"""
Classes and functions for 1D coordinate transformation.
"""

import functools
import warnings

import numpy as np
import xarray as xr
from numba import boolean, float32, float64, guvectorize  # type: ignore

"""Low level functions (numba/numpy)"""


@guvectorize(
    [
        (float64[:], float64[:], float64[:], boolean, boolean, float64[:]),
        (float32[:], float32[:], float32[:], boolean, boolean, float32[:]),
    ],
    "(n),(n),(m),(),()->(m)",
    nopython=True,
)
def _interp_1d_linear(
    phi, theta, target_theta_levels, mask_edges, bypass_checks, output
):
    # rough check if the data is decreasing with depth. If that is the case, flip.
    if not bypass_checks:
        theta_sign_test = theta[~np.isnan(theta)]
        if theta_sign_test[-1] < theta_sign_test[0]:
            theta = theta[::-1]
            phi = phi[::-1]

    output[:] = np.interp(target_theta_levels, theta, phi)

    if mask_edges:
        theta_max = np.nanmax(theta)
        theta_min = np.nanmin(theta)
        for i in range(len(target_theta_levels)):
            theta_lev = target_theta_levels[i]
            if (theta_lev < theta_min) or (theta_lev > theta_max):
                output[i] = np.nan


def interp_1d_linear(
    phi,
    theta,
    target_theta_levels,
    mask_edges=False,
    bypass_checks=False,
    logarithmic=False,
):
    """
    Vectorized interpolation of scalar phi to isosurfaces of scalar theta
    along the final axis.

    Parameters
    ----------
    phi : array_like
        Array of shape (..., n), scalar field to be interpolated
    theta : array_like
        Array of shape (..., n), scalar field which defines the isosurfaces
    target_theta_levels : array_like
        Array of shape (m) specificying target isosurface levels
    mask_edges : bool, optional
        Determines how to handle theta values that exceed the bounds of
        target_theta_levels. If False, fill with nearest valid values. If
        True, fill with NaNs.
    bypass_checks : bool, optional
        Option to bypass logic to flip data if monotonically decreasing along the axis.
        This will improve performance if True, but the user needs to ensure that values
        are increasing alon the axis.
    logarithmic : bool, optional
        Apply a logarithmic transform to theta and target_theta_levels so that the
        interpolation is done in logarithmic, rather than linear, space. In turn, theta
        must be positive. Defaults to False.

    Returns
    -------
    phi_interp : array
        Array of shape (..., m) of phi interpolated to theta isosurfaces.
    """
    if logarithmic:
        theta = np.log(theta)
        target_theta_levels = np.log(target_theta_levels)
    return _interp_1d_linear(phi, theta, target_theta_levels, mask_edges, bypass_checks)


@guvectorize(
    [
        (float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]),
        (float32[:], float32[:], float32[:], float32[:], float32[:], float32[:]),
    ],
    "(n),(n),(n),(m),(m)->(m)",
    nopython=True,
)
def _interp_1d_conservative(phi, theta_1, theta_2, theta_hat_1, theta_hat_2, output):
    output[:] = np.nan

    n = len(theta_1)
    m = len(theta_hat_1)

    for i in range(n):
        # handle missing values
        if np.isnan(theta_1[i]) and np.isnan(theta_2[i]):
            continue
        # in the next two cases, we are effectively applying a boundary condition
        # by assuming that theta is homogenous over the cell
        elif np.isnan(theta_1[i]):
            theta_min = theta_max = theta_2[i]
        elif np.isnan(theta_2[i]):
            theta_min = theta_max = theta_1[i]
        # handle non-monotonic stratification
        elif theta_1[i] < theta_2[i]:
            theta_min = theta_1[i]
            theta_max = theta_2[i]
        else:
            theta_min = theta_2[i]
            theta_max = theta_1[i]

        for j in range(m):
            if np.isnan(phi[i]):
                # missing values in phi (the input data) need to be excluded
                # here or the whole target bin will be NaN
                continue
            if (theta_hat_1[j] > theta_max) or (theta_hat_2[j] < theta_min):
                # there is no overlap between the cell and the bin
                pass
            elif theta_max == theta_min:
                if np.isnan(output[j]):
                    output[j] = phi[i]
                else:
                    output[j] += phi[i]
            else:
                # from here on there is some overlap
                theta_hat_min = max(theta_min, theta_hat_1[j])
                theta_hat_max = min(theta_max, theta_hat_2[j])
                alpha = (theta_hat_max - theta_hat_min) / (theta_max - theta_min)
                # now assign based on this weight
                if np.isnan(output[j]):
                    output[j] = alpha * phi[i]
                else:
                    output[j] += alpha * phi[i]


def _array_module(a):
    """Return the array module (numpy or dask.array) backing ``a``."""
    if hasattr(a, "dask"):
        import dask.array as dask_array

        return dask_array
    return np


def _check_and_align_target_bins(target_theta_bins):
    """Validate target bins column by column and align them for the
    column-wise numba kernel.

    The target bins may carry horizontal/time dimensions in addition to the
    bin dimension, so that each column can define its own target cell
    boundaries. The following explicit rules are applied to every column:

    - A column containing any NaN is considered a *missing* column. Its
      monotonicity is not checked and the corresponding output column is
      filled entirely with NaN.
    - Every non-missing column needs at least two boundaries and has to be
      either strictly increasing or strictly decreasing. Columns with
      different numbers of non-NaN boundaries cannot coexist because they
      cannot share a single output dimension.
    - Strictly decreasing columns are reversed (the resulting output is
      reversed back later, so values stay attached to the user-provided
      bin order).

    Both numpy and dask arrays are accepted. The validation itself only
    evaluates small derived arrays (per-column booleans/counts), so for dask
    inputs the full bin array is never materialized; the returned aligned bins,
    flip and missing arrays stay lazy and preserve the input chunking, which is
    what allows the downstream chunk-wise gufunc to match an eager column loop.

    Parameters
    ----------
    target_theta_bins : np.ndarray or dask.array.Array
        Array of shape (..., m) of target cell boundaries. The last
        dimension is the bin dimension.

    Returns
    -------
    aligned_bins : array
        Array with the same shape/chunking as the input, with each
        non-missing column arranged in strictly increasing order. Missing
        columns are kept as NaN.
    flip : array of bool
        Boolean array of shape (...) flagging the columns that were
        decreasing and had to be reversed.
    missing : array of bool
        Boolean array of shape (...) flagging missing (NaN-containing)
        columns.
    """
    xp = _array_module(target_theta_bins)

    if target_theta_bins.ndim == 0 or target_theta_bins.shape[-1] < 2:
        raise ValueError(
            "`target_theta_bins` needs at least two boundary values "
            f"along its last dimension, got shape {target_theta_bins.shape}."
        )

    # Evaluate small per-column scalar flags. For dask these are reductions
    # over derived arrays; only the small boolean/count results are computed,
    # the full bin array stays lazy.
    def _value(x):
        return x.compute() if hasattr(x, "dask") else x

    nan_mask = xp.isnan(target_theta_bins)
    missing = xp.any(nan_mask, axis=-1)
    counts = xp.sum(~nan_mask, axis=-1)

    # Boundary-count consistency. Fully missing columns (count 0) are exempt;
    # every column with at least one boundary must share the same count. This
    # is expressed with scalar reductions (min/max over distinct valid counts)
    # so it works identically for numpy and dask without boolean indexing,
    # which would produce unknown chunk sizes in dask.
    has_valid = _value(xp.any(counts > 0))
    if has_valid:
        if xp is np:
            distinct = np.unique(counts[counts > 0])
            mismatch = distinct.size > 1
        else:
            # dask: a single valid count means min equals max over the valid
            # columns; this avoids boolean indexing (unknown chunk sizes).
            positive = counts[counts > 0]
            mismatch = _value(positive.min() != positive.max())
        if mismatch:
            raise ValueError(
                "The columns of `target_theta_bins` do not all define the same "
                "number of boundaries. Columns that are entirely NaN (missing "
                "columns) are allowed, but every other column must define the "
                "same number of boundaries. Got per-column non-NaN counts "
                f"{_value(counts)}."
            )

    diffs = xp.diff(target_theta_bins, axis=-1)
    increasing = xp.all(diffs > 0, axis=-1)
    decreasing = xp.all(diffs < 0, axis=-1)

    # NaNs propagate through the comparisons; missing columns are excluded.
    valid = ~missing
    non_monotonic = valid & ~(increasing | decreasing)
    if _value(xp.any(non_monotonic)):
        if xp is np:
            bad_index = np.argwhere(non_monotonic)
            index_msg = (
                f"The first non-monotonic column is at index "
                f"{tuple(bad_index[0])}. "
            )
        else:
            index_msg = ""
        raise ValueError(
            "Target values are not strictly monotonic in every column. "
            + index_msg
            + "Each column of `target_theta_bins` must be either strictly "
            "increasing or strictly decreasing."
        )

    flip = valid & decreasing
    if xp is np:
        aligned_bins = target_theta_bins.copy()
        if np.any(flip):
            aligned_bins[flip] = target_theta_bins[flip][:, ::-1]
    else:
        # dask: reverse the decreasing columns inside each chunk. map_blocks
        # keeps the operation lazy and preserves chunking (boolean-mask
        # assignment would require unknown chunk sizes).
        import dask.array as dask_array

        def _align_block(bins_block):
            nan_block = np.any(np.isnan(bins_block), axis=-1)
            diff_block = np.diff(bins_block, axis=-1)
            dec_block = np.all(diff_block < 0, axis=-1)
            flip_block = (~nan_block) & dec_block
            if np.any(flip_block):
                bins_block = bins_block.copy()
                bins_block[flip_block] = bins_block[flip_block][:, ::-1]
            return bins_block

        aligned_bins = dask_array.map_blocks(
            _align_block, target_theta_bins, dtype=target_theta_bins.dtype
        )

    return aligned_bins, flip, missing


def interp_1d_conservative(phi, theta, target_theta_bins):
    """
    Accumulate extensive cell-centered quantity phi into new vertical coordinate
    defined by scalar theta.

    Parameters
    ----------
    phi : array_like
        Array of shape (..., n) defining an extensive quanitity in a cell
        bounded by two vertices.
    theta : array_like
        Array of shape (..., n+1) giving values of scalar theta  on the
        cell vertices. Phi is assumed to vary linearly between vertices.
    target_theta_bins : array_like
        Array of shape (..., m) defining the bounds of bins in which to
        accumulate phi. In addition to the (last) bin dimension it may carry
        horizontal and/or time dimensions, allowing each column to define its
        own target cell boundaries. Boundary arrays are validated column by
        column: every non-missing column must be strictly monotonic (either
        increasing or decreasing) and all columns must define the same number
        of boundaries. Columns containing NaN boundaries are treated as
        missing and produce an all-NaN output column. Leading dimensions of
        `phi`, `theta` and `target_theta_bins` are broadcast against each
        other following numpy broadcasting rules.

    Returns
    -------
    phi_accum : array_like
        Array of shape (..., m-1) giving the values of phi partitioned into
        specified theta bins. For decreasing target bins the output order
        matches the user-provided bin order.
    """

    assert phi.shape[-1] == (theta.shape[-1] - 1)

    if np.asarray(target_theta_bins).ndim == 1:
        # keep the original, well established code path for 1D targets.
        # np.asarray only triggers a computation for a 1D dask array, which is
        # acceptable because the target carries no chunked loop dimension here.
        target_theta_bins = np.asarray(target_theta_bins)
        target_diff = np.diff(target_theta_bins)
        if all(target_diff < 0):
            flip_switch = True
            target_theta_bins = target_theta_bins[::-1]
        elif all(target_diff > 0):
            flip_switch = False
        else:
            raise ValueError("Target values are not monotonic")

        theta_1 = theta[..., :-1]
        theta_2 = theta[..., 1:]
        theta_hat_1 = target_theta_bins[:-1]
        theta_hat_2 = target_theta_bins[1:]

        out = _interp_1d_conservative(
            phi, theta_1, theta_2, theta_hat_1, theta_hat_2
        )
        if flip_switch:
            out = out[::-1]
        return out

    # multi-dimensional target: each column may define its own boundaries.
    # The validation stays lazy for dask inputs and preserves chunking.
    aligned_bins, flip, missing = _check_and_align_target_bins(target_theta_bins)

    # choose a single array module for broadcasting. If any input is dask we
    # use dask for all of them so that the loop dimensions carry aligned
    # chunking (the gufunc rejects mismatched chunks); otherwise plain numpy.
    is_dask = any(
        hasattr(a, "dask") for a in (phi, theta, aligned_bins, flip, missing)
    )
    if is_dask:
        import dask.array as dask_array

        xp = dask_array
    else:
        xp = np

    # broadcast the leading (horizontal/time) dimensions of all inputs so that
    # every column is paired with its own set of boundaries.
    batch_shape = np.broadcast_shapes(
        np.shape(phi)[:-1],
        np.shape(theta)[:-1],
        target_theta_bins.shape[:-1],
    )
    phi_b = xp.broadcast_to(phi, batch_shape + (phi.shape[-1],))
    theta_b = xp.broadcast_to(theta, batch_shape + (theta.shape[-1],))
    bins_b = xp.broadcast_to(
        aligned_bins, batch_shape + (aligned_bins.shape[-1],)
    )
    flip_b = xp.broadcast_to(flip, batch_shape)
    missing_b = xp.broadcast_to(missing, batch_shape)

    if is_dask:
        # A numpy input broadcast with dask becomes a single chunk, which
        # clashes with the (finer) chunks of the dask inputs. Rechunk every
        # leading (loop) axis to the finest chunk size shared by all arrays,
        # while leaving the core (last) dimensions in one piece as the
        # column-wise gufunc requires.
        lead_ndim = len(batch_shape)
        ref_chunks = []
        for axis in range(lead_ndim):
            sizes = []
            for arr in (phi_b, theta_b, bins_b, flip_b, missing_b):
                sizes.append(min(arr.chunks[axis]))
            ref_chunks.append(min(sizes))

        def _rechunk_lead(arr):
            # keep every core (trailing) dimension in a single piece; only
            # realign the leading loop dimensions
            core_chunks = arr.chunks[lead_ndim:]
            return arr.rechunk(tuple(ref_chunks) + core_chunks)

        phi_b = _rechunk_lead(phi_b)
        theta_b = _rechunk_lead(theta_b)
        bins_b = _rechunk_lead(bins_b)
        flip_b = flip_b.rechunk(tuple(ref_chunks))
        missing_b = missing_b.rechunk(tuple(ref_chunks))

    theta_1 = theta_b[..., :-1]
    theta_2 = theta_b[..., 1:]
    theta_hat_1 = bins_b[..., :-1]
    theta_hat_2 = bins_b[..., 1:]

    out = _interp_1d_conservative(
        phi_b, theta_1, theta_2, theta_hat_1, theta_hat_2
    )

    if xp is np:
        # reverse the output of columns whose target bins were given in
        # decreasing order, so the result follows the user-provided bin order
        if np.any(flip_b):
            out[flip_b] = out[flip_b][:, ::-1]
        # missing target columns propagate as an all-NaN output column
        if np.any(missing_b):
            out[missing_b] = np.nan
    else:
        # apply the per-column flip/missing treatment inside each chunk to
        # stay lazy and keep chunking (boolean-mask assignment on a dask array
        # requires unknown chunk sizes). The boolean flags are broadcast to the
        # full output shape first so map_blocks receives arrays with identical
        # core dimensions instead of mixing in lower-dimensional arguments.
        def _postprocess(out_block, flip_block, missing_block):
            flip_col = flip_block[..., 0]
            missing_col = missing_block[..., 0]
            if np.any(flip_col):
                out_block = out_block.copy()
                out_block[flip_col] = out_block[flip_col][:, ::-1]
            if np.any(missing_col):
                out_block = out_block.copy()
                out_block[missing_col] = np.nan
            return out_block

        target_shape = out.shape
        flip_full = dask_array.broadcast_to(
            flip_b[..., None], target_shape
        )
        missing_full = dask_array.broadcast_to(
            missing_b[..., None], target_shape
        )
        out = dask_array.map_blocks(
            _postprocess,
            out,
            flip_full,
            missing_full,
            dtype=out.dtype,
        )

    return out


"""Mid level functions (xarray)"""


def input_handling(func):
    """Decorator that handles input naming for interpolations."""

    @functools.wraps(func)
    def wrapper_input_handling(*args, **kwargs):
        # unpack args
        phi, theta, target_theta_levels, phi_dim, theta_dim, target_dim = args

        # pop kwargs used for naming
        suffix = kwargs.pop("suffix", "")

        # rename all input dims to unique names to avoid conflicts in xr.apply_ufunc
        temp_dim = "temp_dim_target"
        target_theta_levels = target_theta_levels.rename({target_dim: temp_dim})

        # The phi_dim doesnt matter for the final product, so just rename to
        # # something unique to avoid conflicts in apply_ufunc
        temp_dim2 = "temp_unique"
        phi = phi.rename({phi_dim: temp_dim2})

        # Execute function with temporary names
        args = (phi, theta, target_theta_levels, temp_dim2, theta_dim, temp_dim)
        value = func(*args, **kwargs)

        # rename back to original name
        value = value.rename({temp_dim: target_dim})

        # the cell bounds are stored as an attribute; rename their key and
        # internal dimension back to the user-provided target dimension name
        bounds_attr = temp_dim + "_bounds"
        if bounds_attr in value.attrs:
            bounds = value.attrs.pop(bounds_attr).rename(
                {bounds_attr: target_dim + "_bounds"}
            )
            value.attrs[target_dim + "_bounds"] = bounds

        # when the target is multidimensional its name (e.g. an isosurface
        # coordinate like ``interface_depth_rho``) differs from the target bin
        # dimension name; the center coordinate keeps that target name, mirroring
        # the linear interpolation behavior, while the dimension itself keeps the
        # target dimension name.
        center_attr = temp_dim + "_center_name"
        if center_attr in value.attrs:
            center_name = value.attrs.pop(center_attr)
            if center_name is not None and center_name != target_dim:
                value = value.assign_coords(
                    {center_name: value.coords[target_dim]}
                ).reset_coords(target_dim, drop=True)

        # name the output according to input name and user customizable suffix
        if phi.name:
            value.name = phi.name + suffix

        return value

    return wrapper_input_handling


@input_handling
def linear_interpolation(
    phi, theta, target_theta_levels, phi_dim, theta_dim, target_dim, **kwargs
):
    out = xr.apply_ufunc(
        interp_1d_linear,
        phi,
        theta,
        target_theta_levels,
        kwargs=kwargs,
        input_core_dims=[[phi_dim], [theta_dim], [target_dim]],
        output_core_dims=[[target_dim]],
        exclude_dims=set((phi_dim, theta_dim)),
        dask="parallelized",
        output_dtypes=[phi.dtype],
    )
    return out


@input_handling
def conservative_interpolation(
    phi, theta, target_theta_levels, phi_dim, theta_dim, target_dim, **kwargs
):
    # the target can be 1D (identical bounds for every column) or carry
    # horizontal/time dimensions in addition to the (last) bin dimension.
    # In the latter case every column defines its own target boundaries.
    n_bounds = target_theta_levels.sizes[target_dim]

    # The numba kernel is column-wise, so only the bin dimension is a core
    # dimension; all other (horizontal/time) dimensions are broadcast by numpy.
    out = xr.apply_ufunc(
        interp_1d_conservative,
        phi,
        theta,
        target_theta_levels,
        kwargs=kwargs,
        input_core_dims=[[phi_dim], [theta_dim], [target_dim]],
        output_core_dims=[["remapped"]],
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"remapped": n_bounds - 1}},
        # Since we are introducing a new dimension instead of changing it we need to declare the output size.
        output_dtypes=[phi.dtype],
    ).rename({"remapped": target_dim})

    # assign the per-column target cell centers. For a multidimensional target
    # these are themselves multidimensional: each column keeps its own centers
    # instead of reusing the first column's values. The center coordinate may
    # therefore carry the same horizontal/time dimensions as the target.
    center_dims = tuple(
        d for d in target_theta_levels.dims if d != target_dim
    ) + (target_dim,)
    target_centers = (
        target_theta_levels.isel({target_dim: slice(1, None)}).data
        + target_theta_levels.isel({target_dim: slice(0, -1)}).data
    ) / 2
    out = out.assign_coords({target_dim: (center_dims, target_centers)})

    # remember the target's own name so the decorator can rename the center
    # coordinate to it (e.g. a multidim isosurface coordinate); for a 1D target
    # the name equals the target dimension and no rename is needed.
    out.attrs[target_dim + "_center_name"] = target_theta_levels.name

    # preserve the corresponding per-column cell bounds. Since the output only
    # carries the m-1 cell centers along `target_dim`, the m boundaries cannot
    # be stored as an ordinary coordinate (its vertex dimension would be larger
    # than the data dimension); they are attached as a DataArray attribute.
    # For multidimensional targets the bounds retain the horizontal/time dims.
    out.attrs[target_dim + "_bounds"] = target_theta_levels.rename(
        {target_dim: target_dim + "_bounds"}
    )

    return out


def transform(
    grid,
    axis_name,
    da,
    target,
    target_data=None,
    target_dim=None,
    method="linear",
    mask_edges=True,
    bypass_checks=False,
    suffix="_transformed",
):
    """Convert an array of data to new 1D-coordinates.
    The method takes a multidimensional array of data `da` and
    transforms it onto another data_array `target_data` in the
    direction of the axis (for each 1-dimensional 'column').
    `target_data` can be e.g. the existing coordinate along an
    axis, like depth. xgcm automatically detects the appropriate
    coordinate and then transforms the data from the input
    positions to the desired positions defined in `target`. This
    is the default behavior. If the `target` has more than one
    dimension, the `target_dim` has to be explicitly set. Otherwise,
    it is inferred from `target`. The method can also be used for more
    complex cases like transforming a dataarray into new
    coordinates that are defined by e.g. a tracer field like
    temperature, density, etc.
    Currently three methods are supported to carry out the
    transformation:
    - 'linear': Values are linear interpolated between 1D columns
      along `axis` of `da` and `target_data`. This method requires
      `target_data` to increase/decrease monotonically. `target`
      values are interpreted as new cell centers in this case. By
      default this method will return nan for values in `target` that
      are outside of the range of `target_data`, setting
      `mask_edges=False` results in the default np.interp behavior of
      repeated values.
    - 'log': Same as 'linear', but with values interpolated
      logarithmically between 1D columns. Operates by applying `np.log`
      to the target and target data values prior to linear interpolation.
    - 'conservative': Values are transformed while conserving the
      integral of `da` along each 1D column. This method can be used
      with non-monotonic values of `target_data`. Currently this will
      only work with extensive quantities (like heat, mass, transport)
      but not with intensive quantities (like temperature, density,
      velocity). N given `target` values are interpreted as cell-bounds
      and the returned array will have N-1 elements along the newly
      created coordinate, with coordinate values that are interpolated
      between `target` values. The `target` may be one-dimensional (the
      same boundaries are then used for every column) or carry horizontal
      and/or time dimensions so that each column defines its own target
      cell boundaries. In the multidimensional case the boundaries are
      validated column by column: every non-missing column must be
      strictly monotonic (either increasing or decreasing) and all
      columns must define the same number of boundaries. A column that
      contains NaN boundaries is treated as missing and produces an
      all-NaN output column. Leading dimensions of `da`, `target_data`
      and `target` are broadcast following numpy broadcasting rules.
      The output stores the per-column cell centers as its coordinate and
      the corresponding per-column cell bounds as the
      `<target_dim>_bounds` attribute. With fully covering target bins the
      total of each column is conserved.
    Parameters
    ----------
    grid : xgcm.Grid
        xgcm Grid object
    axis_name : str
        Name of the axis along which to operate
    da : xr.DataArray
        Input data
    target : {np.array, xr.DataArray}
        Target points for transformation. Depending on the method is
        interpreted as cell center (method='linear' and method='log') or
        cell bounds (method='conservative).
        Values correspond to `target_data` or the existing coordinate
        along the axis (if `target_data=None`).
    target_data : xr.DataArray, optional
        Data to transform onto (e.g. a tracer like density or temperature).
        Defaults to None, which infers the appropriate coordinate along
        `axis` (e.g. the depth).
    target_dim : str, optional
        Dimension name associated with the `target` points. If `target` has more than one dimension, this
        parameter must be explicitly set to specify which dimension corresponds to the transformation. If `target`
        is one-dimensional, `target_dim` is inferred automatically.
    method : str, optional
        Method used to transform, by default "linear"
    mask_edges : bool, optional
        If activated, `target` values outside the range of `target_data`
        are masked with nan, by default True. Only applies to 'linear' and 'log'
        methods.
    bypass_checks : bool, optional
        Only applies for `method='linear'` and `method='log'`.
        Option to bypass logic to flip data if monotonically decreasing along the axis.
        This will improve performance if True, but the user needs to ensure that values
        are increasing along the axis.
    suffix : str, optional
        Customizable suffix to the name of the output array. This will
        be added to the original name of `da`. Defaults to `_transformed`.

    Returns
    -------
    xr.DataArray
        The transformed data

    Notes
    -----
    - If `target` is multi-dimensional, you must specify `target_dim` explicitly.
    - If `target_dim` is not specified and `target_data` has no name, a default name "TRANSFORMED_DIMENSION" will be used for the transformed dimension.
    - For `conservative` transformations, `target_data` must be located on cell bounds.

    """

    axis = grid.axes[axis_name]

    # raise error if axis is periodic
    if axis.padding == "periodic":
        raise ValueError(
            "`transform` can only be used on axes that are non-periodic. Set a "
            "non-periodic boundary (e.g. `padding='fill'`, or leave it unset) "
            "for this axis on `xgcm.Grid`."
        )

    # raise error if the target values are not provided as xr.dataarray
    for var_name, variable, allowed_types in [
        ("da", da, [xr.DataArray]),
        ("target", target, [xr.DataArray, np.ndarray]),
        ("target_data", target_data, [xr.DataArray]),
    ]:
        if not (isinstance(variable, tuple(allowed_types)) or variable is None):
            raise ValueError(
                f"`{var_name}` needs to be a {' or '.join([str(a) for a in allowed_types])}. Found {type(variable)}"
            )

    def _target_data_name_handling(target_data):
        """Handle target_data input without a name"""
        if target_data.name is None:
            warnings.warn(
                "Input`target_data` has no name, but we need a name for the transformed dimension. The name `TRANSFORMED_DIMENSION` will be used. To avoid this warning, call `.rename` on `target_data` before calling `transform`."
            )
            target_data.name = "TRANSFORMED_DIMENSION"

    def _check_other_dims(target_da):
        # check if other dimensions (excluding ones associated with the transform axis) are the
        # same between `da` and `target_data`. If not provide instructions how to work around.

        da_other_dims = set(da.dims) - set(axis.coords.values())
        target_da_other_dims = set(target_da.dims) - set(axis.coords.values())
        if not target_da_other_dims.issubset(da_other_dims):
            raise ValueError(
                f"Found additional dimensions [{target_da_other_dims - da_other_dims}]"
                "in `target_data` not found in `da`. This could mean that the target "
                "array is not on the same position along other axes."
                " If the additional dimensions are associated witha staggered axis, "
                "use grid.interp() to move values to other grid position. "
                "If additional dimensions are not related to the grid (e.g. climate "
                "model ensemble members or similar), use xr.broadcast() before using transform."
            )

    def _parse_target(target, target_dim, target_data_dim, target_data):
        """Parse target values into correct xarray naming and set default naming based on input data"""
        # if target_data is not provided, assume the target to be one of the staggered dataset dimensions.
        if target_data is None:
            target_data = grid._ds[target_data_dim]

        if target_dim is None:
            # Infer target_dim from target
            if isinstance(target, xr.DataArray):
                if len(target.dims) == 1:
                    if target_dim is None:
                        target_dim = list(target.dims)[0]
                else:
                    raise ValueError(
                        "When `target` is multi-dimensional, `target_dim` has "
                        "to be set explicitly to specify which of its "
                        f"dimensions {target.dims} is the target bin dimension."
                    )
            else:
                # if the target is not provided as xr.Dataarray we take the name of the target_data as new dimension name
                _target_data_name_handling(target_data)
                target_dim = target_data.name
        if not isinstance(target, xr.DataArray):
            target = xr.DataArray(
                target, dims=[target_dim], coords={target_dim: target}
            )

        if target_dim not in target.dims:
            raise ValueError(
                f"The specified `target_dim` {target_dim!r} is not within the "
                f"dimensions of the target: {target.dims}."
            )

        _check_other_dims(target_data)
        return target, target_dim, target_data

    _, dim = axis._get_position_name(da)
    if method == "linear" or method == "log":
        target, target_dim, target_data = _parse_target(
            target, target_dim, dim, target_data
        )
        out = linear_interpolation(
            da,
            target_data,
            target,
            dim,
            dim,  # in this case the dimension of phi and theta are the same
            target_dim,
            mask_edges=mask_edges,
            bypass_checks=bypass_checks,
            logarithmic=(method == "log"),
        )
    elif method == "conservative":
        # Multi-dimensional targets are supported: each column may define its
        # own target cell boundaries (validated column by column further down).

        # the conservative method requires `target_data` to be on the `outer` coordinate.
        # If that is not the case (a very common use case like transformation on any tracer),
        # we need to infer the boundary values (using the interp logic)
        # for this method we need the `outer` position. Error out if its not there.
        try:
            target_data_dim = axis.coords["outer"]
        except KeyError:
            raise RuntimeError(
                "In order to use the method `conservative` the grid object needs to have `outer` coordinates."
            )

        target, target_dim, target_data = _parse_target(
            target, target_dim, target_data_dim, target_data
        )

        # check on which coordinate `target_data` is, and interpolate if needed
        if target_data_dim not in target_data.dims:
            warnings.warn(
                "The `target data` input is not located on the cell bounds. This method will continue with linear interpolation with repeated boundary values. For most accurate results provide values on cell bounds.",
                UserWarning,
            )
            target_data = grid.interp(target_data, axis_name, padding="extend")
            # This seems to end up with chunks along the axis dimension.
            # Rechunk to keep xr.apply_func from complaining.
            # TODO: This should be made obsolete, when the internals are refactored using numba
            target_data = target_data.chunk(
                {axis._get_position_name(target_data)[1]: -1}
            )

        out = conservative_interpolation(
            da,
            target_data,
            target,
            dim,
            target_data_dim,  # in this case the dimension of phi and theta are the same
            target_dim,
        )

    return out
