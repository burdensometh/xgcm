"""
Classes and functions for 1D coordinate transformation.
"""

import functools
import math
import numbers
import warnings
from typing import Dict, NamedTuple, Optional

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
        Array of shape (m) defining the bounds of bins in which to accumulate
        phi.

    Returns
    -------
    phi_accum : array_like
        Array of shape (..., m-1) giving the values of phi partitioned into
        specified theta bins.
    """

    assert phi.shape[-1] == (theta.shape[-1] - 1)
    assert target_theta_bins.ndim == 1

    # flip target_theta_bins if needed (only needed for the conservative method,
    # np.interp handles this by itself)
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

    out = _interp_1d_conservative(phi, theta_1, theta_2, theta_hat_1, theta_hat_2)
    if flip_switch:
        out = out[::-1]
    return out


"""Mid level functions (xarray)"""


class _TransformChunkPlan(NamedTuple):
    """Chunk sizes chosen for a single transform block.

    The transform dimension (the "core" dimension) is always kept as one
    complete column. ``chunks`` maps every broadcast (non-core) dimension that
    had to be split to the maximum number of elements allowed in one block.
    """

    chunks: Dict[str, int]
    per_column_bytes: int
    max_memory: Optional[int]

    @property
    def columns_per_block(self) -> Optional[int]:
        if not self.chunks:
            return None
        n_columns = 1
        for size in self.chunks.values():
            n_columns *= size
        return n_columns


def _parse_max_memory(max_memory):
    """Normalize a user supplied memory budget to bytes."""
    if max_memory is None:
        return None
    if isinstance(max_memory, bool):
        raise TypeError("`max_memory` must be a number of bytes or a string.")
    if isinstance(max_memory, str):
        try:
            from dask.utils import parse_bytes  # type: ignore
        except ImportError:  # pragma: no cover - dask is a hard dependency
            raise ImportError(
                "Reading `max_memory` from a string requires dask. "
                "Pass an integer number of bytes instead."
            )
        value = parse_bytes(max_memory)
    elif isinstance(max_memory, numbers.Real):
        if not math.isfinite(float(max_memory)):
            raise ValueError("`max_memory` must be finite.")
        value = int(max_memory)
    else:
        raise TypeError("`max_memory` must be a number of bytes or a string.")
    if value <= 0:
        raise ValueError(f"`max_memory` must be positive, received {max_memory!r}.")
    return value


def _column_bytes(length, dtype):
    return int(length) * np.dtype(dtype).itemsize


def _estimate_column_bytes(input_columns, output_length, output_dtype):
    """Estimate peak bytes needed for one transform column.

    ``input_columns`` is an iterable of ``(length, dtype)`` pairs describing
    every input that is in flight at the same time for a single column (e.g.
    both ``da`` and ``target_data``). The estimate adds the output column and
    a scratch term covering the contiguous copies / accumulation buffers
    allocated by the (guvectorized) kernels.
    """
    input_bytes = sum(
        _column_bytes(length, dtype) for length, dtype in input_columns
    )
    output_bytes = _column_bytes(output_length, output_dtype)
    scratch_bytes = input_bytes + output_bytes
    return input_bytes + output_bytes + scratch_bytes


def _collect_non_core_dims(arrays, core_dims):
    non_core_dims = []
    for array in arrays:
        for dim in array.dims:
            if dim in core_dims or dim in non_core_dims:
                continue
            non_core_dims.append(dim)
    return non_core_dims


def _initial_block_size(arrays, dim):
    """Largest existing chunk along ``dim`` (falls back to the full length)."""
    length = None
    largest_chunk = 0
    for array in arrays:
        if dim not in array.dims:
            continue
        length = array.sizes[dim]
        if array.chunks is not None:
            largest_chunk = max(
                largest_chunk, max(array.chunks[array.get_axis_num(dim)])
            )
    if length is None:
        return 1
    return min(length, largest_chunk or length)


def _plan_transform_chunks(
    arrays, core_dims, input_columns, output_length, output_dtype, max_memory
):
    """Plan chunk sizes for a transform.

    The complete transform (vertical) column is always retained. If a memory
    budget is supplied, broadcast (horizontal/time) dimensions are split so
    that the estimated peak memory of a single block -- all inputs, the
    target, the output and kernel temporaries -- fits within ``max_memory``.
    """
    per_column_bytes = _estimate_column_bytes(
        input_columns, output_length, output_dtype
    )
    budget = _parse_max_memory(max_memory)

    if budget is not None and per_column_bytes > budget:
        raise ValueError(
            "Cannot perform transform within `max_memory`="
            f"{budget} bytes: a single transform column requires approximately "
            f"{per_column_bytes} bytes (all inputs, target, output and kernel "
            "temporaries). Increase `max_memory` or split the data before "
            "calling transform."
        )

    dask_backed = any(array.chunks is not None for array in arrays)
    if budget is None or not dask_backed:
        return _TransformChunkPlan(
            chunks={}, per_column_bytes=per_column_bytes, max_memory=budget
        )

    capacity = max(1, budget // max(per_column_bytes, 1))
    non_core_dims = _collect_non_core_dims(arrays, core_dims)
    sizes = {
        dim: _initial_block_size(arrays, dim) for dim in non_core_dims
    }

    # Shrink the largest broadcast chunk until one block fits the budget.
    while math.prod(sizes.values()) > capacity:
        dim = max(sizes, key=lambda key: (sizes[key], key))
        if sizes[dim] <= 1:
            # Every remaining dimension of length > 1 has been split down to
            # a single element. ``capacity`` is at least one, so the loop can
            # only end here when the block already fits.
            break
        sizes[dim] = max(1, sizes[dim] // 2)

    return _TransformChunkPlan(
        chunks=sizes, per_column_bytes=per_column_bytes, max_memory=budget
    )


def _apply_chunk_plan(plan, array, core_dims):
    """Apply a chunk plan to ``array`` without computing it.

    Every core dimension is merged into a single chunk (one complete column).
    Broadcast dimensions are only rechunked when they exceed the planned block
    size. Numpy-backed arrays are returned unchanged: their data is already
    resident in memory, so splitting them cannot lower peak memory and would
    silently make the result lazy. When a dask input is present, xarray aligns
    the numpy inputs to the planned chunks as part of the dask graph.
    """
    if not array.dims or array.chunks is None:
        return array

    core_dims = set(core_dims)
    spec = {}
    for dim in core_dims:
        if dim not in array.dims:
            continue
        # Only merge when the column is actually split across chunks.
        if len(array.chunks[array.get_axis_num(dim)]) > 1:
            spec[dim] = -1

    for dim, size in plan.chunks.items():
        if dim not in array.dims or dim in core_dims:
            continue
        if max(array.chunks[array.get_axis_num(dim)]) > size:
            spec[dim] = size

    if not spec:
        return array
    return array.chunk(spec)


def _prepare_transform_inputs(
    phi, theta, target, phi_dim, theta_dim, target_dim, output_length, max_memory
):
    """Plan and apply budget-constrained chunking to transform inputs.

    All inputs that are simultaneously in flight for a single column (``phi``,
    ``theta`` and ``target``) contribute to the memory estimate. The complete
    column along the transform core dimensions is always retained as one chunk;
    broadcast (horizontal / time) dimensions are split only when required to
    obey ``max_memory``.
    """
    plan = _plan_transform_chunks(
        arrays=[phi, theta, target],
        core_dims=[phi_dim, theta_dim, target_dim],
        input_columns=[
            (phi.sizes[phi_dim], phi.dtype),
            (theta.sizes[theta_dim], theta.dtype),
            (target.sizes[target_dim], target.dtype),
        ],
        output_length=output_length,
        output_dtype=phi.dtype,
        max_memory=max_memory,
    )
    phi = _apply_chunk_plan(plan, phi, [phi_dim])
    theta = _apply_chunk_plan(plan, theta, [theta_dim])
    target = _apply_chunk_plan(plan, target, [target_dim])
    return phi, theta, target


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

        # name the output according to input name and user customizable suffix
        if phi.name:
            value.name = phi.name + suffix

        return value

    return wrapper_input_handling


@input_handling
def linear_interpolation(
    phi, theta, target_theta_levels, phi_dim, theta_dim, target_dim, **kwargs
):
    max_memory = kwargs.pop("max_memory", None)
    phi, theta, target_theta_levels = _prepare_transform_inputs(
        phi,
        theta,
        target_theta_levels,
        phi_dim,
        theta_dim,
        target_dim,
        output_length=target_theta_levels.sizes[target_dim],
        max_memory=max_memory,
    )
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
    max_memory = kwargs.pop("max_memory", None)
    phi, theta, target_theta_levels = _prepare_transform_inputs(
        phi,
        theta,
        target_theta_levels,
        phi_dim,
        theta_dim,
        target_dim,
        # The conservative method returns one value per interval between
        # adjacent target bounds, i.e. one fewer than the number of bounds.
        output_length=target_theta_levels.sizes[target_dim] - 1,
        max_memory=max_memory,
    )
    out = xr.apply_ufunc(
        interp_1d_conservative,
        phi,
        theta,
        target_theta_levels,
        kwargs=kwargs,
        input_core_dims=[[phi_dim], [theta_dim], [target_dim]],
        output_core_dims=[["remapped"]],
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"remapped": len(target_theta_levels) - 1}},
        # Since we are introducing a new dimension instead of changing it we need to declare the output size.
        output_dtypes=[phi.dtype],
    ).rename({"remapped": target_dim})

    # assign the target cell center
    target_centers = (target_theta_levels.data[1:] + target_theta_levels.data[:-1]) / 2
    out = out.assign_coords({target_dim: target_centers})

    # TODO: Somehow preserve the original bounds

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
    max_memory=None,
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
      between `target` values. This method does currently not work
      for multi-dimensional targets.
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
    max_memory : int or str, optional
        Upper bound on the estimated peak memory, in bytes (or a string such
        as ``"500MB"``), used to plan chunking of the transform. The complete
        column along `axis` is always kept in a single chunk. When the inputs
        are dask-backed, the horizontal / time dimensions are split so that a
        single block -- all inputs, the target, the output and kernel
        temporaries -- fits within `max_memory`. If a single column already
        exceeds the budget, a `ValueError` is raised before any work is done.
        Defaults to None, in which case the existing chunking is preserved
        (aside from merging `axis` into a single chunk).

    Returns
    -------
    xr.DataArray
        The transformed data

    Notes
    -----
    - If `target` is multi-dimensional, you must specify `target_dim` explicitly.
    - If `target_dim` is not specified and `target_data` has no name, a default name "TRANSFORMED_DIMENSION" will be used for the transformed dimension.
    - For `conservative` transformations, `target_data` must be located on cell bounds.
    - `max_memory` only affects the dask chunk layout; no data is computed
      eagerly and the returned array stays lazy.

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
                    if target_dim is not None and target_dim not in target.dims:
                        raise ValueError(
                            f"The specified `target_dim` {target_dim} is not within the dimensions of the target: [{target.dims}]."
                        )
            else:
                # if the target is not provided as xr.Dataarray we take the name of the target_data as new dimension name
                _target_data_name_handling(target_data)
                target_dim = target_data.name
        if not isinstance(target, xr.DataArray):
            target = xr.DataArray(
                target, dims=[target_dim], coords={target_dim: target}
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
            max_memory=max_memory,
        )
    elif method == "conservative":
        if isinstance(target, xr.DataArray):
            if len(target.dims) > 1:
                raise NotImplementedError(
                    "Conservative transformation is not yet supported for multi-dimensional targets."
                )

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
            # The internal `grid.interp` is itself a grid ufunc that requires a
            # single chunk along its core dimension, and the intermediate
            # column is in flight together with the transform inputs. Plan the
            # chunking (and reject an over-budget single column) before
            # interpolating rather than after.
            center_dim = axis._get_position_name(target_data)[1]
            plan = _plan_transform_chunks(
                arrays=[da, target_data, target],
                core_dims=[dim, center_dim, target_dim],
                # Peak memory spans the internal interpolation, when the center
                # column of `target_data` is in flight alongside `da`, the
                # interpolated outer column (one element longer), the target
                # and the output.
                input_columns=[
                    (da.sizes[dim], da.dtype),
                    (target_data.sizes[center_dim], target_data.dtype),
                    (target_data.sizes[center_dim] + 1, target_data.dtype),
                    (target.sizes[target_dim], target.dtype),
                ],
                output_length=target.sizes[target_dim] - 1,
                output_dtype=da.dtype,
                max_memory=max_memory,
            )
            da = _apply_chunk_plan(plan, da, [dim])
            target = _apply_chunk_plan(plan, target, [target_dim])
            target_data = _apply_chunk_plan(plan, target_data, [center_dim])

            target_data = grid.interp(target_data, axis_name, padding="extend")
            # This seems to end up with chunks along the axis dimension.
            # Rechunk to keep xr.apply_func from complaining.
            # TODO: This should be made obsolete, when the internals are refactored using numba
            target_data = target_data.chunk({target_data_dim: -1})

        out = conservative_interpolation(
            da,
            target_data,
            target,
            dim,
            target_data_dim,  # in this case the dimension of phi and theta are the same
            target_dim,
            max_memory=max_memory,
        )

    return out
