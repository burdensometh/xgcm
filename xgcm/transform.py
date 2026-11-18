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


@guvectorize(
    [
        (float64[:], float64[:], float64[:], float64[:], float64[:]),
        (float32[:], float32[:], float32[:], float32[:], float32[:]),
    ],
    "(n),(n),(m),(m)->(m)",
    nopython=True,
)
def _coverage_1d_conservative(theta_1, theta_2, theta_hat_1, theta_hat_2, output):
    # Fractional coverage of every target bin by the source cells, measured in
    # units of the target-coordinate (theta) extent of the bin. The same overlap
    # geometry as in `_interp_1d_conservative` is used, so that the coverage
    # describes exactly how much of each target bin received contributions.
    output[:] = np.nan

    n = len(theta_1)
    m = len(theta_hat_1)

    for j in range(m):
        width = theta_hat_2[j] - theta_hat_1[j]
        if width == 0:
            # a zero-width target bin has no meaningful coverage fraction
            output[j] = np.nan
            continue
        overlap = 0.0
        for i in range(n):
            # mirror the boundary-value handling of _interp_1d_conservative
            if np.isnan(theta_1[i]) and np.isnan(theta_2[i]):
                continue
            elif np.isnan(theta_1[i]):
                theta_min = theta_max = theta_2[i]
            elif np.isnan(theta_2[i]):
                theta_min = theta_max = theta_1[i]
            elif theta_1[i] < theta_2[i]:
                theta_min = theta_1[i]
                theta_max = theta_2[i]
            else:
                theta_min = theta_2[i]
                theta_max = theta_1[i]

            if (theta_hat_1[j] > theta_max) or (theta_hat_2[j] < theta_min):
                # no overlap between the source cell and the target bin
                continue
            elif theta_max == theta_min:
                # a homogeneous (zero-extent) source surface contributes no
                # measurable extent to the target bin
                continue
            else:
                theta_hat_min = max(theta_min, theta_hat_1[j])
                theta_hat_max = min(theta_max, theta_hat_2[j])
                overlap += theta_hat_max - theta_hat_min
        output[j] = overlap / width


def coverage_1d_conservative(theta, target_theta_bins):
    """
    Geometric overlap coverage of target bins by the source cells.

    For each target bin the function returns the sum of the fractions of
    the bin's extent in the theta coordinate that overlap with source
    cells. The overlap geometry is identical to the one used by
    :func:`interp_1d_conservative`, including the boundary handling for
    missing vertex values and non-monotonic stratification.

    For a monotonic theta coordinate every part of a target bin overlaps
    at most one source cell, so the result is a true coverage fraction in
    the closed interval [0, 1]. For a non-monotonic (overturning)
    coordinate several source layers can map onto the same target bin; the
    returned value is then an overlap *multiplicity* that can exceed 1,
    meaning the bin is filled by more than one source layer (it is over-,
    not under-represented). Clip the result to 1 when a [0, 1] coverage
    fraction is needed for a threshold.

    Parameters
    ----------
    theta : array_like
        Array of shape (..., n+1) giving values of scalar theta on the
        source cell vertices.
    target_theta_bins : array_like
        Array of shape (m) defining the bounds of the target bins.

    Returns
    -------
    coverage : array_like
        Array of shape (..., m-1) of geometric overlap coverage (see
        above). Target bins without any overlap get zero coverage;
        zero-width target bins get NaN.
    """

    assert target_theta_bins.ndim == 1

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

    out = _coverage_1d_conservative(theta_1, theta_2, theta_hat_1, theta_hat_2)
    if flip_switch:
        out = out[::-1]
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


@input_handling
def conservative_coverage(
    phi, theta, target_theta_levels, phi_dim, theta_dim, target_dim, **kwargs
):
    # `phi` is only passed through the naming decorator; the coverage itself
    # only depends on the coordinate geometry (theta and target levels).
    out = xr.apply_ufunc(
        coverage_1d_conservative,
        theta,
        target_theta_levels,
        kwargs=kwargs,
        input_core_dims=[[theta_dim], [target_dim]],
        output_core_dims=[["remapped"]],
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"remapped": len(target_theta_levels) - 1}},
        output_dtypes=[theta.dtype],
    ).rename({"remapped": target_dim})

    # assign the target cell center
    target_centers = (target_theta_levels.data[1:] + target_theta_levels.data[:-1]) / 2
    out = out.assign_coords({target_dim: target_centers})

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
    quantity="extensive",
    coverage_threshold=1.0,
    return_effective_volume=False,
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
      with non-monotonic values of `target_data`. N given `target`
      values are interpreted as cell-bounds and the returned array
      will have N-1 elements along the newly created coordinate, with
      coordinate values that are interpolated between `target` values.
      This method does currently not work for multi-dimensional targets.
      The interpretation of `da` is controlled by `quantity`:

      * ``quantity='extensive'`` (the default): `da` is an extensive
        quantity (like heat, mass, transport) that is already weighted
        by the source cell volume along the transform axis and is simply
        accumulated into the target bins. Intensive quantities (like
        temperature, density, velocity) cannot be remapped this way.
      * ``quantity='intensive'``: `da` is an intensive quantity (a
        concentration). It is first multiplied with the source effective
        volume (the grid metric along `axis_name`) to form an extensive
        total amount; both the total amount and the effective volume are
        remapped conservatively and independently, and the target
        concentration is their ratio. Target bins whose coverage by the
        source grid is below `coverage_threshold` are set to NaN; in
        particular a bin with zero coverage is never assigned a zero
        concentration. The remapped effective volume can be returned and
        registered as a metric on a target `Grid` (see
        :func:`target_grid`) so that subsequent `average` / `integrate`
        calls use the same partial-coverage accounting as the remap.
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
    quantity : {'extensive', 'intensive'}, optional
        Only applies to `method='conservative'`. Selects whether `da` is
        treated as an extensive quantity (the default, preserves the
        original behavior) or as an intensive concentration. The
        intensive mode requires a grid metric (cell volume along
        `axis_name`) to be registered on `grid`.
    coverage_threshold : float, optional
        Only applies to `method='conservative'` with
        ``quantity='intensive'``. Minimum source coverage fraction (a
        number between 0 and 1) of a target bin for a concentration to be
        returned. The coverage fraction is the geometric overlap between
        source cells and the target bin, clipped to 1 (clipping only
        matters for a non-monotonic / overturning coordinate, where several
        source layers can fill one target bin). Bins with coverage below
        the threshold are set to NaN. The default of 1.0 means that only
        fully covered target bins receive a value; bins with zero
        coverage are always NaN regardless of this setting.
    return_effective_volume : bool, optional
        Only applies to `method='conservative'` with
        ``quantity='intensive'``. If True, return a tuple
        ``(concentration, effective_volume)`` where the second element is
        the conservatively remapped source effective volume on the
        target bins. The volume is finite everywhere, with zero effective
        volume on target bins that receive no source contribution; the
        concentration's own NaN mask (controlled by
        `coverage_threshold`) carries the missingness. The volume can be
        registered as a metric on the target `Grid` (see
        :func:`target_grid`), ensuring that later `average` /
        `integrate` operations use exactly the coverage convention of
        the remap. Defaults to False.

    Returns
    -------
    xr.DataArray or tuple of xr.DataArray
        The transformed data. When `return_effective_volume` is True, a
        tuple ``(concentration, effective_volume)`` is returned.

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

    # validate the intensive / extensive mode selection
    if quantity not in ("extensive", "intensive"):
        raise ValueError(
            f"`quantity` must be either 'extensive' or 'intensive', got {quantity!r}."
        )
    if quantity == "intensive" and method != "conservative":
        raise ValueError(
            "The intensive quantity mode (`quantity='intensive'`) is only "
            "available with `method='conservative'`, since a concentration "
            "cannot be remapped conservatively without the target cell bounds."
        )
    if return_effective_volume and not (
        method == "conservative" and quantity == "intensive"
    ):
        raise ValueError(
            "`return_effective_volume=True` is only supported with "
            "`method='conservative'` and `quantity='intensive'`."
        )
    if not (0.0 <= coverage_threshold <= 1.0):
        raise ValueError(
            "`coverage_threshold` must be a fraction between 0 and 1, "
            f"got {coverage_threshold}."
        )
    if method != "conservative" and coverage_threshold != 1.0:
        warnings.warn(
            "`coverage_threshold` only applies to conservative transformations "
            "of intensive quantities and will be ignored."
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
            target_data = grid.interp(target_data, axis_name, padding="extend")
            # This seems to end up with chunks along the axis dimension.
            # Rechunk to keep xr.apply_func from complaining.
            # TODO: This should be made obsolete, when the internals are refactored using numba
            target_data = target_data.chunk(
                {axis._get_position_name(target_data)[1]: -1}
            )

        if quantity == "intensive":
            # the concentration is never remapped directly: form the total
            # amount and remap it together with the effective volume
            return _conservative_intensive(
                grid=grid,
                axis_name=axis_name,
                da=da,
                dim=dim,
                target_data=target_data,
                target_data_dim=target_data_dim,
                target=target,
                target_dim=target_dim,
                coverage_threshold=coverage_threshold,
                suffix=suffix,
                return_effective_volume=return_effective_volume,
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


def _conservative_intensive(
    grid,
    axis_name,
    da,
    dim,
    target_data,
    target_data_dim,
    target,
    target_dim,
    coverage_threshold,
    suffix,
    return_effective_volume,
):
    """Conservative remap of an intensive quantity (a concentration).

    The concentration is never remapped directly. It is combined with the
    source cell volume to form an extensive total amount, and both the total
    amount and the effective volume are remapped independently with the
    extensive conservative kernel. The target concentration is the ratio of
    the two remapped fields. Target bins not covered enough by the source
    grid are declared missing (NaN); in particular zero coverage can never
    produce a zero concentration.
    """
    # the source effective volume on the cells of `da`: a metric registered
    # along the transform axis (e.g. cell thickness / area / volume)
    try:
        source_volume = grid.get_metric(da, axis_name)
    except KeyError:
        raise KeyError(
            "The intensive quantity mode requires an effective volume metric "
            f"along axis {axis_name!r} to be registered on the source grid "
            "(e.g. a cell thickness or volume metric, see `Grid.set_metrics`)."
        )

    # align the metric to `da`, including its missing-value mask. The same
    # mask is applied to amount and volume so that both remaps use exactly
    # the same contributing source cells.
    mask = np.isfinite(da)
    source_volume = xr.broadcast(source_volume, da)[0].where(mask)

    # 1. form the extensive total amount in each source cell
    total_amount = da * source_volume

    # 2. remap total amount and effective volume independently, using the
    #    unmodified extensive (conservative) kernel
    remapped_amount = conservative_interpolation(
        total_amount,
        target_data,
        target,
        dim,
        target_data_dim,
        target_dim,
    )
    effective_volume = conservative_interpolation(
        source_volume,
        target_data,
        target,
        dim,
        target_data_dim,
        target_dim,
    )

    # 3. geometric overlap coverage of each target bin by source cells,
    #    using the identical overlap geometry as the conservative kernel
    coverage = conservative_coverage(
        da,
        target_data,
        target,
        dim,
        target_data_dim,
        target_dim,
    )

    # The raw coverage is an overlap multiplicity that can exceed 1 for an
    # overturning (non-monotonic) coordinate, where several source layers map
    # onto one target bin. For a coverage threshold we need a [0, 1]
    # fraction: a bin whose whole extent is covered (multiplicity >= 1) is
    # fully represented and must pass every threshold, so clip at 1.
    coverage_fraction = np.minimum(coverage, 1.0)

    # A target bin is declared missing when its coverage is below the
    # declared threshold. Zero coverage is therefore always missing (it also
    # has zero remapped effective volume), so a zero concentration can never
    # be emitted for an uncovered bin.
    insufficient = np.isnan(coverage_fraction) | (
        coverage_fraction < coverage_threshold
    )

    # avoid division-by-zero warnings: bins without any remapped volume are
    # masked before the ratio is taken
    valid = (~insufficient) & (effective_volume > 0)
    concentration = xr.where(
        valid, remapped_amount / effective_volume.where(valid), np.nan
    )

    # naming: inherit the input name and apply the user suffix
    if da.name:
        concentration.name = da.name + suffix
        effective_volume.name = f"{da.name}_effective_volume{suffix}"
    else:
        effective_volume.name = f"effective_volume{suffix}"

    # The effective volume is a physical remapped field: a target bin without
    # any contributing source cell genuinely has *zero* effective volume (the
    # conservative kernel leaves such bins as NaN). Fill those with zero so
    # the returned volume is finite everywhere and can be registered directly
    # as a grid metric -- xarray's weighted reductions reject NaN weights, and
    # a zero weight makes `average`/`integrate` skip exactly these bins. The
    # missingness of the concentration itself is carried by its own NaN mask
    # (driven by `coverage_threshold`), not by the volume.
    effective_volume = effective_volume.fillna(0.0)

    if return_effective_volume:
        return concentration, effective_volume
    return concentration


def target_grid(source_grid, axis_name, transformed, effective_volume):
    """Build a target `xgcm.Grid` with the remapped effective volume registered.

    An intensive conservative remap produces values on a new coordinate whose
    target bins may only be partially covered by the source grid. The remapped
    effective volume returned by
    :func:`transform` (with ``return_effective_volume=True``) encodes exactly
    that coverage. Registering it as the metric along `axis_name` on the
    target grid ensures that subsequent `Grid.average` and `Grid.integrate`
    calls weight with the same effective volume that was used to compute the
    remapped concentration, instead of silently assuming full coverage.

    Parameters
    ----------
    source_grid : xgcm.Grid
        The grid object used for the transformation. Its axes (other than the
        transformed `axis_name`), coordinates and metrics are carried over.
    axis_name : str
        Name of the axis along which the transformation was performed.
    transformed : xr.DataArray
        The concentration returned by :func:`transform`. Only its dimensions
        and the name of the transformed coordinate are used.
    effective_volume : xr.DataArray
        The effective volume returned together with `transformed`. It is
        registered as the metric for `axis_name` on the target grid.

    Returns
    -------
    xgcm.Grid
        A new grid whose transformed axis has its center at the new target
        coordinate and whose metric along `axis_name` is `effective_volume`.

    Examples
    --------
    >>> conc, veff = grid.transform(
    ...     da, "Z", target, method="conservative", quantity="intensive",
    ...     return_effective_volume=True,
    ... )
    >>> tgrid = target_grid(grid, "Z", conc, veff)
    >>> tgrid.average(conc, "Z")  # weighted with the remapped coverage
    """
    # local import to avoid a circular import at module load time
    from .grid import Grid

    # The transformed dimension is the new coordinate introduced by the
    # remap. For a multi-dimensional field the transform axis can end up at
    # any position (it is typically the first core dimension), so it cannot
    # be picked by index. Identify it as the output dimension not belonging
    # to any of the *other* (non-transformed) source axes.
    other_axis_dims = set()
    for other_name, other_ax in source_grid.axes.items():
        if other_name != axis_name:
            other_axis_dims.update(other_ax.coords.values())
    target_candidates = [d for d in transformed.dims if d not in other_axis_dims]
    if len(target_candidates) != 1:
        raise ValueError(
            "Could not uniquely determine the transformed coordinate from "
            f"{transformed.dims!r} (candidates: {target_candidates!r}). "
            "`transformed` must be the concentration returned by "
            "`Grid.transform` along `axis_name`."
        )
    target_dim = target_candidates[0]

    # carry over the axis/coordinate specification, replacing the transformed
    # axis with a center-only definition on the new target coordinate
    coords = {}
    for ax_name, ax in source_grid.axes.items():
        if ax_name == axis_name:
            coords[ax_name] = {"center": target_dim}
        else:
            coords[ax_name] = dict(ax.coords)

    # The transformed coordinate can reuse the *name* of a source outer/index
    # coordinate (e.g. "zc") while carrying different values (target cell
    # centers rather than source bounds). Building the target dataset must
    # therefore avoid index alignment against the source dataset: reset the
    # transformed coordinate to a plain dimension and re-attach the target
    # cell centers explicitly.
    target_centers = transformed.coords[target_dim]
    transformed = transformed.reset_coords(drop=True)
    effective_volume = effective_volume.reset_coords(drop=True)

    # a metric variable has to live in the grid's dataset. The effective
    # volume is finite everywhere (zero on a target bin that receives no
    # source contribution); `fillna` is a defensive guard so a registered
    # metric can never carry NaN weights, which xarray's weighted reductions
    # would reject. A zero weight makes `average`/`integrate` skip exactly
    # the bins with no contributing volume, while the concentration's own
    # NaN mask handles bins rejected by `coverage_threshold`.
    volume_name = effective_volume.name or "effective_volume"
    effective_volume = effective_volume.rename(volume_name).fillna(0.0)

    # start from the source dataset but drop any coordinate sharing the
    # target dimension's name (it would otherwise misalign the assignment)
    source_ds = source_grid._ds
    if target_dim in source_ds.coords:
        source_ds = source_ds.reset_index(target_dim, drop=True)
    ds = source_ds.assign(
        {
            transformed.name
            or "transformed": transformed,
            volume_name: effective_volume,
        }
    ).assign_coords({target_dim: target_centers})

    # carry over existing metrics (by variable name) and register the
    # effective volume for the transformed axis, overwriting any inherited
    # metric for that axis combination
    metrics = {
        tuple(axes): [mv.name for mv in metric_vars]
        for axes, metric_vars in source_grid._metrics.items()
    }
    metric_axes = tuple([axis_name])
    existing = metrics.get(metric_axes, [])
    if volume_name in existing:
        existing = [name for name in existing if name != volume_name]
    metrics[metric_axes] = existing + [volume_name]

    padding = {ax_name: ax.padding for ax_name, ax in source_grid.axes.items()}
    padding = {k: v for k, v in padding.items() if v is not None}

    target = Grid(
        ds,
        coords=coords,
        metrics=metrics,
        padding=padding if padding else None,
        autoparse_metadata=False,
    )
    # `ds.assign` may keep a same-named variable carried over from the source
    # dataset; overwrite the inherited registration so the zero-filled
    # effective volume (not a stale NaN-bearing one) is the target metric.
    target.set_metrics(metric_axes, [volume_name], overwrite=True)
    return target
