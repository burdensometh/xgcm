"""Tests for transform logic.
The tests are split into three levels:
low level: functions operating on numpy(numba) level (located in transform.py)
mid level: function wrappers that operate on xarray objects (located in transform.py)
high level: API implementations in the grid object (located in grid.py)
"""

# TODO:
# - performance test? It would be nice to have these right after so we can track the performance as this feature evolves

import dask  # type: ignore
import numpy as np
import pytest
import xarray as xr

from xgcm.grid import Grid

try:
    import numba  # type: ignore

    from ..transform import (
        conservative_coverage,
        conservative_interpolation,
        coverage_1d_conservative,
        interp_1d_conservative,
        interp_1d_linear,
        linear_interpolation,
        target_grid,
    )
except ImportError:
    numba = None


"""1D Test datasets for various transformations.
This part of the module is organized as follows:

1. A nested dictionary with all parameters needed to construct a set of test datasets
2. Fixtures that use a construct_test_source_data function and all or a subset of the dictionary entries from 1. to returns:
     (source_dataset, grid_kwargs, target, transform_kwargs, expected_dataarray)
3. Test functions

"""

cases = {
    "linear_depth_depth": {
        "source_coord": ("z", [5, 25, 60]),
        "source_data": (
            "data",
            [0.23246861, 0.45175654, 0.58320681],
        ),  # random numbers
        "target_coord": ("z", [0, 7, 30, 60, 70]),
        "target_data": ("z", [0, 7, 30, 60, 70]),
        "expected_coord": ("z", [0, 7, 30, 60, 70]),
        "expected_data": (
            "data",
            np.interp(
                [0, 7, 30, 60, 70],
                [5, 25, 60],
                [0.23246861, 0.45175654, 0.58320681],
            ),
        ),  # same source as in `source_data`
        "expected_data_mask_index": [0, -1],
        "expected_data_mask_value": np.nan,
        "grid_kwargs": {"coords": {"Z": {"center": "z"}}, "autoparse_metadata": False},
        "transform_kwargs": {"mask_edges": True, "method": "linear"},
    },
    "linear_depth_depth_nomask": {
        "source_coord": ("z", [5, 25, 60]),
        "source_data": (
            "data",
            [0.23246861, 0.45175654, 0.58320681],
        ),  # random numbers
        "target_coord": ("z", [0, 7, 30, 60, 70]),
        "target_data": ("z", [0, 7, 30, 60, 70]),
        "expected_coord": ("z", [0, 7, 30, 60, 70]),
        "expected_data": (
            "data",
            np.interp(
                [0, 7, 30, 60, 70],
                [5, 25, 60],
                [0.23246861, 0.45175654, 0.58320681],
            ),
        ),  # same source as in `source_data`
        "grid_kwargs": {"coords": {"Z": {"center": "z"}}, "autoparse_metadata": False},
        "transform_kwargs": {"mask_edges": False, "method": "linear"},
    },
    "linear_depth_depth_renamed": {
        "source_coord": ("test", [5, 25, 60]),
        "source_data": (
            "data",
            [0.23246861, 0.45175654, 0.58320681],
        ),  # random numbers
        "target_coord": ("something", [0, 7, 30, 60, 70]),
        "target_data": ("something", [0, 7, 30, 60, 70]),
        "expected_coord": ("something", [0, 7, 30, 60, 70]),
        "expected_data": (
            "data_test_suffix",
            np.interp(
                [0, 7, 30, 60, 70],
                [5, 25, 60],
                [0.23246861, 0.45175654, 0.58320681],
            ),
        ),  # same source as in `source_data`
        "grid_kwargs": {
            "coords": {"Z": {"center": "test"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "mask_edges": False,
            "method": "linear",
            "suffix": "_test_suffix",
        },
    },
    "linear_depth_depth_nomask_multidim_target": {
        "source_coord": ("z", [5, 25, 60]),
        "source_data": (
            "data",
            [0.23246861, 0.45175654, 0.58320681],
        ),  # random numbers
        "source_additional_data_coord": ("z", [5, 25, 60]),
        "source_additional_data": ("depth", [5, 25, 60]),
        # 2D target
        "target_dims": (
            "eta_rho",
            "s_rho",
        ),  # eta_rho: horizontal dimension; s_rho: vertical dimension
        "target_coord": (
            "layer_depth_rho",
            np.array([[0, 7, 30, 60, 70], [5, 10, 25, 50, 100]]),
        ),
        "target_data": (
            "layer_depth_rho",
            np.array([[0, 7, 30, 60, 70], [5, 10, 25, 50, 100]]),
        ),
        "expected_dims": ("eta_rho", "s_rho"),
        "expected_coord": (
            "layer_depth_rho",
            np.array([[0, 7, 30, 60, 70], [5, 10, 25, 50, 100]]),
        ),
        "expected_data": (
            "data",
            np.array(
                [
                    np.interp(
                        [0, 7, 30, 60, 70],
                        [5, 25, 60],
                        [0.23246861, 0.45175654, 0.58320681],
                    ),
                    np.interp(
                        [5, 10, 25, 50, 100],
                        [5, 25, 60],
                        [0.23246861, 0.45175654, 0.58320681],
                    ),
                ]
            ),  # 2D interpolation for both sets
        ),
        "grid_kwargs": {"coords": {"Z": {"center": "z"}}, "autoparse_metadata": False},
        "transform_kwargs": {
            "mask_edges": False,
            "method": "linear",
            "target_data": "depth",
            "target_dim": "s_rho",
        },
    },
    "linear_depth_depth_multidim_target": {
        "source_coord": ("z", [5, 25, 60]),
        "source_data": (
            "data",
            [0.23246861, 0.45175654, 0.58320681],
        ),  # random numbers
        "source_additional_data_coord": ("z", [5, 25, 60]),
        "source_additional_data": ("depth", [5, 25, 60]),
        # 2D target
        "target_dims": (
            "eta_rho",
            "s_rho",
        ),  # eta_rho: horizontal dimension; s_rho: vertical dimension
        "target_coord": (
            "layer_depth_rho",
            np.array([[0, 7, 30, 60, 70], [5, 10, 25, 50, 100]]),
        ),
        "target_data": (
            "layer_depth_rho",
            np.array([[0, 7, 30, 60, 70], [5, 10, 25, 50, 100]]),
        ),
        "expected_dims": ("eta_rho", "s_rho"),
        "expected_coord": (
            "layer_depth_rho",
            np.array([[0, 7, 30, 60, 70], [5, 10, 25, 50, 100]]),
        ),
        "expected_data": (
            "data",
            np.array(
                [
                    np.interp(
                        [0, 7, 30, 60, 70],
                        [5, 25, 60],
                        [0.23246861, 0.45175654, 0.58320681],
                    ),
                    np.interp(
                        [5, 10, 25, 50, 100],
                        [5, 25, 60],
                        [0.23246861, 0.45175654, 0.58320681],
                    ),
                ]
            ),  # 2D interpolation for both sets
        ),
        "expected_data_mask_index": [[0, 0], [0, -1], [1, -1]],
        "expected_data_mask_value": np.nan,
        "grid_kwargs": {"coords": {"Z": {"center": "z"}}, "autoparse_metadata": False},
        "transform_kwargs": {
            "mask_edges": True,
            "method": "linear",
            "target_data": "depth",
            "target_dim": "s_rho",
        },
    },
    "conservative_depth_depth_multidim_target": {
        "source_coord": ("z", [5, 25, 60]),
        "source_bounds_coord": ("zc", [0, 10, 50, 75]),
        "source_data": ("data", [1, 4, 0]),
        "source_additional_data_coord": (
            "zc",
            [0, 10, 50, 75],
        ),
        "source_additional_data": ("zc", [0, 10, 50, 75]),
        # 2D target
        "target_dims": (
            "eta_rho",
            "s_w",
        ),  # eta_rho: horizontal dimension; s_w: vertical dimension
        "target_coord": (
            "interface_depth_rho",
            np.array([[0, 1, 10, 50, 80], [0, 5, 20, 30, 100]]),
        ),
        "target_data": (
            "interface_depth_rho",
            np.array([[0, 1, 10, 50, 80], [0, 5, 20, 30, 100]]),
        ),
        "expected_dims": ("eta_rho", "s_w"),
        "expected_coord": (
            "interface_depth_rho",
            np.array([[0.5, 5.5, 30, 65], [2.5, 12.5, 25, 65]]),
        ),
        "expected_data": (
            "data",
            np.array([[0.1, 0.9, 4.0, 0.0], [0.5, 1.5, 1.0, 2.0]]),
        ),
        "grid_kwargs": {
            "coords": {"Z": {"center": "z", "outer": "zc"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "method": "conservative",
            "target_data": "zc",
            "target_dim": "s_w",
        },
    },
    # example of interpolating onto a tracer that increases with depth
    # but with inverted target
    "linear_depth_dens": {
        "source_coord": ("depth", [20, 40, 60, 80, 100, 120]),
        "source_data": ("data", [1, 4, 6, 2, 0, -3]),
        "source_additional_data_coord": ("depth", [20, 40, 60, 80, 100, 120]),
        "source_additional_data": ("dens", [1, 5, 10, 20, 24, 35]),
        "target_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "target_data": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_data": (
            "data",
            [1.0, 4.0, 6.0, 5.6, 4.0, 2.0, -0.272727, -0.818182],
        ),
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "method": "linear",
            "target_data": "dens",
            "mask_edges": False,
        },
    },
    # example of interpolating onto a tracer that descreases with depth
    "linear_depth_temp": {
        "source_coord": ("depth", [20, 40, 60, 80, 100, 120]),
        "source_data": ("data", [-3, 0, 2, 6, 4, 1]),
        "source_additional_data_coord": ("depth", [20, 40, 60, 80, 100, 120]),
        "source_additional_data": ("temp", [35, 24, 20, 10, 5, 1]),
        "target_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "target_data": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_data": (
            "data",
            [1.0, 4.0, 6.0, 5.6, 4.0, 2.0, -0.272727, -0.818182],
        ),
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth"}},
            "autoparse_metadata": False,
        },
        # "error": True,  # this currently fails but shouldnt
        "transform_kwargs": {
            "method": "linear",
            "target_data": "temp",
            "mask_edges": False,
        },
    },
    # example of interpolating onto a tracer that descreases with depth
    "linear_depth_temp_missing_values": {
        "source_coord": ("depth", [20, 40, 60, 80, 100, 120]),
        "source_data": ("data", [-3, 0, 2, 6, 4, 1]),
        "source_additional_data_coord": ("depth", [20, 40, 60, 80, 100, 120]),
        "source_additional_data": ("temp", [np.nan, 24, 20, 10, 5, 1]),
        "target_coord": ("something", [0, 5, 10, 11, 15, 20]),
        "target_data": ("something", [0, 5, 10, 11, 15, 20]),
        "expected_coord": ("something", [0, 5, 10, 11, 15, 20]),
        "expected_data": (
            "data",
            [1.0, 4.0, 6.0, 5.6, 4.0, 2.0],
        ),
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth"}},
            "autoparse_metadata": False,
        },
        # "error": True,  # this currently fails but shouldnt
        "transform_kwargs": {
            "method": "linear",
            "target_data": "temp",
            "mask_edges": False,
        },
    },
    "linear_depth_negative_dens": {
        "source_coord": ("depth", [-5, -25, -60, -80, -100, -120]),
        "source_data": ("data", [1, 4, 6, 2, 0, -3]),
        "source_additional_data_coord": ("depth", [-5, -25, -60, -80, -100, -120]),
        "source_additional_data": ("dens", [1, 5, 10, 20, 24, 35]),
        "target_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "target_data": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_data": (
            "data",
            [1.0, 4.0, 6.0, 5.6, 4.0, 2.0, -0.272727, -0.818182],
        ),
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "method": "linear",
            "target_data": "dens",
            "mask_edges": False,
        },
    },
    # example of interpolating onto a tracer that increases with depth
    # with masked values
    "linear_depth_dens_masked": {
        "source_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_data": ("data", [1, 4, 6, 2, 0, -3]),
        "source_additional_data_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_additional_data": ("dens", [1, 5, 10, 20, 24, 35]),
        "target_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "target_data": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_data": (
            "data",
            [np.nan, 4.0, 6.0, 5.6, 4.0, 2.0, -0.272727, -0.818182],
        ),
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "method": "linear",
            "target_data": "dens",
            "mask_edges": True,
        },
    },
    # example of interpolating onto a tracer that increases with depth
    # but with inverted target
    "linear_depth_dens_reverse": {
        "source_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_data": ("data", [1, 4, 6, 2, 0, -3]),
        "source_additional_data_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_additional_data": ("dens", [1, 5, 10, 20, 24, 35]),
        "target_coord": ("something", [27, 25, 20, 15, 11, 10, 5, 0]),
        "target_data": ("something", [27, 25, 20, 15, 11, 10, 5, 0]),
        "expected_coord": ("something", [27, 25, 20, 15, 11, 10, 5, 0]),
        "expected_data": (
            "data",
            [-0.818182, -0.272727, 2.0, 4.0, 5.6, 6.0, 4.0, 1.0],
        ),
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "method": "linear",
            "target_data": "dens",
            "mask_edges": False,
        },
    },
    # example of log-transformed linear interpolation, tested against MetPy's
    # log_interpolate_1d
    "log_sigma_pressure": {
        "source_coord": ("sigma", [0.9969, 0.9558, 0.8631, 0.7046, 0.5117]),
        "source_data": (
            "data",
            [271.75452, 272.79956, 274.8517, 279.22043, 296.48782],
        ),
        "source_additional_data_coord": (
            "sigma",
            [0.9969, 0.9558, 0.8631, 0.7046, 0.5117],
        ),
        "source_additional_data": (
            "pressure",
            [100180.625, 96250.0, 87369.14, 72186.66, 53718.586],
        ),
        "target_coord": ("pressure", [1.0e5, 8.5e4, 7.0e4]),
        "target_data": ("pressure", [1.0e5, 8.5e4, 7.0e4]),
        "expected_coord": ("pressure", [1.0e5, 8.5e4, 7.0e4]),
        "expected_data": ("data", [271.80164362, 275.48088011, 281.01791239]),
        "grid_kwargs": {
            "coords": {"Z": {"center": "sigma"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "target_data": "pressure",
            "mask_edges": True,
            "method": "log",
        },
    },
    "conservative_depth_depth": {
        "source_coord": ("z", [5, 25, 60]),
        "source_bounds_coord": ("zc", [0, 10, 50, 75]),
        "source_data": ("data", [1, 4, 0]),
        "target_coord": ("z", [0, 1, 10, 50, 80]),
        "target_data": ("z", [0, 1, 10, 50, 80]),
        "expected_coord": ("z", [0.5, 5.5, 30, 65]),
        "expected_data": (
            "data",
            [0.1, 0.9, 4.0, 0.0],
        ),  # same source as in `source_data`
        "grid_kwargs": {
            "coords": {"Z": {"center": "z", "outer": "zc"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {"method": "conservative"},
    },
    "conservative_depth_depth_rename": {
        "source_coord": ("depth", [5, 25, 60]),
        "source_bounds_coord": ("depth_bnds", [0, 10, 50, 75]),
        "source_data": ("data", [1, 4, 0]),
        "target_coord": ("something", [0, 1, 10, 50, 80]),
        "target_data": ("something", [0, 1, 10, 50, 80]),
        "expected_coord": ("something", [0.5, 5.5, 30, 65]),
        "expected_data": (
            "data_test_suffix",
            [0.1, 0.9, 4.0, 0.0],
        ),  # same source as in `source_data`
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth", "outer": "depth_bnds"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "method": "conservative",
            "suffix": "_test_suffix",
        },
    },
    # This works but is an uncommon case, where the 'tracer' which is the target
    # is located on the cell bounds
    "conservative_depth_dens_on_bounds": {
        "source_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_bounds_coord": ("depth_bnds", [0, 10, 30, 70, 90, 110, 170]),
        "source_data": ("data", [1, 4, 6, 2, 0, -3]),
        "source_additional_data_coord": (
            "depth_bnds",
            [0, 10, 30, 70, 90, 110, 170],
        ),
        "source_additional_data": ("dens", [1, 5, 10, 20, 24, 35, 37]),
        "target_coord": ("dens", [0, 5, 38]),
        "target_data": ("dens", [0, 5, 38]),
        "expected_coord": ("dens", [2.5, 21.5]),
        "expected_data": (
            "data",
            [1, 9],
        ),
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth", "outer": "depth_bnds"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "method": "conservative",
            "target_data": "dens",
        },
    },
    # missing values in the target
    "conservative_depth_dens_missing_values_bottom": {
        "source_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_bounds_coord": ("depth_bnds", [0, 10, 30, 70, 90, 110, 170]),
        "source_data": ("data", [1, 4, 6, 2, 0, np.nan]),
        "source_additional_data_coord": (
            "depth_bnds",
            [0, 10, 30, 70, 90, 110, 170],
        ),
        "source_additional_data": ("dens", [1, 5, 10, 20, 24, np.nan, np.nan]),
        "target_coord": ("dens", [0, 5, 38]),
        "target_data": ("dens", [0, 5, 38]),
        "expected_coord": ("dens", [2.5, 21.5]),
        "expected_data": (
            "data",
            [1, 12],
        ),
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth", "outer": "depth_bnds"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "method": "conservative",
            "target_data": "dens",
        },
    },
    "conservative_depth_dens_missing_values_top": {
        "source_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_bounds_coord": ("depth_bnds", [0, 10, 30, 70, 90, 110, 170]),
        "source_data": ("data", [np.nan, 4, 6, 2, 0, -3]),
        "source_additional_data_coord": (
            "depth_bnds",
            [0, 10, 30, 70, 90, 110, 170],
        ),
        "source_additional_data": ("dens", [np.nan, np.nan, 10, 20, 24, 35, 37]),
        "target_coord": ("dens", [0, 5, 38]),
        "target_data": ("dens", [0, 5, 38]),
        "expected_coord": ("dens", [2.5, 21.5]),
        "expected_data": (
            "data",
            [np.nan, 9],
        ),
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth", "outer": "depth_bnds"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "method": "conservative",
            "target_data": "dens",
        },
    },
    # decreasing tracer (e.g. temp)
    "conservative_depth_temp_on_bounds": {
        "source_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_bounds_coord": ("depth_bnds", [0, 10, 30, 70, 90, 110, 170]),
        "source_data": ("data", [-3, 0, 2, 6, 4, 1]),
        "source_additional_data_coord": (
            "depth_bnds",
            [0, 10, 30, 70, 90, 110, 170],
        ),
        "source_additional_data": ("temp", [30, 25, 20, 15, 10, 5, 0]),
        "target_coord": ("temp", [0, 5, 36]),
        "target_data": ("temp", [0, 5, 36]),
        "expected_coord": ("temp", [2.5, 20.5]),
        "expected_data": (
            "data",
            [1, 9],
        ),
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth", "outer": "depth_bnds"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "method": "conservative",
            "target_data": "temp",
        },
    },
    # Trying to make sure that these results are the same as `conservative_depth_temp_on_bounds`
    "conservative_depth_temp": {
        "source_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_bounds_coord": ("depth_bnds", [0, 10, 30, 70, 90, 110, 170]),
        "source_data": ("data", [-3, 0, 2, 6, 4, 1]),
        "source_additional_data_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_additional_data": ("temp", [27.5, 22.5, 17.5, 12.5, 7.5, 2.5]),
        "target_coord": ("temp", [0, 5, 36]),
        "target_data": ("temp", [0, 5, 36]),
        "expected_coord": ("temp", [2.5, 20.5]),
        "expected_data": (
            "data",
            [1, 9],
        ),
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth", "outer": "depth_bnds"}},
            "autoparse_metadata": False,
        },
        "error": True,  # this will still fail in the mid level tests but should succeed in the high level version (due to implementing interp on the high level method)
        "transform_kwargs": {
            "method": "conservative",
            "target_data": "temp",
        },
    },
    # Check for bugfix. Tranform onto column with two or more equal values.
    "conservative_depth_dens_equalvalues": {
        "source_coord": ("depth", [0, 20]),
        "source_bounds_coord": ("depth_bnds", [10, 30, 50]),
        "source_data": ("data", [100, 2210]),
        "source_additional_data_coord": ("depth_bnds", [10, 30, 50]),
        "source_additional_data": ("dens", [2, 2, 1.9]),
        "target_coord": ("dens", [1.9, 1.95, 3]),
        "target_data": ("dens", [1.9, 1.95, 3]),
        "expected_coord": ("dens", [1.925, 2.475]),
        "expected_data": (
            "data",
            [2210 / 2, 100 + 2210 / 2],
        ),
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth", "outer": "depth_bnds"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "method": "conservative",
            "target_data": "dens",
        },
    },
    # adopted from G MacGilchrist's example here: https://github.com/gmacgilchrist/so_decadal_variability/blob/master/dev_nonmonotonic-test.ipynb
    "conservative_depth_dens_nonmono_edge": {
        "source_coord": ("depth", [10, 30, 50, 70, 90]),
        "source_bounds_coord": ("depth_bnds", [0, 20, 40, 60, 80, 100]),
        "source_data": ("data", [2200, 2600, 3000, 3400, 3800]),
        "source_additional_data_coord": ("depth_bnds", [0, 20, 40, 60, 80, 100]),
        "source_additional_data": ("dens", [1, 3, 5, 7, 5, 3]),
        "target_coord": ("dens", [0, 2, 4, 6, 8]),
        "target_data": ("dens", [0, 2, 4, 6, 8]),
        "expected_coord": ("dens", [1, 3, 5, 7]),
        "expected_data": (
            "data",
            [1100.0, 4300.0, 6400.0, 3200.0],
        ),
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth", "outer": "depth_bnds"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "method": "conservative",
            "target_data": "dens",
        },
    },
    # tests for failcase with non-ordered target values.
    "linear_reversed_target": {
        "source_coord": ("depth", [1, 2, 3]),
        "source_bounds_coord": ("depth_bnds", [0.5, 1.5, 2.5, 3.5]),
        "source_data": ("data", [10, 20, 30]),
        "source_additional_data_coord": ("depth", [1, 2, 3]),
        "source_additional_data": ("dens", [2, 4, 6]),
        "target_coord": ("dens", [6, 4, 2]),
        "target_data": ("dens", [6, 4, 2]),
        "expected_coord": ("dens", [6, 4, 2]),
        "expected_data": (
            "data",
            [30, 20, 10],
        ),
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth", "outer": "depth_bnds"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "method": "linear",
            "target_data": "dens",
        },
    },
    "conservative_reversed_target": {
        "source_coord": ("depth", [1, 2, 3]),
        "source_bounds_coord": ("depth_bnds", [0.5, 1.5, 2.5, 3.5]),
        "source_data": ("data", [10, 20, 30]),
        "source_additional_data_coord": ("depth_bnds", [0.5, 1.5, 2.5, 3.5]),
        "source_additional_data": ("dens", [1, 3, 5, 7]),
        "target_coord": ("dens", [7, 5, 3, 1]),
        "target_data": ("dens", [7, 5, 3, 1]),
        "expected_coord": ("dens", [6, 4, 2]),
        "expected_data": (
            "data",
            [30, 20, 10],
        ),
        "error": True,  # this will still fail in the mid level tests but should succeed in the high level version (due to implementing interp on the high level method)
        "grid_kwargs": {
            "coords": {"Z": {"center": "depth", "outer": "depth_bnds"}},
            "autoparse_metadata": False,
        },
        "transform_kwargs": {
            "method": "conservative",
            "target_data": "dens",
        },
    },
}


def construct_test_source_data(case_param_dict):
    """create test components from `cases` dictionary parameters"""
    # make sure the original dict is not modified
    case_param_dict = {k: v for k, v in case_param_dict.items()}

    def _construct_da(param_dict, prefix):
        data = param_dict[prefix + "_data"][1]

        if f"{prefix}_dims" in param_dict.keys():
            dims = param_dict[prefix + "_dims"]
            coords = {
                param_dict[prefix + "_coord"][0]: (
                    param_dict[prefix + "_dims"],
                    param_dict[prefix + "_coord"][1],
                )
            }
        else:
            dims = [param_dict[prefix + "_coord"][0]]
            coords = {
                param_dict[prefix + "_coord"][0]: param_dict[prefix + "_coord"][1]
            }

        return xr.DataArray(
            data, dims=dims, coords=coords, name=param_dict[prefix + "_data"][0]
        )

    def _construct_ds(param_dict, prefix):
        da = _construct_da(param_dict, prefix)
        ds = xr.Dataset(data_vars={da.name: da})

        # Add additional data
        if (
            prefix + "_additional_data" in param_dict.keys()
            and prefix + "_additional_data_coord" in param_dict.keys()
        ):
            add_data = param_dict[prefix + "_additional_data"]
            add_data_coord = param_dict[prefix + "_additional_data_coord"]

            ds[add_data[0]] = xr.DataArray(
                add_data[1],
                dims=[add_data_coord[0]],
                coords={add_data_coord[0]: add_data_coord[1]},
            )

        # add additional coords (bounds)
        if prefix + "_bounds_coord" in param_dict.keys():
            bounds = param_dict[prefix + "_bounds_coord"]
            ds = ds.assign_coords({bounds[0]: bounds[1]})

        # mask values from the output
        if prefix + "_data_mask_index" in param_dict.keys():
            for ii in param_dict[prefix + "_data_mask_index"]:
                if isinstance(ii, (list, tuple)):  # Multi-dimensional index case
                    ds.data.data[tuple(ii)] = param_dict[prefix + "_data_mask_value"]
                else:  # One-dimensional index case
                    ds.data.data[ii] = param_dict[prefix + "_data_mask_value"]
        return ds

    source = _construct_ds(case_param_dict, "source")
    expected = _construct_ds(case_param_dict, "expected")
    target = _construct_da(case_param_dict, "target")

    # parse the 'target_data' from the actual source
    transform_kwargs = {k: v for k, v in case_param_dict["transform_kwargs"].items()}
    if "target_data" in transform_kwargs.keys():
        if transform_kwargs["target_data"] is not None:
            transform_kwargs["target_data"] = source[
                transform_kwargs["target_data"]
            ].copy(deep=True)

    error_flag = case_param_dict.pop("error", None)

    return (
        source,
        {k: v for k, v in case_param_dict["grid_kwargs"].items()},
        target,
        {k: v for k, v in transform_kwargs.items()},
        expected,
        error_flag,
    )


# TODO: once conservative method is implemented for multi-dimensional
# targets, remove the exception from this list
@pytest.fixture(
    params=[
        c
        for c in list(cases.keys())
        if not ("conservative" in c and "multidim_target" in c)
    ],
)
def all_cases_except_conservative_multidim(request):
    return construct_test_source_data(cases[request.param])


@pytest.fixture(
    params=[
        c
        for c in list(cases.keys())
        if ("conservative" in c and "multidim_target" in c)
    ],
)
def conservative_multidim_cases(request):
    return construct_test_source_data(cases[request.param])


@pytest.fixture(
    params=[
        c for c in list(cases.keys()) if "linear" in c and "multidim_target" not in c
    ],
)
def linear_cases(request):
    return construct_test_source_data(cases[request.param])


@pytest.fixture(
    params=[
        c
        for c in list(cases.keys())
        if "conservative" in c and "multidim_target" not in c
    ],
)
def conservative_cases(request):
    return construct_test_source_data(cases[request.param])


@pytest.fixture(
    # scope="module",
    params=[
        "conservative_depth_dens_nonmono_edge",
        "linear_depth_dens",
        "linear_depth_depth",
        "conservative_depth_temp",
    ],  # just pick a few to test the broadcasting and dask
)
def multidim_cases(request):
    return construct_test_source_data(cases[request.param])


@pytest.fixture(
    params=[
        "linear_depth_depth_nomask_multidim_target",
        "linear_depth_depth_multidim_target",
    ],
)
def spatially_varying_vertical_target_coord_cases(request):
    return construct_test_source_data(cases[request.param])


"""Test suite."""


def _parse_dim(da):
    # utility that check that the source array has only one dim and return that
    assert len(da.dims) == 1
    return list(da.dims)[0]


"""Low level tests"""


@pytest.mark.skipif(numba is None, reason="numba required")
def test_interp_1d_linear():
    nz, nx = 100, 1000
    z_vertex = np.linspace(0, 1, nz + 1)
    z = 0.5 * (z_vertex[:-1] + z_vertex[1:])
    x = 2 * np.pi * np.linspace(0, 1, nx)
    # uniformly stratified scalar
    theta = z + 0.1 * np.cos(3 * x)[:, None]
    # the scalar to interpolate
    phi = np.sin(theta) + 0.1 * np.cos(5 * x)[:, None]
    target_theta_levels = np.arange(0.2, 0.9, 0.025)
    phi_at_theta_expected = np.sin(target_theta_levels) + 0.1 * np.cos(5 * x)[:, None]

    # the action
    phi_at_theta = interp_1d_linear(phi, theta, target_theta_levels, mask_edges=False)
    np.testing.assert_allclose(phi_at_theta, phi_at_theta_expected, rtol=1e-4)


@pytest.mark.skipif(numba is None, reason="numba required")
def test_interp_1d_conservative():
    nz = 30
    dz = 10 + np.linspace(0, 90, nz - 1)
    z = np.concatenate([[0], np.cumsum(dz)])
    H = z.max()
    theta = z / H + 0.2 * np.cos(np.pi * z / H)

    nbins = 100
    theta_bins = np.linspace(theta.min() - 0.1, theta.max() + 0.1, nbins)

    # lazy way to check that it vectorizes: just copy the 1d array
    nx = 5
    dz_2d = np.tile(dz, (nx, 1))
    theta_2d = np.tile(theta, (nx, 1))

    dz_theta = interp_1d_conservative(dz_2d, theta_2d, theta_bins)

    np.testing.assert_allclose(np.nansum(dz_theta, axis=-1), np.nansum(dz, axis=-1))


@pytest.mark.skipif(numba is None, reason="numba required")
def test_interp_1d_conservative_nans_in_data():
    """We want the algorithm to ignore nans in the data, and preserve
    the integral of the non-nan values. Basically a transform to a single
    cell should always return the sum of all values along the transform axis."""
    phi = np.array([1, 2, np.nan])
    theta = np.array([30, 40, 50, 60])
    target = np.array([30, 50])

    out = interp_1d_conservative(phi, theta, target)
    np.testing.assert_allclose(np.nansum(phi), out)


@pytest.mark.skipif(numba is None, reason="numba required")
def test_conservative_nonmonotonic_target_error():
    nz = 30
    dz = 10 + np.linspace(0, 90, nz - 1)
    z = np.concatenate([[0], np.cumsum(dz)])
    H = z.max()
    theta = z / H + 0.2 * np.cos(np.pi * z / H)
    # phi = np.sin(5 * np.pi * z/H)

    theta_bins = np.array([0, -2, 4])

    # lazy way to check that it vectorizes: just copy the 1d array
    nx = 5
    dz_2d = np.tile(dz, (nx, 1))
    theta_2d = np.tile(theta, (nx, 1))
    with pytest.raises(ValueError):
        _ = interp_1d_conservative(dz_2d, theta_2d, theta_bins)


"""Mid level tests"""


@pytest.mark.skipif(numba is None, reason="numba required")
def test_linear_interpolation_target_value_error():
    """Test that linear_interpolation/conservative_interpolation throws an error when `target` is a np array"""
    (
        source,
        _,
        target,
        _,
        _,
        _,
    ) = construct_test_source_data(cases["linear_depth_depth"])

    with pytest.raises(ValueError):
        _ = linear_interpolation(source.data, source.z, target.data, "z", "z")

    (
        source,
        _,
        target,
        _,
        _,
        _,
    ) = construct_test_source_data(cases["conservative_depth_depth"])
    with pytest.raises(ValueError):
        _ = conservative_interpolation(source.data, source.z, target.data, "z", "z")


@pytest.mark.skipif(numba is None, reason="numba required")
def test_mid_level_linear(linear_cases):
    """Test the linear interpolations on the xarray wrapper level"""
    source, _, target, transform_kwargs, expected, error_flag = linear_cases

    # construct output name
    transform_kwargs.setdefault("suffix", "")
    output_name = "data" + transform_kwargs["suffix"]

    # method keyword is only for high level tests
    transform_kwargs.pop("method")

    source_dim = _parse_dim(source.data)
    target_dim = _parse_dim(target)

    # parse the target_data manually
    target_data = transform_kwargs.pop("target_data", None)
    if target_data is None:
        target_data = source[source_dim]

    if error_flag:
        with pytest.xfail():
            interpolated = linear_interpolation(
                source.data,
                target_data,
                target,
                source_dim,
                source_dim,
                target_dim,
                **transform_kwargs,
            )
    else:
        interpolated = linear_interpolation(
            source.data,
            target_data,
            target,
            source_dim,
            source_dim,
            target_dim,
            **transform_kwargs,
        )
        xr.testing.assert_allclose(interpolated, expected[output_name])
        assert interpolated.name == output_name


@pytest.mark.skipif(numba is None, reason="numba required")
def test_mid_level_conservative(conservative_cases):
    """Test the conservative interpolations on the xarray wrapper level"""
    (
        source,
        grid_kwargs,
        target,
        transform_kwargs,
        expected,
        error_flag,
    ) = conservative_cases

    # construct output name
    transform_kwargs.setdefault("suffix", "")
    output_name = "data" + transform_kwargs["suffix"]

    # method keyword is only for high level tests
    transform_kwargs.pop("method")

    source_dim = grid_kwargs["coords"]["Z"]["center"]
    bounds_dim = grid_kwargs["coords"]["Z"]["outer"]
    target_dim = _parse_dim(target)

    # parse the target_data manually
    target_data = transform_kwargs.pop("target_data", None)
    if target_data is None:
        target_data = source[bounds_dim]
    if error_flag:
        with pytest.xfail():
            transformed = conservative_interpolation(
                source.data,
                target_data,
                target,
                source_dim,
                bounds_dim,
                target_dim,
                **transform_kwargs,
            )
    else:
        transformed = conservative_interpolation(
            source.data,
            target_data,
            target,
            source_dim,
            bounds_dim,
            target_dim,
            **transform_kwargs,
        )
        xr.testing.assert_allclose(transformed, expected[output_name])
        # make sure that the extensive quantitiy is actually conserved
        xr.testing.assert_allclose(transformed.sum(), source.data.sum())
        assert transformed.name == output_name


"""High level tests"""


@pytest.mark.skipif(numba is None, reason="numba required")
def test_grid_transform(all_cases_except_conservative_multidim):
    source, grid_kwargs, target, transform_kwargs, expected, error_flag = (
        all_cases_except_conservative_multidim
    )

    axis = list(grid_kwargs["coords"].keys())[0]

    grid = Grid(source, **grid_kwargs)

    # construct output name
    transform_kwargs.setdefault("suffix", "")
    output_name = "data" + transform_kwargs["suffix"]

    # the high level routines should be able to deal with all cases (no error flag exception like in the mid level)
    transformed = grid.transform(source.data, axis, target, **transform_kwargs)
    xr.testing.assert_allclose(transformed, expected[output_name])


@pytest.mark.skipif(numba is None, reason="numba required")
def test_conservative_interp_error_if_multidim_target_dim(conservative_multidim_cases):
    source, grid_kwargs, target, transform_kwargs, expected, error_flag = (
        conservative_multidim_cases
    )

    axis = list(grid_kwargs["coords"].keys())[0]

    grid = Grid(source, **grid_kwargs)

    with pytest.raises(NotImplementedError):
        _ = grid.transform(source.data, axis, target, **transform_kwargs)


@pytest.mark.skipif(numba is None, reason="numba required")
def test_conservative_transform_explicit_target_dim():
    # Regression test: the multi-dimensional-target guard used to check
    # len(target_dim) — the length of the dimension *name* — so a 1D target
    # combined with an explicit target_dim longer than one character falsely
    # raised NotImplementedError.
    source, grid_kwargs, target, transform_kwargs, expected, error_flag = (
        construct_test_source_data(cases["conservative_depth_depth_rename"])
    )

    axis = list(grid_kwargs["coords"].keys())[0]

    grid = Grid(source, **grid_kwargs)

    (target_dim,) = target.dims
    assert len(target_dim) > 1
    transformed = grid.transform(
        source.data, axis, target, target_dim=target_dim, **transform_kwargs
    )
    output_name = "data" + transform_kwargs.get("suffix", "")
    xr.testing.assert_allclose(transformed, expected[output_name])


@pytest.mark.skipif(numba is None, reason="numba required")
def test_conservative_interp_warn_if_no_cell_bounds():
    (
        source,
        grid_kwargs,
        target,
        transform_kwargs,
        _,
        _,
    ) = construct_test_source_data(cases["conservative_depth_temp"])

    axis = list(grid_kwargs["coords"].keys())[0]

    grid = Grid(source, **grid_kwargs)
    with pytest.warns(
        UserWarning,
        match="The `target data` input is not located on the cell bounds. This method will continue with linear interpolation with repeated boundary values.",
    ):
        _ = grid.transform(source.data, axis, target, **transform_kwargs)


@pytest.mark.skipif(numba is None, reason="numba required")
def test_grid_transform_noname_data(multidim_cases):
    """Check handling of a `da` input without name"""
    source, grid_kwargs, target, transform_kwargs, _, _ = multidim_cases

    axis = list(grid_kwargs["coords"].keys())[0]

    grid = Grid(source, **grid_kwargs)

    source_da = source.data
    source_da.name = None

    # the high level routines should be able to deal with all cases (no error flag exception like in the mid level)
    transformed = grid.transform(source_da, axis, target, **transform_kwargs)
    assert transformed.name is None


@pytest.mark.skipif(numba is None, reason="numba required")
def test_grid_transform_noname_targetdata():
    """Check handling of a `target_data` input without name"""
    (
        source,
        grid_kwargs,
        target,
        transform_kwargs,
        _,
        _,
    ) = construct_test_source_data(cases["linear_depth_dens"])

    axis = list(grid_kwargs["coords"].keys())[0]

    grid = Grid(source, **grid_kwargs)

    source_da = source.data
    target_data = transform_kwargs.pop("target_data")
    target_data.name = None
    # the name of target_data is only used if `target` is provided as numpy array
    target = target.data

    # the high level routines should be able to deal with all cases (no error flag exception like in the mid level)
    with pytest.warns(UserWarning):
        transformed = grid.transform(
            source_da, axis, target, target_data=target_data, **transform_kwargs
        )
    "TRANSFORMED_DIMENSION" in transformed.dims


@pytest.mark.skipif(numba is None, reason="numba required")
def test_transform_error_periodic(multidim_cases):
    source, grid_kwargs, target, transform_kwargs, expected, error_flag = multidim_cases

    axis = list(grid_kwargs["coords"].keys())[0]

    # `transform` is not supported on periodic axes, so build a periodic grid and
    # confirm the operation raises.
    grid = Grid(source, padding="periodic", **grid_kwargs)

    with pytest.raises(ValueError):
        _ = grid.transform(source.data, axis, target, **transform_kwargs)


@pytest.mark.skipif(numba is None, reason="numba required")
def test_grid_transform_auto_naming(multidim_cases):  # only test a few cases
    """Check that the naming for the new dimension is adapted for the output if the target is not passed as xr.Dataarray"""
    source, grid_kwargs, target, transform_kwargs, expected, error_flag = multidim_cases

    axis = list(grid_kwargs["coords"].keys())[0]
    grid = Grid(source, **grid_kwargs)

    # modify the expected naming and convert target to numpy array
    target_data = transform_kwargs.setdefault("target_data", None)

    if transform_kwargs["target_data"] is None:
        # When no target_data is provided default to axis coordinates depending on method
        if transform_kwargs["method"] == "linear":
            expected_data_coord = grid.axes[axis].coords["center"]
        elif transform_kwargs["method"] == "conservative":
            expected_data_coord = grid.axes[axis].coords["outer"]
    else:
        # When target_data is provided check against the name of the dataarray
        expected_data_coord = target_data.name

    target = target.data

    transformed = grid.transform(source.data, axis, target, **transform_kwargs)
    assert expected_data_coord in transformed.coords


@pytest.mark.skipif(numba is None, reason="numba required")
@pytest.mark.parametrize("bypass_checks", [True, False])
def test_grid_transform_bypass_checks(bypass_checks):
    """Check that the bypass checks option still delivers the right results for monotonically increasing data"""
    (
        source,
        grid_kwargs,
        target,
        transform_kwargs,
        expected,
        _,
    ) = construct_test_source_data(cases["linear_depth_dens"])

    axis = list(grid_kwargs["coords"].keys())[0]
    grid = Grid(source, **grid_kwargs)

    target_data = transform_kwargs.pop("target_data", None)

    transformed = grid.transform(
        source.data,
        axis,
        target,
        target_data=target_data,
        bypass_checks=bypass_checks,
        **transform_kwargs,
    )

    xr.testing.assert_allclose(transformed, expected.data)


""" Multidimensional tests with dask scheduler """


# dask client fixtures (adapted from https://github.com/xgcm/fastjmd95/blob/2cbccf2144e73c13a91741029eb7fd273d380d28/fastjmd95/test/test_jmd95.py#L61)
@pytest.fixture
def no_client():
    return None


@pytest.fixture
def threaded_client():
    with dask.config.set(scheduler="threads"):
        yield


@pytest.fixture
def processes_client():
    with dask.config.set(scheduler="processes"):
        yield


@pytest.fixture(scope="module")
def distributed_client():
    from dask.distributed import Client, LocalCluster  # type: ignore

    cluster = LocalCluster(threads_per_worker=1, n_workers=2, processes=True)
    client = Client(cluster)
    yield
    client.close()
    del client
    cluster.close()
    del cluster


# all_clients = ["no_client", "threaded_client", "processes_client", "distributed_client"]
all_clients = ["threaded_client", "processes_client", "distributed_client", "no_client"]


@pytest.mark.skipif(numba is None, reason="numba required")
@pytest.mark.parametrize("client", all_clients)
def test_grid_transform_multidim(request, client, multidim_cases):
    # broadcast the 1d column agains some other dims and make sure that the 1d results are still valid
    source, grid_kwargs, target, transform_kwargs, expected, error_flag = multidim_cases

    na = 8
    source = source.expand_dims(a=na).copy(deep=True)

    # broadcast the target_data manually
    target_data = transform_kwargs.pop("target_data", None)
    if target_data is not None:
        target_data = target_data.expand_dims(a=na).copy(deep=True)
        if client != "no_client":
            target_data = target_data.chunk({"a": 1})

    if client != "no_client":
        source = source.chunk({"a": 1})

    # calculate the multidimensional result
    axis = list(grid_kwargs["coords"].keys())[0]
    grid = Grid(source, **grid_kwargs)

    # the high level tests should deal with all error cases
    client = request.getfixturevalue(client)

    transformed = grid.transform(
        source.data, axis, target, target_data=target_data, **transform_kwargs
    ).load()
    _, expected_broadcasted = xr.broadcast(transformed, expected)

    xr.testing.assert_allclose(transformed, expected_broadcasted.data)


@pytest.mark.skipif(numba is None, reason="numba required")
@pytest.mark.parametrize("client", all_clients)
def test_grid_transform_multidim_with_target_dim(
    request, client, spatially_varying_vertical_target_coord_cases
):
    """This test uses a vertical target coordinate that is spatially varying."""

    source, grid_kwargs, target, transform_kwargs, expected, error_flag = (
        spatially_varying_vertical_target_coord_cases
    )
    target_data = transform_kwargs.pop("target_data", None)
    grid = Grid(source, **grid_kwargs)

    # the high level tests should deal with all error cases
    client = request.getfixturevalue(client)

    axis = list(grid_kwargs["coords"].keys())[0]
    transformed = grid.transform(
        source.data, axis, target, target_data=target_data, **transform_kwargs
    ).load()
    xr.testing.assert_allclose(transformed, expected.data)


@pytest.mark.skipif(numba is None, reason="numba required")
def test_grid_transform_multidim_other_dims_error(request, multidim_cases):
    # broadcast the 1d column agains some other dims and make sure that the 1d results are still valid
    source, grid_kwargs, target, transform_kwargs, expected, error_flag = multidim_cases

    na = 3
    source = source * xr.DataArray(np.ones([na]), dims=["a"])
    # broadcast the target, but in this case
    # rename one of the dimensions of the target array, which is not along the
    # axis of transformation (this could be the case if e.g. temperature is on a different
    # x grid than velocity)
    target_data = transform_kwargs.pop("target_data", None)
    if target_data is not None:
        target_data = target_data * xr.DataArray(np.ones([na]), dims=["a_other"])

        # calculate the multidimensional result
        axis = list(grid_kwargs["coords"].keys())[0]

        grid = Grid(source, **grid_kwargs)
        with pytest.raises(ValueError):
            _ = grid.transform(
                source.data, axis, target, target_data=target_data, **transform_kwargs
            )

    else:
        # When target_data is none its taken as a 1D-coordinate, no checking needed
        pytest.skip()


@pytest.mark.skipif(numba is None, reason="numba required")
def test_chunking_dim_error():
    """Assure that error is raised when we chunk along the 'vertical' dimension"""

    (
        source,
        grid_kwargs,
        target,
        transform_kwargs,
        _,
        _,
    ) = construct_test_source_data(cases["linear_depth_dens"])

    source = source.chunk({"depth": 1})
    axis = list(grid_kwargs["coords"].keys())[0]
    grid = Grid(source, **grid_kwargs)
    with pytest.raises(ValueError):
        _ = grid.transform(source.data, axis, target, **transform_kwargs)


@pytest.mark.skipif(numba is None, reason="numba required")
def test_grid_transform_input_check():
    (
        source,
        grid_kwargs,
        target,
        transform_kwargs,
        _,
        _,
    ) = construct_test_source_data(cases["linear_depth_dens"])

    axis = list(grid_kwargs["coords"].keys())[0]

    grid = Grid(source, **grid_kwargs)

    # construct output name
    transform_kwargs.setdefault("suffix", "")

    # Make sure that a sensible error is raised if xr.Dataset is provided
    # for either one of `source`, `target` or `target_data` input arguments.
    match_msg = r"needs to be a"
    with pytest.raises(ValueError, match=r"`da` " + match_msg):
        grid.transform(source, axis, target, **transform_kwargs)

    with pytest.raises(ValueError, match=match_msg):
        grid.transform(
            source.data, axis, target.to_dataset(name="dummy"), **transform_kwargs
        )

    transform_kwargs["target_data"] = transform_kwargs["target_data"].to_dataset(
        name="dummy"
    )
    with pytest.raises(ValueError, match=match_msg):
        grid.transform(source.data, axis, target, **transform_kwargs)


"""Tests for the intensive (concentration) conservative remap mode."""


def _make_volume_grid(
    conc_values,
    volume_values,
    bounds=(0.0, 10.0, 50.0, 75.0),
    metric_name="dz",
    register_metric=True,
):
    """Build a minimal 1D vertical grid with a concentration and a volume
    (cell thickness) metric."""
    bounds = np.asarray(bounds, dtype=float)
    centers = 0.5 * (bounds[:-1] + bounds[1:])
    dim, outer = "z", "zc"
    conc = xr.DataArray(
        np.asarray(conc_values, dtype=float),
        dims=[dim],
        coords={dim: centers},
        name="conc",
    )
    volume = xr.DataArray(
        np.asarray(volume_values, dtype=float),
        dims=[dim],
        coords={dim: centers},
        name=metric_name,
    )
    ds = xr.Dataset({"conc": conc, metric_name: volume}).assign_coords(
        {outer: bounds}
    )
    metrics = {"Z": [metric_name]} if register_metric else None
    grid = Grid(
        ds,
        coords={"Z": {"center": dim, "outer": outer}},
        metrics=metrics,
        autoparse_metadata=False,
    )
    return grid, ds, conc, volume


# source cells: [0,10] c=1 v=10 ; [10,50] c=4 v=40 ; [50,75] c=2 v=25
_INTENSIVE_TARGET = np.array([0.0, 1.0, 10.0, 50.0, 80.0])


@pytest.mark.skipif(numba is None, reason="numba required")
def test_coverage_1d_conservative_basic():
    # plain overlaps: bins 0-1, 1-10, 10-50 are fully inside source cells;
    # bin 50-80 only overlaps source over 50-75 (25 of 30)
    theta = np.array([0.0, 10.0, 50.0, 75.0])
    coverage = coverage_1d_conservative(theta, _INTENSIVE_TARGET)
    np.testing.assert_allclose(coverage, [1.0, 1.0, 1.0, 25.0 / 30.0])


@pytest.mark.skipif(numba is None, reason="numba required")
def test_coverage_1d_conservative_zero_coverage():
    # a target extending beyond the source domain must report zero coverage,
    # never NaN (zero-width bins excepted), so the masking can distinguish
    # "uncovered" from "no geometry"
    theta = np.array([0.0, 10.0, 50.0, 75.0])
    target = np.array([0.0, 10.0, 50.0, 75.0, 100.0])
    coverage = coverage_1d_conservative(theta, target)
    np.testing.assert_allclose(coverage, [1.0, 1.0, 1.0, 0.0])


@pytest.mark.skipif(numba is None, reason="numba required")
def test_coverage_1d_conservative_reversed_target():
    theta = np.array([0.0, 10.0, 50.0, 75.0])
    forward = coverage_1d_conservative(theta, _INTENSIVE_TARGET)
    reversed_ = coverage_1d_conservative(theta, _INTENSIVE_TARGET[::-1])
    np.testing.assert_allclose(forward, reversed_[::-1])


@pytest.mark.skipif(numba is None, reason="numba required")
def test_coverage_matches_amount_overlap_geometry():
    # remapping an extensive field of ones times the cell extent reproduces
    # the coverage fraction, proving both kernels use the same geometry
    theta = np.array([0.0, 10.0, 50.0, 75.0])
    cell_extents = np.diff(theta)
    amount = interp_1d_conservative(cell_extents, theta, _INTENSIVE_TARGET)
    bin_widths = np.diff(_INTENSIVE_TARGET)
    coverage = coverage_1d_conservative(theta, _INTENSIVE_TARGET)
    np.testing.assert_allclose(amount / bin_widths, coverage)


@pytest.mark.skipif(numba is None, reason="numba required")
def test_coverage_nonmonotonic_is_multiplicity():
    # an overturning coordinate maps several source layers onto one target
    # bin: the geometric coverage is then a multiplicity that can exceed 1
    theta = np.array([1.0, 3.0, 5.0, 7.0, 5.0, 3.0])
    target = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    coverage = coverage_1d_conservative(theta, target)
    assert np.any(coverage > 1.0)
    # while a bin with no overlap remains exactly zero
    assert coverage[0] == 0.5  # bin 0-2 only half covered by the [1,3] layers
    assert coverage[-1] == 1.0  # bin 6-8 covered by the [5,7] layer only


@pytest.mark.skipif(numba is None, reason="numba required")
def test_conservative_coverage_mid_level():
    grid, ds, conc, volume = _make_volume_grid([1.0, 4.0, 2.0], [10.0, 40.0, 25.0])
    target = xr.DataArray(_INTENSIVE_TARGET, dims=["zc"])
    coverage = conservative_coverage(
        conc, ds["zc"], target, "z", "zc", "zc"
    )
    assert coverage.dims == ("zc",)
    np.testing.assert_allclose(coverage, [1.0, 1.0, 1.0, 25.0 / 30.0])


@pytest.mark.skipif(numba is None, reason="numba required")
def test_intensive_conservative_concentration_values():
    grid, ds, conc, volume = _make_volume_grid([1.0, 4.0, 2.0], [10.0, 40.0, 25.0])
    transformed = grid.transform(
        conc,
        "Z",
        _INTENSIVE_TARGET,
        method="conservative",
        quantity="intensive",
    )
    # fully covered bins keep the concentration of the source cell they
    # fall into; the partially covered bin is NaN at the default threshold
    np.testing.assert_allclose(transformed.values[:3], [1.0, 1.0, 4.0])
    assert np.isnan(transformed.values[3])


@pytest.mark.skipif(numba is None, reason="numba required")
def test_intensive_conservative_partial_coverage_threshold():
    grid, ds, conc, volume = _make_volume_grid([1.0, 4.0, 2.0], [10.0, 40.0, 25.0])
    # bin 50-80 is 25/30 ~ 0.833 covered
    transformed = grid.transform(
        conc,
        "Z",
        _INTENSIVE_TARGET,
        method="conservative",
        quantity="intensive",
        coverage_threshold=0.5,
    )
    np.testing.assert_allclose(transformed.values, [1.0, 1.0, 4.0, 2.0])

    # threshold above the actual coverage masks the bin again
    strict = grid.transform(
        conc,
        "Z",
        _INTENSIVE_TARGET,
        method="conservative",
        quantity="intensive",
        coverage_threshold=0.9,
    )
    assert np.isnan(strict.values[3])

    # threshold of exactly the achieved coverage is accepted (>= comparison)
    exact = grid.transform(
        conc,
        "Z",
        _INTENSIVE_TARGET,
        method="conservative",
        quantity="intensive",
        coverage_threshold=25.0 / 30.0,
    )
    np.testing.assert_allclose(exact.values[3], 2.0)


@pytest.mark.skipif(numba is None, reason="numba required")
def test_intensive_zero_coverage_never_zero_concentration():
    grid, ds, conc, volume = _make_volume_grid([1.0, 4.0, 2.0], [10.0, 40.0, 25.0])
    # last bin 75-100 has zero overlap with the source (0-75)
    target = np.array([0.0, 10.0, 50.0, 75.0, 100.0])
    transformed, effective_volume = grid.transform(
        conc,
        "Z",
        target,
        method="conservative",
        quantity="intensive",
        coverage_threshold=0.0,
        return_effective_volume=True,
    )
    np.testing.assert_allclose(transformed.values[:3], [1.0, 4.0, 2.0])
    assert np.isnan(transformed.values[3])
    # zero coverage means zero remapped volume (a physical zero, not a NaN
    # and not a zero concentration): the volume stays finite so it can be a
    # metric while the concentration carries the missing-value mask.
    assert effective_volume.values[3] == 0.0


@pytest.mark.skipif(numba is None, reason="numba required")
def test_intensive_effective_volume_values_and_mask():
    grid, ds, conc, volume = _make_volume_grid([1.0, 4.0, 2.0], [10.0, 40.0, 25.0])
    transformed, effective_volume = grid.transform(
        conc,
        "Z",
        _INTENSIVE_TARGET,
        method="conservative",
        quantity="intensive",
        return_effective_volume=True,
    )
    # effective volumes are the conservatively remapped source cell volumes
    np.testing.assert_allclose(effective_volume.values[:3], [1.0, 9.0, 40.0])
    # the partially covered bin still has a physical volume (25), while the
    # concentration is masked by the default full-coverage threshold
    np.testing.assert_allclose(effective_volume.values[3], 25.0)
    # the volume is finite everywhere so it can be registered as a metric
    assert np.all(np.isfinite(effective_volume.values))
    assert effective_volume.name.endswith("effective_volume_transformed")


@pytest.mark.skipif(numba is None, reason="numba required")
def test_intensive_missing_source_values_excluded_consistently():
    # a NaN source concentration must exclude that cell from BOTH remaps, so
    # the ratio is never contaminated by it
    grid, ds, conc, volume = _make_volume_grid(
        [1.0, np.nan, 2.0], [10.0, 40.0, 25.0]
    )
    target = np.array([0.0, 30.0, 50.0, 75.0])
    transformed = grid.transform(
        conc,
        "Z",
        target,
        method="conservative",
        quantity="intensive",
        coverage_threshold=0.0,
    )
    # bin 0-30: half of cell 0 (c=1) and half of the NaN cell 1. Only the
    # valid half contributes, so the concentration stays 1
    np.testing.assert_allclose(transformed.values[0], 1.0)
    # bin 30-50 lies entirely in the NaN cell -> missing
    assert np.isnan(transformed.values[1])
    # bin 50-75 is fully in cell 2 (c=2)
    np.testing.assert_allclose(transformed.values[2], 2.0)


@pytest.mark.skipif(numba is None, reason="numba required")
def test_intensive_conservative_nonmonotonic_target_data():
    # non-monotonic theta (density overturning): the intensive path must
    # support this case exactly like the extensive kernel, and reproduce an
    # independent amount/volume ratio computed from that kernel.
    bounds = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    centers = 0.5 * (bounds[:-1] + bounds[1:])
    volume_values = np.diff(bounds)
    volume = xr.DataArray(
        volume_values, dims=["z"], coords={"z": centers}, name="dz"
    )
    conc_values = np.array([22.0, 26.0, 30.0, 34.0, 38.0])
    conc = xr.DataArray(
        conc_values, dims=["z"], coords={"z": centers}, name="conc"
    )
    # non-monotonic density on the outer coordinate
    dens_outer = xr.DataArray(
        [1.0, 3.0, 5.0, 7.0, 5.0, 3.0], dims=["zc"]
    )
    ds = xr.Dataset({"conc": conc, "dz": volume}).assign_coords(
        zc=bounds, dens_outer=dens_outer
    )
    grid = Grid(
        ds,
        coords={"Z": {"center": "z", "outer": "zc"}},
        metrics={"Z": ["dz"]},
        autoparse_metadata=False,
    )
    target_dens = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    transformed = grid.transform(
        conc,
        "Z",
        target_dens,
        target_data=ds.dens_outer,
        method="conservative",
        quantity="intensive",
        coverage_threshold=0.0,
    )

    # independent reference: run the extensive kernel directly on the total
    # amount and on the volume, then take the ratio
    ref_amount = interp_1d_conservative(
        conc_values * volume_values, dens_outer.values, target_dens
    )
    ref_volume = interp_1d_conservative(
        volume_values, dens_outer.values, target_dens
    )
    ref_coverage = coverage_1d_conservative(
        dens_outer.values, target_dens
    )
    ref_conc = np.where(
        (ref_coverage > 0.0) & (ref_volume > 0.0),
        ref_amount / ref_volume,
        np.nan,
    )
    np.testing.assert_allclose(transformed.values, ref_conc)
    # the geometry genuinely blends several source cells into one target bin
    # under non-monotonic stratification, so at least one bin is a mixture
    assert np.any(
        (ref_conc[~np.isnan(ref_conc)] > conc_values.min())
        & (ref_conc[~np.isnan(ref_conc)] < conc_values.max())
    )


@pytest.mark.skipif(numba is None, reason="numba required")
def test_intensive_nonmonotonic_coverage_multiplicity_threshold():
    # Under an overturning coordinate several source layers map onto one
    # target bin, so the raw geometric coverage is a multiplicity that can
    # exceed 1. Such a bin is fully represented and must still pass the
    # strict full-coverage (1.0) threshold; only genuinely under-covered bins
    # are masked.
    bounds = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    centers = 0.5 * (bounds[:-1] + bounds[1:])
    dz = xr.DataArray(np.diff(bounds), dims=["z"], coords={"z": centers}, name="dz")
    conc = xr.DataArray(
        [22.0, 26.0, 30.0, 34.0, 38.0],
        dims=["z"],
        coords={"z": centers},
        name="conc",
    )
    dens_outer = xr.DataArray([1.0, 3.0, 5.0, 7.0, 5.0, 3.0], dims=["zc"])
    ds = xr.Dataset({"conc": conc, "dz": dz}).assign_coords(
        zc=bounds, dens_outer=dens_outer
    )
    grid = Grid(
        ds,
        coords={"Z": {"center": "z", "outer": "zc"}},
        metrics={"Z": ["dz"]},
        autoparse_metadata=False,
    )
    target_dens = np.array([0.0, 2.0, 4.0, 6.0, 8.0])

    # the low-level geometric coverage is explicitly a multiplicity here
    multiplicity = coverage_1d_conservative(dens_outer.values, target_dens)
    assert np.any(multiplicity > 1.0)

    strict, volume = grid.transform(
        conc,
        "Z",
        target_dens,
        target_data=ds.dens_outer,
        method="conservative",
        quantity="intensive",
        coverage_threshold=1.0,
        return_effective_volume=True,
    )
    permissive = grid.transform(
        conc,
        "Z",
        target_dens,
        target_data=ds.dens_outer,
        method="conservative",
        quantity="intensive",
        coverage_threshold=0.0,
    )
    # the first bin is only half covered and is the only strict-threshold NaN
    assert np.isnan(strict.values[0])
    assert np.all(np.isfinite(strict.values[1:]))
    # the permissive threshold fills every non-empty bin
    assert np.all(np.isfinite(permissive.values))
    # strict cannot have more valid bins than permissive
    assert np.sum(np.isfinite(strict.values)) <= np.sum(
        np.isfinite(permissive.values)
    )
    # the effective volume is finite even where the concentration is masked
    assert np.all(np.isfinite(volume.values))


@pytest.mark.skipif(numba is None, reason="numba required")
def test_target_grid_registers_effective_volume_for_average_integrate():
    # target with a fully-covered region plus a zero-coverage tail bin
    target = np.array([0.0, 10.0, 50.0, 75.0, 100.0])
    grid, ds, conc, volume = _make_volume_grid([1.0, 4.0, 2.0], [10.0, 40.0, 25.0])
    transformed, effective_volume = grid.transform(
        conc,
        "Z",
        target,
        method="conservative",
        quantity="intensive",
        coverage_threshold=0.0,
        return_effective_volume=True,
    )
    tgrid = target_grid(grid, "Z", transformed, effective_volume)

    # the registered metric lives on the target dimension and has zero weight
    # on the zero-coverage tail bin (75-100)
    metric = tgrid.get_metric(transformed, "Z")
    assert metric.dims == transformed.dims
    np.testing.assert_allclose(metric.values[:3], [10.0, 40.0, 25.0])
    assert metric.values[3] == 0.0

    # average / integrate use the same effective-volume accounting as the
    # remap, skipping both the NaN concentration and the zero-weight tail bin
    valid = np.isfinite(transformed.values)
    expected_average = np.sum(
        transformed.values[valid] * effective_volume.values[valid]
    ) / np.sum(effective_volume.values[valid])
    expected_integral = np.sum(
        transformed.values[valid] * effective_volume.values[valid]
    )
    np.testing.assert_allclose(tgrid.average(transformed, "Z"), expected_average)
    np.testing.assert_allclose(
        tgrid.integrate(transformed, "Z"), expected_integral
    )


@pytest.mark.skipif(numba is None, reason="numba required")
def test_target_grid_threshold_masked_bins_excluded_from_reductions():
    # a partially covered bin (25/30) is masked at the strict default
    # threshold but still carries its physical effective volume. The NaN
    # concentration must be excluded from average/integrate while the metric
    # stays finite and usable (no NaN weights).
    grid, ds, conc, volume = _make_volume_grid([1.0, 4.0, 2.0], [10.0, 40.0, 25.0])
    transformed, effective_volume = grid.transform(
        conc,
        "Z",
        _INTENSIVE_TARGET,
        method="conservative",
        quantity="intensive",
        return_effective_volume=True,
    )
    tgrid = target_grid(grid, "Z", transformed, effective_volume)

    metric = tgrid.get_metric(transformed, "Z")
    assert np.all(np.isfinite(metric.values))
    np.testing.assert_allclose(metric.values, [1.0, 9.0, 40.0, 25.0])

    valid = np.isfinite(transformed.values)
    expected_average = np.sum(
        transformed.values[valid] * effective_volume.values[valid]
    ) / np.sum(effective_volume.values[valid])
    np.testing.assert_allclose(tgrid.average(transformed, "Z"), expected_average)


@pytest.mark.skipif(numba is None, reason="numba required")
def test_target_grid_keeps_other_axes_and_metrics():
    # 2D grid: an X axis with its own metric must survive target_grid
    xc = np.array([0.0, 1.0, 2.0])
    dx = xr.DataArray([2.0, 3.0, 4.0], dims=["x"], coords={"x": xc}, name="dx")
    bounds = np.array([0.0, 10.0, 50.0, 75.0])
    centers = 0.5 * (bounds[:-1] + bounds[1:])
    dz = xr.DataArray(
        np.broadcast_to(np.diff(bounds), (3, 3)).copy(),
        dims=["x", "z"],
        coords={"x": xc, "z": centers},
        name="dz",
    )
    # distinct concentrations per column so reductions cannot pass by accident
    conc = xr.DataArray(
        np.array([[1.0, 4.0, 2.0], [10.0, 20.0, 30.0], [5.0, 6.0, 7.0]]),
        dims=["x", "z"],
        coords={"x": xc, "z": centers},
        name="conc",
    )
    ds = xr.Dataset({"conc": conc, "dx": dx, "dz": dz}).assign_coords(zc=bounds)
    grid = Grid(
        ds,
        coords={"X": {"center": "x"}, "Z": {"center": "z", "outer": "zc"}},
        metrics={"X": ["dx"], "Z": ["dz"]},
        autoparse_metadata=False,
    )
    transformed, effective_volume = grid.transform(
        conc,
        "Z",
        np.array([0.0, 10.0, 50.0, 75.0]),
        method="conservative",
        quantity="intensive",
        coverage_threshold=0.0,
        return_effective_volume=True,
    )
    tgrid = target_grid(grid, "Z", transformed, effective_volume)
    target_dim = [d for d in transformed.dims if d != "x"][0]

    # the transformed axis must point at the new target dimension, while the
    # horizontal axis still points at `x` (regression for dim-order detection)
    assert tgrid.axes["Z"].coords["center"] == target_dim
    assert tgrid.axes["X"].coords["center"] == "x"

    # the horizontal metric is still registered and usable
    np.testing.assert_allclose(
        tgrid.get_metric(transformed, "X").values, [2.0, 3.0, 4.0]
    )

    # vertical integration reduces the target dimension and conserves each
    # column's conc*thickness content
    vertical_integral = tgrid.integrate(transformed, "Z")
    assert target_dim not in vertical_integral.dims
    np.testing.assert_allclose(
        vertical_integral.values, (conc * dz).sum("z").values
    )

    # full 2D content is conserved as well
    full_integral = tgrid.integrate(transformed, ["X", "Z"])
    np.testing.assert_allclose(
        full_integral.values, (conc * dz * dx).sum().values
    )

    # averaging over X still works with the carried-over metric
    horizontal_mean = tgrid.average(transformed, "X")
    assert set(horizontal_mean.dims) == set(transformed.dims) - {"x"}


@pytest.mark.skipif(numba is None, reason="numba required")
def test_target_grid_rejects_ambiguous_target_dimension():
    # if the array handed to target_grid has no dimension that can be
    # uniquely attributed to the transformed axis, raise rather than guess
    grid, ds, conc, volume = _make_volume_grid([1.0, 4.0, 2.0], [10.0, 40.0, 25.0])
    transformed, effective_volume = grid.transform(
        conc,
        "Z",
        _INTENSIVE_TARGET,
        method="conservative",
        quantity="intensive",
        return_effective_volume=True,
    )
    # pretend the transformed axis is an unknown name so every output
    # dimension is treated as belonging to "other" axes -> no candidate
    with pytest.raises(ValueError, match="transformed coordinate"):
        target_grid(grid, "NONEXISTENT_AXIS", transformed, effective_volume)


@pytest.mark.skipif(numba is None, reason="numba required")
def test_intensive_mode_validation_errors():
    grid, ds, conc, volume = _make_volume_grid([1.0, 4.0, 2.0], [10.0, 40.0, 25.0])

    with pytest.raises(ValueError, match="quantity"):
        grid.transform(
            conc, "Z", _INTENSIVE_TARGET,
            method="conservative", quantity="nonsense",
        )

    with pytest.raises(ValueError, match="intensive"):
        grid.transform(
            conc, "Z", _INTENSIVE_TARGET,
            method="linear", quantity="intensive",
        )

    with pytest.raises(ValueError, match="coverage_threshold"):
        grid.transform(
            conc, "Z", _INTENSIVE_TARGET,
            method="conservative", quantity="intensive",
            coverage_threshold=1.5,
        )

    with pytest.raises(ValueError, match="return_effective_volume"):
        grid.transform(
            conc, "Z", _INTENSIVE_TARGET,
            method="linear", return_effective_volume=True,
        )

    # a grid without a registered volume metric cannot run the intensive mode
    bare_grid, _, bare_conc, _ = _make_volume_grid(
        [1.0, 4.0, 2.0], [10.0, 40.0, 25.0], register_metric=False
    )
    with pytest.raises(KeyError, match="effective volume metric"):
        bare_grid.transform(
            bare_conc, "Z", _INTENSIVE_TARGET,
            method="conservative", quantity="intensive",
        )


@pytest.mark.skipif(numba is None, reason="numba required")
def test_extensive_mode_unchanged_with_new_defaults():
    # the new keyword defaults must leave the extensive path bit-for-bit
    # identical to the previous behavior
    grid, ds, conc, volume = _make_volume_grid([1.0, 4.0, 2.0], [10.0, 40.0, 25.0])
    total_amount = conc * volume
    extensive = grid.transform(
        total_amount, "Z", _INTENSIVE_TARGET, method="conservative"
    )
    np.testing.assert_allclose(extensive.values, [1.0, 9.0, 160.0, 50.0])
    # conservation of the extensive total
    np.testing.assert_allclose(
        extensive.sum().values, total_amount.sum().values
    )


@pytest.mark.skipif(numba is None, reason="numba required")
def test_intensive_conservative_multidim_broadcast():
    # broadcast the 1D column over an extra dimension and check the 1D result
    grid, ds, conc, volume = _make_volume_grid([1.0, 4.0, 2.0], [10.0, 40.0, 25.0])
    na = 4
    conc_b = conc.expand_dims(a=na).copy(deep=True)
    transformed = grid.transform(
        conc_b,
        "Z",
        _INTENSIVE_TARGET,
        method="conservative",
        quantity="intensive",
    )
    single = grid.transform(
        conc,
        "Z",
        _INTENSIVE_TARGET,
        method="conservative",
        quantity="intensive",
    )
    # the transformed target dimension comes first; align before comparing
    _, expected = xr.broadcast(transformed, single)
    xr.testing.assert_allclose(transformed, expected)


@pytest.mark.skipif(numba is None, reason="numba required")
def test_intensive_conservative_dask():
    grid, ds, conc, volume = _make_volume_grid([1.0, 4.0, 2.0], [10.0, 40.0, 25.0])
    na = 4
    conc_b = conc.expand_dims(a=na).chunk({"a": 2}).copy(deep=True)
    transformed = grid.transform(
        conc_b,
        "Z",
        _INTENSIVE_TARGET,
        method="conservative",
        quantity="intensive",
    ).load()
    single = grid.transform(
        conc,
        "Z",
        _INTENSIVE_TARGET,
        method="conservative",
        quantity="intensive",
    )
    _, expected = xr.broadcast(transformed, single)
    xr.testing.assert_allclose(transformed, expected)
