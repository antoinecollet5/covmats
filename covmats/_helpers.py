# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Provide some helpers."""

from typing import Sequence, Union

import numpy as np
from scipy._lib._util import check_random_state as check_random_state

from covmats._types import NDArrayFloat, NDArrayInt


def get_pts_coords_regular_grid(
    mesh_dim: Union[float, Sequence[float], NDArrayFloat],
    shape: Union[int, Sequence[int], NDArrayInt],
) -> NDArrayFloat:
    """
    Create an array of points coordinates for regular grids.

    It supports from 1 to n dimensions.

    Parameters
    ----------
    mesh_dim : NDArrayInt
        Dimensions of one mesh of the grid.
    shape : NDArrayInt
        Shape of the grid (number of grid cells along each axis). The number of elements
        in `shape` much match the number
        of elements in `mesh_dim`.

    Returns
    -------
    NDArrayFloat
        Array of coordinates with shape (Npts, Ndims).
    """
    # convert to numpy array
    _mesh_dim = np.array([mesh_dim]).ravel()
    _shape = np.array(shape, dtype=np.int64).ravel()
    # xmin = center of the first mesh
    xmin: NDArrayFloat = np.array(_mesh_dim) / 2.0
    # xmax  = center of the last mesh
    xmax = (_shape - 0.5) * _mesh_dim
    return (
        np.array(
            np.meshgrid(
                *[np.linspace(xmin[i], xmax[i], _shape[i]) for i in range(_shape.size)],
                indexing="ij",
            )
        )
        .reshape(_shape.size, -1, order="F")
        .T
    )
