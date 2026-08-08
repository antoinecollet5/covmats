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
    mesh_dim : Union[float, Sequence[float], NDArrayFloat]
        Dimensions of one mesh (grid cell) of the grid, one value per spatial
        dimension. A single scalar can be given for an isotropic mesh: it is
        then broadcast to all dimensions of `shape`.
    shape : Union[int, Sequence[int], NDArrayInt]
        Shape of the grid (number of grid cells along each axis). If
        `mesh_dim` has more than one element, the number of elements in
        `shape` must match the number of elements in `mesh_dim`.

    Returns
    -------
    NDArrayFloat
        Array of coordinates with shape (Npts, Ndims), Npts being the total
        number of grid points (the product of `shape`) and Ndims the number
        of spatial dimensions (the size of `shape`).

    Raises
    ------
    ValueError
        If `mesh_dim` has more than one element and its size does not match
        the size of `shape`.

    Examples
    --------
    >>> import numpy as np
    >>> from covmats import get_pts_coords_regular_grid
    >>> get_pts_coords_regular_grid(mesh_dim=1.0, shape=(2, 2))
    array([[0.5, 0.5],
           [1.5, 0.5],
           [0.5, 1.5],
           [1.5, 1.5]])
    >>> get_pts_coords_regular_grid(mesh_dim=[1.0, 2.0], shape=(2, 2)).shape
    (4, 2)
    """
    # convert to numpy array
    _shape = np.array(shape, dtype=np.int64).ravel()
    _mesh_dim = np.atleast_1d(np.array(mesh_dim, dtype=np.float64)).ravel()
    if _mesh_dim.size == 1:
        # Broadcast a scalar (isotropic) mesh size to all dimensions
        _mesh_dim = np.full(_shape.size, _mesh_dim[0])
    elif _mesh_dim.size != _shape.size:
        raise ValueError(
            f"`mesh_dim` has {_mesh_dim.size} element(s) while `shape` has "
            f"{_shape.size} element(s)! They must either match, or `mesh_dim` "
            "must be a single scalar value."
        )
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
