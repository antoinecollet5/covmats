# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET
"""Unit tests for :py:mod:`covmats._helpers`."""

import numpy as np
import pytest
from covmats._helpers import get_pts_coords_regular_grid


class TestGetPtsCoordsRegularGrid:
    """Tests for :py:func:`get_pts_coords_regular_grid`."""

    def test_docstring_example_2d_scalar_mesh_dim(self):
        """Reproduce the exact example from the function's docstring."""
        pts = get_pts_coords_regular_grid(mesh_dim=1.0, shape=(2, 2))
        expected = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
        np.testing.assert_allclose(pts, expected)

    def test_docstring_example_2d_vector_mesh_dim(self):
        """Reproduce the second example from the function's docstring."""
        pts = get_pts_coords_regular_grid(mesh_dim=[1.0, 2.0], shape=(2, 2))
        assert pts.shape == (4, 2)

    @pytest.mark.parametrize(
        "shape",
        [(2,), (5,), (2, 3), (3, 3), (2, 3, 4)],
    )
    def test_number_of_points_and_dims_match_shape(self, shape):
        """Npts must equal prod(shape) and Ndims must equal len(shape)."""
        pts = get_pts_coords_regular_grid(mesh_dim=1.0, shape=shape)
        assert pts.shape == (int(np.prod(shape)), len(shape))

    def test_1d_grid_point_positions(self):
        """A 1D grid with unit mesh size should give centers 0.5, 1.5, ..."""
        pts = get_pts_coords_regular_grid(mesh_dim=1.0, shape=5)
        np.testing.assert_allclose(pts.ravel(), [0.5, 1.5, 2.5, 3.5, 4.5])

    def test_scalar_mesh_dim_is_broadcast_isotropically(self):
        """A scalar mesh_dim must be equivalent to repeating it per axis."""
        pts_scalar = get_pts_coords_regular_grid(mesh_dim=2.0, shape=(3, 3))
        pts_vector = get_pts_coords_regular_grid(mesh_dim=[2.0, 2.0], shape=(3, 3))
        np.testing.assert_allclose(pts_scalar, pts_vector)

    def test_scalar_mesh_dim_broadcasts_to_3d(self):
        """The scalar broadcast must also work for 3+ dimensional grids."""
        pts_scalar = get_pts_coords_regular_grid(mesh_dim=1.5, shape=(2, 2, 2))
        pts_vector = get_pts_coords_regular_grid(
            mesh_dim=[1.5, 1.5, 1.5], shape=(2, 2, 2)
        )
        np.testing.assert_allclose(pts_scalar, pts_vector)

    def test_anisotropic_mesh_dim_changes_spacing_per_axis(self):
        """Each axis must be spaced according to its own mesh_dim entry."""
        pts = get_pts_coords_regular_grid(mesh_dim=[1.0, 2.0], shape=(3, 3))
        xs = np.unique(pts[:, 0])
        ys = np.unique(pts[:, 1])
        np.testing.assert_allclose(np.diff(xs), [1.0, 1.0])
        np.testing.assert_allclose(np.diff(ys), [2.0, 2.0])

    def test_first_and_last_point_are_mesh_centers(self):
        """
        xmin must be mesh_dim / 2 and xmax must be (shape - 0.5) * mesh_dim,
        per axis.
        """
        mesh_dim = np.array([2.0, 3.0])
        shape = (4, 5)
        pts = get_pts_coords_regular_grid(mesh_dim=mesh_dim, shape=shape)
        xmin_expected = mesh_dim / 2.0
        xmax_expected = (np.array(shape) - 0.5) * mesh_dim
        np.testing.assert_allclose(pts.min(axis=0), xmin_expected)
        np.testing.assert_allclose(pts.max(axis=0), xmax_expected)

    @pytest.mark.parametrize(
        "mesh_dim, shape",
        [
            ([1.0, 2.0, 3.0], (2, 2)),  # 3 mesh dims, 2D shape
            ([1.0, 2.0], (2, 2, 2)),  # 2 mesh dims, 3D shape
        ],
    )
    def test_mismatched_mesh_dim_and_shape_raises_value_error(self, mesh_dim, shape):
        """A non-scalar mesh_dim whose size disagrees with shape must raise."""
        with pytest.raises(ValueError, match="mesh_dim"):
            get_pts_coords_regular_grid(mesh_dim=mesh_dim, shape=shape)

    def test_output_dtype_is_floating(self):
        pts = get_pts_coords_regular_grid(mesh_dim=1.0, shape=(2, 2))
        assert np.issubdtype(pts.dtype, np.floating)

    def test_accepts_list_tuple_and_ndarray_shape(self):
        """`shape` may be given as a list, a tuple, or an ndarray."""
        pts_list = get_pts_coords_regular_grid(mesh_dim=1.0, shape=[2, 3])
        pts_tuple = get_pts_coords_regular_grid(mesh_dim=1.0, shape=(2, 3))
        pts_array = get_pts_coords_regular_grid(mesh_dim=1.0, shape=np.array([2, 3]))
        np.testing.assert_allclose(pts_list, pts_tuple)
        np.testing.assert_allclose(pts_tuple, pts_array)

    def test_accepts_list_tuple_and_ndarray_mesh_dim(self):
        """`mesh_dim` may be given as a list, a tuple, or an ndarray."""
        pts_list = get_pts_coords_regular_grid(mesh_dim=[1.0, 2.0], shape=(2, 2))
        pts_tuple = get_pts_coords_regular_grid(mesh_dim=(1.0, 2.0), shape=(2, 2))
        pts_array = get_pts_coords_regular_grid(
            mesh_dim=np.array([1.0, 2.0]), shape=(2, 2)
        )
        np.testing.assert_allclose(pts_list, pts_tuple)
        np.testing.assert_allclose(pts_tuple, pts_array)

    def test_no_duplicate_points(self):
        pts = get_pts_coords_regular_grid(mesh_dim=1.0, shape=(4, 4))
        unique_pts = np.unique(pts, axis=0)
        assert unique_pts.shape[0] == pts.shape[0]

    def test_shape_with_singleton_dimension(self):
        """A grid with a size-1 axis should collapse that axis to one value."""
        pts = get_pts_coords_regular_grid(mesh_dim=1.0, shape=(1, 3))
        assert pts.shape == (3, 2)
        np.testing.assert_allclose(pts[:, 0], np.full(3, 0.5))

    def test_int_mesh_dim_is_accepted(self):
        """An integer mesh_dim should behave the same as its float equivalent."""
        pts_int = get_pts_coords_regular_grid(mesh_dim=2, shape=(2, 2))
        pts_float = get_pts_coords_regular_grid(mesh_dim=2.0, shape=(2, 2))
        np.testing.assert_allclose(pts_int, pts_float)

    def test_single_point_grid(self):
        """A (1, 1) grid must return a single point centered at mesh_dim / 2."""
        pts = get_pts_coords_regular_grid(mesh_dim=4.0, shape=(1, 1))
        np.testing.assert_allclose(pts, [[2.0, 2.0]])
