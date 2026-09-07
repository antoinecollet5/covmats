# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET
"""Unit tests for :py:mod:`covmats._cond_sim`."""

import covmats
import numpy as np
import pytest
from covmats import (
    CovarianceMatrix,
    CovViaCholesky,
    CovViaDiagonal,
    conditional_mean,
    conditional_simulate,
    make_point_observation_operator,
)
from covmats._cond_sim import (
    _as_covariance,
    _as_linear_operator,
    _build_dense_system,
    _get_colorizer,
)
from scipy.sparse.linalg import LinearOperator

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------
N = 25


@pytest.fixture
def x_coords():
    """1D regularly spaced coordinates used to build a squared-exponential prior."""
    return np.linspace(0.0, 10.0, N)


@pytest.fixture
def dense_cov_matrix(x_coords):
    """Dense squared-exponential covariance matrix (SPD)."""
    len_scale = 2.0
    C = np.exp(-0.5 * (x_coords[:, None] - x_coords[None, :]) ** 2 / len_scale**2)
    return C + 1e-10 * np.eye(N)


@pytest.fixture
def cov(dense_cov_matrix):
    """A CovViaCholesky wrapping the dense prior covariance."""
    return CovViaCholesky(np.linalg.cholesky(dense_cov_matrix))


@pytest.fixture
def obs_idx():
    return np.array([2, 8, 15, 20])


@pytest.fixture
def sigma():
    return np.array([0.05, 0.3, 0.1, 0.5])


@pytest.fixture
def obs_op(obs_idx):
    return make_point_observation_operator(obs_idx, N)


@pytest.fixture
def obs_values(dense_cov_matrix, obs_idx, sigma):
    rng = np.random.default_rng(0)
    true_field = rng.multivariate_normal(np.zeros(N), dense_cov_matrix)
    return true_field[obs_idx] + sigma * rng.standard_normal(obs_idx.size)


def dense_reference(dense_cov_matrix, obs_idx, sigma, obs_values, mean=0.0):
    """Exact dense-Gaussian conditioning formula, used as ground truth."""
    n = dense_cov_matrix.shape[0]
    mean_vec = np.broadcast_to(np.asarray(mean, dtype=float), (n,))
    Hd = np.eye(n)[obs_idx]
    R = np.diag(sigma**2)
    K = Hd @ dense_cov_matrix @ Hd.T + R
    r = obs_values - Hd @ mean_vec
    lam = np.linalg.solve(K, r)
    post_mean = mean_vec + dense_cov_matrix @ Hd.T @ lam
    post_cov = dense_cov_matrix - dense_cov_matrix @ Hd.T @ np.linalg.solve(
        K, Hd @ dense_cov_matrix
    )
    return post_mean, post_cov


# ---------------------------------------------------------------------------
# make_point_observation_operator
# ---------------------------------------------------------------------------
class TestMakePointObservationOperator:
    """Tests for :py:func:`make_point_observation_operator`."""

    def test_shape(self):
        H = make_point_observation_operator([0, 2], n=4)
        assert H.shape == (2, 4)

    def test_matvec_picks_entries(self):
        H = make_point_observation_operator([0, 2], n=4)
        out = H.matvec(np.array([10.0, 20.0, 30.0, 40.0]))
        np.testing.assert_allclose(out, [10.0, 30.0])

    def test_rmatvec_scatters_entries(self):
        H = make_point_observation_operator([0, 2], n=4)
        out = H.rmatvec(np.array([1.0, 2.0]))
        np.testing.assert_allclose(out, [1.0, 0.0, 2.0, 0.0])

    def test_matmat(self):
        H = make_point_observation_operator([0, 2], n=4)
        Z = np.arange(4 * 3, dtype=float).reshape(4, 3)
        out = H.matmat(Z)
        np.testing.assert_allclose(out, Z[[0, 2], :])

    def test_rmatmat(self):
        H = make_point_observation_operator([0, 2], n=4)
        Y = np.arange(2 * 3, dtype=float).reshape(2, 3)
        out = H.rmatmat(Y)
        expected = np.zeros((4, 3))
        expected[0] = Y[0]
        expected[2] = Y[1]
        np.testing.assert_allclose(out, expected)

    def test_repeated_indices_are_accumulated_in_rmatvec(self):
        H = make_point_observation_operator([0, 0, 2], n=4)
        out = H.rmatvec(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(out, [3.0, 0.0, 3.0, 0.0])

    def test_repeated_indices_are_accumulated_in_rmatmat(self):
        H = make_point_observation_operator([0, 0, 2], n=4)
        Y = np.array([[1.0], [2.0], [3.0]])
        out = H.rmatmat(Y)
        np.testing.assert_allclose(out, [[3.0], [0.0], [3.0], [0.0]])

    def test_empty_indices_raises(self):
        with pytest.raises(ValueError, match="at least one element"):
            make_point_observation_operator([], n=4)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match=r"in \[0, 4\)"):
            make_point_observation_operator([-1, 2], n=4)

    def test_too_large_index_raises(self):
        with pytest.raises(ValueError, match=r"in \[0, 4\)"):
            make_point_observation_operator([0, 4], n=4)


# ---------------------------------------------------------------------------
# _as_linear_operator
# ---------------------------------------------------------------------------
class TestAsLinearOperator:
    """Tests for the internal :py:func:`_as_linear_operator` helper."""

    def test_passthrough_for_linear_operator(self):
        H = make_point_observation_operator([0, 1], n=3)
        assert _as_linear_operator(H, n=3) is H

    def test_wraps_dense_array(self):
        mat = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        H = _as_linear_operator(mat, n=3)
        assert isinstance(H, LinearOperator)
        assert H.shape == (2, 3)
        np.testing.assert_allclose(H.matvec(np.array([5.0, 6.0, 7.0])), [5.0, 7.0])
        np.testing.assert_allclose(
            H.rmatvec(np.array([1.0, 2.0])), mat.T @ np.array([1.0, 2.0])
        )
        np.testing.assert_allclose(H.matmat(np.eye(3)), mat)
        np.testing.assert_allclose(H.rmatmat(np.eye(2)), mat.T)

    def test_non_2d_array_raises(self):
        with pytest.raises(ValueError, match="LinearOperator or a 2D"):
            _as_linear_operator(np.array([1.0, 2.0, 3.0]), n=3)

    def test_wrong_number_of_columns_raises(self):
        with pytest.raises(ValueError, match="LinearOperator or a 2D"):
            _as_linear_operator(np.zeros((2, 4)), n=3)


# ---------------------------------------------------------------------------
# _as_covariance
# ---------------------------------------------------------------------------
class TestAsCovariance:
    """Tests for the internal :py:func:`_as_covariance` helper."""

    def test_passthrough_for_covariance_matrix(self):
        R = CovViaDiagonal(np.array([0.1, 0.2]))
        assert _as_covariance(R, n_obs=2) is R

    def test_wraps_scalar(self):
        R = _as_covariance(0.25, n_obs=3)
        assert isinstance(R, CovViaDiagonal)
        np.testing.assert_allclose(R.get_diagonal(), [0.25, 0.25, 0.25])

    def test_wraps_array(self):
        variances = np.array([0.1, 0.2, 0.3])
        R = _as_covariance(variances, n_obs=3)
        assert isinstance(R, CovViaDiagonal)
        np.testing.assert_allclose(R.get_diagonal(), variances)


# ---------------------------------------------------------------------------
# _get_colorizer
# ---------------------------------------------------------------------------
class _StubCovarianceMatrix(CovarianceMatrix):
    """
    Minimal concrete stand-in for :py:class:`CovarianceMatrix`.
    """

    def __init__(self) -> None:
        pass  # intentionally bypass CovarianceMatrix.__init__

    def solve(self, b):
        raise NotImplementedError

    def get_diagonal(self):
        raise NotImplementedError

    def _todense(self):
        raise NotImplementedError

    @property
    def precision(self):
        raise NotImplementedError

    def _whiten(self, x):
        raise NotImplementedError

    def _colorize(self, x):
        raise NotImplementedError


class _NoColorizeNoSubspace(_StubCovarianceMatrix):
    """A duck-typed operator with neither `colorize` nor `subspace_size`."""

    shape = (5, 5)

    @property
    def colorize(self):
        # Shadow the concrete `colorize` that CovarianceMatrix would
        # otherwise provide, so this behaves like a genuine matrix-free
        # operator (e.g. CovKernelAsLinop) that offers no direct sampler:
        # `getattr(obj, "colorize", None)` must see it as absent.
        raise AttributeError(
            "`colorize` is intentionally unavailable on this test double"
        )


class TestGetColorizer:
    """Tests for the internal :py:func:`_get_colorizer` helper."""

    def test_default_uses_cov_colorize_and_subspace_size(self, cov):
        colorize, dim = _get_colorizer(cov, n=N, colorize_fn=None, colorize_dim=None)
        assert colorize == cov.colorize
        assert dim == cov.subspace_size

    def test_explicit_colorize_dim_with_default_colorizer(self, cov):
        colorize, dim = _get_colorizer(cov, n=N, colorize_fn=None, colorize_dim=7)
        assert colorize == cov.colorize
        assert dim == 7

    def test_custom_colorize_fn_defaults_dim_to_n(self, cov):
        def my_colorize(w):
            return w

        colorize, dim = _get_colorizer(
            cov, n=N, colorize_fn=my_colorize, colorize_dim=None
        )
        assert colorize is my_colorize
        assert dim == N

    def test_custom_colorize_fn_with_explicit_dim(self, cov):
        def my_colorize(w):
            return w

        colorize, dim = _get_colorizer(
            cov, n=N, colorize_fn=my_colorize, colorize_dim=13
        )
        assert colorize is my_colorize
        assert dim == 13

    def test_missing_colorizer_raises(self):
        with pytest.raises(ValueError, match="no `colorize` method"):
            _get_colorizer(
                _NoColorizeNoSubspace(), n=5, colorize_fn=None, colorize_dim=None
            )

    def test_subspace_size_fallback_to_n_when_absent(self):
        class _ColorizeNoSubspace(_StubCovarianceMatrix):
            def _colorize(self, x):
                return x

        _colorize, dim = _get_colorizer(
            _ColorizeNoSubspace(), n=9, colorize_fn=None, colorize_dim=None
        )
        assert dim == 9


# ---------------------------------------------------------------------------
# conditional_simulate
# ---------------------------------------------------------------------------
class TestConditionalSimulate:
    """Tests for :py:func:`conditional_simulate`."""

    def test_matches_exact_dense_conditioning(
        self, dense_cov_matrix, cov, obs_op, obs_idx, sigma, obs_values
    ):
        n_reals = 3000
        z_cond = conditional_simulate(
            cov,
            obs_op,
            obs_values,
            sigma**2,
            n_reals=n_reals,
            random_state=1,
            rtol=1e-12,
        )
        assert not isinstance(z_cond, tuple)
        assert z_cond.shape == (n_reals, N)

        mean_ref, cov_ref = dense_reference(
            dense_cov_matrix, obs_idx, sigma, obs_values
        )
        mean_mc = z_cond.mean(axis=0)
        cov_mc = np.cov(z_cond, rowvar=False)

        np.testing.assert_allclose(mean_mc, mean_ref, atol=0.05)
        np.testing.assert_allclose(np.diag(cov_mc), np.diag(cov_ref), atol=0.05)

    def test_nonzero_mean_is_honored(
        self, dense_cov_matrix, cov, obs_op, obs_idx, sigma, obs_values
    ):
        mean = 3.0
        z_cond = conditional_simulate(
            cov,
            obs_op,
            obs_values,
            sigma**2,
            n_reals=2000,
            random_state=2,
            mean=mean,
            rtol=1e-12,
        )
        mean_ref, _ = dense_reference(
            dense_cov_matrix, obs_idx, sigma, obs_values, mean=mean
        )
        assert not isinstance(z_cond, tuple)
        np.testing.assert_allclose(z_cond.mean(axis=0), mean_ref, atol=0.06)

    def test_scalar_obs_cov(self, cov, obs_op, obs_idx):
        d = np.zeros(obs_idx.size)
        z_cond = conditional_simulate(
            cov, obs_op, d, obs_cov=0.01, n_reals=2, random_state=3
        )
        assert not isinstance(z_cond, tuple)
        assert z_cond.shape == (2, N)

    def test_covariance_matrix_obs_cov(self, cov, obs_op, obs_idx):
        d = np.zeros(obs_idx.size)
        R = CovViaDiagonal(np.full(obs_idx.size, 0.02))
        z_cond = conditional_simulate(
            cov, obs_op, d, obs_cov=R, n_reals=2, random_state=4
        )
        assert not isinstance(z_cond, tuple)
        assert z_cond.shape == (2, N)

    def test_dense_array_obs_op_accepted(self, cov, obs_idx):
        dense_H = np.eye(N)[obs_idx]
        d = np.zeros(obs_idx.size)
        z_cond = conditional_simulate(
            cov, dense_H, d, obs_cov=0.01, n_reals=1, random_state=5
        )
        assert not isinstance(z_cond, tuple)
        assert z_cond.shape == (1, N)

    def test_return_unconditional_true(self, cov, obs_op, obs_idx):
        d = np.zeros(obs_idx.size)
        z_cond, z_unc = conditional_simulate(
            cov,
            obs_op,
            d,
            obs_cov=0.01,
            n_reals=2,
            random_state=6,
            return_unconditional=True,
        )
        assert z_cond.shape == (2, N)
        assert z_unc.shape == (2, N)
        assert not np.allclose(z_cond, z_unc)

    def test_return_unconditional_false_returns_array_only(self, cov, obs_op, obs_idx):
        d = np.zeros(obs_idx.size)
        out = conditional_simulate(
            cov, obs_op, d, obs_cov=0.01, n_reals=1, random_state=7
        )
        assert isinstance(out, np.ndarray)

    def test_zero_realizations(self, cov, obs_op, obs_idx):
        d = np.zeros(obs_idx.size)
        z_cond = conditional_simulate(
            cov, obs_op, d, obs_cov=0.01, n_reals=0, random_state=8
        )
        assert not isinstance(z_cond, tuple)
        assert z_cond.shape == (0, N)

    def test_custom_colorize_fn_and_dim(self, cov, obs_op, obs_idx):
        d = np.zeros(obs_idx.size)
        z_cond = conditional_simulate(
            cov,
            obs_op,
            d,
            obs_cov=0.01,
            n_reals=1,
            random_state=9,
            colorize_fn=cov.colorize,
            colorize_dim=cov.subspace_size,
        )
        assert not isinstance(z_cond, tuple)
        assert z_cond.shape == (1, N)

    def test_matrix_free_kernel_operator_with_lowrank_colorizer(self):
        pts = covmats.get_pts_coords_regular_grid(mesh_dim=1.0, shape=(6, 6))
        n = pts.shape[0]
        kernel_cov = covmats.CovKernelAsLinop(
            pts, lambda dist: np.exp(-dist), len_scale=np.array([2.0, 2.0])
        )
        lowrank = covmats.eigen_factorize_cov_mat(kernel_cov, n_pc=20, random_state=0)
        idx = np.array([0, 10, 20])
        H = make_point_observation_operator(idx, n)
        d = np.zeros(idx.size)
        z_cond = conditional_simulate(
            kernel_cov,
            H,
            d,
            obs_cov=0.05,
            n_reals=1,
            random_state=10,
            colorize_fn=lowrank.colorize,
            colorize_dim=lowrank.subspace_size,
        )
        assert not isinstance(z_cond, tuple)
        assert z_cond.shape == (1, n)

    def test_missing_colorizer_raises(self):
        pts = covmats.get_pts_coords_regular_grid(mesh_dim=1.0, shape=(4, 4))
        n = pts.shape[0]
        kernel_cov = covmats.CovKernelAsLinop(
            pts, lambda dist: np.exp(-dist), len_scale=np.array([2.0, 2.0])
        )
        H = make_point_observation_operator([0], n)
        with pytest.raises(ValueError, match="no `colorize` method"):
            conditional_simulate(kernel_cov, H, np.zeros(1), obs_cov=0.1, n_reals=1)

    def test_non_1d_obs_values_raises(self, cov, obs_op):
        with pytest.raises(ValueError, match="must be 1D"):
            conditional_simulate(cov, obs_op, np.zeros((2, 2)), obs_cov=0.1, n_reals=1)

    def test_mismatched_obs_op_shape_raises(self, cov, obs_idx):
        wrong_H = make_point_observation_operator(obs_idx[:2], N)  # 2 rows, but...
        d = np.zeros(3)  # ...3 observations expected
        with pytest.raises(ValueError, match="expected"):
            conditional_simulate(cov, wrong_H, d, obs_cov=0.1, n_reals=1)

    def test_cg_non_convergence_raises_runtime_error(self, dense_cov_matrix):
        n = 40
        x = np.linspace(0.0, 10.0, n)
        C = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / 1.0**2) + 1e-10 * np.eye(n)
        c = CovViaCholesky(np.linalg.cholesky(C))
        idx = np.arange(0, n, 2)
        H = make_point_observation_operator(idx, n)
        d = np.ones(idx.size)
        with pytest.raises(RuntimeError, match="did not converge"):
            conditional_simulate(
                c, H, d, obs_cov=1e-6, n_reals=1, random_state=0, maxiter=1, rtol=1e-14
            )

    def test_invalid_solver_raises(self, cov, obs_op, obs_idx):
        d = np.zeros(obs_idx.size)
        with pytest.raises(ValueError, match="`solver` must be one of"):
            conditional_simulate(cov, obs_op, d, obs_cov=0.1, n_reals=1, solver="bogus")

    def test_direct_solver_matches_exact_dense_conditioning(
        self, dense_cov_matrix, cov, obs_op, obs_idx, sigma, obs_values
    ):
        n_reals = 3000
        z_cond = conditional_simulate(
            cov,
            obs_op,
            obs_values,
            sigma**2,
            n_reals=n_reals,
            random_state=1,
            solver="direct",
        )
        assert not isinstance(z_cond, tuple)
        assert z_cond.shape == (n_reals, N)

        mean_ref, cov_ref = dense_reference(
            dense_cov_matrix, obs_idx, sigma, obs_values
        )
        mean_mc = z_cond.mean(axis=0)
        cov_mc = np.cov(z_cond, rowvar=False)

        np.testing.assert_allclose(mean_mc, mean_ref, atol=0.05)
        np.testing.assert_allclose(np.diag(cov_mc), np.diag(cov_ref), atol=0.05)

    def test_direct_solver_matches_cg_solver_statistically(
        self, cov, obs_op, obs_idx, obs_values, sigma
    ):
        n_reals = 2000
        z_direct = conditional_simulate(
            cov,
            obs_op,
            obs_values,
            sigma**2,
            n_reals=n_reals,
            random_state=7,
            solver="direct",
        )
        z_cg = conditional_simulate(
            cov,
            obs_op,
            obs_values,
            sigma**2,
            n_reals=n_reals,
            random_state=7,
            solver="cg",
            rtol=1e-12,
        )
        assert not isinstance(z_direct, tuple)
        assert not isinstance(z_cg, tuple)
        np.testing.assert_allclose(z_direct.mean(axis=0), z_cg.mean(axis=0), atol=0.06)
        np.testing.assert_allclose(
            np.diag(np.cov(z_direct, rowvar=False)),
            np.diag(np.cov(z_cg, rowvar=False)),
            atol=0.06,
        )

    def test_direct_solver_return_unconditional(self, cov, obs_op, obs_idx):
        d = np.zeros(obs_idx.size)
        z_cond, z_unc = conditional_simulate(
            cov,
            obs_op,
            d,
            obs_cov=0.01,
            n_reals=3,
            random_state=6,
            solver="direct",
            return_unconditional=True,
        )
        assert z_cond.shape == (3, N)
        assert z_unc.shape == (3, N)
        assert not np.allclose(z_cond, z_unc)

    def test_direct_solver_zero_realizations(self, cov, obs_op, obs_idx):
        d = np.zeros(obs_idx.size)
        z_cond = conditional_simulate(
            cov, obs_op, d, obs_cov=0.01, n_reals=0, solver="direct"
        )
        assert not isinstance(z_cond, tuple)
        assert z_cond.shape == (0, N)

    def test_direct_solver_nonsingular_check_raises_runtime_error(self):
        class _ZeroCov(_StubCovarianceMatrix):
            def matmat(self, X):
                return np.zeros_like(X)

        class _ZeroR(_StubCovarianceMatrix):
            def todense(self):
                return np.zeros((2, 2))

        H = make_point_observation_operator([0, 1], n=5)
        with pytest.raises(RuntimeError, match="not numerically SPD"):
            _build_dense_system(_ZeroCov(), H, _ZeroR(), n_obs=2)


# ---------------------------------------------------------------------------
# conditional_mean
# ---------------------------------------------------------------------------
class TestConditionalMean:
    """Tests for :py:func:`conditional_mean`."""

    def test_matches_exact_dense_conditioning_mean(
        self, dense_cov_matrix, cov, obs_op, obs_idx, sigma, obs_values
    ):
        mean_ref, _ = dense_reference(dense_cov_matrix, obs_idx, sigma, obs_values)
        mean_field = conditional_mean(cov, obs_op, obs_values, sigma**2)
        np.testing.assert_allclose(mean_field, mean_ref, atol=1e-6)

    def test_nonzero_prior_mean(
        self, dense_cov_matrix, cov, obs_op, obs_idx, sigma, obs_values
    ):
        mean = -1.5
        mean_ref, _ = dense_reference(
            dense_cov_matrix, obs_idx, sigma, obs_values, mean=mean
        )
        mean_field = conditional_mean(cov, obs_op, obs_values, sigma**2, mean=mean)
        np.testing.assert_allclose(mean_field, mean_ref, atol=1e-6)

    def test_dense_array_obs_op_accepted(self, cov, obs_idx):
        dense_H = np.eye(N)[obs_idx]
        d = np.ones(obs_idx.size)
        mean_field = conditional_mean(cov, dense_H, d, obs_cov=1e-4)
        assert mean_field.shape == (N,)

    def test_covariance_matrix_obs_cov(self, cov, obs_op, obs_idx):
        d = np.ones(obs_idx.size)
        R = CovViaDiagonal(np.full(obs_idx.size, 1e-4))
        mean_field = conditional_mean(cov, obs_op, d, obs_cov=R)
        assert mean_field.shape == (N,)

    def test_exact_interpolation_in_low_noise_limit(self, cov, obs_op, obs_idx):
        d = np.linspace(-1.0, 1.0, obs_idx.size)
        mean_field = conditional_mean(cov, obs_op, d, obs_cov=1e-10)
        np.testing.assert_allclose(mean_field[obs_idx], d, atol=1e-4)

    def test_non_1d_obs_values_raises(self, cov, obs_op):
        with pytest.raises(ValueError, match="must be 1D"):
            conditional_mean(cov, obs_op, np.zeros((2, 2)), obs_cov=0.1)

    def test_mismatched_obs_op_shape_raises(self, cov, obs_idx):
        wrong_H = make_point_observation_operator(obs_idx[:2], N)
        d = np.zeros(3)
        with pytest.raises(ValueError, match="expected"):
            conditional_mean(cov, wrong_H, d, obs_cov=0.1)

    def test_cg_non_convergence_raises_runtime_error(self):
        n = 40
        x = np.linspace(0.0, 10.0, n)
        C = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / 1.0**2) + 1e-10 * np.eye(n)
        c = CovViaCholesky(np.linalg.cholesky(C))
        idx = np.arange(0, n, 2)
        H = make_point_observation_operator(idx, n)
        d = np.ones(idx.size)
        with pytest.raises(RuntimeError, match="did not converge"):
            conditional_mean(c, H, d, obs_cov=1e-6, maxiter=1, rtol=1e-14)

    def test_invalid_solver_raises(self, cov, obs_op, obs_idx):
        d = np.zeros(obs_idx.size)
        with pytest.raises(ValueError, match="`solver` must be one of"):
            conditional_mean(cov, obs_op, d, obs_cov=0.1, solver="bogus")

    def test_direct_solver_matches_exact_dense_conditioning_mean(
        self, dense_cov_matrix, cov, obs_op, obs_idx, sigma, obs_values
    ):
        mean_ref, _ = dense_reference(dense_cov_matrix, obs_idx, sigma, obs_values)
        mean_field = conditional_mean(
            cov, obs_op, obs_values, sigma**2, solver="direct"
        )
        np.testing.assert_allclose(mean_field, mean_ref, atol=1e-6)

    def test_direct_solver_matches_cg_solver(
        self, cov, obs_op, obs_idx, obs_values, sigma
    ):
        mean_cg = conditional_mean(
            cov, obs_op, obs_values, sigma**2, solver="cg", rtol=1e-12
        )
        mean_direct = conditional_mean(
            cov, obs_op, obs_values, sigma**2, solver="direct"
        )
        np.testing.assert_allclose(mean_direct, mean_cg, atol=1e-6)


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------
class TestPublicExports:
    """Make sure the new API is properly wired into the top-level `covmats` package."""

    @pytest.mark.parametrize(
        "name",
        ["conditional_simulate", "conditional_mean", "make_point_observation_operator"],
    )
    def test_reexported_from_covmats(self, name):
        assert hasattr(covmats, name)
        assert name in covmats.__all__

    def test_is_covariance_matrix_subclass_check_available(self, cov):
        # sanity check used implicitly by _as_covariance
        assert isinstance(cov, CovarianceMatrix)
