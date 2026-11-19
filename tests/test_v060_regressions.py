# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Regression tests for the fixes and optimizations of v0.6.0."""

import covmats
import numpy as np
import pytest
import scipy as sp
from covmats import SparseCholeskyFactor, sparse_cholesky_superlu
from covmats._covariances import _build_kernel_preconditioner
from covmats._toeplitz import (
    toeplitz_circulant_spectrum,
    toeplitz_product,
    toeplitz_product_from_spectrum,
)


def _permuted_factor(n: int = 7, seed: int = 0):
    """Return a dense SPD matrix A and the exact factor with P != identity."""
    rng = np.random.default_rng(seed)
    B = rng.random((n, n))
    A = B @ B.T + n * np.eye(n)
    P = rng.permutation(n)
    Lc = np.linalg.cholesky(A[P][:, P])
    d = np.diag(Lc)
    scf = SparseCholeskyFactor(
        sp.sparse.csc_array(Lc / d), sp.sparse.csc_array(np.diag(d**2)), P
    )
    return A, scf


# --------------------------------------------------------------------------- #
# SparseCholeskyFactor fixes
# --------------------------------------------------------------------------- #
def test_scf_get_diagonal_with_permutation() -> None:
    A, scf = _permuted_factor()
    np.testing.assert_allclose(scf.get_diagonal(), np.diag(A))


def test_scf_get_invdiagonal() -> None:
    A, scf = _permuted_factor()
    ref = np.diag(np.linalg.inv(A))
    np.testing.assert_allclose(scf.get_invdiagonal(), ref)
    # several blocks
    np.testing.assert_allclose(scf.get_invdiagonal(block_size=3), ref)


def test_scf_whiten_inv_is_consistent() -> None:
    A, scf = _permuted_factor()
    n = A.shape[0]
    rng = np.random.default_rng(1)
    # whiten_inv(x) = W x with W' W = A
    for m in (1, 3, n):  # the n x n case used to silently return a wrong result
        X = rng.standard_normal((m, n))
        Y = scf.whiten_inv(X)
        assert Y.shape == X.shape
        np.testing.assert_allclose(Y @ Y.T, X @ A @ X.T, atol=1e-10)
    x = rng.standard_normal(n)
    np.testing.assert_allclose(scf.whiten_inv(x) @ scf.whiten_inv(x), x @ A @ x)


def test_scf_log_pdet_no_overflow() -> None:
    m = 800
    D = sp.sparse.diags_array(np.full(m, 50.0)).tocsc()
    scf = SparseCholeskyFactor(sp.sparse.eye_array(m).tocsc(), D, np.arange(m))
    np.testing.assert_allclose(scf.log_pdet, m * np.log(50.0))


def test_scf_cached_properties_are_invalidated() -> None:
    A, scf = _permuted_factor()
    _ = scf.invD, scf.sqrtD, scf.sqrtinvD, scf.Pt
    b = np.ones(A.shape[0])
    _ = scf.solve(b)  # fills the CSR caches

    scf.D = 2.0 * scf.D
    np.testing.assert_allclose(scf.invD.diagonal(), 1.0 / scf.D.diagonal())
    np.testing.assert_allclose(scf.sqrtD.diagonal(), np.sqrt(scf.D.diagonal()))
    np.testing.assert_allclose(scf.sqrtinvD.diagonal(), 1 / np.sqrt(scf.D.diagonal()))
    # solve uses the new D
    np.testing.assert_allclose(scf.solve(b), np.linalg.solve(scf.todense(), b))

    P = np.roll(scf.P, 1)
    scf.P = P
    np.testing.assert_array_equal(scf.Pt, np.argsort(P))


def test_scf_failed_L_set_keeps_previous_value() -> None:
    _, scf = _permuted_factor()
    old = scf.L
    bad = sp.sparse.csc_array(np.ones((7, 7)))  # not lower triangular
    with pytest.raises(ValueError, match="lower-triangular"):
        scf.L = bad
    assert scf.L is old


def test_scf_P_must_be_a_permutation() -> None:
    n = 4
    with pytest.raises(ValueError, match="permutation"):
        SparseCholeskyFactor(
            sp.sparse.eye_array(n).tocsc(),
            sp.sparse.eye_array(n).tocsc(),
            [0, 1, 1, 3],
        )


def test_assert_allclose_sparse_shape_mismatch() -> None:
    from covmats._sparse_helpers import assert_allclose_sparse

    with pytest.raises(AssertionError):
        assert_allclose_sparse(sp.sparse.eye_array(2), sp.sparse.eye_array(3))


def test_get_SPD_sparse_example_honours_seed() -> None:
    state = np.random.get_state()[1].copy()
    A1 = covmats.get_SPD_sparse_example(60, 1)
    A1b = covmats.get_SPD_sparse_example(60, 1)
    A2 = covmats.get_SPD_sparse_example(60, 2)
    assert abs(A1 - A1b).max() == 0.0
    assert abs(A1 - A2).max() > 0.0
    # the global numpy state is left untouched
    np.testing.assert_array_equal(state, np.random.get_state()[1])
    # symmetric positive definite
    assert abs(A1 - A1.T).max() == 0.0
    assert np.linalg.eigvalsh(A1.toarray()).min() > 0.0


# --------------------------------------------------------------------------- #
# sparse_cholesky_superlu
# --------------------------------------------------------------------------- #
def _laplacian_2d(k: int) -> sp.sparse.csc_array:
    T = sp.sparse.diags_array(
        [-1.0, 2.0, -1.0], offsets=[-1, 0, 1], shape=(k, k), dtype=float
    )
    eye = sp.sparse.eye_array(k)
    return (
        sp.sparse.kron(eye, T)
        + sp.sparse.kron(T, eye)
        + 0.1 * sp.sparse.eye_array(k * k)
    ).tocsc()


@pytest.mark.parametrize("ordering", ["amd", "rcm", "natural"])
@pytest.mark.parametrize("which", ["n11", "random", "laplacian"])
def test_sparse_cholesky_superlu(which: str, ordering: str) -> None:
    A = {
        "n11": covmats.get_SPD_sparse_n11_example(3),
        "random": covmats.get_SPD_sparse_example(150, 4),
        "laplacian": _laplacian_2d(12),
    }[which]
    scf = sparse_cholesky_superlu(A, ordering=ordering)  # ty: ignore[invalid-argument-type]
    assert isinstance(scf, SparseCholeskyFactor)
    np.testing.assert_allclose(scf.todense(), A.toarray(), atol=1e-10)
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    np.testing.assert_allclose(A @ scf.solve(b), b, atol=1e-9)
    np.testing.assert_allclose(scf.get_diagonal(), A.diagonal())
    np.testing.assert_allclose(
        scf.log_pdet, np.linalg.slogdet(A.toarray())[1], rtol=1e-10
    )


def test_sparse_cholesky_superlu_custom_permutation() -> None:
    A = _laplacian_2d(6)
    p = np.random.default_rng(0).permutation(A.shape[0])
    scf = sparse_cholesky_superlu(A, ordering=p)
    np.testing.assert_allclose(scf.todense(), A.toarray(), atol=1e-10)


def test_sparse_cholesky_superlu_errors() -> None:
    with pytest.raises(ValueError, match="square"):
        sparse_cholesky_superlu(sp.sparse.csc_array(np.ones((2, 3))))
    with pytest.raises(ValueError, match="Unknown ordering"):
        sparse_cholesky_superlu(sp.sparse.eye_array(3), ordering="foo")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="permutation"):
        sparse_cholesky_superlu(sp.sparse.eye_array(3), ordering=[0, 0, 1])  # ty: ignore[invalid-argument-type]
    # indefinite, and symmetric but with a negative pivot
    for bad in ([[1.0, 2.0], [2.0, 1.0]], np.diag([1.0, -1.0, 2.0])):
        with pytest.raises(ValueError, match="SPD|singular"):
            sparse_cholesky_superlu(sp.sparse.csc_array(np.asarray(bad)))
    with pytest.raises(ValueError, match="SPD|singular"):
        sparse_cholesky_superlu(sp.sparse.csc_array(np.zeros((3, 3))))


# --------------------------------------------------------------------------- #
# Toeplitz / FFT
# --------------------------------------------------------------------------- #
def _fft_cov(shape, len_scale=1.0, **kw):
    return covmats.CovKernelAsLinopViaFFT(
        lambda r: 2.0 * np.exp(-r),
        mesh_dim=[0.3] * len(shape),
        domain_shape=shape,
        len_scale=np.full(len(shape), len_scale),
        **kw,
    )


@pytest.mark.parametrize("shape", [(1,), (2,), (9,), (4, 5), (3, 1), (3, 4, 5)])
def test_toeplitz_matvec_matches_dense(shape) -> None:
    cov = _fft_cov(shape)
    dense = cov.todense()
    x = np.random.default_rng(0).standard_normal(dense.shape[0])
    np.testing.assert_allclose(cov @ x, dense @ x, atol=1e-12)
    # functional interface, with and without the precomputed spectrum
    first_row = cov._first_row
    np.testing.assert_allclose(
        toeplitz_product(x, first_row, shape), dense @ x, atol=1e-12
    )
    spec = toeplitz_circulant_spectrum(first_row, shape)
    np.testing.assert_allclose(
        toeplitz_product_from_spectrum(x, spec, shape), dense @ x, atol=1e-12
    )


# --------------------------------------------------------------------------- #
# CovKernelAsLinop
# --------------------------------------------------------------------------- #
def test_dense_matrix_accepts_scalar_len_scale() -> None:
    pts = np.array([[0, 0], [1, 0], [0, 2]])  # also integer coordinates
    kernel = lambda r: np.exp(-r)  # noqa: E731
    C_scalar = covmats._covariances._generate_dense_matrix_from_kernel(
        pts, kernel, np.array([2.0])
    )
    C_vec = covmats._covariances._generate_dense_matrix_from_kernel(
        pts, kernel, np.array([2.0, 2.0])
    )
    np.testing.assert_allclose(C_scalar, C_vec)
    assert C_scalar[0, 1] == pytest.approx(np.exp(-0.5))


def test_solve_counts_iterations_per_column() -> None:
    cov = _fft_cov((6, 6), nugget=1e-3)
    B = np.random.default_rng(0).standard_normal((36, 4))
    cov.reset_comptors()
    X = cov.solve(B)
    total = cov.itercount()

    per_col = 0
    for j in range(B.shape[1]):
        cov.reset_comptors()
        np.testing.assert_allclose(cov.solve(B[:, j]), X[:, j], atol=1e-9)
        per_col += cov.itercount()
    assert total == per_col


def test_solve_integer_rhs_is_not_truncated() -> None:
    cov = _fft_cov((4, 4))
    b = np.arange(16)
    np.testing.assert_allclose(cov @ cov.solve(b), b, atol=1e-8)


def test_solve_warns_when_gmres_does_not_converge() -> None:
    cov = _fft_cov((6, 6))
    with pytest.warns(RuntimeWarning, match="GMRES did not converge"):
        cov.solve(np.ones(36), maxiter=1, rtol=1e-14)


def _loop_preconditioner(pts, kernel, k):
    """Reference implementation (the python loop of v0.5.0)."""
    tree = sp.spatial.cKDTree(pts, leafsize=32)
    _, ind = tree.query(pts, k=k)
    y = np.zeros(k)
    y[0] = 1.0
    nu = np.zeros((len(pts), k))
    for i in range(len(pts)):
        Q = kernel(sp.spatial.distance.cdist(pts[ind[i]], pts[ind[i]]))
        nu[i] = np.linalg.solve(Q, y)
    return sp.sparse.csr_array(
        (nu.ravel(), (np.repeat(np.arange(len(pts)), k), ind.ravel())),
        shape=(len(pts),) * 2,
    )


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_preconditioner_matches_reference(ndim: int) -> None:
    rng = np.random.default_rng(0)
    pts = rng.random((80, ndim))
    kernel = lambda r: np.exp(-r)  # noqa: E731
    new = _build_kernel_preconditioner(pts, kernel, k=12)
    ref = _loop_preconditioner(pts, kernel, 12)
    np.testing.assert_allclose(new.toarray(), ref.toarray(), atol=1e-8, rtol=1e-8)


def test_preconditioner_uses_scaled_coordinates() -> None:
    pts = np.random.default_rng(0).random((60, 2)) * 10.0
    kernel = lambda r: np.exp(-r)  # noqa: E731
    len_scale = np.array([5.0, 2.0])
    cov = covmats.CovKernelAsLinop(
        pts, kernel, len_scale, k=10, is_use_preconditioner=True
    )
    ref = _build_kernel_preconditioner(pts / len_scale, kernel, k=10)
    if cov._preconditioner is not None:
        np.testing.assert_allclose(cov._preconditioner.toarray(), ref.toarray())


# --------------------------------------------------------------------------- #
# Eigen / ensemble
# --------------------------------------------------------------------------- #
def test_ensemble_can_be_reassigned() -> None:
    rng = np.random.default_rng(0)
    cov = covmats.CovViaEnsemble(rng.standard_normal((10, 5)))
    _ = cov.anomalies, cov.n_ens
    new = rng.standard_normal((20, 5))
    cov.ensemble = new
    assert cov.n_ens == 20
    np.testing.assert_allclose(cov.anomalies, new - new.mean(axis=0))
    np.testing.assert_allclose(cov.todense(), np.cov(new, rowvar=False), atol=1e-12)


def test_ensemble_rank_and_log_pdet() -> None:
    rng = np.random.default_rng(0)
    ens = rng.standard_normal((30, 4))  # full rank
    cov = covmats.CovViaEnsemble(ens)
    assert cov.rank == 4
    np.testing.assert_allclose(
        cov.log_pdet, np.linalg.slogdet(np.cov(ens, rowvar=False))[1]
    )
    low = covmats.CovViaEnsemble(rng.standard_normal((3, 6)))  # rank ne - 1 = 2
    assert low.rank == 2
    eig = np.linalg.eigvalsh(low.todense())
    np.testing.assert_allclose(low.log_pdet, np.log(eig[eig > 1e-10]).sum())


def test_eigen_sparse_LLT_factor() -> None:
    rng = np.random.default_rng(0)
    cov = covmats.CovViaEnsemble(rng.standard_normal((4, 9)))
    eig = covmats.CovViaEigenFactorization(
        (cov.eig_vals if hasattr(cov, "eig_vals") else cov._w, cov._v)
    )
    L = eig.get_sparse_LLT_factor()
    assert L.shape == (9, 9)
    np.testing.assert_allclose((L @ L.T).toarray(), eig.todense(), atol=1e-10)


# --------------------------------------------------------------------------- #
# Nugget: C = K + nugget * I, handled once in the base class
# --------------------------------------------------------------------------- #
NUGGET = 0.37


def _kernel_exp(r):
    return 2.0 * np.exp(-r)


def _dense_pair(nugget: float = NUGGET, **kw):
    """Same 2D grid operator through the dense and the FFT code paths."""
    shape, mesh = (5, 6), [0.3, 0.2]
    fft = covmats.CovKernelAsLinopViaFFT(
        _kernel_exp, mesh, shape, np.array([1.0, 1.5]), nugget=nugget, **kw
    )
    dense = covmats.CovKernelAsLinop(
        fft._pts, _kernel_exp, np.array([1.0, 1.5]), nugget=nugget, **kw
    )
    return dense, fft


def test_nugget_is_additive_and_consistent_everywhere() -> None:
    dense, fft = _dense_pair()
    n = dense.n_pts
    K = dense._kernel_todense()
    C = K + NUGGET * np.eye(n)
    x = np.random.default_rng(0).standard_normal((n, 3))
    for op in (dense, fft):
        np.testing.assert_allclose(op.todense(), C, atol=1e-12)
        np.testing.assert_allclose(op @ x, C @ x, atol=1e-11)
        np.testing.assert_allclose(op.get_diagonal(), np.diag(C))
        np.testing.assert_allclose(op @ op.solve(x), x, atol=1e-8)
        assert op.nugget == NUGGET


def test_nugget_zero_is_the_pure_kernel() -> None:
    dense, fft = _dense_pair(nugget=0.0)
    K = dense._kernel_todense()
    np.testing.assert_allclose(dense.todense(), K)
    np.testing.assert_allclose(fft.todense(), K)
    np.testing.assert_allclose(fft.get_diagonal(), np.diag(K))


def test_nugget_lower_bounds_the_smallest_eigenvalue() -> None:
    dense, _ = _dense_pair()
    assert np.linalg.eigvalsh(dense.todense()).min() >= NUGGET - 1e-12


@pytest.mark.parametrize("bad", [-1e-3, np.nan, np.inf])
def test_nugget_must_be_a_finite_non_negative_variance(bad: float) -> None:
    with pytest.raises(ValueError, match="nugget"):
        covmats.CovKernelAsLinop(
            np.zeros((3, 1)), _kernel_exp, np.array([1.0]), nugget=bad
        )
    with pytest.raises(ValueError, match="nugget"):
        covmats.CovKernelAsLinopViaFFT(_kernel_exp, [0.1], [3], np.ones(1), nugget=bad)


def test_preconditioner_includes_the_nugget() -> None:
    pts = np.random.default_rng(0).random((60, 2))
    new = _build_kernel_preconditioner(pts, _kernel_exp, k=10, nugget=NUGGET)
    ref = _loop_preconditioner(pts, lambda r: _kernel_exp(r) + NUGGET * (r == 0), 10)
    np.testing.assert_allclose(new.toarray(), ref.toarray(), atol=1e-8, rtol=1e-8)
    # and it differs from the one without nugget
    no = _build_kernel_preconditioner(pts, _kernel_exp, k=10)
    assert abs(new - no).max() > 1e-3


def test_preconditioner_reduces_iterations_with_nugget() -> None:
    _, plain = _dense_pair(nugget=1e-2)
    _, prec = _dense_pair(nugget=1e-2, is_use_preconditioner=True, k=15)
    b = np.ones(plain.n_pts)
    plain.solve(b)
    prec.solve(b)
    assert prec.itercount() <= plain.itercount()


def test_new_matrix_type_only_implements_kernel_matvec() -> None:
    """A subclass gets nugget, diagonal, dense, solve and counters for free."""

    class CustomKernelOperator(covmats.CovKernelAsLinop):
        def _kernel_matvec(self, x):
            return self._kernel_todense() @ x  # any K x implementation

    pts = np.random.default_rng(0).random((20, 2))
    op = CustomKernelOperator(pts, _kernel_exp, np.array([0.5]), nugget=NUGGET)
    ref = covmats.CovKernelAsLinop(pts, _kernel_exp, np.array([0.5]), nugget=NUGGET)
    x = np.random.default_rng(1).standard_normal(20)
    np.testing.assert_allclose(op @ x, ref @ x)
    np.testing.assert_allclose(op.get_diagonal(), ref.get_diagonal())
    np.testing.assert_allclose(op.solve(x), ref.solve(x), atol=1e-9)
    assert op.count > 0


def test_matvec_counter_is_incremented_by_every_operator_type() -> None:
    dense, fft = _dense_pair()
    for op in (dense, fft):
        op.reset_comptors()
        _ = op @ np.ones(op.n_pts)
        assert op.count == 1
