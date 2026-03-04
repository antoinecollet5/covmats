"""Some tests to refactor."""

import re

import covmats
import numpy as np
import pytest
import scipy as sp
from covmats._covariances import (
    CallBack,
    _build_kernel_preconditioner,
    get_pts_coords_regular_grid,
)
from covmats._sparse_helpers import get_SPD_sparse_n11_example
from covmats._types import NDArrayFloat
from covmats.data import load_precision_example_4225x

from .sparse_helpers import _get_L_D_P  # ty:ignore[unresolved-import]

v3 = np.array([3.0, 1.0, 8.0])
V34 = np.vstack([v3] * 4).T
assert np.shape(V34) == (3, 4)

v5 = np.array([3.0, 0.0, 8.0, 9.76, -1.87])
V54 = np.vstack([v5] * 4).T
assert np.shape(V54) == (5, 4)


def test_CallBack():
    c = CallBack()
    assert c.itercount == 0
    for i in range(3):
        c(np.ones(10))
    assert c.itercount == 3
    c.clear()
    assert c.itercount == 0


def test_removed_from_cholesky() -> None:

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "`from_cholesky` is not available, please instantiate"
            " with `CovViaCholesky(...)` directly!"
        ),
    ):
        covmats.CovarianceMatrix.from_cholesky()


def test_removed_from_diagonal() -> None:

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "`from_diagonal` is not available, please instantiate"
            " with `CovViaDiagonal(...)` directly!"
        ),
    ):
        covmats.CovarianceMatrix.from_diagonal()


def test_removed_from_eigendecomposition() -> None:

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "`from_eigendecomposition` is not available, please instantiate"
            " with `CovViaEigenFactorization(...)` directly!"
        ),
    ):
        covmats.CovarianceMatrix.from_eigendecomposition()


def test_removed_from_precision() -> None:

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "`from_precision` is not available, please instantiate"
            " with `CovViaPrecisionCholesky(...)` directly!"
        ),
    ):
        covmats.CovarianceMatrix.from_precision()


def test_validate_dense_matrix() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "The input `my_arg` must be a square, "
            "two-dimensional array of real numbers."
        ),
    ):
        covmats.CovarianceMatrix._validate_dense_matrix(np.ones((10, 9)), "my_arg")

    # no issue
    covmats.CovarianceMatrix._validate_dense_matrix(np.ones((10, 10)), "my_arg")


def test_validate_sparse_matrix():
    with pytest.raises(
        ValueError,
        match=(
            "The input `my_arg` must be a square, "
            "two-dimensional array of real numbers."
        ),
    ):
        covmats.CovarianceMatrix._validate_sparse_matrix(
            sp.sparse.csc_array((10, 9)), "my_arg"
        )

    # No issue
    covmats.CovarianceMatrix._validate_sparse_matrix(
        sp.sparse.csc_array((10, 10)), "my_arg"
    )


def test_validate_vector():
    with pytest.raises(
        ValueError,
        match=("The input `my_arg` must be a one-dimensional array of real numbers."),
    ):
        covmats.CovarianceMatrix._validate_vector(np.ones((2, 2)), "my_arg")

    with pytest.raises(
        ValueError,
        match=("The input `my_arg` must be a one-dimensional array of real numbers."),
    ):
        covmats.CovarianceMatrix._validate_vector(np.ones((2, 1)), "my_arg")

    # No issue
    covmats.CovarianceMatrix._validate_vector(np.ones(((2,))), "my_arg")


def test_validate_dense_lower_triangle():
    with pytest.raises(
        ValueError,
        match=("The input `D` must be a lower-triangular matrix."),
    ):
        covmats.CovarianceMatrix._validate_dense_lower_triangle(np.ones((2, 2)), "D")

    # No issue
    covmats.CovarianceMatrix._validate_dense_lower_triangle(
        np.array([[1.0, 0.0], [1.0, 1.0]]), "D"
    )


def test_CovViaDiagonal_singular_or_indefinite() -> None:

    with pytest.raises(
        ValueError,
        match=(
            "2 null values have been detected in `D` which "
            "means the matrix is singular and non invertible."
        ),
    ):
        covmats.CovViaDiagonal(np.array([1.0, 0.0, 0.0, 8.789]))

    with pytest.raises(
        ValueError,
        match=(
            "1 negative values have been detected in `D` which "
            "means the matrix is indefinite and non invertible."
        ),
    ):
        covmats.CovViaDiagonal(np.array([1.0, 0.001, -1.9, 8.789]))


def test_CovViaDiagonal_stats1() -> None:

    d = [1, 2, 3]
    A33 = np.diag(d)  # a diagonal covariance matrix
    x = [4, -2, 5]  # a point of interest
    ref_dist = sp.stats.multivariate_normal(mean=[0, 0, 0], cov=A33)

    # Make a diagonal covariance matrix
    cov_diag33 = covmats.CovViaDiagonal(d)
    assert cov_diag33.rank == 3
    np.testing.assert_allclose(cov_diag33.log_pdet, 1.791759, rtol=1e-6)
    assert cov_diag33.get_trace() == 6

    cov_dist = sp.stats.multivariate_normal(mean=[0, 0, 0], cov=cov_diag33)
    np.testing.assert_allclose(ref_dist.pdf(x), cov_dist.pdf(x))

    # Test to dense and covariance
    np.testing.assert_allclose(cov_diag33.todense(), A33)
    np.testing.assert_allclose(cov_diag33.covariance, A33)


def test_CovViaDiagonal_stats2() -> None:

    rng = np.random.default_rng(2026)
    n = 5
    A = np.diag(rng.random(n))
    x = rng.random(size=n)

    d = np.diag(A)
    cov = covmats.CovViaDiagonal(d)

    res = cov.whiten(x)
    ref = np.diag(d**-0.5) @ x
    assert np.allclose(res, ref)

    res = cov.log_pdet
    ref = np.linalg.slogdet(A)[-1]
    assert np.allclose(res, ref)


def test_CovViaDiagonal_aslinop() -> None:

    d = [1, 2, 3]
    # Make a diagonal covariance matrix
    cov_diag33 = covmats.CovViaDiagonal(d)

    # matvec and rmatvec
    np.testing.assert_allclose(cov_diag33 @ v3, np.array([3.0, 2.0, 24.0]))
    np.testing.assert_allclose(cov_diag33 @ v3, cov_diag33.T @ v3)

    # matmat
    np.testing.assert_allclose(
        cov_diag33 @ V34, np.vstack([np.array([3.0, 2.0, 24.0])] * 4).T
    )

    # rmatmat
    np.testing.assert_allclose(cov_diag33 @ V34, cov_diag33.T @ V34)

    # solve
    np.testing.assert_allclose(cov_diag33.solve(cov_diag33 @ v3), v3)
    np.testing.assert_allclose(cov_diag33.solve(cov_diag33 @ V34), V34)

    # Get precision matrix (inverse)
    np.testing.assert_allclose(
        np.linalg.inv(cov_diag33.todense()), cov_diag33.precision
    )


def test_CovViaDiagonal_mvnormal() -> None:

    covd = covmats.CovViaDiagonal(np.array([5.0, 10.0, 15.0]))
    rng_seed = 42
    covd.sample_mvnormal(shape=[2], random_state=rng_seed)
    x = covd.sample_mvnormal(shape=[2, 4], random_state=rng_seed)
    np.testing.assert_allclose(
        x,
        np.array(
            [
                [
                    [1.11068661, -0.43723011, 2.50848692],
                    [3.40559829, -0.74045799, -0.90680853],
                    [3.53122721, 2.4268417, -1.81826648],
                    [1.21320114, -1.46545542, -1.80376358],
                ],
                [
                    [0.54104409, -6.05032338, -6.68057804],
                    [-1.25731314, -3.20285323, 1.21707469],
                    [-2.03040356, -4.46609644, 5.67643327],
                    [-0.50485116, 0.21354293, -5.518026],
                ],
            ]
        ),
    )
    assert x.shape == (2, 4, 3)


def test_CovViaCholesky_log_stats() -> None:

    rng = np.random.default_rng(2026)
    n = 5
    A = rng.random(size=(n, n))
    A = A @ A.T  # make the covariance symmetric positive definite
    x = rng.random(size=n)

    L = np.linalg.cholesky(A)
    cov = covmats.CovViaCholesky(L)

    res = cov.whiten(x)
    ref = sp.linalg.solve_triangular(L, x, lower=True)
    assert np.allclose(res, ref)

    res = cov.log_pdet
    ref = np.linalg.slogdet(A)[-1]
    assert np.allclose(res, ref)


def test_CovViaCholesky_aslinop() -> None:

    rng = np.random.default_rng(2026)
    n = 5
    A = rng.random(size=(n, n))
    A = A @ A.T  # make the covariance symmetric positive definite

    L = np.linalg.cholesky(A)
    cov = covmats.CovViaCholesky(L)

    # Check to dense
    np.testing.assert_allclose(cov.todense(), A)

    expected = np.array([20.932923, 33.649514, 42.873409, 25.55103, 22.343875])

    # matvec and rmatvec
    np.testing.assert_allclose(cov @ v5, expected, rtol=1e-6)
    np.testing.assert_allclose(cov @ v5, cov.T @ v5)

    # matmat
    np.testing.assert_allclose(cov @ V54, np.vstack([expected] * 4).T, rtol=1e-6)

    # rmatmat
    np.testing.assert_allclose(cov @ V54, cov.T @ V54)

    # solve
    np.testing.assert_allclose(cov.solve(cov @ v5), v5, atol=1e-10)
    np.testing.assert_allclose(cov.solve(cov @ V54), V54, atol=1e-10)

    # Get precision matrix (inverse)
    np.testing.assert_allclose(np.linalg.inv(A), cov.precision)

    # Test to dense with original covariance stored
    cov2 = covmats.CovViaCholesky(L, covariance=A)
    np.testing.assert_allclose(cov2.todense(), A)

    # Test  diagonal
    np.testing.assert_allclose(cov2.get_diagonal(), np.diagonal(A))


def test_CovViaCholesky_mvnormal() -> None:

    covd = covmats.CovViaDiagonal(np.array([5.0, 10.0, 15.0]))
    cov_cho = covmats.CovViaCholesky(sp.linalg.cholesky(covd.todense()))

    rng_seed = 42
    covd.sample_mvnormal(shape=[2], random_state=rng_seed)
    x = cov_cho.sample_mvnormal(shape=[2, 4], random_state=rng_seed)
    np.testing.assert_allclose(
        x,
        np.array(
            [
                [
                    [1.11068661, -0.43723011, 2.50848692],
                    [3.40559829, -0.74045799, -0.90680853],
                    [3.53122721, 2.4268417, -1.81826648],
                    [1.21320114, -1.46545542, -1.80376358],
                ],
                [
                    [0.54104409, -6.05032338, -6.68057804],
                    [-1.25731314, -3.20285323, 1.21707469],
                    [-2.03040356, -4.46609644, 5.67643327],
                    [-0.50485116, 0.21354293, -5.518026],
                ],
            ]
        ),
    )
    assert x.shape == (2, 4, 3)


@pytest.mark.parametrize("is_embbed_covariance", (True, False))
def test_CovViaSparseCholesky(is_embbed_covariance: bool) -> None:

    N: int = 11
    # precision matrix (sparse)
    A = get_SPD_sparse_n11_example(seed=2026)
    # dense covariance matrix
    Q = np.linalg.inv(A.toarray())
    cov = covmats.CovViaSparseCholesky(
        covmats.SparseCholeskyFactor(*_get_L_D_P(A)),
        sparse_covariance=A if is_embbed_covariance else None,
    )

    # test access scf
    cov.scf

    # Test to dense
    np.testing.assert_allclose(cov.todense(), A.toarray())
    # Test shape
    assert cov.shape == (N, N)

    # Diagonal
    np.testing.assert_allclose(cov.get_diagonal(), A.diagonal())

    # Dense
    np.testing.assert_allclose(cov.precision, Q)
    np.testing.assert_allclose(cov.todense(), A.toarray())

    # Test solve
    expected_x = np.arange(N, dtype=np.float64)
    b = A @ expected_x
    np.allclose(cov @ b, expected_x)
    np.allclose(cov.solve(b), expected_x)

    # Test colorize and whiten
    np.testing.assert_allclose(
        cov.whiten(cov.colorize(np.eye(N))), np.eye(N), atol=1e-15
    )

    # Test colorize with an ensemble
    colored_samples = cov.sample_mvnormal(
        shape=(500_000,), random_state=np.random.default_rng(2027)
    )
    np.testing.assert_allclose(
        covmats.CovViaEnsemble(colored_samples).todense(), A.toarray(), atol=2e-1
    )

    # Logdet
    np.testing.assert_allclose(np.linalg.slogdet(A.toarray())[1], cov.log_pdet)


@pytest.mark.parametrize("is_embbed_precision", (True, False))
def test_CovViaPrecisionCholesky(is_embbed_precision: bool) -> None:

    rng = np.random.default_rng()
    n = 5
    P = rng.random(size=(n, n))
    Q = P @ P.T  # a precision matrix must be positive definite
    x = rng.random(size=n)

    A = np.linalg.inv(Q)

    cov = covmats.CovViaPrecisionCholesky(
        np.linalg.cholesky(Q), precision=Q if is_embbed_precision else None
    )

    res = cov.whiten(x)
    ref = x @ np.linalg.cholesky(Q)
    assert np.allclose(res, ref)

    res = cov.log_pdet
    ref = -np.linalg.slogdet(Q)[-1]
    assert np.allclose(res, ref)

    # Test to dense
    np.testing.assert_allclose(cov.todense(), A)
    # Test shape
    assert cov.shape == (n, n)

    # Diagonal
    np.testing.assert_allclose(cov.get_diagonal(), A.diagonal())

    # Dense
    np.testing.assert_allclose(cov.precision, Q)
    np.testing.assert_allclose(cov.todense(), A)

    # Test matvec and solve
    expected_x = np.arange(n, dtype=np.float64)
    b = A @ expected_x
    np.allclose(cov @ b, expected_x)
    np.allclose(cov.solve(b), expected_x)

    # Test colorize and whiten
    np.testing.assert_allclose(
        cov.whiten(cov.colorize(np.eye(n))), np.eye(n), atol=1e-10, rtol=1e-10
    )

    # Test colorize with an ensemble
    colored_samples = cov.sample_mvnormal(
        shape=(500_000,), random_state=np.random.default_rng(2027)
    )
    np.testing.assert_allclose(
        covmats.CovViaEnsemble(colored_samples).todense(), A, rtol=2e-1, atol=5e-2
    )

    np.testing.assert_allclose(np.linalg.slogdet(A)[1], cov.log_pdet)


def test_CovViaPrecisionCholesky_whiten() -> None:

    rng = np.random.default_rng(2026)
    n = 3
    A = rng.random(size=(n, n))
    cov_array = A @ A.T  # make matrix symmetric positive definite
    precision = np.linalg.inv(cov_array)
    cov_object = covmats.CovViaPrecisionCholesky(
        sp.linalg.cholesky(precision, lower=True)
    )
    x = rng.multivariate_normal(np.zeros(n), cov_array, size=(10000))
    x_ = cov_object.whiten(x)
    # near-identity covariance is expected
    np.testing.assert_allclose(np.cov(x_, rowvar=False), np.eye(3), atol=0.01)


def test_ensemble_covariance_matrix() -> None:
    """Test the inversion."""

    cov = covmats.CovViaEnsemble(np.random.default_rng(2023).random((200, 77)))
    x = np.random.default_rng(2023).random(77)

    np.testing.assert_allclose(cov.solve(x), np.linalg.inv(cov.todense()).dot(x))
    np.testing.assert_allclose(np.trace(cov.todense()), cov.get_trace(), rtol=1e-12)


@pytest.mark.parametrize("is_embbed_precision", (True, False))
def test_CovViaSparsePrecisionCholesky(is_embbed_precision: bool) -> None:

    N: int = 11
    # precision matrix (sparse)
    Q = get_SPD_sparse_n11_example(seed=2026)
    # dense covariance matrix
    A = np.linalg.inv(Q.toarray())
    cov = covmats.CovViaSparsePrecisionCholesky(
        covmats.SparseCholeskyFactor(*_get_L_D_P(Q)),
        sparse_precision=Q if is_embbed_precision else None,
    )

    # test access scf
    cov.scf

    # Test to dense
    np.testing.assert_allclose(cov.todense(), A)
    # Test shape
    assert cov.shape == (N, N)

    # Diagonal
    np.testing.assert_allclose(cov.get_diagonal(), A.diagonal())

    # Dense
    np.testing.assert_allclose(cov.precision.todense(), Q.toarray())
    np.testing.assert_allclose(cov.todense(), A)

    # Test solve
    expected_x = np.arange(N, dtype=np.float64)
    b = A @ expected_x
    np.allclose(cov @ expected_x, A @ expected_x)
    np.allclose(cov.solve(b), expected_x)

    # Test colorize and whiten
    np.testing.assert_allclose(
        cov.whiten(cov.colorize(np.eye(N))), np.eye(N), atol=1e-15
    )

    # Test colorize with an ensemble
    colored_samples = cov.sample_mvnormal(
        shape=(500_000,), random_state=np.random.default_rng(2027)
    )
    np.testing.assert_allclose(
        covmats.CovViaEnsemble(colored_samples).todense(), A, atol=1e-2
    )

    np.testing.assert_allclose(np.linalg.slogdet(A)[1], cov.log_pdet)


def test_build_kernel_preconditioner() -> None:

    _number_grid_cells = 225
    prior_std = 2.0

    # Exponential covariance model
    def exponential_kernel(r: float) -> NDArrayFloat:
        return (prior_std**2) * np.exp(-r)

    param_shape = np.array(
        [np.sqrt(_number_grid_cells), np.sqrt(_number_grid_cells)], dtype=np.int8
    )
    # _params = {"R": 1.0e-4, "kappa": 100}
    dx = 1.0 / 50.0
    dy = 1.0 / 50.0
    mesh_dim = (dx, dy)

    pts = get_pts_coords_regular_grid(mesh_dim, param_shape)

    # No issue
    _build_kernel_preconditioner(pts, exponential_kernel, k=100)

    # Issue
    with pytest.raises(ValueError, match="The number of points cannot be null !"):
        _build_kernel_preconditioner(np.array([]), exponential_kernel, k=100)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "k (1000) must be lower or equal to the number of points (225)!"
        ),
    ):
        _build_kernel_preconditioner(np.array(pts), exponential_kernel, k=1000)


def test_fft_covariance_matrix() -> None:
    _number_grid_cells = 225
    prior_std = 2.0

    # Exponential covariance model
    def exponential_kernel(r: float) -> NDArrayFloat:
        return (prior_std**2) * np.exp(-r)

    param_shape = np.array(
        [np.sqrt(_number_grid_cells), np.sqrt(_number_grid_cells)], dtype=np.int8
    )
    # _params = {"R": 1.0e-4, "kappa": 100}
    dx = 1.0 / 50.0
    dy = 1.0 / 50.0
    len_scale = np.array([1, 1])
    mesh_dim = (dx, dy)

    cov = covmats.CovKernelAsLinopViaFFT(
        exponential_kernel,
        mesh_dim=mesh_dim,
        domain_shape=param_shape,
        len_scale=len_scale,
        nugget=1e-4,
        is_use_preconditioner=True,
    )

    # tests
    assert cov.n_pts == 225
    np.testing.assert_allclose(cov.get_diagonal(), np.ones(_number_grid_cells) * 4.0)
    assert np.sum(cov.get_diagonal()) == 900

    # reinitiate comptors
    cov.reset_comptors()
    assert cov.itercount() == 0

    # Dense version
    np.testing.assert_allclose(cov @ np.eye(cov.n_pts), cov.todense(), rtol=1e-3)

    # Test to matvec and solve
    np.testing.assert_allclose(cov.solve(cov @ np.ones(225)), np.ones(225))


@pytest.mark.parametrize("is_use_preconditioner,", ((True), (False)))
def test_eigen_decompose_and_associated_functions(is_use_preconditioner: bool) -> None:
    _number_grid_cells = 225
    prior_std = 2.0

    # Exponential covariance model
    def exponential_kernel(r: float) -> NDArrayFloat:
        return (prior_std**2) * np.exp(-r)

    param_shape = np.array(
        [np.sqrt(_number_grid_cells), np.sqrt(_number_grid_cells)], dtype=np.int8
    )
    # _params = {"R": 1.0e-4, "kappa": 100}
    dx = 1.0 / 50.0
    dy = 1.0 / 50.0
    len_scale = np.array([1, 1])
    mesh_dim = (dx, dy)

    cov_mat_fft = covmats.CovKernelAsLinopViaFFT(
        exponential_kernel,
        mesh_dim=mesh_dim,
        domain_shape=param_shape,
        len_scale=len_scale,
        nugget=1e-4,
        is_use_preconditioner=is_use_preconditioner,
    )

    eig_mat = covmats.eigen_factorize_cov_mat(cov_mat_fft, n_pc=100, random_state=25652)
    assert eig_mat.n_pc == 100
    # should return the matrix as is
    eig_mat = covmats.eigen_factorize_cov_mat(eig_mat, 50)
    assert eig_mat.n_pc == 100  # and not 50 !

    # no random state for the test
    _ = covmats.get_linop_eigen_factorization(eig_mat, 50, n_pc=12)

    # This is determined form the eigen vectors
    assert eig_mat.n_pts == 225

    np.testing.assert_allclose(
        eig_mat.get_diagonal(), cov_mat_fft.get_diagonal(), rtol=0.05
    )
    # The trace should be around 900 (225 * 2.0 ** 2)
    np.testing.assert_allclose(eig_mat.get_trace(), 900, rtol=0.05)

    samples = eig_mat.sample_mvnormal(shape=(20,), random_state=2026)
    assert samples.shape == (20, 225)
    samples = eig_mat.sample_mvnormal(
        shape=(
            10,
            7,
            99,
        ),
        random_state=2026,
    )
    assert samples.shape == (10, 7, 99, 225)
    assert eig_mat.todense().shape == (225, 225)

    eig_mat.precision
    # np.testing.assert_allclose(
    #     eig_mat.precision, np.linalg.inv(cov_mat_fft.todense()), rtol=0.1
    # )

    L = eig_mat.get_sparse_LLT_factor()
    np.testing.assert_allclose((L @ L.T).toarray(), eig_mat.todense())

    # Test to dense
    np.testing.assert_allclose(cov_mat_fft.todense(), eig_mat.todense(), rtol=1e-2)
    # Test trace and diagonal
    _trace = eig_mat.get_trace()
    np.testing.assert_allclose(_trace, np.sum(cov_mat_fft.get_diagonal()), rtol=1e-2)

    # both covariance matrice instance and trace
    covmats.get_explained_var(eig_mat.eig_vals, eig_mat, _trace)
    # no trace
    covmats.get_explained_var(eig_mat.eig_vals, eig_mat)
    # no matrix
    covmats.get_explained_var(eig_mat.eig_vals, trace_cov_mat=_trace)
    # none
    with pytest.raises(
        ValueError, match="You must provide a Covariance matrix instance or the trace !"
    ):
        covmats.get_explained_var(eig_mat.eig_vals)

    # Test linop capability
    test_v = np.ones(225)
    eig_mat.solve(eig_mat @ test_v)  # cannot test equality because we will not get it

    test_V = np.ones((225, 89))
    eig_mat.solve(eig_mat @ test_V)

    # Test sampling
    samples = np.random.default_rng(2027).standard_normal(
        size=(4, 70, eig_mat._subspace_size)
    )
    np.testing.assert_allclose(
        samples, eig_mat.whiten(eig_mat.colorize(samples)), rtol=1e-2, atol=1e-2
    )


def test_negative_eigen_values() -> None:
    # we build a matrix with negative eigen values
    # matrix 4 x 4
    U = np.arange(16).reshape((4, 4)).astype(np.float64)
    V = np.diag([5.0, 4.0, -1.0, -2.0])

    # This is the dense matrix to decompose
    cov_mat = U @ V @ U.T

    cov_mat_eigen = covmats.eigen_factorize_cov_mat(cov_mat, n_pc=3, random_state=2023)
    assert cov_mat_eigen.n_pc == 2
    assert cov_mat_eigen.eig_vects.size == 8


def test_CovViaEnsemble_0() -> None:

    rng = np.random.default_rng(2026)
    n = 5
    A = rng.random(size=(n, n))
    A = A @ A.T  # make the covariance symmetric positive definite

    L = np.linalg.cholesky(A)
    cov_cho = covmats.CovViaCholesky(L)

    # Test colorize with an ensemble
    colored_samples = cov_cho.sample_mvnormal(
        shape=(5000,), random_state=np.random.default_rng(2027)
    )
    cov_ens = covmats.CovViaEnsemble(colored_samples)

    # Check to dense
    np.testing.assert_allclose(cov_ens.todense(), A, rtol=0.02)

    expected = np.array([20.926541, 33.896138, 43.155498, 25.852243, 22.575929])

    # matvec and rmatvec
    np.testing.assert_allclose(cov_ens @ v5, expected, rtol=1e-6)
    np.testing.assert_allclose(cov_ens @ v5, cov_ens.T @ v5)

    # matmat
    np.testing.assert_allclose(cov_ens @ V54, np.vstack([expected] * 4).T, rtol=1e-6)

    # rmatmat
    np.testing.assert_allclose(cov_ens @ V54, cov_ens.T @ V54)

    # solve
    np.testing.assert_allclose(cov_ens.solve(cov_ens @ v5), v5, atol=1e-10)
    np.testing.assert_allclose(cov_ens.solve(cov_ens @ V54), V54, atol=1e-10)

    # Get precision matrix (inverse)
    np.testing.assert_allclose(np.linalg.inv(A), cov_ens.precision, rtol=0.05)


def test_CovViaEnsemble_1() -> None:

    # 4225 x 4225 sparse precision matrix
    Q = load_precision_example_4225x()
    # 4225 x 4225 dense covariance matrix
    # C = np.linalg.inv(Q.toarray())

    cov = covmats.CovViaSparsePrecisionCholesky(
        covmats.SparseCholeskyFactor(*_get_L_D_P(Q))
    )

    # Sample from this covariance
    ne = 1000
    ensemble = cov.sample_mvnormal((ne,), random_state=np.random.default_rng(209))

    # Create a new Cov instance using the ensemble
    Ce = covmats.CovViaEnsemble(ensemble=ensemble)

    # Make it dense
    Ce.todense()
    # np.testing.assert_allclose(Ce.todense(), C, rtol=0.5, atol=1.0)

    # Invert it
    np.testing.assert_allclose(Ce.precision, Q.toarray(), rtol=1e-2, atol=2e-1)

    # Sample from this ensemble
    samples = np.random.default_rng(2027).standard_normal(size=(ne, Ce.subspace_size))
    np.testing.assert_allclose(
        samples, Ce.whiten(Ce.colorize(samples)), rtol=1e-2, atol=1e-2
    )

    # Test linop capability
    # TODO


def test_CovViaEnsemble_2() -> None:
    # precision matrix (sparse)
    Q = get_SPD_sparse_n11_example(seed=2026)
    # dense covariance matrix
    C = np.linalg.inv(Q.toarray())
    cvspc = covmats.CovViaSparsePrecisionCholesky(
        covmats.SparseCholeskyFactor(*_get_L_D_P(Q))
    )

    # White noise sampling
    samples = np.random.default_rng(2027).standard_normal(size=(50_000, 11))
    # Sample from the covariance matrix
    colored_samples = cvspc.colorize(samples)

    # New representation from the ensemble
    Ce = covmats.CovViaEnsemble(colored_samples)
    # Test sampling using the new representation
    colored_samples2 = Ce.colorize(samples)
    np.testing.assert_allclose(samples, Ce.whiten(colored_samples2))

    # Make a second covariance instancefrom the new ensemble
    Ce2 = covmats.CovViaEnsemble(colored_samples2)

    # Test the diagonal extraction
    np.testing.assert_allclose(np.diag(C), Ce.get_diagonal(), rtol=0.05)

    # Check consistency
    np.testing.assert_allclose(C, Ce.todense(), atol=0.05)
    np.testing.assert_allclose(C, Ce2.todense(), atol=0.1)
    np.testing.assert_allclose(Q.toarray(), Ce.precision, atol=0.05)
    np.testing.assert_allclose(Q.toarray(), Ce2.precision, atol=0.05)

    # Test linop capability
    # TODO
