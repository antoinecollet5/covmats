"""Some tests to refactor."""

import re

import covmats
import numpy as np
import pytest
import scipy as sp
from covmats._covariances import CallBack
from covmats._sparse_helpers import get_SPD_sparse_n11_example
from covmats._types import NDArrayFloat

from .sparse_helpers import _get_L_D_P  # ty:ignore[unresolved-import]

v3 = np.array([3.0, 1.0, 8.0])
V34 = np.vstack([v3] * 4).T
assert np.shape(V34) == (3, 4)


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


def test_CovViaCholesky_log_pdet() -> None:

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
    v5 = rng.random(size=n)

    L = np.linalg.cholesky(A)
    cov = covmats.CovViaCholesky(L)

    expected = np.array([2.057481, 3.125178, 4.043185, 2.370566, 2.195569])

    # matvec and rmatvec
    np.testing.assert_allclose(cov @ v5, expected, rtol=1e-6)
    np.testing.assert_allclose(cov @ v5, cov.T @ v5)

    V5 = np.vstack([v5] * 4).T
    assert np.shape(V5) == (5, 4)

    # matmat
    np.testing.assert_allclose(cov @ V5, np.vstack([expected] * 4).T, rtol=1e-6)

    # rmatmat
    np.testing.assert_allclose(cov @ V5, cov.T @ V5)


def test_CovViaDioagonal_mvnormal() -> None:

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


def test_CovViaPrecisionCholesky() -> None:

    rng = np.random.default_rng()
    n = 5
    P = rng.random(size=(n, n))
    P = P @ P.T  # a precision matrix must be positive definite
    x = rng.random(size=n)

    cov = covmats.CovViaPrecisionCholesky(np.linalg.cholesky(P))

    res = cov.whiten(x)
    ref = x @ np.linalg.cholesky(P)
    assert np.allclose(res, ref)

    res = cov.log_pdet
    ref = -np.linalg.slogdet(P)[-1]
    assert np.allclose(res, ref)


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


def test_CovViaSparseCholesky() -> None:

    N: int = 11
    A = get_SPD_sparse_n11_example(seed=2026)
    cov = covmats.CovViaSparseCholesky(covmats.SparseCholeskyFactor(*_get_L_D_P(A)))

    # Test to dense
    np.testing.assert_allclose(cov.todense(), A.toarray())
    # Test shape
    assert cov.shape == (N, N)

    # Diagonal
    np.testing.assert_allclose(cov.get_diagonal(), A.diagonal())

    # Test solve
    expected_x = np.arange(N, dtype=np.float64)
    b = A @ expected_x
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
        covmats.CovViaEnsemble(colored_samples).todense(), A.toarray(), atol=2e-2
    )


def test_CovViaSparsePrecisionCholesky() -> None:

    N: int = 11
    # precision matrix (sparse)
    Q = get_SPD_sparse_n11_example(seed=2026)
    # dense covariance matrix
    A = np.linalg.inv(Q.toarray())
    cov = covmats.CovViaSparsePrecisionCholesky(
        covmats.SparseCholeskyFactor(*_get_L_D_P(Q))
    )

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


def test_eigen_decompose_and_associated_functions() -> None:
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
        is_use_preconditioner=True,
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

    samples = covmats.sample_from_sparse_cov_factor(
        np.ones(225) * 100.0, eig_mat.get_sparse_LLT_factor(), 20
    )
    assert samples.shape == (225, 20)
    samples = covmats.sample_from_sparse_cov_factor(
        np.ones(225) * 100.0, eig_mat.get_sparse_LLT_factor(), 10, random_state=2012
    )
    assert samples.shape == (225, 10)
    assert eig_mat.todense().shape == (225, 225)

    _trace = eig_mat.get_trace()
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


# def test_sparse_precision_matrix() -> None:
#     nx = (
#         10  # number of voxels along the x axis + 4 * 2 for the borders
# (regularization)
#     )
#     ny = 10  # number of voxels along the y axis
#     nz = 1
#     dx = 5.0  # voxel dimension along the x axis
#     dy = 5.0  # voxel dimension along the y axis
#     dz = 1.0

#     len_scale = 20.0  # m
#     kappa = 1 / len_scale
#     alpha = 1.0

#     mean = 300.0  # trend of the field
#     std = 150.0  # standard deviation of the field

#     # Create a presison matrix
#     Q_ref = spde.get_precision_matrix(
#         nx, ny, nz, dx, dy, dz, kappa, alpha, spatial_dim=2, sigma=std
#     )
#     cholQ_ref = sparse_cholesky(Q_ref)

#     n_fields = 50
#     # 200 non conditional simulations
#     tmp = []
#     for i in range(n_fields):
#         _field = np.abs(
#             spde.simu_nc(cholQ_ref, random_state=i).reshape(nx, ny, order="F") + mean
#         )

#         tmp.append(np.where(_field < 0.0, 0.0, _field).ravel("F"))
#     X = np.array(tmp).T

#     assert X.shape == (nx * ny, n_fields)

#     cov_mat = CovViaSparsePrecision(Q_ref, cholQ_ref)
#     assert (cov_mat @ np.ones(nx * ny)).size == nx * ny
#     assert cov_mat.get_diagonal().size == nx * ny


# def test_dense_covariance_matrix() -> None:
#     cov = CovViaDense(np.arange(1, 10).reshape(3, 3).astype(np.float64))
#     np.testing.assert_array_equal(
#         cov @ np.ones(3, dtype=np.float64), np.array([6.0, 15.0, 24.0])
#     )
#     np.testing.assert_array_equal(cov.T @ np.ones(3), np.array([12.0, 15.0, 18.0]))
#     np.testing.assert_array_equal(cov.get_diagonal(), np.array([1.0, 5.0, 9.0]))
#     assert cov.get_trace() == 15.0
