import re

import covmats
import numpy as np
import pytest
import scipy as sp
from covmats import (
    SparseCholeskyFactor,
    load_precision_example_4225x,
    load_precision_example_4225x_SCF,
)
from covmats._sparse_helpers import (
    assert_allclose_sparse,
    get_SPD_sparse_example,
    get_SPD_sparse_n11_example,
)


def test_SparseCholeskyFactor_construtor() -> None:

    A = get_SPD_sparse_n11_example(seed=2026)
    L, D, P = covmats.get_L_D_P(A)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`L` has shape (11, 11), "
            "`D` has shape (11, 11), and `P` has shape (10,)!\n"
            "To be consistent, `D` and "
            "`P` are expected with shape (11, 11) and (11,) respectively."
        ),
    ):
        SparseCholeskyFactor(L, D, P[:-1])

    with pytest.raises(
        ValueError,
        match=("The diagonal elements of `L` are assumed to be 1!"),
    ):
        L[0, 0] = 3.0  # no unit diagonal
        SparseCholeskyFactor(L, D, P)

    with pytest.raises(
        ValueError,
        match=("The input `L` must be a lower-triangular sparse array."),
    ):
        L = L @ L.T  # not a lower triangle
        SparseCholeskyFactor(L, D, P)


def test_SparseCholeskyFactor() -> None:

    A = get_SPD_sparse_n11_example(seed=2026)
    Q = np.linalg.inv(A.toarray())
    scf = SparseCholeskyFactor(*covmats.get_L_D_P(A))

    # Test to dense
    np.testing.assert_allclose(scf.todense(), A.toarray(), atol=1e-12)
    # Test shape
    assert scf.shape == (11, 11)
    assert scf.n_pts == 11

    # Diagonal
    np.testing.assert_allclose(scf.get_diagonal(), A.diagonal())
    np.testing.assert_allclose(scf.get_invdiagonal(), Q.diagonal())

    # Test solve
    expected_x = np.arange(scf.n_pts, dtype=np.float64)
    b = A @ expected_x
    np.allclose(scf.solve(b), expected_x)

    # Test colorize and whiten
    np.testing.assert_allclose(
        scf.whiten(scf.colorize(np.eye(11))), np.eye(11), atol=1e-15
    )

    # Test colorize with an ensemble
    colored_samples = scf.colorize(
        np.random.default_rng(2027).standard_normal(size=(500_000, 11))
    )
    np.testing.assert_allclose(
        covmats.CovViaEnsemble(colored_samples).todense(), A.toarray(), atol=2e-2
    )

    # slogdet
    np.testing.assert_allclose(np.linalg.slogdet(A.toarray())[1], scf.log_pdet)


def test_SparseCholeskyFactor_colorize_adjoint() -> None:

    A = get_SPD_sparse_n11_example(seed=2026)
    scf = SparseCholeskyFactor(*covmats.get_L_D_P(A))

    rng = np.random.default_rng(2028)
    n = scf.n_pts
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)

    # colorize_adjoint is the adjoint of colorize: <colorize(x), y> == <x,
    # colorize_adjoint(y)>
    np.testing.assert_allclose(
        scf.colorize(x[None, :])[0] @ y, x @ scf.colorize_adjoint(y[None, :])[0]
    )

    # colorize_inv_adjoint is the adjoint of colorize_inv
    np.testing.assert_allclose(
        scf.colorize_inv(x[None, :])[0] @ y,
        x @ scf.colorize_inv_adjoint(y[None, :])[0],
    )

    # same checks, batched
    X = rng.standard_normal((4, n))
    Y = rng.standard_normal((4, n))
    np.testing.assert_allclose(
        np.sum(scf.colorize(X) * Y), np.sum(X * scf.colorize_adjoint(Y))
    )
    np.testing.assert_allclose(
        np.sum(scf.colorize_inv(X) * Y), np.sum(X * scf.colorize_inv_adjoint(Y))
    )


def test_assert_allclose_sparse() -> None:
    L = sp.sparse.csc_array([[1.0, 0.0, 0.1], [0.1, 1.2, 0.0], [0.0, 0.0, 1.5]])
    A = L.T @ L

    # works
    assert_allclose_sparse(A, A)
    assert_allclose_sparse(A.T, A)
    assert_allclose_sparse(L, L)

    # shapes are not the same
    with pytest.raises(AssertionError):
        assert_allclose_sparse(L, L.T)

    # Values are not the same
    with pytest.raises(AssertionError):
        assert_allclose_sparse(A, L)

    # shape is not the same
    with pytest.raises(AssertionError):
        assert_allclose_sparse(
            A, sp.sparse.csc_array([[1.0, 0.0, 0.1], [0.0, 0.0, 1.5]])
        )

    # empty
    assert_allclose_sparse(sp.sparse.csc_array([[]]), sp.sparse.csc_array([[]]))


def test_get_SPD_sparse_example() -> None:

    A = get_SPD_sparse_example(100, 2025, diag_mean=4.0)
    SparseCholeskyFactor(*covmats.get_L_D_P(A))


def test_load_precision_example_4225x() -> None:

    A = load_precision_example_4225x()
    SparseCholeskyFactor(*covmats.get_L_D_P(A))

    # Load the cholesky factor
    load_precision_example_4225x_SCF()


def test_indefinite():

    A = get_SPD_sparse_example(100, 2025, diag_mean=5.0)
    L, D, P = covmats.get_L_D_P(A)
    D = sp.sparse.diags(np.hstack([np.ones(100), 0.0]))

    # And sparse cholesky should fail as well
    with pytest.raises(
        ValueError,
        match="1 null values have been detected in `D` which means the factorized"
        " matrix is singular and non invertible.",
    ):
        SparseCholeskyFactor(L, D, P)


def test_singular():

    # Produce a singular matrix
    A = get_SPD_sparse_example(100, 2025, diag_mean=0.0)
    # Compute the eigenvalues of a complex Hermitian or real symmetric matrix
    sigma = np.linalg.eigvalsh(A.toarray())
    # The matrix is indefinite (negative eigen values)
    assert np.count_nonzero(sigma < 0) > 0

    # classic cholesky fails
    with pytest.raises(np.linalg.LinAlgError):
        sp.linalg.cholesky(A.toarray())

    # And sparse cholesky should fail as well
    with pytest.raises(
        ValueError,
        match=r"\d+ negative values have been detected in `D` which means the "
        "factorized matrix is indefinite and non invertible.",
    ):
        SparseCholeskyFactor(*covmats.get_L_D_P(A))
