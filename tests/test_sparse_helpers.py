import re

import covmats
import numpy as np
import pytest
import scipy as sp
import sksparse.cholmod as cholmod  # ty:ignore[unresolved-import]
from covmats import SparseCholeskyFactor, load_precision_example_4225x
from covmats._sparse_helpers import (
    assert_allclose_sparse,
    get_SPD_sparse_example,
    get_SPD_sparse_n11_example,
)


def _get_L_D_P(A: sp.sparse.sparray):
    """
    Return L, D and P from the factorization L @ D @ L' = P @ A @ P' using sksparse.

    Note that sksparse uses SuiteSparse which is LGPL licence.
    """
    # Need to take the API change into account
    try:
        # sksparse 4.x
        L, D, P = cholmod.ldl(A, order="amd")
    except AttributeError:
        # sksparse 5.x
        f = cholmod.cholesky(A)
        (L, D), P = f.L_D(), f.P()
    return L, D, P


def test_SparseCholeskyFactor_construtor() -> None:

    A = get_SPD_sparse_n11_example(seed=2026)
    L, D, P = _get_L_D_P(A)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`L` has shape (11, 11), "
            "`D` has shape (11, 11), and `L` has shape (10,)!\n"
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


def test_SparseCholeskyFactor() -> None:

    A = get_SPD_sparse_n11_example(seed=2026)
    Q = np.linalg.inv(A.toarray())
    scf = SparseCholeskyFactor(*_get_L_D_P(A))

    # Test to dense
    np.testing.assert_allclose(scf.todense(), A.toarray())
    # Test shape
    assert scf.shape == (11, 11)
    assert scf.n == 11

    # Diagonal
    np.testing.assert_allclose(scf.get_diagonal(), A.diagonal())
    np.testing.assert_allclose(scf.get_invdiagonal(), Q.diagonal())

    # Test solve
    expected_x = np.arange(scf.n, dtype=np.float64)
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

    A = get_SPD_sparse_example(100, 2025)
    SparseCholeskyFactor(*_get_L_D_P(A))


def test_load_precision_example_4225x() -> None:

    A = load_precision_example_4225x()
    SparseCholeskyFactor(*_get_L_D_P(A))
