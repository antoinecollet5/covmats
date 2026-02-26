import numpy as np
import scipy as sp
import sksparse
from covmats import SparseCholeskyFactor
from covmats._sparse_helpers import get_SPD_sparse_n11_example


def _get_L_D_P(A: sp.sparse.sparray):
    """
    Return L, D and P from the factorization L @ D @ L' = P @ A @ P' using sksparse.

    Note that sksparse uses SuiteSparse which is LGPL licence.
    """
    # Need to take the API change into account
    try:
        # sksparse 4.x
        L, D, P = sksparse.cholmod.ldl(A, order="amd")
    except AttributeError:
        # sksparse 5.x
        f = sksparse.cholmod.cholesky(A)
        (L, D), P = f.L_D(), f.P()
    return L, D, P


def test_SparseCholeskyFactor() -> None:

    A = get_SPD_sparse_n11_example(seed=2026)
    scf = SparseCholeskyFactor(*_get_L_D_P(A))

    # Test to dense
    np.testing.assert_allclose(scf.todense(), A.toarray())
    # Test shape
    assert scf.shape == (11, 11)
    assert scf.n == 11

    # Test solve
    expected_x = np.arange(scf.n, dtype=np.float64)
    b = A @ expected_x
    np.allclose(scf.solve(b), expected_x)

    # Test colorize and whiten
    np.testing.assert_allclose(
        scf.whiten(scf.colorize(np.eye(11))), np.eye(11), atol=1e-15
    )

    # Test colorize with an ensemble
    pass
