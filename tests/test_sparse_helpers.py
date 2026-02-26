import covmats
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
