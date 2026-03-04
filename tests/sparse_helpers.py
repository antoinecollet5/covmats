from typing import no_type_check

import scipy as sp


@no_type_check
def _get_L_D_P(A: sp.sparse.sparray):
    """
    Return L, D and P from the factorization L @ D @ L' = P @ A @ P' using sksparse.

    Note that sksparse uses SuiteSparse which is LGPL licence.
    """
    import sksparse.cholmod as cholmod

    # Need to take the API change into account
    try:
        # sksparse 4.x
        L, D, P = cholmod.ldl(A, order="amd")
    except AttributeError:
        # sksparse 5.x
        f = cholmod.cholesky(A)
        (L, D), P = f.L_D(), f.P()
    return L, D, P
