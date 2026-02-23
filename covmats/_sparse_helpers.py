# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Wrap the sparse cholesky factorization from"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy.sparse import csc_array, csc_matrix, find

from covmats._helpers import check_random_state
from covmats._types import NDArrayFloat

try:
    from sksparse.cholmod import Factor as SparseChoFactor
    from sksparse.cholmod import cholesky
except (ImportError, ModuleNotFoundError):

    class SparseChoFactor:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "sksparse could not be loaded. Please "
                " install it to use SparseChoFactor (see "
                "https://scikit-sparse.readthedocs.io/en/latest/overview.html#installation"
                ")"
            )

        def __call__(self, values: np.ndarray) -> np.ndarray: ...

        def D(self) -> NDArrayFloat:  # ty:ignore[empty-body]
            ...

        def L(self) -> NDArrayFloat:  # ty:ignore[empty-body]
            ...

        def apply_P(self, x: NDArrayFloat) -> NDArrayFloat:  # ty:ignore[empty-body]
            ...

        def apply_Pt(self, x: NDArrayFloat) -> NDArrayFloat:  # ty:ignore[empty-body]
            ...

        def solve_Lt(self, x: NDArrayFloat) -> NDArrayFloat:  # ty:ignore[empty-body]
            ...

        def slogdet(self) -> float:  # ty:ignore[empty-body]
            ...

        @property
        def shape(self) -> Tuple[int, int]:  # ty:ignore[empty-body]
            ...

    def cholesky(mat: csc_matrix) -> SparseChoFactor:
        raise ModuleNotFoundError(
            "sksparse could not be loaded. Please"
            " install it to use sparse_cholesky (see "
            "https://scikit-sparse.readthedocs.io/en/latest/overview.html#installation"
            ")"
        )


def sparse_cholesky(arr: Union[csc_matrix, csc_array]) -> SparseChoFactor:
    # see: https://github.com/scikit-sparse/scikit-sparse/issues/108
    # see: https://github.com/scikit-sparse/scikit-sparse/pull/102
    return cholesky(csc_matrix(arr))


def assert_allclose_sparse(A, B, atol=1e-8, rtol=1e-8) -> None:
    """Assert that two sparse matrices or arrays are almost equal."""
    # If you want to check matrix shapes as well
    assert np.array_equal(A.shape, B.shape)
    r1, c1, v1 = find(A)
    r2, c2, v2 = find(B)
    np.testing.assert_equal(r1, r2)
    np.testing.assert_equal(c1, c2)
    np.testing.assert_allclose(v1, v2, atol=atol, rtol=rtol)


def get_sparse_covmat_variance(
    Q: csc_array, cholQ: Optional[SparseChoFactor]
) -> NDArrayFloat:
    """
    Extract efficiently the diagonal of the covariance matrix from the precision matrix.

    It relies on the linear operator `matvec` operation and consequenlty does not
    require to build the dense matrix which is much longer and generally untractable
    for large-scale problems.

    Parameters
    ----------
    hess_inv : LbfgsInvHessProduct
        Linear operator for the L-BFGS approximate inverse Hessian.

    Returns
    -------
    NDArrayFloat
        The diagonal of the L-BFGS approximated inverse Hessian.
    """
    # perform the cholesky factorization -> solving is then much faster
    if cholQ is None:
        _cholQ = sparse_cholesky(Q.tocsc())
    else:
        _cholQ = cholQ
    n_params = Q.shape[0]
    cov_mat_diag = np.zeros(n_params)
    v = np.zeros(n_params)
    for i in range(n_params):
        v[i - 1] = 0.0
        v[i] = 1.0
        cov_mat_diag[i] = _cholQ(v)[i]
    return cov_mat_diag


def sample_from_sparse_cov_factor(
    mean: NDArrayFloat,
    factor: csc_array,
    n_samples: int = 100,
    random_state: Optional[
        Union[int, np.random.Generator, np.random.RandomState]
    ] = None,
) -> NDArrayFloat:
    r"""
    Sample from the given sparse factor of the covariance matrix and the given mean.

    Parameters
    ----------
    mean: NDArrayFloat
        Mean of the field with shape $N_{\mathrm{s}}$.
    factor: NDArrayFloat
        Sparse factor of the covariance matrix from which to sample from. It has shape
        $(N_{\mathrm{s}} \times N_{\mathrm{s}})$.
    n_samples: int
        The number of samples required ($N_{\mathrm{e}}$). By default 100.
    random_state : Optional[Union[int, np.random.Generator, np.random.RandomState]]
        Pseudorandom number generator state used to generate resamples.
        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used. The default is None

    Returns
    -------
    NDArrayFloat
        The ensemble of realizations with shape
        $(N_{\mathrm{s}} \times N_{\mathrm{e}})$
    """
    if random_state is not None:
        _random_state = check_random_state(random_state)
    else:
        _random_state = np.random.default_rng()
    return factor @ _random_state.normal(
        scale=1.0, size=(factor.shape[0], n_samples)
    ) + mean.reshape(-1, 1)
