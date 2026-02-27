# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Wrap the sparse cholesky factorization from"""

from typing import Optional, Tuple, TypeVar, Union, overload

import numpy as np
import scipy as sp

from covmats._helpers import check_random_state
from covmats._types import ArrayLike, NDArrayFloat, NDArrayInt

# Define a type variable for the input/output type
T = TypeVar("T", NDArrayFloat, sp.sparse.csc_array)


class SparseCholeskyFactor:
    """Sparse cholesky factor of a matrix A.

    TODO.
    """

    __slots__ = ["_P", "_Pt", "_D", "_invD", "_sqrtD", "_sqrtinvD", "_L"]

    def __init__(
        self, L: sp.sparse.sparray, D: sp.sparse.sparray, P: ArrayLike
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        L : sp.sparse.sparray
            "The diagonal elements of `L` are assumed to be 1!".
        D : sp.sparse.sparray
            _description_
        P : ArrayLike
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        # Make sure that is it 1D and integer
        self.P = np.asarray(P, dtype=np.int64).ravel()
        self.D = D
        self.L = L

        # Assert shapes are ok.
        if self.D.shape != self.shape or self.P.size != self.n:
            raise ValueError(
                f"`L` has shape {self.shape},"
                f" `D` has shape {D.shape}, and `L` has shape {self.P.shape}!\n"
                f"To be consistent, `D` and `P` are expected with shape {self.shape}"
                f" and ({self.n},) respectively."
            )

    @property
    def L(self) -> sp.sparse.csc_array:
        return self._L

    @L.setter
    def L(self, L: sp.sparse.csc_array) -> None:
        # Extract the diagonal
        try:
            np.testing.assert_allclose(L.diagonal(), np.ones(L.shape[0]))
        except AssertionError as e:
            raise ValueError("The diagonal elements of `L` are assumed to be 1!") from e

        self._L = L

    @property
    def D(self) -> sp.sparse.csc_array:
        return self._D

    @D.setter
    def D(self, D: sp.sparse.csc_array) -> None:
        nnv = np.count_nonzero(D.diagonal() < 0.0)
        if nnv != 0:
            raise ValueError(
                f"{nnv} negative values have been detected in `D` which "
                "means the factorized matrix is indefinite and non invertible."
            )

        nnv = np.count_nonzero(D.diagonal() == 0.0)
        if nnv != 0:
            raise ValueError(
                f"{nnv} null values have been detected in `D` which "
                "means the factorized matrix is singular and non invertible."
            )

        self._D = D
        # Update the pseudo inverse as well
        self._invD = sp.sparse.diags(1.0 / self.D.diagonal())
        # Store the square roots for whitening and colorizing transformations
        self._sqrtD = np.sqrt(self.D)
        self._sqrtinvD = np.sqrt(self.invD)

    @property
    def invD(self) -> sp.sparse.csc_array:
        return self._invD

    @property
    def sqrtinvD(self) -> sp.sparse.csc_array:
        return self._sqrtinvD

    @property
    def sqrtD(self) -> sp.sparse.csc_array:
        return self._sqrtD

    @property
    def P(self) -> NDArrayInt:
        return self._P

    @P.setter
    def P(self, P: ArrayLike) -> None:
        self._P = np.asarray(P, dtype=np.int64).ravel()
        # Update the back pivot
        self._Pt: NDArrayInt = np.argsort(self._P)

    @property
    def Pt(self) -> NDArrayInt:
        return self._Pt

    @overload
    def apply_P(self, x: NDArrayFloat) -> NDArrayFloat: ...

    @overload
    def apply_P(self, x: sp.sparse.csc_array) -> sp.sparse.csc_array: ...

    def apply_P(self, x: T) -> T:
        return x[self.P]

    @overload
    def apply_Pt(self, x: NDArrayFloat) -> NDArrayFloat: ...

    @overload
    def apply_Pt(self, x: sp.sparse.csc_array) -> sp.sparse.csc_array: ...

    def apply_Pt(self, x: T) -> T:
        return x[self.Pt]

    def solve(self, b: NDArrayFloat) -> NDArrayFloat:
        # L @ D @ L' = P @ A @ P'
        # A x = b
        # P' L @ D @ L' P x = b
        # x = P' L^-T @ D^{-1} @ L^{-1} P b
        return self.apply_Pt(
            sp.sparse.linalg.spsolve_triangular(
                self.L.T,
                self.invD
                @ sp.sparse.linalg.spsolve_triangular(
                    self.L, self.apply_P(b), lower=True, unit_diagonal=True
                ),
                lower=False,
                unit_diagonal=True,
            )
        )

    @property
    def log_pdet(self) -> float:
        return np.log(np.prod(self.D.diagonal()))

    @property
    def n(self) -> int:
        return self.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.L.shape[:2]

    @property
    def mat(self) -> sp.sparse.csc_array:
        # Permute and rebuild
        L = self.apply_Pt(self.L)
        return L @ self.D @ L.T

    def todense(self) -> NDArrayFloat:
        # Permute and rebuild
        return self.mat.toarray()

    def inv(self) -> NDArrayFloat:
        return self.solve(np.eye(self.n))

    def get_diagonal(self) -> NDArrayFloat:
        # Permute and rebuild
        return self.mat.diagonal()

    def get_invdiagonal(self) -> NDArrayFloat:
        cov_mat_diag = np.zeros(self.n)
        v = np.zeros(self.n)
        for i in range(self.n):
            v[i - 1] = 0.0
            v[i] = 1.0
            cov_mat_diag[i] = self.solve(v)[i]
        return cov_mat_diag

    def colorize(self, x: NDArrayFloat) -> NDArrayFloat:
        # We want to solve z.T = x @ K.T, where A = K @ K^{T}
        # We use the cholesky factorization LDL' = PA'AP'
        # with P' = P^{-1} the permutation that makes the decomposition unique.
        # So LD^{1/2} = PA' and A = D^{1/2}L'P
        # Finally z = (P' L D^{1/2} x')'
        return self.apply_Pt(self.L @ self.sqrtD @ x.T).T

    def colorize_inv(self, x: NDArrayFloat) -> NDArrayFloat:
        return self.apply_Pt(
            sp.sparse.linalg.spsolve_triangular(
                self.L.T, self.sqrtinvD @ x.T, unit_diagonal=True, lower=False
            )
        ).T

    def whiten(self, x: NDArrayFloat) -> NDArrayFloat:
        # We use the cholesky factorization LDL' = PA'AP'
        # We want to solve x = z @ A.T =>  z = x A^{-T}
        # with P' = P^{-1} the permutation that makes the decomposition unique.
        # So LD^{1/2} = PA' and A = D^{1/2}L'P
        # Finally z = (D^{-1/2} L^{-1} P x')'
        return (
            self.sqrtinvD
            @ sp.sparse.linalg.spsolve_triangular(
                self.L, self.apply_P(x.T), unit_diagonal=True, lower=True
            )
        ).T

    def whiten_inv(self, x: NDArrayFloat) -> NDArrayFloat:
        return self.sqrtD @ (self.L.T @ self.apply_P(x.T)).T


def assert_allclose_sparse(A, B, atol=1e-8, rtol=1e-8) -> None:
    """Assert that two sparse matrices or arrays are almost equal."""
    # If you want to check matrix shapes as well
    assert np.array_equal(A.shape, B.shape)
    r1, c1, v1 = sp.sparse.find(A)
    r2, c2, v2 = sp.sparse.find(B)
    np.testing.assert_equal(r1, r2)
    np.testing.assert_equal(c1, c2)
    np.testing.assert_allclose(v1, v2, atol=atol, rtol=rtol)


def sample_from_sparse_cov_factor(
    mean: NDArrayFloat,
    factor: sp.sparse.csc_array,
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


def get_SPD_sparse_n11_example(seed: int) -> sp.sparse.csc_array:
    """
    Create a symmetric positive definite matrix of shape(11, 11).

    Parameters
    ----------
    seed: int
        Random seed.
    """
    N = 11
    rows = np.array([5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10])
    cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 9])
    rng = np.random.default_rng(seed)
    vals = rng.random(len(rows), dtype=np.float64)
    L = sp.sparse.coo_array((vals, (rows, cols)), shape=(N, N))
    A = L + L.T  # make it symmetric
    A.setdiag(2.0)  # make it strongly positive definite
    A = A.tocsc()
    return A


def get_SPD_sparse_example(
    n: int, seed: int, diag_mean: float = 10.0
) -> sp.sparse.csc_array:
    """
    Create a symmetric positive definite matrix.

    Parameters
    ----------
    n : int
        Number of elements.
    seed : int
        Random seed.

    Returns
    -------
    sp.sparse.cscarray
        CSC sparse SPD.
    """
    # Initialize a sparse matrix
    A = sp.sparse.lil_matrix((n, n))

    # Fill the diagonal with variances (non-zero values)
    np.random.seed(42)  # for reproducibility
    variances = np.random.uniform(-1.5, 1.5, n) + diag_mean
    for i in range(n):
        A[i, i] = variances[i]

    # Add some random non-zero covariances
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < 0.1:  # 10% chance of non-zero covariance
                cov = np.random.uniform(-0.5, 0.5)
                A[i, j] = cov
                A[j, i] = cov  # symmetry
    return A.tocsc()
