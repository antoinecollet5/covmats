# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Provide covariance matrix representation.

Note: add some notes about:
https://github.com/arvindks/kle/blob/master/covariance/covariance.py

And cite Saibaba's phd thesis about the uncertainty and all.
"""

from __future__ import annotations

import abc
import logging
from abc import abstractmethod
from functools import cached_property
from time import time
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy as sp
from numpy.random import Generator, RandomState
from scipy.sparse import csc_array, csr_array
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.spatial.distance import cdist

from covmats._helpers import (
    check_random_state,
    get_pts_coords_regular_grid,
)
from covmats._sparse_helpers import (
    SparseCholeskyFactor,
)
from covmats._toeplitz import (
    create_toepliz_first_row,
    toeplitz_product,
)
from covmats._types import ArrayLike, NDArrayFloat, NDArrayInt


class CallBack:
    """Represents a callback instance."""

    __slots__: List[str] = ["res"]

    def __init__(self) -> None:
        """Initialize the instance."""
        self.res: List[NDArrayFloat] = []

    def __call__(self, rk) -> None:
        self.res.append(rk)

    @property
    def itercount(self) -> int:
        """Return the number of times the callback as been called."""
        return len(self.res)

    def clear(self) -> None:
        """Delete all results."""
        self.res = []


class CovarianceMatrix(LinearOperator, sp.stats.Covariance, abc.ABC):
    """
    Abstract representation of a covariance matrix.

    Calculations involving covariance matrices (e.g. data whitening,
    multivariate normal function evaluation) are often performed more
    efficiently using a decomposition of the covariance matrix instead of the
    covariance matrix itself. This class allows the user to construct an
    object representing a covariance matrix using any of several
    decompositions and perform calculations using a common interface.

    .. note::

        The `CovarianceMatrix` class cannot be instantiated directly. Instead, use
        one of the derived class:

        - :py:class:`CovViaDiagonal`
        - :py:class:`CovViaCholesky`
        - :py:class:`CovViaSparseCholesky`
        - :py:class:`CovViaPrecisionCholesky`
        - :py:class:`CovViaSparsePrecisionCholesky`
        - :py:class:`CovViaEigenFactorization`
        - :py:class:`CovViaEnsemble`

    """

    __slots__ = [
        "_rank",
        "_n_pts",
        "_log_pdet",
        "_allow_singular",
        "_subspace_size",
    ]

    def __init__(self) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        shape : Tuple[int, int]
            Shape (n, n) of the covariance matrix.
        log_pdet : float
            Log pseudo determinant of the matrix.
        rank : int
            Rank of the matrix representation.
        """
        # counters
        self.count: int = 0
        self.solvmatvecs: int = 0
        self.dtype = np.dtype("d")  # float64 for LinearOperator
        self._allow_singular = False  # must be invertible

    @property
    def n_pts(self) -> int:
        """Number of points in the domain (n)."""
        return self._n_pts

    @n_pts.setter
    def n_pts(self, n_pts: int) -> None:
        """Set the number of points in the domain (n)."""
        assert int(n_pts) > 0
        self._n_pts = n_pts

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_pts, self.n_pts)

    @abstractmethod
    def solve(self, b: NDArrayFloat) -> NDArrayFloat:
        """Solve Ax = b, with A, the current covariance matrix instance."""

    def get_diagonal(self) -> NDArrayFloat:
        """
        Return the diagonal entries of the matrix (variances).

        The matrix is never built explicitly. Instead the matvec interface is
        used to multiply all column of the identity matrix.
        """
        # Otherwise extract it using matrix vector products
        approx_diag = np.zeros(self.n_pts)
        for i in range(self.n_pts):
            # construct the ith row of the identity matrix
            v = np.zeros(self.n_pts)
            v[i] = 1.0
            approx_diag[i] = self.matvec(v)[i]
        return approx_diag

    def get_trace(self) -> float:
        """Return the trace of the covariance matrix."""
        return np.sum(self.get_diagonal()).item()

    @property
    def log_pdet(self) -> float:
        """
        Log of the pseudo-determinant of the covariance matrix. In linear algebra and
        statistics, the pseudo-determinant[1] is the product of all non-zero
        eigenvalues of a square matrix. It coincides with the regular determinant
        when the matrix is non-singular.
        """
        return np.array(self._log_pdet, dtype=float)[()]

    @property
    def rank(self) -> int:
        """
        Rank of the covariance matrix.
        """
        return np.array(self._rank, dtype=int)[()]

    @property
    def subspace_size(self) -> int:
        """
        Subspace size of the covariance matrix.
        """
        return self._subspace_size

    @abstractmethod
    def _todense(self) -> NDArrayFloat:
        """
        Return an explicit dense representation of the covariance matrix.
        """
        ...

    def todense(self) -> NDArrayFloat:
        """Explicit dense representation of the covariance matrix."""
        return self._todense()

    @property
    def covariance(self) -> NDArrayFloat:
        """
        Explicit dense representation of the covariance matrix.

        Alias for `todense()`.
        """
        return self.todense()

    @property
    @abstractmethod
    def precision(self) -> NDArrayFloat:
        """
        Explicit dense representation of the precision matrix.
        """
        ...

    @staticmethod
    def _validate_cov_mat(A: Union[NDArrayFloat, sp.sparse.sparray], name: str) -> None:
        m, n = A.shape[-2:]
        if (
            m != n
            or A.ndim != 2
            or not (
                np.issubdtype(A.dtype, np.integer)
                or np.issubdtype(A.dtype, np.floating)
            )
        ):
            message = (
                f"The input `{name}` must be a square, "
                "two-dimensional array of real numbers."
            )
            raise ValueError(message)

    @staticmethod
    def _validate_dense_matrix(A: ArrayLike, name: str) -> NDArrayFloat:
        A = np.atleast_2d(A)
        CovarianceMatrix._validate_cov_mat(A, name)
        return A

    @staticmethod
    def _validate_sparse_matrix(A: sp.sparse.sparray, name: str) -> sp.sparse.sparray:
        CovarianceMatrix._validate_cov_mat(A, name)
        return A

    @staticmethod
    def _validate_vector(A, name):
        A = np.atleast_1d(A)
        if A.ndim != 1 or not (
            np.issubdtype(A.dtype, np.integer) or np.issubdtype(A.dtype, np.floating)
        ):
            message = (
                f"The input `{name}` must be a one-dimensional array of real numbers."
            )
            raise ValueError(message)
        return A

    def _rmatvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return cov_obs @ x."""
        return self._matvec(x)

    def _matmat(self, X: NDArrayFloat) -> NDArrayFloat:
        """Return cov_obs @ X."""
        return self._matvec(X)

    def _rmatmat(self, X: NDArrayFloat) -> NDArrayFloat:
        """Return cov_obs @ X."""
        return self._matmat(X)

    @abc.abstractmethod
    def _whiten(self, x: NDArrayFloat) -> NDArrayFloat: ...

    def whiten(self, x):
        """
        Perform a whitening transformation on data.

        "Whitening" ("white" as in "white noise", in which each frequency has
        equal magnitude) transforms a set of random variables into a new set of
        random variables with unit-diagonal covariance. When a whitening
        transform is applied to a sample of points distributed according to
        a multivariate normal distribution with zero mean, the covariance of
        the transformed sample is approximately the identity matrix
        :cite:p:`WhiteningTransformation2025, novakGeneralizationColoringLinear2019`.

        Parameters
        ----------
        x : array_like
            An array of points. The last dimension must correspond with the
            dimensionality of the space, i.e., the number of columns in the
            covariance matrix.

        Returns
        -------
        x_ : array_like
            The transformed array of points.

        References
        ----------
        .. bibliography::
            :filter: False

            novakGeneralizationColoringLinear2019
            WhiteningTransformation2025

        Examples
        --------
        >>> import numpy as np
        >>> import covmats
        >>> rng = np.random.default_rng()
        >>> n = 3
        >>> A = rng.random(size=(n, n))
        >>> cov_array = A @ A.T  # make matrix symmetric positive definite
        >>> precision = np.linalg.inv(cov_array)
        >>> cov_object = covmats.CovViaPrecisionCholesky(
        .. sp.linalg.cholesky(precision, lower=True))
        >>> x = rng.multivariate_normal(np.zeros(n), cov_array, size=(10000))
        >>> x_ = cov_object.whiten(x)
        >>> np.cov(x_, rowvar=False)  # near-identity covariance
        array([[0.97862122, 0.00893147, 0.02430451],
               [0.00893147, 0.96719062, 0.02201312],
               [0.02430451, 0.02201312, 0.99206881]])

        """
        return self._whiten(np.asarray(x))

    @abc.abstractmethod
    def _colorize(self, x: NDArrayFloat) -> NDArrayFloat: ...

    def colorize(self, x):
        """
        Perform a colorizing transformation on data.

        "Colorizing" ("color" as in "colored noise", in which different
        frequencies may have different magnitudes) transforms a set of
        uncorrelated random variables into a new set of random variables with
        the desired covariance. When a coloring transform is applied to a
        sample of points distributed according to a multivariate normal
        distribution with identity covariance and zero mean, the covariance of
        the transformed sample is approximately the covariance matrix used
        in the coloring transform
        :cite:p:`WhiteningTransformation2025,novakGeneralizationColoringLinear2019`.

        Parameters
        ----------
        x : array_like
            An array of points. The last dimension must correspond with the
            dimensionality of the space, i.e., the number of columns in the
            covariance matrix.

        Returns
        -------
        x_ : array_like
            The transformed array of points.

        References
        ----------
        .. bibliography::
            :filter: False

            WhiteningTransformation2025
            novakGeneralizationColoringLinear2019

        Examples
        --------
        >>> import numpy as np
        >>> import covmats
        >>> rng = np.random.default_rng(1638083107694713882823079058616272161)
        >>> n = 3
        >>> A = rng.random(size=(n, n))
        >>> cov_array = A @ A.T  # make matrix symmetric positive definite
        >>> cholesky = np.linalg.cholesky(cov_array)
        >>> cov_object = covmats.CovViaCholesky(cholesky)
        >>> x = rng.multivariate_normal(np.zeros(n), np.eye(n), size=(10000))
        >>> x_ = cov_object.colorize(x)
        >>> cov_data = np.cov(x_, rowvar=False)
        >>> np.allclose(cov_data, cov_array, rtol=3e-2)
        True
        """
        return self._colorize(np.asarray(x))

    def sample_mvnormal(
        self,
        shape: Sequence[int],
        random_state: Optional[Union[int, RandomState, Generator]] = None,
    ) -> NDArrayFloat:
        """
        Draw samples from the multivariate normal N(0, Q).

        Parameters
        ----------
        shape: Sequence[int]
            Number of random vectors to sample. The resulting array will be of shape
            (*shape, n), n being the number of elements per random vector
            (the covariance) matrix has shape (n, n).
        random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]]
            Pseudorandom number generator state used to generate resamples.
            If `random_state` is ``None`` (or `np.random`), the
            `numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is used,
            seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance then that instance is used. The default is None.

        Return
        ------
        X: The transformed array of points. It has shape (*input shape, n), n being
        the number of elements per random vector (the covariance)
        matrix has shape (n, n).

        Examples
        --------
        >>> import numpy as np
        >>> import scipy as sp
        >>> import covmats
        >>> covd = covmats.CovViaDiagonal(np.array([5.0, 10.0, 15.0]))
        >>> rng_seed = 42
        >>> covd.sample_mvnormal(shape=[2], random_state=rng_seed)
        array([[ 1.11068661, -0.43723011,  2.50848692],
            [ 3.40559829, -0.74045799, -0.90680853]])
        >>> x = covd.sample_mvnormal(shape=[2, 4], random_state=rng_seed)
        >>> x
        array([[[ 1.11068661, -0.43723011,  2.50848692],
                [ 3.40559829, -0.74045799, -0.90680853],
                [ 3.53122721,  2.4268417 , -1.81826648],
                [ 1.21320114, -1.46545542, -1.80376358]],
               [[ 0.54104409, -6.05032338, -6.68057804],
                [-1.25731314, -3.20285323,  1.21707469],
                [-2.03040356, -4.46609644,  5.67643327],
                [-0.50485116,  0.21354293, -5.518026  ]]])
        >>> x.shape
        (2, 4, 3)
        >>> cov_cho = covmats.CovViaCholesky(sp.linalg.cholesky(covd.todense()))
        >>> cov_cho.sample_mvnormal(shape=[2], random_state=rng_seed)
        array([[-0.95777013, -1.11354406,  2.06162461],
            [ 0.81715777,  1.30517512,  1.66856257]])
        >>> cov_cho.sample_mvnormal(shape=[2, 2], random_state=rng_seed)
        array([[[ 1.11068661, -0.43723011,  2.50848692],
                [ 3.40559829, -0.74045799, -0.90680853],
                [ 3.53122721,  2.4268417 , -1.81826648],
                [ 1.21320114, -1.46545542, -1.80376358]],
               [[ 0.54104409, -6.05032338, -6.68057804],
                [-1.25731314, -3.20285323,  1.21707469],
                [-2.03040356, -4.46609644,  5.67643327],
                [-0.50485116,  0.21354293, -5.518026  ]]])
        """
        # A 1D diagonal of a covariance matrix was passed
        return self._colorize(
            check_random_state(seed=random_state).standard_normal(
                size=(*shape, self.subspace_size)
            )
        )

    @classmethod
    def from_cholesky(self, *args):
        """Representation of a covariance provided via choleksy factorization.

        .. error::
            Removed from parent class, do not use.
        """
        raise NotImplementedError(
            "`from_cholesky` is not available, please instantiate"
            " with `CovViaCholesky(...)` directly!"
        )

    @classmethod
    def from_diagonal(self, *args):
        """Representation of a covariance provided via diagonal.

        .. error::
            Removed from parent class, do not use.
        """
        raise NotImplementedError(
            "`from_diagonal` is not available, please instantiate"
            " with `CovViaDiagonal(...)` directly!"
        )

    @classmethod
    def from_eigendecomposition(self, *args):
        """
        Representation of a covariance provided via eigendecomposition.

        .. error::
            Removed from parent class, do not use.
        """
        raise NotImplementedError(
            "`from_eigendecomposition` is not available, please instantiate"
            " with `CovViaEigenFactorization(...)` directly!"
        )

    @classmethod
    def from_precision(self, *args):
        """
        Return a representation of a covariance from its precision matrix.

        .. error::
            Removed from parent class, do not use.
        """
        raise NotImplementedError(
            "`from_precision` is not available, please instantiate"
            " with `CovViaPrecisionCholesky(...)` directly!"
        )


def _dot_diag(x: NDArrayFloat, d: NDArrayFloat):
    # If d were a full diagonal matrix, x @ d would always do what we want.
    # Special treatment is needed for n-dimensional `d` in which each row
    # includes only the diagonal elements of a covariance matrix.
    return x * d if x.ndim < 2 else x * np.expand_dims(d, -2)


class CovViaDiagonal(CovarianceMatrix):
    r"""
    Representation of a covariance matrix via its diagonal.

    Attributes
    ----------
    diagonal : NDArrayFloat
        The diagonal elements of a diagonal matrix.

    Notes
    -----
    Let the diagonal elements of a diagonal covariance matrix :math:`D` be
    stored in the vector :math:`d`.

    When all elements of :math:`d` are strictly positive, whitening of a
    data point :math:`x` is performed by computing
    :math:`x \cdot d^{-1/2}`, where the inverse square root can be taken
    element-wise.
    :math:`\log\det{D}` is calculated as :math:`-2 \sum(\log{d})`,
    where the :math:`\log` operation is performed element-wise.

    This `Covariance` class supports singular covariance matrices. When
    computing ``_log_pdet``, non-positive elements of :math:`d` are
    ignored. Whitening is not well defined when the point to be whitened
    does not lie in the span of the columns of the covariance matrix. The
    convention taken here is to treat the inverse square root of
    non-positive elements of :math:`d` as zeros.

    Examples
    --------
    Prepare a symmetric positive definite covariance matrix ``A`` and a
    data point ``x``.

    >>> import numpy as np
    >>> import covmats
    >>> rng = np.random.default_rng()
    >>> n = 5
    >>> A = np.diag(rng.random(n))
    >>> x = rng.random(size=n)

    Extract the diagonal from ``A`` and create the `Covariance` object.

    >>> d = np.diag(A)
    >>> cov = covmats.CovViaDiagonal(d)

    Compare the functionality of the `Covariance` object against a
    reference implementations.

    >>> res = cov.whiten(x)
    >>> ref = np.diag(d**-0.5) @ x
    >>> np.allclose(res, ref)
    True
    >>> res = cov.log_pdet
    >>> ref = np.linalg.slogdet(A)[-1]
    >>> np.allclose(res, ref)
    True

    """

    __slots__ = ["_D", "_sqrtD", "_invD", "_i_zero"]

    def __init__(self, diagonal: ArrayLike) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        diagonal : ArrayLike
            The diagonal elements of a diagonal matrix.
        """
        self.D = diagonal
        super().__init__()

    @property
    def D(self) -> NDArrayFloat:
        return self._D

    @D.setter
    def D(self, D: NDArrayFloat) -> None:
        self._D = self._validate_vector(
            A=np.asarray(D, dtype=np.float64), name="diagonal"
        )
        nnv = np.count_nonzero(self.D < 0.0)
        if nnv != 0:
            raise ValueError(
                f"{nnv} negative values have been detected in `D` which "
                "means the matrix is indefinite and non invertible."
            )

        nnv = np.count_nonzero(self.D == 0.0)
        if nnv != 0:
            raise ValueError(
                f"{nnv} null values have been detected in `D` which "
                "means the matrix is singular and non invertible."
            )
        self.n_pts = np.size(self.D)
        self._subspace_size = self.n_pts
        self._rank = self.n_pts
        # Update the pseudo inverse as well
        self._invD = 1.0 / self.D
        # Store the square roots for whitening and colorizing transformations
        self._sqrtD = np.sqrt(self.D)
        self._sqrtinvD = np.sqrt(self._invD)
        # Update the log pseudo determinant
        self._log_pdet = np.sum(np.log(self.D), axis=-1)

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        return self.get_diagonal() * x

    def _matmat(self, X: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the matrix X."""
        return self.get_diagonal()[:, np.newaxis] * X

    def _whiten(self, x: NDArrayFloat) -> NDArrayFloat:
        return _dot_diag(x, self._sqrtinvD)

    def _colorize(self, x: NDArrayFloat) -> NDArrayFloat:
        return _dot_diag(x, self._sqrtD)

    def _support_mask(self, x):
        """
        Check whether x lies in the support of the distribution.
        """
        return ~np.any(_dot_diag(x, self._i_zero), axis=-1)

    def _todense(self) -> NDArrayFloat:
        return np.apply_along_axis(np.diag, -1, self.D)

    def solve(self, b: NDArrayFloat) -> NDArrayFloat:
        """Solve Ax = b, with A, the current covariance matrix instance."""

        return self._invD * b if b.ndim < 2 else (self._invD)[:, np.newaxis] * b

    def get_diagonal(self) -> NDArrayFloat:
        """Return the diagonal entries of the matrix (variances)."""
        return self.D

    @property
    def precision(self) -> NDArrayFloat:
        """
        Explicit dense representation of the precision matrix.
        """
        return np.diag(self._invD)


class CovViaCholesky(CovarianceMatrix):
    r"""
    Representation of a covariance via a Cholesky factorization.

    Parameters
    ----------
    cholesky : array_like
        The lower triangular Cholesky factor of the covariance matrix.

    Notes
    -----
    Let the covariance matrix be :math:`A` and :math:`L` be the lower
    Cholesky factor such that :math:`L L^T = A`.
    Whitening of a data point :math:`x` is performed by computing
    :math:`L^{-1} x`. :math:`\log\det{A}` is calculated as
    :math:`2tr(\log{L})`, where the :math:`\log` operation is performed
    element-wise.

    This `Covariance` class does not support singular covariance matrices
    because the Cholesky decomposition does not exist for a singular
    covariance matrix.

    Examples
    --------
    Prepare a symmetric positive definite covariance matrix ``A`` and a
    data point ``x``.

    >>> import numpy as np
    >>> import covmats
    >>> rng = np.random.default_rng()
    >>> n = 5
    >>> A = rng.random(size=(n, n))
    >>> A = A @ A.T  # make the covariance symmetric positive definite
    >>> x = rng.random(size=n)

    Perform the Cholesky decomposition of ``A`` and create the
    `Covariance` object.

    >>> L = np.linalg.cholesky(A)
    >>> cov = covmats.CovViaCholesky(L)

    Compare the functionality of the `Covariance` object against
    reference implementation.

    >>> from scipy.linalg import solve_triangular
    >>> res = cov.whiten(x)
    >>> ref = solve_triangular(L, x, lower=True)
    >>> np.allclose(res, ref)
    True
    >>> res = cov.log_pdet
    >>> ref = np.linalg.slogdet(A)[-1]
    >>> np.allclose(res, ref)
    True

    """

    __slots__ = ["_L"]

    def __init__(self, L: ArrayLike, covariance: Optional[ArrayLike] = None) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        L : NDArrayFloat
            Lower triangle of the covariance matrix Cholesky factorization.
        covariance : Optional[NDArrayFloat], optional
            Dense covariance matrix, by default None
        """
        self.L = L
        super().__init__()
        if covariance is not None:
            self._covariance = self.validate_matrix(covariance)

    @property
    def L(self) -> NDArrayFloat:
        """Lower triangle of the covariance matrix Cholesky factorization."""
        return self._L

    @L.setter
    def L(self, L: NDArrayFloat) -> None:
        """Set the lower triangle of the covariance matrix Cholesky factorization."""
        # TODO: validate lower triangle
        self._L = self._validate_matrix(L, "L")
        self._log_pdet = 2 * np.log(np.diag(self._L)).sum(axis=-1)
        self.n_pts = np.shape(self.L)[0]
        self._subspace_size = self.n_pts
        self._rank = self.n_pts

    @property
    def precision(self) -> NDArrayFloat:
        """
        Explicit dense representation of the precision matrix.
        """
        return None

    def _todense(self) -> NDArrayFloat:
        return self.L @ self.L.T

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        return np.linalg.multi_dot([self.L, self.L.T, x])

    def _whiten(self, x: NDArrayFloat) -> NDArrayFloat:
        m = x.T.shape[0]
        res = sp.linalg.solve_triangular(self.L, x.T.reshape(m, -1), lower=True)
        return res.reshape(x.T.shape).T

    def _colorize(self, x: NDArrayFloat) -> NDArrayFloat:
        return x @ self.L.T

    def solve(self, b: NDArrayFloat) -> NDArrayFloat:
        """Solve Ax = b, with A, the current covariance matrix instance."""
        return sp.linalg.cho_solve((self.L, True), b)


class CovViaSparseCholesky(CovarianceMatrix):
    r"""
    Representation of a covariance via a sparse Cholesky factorization.

    Notes
    -----
    Let the covariance matrix be :math:`A` and :math:`L` be the lower
    Cholesky factor such that :math:`L L^T = A`.
    Whitening of a data point :math:`x` is performed by computing
    :math:`L^{-1} x`. :math:`\log\det{A}` is calculated as
    :math:`2tr(\log{L})`, where the :math:`\log` operation is performed
    element-wise.

    This `Covariance` class does not support singular covariance matrices
    because the Cholesky decomposition does not exist for a singular
    covariance matrix.

    TODO: indicate that LGPL licence is expected here because it relies on CHOLMOD.

    Examples
    --------
    Prepare a symmetric positive definite covariance matrix ``A`` and a
    data point ``x``.

    >>> import numpy as np
    >>> import covmats
    >>> rng = np.random.default_rng()
    >>> n = 5
    >>> A = rng.random(size=(n, n))
    >>> A = A @ A.T  # make the covariance symmetric positive definite
    >>> x = rng.random(size=n)

    Perform the Cholesky decomposition of ``A`` and create the
    `Covariance` object.

    >>> L = np.linalg.cholesky(A)
    >>> cov = covmats.CovViaCholesky(L)

    Compare the functionality of the `Covariance` object against
    reference implementation.

    >>> from scipy.linalg import solve_triangular
    >>> res = cov.whiten(x)
    >>> ref = solve_triangular(L, x, lower=True)
    >>> np.allclose(res, ref)
    True
    >>> res = cov.log_pdet
    >>> ref = np.linalg.slogdet(A)[-1]
    >>> np.allclose(res, ref)
    True

    """

    __slots__ = ["_scf", "_sparse_covariance"]

    def __init__(
        self,
        scf: SparseCholeskyFactor,
        sparse_covariance: Optional[sp.sparse.sparray] = None,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        scf : SparseCholeskyFactor
            Lower triangle of the sparse covariance matrix Cholesky factorization.
        sparse_covariance : Optional[sp.sparse.sparray], optional
            Sparse covariance matrix, by default None
        """
        self.scf = scf
        if sparse_covariance is not None:
            self._validate_sparse_matrix(sparse_covariance, name="sparse_covariance")
        self._sparse_covariance: Optional[sp.sparse.sparray] = sparse_covariance
        super().__init__()

    @property
    def scf(self) -> SparseCholeskyFactor:
        return self._scf

    @scf.setter
    def scf(self, scf: SparseCholeskyFactor) -> None:
        self._scf = scf
        self._n_pts = self._scf.L.shape[0]
        self._log_pdet = self._scf.log_pdet
        self._subspace_size = self.n_pts
        self._rank = self._n_pts

    def _todense(self):
        return self._scf.todense()

    @property
    def precision(self) -> NDArrayFloat:
        """
        Explicit dense representation of the precision matrix.
        """
        return self._scf.inv()

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        return self._scf(x)

    def _whiten(self, x: NDArrayFloat) -> NDArrayFloat:
        return self._scf.whiten(x)

    def _colorize(self, x: NDArrayFloat) -> NDArrayFloat:
        return self._scf.colorize(x)

    def solve(self, b: NDArrayFloat) -> NDArrayFloat:
        """Solve Ax = b, with A, the current covariance matrix instance."""
        return self._scf.solve(b)

    def get_diagonal(self) -> NDArrayFloat:
        """
        Return the diagonal entries of the matrix (variances).
        The matrix is never built explicitly. Instead the matvec interface is
        used to multiply all column of the identity matrix.
        """
        return self._scf.get_diagonal()


class CovViaPrecisionCholesky(CovarianceMatrix):
    r"""
    Representation of a covariance via the Cholesky factorization of its inverse
    (aka the precision matrix).

    Notes
    -----
    Let the covariance matrix be :math:`A`, its precision matrix be
    :math:`P = A^{-1}`, and :math:`L` be the lower Cholesky factor such
    that :math:`L L^T = P`.
    Whitening of a data point :math:`x` is performed by computing
    :math:`x^T L`. :math:`\log\det{A}` is calculated as
    :math:`-2tr(\log{L})`, where the :math:`\log` operation is performed
    element-wise.

    This `Covariance` class does not support singular covariance matrices
    because the precision matrix does not exist for a singular covariance
    matrix.

    Examples
    --------
    Prepare a symmetric positive definite precision matrix ``P`` and a
    data point ``x``. (If the precision matrix is not already available,
    consider the other factory methods of the `Covariance` class.)

    >>> import numpy as np
    >>> import covmats
    >>> rng = np.random.default_rng()
    >>> n = 5
    >>> P = rng.random(size=(n, n))
    >>> P = P @ P.T  # a precision matrix must be positive definite
    >>> x = rng.random(size=n)

    Create the `Covariance` object.

    >>> cov = covmats.CovViaPrecisionCholesky(np.linalg.cholesky(P))

    Compare the functionality of the `Covariance` object against
    reference implementations.

    >>> res = cov.whiten(x)
    >>> ref = x @ np.linalg.cholesky(P)
    >>> np.allclose(res, ref)
    True
    >>> res = cov.log_pdet
    >>> ref = -np.linalg.slogdet(P)[-1]
    >>> np.allclose(res, ref)
    True

    """

    __slots__: List[str] = ["_L", "_precision"]

    def __init__(self, L: ArrayLike, precision: Optional[ArrayLike] = None) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        L : NDArrayFloat
            Lower triangle of the precision matrix Cholesky factorization.
        covariance : Optional[NDArrayFloat], optional
            Dense precision matrix, by default None

        Raises
        ------
        ValueError
            _description_
        """
        self.L = L
        super().__init__()
        if precision is not None:
            self._precision: Optional[NDArrayFloat] = self._validate_dense_matrix(
                precision, "precision"
            )
        else:
            self._precision = None

    @property
    def L(self) -> NDArrayFloat:
        """Lower triangle of the precision matrix Cholesky factorization."""
        return self._L

    @L.setter
    def L(self, L: NDArrayFloat) -> None:
        """Set the lower triangle of the precision matrix Cholesky factorization."""
        # TODO: validate lower triangle
        self._L = self._validate_matrix(L, "L")
        self._log_pdet = -2 * np.log(np.diag(self.L)).sum(axis=-1)
        self.n_pts = np.shape(self.L)[0]
        self._subspace_size = self.n_pts
        self._rank = self.n_pts

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        return sp.linalg.cho_solve((self.L, True), x)

    def _todense(self) -> NDArrayFloat:
        return sp.linalg.cho_solve((self.L, True), np.eye(self.n_pts))

    @property
    def precision(self) -> NDArrayFloat:
        """
        Explicit dense representation of the precision matrix.
        """
        if self._precision is not None:
            return self._precision
        return self.L @ self.L.T

    def _whiten(self, x: NDArrayFloat) -> NDArrayFloat:
        return x @ self.L

    def _colorize(self, x: NDArrayFloat) -> NDArrayFloat:
        m = x.T.shape[0]
        res = sp.linalg.solve_triangular(self.L.T, x.T.reshape(m, -1), lower=False)
        # L1^T @ b.T = x.T
        return res.reshape(x.T.shape).T

    def solve(self, b: NDArrayFloat) -> NDArrayFloat:
        """Solve Ax = b, with A, the current covariance matrix instance."""
        if self._precision is not None:
            return self._precision @ b
        return self.L @ (self.L.T) @ b


class CovViaSparsePrecisionCholesky(CovarianceMatrix):
    """
    Representation of a covariance via the sparse Cholesky factorization of its
    sparse inverse (aka the precision matrix).

    Warning
    -------
    This piece of code rely on
    `scikit-sparse <https://github.com/scikit-sparse/scikit-sparse>`_
    which itself depends on external libraries with GPL licenses, such as
    `SuiteSparse <https://github.com/DrTimothyAldenDavis/SuiteSparse?tab=License-1-ov-file>`_.
    As a consequence these pieces of code must adopt that license as well.
    Please look into the terms of this license before creating a dynamic
    link to this pieces in your downstream package and understand
    commercial use limitations. We are not lawyers and cannot provide any
    guidance on the terms of this license.

    Please see https://www.gnu.org/licenses/licenses.html#LGPL


    Notes
    -----
    Blablabla.
    """

    __slots__: List[str] = [
        "_sparse_precision",
        "_scf",
    ]

    def __init__(
        self,
        scf: SparseCholeskyFactor,
        sparse_precision: Optional[sp.sparse.sparray] = None,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        scf : SparseCholeskyFactor
            Lower triangle of the sparse precision matrix Cholesky factorization.
        sparse_precision : Optional[sp.sparse.sparray], optional
            Sparse precision matrix (inverse of the covariance matrix), by default None
        """
        self.scf = scf

        if sparse_precision is not None:
            self._sparse_precision = self._validate_sparse_matrix(
                sparse_precision, "sparse_precision"
            )
        self._sparse_precision: Optional[sp.sparse.sparray] = sparse_precision
        super().__init__()

    @property
    def scf(self) -> SparseCholeskyFactor:
        return self._scf

    @scf.setter
    def scf(self, scf: SparseCholeskyFactor) -> None:
        self._scf = scf
        self._n_pts = self._scf.L.shape[0]
        self._log_pdet = -self._scf.log_pdet
        self._subspace_size = self.n_pts
        self._rank = self._n_pts

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        return self._scf.solve(x)

    def _whiten(self, x: NDArrayFloat) -> NDArrayFloat:
        return self._scf.whiten_inv(x)

    def _colorize(self, x: NDArrayFloat) -> NDArrayFloat:
        return self._scf.colorize_inv(x)

    def solve(self, b: NDArrayFloat) -> NDArrayFloat:
        """Return x = A^{-1} b."""
        if self._sparse_precision is not None:
            return self._sparse_precision.dot(b)
        return self._scf.mat @ b

    def get_diagonal(self) -> NDArrayFloat:
        """
        Return the diagonal entries of the matrix (variances).
        The matrix is never built explicitly. Instead the matvec interface is
        used to multiply all column of the identity matrix.
        """
        return self._scf.get_invdiagonal()

    def _todense(self):
        return self._scf.inv()

    @property
    def precision(self) -> sp.sparse.sparray:
        if self._sparse_precision is not None:
            return self._sparse_precision
        return self._scf.mat


class CovViaEigenFactorization(CovarianceMatrix):
    r"""
    Representation of a covariance provided via eigenfactorization

    Parameters
    ----------
    eigenfactorization : sequence
        A sequence (nominally a tuple) containing the eigenvalue and
        eigenvector arrays as computed by `scipy.sparse.eigsh`.

    Notes
    -----
    Let the covariance matrix be :math:`A`, let :math:`V` be matrix of
    eigenvectors, and let :math:`W` be the diagonal matrix of eigenvalues
    such that `V W V^T = A`.

    When all of the eigenvalues are strictly positive, whitening of a
    data point :math:`x` is performed by computing
    :math:`x^T (V W^{-1/2})`, where the inverse square root can be taken
    element-wise.
    :math:`\log\det{A}` is calculated as  :math:`tr(\log{W})`,
    where the :math:`\log` operation is performed element-wise.

    This `Covariance` class supports singular covariance matrices. When
    computing ``_log_pdet``, non-positive eigenvalues are ignored.
    Whitening is not well defined when the point to be whitened
    does not lie in the span of the columns of the covariance matrix. The
    convention taken here is to treat the inverse square root of
    non-positive eigenvalues as zeros.

    Examples
    --------
    Prepare a symmetric positive definite covariance matrix ``A`` and a
    data point ``x``.

    >>> import numpy as np
    >>> import covmats
    >>> rng = np.random.default_rng()
    >>> n = 5
    >>> A = rng.random(size=(n, n))
    >>> A = A @ A.T  # make the covariance symmetric positive definite
    >>> x = rng.random(size=n)

    Perform the eigenfactorization of ``A`` and create the `Covariance`
    object.

    >>> w, v = np.linalg.eigh(A)
    >>> cov = covmats.CovViaEigenFactorization((w, v))

    Compare the functionality of the `Covariance` object against
    reference implementations.

    >>> res = cov.whiten(x)
    >>> ref = x @ (v @ np.diag(w**-0.5))
    >>> np.allclose(res, ref)
    True
    >>> res = cov.log_pdet
    >>> ref = np.linalg.slogdet(A)[-1]
    >>> np.allclose(res, ref)
    True

    """

    __slots__: List[str] = [
        "_LP",
        "_LA",
        "_w",
        "_v",
        "_null_basis",
        "_eps",
    ]

    def __init__(self, eigenfactorization: Tuple[NDArrayFloat, NDArrayFloat]) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        (eig_vals, eig_vects) : Tuple[NDArrayFloat, NDArrayFloat]
            - 1D vector of eigen values with size `n_pc`.
            - 2D arrays of eigen vectors (columns) with size `(Ns, n_pc)`. Ns being the
            number of elements in the original covariance matrix.
        """
        eigenvalues, eigenvectors = eigenfactorization
        # See if we do a common setter interface ?
        self.eig_vals = eigenvalues
        self.eig_vects = eigenvectors
        super().__init__()

    @property
    def n_pc(self) -> int:
        """
        Return the number of eigen vectors/values, i.e. principal components.

        It is determined from the eigen values vector size.
        """
        return self._w.size

    @property
    def subspace_size(self) -> int:
        return self.n_pc

    @property
    def eig_vals(self) -> NDArrayFloat:
        """Return the Eigen values."""
        return self._w

    @eig_vals.setter
    def eig_vals(self, value: NDArrayFloat) -> None:
        """Return the Eigen values."""
        self._w = np.reshape(value, (-1, 1))
        self._log_pdet = np.sum(np.log(self._w), axis=0).item()
        self._subspace_size = np.size(self._w)
        self._rank = self._subspace_size

    @property
    def eig_vects(self) -> NDArrayFloat:
        """Return the Eigen vectors."""
        return self._v

    @eig_vects.setter
    def eig_vects(self, value: NDArrayFloat) -> None:
        self._v = value
        self.n_pts = self._v.shape[0]

    def get_diagonal(self) -> NDArrayFloat:
        """
        Return the diagonal entries of the matrix (variances).
        """
        return np.sum(((self._v.T * np.sqrt(self._w)) ** 2), axis=0)

    def _todense(self) -> NDArrayFloat:
        return np.dot(self._v, np.multiply(self._w, self._v.T))

    def _whiten(self, x: NDArrayFloat) -> NDArrayFloat:
        # shape (r, n)
        return (
            (self._v.T * (1.0 / np.sqrt(self._w))).T @ self._v.T
        ) @ x  # x @ self._LP

    def _colorize(self, x: NDArrayFloat) -> NDArrayFloat:
        return x @ (self._v.T * np.sqrt(self._w))

    def _support_mask(self, x):
        """
        Check whether x lies in the support of the distribution.
        """
        # TODO: this is not correct either
        raise NotImplementedError(
            "_support_mask is not implemented for EigenFactorization"
        )
        residual = np.linalg.norm(x @ self._null_basis, axis=-1)
        in_support = residual < self._eps
        return in_support

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        return np.dot(
            self._v,
            np.multiply(self._w, np.dot(self._v.T, x.reshape(-1, 1))),
        )

    def _matmat(self, X: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        assert np.shape(X)[0] == self.shape[0]
        return np.dot(
            self._v,
            np.multiply(self._w, np.dot(self._v.T, X)),
        )

    def solve(self, b: NDArrayFloat) -> NDArrayFloat:
        r"""
        Return $Q^{-1} b = ZD^{-1}Z^{T}b$.

        Parameters
        ----------
        b: NDArrayFloat
            Column vector with shape ($N_{\mathrm{s}}$, 1) or ensemble matrix with
            shape ($N_{\mathrm{s}}$, $N_e$).

        Returns
        -------
        NDAarrayFloat
            Column vector with shape ($N_{\mathrm{s}}$, 1) or ensemble matrix with
            shape ($N_{\mathrm{s}}$, $N_e$).
        """
        # np.dot(invZs.T, invZs)
        # Note: x must be a column vector of a matrix with size (Ns, Ne)
        ne = 1  # case of a column vector
        if b.ndim > 1:
            ne = b.shape[1]
        return np.dot(
            self._v,
            np.multiply(1.0 / self._w, np.dot(self._v.T, b.reshape(-1, ne))),
        )

    def get_sparse_LLT_factor(self) -> csc_array:
        """
        Return the sparse factor L of the LL^T factorization of the eigen matrix.

        Return
        ------
        L: csc_array
            L = U * V^{T/2}.
        """
        # 1) Convert U sqrt(V) to a sparse format
        sp_mat = sp.sparse.lil_array(self._v * np.sqrt(self._w).T)
        # 2) Resize -> we now have a square matrix and indices are preserved
        sp_mat.resize(self.shape)
        # 3) Convert to column format
        return sp_mat.tocsc()

    @property
    def precision(self) -> sp.sparse.sparray:
        return np.dot(
            self._v,
            np.multiply(1.0 / self._w, self._v.T),
        )


class CovViaEnsemble(CovarianceMatrix):
    r"""
    Represents a covariance matrix as an ensemble of realizations.

    For a given ensemble with shape (:math:`N_{s}`, :math:`N_{e}`), the number of
    points and the number of members in the ensemble respectively, the covariance
    matrix :math:`\mathbf{\Sigma_{ss}}` is approximated from the ensemble
    in the standard way of EnKF
    :cite:p:`evensenDataAssimilationEnsemble2007,aanonsenEnsembleKalmanFilter2009`:

    .. math::
        \mathbf{\Sigma_{ss}} = \frac{1}{N_{e} - 1} \sum_{j=1}^{N_{e}}\left(s_{j} -
        \overline{s}\right)\left(s_{j}
        - \overline{s^{l}} \right)^{T}

    Or by defining a matrix of anomalies
    :math:`\mathbf{A} = \mathbf{S} - \overline{\mathbf{S}}`
    with shape  (:math:`N_{s}`, :math:`N_{e}`):

    .. math::
        \mathbf{\Sigma_{ss}} = \frac{1}{N_{e} - 1} \mathbf{A}^{T}\mathbf{A}

    Note
    ----
    Practically, the dense covariance matrix is never built,
    only the anomalies matrix :math:`\mathbf{A}` is used. The product between the
    inverse of the covariance matrix and a vector
    :math:`\mathbf{x} = \mathbf{\Sigma_{ss}}^{-1}\mathbf{b}`
    is obtained solving the system :math:`\mathbf{A}^{T}\mathbf{Ax} = \mathbf{b}`,
    using gmres, where only anomalies matrix vector products are required.

    References
    ----------
    .. bibliography::
        :filter: False

        evensenDataAssimilationEnsemble2007
        aanonsenEnsembleKalmanFilter2009

    """

    __slots__ = ["_ensemble"]

    def __init__(
        self,
        ensemble: NDArrayFloat,
    ) -> None:
        """
        Initiate the instance.

        Parameters
        ----------
        ensemble : NDArrayFloat
            Ensemble of realization with shape (:math:`N_{e}`, :math:`N_{s}`).
        """
        self.ensemble = ensemble
        super().__init__()
        # TODO Add SVD of anomalies

    @property
    def ensemble(self) -> NDArrayFloat:
        return self._ensemble

    @ensemble.setter
    def ensemble(self, value: NDArrayFloat) -> None:
        self._ensemble = value
        self._subspace_size, self.n_pts = np.shape(self.ensemble)
        self._rank = 0
        self._log_pdet = 0.0

    @cached_property
    def anomalies(self) -> NDArrayFloat:
        """
        Return the matrix of anomalies.

        """
        return self.ensemble - np.mean(self.ensemble, axis=0, keepdims=True)

    @cached_property
    def n_ens(self) -> int:
        """Return the number of members in the ensemble."""
        return self.ensemble.shape[0]

    @property
    def subspace_size(self) -> int:
        """
        Subspace size of the covariance matrix.
        """
        return self.n_ens

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x (dot product)."""
        return np.linalg.multi_dot([self.anomalies.T, self.anomalies, x]) / (
            self.n_ens - 1
        )

    def _whiten(self, x: NDArrayFloat) -> NDArrayFloat:
        # TODO with SVD of anomaly matrix
        raise NotImplementedError(
            "`whitening` is not implemented for an ensemble matrix yet!\n"
            "Please be patient!"
        )

    def _colorize(self, x: NDArrayFloat) -> NDArrayFloat:
        # TODO with SVD of anomaly matrix
        raise NotImplementedError(
            "`colorize` is not implemented for an ensemble matrix yet!\n"
            "Please be patient!"
        )

    def _todense(self) -> NDArrayFloat:
        """
        Return a dense representation of the matrix.
        """
        return self.anomalies.T @ self.anomalies / (self.n_ens - 1)

    def solve(
        self, b: NDArrayFloat, rtol: float = 1e-12, maxiter: int = 1000
    ) -> NDArrayFloat:
        """
        Solve A^{T}Ax = b, with A, the anomalies matrix instance.

        Note that the dense covariance matrix is never built.
        """
        residual = CallBack()

        x, info = gmres(
            self,
            b=b,
            rtol=rtol,
            maxiter=maxiter,
            callback=residual,
            callback_type="legacy",
            atol=0.0,
        )
        self.solvmatvecs += residual.itercount
        return x

    def get_diagonal(self) -> NDArrayFloat:
        """Return the diagonal entries of the matrix (variances)."""
        return np.sum((self.anomalies**2), axis=0) / (self.n_ens - 1.0)

    @property
    def precision(self) -> sp.sparse.sparray:
        raise NotImplementedError("Be patient...")


def _generate_dense_matrix_from_kernel(
    pts: NDArrayFloat, kernel: Callable, len_scale: NDArrayFloat, nugget: float = 0.0
) -> NDArrayFloat:
    """
    Generate a dense matrix.

    Compute O(dim^2) interactions.

    Parameters
    ----------
    pts : NDArrayFloat
        DESCRIPTION.
    kernel : TYPE
        DESCRIPTION.
    len_scale: NDArrayFloat
        DESCRIPTION.

    Returns
    -------
    NDArrayFloat
        The dense matrix.
    """
    # Scale the points coordinates
    scaled_pts = np.array(pts, copy=True)
    for dim in range(scaled_pts.shape[1]):
        scaled_pts[:, dim] /= len_scale[dim]
    return kernel(sp.spatial.distance_matrix(scaled_pts, scaled_pts))


class CovKernelAsLinop(abc.ABC, LinearOperator):
    __slots__: List[str] = [
        "_kernel",
        "_pts",
        "_nugget",
        "_preconditioner",
        "count",
        "solvematvecs",
    ]

    def __init__(
        self,
        pts: NDArrayFloat,
        kernel: Callable,
        len_scale: NDArrayFloat,
        nugget: float = 0.0,
        k: int = 100,
        is_use_preconditioner: bool = False,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        pts : NDArrayFloat
            _description_
        kernel : Callable
            _description_
        nugget : float, optional
            _description_, by default 0.0
        """
        self._kernel: Callable = kernel
        self._pts: NDArrayFloat = pts
        self._nugget: float = nugget
        self._len_scale: NDArrayFloat = len_scale
        # counters
        self.count: int = 0
        self.solvmatvecs: int = 0

        super().__init__(shape=(self.n_pts, self.n_pts), dtype="d")

        if is_use_preconditioner:
            self._preconditioner: Optional[csr_array] = _build_kernel_preconditioner(
                pts, kernel, k=k
            )
        else:
            self._preconditioner = None

    @property
    def n_pts(self) -> int:
        """Return the number of points covered."""
        return np.shape(self._pts)[0]

    def _todense(self) -> NDArrayFloat:
        """
        Explicit dense representation of the covariance matrix.
        """
        return _generate_dense_matrix_from_kernel(
            self._pts, self._kernel, self._len_scale, self._nugget
        )

    def get_diagonal(self) -> NDArrayFloat:
        """Return the diagonal entries of the matrix (variances)."""
        return self._kernel(np.zeros(len(self._pts)))

    def reset_comptors(self) -> None:
        """Set the comptors to zero."""
        self.count = 0
        self.solvmatvecs = 0

    def itercount(self) -> int:
        """Return the number of counts."""
        return self.count

    def solve(
        self, b: NDArrayFloat, rtol: float = 1e-12, maxiter: int = 1000
    ) -> NDArrayFloat:
        """Solve Ax = b, with A, the current covariance matrix instance."""
        residual = CallBack()
        x, info = gmres(
            self,
            b=b,
            rtol=rtol,
            maxiter=maxiter,
            callback=residual,
            callback_type="legacy",
            M=self._preconditioner,
            atol=0.0,
        )

        self.solvmatvecs += residual.itercount
        return x


def _build_kernel_preconditioner(
    pts: NDArrayFloat, kernel: Callable, k: int = 100
) -> csr_array:
    """
    Implementation of the preconditioner based on changing basis.

    Parameters
    ----------
    pts : NDArrayFloat
        The points (n, m) with n the number of data points and m the dimension of
        coordinates.
    kernel: Callable
        TODO
    k : int, optional
        Number of local centers in the preconditioner. Controls the sparity of
        the preconditioner. By default 100.

    Returns
    -------
    csr_array
        _description_

    Raises
    ------
    ValueError
        _description_

    Notes:
    ------
    Implementation of the preconditioner based on local centers.
    The parameter k controls the sparsity and the effectiveness of the preconditioner.
    Larger k is more expensive but results in fewer iterations.
    For large ill-conditioned systems, it was best to use a nugget effect to make the
    problem better conditioned.
    To Do: implementation based on local centers and additional points. Will remove the
    hack of using nugget effect.

    """
    nb_pts: int = pts.shape[0]
    if nb_pts <= 0:
        raise ValueError("The number of points cannot be null !")
    if nb_pts < k:
        raise ValueError("k must be superior to the number of points !")

    # Build the tree
    start: float = time()
    tree: sp.spatial.cKDTree = sp.spatial.cKDTree(pts, leafsize=32)
    end: float = time()

    logging.log(logging.INFO, f"Tree building time = {end - start}")

    # Find the nearest neighbors of all the points
    start = time()
    _dist, ind = tree.query(pts, k=k)
    end = time()

    logging.log(logging.INFO, f"Nearest neighbor computation time = {end - start}")

    Q = np.zeros((k, k), dtype="d")
    y = np.zeros((k, 1), dtype="d")

    row = np.tile(np.arange(nb_pts), (k, 1)).transpose()
    col = np.copy(ind)
    nu = np.zeros((nb_pts, k), dtype="d")

    y[0] = 1.0
    start = time()

    # TODO: This is very inefficient and must be re-written
    for i in np.arange(nb_pts):
        Q = kernel(cdist(pts[ind[i, :], :], pts[ind[i, :], :]))
        nui = sp.linalg.solve(Q, y)
        nu[i] = nui.ravel()

    end = time()

    logging.log(logging.INFO, "Elapsed time = %g" % (end - start))

    ij = np.zeros((nb_pts * k, 2), dtype="i")
    ij[:, 0] = np.copy(np.reshape(row, nb_pts * k, order="F").transpose())
    ij[:, 1] = np.copy(np.reshape(col, nb_pts * k, order="F").transpose())

    data = np.copy(np.reshape(nu, nb_pts * k, order="F").transpose())
    return csr_array((data, ij.transpose()), shape=(nb_pts, nb_pts), dtype="d")


class CovKernelAsLinopViaFFT(CovKernelAsLinop):
    """
    Represents a fast fourier transform covariance matrix.

    FFT based operations if kernel is stationary or translation invariant and points
    are on a regular grid.
    """

    __slots__: List[str] = ["_first_row"]

    def __init__(
        self,
        kernel,
        mesh_dim: Union[float, NDArrayFloat, Sequence[float]],
        domain_shape: Union[int, NDArrayInt, Sequence[int]],
        len_scale: NDArrayFloat,
        nugget: float = 0.0,
        k: int = 100,
        is_use_preconditioner: bool = False,
    ) -> None:
        """_summary_

        Parameters
        ----------
        kernel : _type_
            _description_
        mesh_dim : Union[NDArrayInt, Tuple[float, float]]
            _description_
        domain_shape : Union[NDArrayInt, Tuple[int, int]]
            _description_
        len_scale : NDArrayFloat
            _description_
        nugget : float, optional
            _description_, by default 0.0
        k : int, optional
            Number of local centers in the preconditioner. Controls the sparity of
            the preconditioner. It should be inferior to the number of points.
            By default 100.
        is_use_preconditioner: bool
            Whether to build the preconditioner at instance creation and use it to
            solve Ax = b systems. The default is False.
        """
        self.param_shape: NDArrayInt = np.array(domain_shape, dtype=np.int8)
        # Coordinates of the points in the grid with shape (Npts, Ndim)
        pts = get_pts_coords_regular_grid(mesh_dim, self.param_shape)

        self._first_row = create_toepliz_first_row(pts, kernel, len_scale)
        super().__init__(
            pts,
            kernel,
            len_scale=len_scale,
            nugget=nugget,
            k=k,
            is_use_preconditioner=is_use_preconditioner,
        )

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        return toeplitz_product(x, self._first_row, self.param_shape) * (
            1 + self._nugget
        )


def get_linop_eigen_factorization(
    linop: LinearOperator,
    size: int,
    n_pc: int,
    random_state: Optional[Union[int, RandomState, Generator]] = None,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Compute Eigenmodes of the covariance.

    Parameters
    ----------
    linop: LinearOperator
        The covariance matrix as a linear operator to factorize.
    size: int
        Input vector size for the linear operator.
    n_pc : int
        Number of principal component in the matrix.
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]]
        Pseudorandom number generator state used to generate resamples.
        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used.

    Raises
    ------
    NotImplementedError
        If a method difference from arpack is used for decomposition.

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat]
        Eigen values and eigen vectors.
    """
    logging.info("Eigen factorization of linear operator")

    # twopass = False if not 'twopass' in self.params else self.params['twopass']
    start = time()

    # Random state for v0 vector used by eigsh and svds
    if random_state is not None:
        random_state = check_random_state(random_state)
        v0 = random_state.uniform(size=(size,))
    else:
        v0 = None

    eig_vals, eig_vects = sp.sparse.linalg.eigsh(linop, k=n_pc, v0=v0)
    eig_vals = eig_vals[::-1]
    eig_vals = eig_vals.reshape(-1, 1)  # make a column vector
    eig_vects = eig_vects[:, ::-1]

    logging.info(
        "- time for eigenfactorization with k = %d is %g sec"
        % (n_pc, round(time() - start))
    )

    if (eig_vals > 0).sum() < n_pc:
        n_pc = (eig_vals > 0).sum()
        eig_vals = eig_vals[:n_pc, :]
        eig_vects = eig_vects[:, :n_pc]
        logging.warning("Warning: n_pc changed to %d for positive eigenvalues" % (n_pc))

    logging.info(
        f"- 1st eigv : {eig_vals[0]}, {n_pc}-th eigv : {eig_vals[-1]}, "
        f"ratio: {eig_vals[-1] / eig_vals[0]}"
    )
    return eig_vals, eig_vects


def eigen_factorize_cov_mat(
    cov_mat: Union[CovarianceMatrix, CovKernelAsLinop],
    n_pc: int,
    random_state: Optional[Union[int, RandomState, Generator]] = None,
) -> CovViaEigenFactorization:
    """
    Return an eigen factorized covariance matrix from the input covariance matrix.

    Parameters
    ----------
    cov_mat : Union[CovarianceMatrix, CovKernelAsLinop]
        The covariance matrix instance to decompose.
    n_pc : int
        Number of principal component in the matrix.
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]]
        Pseudorandom number generator state used to generate resamples.
        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used.

    Returns
    -------
    CovViaEigenFactorization
        Decomposed matrix instance.
    """
    if isinstance(cov_mat, CovViaEigenFactorization):
        return cov_mat
    return CovViaEigenFactorization(
        get_linop_eigen_factorization(cov_mat, cov_mat.shape[0], n_pc, random_state)
    )


def get_explained_var(
    eigval: NDArrayFloat,
    cov_mat: Optional[CovarianceMatrix] = None,
    trace_cov_mat: Optional[float] = None,
) -> NDArrayFloat:
    """Return the variance explained by each eigen value."""
    if trace_cov_mat is not None:
        return eigval / trace_cov_mat
    if cov_mat is not None:
        return eigval / cov_mat.get_trace()
    else:
        raise ValueError("You must provide a Covariance matrix instance or the trace !")
