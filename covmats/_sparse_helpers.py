# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Wrap the sparse cholesky factorization from"""

import warnings
from functools import cached_property
from typing import Tuple, TypeVar, overload

import numpy as np
import scipy as sp

from covmats._types import ArrayLike, NDArrayFloat, NDArrayInt

# Define a type variable for the input/output type
T = TypeVar("T", NDArrayFloat, sp.sparse.csc_array)

_fact_ds = (
    r":math:`\mathbf{LDL}^{\mathrm{T}} = \mathbf{PAP}^{\mathrm{T}}` factorization"
)


class _PickleSafeLinearOperator(sp.sparse.linalg.LinearOperator):
    """
    A LinearOperator subclass whose pickling correctly preserves state stored in
    __slots__, on every scipy/Python version.

    Background
    ----------
    scipy >= 1.18.0 added ``LinearOperator.__getstate__``, which serializes only
    ``self.__dict__`` (apparently to make the new Array API namespace reference
    ``_xp`` picklable) and has no awareness of ``__slots__`` at all. Any subclass
    storing its state in ``__slots__`` -- a common, memory-conscious pattern for
    ``LinearOperator`` subclasses -- silently loses that state on every pickle/unpickle
    round-trip. No error is raised at pickle time; the failure only surfaces later,
    wherever the first dropped attribute happens to get accessed (which can be
    arbitrarily far from the pickling site, e.g. inside a multiprocessing/joblib
    worker process). This has been reported upstream;
    see https://github.com/scipy/scipy/issues/25871.

    This class normalizes this issue in one consistent implementation, so subclasses
    just inherit from it in place of ``LinearOperator`` directly and never have to
    think about this again.

    Usage
    -----
    Replace::

        class Foo(LinearOperator, ...):
            ...

    with::

        class Foo(_PickleSafeLinearOperator, ...):
            ...

    No other changes needed. Works whether ``Foo`` declares its own ``__slots__`` or
    not, whether it has multiple bases, and whether ``Foo``
    (or anything else in its MRO) also has a ``__dict__``.

    Examples
    --------
    >>> import copy
    >>> import numpy as np
    >>> class Diagonal(_PickleSafeLinearOperator):
    ...     __slots__ = ["_d"]
    ...     def __init__(self, d):
    ...         self._d = np.asarray(d, dtype=np.float64)
    ...         super().__init__(dtype=self._d.dtype, shape=(len(d),) * 2)
    ...     def _matvec(self, x):
    ...         return self._d * x
    >>> op = Diagonal([1.0, 2.0, 3.0])
    >>> op2 = copy.deepcopy(op)
    >>> op2.matvec(np.ones(3))
    array([1., 2., 3.])
    """

    __slots__: list = []

    def __getstate__(self) -> dict:
        base_getstate = getattr(super(), "__getstate__", None)
        raw = base_getstate() if base_getstate is not None else None

        state: dict = {}
        if isinstance(raw, tuple):
            # object.__getstate__()'s default form when __slots__ is present:
            # (dict_state_or_None, slots_state_or_None).
            dict_part, slots_part = raw
            if dict_part:
                state.update(dict_part)
            if slots_part:
                for slot_name, slot_value in slots_part.items():
                    state[f"__slot__{slot_name}"] = slot_value
        elif raw:
            # e.g. scipy>=1.18's LinearOperator.__getstate__: a plain dict,
            # __dict__ only.
            state.update(raw)

        # Add every __slots__ attribute declared anywhere in the MRO that isn't already
        # accounted for above (harmless to re-add ones that are, since the value is
        # identical either way).
        for klass in type(self).__mro__:
            for slot in getattr(klass, "__slots__", ()) or ():
                if slot in ("__dict__", "__weakref__"):
                    continue
                if hasattr(self, slot):
                    state[f"__slot__{slot}"] = getattr(self, slot)
        return state

    def __setstate__(self, state) -> None:
        if isinstance(state, tuple):
            dict_part, slots_part = state
            merged: dict = {}
            if dict_part:
                merged.update(dict_part)
            if slots_part:
                for slot_name, slot_value in slots_part.items():
                    merged[f"__slot__{slot_name}"] = slot_value
            state = merged

        slot_items = {
            k[len("__slot__") :]: v
            for k, v in state.items()
            if k.startswith("__slot__")
        }
        dict_state = {k: v for k, v in state.items() if not k.startswith("__slot__")}

        base_setstate = getattr(super(), "__setstate__", None)
        if base_setstate is not None:
            try:
                base_setstate(dict_state)
            except TypeError:
                if dict_state and hasattr(self, "__dict__"):
                    self.__dict__.update(dict_state)
        elif dict_state and hasattr(self, "__dict__"):
            self.__dict__.update(dict_state)

        for slot, value in slot_items.items():
            setattr(self, slot, value)


class SparseCholeskyFactor(_PickleSafeLinearOperator):
    r"""
    Sparse Cholesky factor of a matrix :math:`\mathbf{A}`.

    .. _factref:

    .. admonition:: Factorization

        It relies on the :math:`\mathbf{LDL}^{\mathrm{T}} =`
        :math:`\mathbf{PAP}^{\mathrm{T}}` factorization where

        - :math:`\mathbf{L}` is the sparse lower triangle with unit diagonal.
        - :math:`\mathbf{D}` is the sparse diagonal matrix with diagonal entries.
        - :math:`\mathbf{P}` is the rows permutation vector.

    Note
    ----
    Since this code is released under BSD 3-Clause License, it is not possible to
    include sparse factors provided by
    `scikit-sparse <https://github.com/scikit-sparse/scikit-sparse>`_,
    as it depends on external libraries with GPL licenses, such as
    `SuiteSparse <https://github.com/DrTimothyAldenDavis/SuiteSparse?tab=License-1-ov-file>`_.
    The main goal of this implementation is therefore to mimic useful features while
    preserving the BSD 3-Clause License compatibility.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> import covmats
    >>> Q = covmats.get_SPD_sparse_n11_example(seed=42)  # a sparse SPD matrix
    >>> L, D, P = sp.linalg.ldl(Q.toarray())  # dense LDL' for this example
    >>> factor = covmats.SparseCholeskyFactor(
    ...     sp.sparse.csc_array(L), sp.sparse.csc_array(D), P
    ... )
    >>> x = np.ones(Q.shape[0])
    >>> np.allclose(factor.matvec(x), Q @ x)
    True
    """

    __slots__ = ["_P", "_D", "_L", "_n_pts"]

    def __init__(
        self, L: sp.sparse.sparray, D: sp.sparse.sparray, P: ArrayLike
    ) -> None:
        r"""
        Initialize the instance.

        Parameters
        ----------
        L : sp.sparse.sparray
            Sparse lower triangle :math:`\mathbf{L}` of
            the :ref:`factorization <factref>`.
        D : sp.sparse.sparray
            Sparse diagonal matrix $\mathbf{{D}}$ of the :ref:`factorization <factref>`.
        P : ArrayLike
            1D vector $\mathbf{{P}}$ storing rows permutations of the
            :ref:`factorization <factref>`.

        Raises
        ------
        ValueError
            If the dimensions of `L`, `D` and `P` do not match.
        """
        # Make sure that is it 1D and integer
        self.P = np.asarray(P, dtype=np.int64).ravel()
        self.D = D
        self.L = L

        # Assert shapes are ok.
        if self.D.shape != self.shape or self.P.size != self.n_pts:
            raise ValueError(
                f"`L` has shape {self.shape},"
                f" `D` has shape {D.shape}, and `L` has shape {self.P.shape}!\n"
                f"To be consistent, `D` and `P` are expected with shape {self.shape}"
                f" and ({self.n_pts},) respectively."
            )

        super().__init__("d", shape=(self.n_pts, self.n_pts))

    @staticmethod
    def _validate_lower_triangle(A: sp.sparse.sparray, name: str) -> sp.sparse.sparray:
        """Assert that the input matrix A is lower triangular."""
        try:
            assert_allclose_sparse(A, sp.sparse.tril(A))
        except AssertionError as e:
            message = f"The input `{name}` must be a lower-triangular sparse array."
            raise ValueError(message) from e
        return A

    @property
    def L(self) -> sp.sparse.csc_array:
        r"""
        Sparse lower triangle :math:`\mathbf{L}` with unit diagonal of
        the :ref:`factorization <factref>`.
        """
        return self._L

    @L.setter
    def L(self, L: sp.sparse.csc_array) -> None:
        r"""
        Set the sparse lower triangle of
        the :ref:`factorization <factref>`
        """
        self._L = self._validate_lower_triangle(L, "L")
        # Extract the diagonal
        try:
            np.testing.assert_allclose(L.diagonal(), np.ones(L.shape[0]))
        except AssertionError as e:
            raise ValueError("The diagonal elements of `L` are assumed to be 1!") from e

    @property
    def D(self) -> sp.sparse.csc_array:
        r"""
        Sparse diagonal matrix :math:`\mathbf{D}` of
        the :ref:`factorization <factref>`.
        """
        return self._D

    @D.setter
    def D(self, D: sp.sparse.csc_array) -> None:
        r"""
        Set the sparse diagonal matrix :math:`\mathbf{D}` of
        the :ref:`factorization <factref>`.
        """
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

    @cached_property
    def invD(self) -> sp.sparse.csc_array:
        r"""
        Inverse of :math:`\mathbf{D}` in
        the :ref:`factorization <factref>`.

        Notes
        -----
        The property is read-only and updated when setting `D`.
        """
        return sp.sparse.diags(1.0 / self.D.diagonal())

    @cached_property
    def sqrtinvD(self) -> sp.sparse.csc_array:
        r"""
        Squared root inverse of :math:`\mathbf{D}` in
        the :ref:`factorization <factref>`.

        Notes
        -----
        The property is read-only and updated when setting `D`.
        """
        return np.sqrt(self.invD)

    @cached_property
    def sqrtD(self) -> sp.sparse.csc_array:
        r"""
        Squared root of :math:`\mathbf{D}` in
        the :ref:`factorization <factref>`.

        Notes
        -----
        The property is read-only and updated when setting `D`.
        """
        return np.sqrt(self.D)

    @property
    def P(self) -> NDArrayInt:
        r"""
        1D vector :math:`\mathbf{P}` storing rows permutations of the
        :ref:`factorization <factref>`.
        """
        return self._P

    @P.setter
    def P(self, P: ArrayLike) -> None:
        r"""
        Set the 1D vector :math:`\mathbf{P}` storing rows permutations of the
        :ref:`factorization <factref>`.
        """
        self._P = np.asarray(P, dtype=np.int64).ravel()

    @cached_property
    def Pt(self) -> NDArrayInt:
        r"""
        1D vector :math:`\mathbf{P}` storing rows back-permutations of the
        :ref:`factorization <factref>`.
        """
        return np.argsort(self._P)

    @overload
    def apply_P(self, x: NDArrayFloat) -> NDArrayFloat: ...

    @overload
    def apply_P(self, x: sp.sparse.csc_array) -> sp.sparse.csc_array: ...

    def apply_P(self, x: T) -> T:
        """Apply the permutation of rows."""
        return x[self.P]

    @overload
    def apply_Pt(self, x: NDArrayFloat) -> NDArrayFloat: ...

    @overload
    def apply_Pt(self, x: sp.sparse.csc_array) -> sp.sparse.csc_array: ...

    def apply_Pt(self, x: T) -> T:
        """Apply the reverse permutation of rows."""
        return x[self.Pt]

    def solve(self, b: NDArrayFloat) -> NDArrayFloat:
        r"""
        Return :math:`\mathbf{x} = \mathbf{A}^{-1} \mathbf{b}.`

        # L @ D @ L' = P @ A @ P'
        # A x = b
        # P' L @ D @ L' P x = b
        # x = P' L^-T @ D^{-1} @ L^{-1} P b

        Parameters
        ----------
        b: NDArrayFloat
            Column vector with shape (n,) or ensemble matrix with shape (n, ne).

        Returns
        -------
        NDArrayFloat
            :math:`\mathbf{x}`, with the same shape as `b`.
        """
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

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return A @ x."""
        L = self.apply_Pt(self.L)
        return L @ (self.D @ (L.T @ x))

    @property
    def log_pdet(self) -> float:
        """
        Log of the pseudo-determinant of the covariance matrix.

        Note
        ----
        In linear algebra and statistics, the pseudo-determinant [1] is the product of
        all non-zero eigenvalues of a square matrix. It coincides with the regular
        determinant when the matrix is non-singular.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.log(np.prod(self.D.diagonal()))

    @property
    def n_pts(self) -> int:
        """Number of points in the domain (n). This is read-only."""
        return self.L.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Shape of the square matrix (n, n). Read-only.
        """
        return self.L.shape[:2]

    @shape.setter
    def shape(self, value: Tuple[int, int]) -> None:
        pass

    @property
    def mat(self) -> sp.sparse.csc_array:
        r"""
        Return :math:`\mathbf{A}` as a dense array.

        Note
        ----
        Alias for :py:meth:`SparseCholeskyFactor.todense()`. This is read-only.
        """
        # Permute and rebuild
        L = self.apply_Pt(self.L)
        return L @ self.D @ L.T

    def todense(self) -> NDArrayFloat:
        r"""Return :math:`\mathbf{A}` as a dense array."""
        # Permute and rebuild
        return self.mat.toarray()

    def inv(self) -> NDArrayFloat:
        r"""Return the inverse :math:`\mathbf{A}^{-1}` as a dense array."""
        return self.solve(np.eye(self.n_pts))

    def get_diagonal(self) -> NDArrayFloat:
        r"""
        Return the diagonal entries of the matrix :math:`\mathbf{A}`.

        It as shape (n,).
        """
        return (self.L @ self.D @ self.L.T).diagonal()

    def get_invdiagonal(self) -> NDArrayFloat:
        r"""
        Return the diagonal entries of the inverse matrix :math:`\mathbf{A}^{-1}`.

        It as shape (n,).
        """
        cov_mat_diag = np.zeros(self.n_pts)
        v = np.zeros(self.n_pts)
        for i in range(self.n_pts):
            v[i - 1] = 0.0
            v[i] = 1.0
            cov_mat_diag[i] = self.solve(v)[i]
        return cov_mat_diag

    def colorize(self, x: NDArrayFloat) -> NDArrayFloat:
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
        :cite:p:`wikipediaWhiteningTransformation2025,novakGeneralizationColoringLinear2019`.

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

        Note
        ----
        We want to solve z.T = x @ K.T, where A = K @ K^{T}
        We use the cholesky factorization LDL' = PA'AP'
        with P' = P^{-1} the permutation that makes the decomposition unique.
        So LD^{1/2} = PA' and A = D^{1/2}L'P
        Finally z = (P' L D^{1/2} x')'

        References
        ----------
        .. bibliography::
            :list: enumerated
            :filter: False

            wikipediaWhiteningTransformation2025
            novakGeneralizationColoringLinear2019

        Examples
        --------
        >>> import numpy as np
        >>> import scipy as sp
        >>> import covmats
        >>> rng = np.random.default_rng(0)
        >>> n = 4
        >>> B = rng.random(size=(n, n))
        >>> cov_array = B @ B.T + n * np.eye(n)  # SPD covariance matrix
        >>> L, D, P = sp.linalg.ldl(cov_array)  # dense LDL' for this example
        >>> scf = covmats.SparseCholeskyFactor(
        ...     sp.sparse.csc_array(L), sp.sparse.csc_array(D), P
        ... )
        >>> x = rng.multivariate_normal(np.zeros(n), np.eye(n), size=10000)
        >>> x_ = scf.colorize(x)
        >>> cov_data = np.cov(x_, rowvar=False)
        >>> np.allclose(cov_data, cov_array, atol=0.15)
        True
        """
        return self.apply_Pt(self.L @ self.sqrtD @ x.T).T

    def colorize_inv(self, x: NDArrayFloat) -> NDArrayFloat:
        r"""
        Perform a colorizing transformation on data as in
        :py:meth:`SparseCholeskyFactor:colorize` but for the inverse
        :math:`\mathbf{A}^{-1}`.
        """
        return self.apply_Pt(
            sp.sparse.linalg.spsolve_triangular(
                self.L.T, self.sqrtinvD @ x.T, unit_diagonal=True, lower=False
            )
        ).T

    def whiten(self, x: NDArrayFloat) -> NDArrayFloat:
        """
        Perform a whitening transformation on data.

        "Whitening" ("white" as in "white noise", in which each frequency has
        equal magnitude) transforms a set of random variables into a new set of
        random variables with unit-diagonal covariance. When a whitening
        transform is applied to a sample of points distributed according to
        a multivariate normal distribution with zero mean, the covariance of
        the transformed sample is approximately the identity matrix
        :cite:p:`wikipediaWhiteningTransformation2025,
        novakGeneralizationColoringLinear2019`.

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

        Note
        ----
        We use the cholesky factorization LDL' = PA'AP'
        We want to solve x = z @ A.T =>  z = x A^{-T}
        with P' = P^{-1} the permutation that makes the decomposition unique.
        So LD^{1/2} = PA' and A = D^{1/2}L'P
        Finally z = (D^{-1/2} L^{-1} P x')'

        References
        ----------
        .. bibliography::
            :list: enumerated
            :filter: False

            novakGeneralizationColoringLinear2019
            wikipediaWhiteningTransformation2025

        Examples
        --------
        >>> import numpy as np
        >>> import scipy as sp
        >>> import covmats
        >>> rng = np.random.default_rng(0)
        >>> n = 4
        >>> B = rng.random(size=(n, n))
        >>> cov_array = B @ B.T + n * np.eye(n)  # SPD covariance matrix
        >>> L, D, P = sp.linalg.ldl(cov_array)  # dense LDL' for this example
        >>> scf = covmats.SparseCholeskyFactor(
        ...     sp.sparse.csc_array(L), sp.sparse.csc_array(D), P
        ... )
        >>> x = rng.multivariate_normal(np.zeros(n), cov_array, size=10000)
        >>> x_ = scf.whiten(x)
        >>> np.round(np.cov(x_, rowvar=False), 2)  # near-identity covariance
        array([[ 1.01, -0.  , -0.02, -0.  ],
               [-0.  ,  1.02,  0.  ,  0.  ],
               [-0.02,  0.  ,  1.  ,  0.  ],
               [-0.  ,  0.  ,  0.  ,  0.99]])

        """
        return (
            self.sqrtinvD
            @ sp.sparse.linalg.spsolve_triangular(
                self.L, self.apply_P(x.T), unit_diagonal=True, lower=True
            )
        ).T

    def whiten_inv(self, x: NDArrayFloat) -> NDArrayFloat:
        r"""
        Perform a whitening transformation on data as in
        :py:meth:`SparseCholeskyFactor:whiten` but for the inverse
        :math:`\mathbf{A}^{-1}`.
        """
        return self.sqrtD @ (self.L.T @ self.apply_P(x.T)).T


def assert_allclose_sparse(
    A: sp.sparse.sparray, B: sp.sparse.sparray, atol: float = 1e-8, rtol: float = 1e-8
) -> None:
    """
    Assert that two sparse matrices or arrays are almost equal.

    Parameters
    ----------
    A : sp.sparse.sparray
        First sparse matrix.
    B : sp.sparse.sparray
        Second sparse matrix.
    atol : float, optional
        Absolute tolerance for the equality test, by default 1e-8.
    rtol : float, optional
        Relative tolerance for the equality test, by default 1e-8.
    """
    # If you want to check matrix shapes as well
    assert np.array_equal(A.shape, B.shape)
    r1, c1, v1 = sp.sparse.find(A)
    r2, c2, v2 = sp.sparse.find(B)
    np.testing.assert_equal(r1, r2)
    np.testing.assert_equal(c1, c2)
    np.testing.assert_allclose(v1, v2, atol=atol, rtol=rtol)


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
