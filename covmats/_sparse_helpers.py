# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Wrap the sparse cholesky factorization from"""

from functools import cached_property
from typing import Literal, Tuple, TypeVar, Union, overload

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
        self.P = P
        self.D = D
        self.L = L

        # Assert shapes are ok.
        if self.D.shape != self.shape or self.P.size != self.n_pts:
            raise ValueError(
                f"`L` has shape {self.shape},"
                f" `D` has shape {D.shape}, and `P` has shape {self.P.shape}!\n"
                f"To be consistent, `D` and `P` are expected with shape {self.shape}"
                f" and ({self.n_pts},) respectively."
            )

        # Also validate that P is a permutation (needs n_pts, hence done here)
        if not np.array_equal(np.sort(self.P), np.arange(self.n_pts)):
            raise ValueError("`P` must be a permutation of `range(n)`!")

        super().__init__("d", shape=(self.n_pts, self.n_pts))

    # Cached quantities (stored in the instance ``__dict__``) that must be dropped
    # when the attribute they derive from is set again.
    _CACHE_FROM_D = (
        "invD",
        "sqrtD",
        "sqrtinvD",
        "_d",
        "_inv_d",
        "_sqrt_d",
        "_inv_sqrt_d",
    )
    _CACHE_FROM_L = ("_L_csr", "_Lt_csr")
    _CACHE_FROM_P = ("Pt",)

    def _drop_cache(self, names: Tuple[str, ...]) -> None:
        """Forget cached values so that they are recomputed on next access."""
        for name in names:
            self.__dict__.pop(name, None)

    @staticmethod
    def _validate_lower_triangle(A: sp.sparse.sparray, name: str) -> sp.sparse.sparray:
        """Assert that the input matrix A is lower triangular."""
        if sp.sparse.triu(A, k=1).count_nonzero() != 0:
            message = f"The input `{name}` must be a lower-triangular sparse array."
            raise ValueError(message)
        return A

    @staticmethod
    def _scale_rows(v: NDArrayFloat, x: NDArrayFloat) -> NDArrayFloat:
        """Return ``diag(v) @ x`` for ``x`` with shape (n,) or (n, m)."""
        return v * x if x.ndim == 1 else v[:, None] * x

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
        # Validate first, assign afterwards: a failed set leaves the instance intact.
        self._validate_lower_triangle(L, "L")
        if not np.allclose(L.diagonal(), 1.0, rtol=1e-7, atol=0.0):
            raise ValueError("The diagonal elements of `L` are assumed to be 1!")
        self._L = L
        self._drop_cache(self._CACHE_FROM_L)

    @cached_property
    def _L_csr(self) -> sp.sparse.csr_array:
        """CSR copy of L, required by ``spsolve_triangular`` (converted once)."""
        return sp.sparse.csr_array(self._L)

    @cached_property
    def _Lt_csr(self) -> sp.sparse.csr_array:
        """CSR copy of L^T, required by ``spsolve_triangular`` (converted once)."""
        return sp.sparse.csr_array(self._L.T)

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
        self._drop_cache(self._CACHE_FROM_D)

    @cached_property
    def _d(self) -> NDArrayFloat:
        """Diagonal of D as a 1D vector."""
        return np.asarray(self._D.diagonal(), dtype=np.float64)

    @cached_property
    def _inv_d(self) -> NDArrayFloat:
        return 1.0 / self._d

    @cached_property
    def _sqrt_d(self) -> NDArrayFloat:
        return np.sqrt(self._d)

    @cached_property
    def _inv_sqrt_d(self) -> NDArrayFloat:
        return 1.0 / self._sqrt_d

    @cached_property
    def invD(self) -> sp.sparse.dia_array:
        r"""
        Inverse of :math:`\mathbf{D}` in
        the :ref:`factorization <factref>`.

        Notes
        -----
        The property is read-only and updated when setting `D`.
        """
        return sp.sparse.diags_array(self._inv_d)

    @cached_property
    def sqrtinvD(self) -> sp.sparse.dia_array:
        r"""
        Squared root inverse of :math:`\mathbf{D}` in
        the :ref:`factorization <factref>`.

        Notes
        -----
        The property is read-only and updated when setting `D`.
        """
        return sp.sparse.diags_array(self._inv_sqrt_d)

    @cached_property
    def sqrtD(self) -> sp.sparse.dia_array:
        r"""
        Squared root of :math:`\mathbf{D}` in
        the :ref:`factorization <factref>`.

        Notes
        -----
        The property is read-only and updated when setting `D`.
        """
        return sp.sparse.diags_array(self._sqrt_d)

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
        self._drop_cache(self._CACHE_FROM_P)

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
        y = sp.sparse.linalg.spsolve_triangular(
            self._L_csr, self.apply_P(b), lower=True, unit_diagonal=True
        )
        return self.apply_Pt(
            sp.sparse.linalg.spsolve_triangular(
                self._Lt_csr,
                self._scale_rows(self._inv_d, y),
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
        # Sum of logs: the product of the pivots overflows/underflows for large n.
        return float(np.sum(np.log(self._d)))

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

        Note
        ----
        :math:`\mathbf{A} = \mathbf{P}^{T}\mathbf{L}\mathbf{D}\mathbf{L}^{T}
        \mathbf{P}`, so that :math:`A_{ii} = \sum_k L_{ik}^2 D_{kk}` is computed in
        O(nnz(L)) without forming the product, and then un-permuted.
        """
        return self.apply_Pt(self.L.power(2) @ self._d)

    def get_invdiagonal(self, block_size: int = 256) -> NDArrayFloat:
        r"""
        Return the diagonal entries of the inverse matrix :math:`\mathbf{A}^{-1}`.

        It as shape (n,).

        Parameters
        ----------
        block_size : int, optional
            Number of unit vectors processed per (multi right-hand side) triangular
            solve. It bounds the memory to ``n * block_size`` floats. By default 256.

        Note
        ----
        :math:`(A^{-1})_{ii} = \lVert \mathbf{D}^{-1/2}\mathbf{L}^{-1}\mathbf{P}
        \mathbf{e}_i \rVert^2`: a single forward solve per unit vector is needed
        (instead of a full forward + backward :py:meth:`solve`).
        """
        n = self.n_pts
        out = np.empty(n)
        for start in range(0, n, block_size):
            idx = np.arange(start, min(start + block_size, n))
            E = np.zeros((n, idx.size))
            E[idx, np.arange(idx.size)] = 1.0
            Z = sp.sparse.linalg.spsolve_triangular(
                self._L_csr, self.apply_P(E), lower=True, unit_diagonal=True
            )
            out[idx] = np.einsum("ij,ij,i->j", Z, Z, self._inv_d)
        return out

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

    def colorize_adjoint(self, y: NDArrayFloat) -> NDArrayFloat:
        r"""
        Perform the adjoint (transpose) of :py:meth:`SparseCholeskyFactor.colorize`.

        Parameters
        ----------
        y : array_like
            An array of points in data space, with the last dimension equal
            to :py:attr:`n_pts`.

        Returns
        -------
        y_ : array_like
            The transformed array, same shape as `y`.

        Note
        ----
        Writing :py:meth:`colorize` as the linear map
        :math:`\mathrm{colorize}(x) = \mathbf{P}^{T} \mathbf{L} \mathbf{D}^{1/2} x`
        (with :math:`\mathbf{P}` the permutation matrix such that
        :py:meth:`apply_P` computes :math:`\mathbf{P} x`, so that
        :py:meth:`apply_Pt` computes :math:`\mathbf{P}^{T} x`), its adjoint is
        :math:`\mathrm{colorize}^{T}(y) = \mathbf{D}^{1/2} \mathbf{L}^{T}
        \mathbf{P} y`, obtained here as
        ``sqrtD @ (L.T @ apply_P(y))`` -- no triangular solve needed, unlike
        :py:meth:`colorize_inv_adjoint`.

        Examples
        --------
        >>> import numpy as np
        >>> import scipy as sp
        >>> import covmats
        >>> rng = np.random.default_rng(0)
        >>> n = 4
        >>> B = rng.random(size=(n, n))
        >>> cov_array = B @ B.T + n * np.eye(n)
        >>> L, D, P = sp.linalg.ldl(cov_array)
        >>> scf = covmats.SparseCholeskyFactor(
        ...     sp.sparse.csc_array(L), sp.sparse.csc_array(D), P
        ... )
        >>> x = rng.standard_normal(n)
        >>> y = rng.standard_normal(n)
        >>> np.allclose(
        ...     scf.colorize(x[None, :])[0] @ y, x @ scf.colorize_adjoint(y[None, :])[0]
        ... )
        True
        """
        return (self.sqrtD @ (self.L.T @ self.apply_P(y.T))).T

    def colorize_inv(self, x: NDArrayFloat) -> NDArrayFloat:
        r"""
        Perform a colorizing transformation on data as in
        :py:meth:`SparseCholeskyFactor:colorize` but for the inverse
        :math:`\mathbf{A}^{-1}`.
        """
        return self.apply_Pt(
            sp.sparse.linalg.spsolve_triangular(
                self._Lt_csr, self.sqrtinvD @ x.T, unit_diagonal=True, lower=False
            )
        ).T

    def colorize_inv_adjoint(self, y: NDArrayFloat) -> NDArrayFloat:
        r"""
        Perform the adjoint (transpose) of :py:meth:`SparseCholeskyFactor.colorize_inv`.

        Parameters
        ----------
        y : array_like
            An array of points in data space, with the last dimension equal
            to :py:attr:`n_pts`.

        Returns
        -------
        y_ : array_like
            The transformed array, same shape as `y`.

        Note
        ----
        Writing :py:meth:`colorize_inv` as the linear map
        :math:`\mathrm{colorize\_inv}(x) = \mathbf{P}^{T} \mathbf{L}^{-T}
        \mathbf{D}^{-1/2} x` (:py:meth:`colorize_inv` solves
        :math:`\mathbf{L}^{T} z = \mathbf{D}^{-1/2} x` then applies
        :math:`\mathbf{P}^{T}`), its adjoint is
        :math:`\mathrm{colorize\_inv}^{T}(y) = \mathbf{D}^{-1/2}
        \mathbf{L}^{-1} \mathbf{P} y`, obtained here by forward-solving
        :math:`\mathbf{L} z = \mathbf{P} y` (lower-triangular, unlike the
        backward solve used by :py:meth:`colorize_inv` itself) and then
        applying ``sqrtinvD``.

        Examples
        --------
        >>> import numpy as np
        >>> import scipy as sp
        >>> import covmats
        >>> rng = np.random.default_rng(0)
        >>> n = 4
        >>> B = rng.random(size=(n, n))
        >>> cov_array = B @ B.T + n * np.eye(n)
        >>> L, D, P = sp.linalg.ldl(cov_array)
        >>> scf = covmats.SparseCholeskyFactor(
        ...     sp.sparse.csc_array(L), sp.sparse.csc_array(D), P
        ... )
        >>> x = rng.standard_normal(n)
        >>> y = rng.standard_normal(n)
        >>> np.allclose(
        ...     scf.colorize_inv(x[None, :])[0] @ y,
        ...     x @ scf.colorize_inv_adjoint(y[None, :])[0],
        ... )
        True
        """
        z = sp.sparse.linalg.spsolve_triangular(
            self._L_csr, self.apply_P(y.T), unit_diagonal=True, lower=True
        )
        return (self.sqrtinvD @ z).T

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
                self._L_csr, self.apply_P(x.T), unit_diagonal=True, lower=True
            )
        ).T

    def whiten_inv(self, x: NDArrayFloat) -> NDArrayFloat:
        r"""
        Perform a whitening transformation on data as in
        :py:meth:`SparseCholeskyFactor:whiten` but for the inverse
        :math:`\mathbf{A}^{-1}`.
        """
        return (self.sqrtD @ (self.L.T @ self.apply_P(x.T))).T


_Ordering = Union[Literal["amd", "rcm", "natural"], NDArrayInt]


def sparse_cholesky_superlu(
    A: sp.sparse.sparray,
    ordering: _Ordering = "amd",
    check: bool = True,
) -> SparseCholeskyFactor:
    r"""
    Sparse :math:`\mathbf{LDL}^{\mathrm{T}} = \mathbf{PAP}^{\mathrm{T}}` factorization.

    SuperLU (shipped with SciPy, BSD) is a general LU solver. For a symmetric
    positive definite matrix, restricting it to diagonal pivots and a symmetric
    ordering gives :math:`\mathbf{PAP}^{\mathrm{T}} = \mathbf{L}\mathbf{U}` with
    :math:`\mathbf{L}` unit lower triangular and
    :math:`\mathbf{U} = \mathbf{D}\mathbf{L}^{\mathrm{T}}`, which is exactly
    the representation expected by :py:class:`SparseCholeskyFactor`.

    Parameters
    ----------
    A : sp.sparse.sparray
        Symmetric positive definite matrix, shape (n, n). Only symmetry is
        assumed, not checked (see `check`).
    ordering : {"amd", "rcm", "natural"} or array_like of int, optional
        Fill-reducing symmetric ordering. ``"amd"`` uses SuperLU's minimum degree
        on :math:`A + A^T` (default), ``"rcm"`` uses reverse Cuthill-McKee
        (good for banded / grid-like matrices), ``"natural"`` keeps the input
        order. An integer vector is used as the permutation ``p`` directly
        (``A[p][:, p]`` is factorized).
    check : bool, optional
        If True, verify that SuperLU did not pivot off the diagonal and that all
        pivots are strictly positive (SPD). By default True.

    Returns
    -------
    SparseCholeskyFactor

    Raises
    ------
    ValueError
        If `A` is not square, is not positive definite, or the ordering is
        invalid.
    """
    A = sp.sparse.csc_array(A)
    n = A.shape[0]
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError(f"`A` must be square, got shape {A.shape}.")

    # 1) Symmetric pre-ordering p (outer permutation)
    if isinstance(ordering, str):
        if ordering == "rcm":
            p = sp.sparse.csgraph.reverse_cuthill_mckee(
                A.tocsr(), symmetric_mode=True
            ).astype(np.int64)
            permc = "NATURAL"
        elif ordering == "natural":
            p, permc = np.arange(n, dtype=np.int64), "NATURAL"
        elif ordering == "amd":
            p, permc = None, "MMD_AT_PLUS_A"
        else:
            raise ValueError(f"Unknown ordering {ordering!r}.")
    else:
        p = np.asarray(ordering, dtype=np.int64).ravel()
        if not np.array_equal(np.sort(p), np.arange(n)):
            raise ValueError("`ordering` must be a permutation of range(n).")
        permc = "NATURAL"

    Ap = A if p is None else A[p][:, p].tocsc()

    # 2) SuperLU, forced to behave symmetrically:
    #  - diag_pivot_thresh=0 : always pivot on the diagonal (no row swaps),
    #  - SymmetricMode       : symmetric structure heuristics,
    #  - Equil=False         : safeguard; row/column equilibration could scale
    #                          the two sides differently and break L U = L D L'.
    try:
        lu = sp.sparse.linalg.splu(
            Ap,
            permc_spec=permc,
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True, "Equil": False},
        )
    except RuntimeError as e:  # exactly singular
        raise ValueError("`A` is singular or not positive definite.") from e

    # 3) Combine permutations. SuperLU: Pr @ Ap @ Pc = L @ U with
    #    Ap[argsort(perm_r)][:, argsort(perm_c)] = L U.
    q_r, q_c = np.argsort(lu.perm_r), np.argsort(lu.perm_c)
    if check and not np.array_equal(q_r, q_c):
        raise ValueError(
            "SuperLU pivoted off the diagonal: `A` is not SPD."
        )  # pragma: no cover
    P = q_c if p is None else p[q_c]

    d = lu.U.diagonal()
    if check and not np.all(d > 0.0):
        raise ValueError(
            f"{np.count_nonzero(d <= 0.0)} non-positive pivot(s): `A` is not SPD."
        )

    L = sp.sparse.csc_array(lu.L)
    D = sp.sparse.dia_array((d[None, :], [0]), shape=(n, n)).tocsc()
    return SparseCholeskyFactor(L, D, P)


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
    if tuple(A.shape) != tuple(B.shape):
        raise AssertionError(f"Shape mismatch: {A.shape} != {B.shape}")
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


def get_L_D_P(A: sp.sparse.sparray):
    """
    Return L, D and P from the factorization L @ D @ L' = P @ A @ P' using
    ``sparse_cholesky_superlu``.
    """
    scf = sparse_cholesky_superlu(A, check=False)
    return scf.L, scf.D, scf.P


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
    diag_mean : float, optional
        Mean of the diagonal entries (variances), by default 10.0.

    Returns
    -------
    sp.sparse.csc_array
        CSC sparse SPD.

    Note
    ----
    Since v0.6.0 the `seed` is honoured (it was previously ignored in favor of a
    hard-coded seed of 42) and the random generator is local (the global numpy
    state is no longer modified).
    """
    rng = np.random.default_rng(seed)
    variances = rng.uniform(-1.5, 1.5, n) + diag_mean

    # ~10% of the strictly upper triangle is filled with small covariances
    rows, cols = np.triu_indices(n, k=1)
    keep = rng.random(rows.size) < 0.1
    rows, cols = rows[keep], cols[keep]
    covs = rng.uniform(-0.5, 0.5, rows.size)

    A = sp.sparse.coo_array(
        (
            np.concatenate([variances, covs, covs]),
            (
                np.concatenate([np.arange(n), rows, cols]),
                np.concatenate([np.arange(n), cols, rows]),
            ),
        ),
        shape=(n, n),
    )
    return A.tocsc()
