# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""
Provide conditional (Gaussian) simulation on top of a :py:class:`CovarianceMatrix`.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.random import Generator, RandomState
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg

from covmats._covariances import CovarianceMatrix, CovViaDiagonal
from covmats._helpers import check_random_state
from covmats._types import ArrayLike, NDArrayFloat, NDArrayInt

_SOLVERS = ("cg", "direct")


def make_point_observation_operator(indices: ArrayLike, n: int) -> LinearOperator:
    """
    Build a point-picking observation operator ``H`` with shape ``(n_obs, n)``.

    Such that ``(H z)[i] = z[indices[i]]``. This is the common case where
    measurements are made directly at (a subset of) the field's
    discretization nodes/points.

    Parameters
    ----------
    indices : ArrayLike
        Indices, in ``[0, n)``, of the observed entries of the field.
        Repeated indices are supported (both in the forward and adjoint
        directions).
    n : int
        Size of the field (number of rows/columns of the covariance matrix).

    Returns
    -------
    LinearOperator
        A matrix-free linear operator with shape ``(n_obs, n)``, ``n_obs``
        being the number of indices.

    Raises
    ------
    ValueError
        If ``indices`` is empty, or if some index falls outside ``[0, n)``.

    Examples
    --------
    >>> import numpy as np
    >>> from covmats import make_point_observation_operator
    >>> H = make_point_observation_operator([0, 2], n=4)
    >>> H.matvec(np.array([10.0, 20.0, 30.0, 40.0]))
    array([10., 30.])
    >>> H.rmatvec(np.array([1.0, 2.0]))
    array([1., 0., 2., 0.])
    """
    _indices: NDArrayInt = np.asarray(indices, dtype=np.int64).ravel()
    if _indices.size == 0:
        raise ValueError("`indices` must contain at least one element!")
    if np.any(_indices < 0) or np.any(_indices >= n):
        raise ValueError(
            f"All `indices` must be in [0, {n}), got min={_indices.min()}, "
            f"max={_indices.max()}!"
        )
    n_obs = _indices.size

    def matvec(z: NDArrayFloat) -> NDArrayFloat:
        return np.asarray(z)[_indices]

    def rmatvec(y: NDArrayFloat) -> NDArrayFloat:
        out = np.zeros(n)
        np.add.at(out, _indices, y)
        return out

    def matmat(z: NDArrayFloat) -> NDArrayFloat:
        return np.asarray(z)[_indices, :]

    def rmatmat(y: NDArrayFloat) -> NDArrayFloat:
        out = np.zeros((n, y.shape[1]))
        np.add.at(out, _indices, y)
        return out

    return LinearOperator(
        shape=(n_obs, n),
        matvec=matvec,  # ty: ignore[unknown-argument]
        rmatvec=rmatvec,  # ty: ignore[unknown-argument]
        matmat=matmat,  # ty: ignore[unknown-argument]
        rmatmat=rmatmat,  # ty: ignore[unknown-argument]
        dtype=np.float64,
    )


def _as_linear_operator(
    obs_op: Union[LinearOperator, ArrayLike], n: int
) -> LinearOperator:
    """Wrap a dense/sparse array-like observation operator as a LinearOperator."""
    if isinstance(obs_op, LinearOperator):
        return obs_op
    mat = np.asarray(obs_op, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] != n:
        raise ValueError(
            f"`obs_op` must be a LinearOperator or a 2D array-like with "
            f"{n} columns, got shape {mat.shape}!"
        )
    return LinearOperator(
        shape=mat.shape,
        matvec=lambda x: mat @ x,  # ty: ignore[unknown-argument]
        rmatvec=lambda y: mat.T @ y,  # ty: ignore[unknown-argument]
        matmat=lambda x: mat @ x,  # ty: ignore[unknown-argument]
        rmatmat=lambda y: mat.T @ y,  # ty: ignore[unknown-argument]
        dtype=np.float64,
    )


def _as_covariance(
    obs_cov: Union[CovarianceMatrix, float, ArrayLike], n_obs: int
) -> CovarianceMatrix:
    """Wrap a scalar/array of independent variances into a CovViaDiagonal."""
    if isinstance(obs_cov, CovarianceMatrix):
        return obs_cov
    variances = np.broadcast_to(np.asarray(obs_cov, dtype=np.float64), (n_obs,))
    return CovViaDiagonal(variances)


def _get_colorizer(
    cov: CovarianceMatrix,
    n: int,
    colorize_fn: Optional[Callable[[NDArrayFloat], NDArrayFloat]],
    colorize_dim: Optional[int],
) -> Tuple[Callable[[NDArrayFloat], NDArrayFloat], int]:
    """Resolve the (colorizer, white-noise dimension) pair for unconditional draws."""
    colorize = (
        colorize_fn if colorize_fn is not None else getattr(cov, "colorize", None)
    )
    if colorize is None:
        raise ValueError(
            "`cov` has no `colorize` method (this is expected for matrix-free "
            "operators such as CovKernelAsLinop / CovKernelAsLinopViaFFT, which "
            "provide no cheap direct sampler). Pass `colorize_fn` explicitly, "
            "e.g. built from `eigen_factorize_cov_mat(cov, ...).colorize`."
        )
    if colorize_dim is not None:
        return colorize, colorize_dim
    if colorize_fn is None:
        return colorize, getattr(cov, "subspace_size", n)
    return colorize, n


def _prepare(
    cov: CovarianceMatrix,
    obs_op: Union[LinearOperator, ArrayLike],
    obs_values: ArrayLike,
    obs_cov: Union[CovarianceMatrix, float, ArrayLike],
    mean: Union[float, ArrayLike],
    solver: str,
) -> Tuple[int, int, LinearOperator, CovarianceMatrix, NDArrayFloat, NDArrayFloat]:
    """Validate inputs shared by conditional_simulate and conditional_mean."""
    if solver not in _SOLVERS:
        raise ValueError(f"`solver` must be one of {_SOLVERS}, got {solver!r}!")
    n = cov.shape[0]
    _obs_values: NDArrayFloat = np.asarray(obs_values, dtype=np.float64)
    if _obs_values.ndim != 1:
        raise ValueError(f"`obs_values` must be 1D, got shape {_obs_values.shape}!")
    n_obs = _obs_values.shape[0]

    H = _as_linear_operator(obs_op, n)
    if H.shape != (n_obs, n):
        raise ValueError(
            f"`obs_op` has shape {H.shape}, expected ({n_obs}, {n}) to match "
            "`obs_values` and `cov`!"
        )
    R = _as_covariance(obs_cov, n_obs)
    mean_vec: NDArrayFloat = np.broadcast_to(np.asarray(mean, dtype=np.float64), (n,))
    return n, n_obs, H, R, mean_vec, _obs_values


def _build_dense_system(
    cov: CovarianceMatrix, H: LinearOperator, R: CovarianceMatrix, n_obs: int
) -> Tuple[NDArrayFloat, Tuple[NDArrayFloat, bool]]:
    """
    Assemble ``CHt = cov @ H^T`` and factor the dense system ``A = H CHt + R``.

    Costs O(n_obs) applications of ``cov.matmat``/``H``, independent of the
    number of realizations that will later be solved for.
    """
    Ht_dense = H.rmatmat(np.eye(n_obs))  # (n, n_obs)
    CHt = cov.matmat(Ht_dense)  # (n, n_obs)
    A_dense = H.matmat(CHt) + R.todense()  # (n_obs, n_obs)
    try:
        factor = cho_factor(A_dense)
    except LinAlgError as err:
        raise RuntimeError(
            "Cholesky factorization of `H cov H^T + R` failed (the assembled "
            "system is not numerically SPD); try `solver='cg'` instead, or add "
            "regularization to `obs_cov`."
        ) from err
    return CHt, factor


def conditional_simulate(
    cov: CovarianceMatrix,
    obs_op: Union[LinearOperator, ArrayLike],
    obs_values: ArrayLike,
    obs_cov: Union[CovarianceMatrix, float, ArrayLike],
    n_reals: int = 1,
    mean: Union[float, ArrayLike] = 0.0,
    random_state: Optional[Union[int, RandomState, Generator]] = None,
    solver: str = "cg",
    rtol: float = 1e-8,
    maxiter: Optional[int] = None,
    preconditioner: Optional[LinearOperator] = None,
    return_unconditional: bool = False,
    colorize_fn: Optional[Callable[[NDArrayFloat], NDArrayFloat]] = None,
    colorize_dim: Optional[int] = None,
) -> Union[NDArrayFloat, Tuple[NDArrayFloat, NDArrayFloat]]:
    r"""
    Draw exact conditional realizations of a Gaussian field given noisy data.

    The field prior is ``z ~ N(mean, cov)``. Observations are
    ``obs_values = obs_op @ z + eps``, ``eps ~ N(0, obs_cov)``. This uses
    Matheron's rule / pathwise conditioning
    :cite:p:`wilsonPathwiseConditioningGaussian2021`, so it is exact (up to
    the linear solve tolerance and Monte Carlo sampling error), and only
    ever needs forward operators (``matvec``/``matmat``/``colorize``) on
    ``cov``, never a factorization or dense assembly of the *posterior*
    covariance (the much smaller ``(n_obs, n_obs)`` *prior* data-space
    system may be assembled and factored when ``solver="direct"``, see
    below).

    Parameters
    ----------
    cov : CovarianceMatrix
        Prior covariance representation of the field, with shape ``(n, n)``.
        Only ``cov.matvec``/``cov.matmat`` are required for the
        conditioning step itself, so this works identically for dense
        representations (:py:class:`CovViaCholesky`, ...), low-rank ones
        (:py:class:`CovViaEnsemble`, :py:class:`CovViaEigenFactorization`)
        and matrix-free kernel operators (:py:class:`CovKernelAsLinop`,
        :py:class:`CovKernelAsLinopViaFFT`). Unconditional draws
        additionally require a colorizer: :py:class:`CovarianceMatrix`
        subclasses provide ``cov.colorize`` directly and it is used by
        default; :py:class:`CovKernelAsLinop` and
        :py:class:`CovKernelAsLinopViaFFT` are plain matrix-free linear
        operators with no cheap direct sampler, so for those pass
        `colorize_fn` explicitly (e.g. built from
        :py:func:`eigen_factorize_cov_mat`, whose result is a
        :py:class:`CovarianceMatrix` and does expose ``colorize``).
    obs_op : LinearOperator or ArrayLike, shape (n_obs, n)
        Observation/measurement operator ``H``. Use
        :py:func:`make_point_observation_operator` for plain point
        observations at grid/discretization nodes, or provide any other
        :py:class:`~scipy.sparse.linalg.LinearOperator` (e.g. for block
        averages, gradients, ...). A dense/sparse array-like is also
        accepted and wrapped automatically.
    obs_values : ArrayLike, shape (n_obs,)
        Observed data.
    obs_cov : CovarianceMatrix, float, or ArrayLike
        Observation-error covariance ``R``. A float or 1D array of length
        ``n_obs`` is treated as independent per-observation variances
        (:math:`\sigma_i^2`) and wrapped into a :py:class:`CovViaDiagonal`.
        Pass any other :py:class:`CovarianceMatrix` instance directly to
        represent correlated observation errors.
    n_reals : int, default 1
        Number of independent conditional realizations to draw.
    mean : float or ArrayLike, shape (n,), default 0.0
        Prior mean of the field.
    random_state : int, RandomState, Generator, or None
        Pseudorandom number generator state, forwarded to
        :py:func:`~scipy._lib._util.check_random_state`.
    solver : {"cg", "direct"}, default "cg"
        How the ``(H cov H^T + R) lam = r`` system is solved.

        - ``"cg"``: matrix-free, one conjugate-gradient solve per
          realization (:py:func:`scipy.sparse.linalg.cg`), using only
          ``cov.matvec``. Scales to arbitrarily large ``n_obs`` since the
          system is never assembled, at the cost of a Python loop over
          `n_reals`.
        - ``"direct"``: assembles ``H cov H^T + R`` and Cholesky-factors it
          *once* (:math:`O(n_{obs})` applications of ``cov.matmat``,
          independent of `n_reals`), then solves for *all* realizations in
          a single vectorized :py:func:`scipy.linalg.cho_solve` call -- no
          Python loop over realizations at all. Typically much faster
          whenever ``n_obs`` is small to moderate relative to what is
          affordable to factor densely (the common case in practice: far
          fewer observations than field points).
    rtol, maxiter, preconditioner :
        Only used when ``solver="cg"``, forwarded to
        :py:func:`scipy.sparse.linalg.cg` for the
        ``(H cov H^T + R) lam = r`` solve.
    return_unconditional : bool, default False
        If True, also return the unconditional draws (useful for QC, or to
        visualize the conditioning correction term in isolation).
    colorize_fn : callable(w) -> x, optional
        Overrides ``cov.colorize``. Must map a standard-normal white-noise
        array to a draw with covariance ``cov``, and must accept a batch of
        white-noise vectors stacked along the first axis (as
        ``cov.colorize`` does) when `n_reals` > 1 and ``solver="direct"``.
    colorize_dim : int, optional
        Dimension of the white-noise input expected by the colorizer. Equal
        to ``n`` for full-rank representations, but equal to
        ``subspace_size`` (the retained rank) for low-rank ones. Detected
        automatically from ``cov.subspace_size`` when available and
        `colorize_fn` is None; otherwise defaults to ``n``.

    Returns
    -------
    NDArrayFloat, shape (n_reals, n)
        Conditional realizations.
    NDArrayFloat, shape (n_reals, n), optional
        Unconditional realizations. Only returned if
        `return_unconditional` is True, as the second element of a tuple.

    Raises
    ------
    ValueError
        If `obs_values` is not 1D, if `obs_op` has an incompatible shape,
        if `solver` is not one of ``"cg"``/``"direct"``, or if `cov`
        provides no colorizer and `colorize_fn` is not supplied.
    RuntimeError
        If the linear solve does not converge (``solver="cg"``) or if the
        dense data-space system is not numerically SPD (``solver="direct"``).

    Examples
    --------
    >>> import numpy as np
    >>> import covmats
    >>> x = np.linspace(0, 10, 25)
    >>> C = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / 2.0**2)
    >>> cov = covmats.CovViaCholesky(np.linalg.cholesky(C + 1e-10 * np.eye(25)))
    >>> H = covmats.make_point_observation_operator([2, 12, 20], n=25)
    >>> d = np.array([0.5, -0.3, 0.8])
    >>> z = covmats.conditional_simulate(cov, H, d, obs_cov=0.01, n_reals=3,
    ... random_state=0)
    >>> z.shape
    (3, 25)
    >>> z_fast = covmats.conditional_simulate(cov, H, d, obs_cov=0.01, n_reals=3,
    ... random_state=0, solver="direct")
    >>> z_fast.shape
    (3, 25)
    """
    n, n_obs, H, R, mean_vec, _obs_values = _prepare(
        cov, obs_op, obs_values, obs_cov, mean, solver
    )
    colorize, colorize_dim = _get_colorizer(cov, n, colorize_fn, colorize_dim)
    rng = check_random_state(random_state)

    if solver == "direct":
        CHt, factor = _build_dense_system(cov, H, R, n_obs)

        W = rng.standard_normal(size=(n_reals, colorize_dim))
        Z_u = mean_vec + colorize(W)  # (n_reals, n)

        Eps_u = R.sample_mvnormal(shape=[n_reals], random_state=rng)  # (n_reals, n_obs)
        D_u = H.matmat(Z_u.T).T + Eps_u  # (n_reals, n_obs)

        Resid = (_obs_values[None, :] - D_u).T  # (n_obs, n_reals)
        Lam = cho_solve(factor, Resid)  # (n_obs, n_reals), all realizations at once

        z_cond = Z_u + (CHt @ Lam).T
        if return_unconditional:
            return z_cond, Z_u
        return z_cond

    def A_matvec(lam: NDArrayFloat) -> NDArrayFloat:
        return H.matvec(cov.matvec(H.rmatvec(lam))) + R.matvec(lam)

    A = LinearOperator(shape=(n_obs, n_obs), matvec=A_matvec, dtype=np.float64)  # ty: ignore[unknown-argument]

    z_cond = np.empty((n_reals, n))
    z_unc = np.empty((n_reals, n)) if return_unconditional else None

    for k in range(n_reals):
        # 1. unconditional draw
        w = rng.standard_normal(size=colorize_dim)
        z_u = mean_vec + colorize(w)

        # 2-3. synthetic data
        eps_u = R.sample_mvnormal(shape=[1], random_state=rng)[0]
        d_u = H.matvec(z_u) + eps_u

        # 4. residual
        r = _obs_values - d_u

        # 5. CG solve of (H cov H^T + R) lam = r
        lam, info = cg(A, r, rtol=rtol, maxiter=maxiter, M=preconditioner)
        if info != 0:
            raise RuntimeError(
                f"CG did not converge for realization {k} (info={info}); "
                "consider a preconditioner, a looser `rtol`, a larger "
                "`maxiter`, or `solver='direct'`."
            )

        # 6. correction and output
        z_cond[k] = z_u + cov.matvec(H.rmatvec(lam))
        if z_unc is not None:
            z_unc[k] = z_u

    if return_unconditional and z_unc is not None:
        return z_cond, z_unc
    return z_cond


def conditional_mean(
    cov: CovarianceMatrix,
    obs_op: Union[LinearOperator, ArrayLike],
    obs_values: ArrayLike,
    obs_cov: Union[CovarianceMatrix, float, ArrayLike],
    mean: Union[float, ArrayLike] = 0.0,
    solver: str = "cg",
    rtol: float = 1e-10,
    maxiter: Optional[int] = None,
    preconditioner: Optional[LinearOperator] = None,
) -> NDArrayFloat:
    r"""
    Compute the exact (simple) kriging mean of the conditional distribution.

    This solves the same ``(H cov H^T + R) lam = r`` system used by
    :py:func:`conditional_simulate`, but only once, with the true residual
    ``obs_values - obs_op @ mean``. It is deterministic (no Monte Carlo
    error), unlike the sample mean of :py:func:`conditional_simulate`'s
    output.

    Note that this only returns the posterior mean field; the posterior
    covariance/variance is not computed by this function (it can be
    estimated from the empirical covariance of :py:func:`conditional_simulate`
    realizations).

    Parameters
    ----------
    cov : CovarianceMatrix
        Prior covariance representation of the field, with shape ``(n, n)``.
    obs_op : LinearOperator or ArrayLike, shape (n_obs, n)
        Observation/measurement operator ``H``.
    obs_values : ArrayLike, shape (n_obs,)
        Observed data.
    obs_cov : CovarianceMatrix, float, or ArrayLike
        Observation-error covariance ``R``, see :py:func:`conditional_simulate`.
    mean : float or ArrayLike, shape (n,), default 0.0
        Prior mean of the field.
    solver : {"cg", "direct"}, default "cg"
        How the ``(H cov H^T + R) lam = r`` system is solved, see
        :py:func:`conditional_simulate`.
    rtol, maxiter, preconditioner :
        Only used when ``solver="cg"``, forwarded to
        :py:func:`scipy.sparse.linalg.cg`.

    Returns
    -------
    NDArrayFloat, shape (n,)
        The conditional (posterior) mean field.

    Raises
    ------
    ValueError
        If `obs_values` is not 1D, if `obs_op` has an incompatible shape, or
        if `solver` is not one of ``"cg"``/``"direct"``.
    RuntimeError
        If the linear solve does not converge (``solver="cg"``) or if the
        dense data-space system is not numerically SPD (``solver="direct"``).

    Examples
    --------
    >>> import numpy as np
    >>> import covmats
    >>> x = np.linspace(0, 10, 25)
    >>> C = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / 2.0**2)
    >>> cov = covmats.CovViaCholesky(np.linalg.cholesky(C + 1e-10 * np.eye(25)))
    >>> H = covmats.make_point_observation_operator([12], n=25)
    >>> m = covmats.conditional_mean(cov, H, np.array([1.0]), obs_cov=1e-8)
    >>> round(float(m[12]), 3)
    1.0
    """
    n, n_obs, H, R, mean_vec, _obs_values = _prepare(
        cov, obs_op, obs_values, obs_cov, mean, solver
    )
    r = _obs_values - H.matvec(mean_vec)

    if solver == "direct":
        CHt, factor = _build_dense_system(cov, H, R, n_obs)
        lam = cho_solve(factor, r)
        return mean_vec + CHt @ lam

    def A_matvec(lam: NDArrayFloat) -> NDArrayFloat:
        return H.matvec(cov.matvec(H.rmatvec(lam))) + R.matvec(lam)

    A = LinearOperator(shape=(n_obs, n_obs), matvec=A_matvec, dtype=np.float64)  # ty: ignore[unknown-argument]
    lam, info = cg(A, r, rtol=rtol, maxiter=maxiter, M=preconditioner)
    if info != 0:
        raise RuntimeError(f"CG did not converge for the mean solve (info={info})!")
    return mean_vec + cov.matvec(H.rmatvec(lam))
