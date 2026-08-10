# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""
Regression tests: pickling must preserve all state on every CovarianceMatrix
(and other LinearOperator-based) subclass.

Context
-------
scipy 1.18.0 introduced ``LinearOperator.__getstate__``, which serializes
only ``self.__dict__`` and has no awareness of ``__slots__``. Because every
class in this module stores its actual state in ``__slots__`` (for memory
efficiency), that scipy change silently drops all of it on pickle -- no
error at pickle time, only a confusing ``AttributeError`` later, wherever
the first dropped attribute happens to get accessed (which can be far away
from the pickling site, e.g. inside a worker process).

This file has two jobs:

1. ``test_scipy_linearoperator_slots_pickle_compat`` is a *canary* test
   against scipy itself, using a minimal LinearOperator subclass with no
   dependency on any of our own classes. If this ever starts failing again
   (e.g. a scipy regression reappears, or a fix gets reverted), it tells you
   immediately, without needing to reason through any of our own code.

2. The parametrized ``test_pickle_preserves_state`` and
   ``test_pickle_across_process_pool`` tests exercise every concrete
   CovarianceMatrix subclass (plus CovKernelAsLinop, which is not a
   CovarianceMatrix but shares the same LinearOperator + __slots__ pattern
   and is equally exposed to this bug) to make sure our own
   __getstate__/__setstate__ overrides actually cover every subclass, not
   just the ones we happened to think of when writing the fix.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Tuple, Union

import covmats
import numpy as np
import pytest
import scipy as sp
from scipy.sparse.linalg import LinearOperator

# --------------------------------------------------------------------------- #
# 1. Fixture factories: one per concrete subclass exposed to the bug.
#    Each returns (instance, a compatible right-hand-side vector `b`).
# --------------------------------------------------------------------------- #

_N = 5
_SEED = 0

# NOTE: each factory below creates its own freshly-seeded RNG (rather than
# drawing from one shared, stateful module-level RNG) so that calling a
# factory twice -- as test_pickle_across_process_pool deliberately does, to
# get an independent reference instance -- produces two *identical* objects.
# A shared, advancing RNG would make the two calls diverge and turn every
# comparison into a false failure unrelated to pickling.


def _make_diagonal() -> Tuple[covmats.CovViaDiagonal, np.ndarray]:
    rng = np.random.default_rng(_SEED)
    d = rng.random(_N) + 0.5  # keep strictly positive
    return covmats.CovViaDiagonal(d), rng.random(_N)


def _make_cholesky() -> Tuple[covmats.CovViaCholesky, np.ndarray]:
    rng = np.random.default_rng(_SEED)
    A = rng.random((_N, _N))
    A = A @ A.T + _N * np.eye(_N)  # SPD
    L = np.linalg.cholesky(A)
    return covmats.CovViaCholesky(L), rng.random(_N)


def _make_precision_cholesky() -> Tuple[covmats.CovViaPrecisionCholesky, np.ndarray]:
    rng = np.random.default_rng(_SEED)
    P = rng.random((_N, _N))
    P = P @ P.T + _N * np.eye(_N)  # SPD precision
    L = np.linalg.cholesky(P)
    return covmats.CovViaPrecisionCholesky(L), rng.random(_N)


def _make_eigen() -> Tuple[covmats.CovViaEigenFactorization, np.ndarray]:
    rng = np.random.default_rng(_SEED)
    A = rng.random((_N, _N))
    A = A @ A.T + _N * np.eye(_N)  # SPD
    w, v = np.linalg.eigh(A)
    return covmats.CovViaEigenFactorization((w, v)), rng.random(_N)


def _make_ensemble() -> Tuple[covmats.CovViaEnsemble, np.ndarray]:
    rng = np.random.default_rng(_SEED)
    n_ens = 20
    ensemble = rng.random((n_ens, _N))
    return covmats.CovViaEnsemble(ensemble), rng.random(_N)


def _make_sparse_precision_cholesky() -> Tuple[
    covmats.CovViaSparsePrecisionCholesky, np.ndarray
]:
    rng = np.random.default_rng(_SEED)
    Q = covmats.get_SPD_sparse_n11_example(seed=42)
    L, D, P = sp.linalg.ldl(Q.toarray())
    scf = covmats.SparseCholeskyFactor(
        sp.sparse.csc_array(L), sp.sparse.csc_array(D), P
    )
    cov = covmats.CovViaSparsePrecisionCholesky(scf)
    return cov, rng.random(cov.n_pts)


def _exponential_kernel(dist: np.ndarray) -> np.ndarray:
    """Module-level (not a closure) so it, and objects holding a reference
    to it, remain picklable -- a local/nested function would fail to pickle
    for reasons entirely unrelated to the bug this suite targets, and would
    turn that unrelated failure into a false positive for this test."""
    return np.exp(-dist)


def _make_kernel_linop() -> Tuple[covmats.CovKernelAsLinop, np.ndarray]:
    rng = np.random.default_rng(_SEED)
    pts = rng.random((10, 2))
    op = covmats.CovKernelAsLinop(
        pts, _exponential_kernel, len_scale=np.array([1.0, 1.0])
    )
    return op, rng.random(10)


# NOTE: CovViaSparseCholesky (the covariance-based, as opposed to
# precision-based, sparse Cholesky representation) is intentionally not
# included here: we don't yet have a documented standalone construction
# example for it (see CovViaSparsePrecisionCholesky's docstring for the
# closest analogue). Add a factory here once that's available -- until then,
# that class is NOT covered by this regression suite.

_FACTORIES: list[Tuple[str, Callable[[], Tuple[LinearOperator, np.ndarray]]]] = [
    ("diagonal", _make_diagonal),
    ("cholesky", _make_cholesky),
    ("precision_cholesky", _make_precision_cholesky),
    ("eigen_factorization", _make_eigen),
    ("ensemble", _make_ensemble),
    ("sparse_precision_cholesky", _make_sparse_precision_cholesky),
    ("kernel_linop", _make_kernel_linop),
]


def _all_slots(obj: object) -> set:
    """Collect every __slots__ name declared anywhere in obj's class MRO."""
    slots: set = set()
    for klass in type(obj).__mro__:
        slots.update(getattr(klass, "__slots__", ()) or ())
    return slots


def _solve_or_matvec(
    cov: Union[covmats.CovarianceMatrix, covmats.CovKernelAsLinop], b: np.ndarray
) -> np.ndarray:
    """CovarianceMatrix subclasses expose solve(); CovKernelAsLinop also
    exposes solve() (via GMRES), so this is uniform across all fixtures."""
    return cov.solve(b)


def _assert_slot_value_preserved(name: str, slot: str, value, value2) -> None:
    """
    Compare one slot's pre- and post-pickle value.

    Handles two cases plain `==` gets wrong:
      - numpy arrays, where `==` produces an elementwise array rather than
        a bool;
      - nested LinearOperator-like objects (e.g. SparseCholeskyFactor stored
        inside CovViaSparsePrecisionCholesky._scf), which generally don't
        define a meaningful __eq__ at all, so `==` falls back to identity
        and would report "changed" for any two distinct-but-equal
        instances, pickling bug or not. These are compared recursively by
        their own __slots__ state instead.
    """
    if isinstance(value, np.ndarray):
        np.testing.assert_allclose(
            value2, value, err_msg=f"{name}: slot '{slot}' changed after pickling"
        )
    elif sp.sparse.issparse(value):
        # Sparse arrays' `==` returns an elementwise sparse array, not a
        # bool -- densify for comparison. Fixture sizes here are small
        # enough that this is fine for a test; not something you'd do on
        # production-sized matrices.
        np.testing.assert_allclose(
            value2.toarray(),
            value.toarray(),
            err_msg=f"{name}: slot '{slot}' changed after pickling",
        )
    elif isinstance(value, LinearOperator):
        assert isinstance(value2, type(value)), (
            f"{name}: slot '{slot}' changed type after pickling "
            f"({type(value2)} vs {type(value)})"
        )
        nested_slots = _all_slots(value)
        nested_present = {
            s: getattr(value, s) for s in nested_slots if hasattr(value, s)
        }
        assert nested_present, (
            f"{name}: nested object in slot '{slot}' has no __slots__ state "
            f"set -- fixture looks broken."
        )
        for nested_slot, nested_value in nested_present.items():
            assert hasattr(value2, nested_slot), (
                f"{name}: nested slot '{slot}.{nested_slot}' lost after pickling"
            )
            _assert_slot_value_preserved(
                name,
                f"{slot}.{nested_slot}",
                nested_value,
                getattr(value2, nested_slot),
            )
    else:
        assert value2 == value, f"{name}: slot '{slot}' changed after pickling"


# --------------------------------------------------------------------------- #
# 2. Plain pickle round-trip: every slot must survive, and results computed
#    before/after pickling must match.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name,factory", _FACTORIES, ids=[f[0] for f in _FACTORIES])
def test_pickle_preserves_state(name, factory) -> None:
    cov, b = factory()

    ref_result = _solve_or_matvec(cov, b)

    # Snapshot state *after* calling solve()/matvec(), not before: some
    # classes (e.g. CovKernelAsLinop.count, a running call counter) mutate
    # their own __slots__ state as a side effect of that call. We want to
    # check that whatever state exists at the moment of pickling survives
    # the round-trip -- not compare against a stale pre-call snapshot,
    # which would flag normal, expected mutation as a false failure.
    slots_before = _all_slots(cov)
    present_before = {s: getattr(cov, s) for s in slots_before if hasattr(cov, s)}
    assert present_before, (
        f"{name}: no __slots__ attributes were actually set on this "
        f"instance -- the fixture itself looks broken, fix it before "
        f"trusting this test."
    )

    cov2 = pickle.loads(pickle.dumps(cov))

    missing = [s for s in present_before if not hasattr(cov2, s)]
    assert not missing, (
        f"{name}: lost __slots__ attributes {missing} after pickling "
        f"(scipy {sp.__version__}). This is the "
        f"LinearOperator.__getstate__ regression -- see "
        f"test_scipy_linearoperator_slots_pickle_compat."
    )

    for slot, value in present_before.items():
        _assert_slot_value_preserved(name, slot, value, getattr(cov2, slot))

    np.testing.assert_allclose(
        _solve_or_matvec(cov2, b),
        ref_result,
        err_msg=f"{name}: solve()/matvec() result differs after pickling",
    )


# --------------------------------------------------------------------------- #
# 3. Real multiprocessing round-trip: mirrors how pyesmda actually uses
#    these objects (dispatched via a worker pool, potentially to more than
#    one worker across more than one submitted task).
# --------------------------------------------------------------------------- #


def _mp_worker(
    cov: Union[covmats.CovKernelAsLinop, covmats.CovarianceMatrix], b: np.ndarray
) -> np.ndarray:
    """Module-level so it (and its arguments) can be pickled for dispatch."""
    return _solve_or_matvec(cov, b)


@pytest.mark.parametrize("name,factory", _FACTORIES, ids=[f[0] for f in _FACTORIES])
def test_pickle_across_process_pool(name, factory) -> None:
    cov, b = factory()

    # Compute the reference result from an *independent*, never-pickled
    # instance, and do not call solve()/matvec() on `cov` itself before
    # dispatching it. Some classes (e.g. CovViaDiagonal.invD) memoize
    # derived quantities via @cached_property, which stores its result in
    # __dict__ rather than __slots__. __dict__ *does* survive pickling even
    # when __slots__ doesn't -- so calling solve() on `cov` here first would
    # pre-warm that cache and silently mask exactly the bug this test exists
    # to catch. The whole point is to force each worker to compute
    # everything from scratch, the way a freshly-dispatched batch would.
    cov_ref, _ = factory()
    ref_result = _solve_or_matvec(cov_ref, b)

    with ProcessPoolExecutor(
        max_workers=2, mp_context=mp.get_context("spawn")
    ) as executor:
        # Submit more tasks than workers, so the same object gets pickled
        # and dispatched multiple times -- this is the shape that originally
        # surfaced the bug (multiple batches sharing one covariance object).
        futures = [executor.submit(_mp_worker, cov, b) for _ in range(4)]
        for future in futures:
            np.testing.assert_allclose(
                future.result(),
                ref_result,
                err_msg=(
                    f"{name}: result from worker process differs from the "
                    f"main-process result -- state was lost crossing the "
                    f"process boundary."
                ),
            )
