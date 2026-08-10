# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET
"""
Full-coverage unit tests for ``covmats._sparse_helpers._PickleSafeLinearOperator``.
"""

from __future__ import annotations

import builtins
import copy
import pickle
from typing import Tuple

import covmats._sparse_helpers as sh
import numpy as np
import pytest
import scipy as sp
from covmats._sparse_helpers import _PickleSafeLinearOperator

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class _Leaf(_PickleSafeLinearOperator):
    """Minimal concrete subclass: one level, one slot."""

    __slots__ = ["_a"]

    def __init__(self, a):
        self._a = np.asarray(a, dtype=np.float64)
        super().__init__(dtype=self._a.dtype, shape=(len(a), len(a)))

    def _matvec(self, x):
        return self._a * x

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the covariance matrix (n, n)."""
        return (len(self._a), len(self._a))

    @shape.setter
    def shape(self, value: Tuple[int, int]) -> None:
        """Shape of the covariance matrix (n, n)."""
        pass


class _Mid(_PickleSafeLinearOperator):
    """Intermediate class contributing its own slot, to exercise MRO-wide
    slot collection (not just the leaf class's own __slots__)."""

    __slots__ = ["_mid"]


class _DeepLeaf(_Mid):
    """Two levels of __slots__ above _PickleSafeLinearOperator itself."""

    __slots__ = ["_leaf"]

    def __init__(self, mid, leaf):
        self._mid = np.asarray(mid, dtype=np.float64)
        self._leaf = np.asarray(leaf, dtype=np.float64)
        super().__init__(dtype=self._leaf.dtype, shape=(len(leaf), len(leaf)))

    def _matvec(self, x):
        return self._leaf * x


class _PartiallyInitialized(_PickleSafeLinearOperator):
    """A slot declared but deliberately never assigned, to exercise the
    `hasattr(self, slot)` guard skipping unset slots."""

    __slots__ = ["_set", "_unset"]

    def __init__(self, value):
        self._set = np.asarray(value, dtype=np.float64)
        # _unset is intentionally never assigned.
        super().__init__(dtype=self._set.dtype, shape=(len(value), len(value)))

    def _matvec(self, x):
        return self._set * x


def test_getstate_skips_dunder_dict_and_weakref_slot_names(monkeypatch):
    """
    Exercise the `if slot in ("__dict__", "__weakref__"): continue` guard.

    A real class can't declare "__dict__"/"__weakref__" in its own
    __slots__ when a __dict__-granting ancestor is already present in the
    MRO (LinearOperator doesn't declare __slots__ itself, so it already
    grants every subclass a __dict__) -- Python raises a TypeError at class
    *definition* time for that. Since the slot-collection loop only reads
    `klass.__slots__` as a plain iterable (it never touches the actual slot
    descriptor machinery), we can safely monkeypatch that class attribute
    on an already-defined, already-working class to include those names,
    without needing to construct an otherwise-illegal class.
    """
    op = _make_leaf()
    monkeypatch.setattr(
        _Leaf, "__slots__", ["_a", "__dict__", "__weakref__"], raising=False
    )
    state = op.__getstate__()
    assert "__slot____dict__" not in state
    assert "__slot____weakref__" not in state
    assert "__slot___a" in state


def _make_leaf():
    return _Leaf([1.0, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# Real, end-to-end round-trips: whatever __getstate__/__setstate__ scipy
# actually provides in this environment, right now.
# --------------------------------------------------------------------------- #


def test_pickle_roundtrip_preserves_slot():
    op = _make_leaf()
    op2 = pickle.loads(pickle.dumps(op))
    np.testing.assert_array_equal(op2._a, op._a)
    np.testing.assert_array_equal(op2.matvec(np.ones(3)), op.matvec(np.ones(3)))


def test_deepcopy_roundtrip_preserves_slot():
    op = _make_leaf()
    op2 = copy.deepcopy(op)
    np.testing.assert_array_equal(op2._a, op._a)


def test_pickle_roundtrip_preserves_slots_across_mro():
    op = _DeepLeaf(mid=[1.0, 2.0], leaf=[3.0, 4.0])
    op2 = pickle.loads(pickle.dumps(op))
    np.testing.assert_array_equal(op2._mid, op._mid)
    np.testing.assert_array_equal(op2._leaf, op._leaf)


def test_pickle_roundtrip_skips_unset_slot():
    op = _PartiallyInitialized([1.0, 2.0])
    op2 = pickle.loads(pickle.dumps(op))
    np.testing.assert_array_equal(op2._set, op._set)
    assert not hasattr(op2, "_unset")


def test_getstate_never_includes_unset_slot():
    op = _PartiallyInitialized([1.0, 2.0])
    state = op.__getstate__()
    assert "__slot___unset" not in state
    assert "__slot___set" in state


# --------------------------------------------------------------------------- #
# __getstate__: directly controlling what super().__getstate__() returns,
# to hit every branch deterministically regardless of installed scipy.
# --------------------------------------------------------------------------- #


def test_getstate_base_returns_none(monkeypatch):
    """super().__getstate__ exists but returns None/falsy -> only slots added."""
    monkeypatch.setattr(
        sp.sparse.linalg.LinearOperator,
        "__getstate__",
        lambda self: None,
        raising=False,
    )
    op = _make_leaf()
    state = op.__getstate__()
    assert state == {"__slot___a": pytest.approx(op._a)} or list(state) == [
        "__slot___a"
    ]


def test_getstate_base_returns_nonempty_dict(monkeypatch):
    """super().__getstate__ returns a plain, non-empty dict (scipy>=1.18 shape)."""
    monkeypatch.setattr(
        sp.sparse.linalg.LinearOperator,
        "__getstate__",
        lambda self: {"dtype": self.dtype},
        raising=False,
    )
    op = _make_leaf()
    state = op.__getstate__()
    assert state["dtype"] == op.dtype
    assert "__slot___a" in state


def test_getstate_base_returns_empty_dict(monkeypatch):
    """An empty dict is falsy: the `elif raw:` branch must NOT run, but
    slot collection must still happen."""
    monkeypatch.setattr(
        sp.sparse.linalg.LinearOperator, "__getstate__", lambda self: {}, raising=False
    )
    op = _make_leaf()
    state = op.__getstate__()
    assert "__slot___a" in state
    assert len(state) == 1


@pytest.mark.parametrize(
    "dict_part,slots_part",
    [
        ({"y": 2}, {"_hidden": 99}),  # both present
        (None, {"_hidden": 99}),  # dict part absent
        ({"y": 2}, None),  # slots part absent
        (None, None),  # both absent
        ({}, {}),  # both present but empty (falsy)
    ],
    ids=["both", "dict-only", "slots-only", "neither", "both-empty"],
)
def test_getstate_base_returns_tuple_form(monkeypatch, dict_part, slots_part):
    """super().__getstate__ returns the (dict, slots) tuple form used by
    Python's default object.__getstate__() on slotted instances (the shape
    hit when scipy has no LinearOperator.__getstate__ of its own, on
    Python >= 3.11)."""
    monkeypatch.setattr(
        sp.sparse.linalg.LinearOperator,
        "__getstate__",
        lambda self: (dict_part, slots_part),
        raising=False,
    )
    op = _make_leaf()
    state = op.__getstate__()

    if dict_part:
        assert state["y"] == 2
    else:
        assert "y" not in state

    if slots_part:
        assert state["__slot___hidden"] == 99

    # Real __slots__ attributes are always added regardless of what the
    # tuple contained.
    assert "__slot___a" in state


def test_getstate_no_base_getstate_reachable(monkeypatch):
    """
    Simulate Python < 3.11 with scipy < 1.18.0: nothing in the MRO defines
    __getstate__ at all, so `getattr(super(), "__getstate__", None)` is
    None. Unreachable through a real instance on this environment (object
    itself provides a default from Python 3.11 on), so we shadow the
    module-level `super` name that _PickleSafeLinearOperator.__getstate__
    actually calls.
    """

    class _FakeSuperNoGetstate:
        pass

    monkeypatch.setattr(
        sh, "super", lambda *a, **k: _FakeSuperNoGetstate(), raising=False
    )
    op = _make_leaf()
    state = op.__getstate__()
    assert state == {"__slot___a": op._a} or list(state) == ["__slot___a"]
    np.testing.assert_array_equal(state["__slot___a"], op._a)


# --------------------------------------------------------------------------- #
# __setstate__: symmetric coverage of every branch.
# --------------------------------------------------------------------------- #


def test_setstate_flat_dict_with_slot_prefixed_and_plain_keys(monkeypatch):
    """A flat dict mixing '__slot__'-prefixed entries with plain dict
    entries is split and applied correctly (the scipy>=1.18 shape)."""
    monkeypatch.setattr(
        sp.sparse.linalg.LinearOperator,
        "__getstate__",
        lambda self: {"dtype": self.dtype},
        raising=False,
    )
    monkeypatch.setattr(
        sp.sparse.linalg.LinearOperator,
        "__setstate__",
        lambda self, state: self.__dict__.update(state),
        raising=False,
    )
    op = _make_leaf()
    state = op.__getstate__()

    op2 = _Leaf.__new__(_Leaf)
    op2.__setstate__(state)
    np.testing.assert_array_equal(op2._a, op._a)
    assert op2.dtype == op.dtype


@pytest.mark.parametrize(
    "dict_part,slots_part",
    [
        ({"y": 2}, {"_hidden": 99}),
        (None, {"_hidden": 99}),
        ({"y": 2}, None),
        (None, None),
    ],
    ids=["both", "dict-only", "slots-only", "neither"],
)
def test_setstate_tuple_form(monkeypatch, dict_part, slots_part):
    """__setstate__ receiving the (dict, slots) tuple form directly."""
    calls = []

    def fake_base_setstate(self, d):
        calls.append(dict(d))
        self.__dict__.update(d)

    monkeypatch.setattr(
        sp.sparse.linalg.LinearOperator,
        "__setstate__",
        fake_base_setstate,
        raising=False,
    )
    op2 = _Leaf.__new__(_Leaf)
    op2.__setstate__((dict_part, slots_part))

    if dict_part:
        assert op2.y == 2
    if slots_part:
        assert op2._hidden == 99
    assert calls, "base __setstate__ should have been invoked with the dict part"


def test_setstate_base_setstate_raises_typeerror_falls_back_to_dict_update(monkeypatch):
    """If super().__setstate__ exists but rejects our reconstructed dict
    (TypeError), fall back to updating self.__dict__ directly."""

    def rejecting_setstate(self, state):
        raise TypeError("nope")

    monkeypatch.setattr(
        sp.sparse.linalg.LinearOperator,
        "__setstate__",
        rejecting_setstate,
        raising=False,
    )
    op2 = _Leaf.__new__(_Leaf)
    op2.__setstate__({"y": 5, "__slot___a": np.array([1.0, 2.0, 3.0])})
    assert op2.y == 5
    np.testing.assert_array_equal(op2._a, [1.0, 2.0, 3.0])


def test_setstate_no_base_setstate_updates_dict_directly(monkeypatch):
    """No super().__setstate__ reachable at all: dict_state is applied via
    self.__dict__.update directly."""

    class _FakeSuperNoSetstate:
        pass

    monkeypatch.setattr(
        sh, "super", lambda *a, **k: _FakeSuperNoSetstate(), raising=False
    )
    op2 = _Leaf.__new__(_Leaf)
    op2.__setstate__({"y": 7, "__slot___a": np.array([9.0])})
    assert op2.y == 7
    np.testing.assert_array_equal(op2._a, [9.0])


def test_setstate_no_base_setstate_and_empty_dict_state_is_noop(monkeypatch):
    """No super().__setstate__, and no plain dict entries at all: the
    `elif dict_state and hasattr(...)` guard must not blow up on an empty
    dict_state (short-circuits before touching __dict__)."""

    class _FakeSuperNoSetstate:
        pass

    monkeypatch.setattr(
        sh, "super", lambda *a, **k: _FakeSuperNoSetstate(), raising=False
    )
    op2 = _Leaf.__new__(_Leaf)
    op2.__setstate__({"__slot___a": np.array([4.0])})
    np.testing.assert_array_equal(op2._a, [4.0])


def test_setstate_self_has_no_dict_is_skipped_safely(monkeypatch):
    """
    Simulate a hypothetical reuse of this class where `self` lacks a
    __dict__ entirely (never happens through real scipy.LinearOperator,
    which doesn't declare __slots__ itself and so always grants one -- this
    guards the mixin for correctness if it's ever reused in a context that
    doesn't). Shadow `hasattr` at the module level so
    `hasattr(self, "__dict__")` reports False without needing a genuinely
    dict-less instance, which isn't constructible here.
    """
    real_hasattr = builtins.hasattr

    def fake_hasattr(obj, name):
        if name == "__dict__":
            return False
        return real_hasattr(obj, name)

    class _FakeSuperNoSetstate:
        pass

    monkeypatch.setattr(
        sh, "super", lambda *a, **k: _FakeSuperNoSetstate(), raising=False
    )
    monkeypatch.setattr(sh, "hasattr", fake_hasattr, raising=False)

    op2 = _Leaf.__new__(_Leaf)
    # dict_state is non-empty, but hasattr(self, "__dict__") now reports
    # False, so the update must be skipped rather than raising.
    op2.__setstate__({"y": 1, "__slot___a": np.array([2.0])})
    np.testing.assert_array_equal(op2._a, [2.0])
    assert "y" not in op2.__dict__  # skipped, as intended


def test_setstate_base_setstate_raises_and_self_has_no_dict(monkeypatch):
    """Same as above, but via the except-TypeError fallback path rather
    than the no-base-setstate path."""
    real_hasattr = builtins.hasattr

    def fake_hasattr(obj, name):
        if name == "__dict__":
            return False
        return real_hasattr(obj, name)

    def rejecting_setstate(self, state):
        raise TypeError("nope")

    monkeypatch.setattr(
        sp.sparse.linalg.LinearOperator,
        "__setstate__",
        rejecting_setstate,
        raising=False,
    )
    monkeypatch.setattr(sh, "hasattr", fake_hasattr, raising=False)

    op2 = _Leaf.__new__(_Leaf)
    op2.__setstate__({"y": 1, "__slot___a": np.array([3.0])})
    np.testing.assert_array_equal(op2._a, [3.0])
    assert "y" not in op2.__dict__


# --------------------------------------------------------------------------- #
# End-to-end sanity against the real SparseCholeskyFactor built on top of
# _PickleSafeLinearOperator, as a final integration check.
# --------------------------------------------------------------------------- #


def test_sparse_cholesky_factor_pickle_roundtrip():
    import covmats

    Q = covmats.get_SPD_sparse_n11_example(seed=42)
    L, D, P = sp.linalg.ldl(Q.toarray())
    scf = covmats.SparseCholeskyFactor(
        sp.sparse.csc_array(L), sp.sparse.csc_array(D), P
    )
    scf2 = pickle.loads(pickle.dumps(scf))
    x = np.ones(Q.shape[0])
    np.testing.assert_allclose(scf2.matvec(x), Q @ x)
