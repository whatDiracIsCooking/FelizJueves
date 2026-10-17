"""passes.py: `atomize`, the atomic write-back of `[JEngine, v2, §5]`, as a pass.

The property is bit-exact: onto a zeroed `J`, one atomic add of `x` is `x`, so
`interpret(atomize(k))` and `interpret(k)` agree under `np.array_equal`, with
no tolerance to hide a real difference behind. The structural half pins what
the pass changed (the write) and what it must not have (the lane-0 guard, the
census). Nothing compiles and nothing launches on a card.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from tiers import by_l, split_by_l

from felizjueves.interp import Launch, interpret
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Call,
    Fn,
    If,
    Index,
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Raw,
    Scalar,
    walk,
)
from felizjueves.kernel import (
    bra_functions,
    build_pq_kernel,
    ket_functions,
    launch_shape,
    pq_spec,
)
from felizjueves.passes import ATOMIC_NOTE, LANE0, atomize
from felizjueves.reference.boys import boys


@cache
def _pair(L: int) -> tuple[Kernel, Kernel]:
    k = build_pq_kernel(L)
    return k, atomize(k)


def _inputs(L: int, n_bra: int, n_ket: int, seed: int) -> dict:
    """As test_kernel.py's: kets sorted descending by Iq * Dmax, J zeroed."""
    spec = pq_spec(L)
    rng = np.random.default_rng(seed)
    n_p, n_q = len(bra_functions(spec)), len(ket_functions(spec))
    Iq = rng.uniform(0.2, 1.0, n_ket)
    Dmax = rng.uniform(0.5, 1.0, n_ket)
    srt = np.argsort(-(Iq * Dmax), kind="stable")
    return dict(
        n_bra=n_bra, n_ket=n_ket, tau=0.0,
        P=rng.uniform(-0.5, 0.5, (n_bra, 3)), zeta=rng.uniform(0.6, 1.4, n_bra),
        Ip=rng.uniform(0.5, 1.0, n_bra),
        Q=rng.uniform(-0.5, 0.5, (n_ket, 3)), eta=rng.uniform(0.6, 1.4, n_ket),
        Iq=Iq[srt].copy(), Dmax=Dmax[srt].copy(),
        D=rng.uniform(-1.0, 1.0, (n_ket, n_q)),
        J=np.zeros((n_bra, n_p)),
    )


def _J(k: Kernel, args: dict, lanes: int) -> np.ndarray:
    grid, block = launch_shape(args["n_bra"], lanes)
    return interpret(k, launch=Launch(grid, block, lanes), boys=boys, **args)["J"]


def _j_writes(k: Kernel) -> tuple[list[Assign], list[AtomicAdd]]:
    stmts = list(walk(k.body))

    def into_j(s: object) -> bool:
        return isinstance(s.dst, Index) and s.dst.array == "J"  # type: ignore[attr-defined]

    return ([s for s in stmts if isinstance(s, Assign) and into_j(s)],
            [s for s in stmts if isinstance(s, AtomicAdd) and into_j(s)])


# ---------------------------------------------------------------------------
# The property, bit-exact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", [
    by_l(c, c[0], id=f"{c[0]}-{c[1]}") for c in ((0, 1), (3, 2), (4, 4), (7, 2))])
def test_atomize_onto_zeros_is_bit_exact(case: tuple[int, int]) -> None:
    L, lanes = case
    k, ak = _pair(L)
    args = _inputs(L, n_bra=3, n_ket=5, seed=48 + L)
    plain, atomic = _J(k, args, lanes), _J(ak, args, lanes)
    assert np.any(plain != 0.0)
    assert np.array_equal(atomic, plain)


def test_atomize_accumulates_where_the_plain_write_overwrites() -> None:
    """The point of the pass: a second kernel's J_p adds to the first's."""
    k, ak = _pair(4)
    args = _inputs(4, n_bra=2, n_ket=4, seed=480)
    x = _J(k, args, 2)
    seeded = {**args, "J": np.full_like(args["J"], 0.375)}
    assert np.array_equal(_J(k, seeded, 2), x)
    assert np.array_equal(_J(ak, seeded, 2), seeded["J"] + x)
    assert np.array_equal(args["J"], np.zeros_like(x))  # caller's J untouched


# ---------------------------------------------------------------------------
# Structure and census
# ---------------------------------------------------------------------------


@pytest.mark.slow  # L = 6 > FAST_L_MAX (tiers.py)
@pytest.mark.claim("jengine/alg1-line10-atomic-free")
def test_atomize_writes_j_only_by_atomic_add() -> None:
    """Alg. 1 line 10 is a plain write; §5 corrects it, and so does atomize:
    exactly n_p AtomicAdds into J, no Assign into J, same dst and value,
    still under the unchanged lane-0 guard, with the Note saying why."""
    k, ak = _pair(6)
    n_p = len(bra_functions(pq_spec(6)))
    plain, none = _j_writes(k)
    assigns, atomics = _j_writes(ak)
    assert (len(plain), none, assigns, len(atomics)) == (n_p, [], [], n_p)
    assert [(a.dst, a.value) for a in atomics] == [(w.dst, w.value) for w in plain]

    (g0,) = [s for s in walk(k.body) if isinstance(s, If) and s.cond == LANE0]
    (g1,) = [s for s in walk(ak.body) if isinstance(s, If) and s.cond == LANE0]
    assert LANE0 == Call(Fn.LT, (Call(Fn.THREAD_Y), Lit(1)))
    assert g1.then == tuple(atomics) and g1.orelse == g0.orelse == ()

    notes = [s.prov for s in walk(ak.body) if isinstance(s, Note)]
    assert ATOMIC_NOTE in notes
    assert "§5" in ATOMIC_NOTE and "Alg. 1 line 10" in ATOMIC_NOTE
    assert "r = p + q" in ATOMIC_NOTE
    assert not any("NOT atomic" in n or "no atomics" in n for n in notes)


@pytest.mark.slow  # L = 5 > FAST_L_MAX (tiers.py)
def test_only_the_write_and_its_notes_change() -> None:
    k, ak = _pair(5)
    assert (ak.name, ak.params) == (k.name, k.params)
    before, after = list(walk(k.body)), list(walk(ak.body))
    assert len(before) == len(after)
    changed = [(a, b) for a, b in zip(before, after, strict=True) if a != b]
    for a, b in changed:
        assert isinstance(a, (Assign, Note, If)), a
        assert isinstance(b, (AtomicAdd, Note, If)), b


@pytest.mark.parametrize("L", split_by_l([0, 6, 16], lambda L: L))
def test_the_census_does_not_move(L: int) -> None:
    """build_pq_kernel stays atomic-free (test_kernel.py's census, mirrored);
    atomize adds only AtomicAdds into J, and no Raw anywhere."""
    k, ak = _pair(L)
    assert not any(isinstance(s, (AtomicAdd, Raw)) for s in walk(k.body))
    assert not any(isinstance(s, Raw) for s in walk(ak.body))
    atomics = [s for s in walk(ak.body) if isinstance(s, AtomicAdd)]
    assert atomics and all(s.dst.array == "J" for s in atomics)
    assert len(atomics) == len(bra_functions(pq_spec(L)))


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_atomize_twice_is_refused() -> None:
    _, ak = _pair(3)
    with pytest.raises(ValueError, match="already atomics"):
        atomize(ak)


def _tiny(*body: object) -> Kernel:
    J = Param("J", Scalar.F64, array=True, intent=Intent.OUT)
    return Kernel("tiny", (J,), tuple(body))  # type: ignore[arg-type]


def test_a_kernel_with_no_write_is_refused() -> None:
    with pytest.raises(ValueError, match="no plain write"):
        atomize(_tiny(Note("nothing here")))


def test_a_write_outside_the_lane0_guard_is_refused() -> None:
    w = Assign(Index("J", (Lit(0),)), Lit(1.0))
    with pytest.raises(ValueError, match="lane-0 guard"):
        atomize(_tiny(w))
    with pytest.raises(ValueError, match="lane-0 guard"):
        atomize(_tiny(If(LANE0, (w,)), w))


def test_a_guarded_write_is_atomized() -> None:
    w = Assign(Index("J", (Lit(0),)), Lit(1.0))
    got = atomize(_tiny(If(LANE0, (w,))))
    assert got.body == (If(LANE0, (AtomicAdd(w.dst, w.value),)),)


def test_atomize_takes_a_kernel() -> None:
    with pytest.raises(TypeError):
        atomize("kernel")  # type: ignore[arg-type]
