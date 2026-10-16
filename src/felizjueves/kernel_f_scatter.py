"""The scatter kernel of the atomic-free F write-back (#358), and its host map.

    build_scatter_kernel()                         -> Kernel   `f_scatter`
    scatter_map(cls, n_ket, bra_ao, ket_ao)        -> ScatterMap
    footprint_bound(cls, bra_ao, ket_ao)           -> int      elements of F
    launch_shape(n_el, threads)                    -> (grid, (threads, 1))
    n_slot(cls, n_bra, n_ket)                      `f_levers.scatter`'s

The second kernel of `f_levers/scatter.py`'s build. A `scatter` F launch
stores every contribution it would have added to `F` into its slot buffer
`W`, with no atomics; this kernel then sums `W` into `F`:

    e = blockIdx.x blockDim.x + threadIdx.x      one thread per ELEMENT of F
    if e < n_el:                                 the launch touches
        r, c = el[e]
        s = 0
        for t = ptr[e]; t < ptr[e + 1]; t++:     that element's slots, in
            s = s + W[src[t]]                    ascending slot order
        F[r, c] = F[r, c] + s                    one writer: a plain store

CLASS-AGNOSTIC: the kernel names no class, no AO count and no block; one
unit (`f_scatter`, `NAME`) serves every F class, and everything about the
launch is in its map. DETERMINISTIC: the elements are distinct (one thread
each) and each sums its slots in a fixed order, so with the F launches and
their scatters in stream order the closed `F` is bit-identical from run to
run -- the property no atomic path has.

THE MAP (`scatter_map`) is host data, as `kernel_f.pack` is: from the
class, the launch's ket count and the two sides' AO offsets (`Packed.ao`)
it lists, for EVERY slot of `W` (`f_levers.scatter`'s layout), the element
of `F` the literal kernel would have added it to, then groups the slots by
element: `el` the distinct elements in ascending `(row, col)` order, `ptr`
their CSR offsets, `src` the slots, each element's ascending. Every slot is
listed exactly once (`src` is a permutation of `range(n_slot)`), which the
builder checks; a slot no thread stored (a screened quartet, the canonical
half) holds the host's zero and adds exactly.

THE AO FOOTPRINT, DERIVED (`footprint_bound`). The elements a launch touches
are the bra blocks `(a, b)` -- `n_bra n_ab` of them, the bra pairs being
distinct -- the ket blocks `(c, d)` -- `n_ket n_cd` -- and the four cross
blocks, whose rows are a bra shell's AOs and columns a ket shell's
(`kernel_f`'s `ao(b, i, j)`), so at most `|A_bra| |A_ket|`, `A_bra` the AOs
of the launch's bra shells and `A_ket` its ket shells':

    n_el <= n_bra n_ab + n_ket n_cd + |A_bra| |A_ket|

an equality when the two AO sets are disjoint and every bra meets every ket
(an off-diagonal launch of distinct shells). The map refuses a launch whose
elements exceed it, so a footprint too small is an error, never a dropped
term; `launch_f.launch_footprint` declares the same bound from the plan.

Nothing here imports the plan layer, the AO layer, `reference/`, the
interpreter, a backend or a device: the `Kernel` is arch-neutral and the
map is plain ints.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from felizjueves.f_levers import scatter as _sc
from felizjueves.f_levers.seams import BLOCKS, FOCK
from felizjueves.hgp import vrr as _vrr
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Fn,
    For,
    If,
    Index,
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Scalar,
    Var,
)

__all__ = [
    "NAME",
    "PARAMS",
    "SLOTS",
    "ScatterMap",
    "build_scatter_kernel",
    "footprint_bound",
    "launch_shape",
    "n_slot",
    "scatter_map",
]

NAME = "f_scatter"
SLOTS = _sc.SLOTS
PARAMS = ("n_el", "n_slot", "n_ao", "el", "ptr", "src", SLOTS, FOCK)
n_slot = _sc.n_slot


def _c(fn: Fn, *args) -> Call:
    return Call(fn, args)


def build_scatter_kernel() -> Kernel:
    """`f_scatter` (module docstring): one thread per element of `F`."""
    nel, ns, nao = Var("n_el"), Var("n_slot"), Var("n_ao")
    e = Var("e")

    def i32(n: str, *shape) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    params = (
        Param("n_el", Scalar.I32), Param("n_slot", Scalar.I32),
        Param("n_ao", Scalar.I32),
        i32("el", nel, Lit(2)), i32("ptr", _c(Fn.ADD, nel, Lit(1))),
        i32("src", ns),
        Param(SLOTS, Scalar.F64, array=True, shape=(ns,)),
        Param(FOCK, Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nao, nao)),
    )
    assert tuple(p.name for p in params) == PARAMS
    body = (
        Note("#358: the atomic-free F write-back's scatter, class-agnostic: "
             "one thread per element of F the launch touches, summing its "
             "slots of W in ascending slot order, then one plain store; "
             "deterministic, no atomics"),
        Note("launch: grid = ceil(n_el / threads), blockDim = (threads, 1), "
             "after every F launch of the scatter lever, before the next"),
        Decl("e", Scalar.I32, init=_c(Fn.ADD, _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)),
            Call(Fn.THREAD_X))),
        If(_c(Fn.LT, e, nel), (
            Decl("r", Scalar.I32, init=Index("el", (e, Lit(0)))),
            Decl("c", Scalar.I32, init=Index("el", (e, Lit(1)))),
            Decl("s", Scalar.F64, init=Lit(0.0)),
            For("t", Index("ptr", (e,)), Index("ptr", (_c(Fn.ADD, e, Lit(1)),)),
                Lit(1), (Assign(Var("s"), _c(
                    Fn.ADD, Var("s"), Index(SLOTS, (Index("src", (Var("t"),)),)))),)),
            Assign(Index(FOCK, (Var("r"), Var("c"))), _c(
                Fn.ADD, Index(FOCK, (Var("r"), Var("c"))), Var("s"))),
        )),
    )
    return Kernel(NAME, params, body)


def launch_shape(n_el: int, threads: int) -> tuple[int, tuple[int, int]]:
    """`(ceil(n_el / threads), (threads, 1))`."""
    for name, v in (("n_el", n_el), ("threads", threads)):
        if type(v) is not int or v < 1:
            raise ValueError(f"{name} = {v!r} is not a positive int")
    return -(-n_el // threads), (threads, 1)


# --- the host's map ----------------------------------------------------------


@dataclass(frozen=True)
class ScatterMap:
    """One launch's map (module docstring): `el` flat `n_el x 2`, `ptr`
    `n_el + 1`, `src` `n_slot`, all plain ints."""

    n_el: int
    n_slot: int
    el: tuple[int, ...]
    ptr: tuple[int, ...]
    src: tuple[int, ...]

    def args(self, W, F, n_ao: int) -> dict[str, object]:
        """The scatter kernel's arguments by name."""
        return {"n_el": self.n_el, "n_slot": self.n_slot, "n_ao": n_ao,
                "el": self.el, "ptr": self.ptr, "src": self.src, SLOTS: W,
                FOCK: F}


def _pairs(ao: Sequence[int], what: str) -> list[tuple[int, int]]:
    ao = [int(x) for x in ao]
    if not ao or len(ao) % 2 or any(x < 0 for x in ao):
        raise ValueError(f"{what} = {ao!r} is not n x 2 non-negative ints")
    return list(zip(ao[::2], ao[1::2], strict=True))


def _aos(cls, pairs, first: int) -> set[int]:
    n = [len(_vrr.cartesians(x)) for x in tuple(cls)]
    return {o + i for p in pairs for s, o in enumerate(p)
            for i in range(n[first + s])}


def footprint_bound(cls, bra_ao: Sequence[int], ket_ao: Sequence[int]) -> int:
    """`n_bra n_ab + n_ket n_cd + |A_bra| |A_ket|` (module docstring)."""
    bras, kets = _pairs(bra_ao, "bra_ao"), _pairs(ket_ao, "ket_ao")
    return (len(bras) * _sc.block_size(cls, BLOCKS[0])
            + len(kets) * _sc.block_size(cls, BLOCKS[1])
            + len(_aos(cls, bras, 0)) * len(_aos(cls, kets, 2)))


def scatter_map(cls, n_ket: int, bra_ao: Sequence[int],
                ket_ao: Sequence[int]) -> ScatterMap:
    """The map of one launch of class `cls` (module docstring)."""
    cls = tuple(cls)
    bras, kets = _pairs(bra_ao, "bra_ao"), _pairs(ket_ao, "ket_ao")
    if len(kets) != n_ket:
        raise ValueError(f"n_ket = {n_ket}, but ket_ao holds {len(kets)} pairs")
    n = [len(_vrr.cartesians(x)) for x in cls]
    total = _sc.n_slot(cls, len(bras), n_ket)
    target: list[tuple[int, int] | None] = [None] * total

    def put(slot: int, rc: tuple[int, int]) -> None:
        if target[slot] is not None:
            raise AssertionError(f"slot {slot} written twice")
        target[slot] = rc

    ab = BLOCKS[0]
    for bra, (oa, ob) in enumerate(bras):
        for i in range(n[ab[0]]):
            for j in range(n[ab[1]]):
                put(_sc.slot_ab(cls, n_ket, bra, i * n[ab[1]] + j),
                    (oa + i, ob + j))
        for k, (oc, od) in enumerate(kets):
            o = (oa, ob, oc, od)
            for b in BLOCKS[1:]:
                for i in range(n[b[0]]):
                    for j in range(n[b[1]]):
                        put(_sc.slot_ket(cls, n_ket, bra, k, b, i, j),
                            (o[b[0]] + i, o[b[1]] + j))
    if any(t is None for t in target):
        raise AssertionError("a slot of W has no element of F")
    groups: dict[tuple[int, int], list[int]] = {}
    for slot, rc in enumerate(target):
        groups.setdefault(rc, []).append(slot)  # ascending slot order
    elems = sorted(groups)
    bound = footprint_bound(cls, bra_ao, ket_ao)
    if len(elems) > bound:
        raise AssertionError(f"{len(elems)} elements exceed the derived "
                             f"footprint {bound}")
    ptr = [0]
    for rc in elems:
        ptr.append(ptr[-1] + len(groups[rc]))
    return ScatterMap(len(elems), total,
                      tuple(x for rc in elems for x in rc), tuple(ptr),
                      tuple(s for rc in elems for s in groups[rc]))
