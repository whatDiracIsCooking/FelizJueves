"""Interleaved `I_p`, the warp arm, `k_levers/interleave.py` (#343), interpreted.

The lever replaces the K kernel's per-`(b,d)`-row contraction: instead of
staging the whole `I[p]` column in `Ksh` and then folding each `(a,c)` row's
`V`, it forms ONE `I_p` in a local, feeds every running `V` it reaches, and
discards it. What is asserted:

1. THE DECLARATION: `interleave` is legal on `Target.K` only, refused on the
   other targets with a stated reason, keyed as any lever is, registered in
   `k_levers.REGISTRY`, and absent from every literal build (the literal K
   kernels' digest is test_k_seams.py's pin, unchanged).
2. THE SHAPE: the hooks it moves (`contract`, and `shared_decl` /
   `init_acc` / `write_back` re-slotted), `Ksh` holding the `K` slots only,
   no `I` row ever written to shared memory, every `I_p` formed once and
   read by a `V` fold, the `[p|q]` program before `E^bd` untouched, the
   lever's provenance and the trade's `Note` in the kernel.
3. PARITY, the issue's acceptance (host shim, L <= 2): per bra record the
   kernel's `K` is `ao.k.k_quartet` BIT FOR BIT and `==` the literal
   kernel's, every class with L <= 2 (fast) and a sample through L = 4 with
   no shell above d (`slow`, milestone 13's cap); with a 3-batch split the
   atomized batch kernels onto a zeroed `K` are the literal batched kernels'
   `K` bit for bit. Bitwise because nothing is reassociated (the module
   docstring): no tolerance is involved.
4. COMPOSITION, for #348: with #341's `split` in the same spec, the element
   kernels carry both levers, each holds one `K` slot and no `I` row, and
   the union is the split-alone (hence the literal) block bit for bit.
5. THE GOLDENS: `(p s|p s)`, one batch, per backend.

Fast tier unless `slow`: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import test_kernel_k as tk

from felizjueves import k_levers, levers
from felizjueves.emit import cuda, hip
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Decl,
    For,
    Index,
    Kernel,
    Note,
    Var,
    walk,
)
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.k_levers import interleave, split
from felizjueves.k_levers.seams import DEFAULT_HOOKS, KSH, KHooks, ncart
from felizjueves.kernel_k import (
    KET,
    build_k_kernel,
    build_k_kernels,
    launch_shape,
)
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.reference.boys import boys
from felizjueves.spec import Operator, Spec, Target

Am = tuple[int, int, int, int]

FAST: list[Am] = tk.FAST  # every K class with L <= 2
SLOW_SAMPLE: list[Am] = [(1, 1, 1, 0), (1, 0, 1, 1), (1, 1, 1, 1),
                         (2, 0, 2, 0), (2, 2, 0, 0), (0, 0, 2, 2),
                         (2, 1, 1, 0), (0, 2, 1, 1)]
CLASSES = ([pytest.param(am, id=tk._id(am)) for am in FAST]
           + [pytest.param(am, id=tk._id(am), marks=pytest.mark.slow)
              for am in SLOW_SAMPLE])
FAST_IDS = [pytest.param(a, id=tk._id(a)) for a in FAST]


def lever_spec(am: Am, n_batches: int = 1, *extra: str) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.K,
                tuple(sorted((interleave.NAME, *extra))), n_batches)


def literal_spec(am: Am, n_batches: int = 1) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.K, (), n_batches)


def _n(am: Am) -> int:
    return ncart(am[0]) * ncart(am[2])


def _ksh_rows(kern: Kernel) -> int:
    (sh,) = [s for s in walk(kern.body)
             if isinstance(s, Decl) and s.name == KSH]
    return sh.shape[0].value


def _notes(kern: Kernel) -> list[str]:
    return [s.prov for s in walk(kern.body) if isinstance(s, Note)]


# ---------------------------------------------------------------------------
# 1. The declaration
# ---------------------------------------------------------------------------


def test_interleave_is_declared_legal_on_k_only_with_its_own_reasons() -> None:
    lever = levers.VOCABULARY[interleave.NAME]
    assert (lever.name, lever.issue, lever.legal) == ("interleave", 343,
                                                      frozenset({"k"}))
    assert "stages no I column" in lever.refusals["j"]
    assert "#340" in lever.refusals["f"]
    for slot in ("dq", "jab", "s", "t", "v"):
        assert lever.refusals[slot] == levers.DEFAULT_REFUSALS[slot]
    with pytest.raises(ValueError, match="stages no I column"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J, (interleave.NAME,))


def test_interleave_is_registered_keyed_and_absent_from_the_literal() -> None:
    assert k_levers.REGISTRY[interleave.NAME] is interleave.apply
    spec = lever_spec((1, 0, 1, 0))
    assert spec.key() == "coulomb/d0/(1,0|1,0)/b1/k/interleave"
    assert spec.kernel_body_levers() == (interleave.NAME,)
    assert k_levers.hooks_for(literal_spec((1, 0, 1, 0))) is DEFAULT_HOOKS
    for am in FAST:
        assert interleave.TRADE not in _notes(build_k_kernel(am))
    # one kernel per batch, under the literal kernel's name
    (kern,) = build_k_kernels(spec)
    assert kern.name == build_k_kernel((1, 0, 1, 0)).name == "k_10_10"
    assert build_k_kernel(spec) == kern


def test_the_hooks_it_moves() -> None:
    spec = lever_spec((1, 0, 1, 0))
    hooks = interleave.apply(DEFAULT_HOOKS, spec)
    moved = {f.name for f in dataclasses.fields(KHooks)
             if not hooks.is_default(f.name)}
    assert moved == {"contract", "shared_decl", "init_acc", "write_back"}
    assert hooks.contract is interleave.contract
    # keep / fan_out / finish untouched: #341's split composes (#348)
    assert {"keep", "fan_out", "finish"}.isdisjoint(moved)


# ---------------------------------------------------------------------------
# 2. The shape: K-only Ksh, I_p in registers, the program untouched
# ---------------------------------------------------------------------------


def _ket_loop(kern: Kernel) -> tuple:
    (loop,) = [s for s in walk(kern.body) if isinstance(s, For)
               and s.var == KET]
    return loop.body


def _program(kern: Kernel) -> tuple:
    """The ket loop up to and including the `dk` row: everything before the
    contraction region."""
    body = _ket_loop(kern)
    end = next(i for i, s in enumerate(body)
               if isinstance(s, Decl) and s.name == "dk")
    return body[:end + 1]


@pytest.mark.parametrize("am", FAST_IDS)
def test_ksh_is_the_k_slots_only_and_no_i_row_reaches_it(am: Am) -> None:
    literal = build_k_kernel(am)
    (kern,) = build_k_kernels(lever_spec(am))
    assert _ksh_rows(kern) == _n(am)
    assert _ksh_rows(literal) > _n(am)  # the literal's I column, then K
    # every Ksh write in the ket loop is a K digest: Ksh[s] = Ksh[s] + D V
    for s in _ket_loop(kern):
        if isinstance(s, Assign) and isinstance(s.dst, Index):
            assert s.dst.array == KSH
            assert s.dst.at[0].value < _n(am)
            (_, prod) = s.value.args
            (_, v) = prod.args
            assert isinstance(v, Var) and v.name.startswith("V")


@pytest.mark.parametrize("am", FAST_IDS)
def test_each_i_p_is_formed_once_and_folded_into_a_v(am: Am) -> None:
    (kern,) = build_k_kernels(lever_spec(am))
    body = _ket_loop(kern)
    i_decls = [s.name for s in body if isinstance(s, Decl)
               and s.name.startswith("I") and s.init is not None]
    assert i_decls and len(i_decls) == len(set(i_decls))
    folded = {e.name for s in body
              if isinstance(s, Assign) and isinstance(s.dst, Var)
              for e in _vars(s.value)}
    assert set(i_decls) <= folded
    # the program before the contraction is the literal kernel's
    assert _program(kern) == _program(build_k_kernel(am))
    notes = _notes(kern)
    assert levers.provenance(interleave.NAME) in notes
    assert interleave.TRADE in notes
    assert not any(n.startswith("per (b,d) row: stage 1") for n in notes)


def _vars(e) -> list[Var]:
    out = [e] if isinstance(e, Var) else []
    for a in getattr(e, "args", ()):
        out += _vars(a)
    return out


# ---------------------------------------------------------------------------
# 3. Parity: against ao.k.k_quartet per bra record, and the literal kernel
# ---------------------------------------------------------------------------


def run(kernels: tuple[Kernel, ...], am: Am, *, zero: bool,
        contracted: bool = True, shift: float = 0.0) -> np.ndarray:
    """The kernels in order onto one `K`, one lane, test_kernel_k's shells
    (NaN-filled unless `zero`: then every unwritten entry would show)."""
    bra, ket = tk._pairs(am, contracted=contracted)
    args = tk.k_args(am, bra, [(ket, 0)], [tk._dcd(am[1], am[3], shift)])
    if zero:
        args["K"] = np.zeros_like(args["K"])
    grid, block = launch_shape(args["n_bra"], 1)
    for kern in kernels:
        args["K"] = Interpreter(kern, launch=Launch(grid, block, 1),
                                boys=boys).run(**args)["K"]
    return args["K"]


@pytest.mark.parametrize("am", CLASSES)
def test_each_bra_record_is_the_host_staged_quartet_bitwise(am: Am) -> None:
    """The issue's acceptance: host-shim parity with the literal kernel."""
    bra, ket = tk._pairs(am)
    dcd = tk._dcd(am[1], am[3])
    K = run(build_k_kernels(lever_spec(am)), am, zero=False)
    assert not np.isnan(K).any()
    for i in range(len(bra.recs)):
        assert K[i].tolist() == tk._host(tk._one(bra, i), ket, dcd, am), i
    assert np.any(K != 0.0)
    assert K.tolist() == run((build_k_kernel(am),), am, zero=False).tolist()


@pytest.mark.parametrize("am", FAST_IDS)
def test_an_uncontracted_bra_is_the_whole_quartet_bitwise(am: Am) -> None:
    bra, ket = tk._pairs(am, contracted=False)
    dcd = tk._dcd(am[1], am[3], shift=-0.2)
    K = run(build_k_kernels(lever_spec(am)), am, zero=False,
            contracted=False, shift=-0.2)
    assert K[0].tolist() == tk._host(bra, ket, dcd, am)


@pytest.mark.parametrize("lanes", [2, 4])
def test_more_lanes_match_the_literal_at_the_same_lanes(lanes: int) -> None:
    """Strip-mined over lanes, the warp reduction is the literal one: the
    lever's K at `lanes` lanes is the literal kernel's at `lanes` lanes."""
    am = (1, 1, 1, 0)
    bra, ket = tk._pairs(am)
    kets = [(ket, 0), (ket, 0), (ket, 0)]
    dcds = [tk._dcd(am[1], am[3])]

    def k_of(kern: Kernel) -> list:
        args = tk.k_args(am, bra, kets, dcds)
        grid, block = launch_shape(args["n_bra"], lanes)
        return Interpreter(kern, launch=Launch(grid, block, lanes),
                           boys=boys).run(**args)["K"].tolist()

    (kern,) = build_k_kernels(lever_spec(am))
    assert k_of(kern) == k_of(build_k_kernel(am))


SPLIT3 = ([pytest.param(a, id=tk._id(a)) for a in FAST if sum(a) > 0]
          + [pytest.param(a, id=tk._id(a), marks=pytest.mark.slow)
             for a in SLOW_SAMPLE])


@pytest.mark.parametrize("am", SPLIT3)
def test_with_a_batch_split_it_is_the_literal_batched_kernels_bitwise(
        am: Am) -> None:
    kernels = build_k_kernels(lever_spec(am, 3))
    literal = build_k_kernels(literal_spec(am, 3))
    assert len(kernels) == len(literal) == 3
    for kern, lit in zip(kernels, literal, strict=True):
        assert kern.name == lit.name
        assert any(isinstance(s, AtomicAdd) for s in walk(kern.body))
        assert _ksh_rows(kern) < _ksh_rows(lit)
    assert (run(kernels, am, zero=True).tolist()
            == run(literal, am, zero=True).tolist())


# ---------------------------------------------------------------------------
# 4. Composition with #341's split (the shape #348 builds on)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("am", [pytest.param(a, id=tk._id(a))
                                for a in [(1, 0, 1, 0), (1, 1, 0, 0),
                                          (0, 1, 1, 0), (2, 0, 0, 0)]])
@pytest.mark.parametrize("n_batches", [1, 3])
def test_interleave_composes_with_split(am: Am, n_batches: int) -> None:
    both = build_k_kernels(lever_spec(am, n_batches, split.NAME))
    alone = build_k_kernels(Spec(Operator.COULOMB, 0, am, Target.K,
                                 (split.NAME,), n_batches))
    assert [k.name for k in both] == [k.name for k in alone]
    for kern in both:
        notes = _notes(kern)
        assert interleave.TRADE in notes and split.TRADE in notes
        assert _ksh_rows(kern) == 1  # one K slot, no I row
    assert (run(both, am, zero=True).tolist()
            == run(alone, am, zero=True).tolist())


# ---------------------------------------------------------------------------
# 5. The goldens: (p s|p s) interleaved, one batch, per backend
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.spec import Spec, Operator, Target; \
#   from felizjueves.kernel_k import build_k_kernels; \
#   from felizjueves.performance.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   [(Path(f'test/golden/cuda/k_interleave/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)), Path(f'test/golden/hip/k_interleave/{k.name}.hip' \
#   ).write_text(hip.emit(k, GFX1200))) for k in build_k_kernels(Spec( \
#   Operator.COULOMB, 0, (1, 0, 1, 0), Target.K, ('interleave',)))]"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_AM: Am = (1, 0, 1, 0)
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_interleave_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    kernels = build_k_kernels(lever_spec(GOLDEN_AM))
    where = GOLDEN / backend / "k_interleave"
    assert sorted(p.name for p in where.iterdir()) == sorted(
        f"{k.name}.{ext}" for k in kernels)
    for kern in kernels:
        assert emit(kern, arch) == (where / f"{kern.name}.{ext}").read_text()
