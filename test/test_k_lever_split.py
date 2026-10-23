"""The per-element split K lever, `k_levers/split.py` (#341), interpreted.

The lever fans each batch's K kernel out into `n_a * n_b` element kernels,
one per `K[a_i, b_j]`, each carrying that one accumulator. What is asserted:

1. THE DECLARATION: `split` is legal on `Target.K` only, refused on the
   other targets with a stated reason, keyed as any lever is, registered in
   `k_levers.REGISTRY`, and absent from every literal build (the literal K
   kernels' digest is test_k_seams.py's pin, unchanged).
2. THE TILING: `n_a * n_b` kernels at one batch, named `..._e{i}_{j}`, each
   writing exactly its own element, together every element once; each holds
   ONE accumulator slot after the `I` rows its element reads; each carries
   the lever's provenance and the trade's `Note`; and each re-runs the whole
   `[p|q]` program (the trade, stated: the literal kernel's recurrence,
   statement for statement, in every element kernel).
3. EXACTNESS, against `ao.k.k_quartet` (the issue's acceptance): every
   element kernel performs the literal kernel's operations for its element in
   the literal order, so per bra record the element kernels' union is
   `k_quartet` BIT FOR BIT, and `==` the literal block kernel -- every class
   with L <= 2 (fast), and the classes through L = 4 with no shell above d
   (`slow`, milestone 13's cap). With a 3-batch split, each element kernel is
   atomized and the class's kernels onto a zeroed `K` are the literal
   batched kernels' `K` bit for bit (each element sees its batches' partials
   in the literal batch order); an element a batch never reaches is dropped,
   not emitted empty.
4. COMPOSITION, for #348: `split` wraps whatever `fan_out` it is handed and
   moves only `keep`, `fan_out` and `init_acc`, so a lever applied before it
   that moves the contraction is seen in every element kernel.

Fast tier unless `slow`: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
import itertools
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
    walk,
)
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.k_levers import split
from felizjueves.k_levers.seams import DEFAULT_HOOKS, KSH, KHooks, ncart
from felizjueves.kernel_k import (
    KET,
    build_k_kernel,
    build_k_kernels,
    k_class_spec,
    launch_shape,
)
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.reference.boys import boys
from felizjueves.spec import Operator, Spec, Target

Am = tuple[int, int, int, int]

FAST: list[Am] = tk.FAST  # every K class with L <= 2
# Above L = 2, within milestone 13's cap of L = 4, no shell above d: `slow`.
WIDE: list[Am] = [am for am in itertools.product(range(3), repeat=4)
                  if 2 < sum(am) <= 4]
SLOW_SAMPLE: list[Am] = [(1, 1, 1, 0), (1, 0, 1, 1), (1, 1, 1, 1),
                         (2, 0, 2, 0), (2, 2, 0, 0), (0, 0, 2, 2),
                         (2, 1, 1, 0), (0, 2, 1, 1)]
CLASSES = ([pytest.param(am, id=tk._id(am)) for am in FAST]
           + [pytest.param(am, id=tk._id(am), marks=pytest.mark.slow)
              for am in SLOW_SAMPLE])


def lever_spec(am: Am, n_batches: int = 1) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.K, (split.NAME,), n_batches)


def literal_spec(am: Am, n_batches: int = 1) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.K, (), n_batches)


def _n(am: Am) -> int:
    return ncart(am[0]) * ncart(am[2])


def _writes(kern: Kernel) -> set[int]:
    """The flat `K[a, b]` indices the kernel writes (plain or atomic)."""
    out = set()
    for s in walk(kern.body):
        tgt = s.dst if isinstance(s, Assign | AtomicAdd) else None
        if isinstance(tgt, Index) and tgt.array == "K":
            (_, slot) = tgt.at
            out.add(slot.value)
    return out


# ---------------------------------------------------------------------------
# 1. The declaration
# ---------------------------------------------------------------------------


def test_split_is_declared_legal_on_k_only_with_its_own_reasons() -> None:
    lever = levers.VOCABULARY[split.NAME]
    assert (lever.name, lever.issue, lever.legal) == ("split", 341,
                                                      frozenset({"k"}))
    assert "K[a, b]" in lever.refusals["j"]
    assert "#340" in lever.refusals["f"]
    for slot in ("dq", "jab", "s", "t", "v"):
        assert lever.refusals[slot] == levers.DEFAULT_REFUSALS[slot]
    with pytest.raises(ValueError, match="no Cartesian element block"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J, (split.NAME,))


def test_split_is_registered_keyed_and_absent_from_the_literal() -> None:
    assert k_levers.REGISTRY[split.NAME] is split.apply
    spec = lever_spec((1, 0, 1, 0))
    assert spec.key() == "coulomb/d0/(1,0|1,0)/b1/k/split"
    assert spec.kernel_body_levers() == (split.NAME,)
    assert k_levers.hooks_for(spec) != DEFAULT_HOOKS
    assert k_levers.hooks_for(literal_spec((1, 0, 1, 0))) is DEFAULT_HOOKS
    # the element kernels' common prefix is the literal kernel's name
    assert build_k_kernel((1, 0, 1, 0)).name == "k_10_10"
    with pytest.raises(ValueError, match="fans out into 9 kernels"):
        build_k_kernel(spec)


def test_the_interface_348_builds_on() -> None:
    spec = lever_spec((2, 0, 1, 0))
    assert split.elements(spec) == tuple(
        (i, j) for i in range(6) for j in range(3))
    assert split.suffix(4, 2) == "_e4_2"
    keep = split.keep_element(4, 2)
    assert keep(spec, 4, 2) and not keep(spec, 2, 4)
    moved = {f.name for f in dataclasses.fields(KHooks)
             if not split.apply(DEFAULT_HOOKS, spec).is_default(f.name)}
    assert moved == {"fan_out"}
    ((_, h), *_) = split.apply(DEFAULT_HOOKS, spec).fan_out(spec,
                                                            DEFAULT_HOOKS)
    per_variant = {f.name for f in dataclasses.fields(KHooks)
                   if not h.is_default(f.name)}
    assert per_variant == {"keep", "init_acc", "finish"}


# ---------------------------------------------------------------------------
# 2. The tiling, the footprint, the notes, the trade
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("am", [pytest.param(a, id=tk._id(a)) for a in FAST])
def test_n_a_n_b_kernels_each_writing_its_own_element(am: Am) -> None:
    kernels = build_k_kernels(lever_spec(am))
    la, lc, lb, ld = am
    assert [k.name for k in kernels] == [
        f"k_{la}{lc}_{lb}{ld}_e{i}_{j}"
        for i in range(ncart(la)) for j in range(ncart(lb))]
    nb = ncart(lb)
    for kern, (i, j) in zip(kernels, split.elements(lever_spec(am)),
                            strict=True):
        assert _writes(kern) == {i * nb + j}
        assert not any(isinstance(s, AtomicAdd) for s in walk(kern.body))


@pytest.mark.parametrize("am", [pytest.param(a, id=tk._id(a)) for a in FAST])
def test_one_accumulator_slot_and_the_notes(am: Am) -> None:
    literal = build_k_kernel(am)
    (lit_sh,) = [s for s in walk(literal.body)
                 if isinstance(s, Decl) and s.name == KSH]
    n_p = lit_sh.shape[0].value - _n(am)
    for kern in build_k_kernels(lever_spec(am)):
        (sh,) = [s for s in walk(kern.body)
                 if isinstance(s, Decl) and s.name == KSH]
        assert 2 <= sh.shape[0].value <= n_p + 1  # >= one I row, one K slot
        notes = [s.prov for s in walk(kern.body) if isinstance(s, Note)]
        assert levers.provenance(split.NAME) in notes
        assert split.TRADE in notes
        assert split.PLAIN_WRITE in notes  # its element, not its row, of K
        assert not any("its row of K" in n for n in notes)


def _ket_program(kern: Kernel) -> tuple:
    """The ket loop's `[p|q]` program: everything before the E^bd binding."""
    (loop,) = [s for s in walk(kern.body) if isinstance(s, For)
               and s.var == KET]
    end = next(i for i, s in enumerate(loop.body)
               if isinstance(s, Note) and s.prov.startswith("the undressed E^bd"))
    return loop.body[:end]


def _subsequence(sub: tuple, full: tuple) -> bool:
    it = iter(full)
    return all(any(s == f for f in it) for s in sub)


@pytest.mark.parametrize("am", [pytest.param(a, id=tk._id(a)) for a in FAST])
def test_every_element_kernel_re_runs_the_recurrence_its_element_reads(
        am: Am) -> None:
    """The trade the issue says is reported, not hidden: the recurrence is
    not split, it is repeated -- each element kernel runs the literal
    program less only the nodes its element never reads, in the literal
    order, statement for statement; together they need all of it, and the
    top node `[r]^(0)` with `|r| = L` is in every one of them."""
    program = _ket_program(build_k_kernel(am))
    kernels = build_k_kernels(lever_spec(am))
    assert len(kernels) == _n(am)
    seen: set = set()
    for kern in kernels:
        mine = _ket_program(kern)
        assert _subsequence(mine, program)
        seen |= set(mine)
        assert sum(isinstance(s, Decl) and s.name.startswith("R")
                   for s in mine) >= 1
    assert seen == set(program)


@pytest.mark.parametrize("am", [pytest.param(a, id=tk._id(a)) for a in FAST]
                         + [pytest.param(a, id=tk._id(a),
                                         marks=pytest.mark.slow)
                            for a in SLOW_SAMPLE])
def test_prune_leaves_the_literal_kernel_alone(am: Am) -> None:
    """The literal kernel declares nothing it does not read, so `prune` is
    the identity on it: it removes only what a narrowed `keep` left dead."""
    assert split.prune(build_k_kernel(am)) == build_k_kernel(am)
    for spec in (literal_spec(am, 3),) if sum(am) else ():
        for kern in build_k_kernels(spec):
            assert split.prune(kern) == kern


# ---------------------------------------------------------------------------
# 3. Exactness: against ao.k.k_quartet per bra record, and the literal block
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
    """The issue's acceptance: the split kernel set reproduces `kernel_k`'s
    block against `ao.k.k_quartet`, per bra record, bit for bit."""
    bra, ket = tk._pairs(am)
    dcd = tk._dcd(am[1], am[3])
    K = run(build_k_kernels(lever_spec(am)), am, zero=False)
    assert not np.isnan(K).any()  # every element written
    for i in range(len(bra.recs)):
        assert K[i].tolist() == tk._host(tk._one(bra, i), ket, dcd, am), i
    assert np.any(K != 0.0)
    assert K.tolist() == run((build_k_kernel(am),), am, zero=False).tolist()


@pytest.mark.parametrize("am", [pytest.param(a, id=tk._id(a)) for a in FAST])
def test_an_uncontracted_bra_is_the_whole_quartet_bitwise(am: Am) -> None:
    bra, ket = tk._pairs(am, contracted=False)
    dcd = tk._dcd(am[1], am[3], shift=-0.2)
    K = run(build_k_kernels(lever_spec(am)), am, zero=False,
            contracted=False, shift=-0.2)
    assert K[0].tolist() == tk._host(bra, ket, dcd, am)


SPLIT3 = ([pytest.param(a, id=tk._id(a)) for a in FAST if sum(a) > 0]
          + [pytest.param(a, id=tk._id(a), marks=pytest.mark.slow)
             for a in SLOW_SAMPLE])


@pytest.mark.parametrize("am", SPLIT3)
def test_with_a_batch_split_it_is_the_literal_batched_kernels_bitwise(
        am: Am) -> None:
    kernels = build_k_kernels(lever_spec(am, 3))
    literal = build_k_kernels(literal_spec(am, 3))
    assert len(literal) == 3
    assert _n(am) <= len(kernels) <= 3 * _n(am)
    assert all(any(isinstance(s, AtomicAdd) for s in walk(k.body))
               for k in kernels)
    # every element is written by at least one batch, and no kernel is empty
    assert set().union(*map(_writes, kernels)) == set(range(_n(am)))
    assert all(len(_writes(k)) == 1 for k in kernels)
    assert (run(kernels, am, zero=True).tolist()
            == run(literal, am, zero=True).tolist())


def test_an_element_a_batch_never_reaches_is_dropped() -> None:
    """(p s|p s) at 3 batches: 19 element kernels, not 27 -- a batch whose
    targets reach no `p` some element reads emits nothing for it."""
    kernels = build_k_kernels(lever_spec((1, 0, 1, 0), 3))
    assert len(kernels) == 19
    names = [k.name for k in kernels]
    assert len(set(names)) == len(names)
    assert all(n.startswith("k_10_10_") and "_e" in n for n in names)


@pytest.mark.slow
@pytest.mark.parametrize("am", [pytest.param(a, id=tk._id(a)) for a in WIDE])
def test_every_class_through_l4_builds_and_tiles(am: Am) -> None:
    """Every K class with 2 < L <= 4 and no shell above d: the tiling holds
    (built, not interpreted)."""
    kernels = build_k_kernels(lever_spec(am))
    nb = ncart(am[2])
    assert [_writes(k) for k in kernels] == [
        {i * nb + j} for i, j in split.elements(lever_spec(am))]


# ---------------------------------------------------------------------------
# 4. Composition: a lever applied before split is seen in every element
# ---------------------------------------------------------------------------


PROBE = "aa-probe"  # sorts before "split": the key order applies it first


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setitem(levers.VOCABULARY, PROBE, levers.Lever.of(
        PROBE, 341, "a probe: marks the contraction and fans out in two",
        {"k"}))

    def apply(h: KHooks, spec: Spec) -> KHooks:
        prior = h.contract

        def contract(ctx):
            return (*prior(ctx), Note("PROBE"))

        def fan_out(spec, hh):
            return (("_u", hh), ("_v", hh))
        return dataclasses.replace(h, contract=contract, fan_out=fan_out)

    monkeypatch.setitem(k_levers.REGISTRY, PROBE, apply)
    return PROBE


def test_split_composes_with_a_lever_applied_before_it(probe: str) -> None:
    am = (1, 0, 1, 0)
    spec = Spec(Operator.COULOMB, 0, am, Target.K, (probe, split.NAME))
    kernels = build_k_kernels(spec)
    assert [k.name for k in kernels] == [
        f"k_10_10_{v}_e{i}_{j}" for v in "uv"
        for i in range(3) for j in range(3)]
    for kern in kernels:
        notes = [s.prov for s in walk(kern.body) if isinstance(s, Note)]
        assert "PROBE" in notes and split.TRADE in notes
        assert len(_writes(kern)) == 1
    # the probe moves no arithmetic: its _u half is the split block, bitwise
    assert (run(kernels[:9], am, zero=False).tolist()
            == run(build_k_kernels(lever_spec(am)), am, zero=False).tolist())


def test_the_one_batch_literal_is_untouched() -> None:
    for am in FAST:
        assert build_k_kernels(k_class_spec(*am)) == (build_k_kernel(am),)
        assert not any(isinstance(s, Note) and s.prov == split.TRADE
                       for s in walk(build_k_kernel(am).body))


# ---------------------------------------------------------------------------
# 5. The goldens: (p s|s s) split, three element kernels per backend
# ---------------------------------------------------------------------------
#
# One class, one batch: the emitted text of its three element kernels, so a
# change in the lever's text is a reviewed diff. Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.spec import Spec, Operator, Target; \
#   from felizjueves.kernel_k import build_k_kernels; \
#   from felizjueves.performance.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   [(Path(f'test/golden/cuda/k_split/{k.name}.cu').write_text(cuda.emit(k, \
#   SM_86)), Path(f'test/golden/hip/k_split/{k.name}.hip').write_text( \
#   hip.emit(k, GFX1200))) for k in build_k_kernels(Spec(Operator.COULOMB, \
#   0, (1, 0, 0, 0), Target.K, ('split',)))]"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_AM: Am = (1, 0, 0, 0)
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_split_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    kernels = build_k_kernels(lever_spec(GOLDEN_AM))
    where = GOLDEN / backend / "k_split"
    assert sorted(p.name for p in where.iterdir()) == sorted(
        f"{k.name}.{ext}" for k in kernels)
    for kern in kernels:
        assert emit(kern, arch) == (where / f"{kern.name}.{ext}").read_text()
