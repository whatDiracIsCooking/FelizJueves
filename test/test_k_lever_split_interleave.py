"""Interleaved `I_p`, the split arm (#348): `interleave` on `split`'s element kernels.

THE DECISION. The split arm is NOT a new lever. It is the COMPOSITION of the
two existing ones, `("interleave", "split")` in one spec (key
`.../k/interleave+split`): `k_levers.hooks_for` applies #343's `interleave`
(it moves `contract` and re-slots `shared_decl` / `init_acc` /
`write_back`) and then #341's `split` (it moves `fan_out`, and per variant
`keep`, `init_acc`, `finish`), and the two touch disjoint hooks. A third
vocabulary entry would name one behaviour twice. What the composition adds
is one `Note`, `interleave.SPLIT_ARM`, in each element kernel's region
(`interleave.arm(spec)`), so the arm is legible in the generated text; no
arithmetic.

What is asserted (host shim, fast unless `slow`; no compiler, no card):

1. THE VOCABULARY stays coherent: no entry or registry line for the arm, the
   key is the two levers', the arm is named by `interleave.arm`, and the
   composition is the same kernels in either application order.
2. THE SHAPE: every element kernel holds ONE `K` slot in `Ksh` and no `I`
   row (the split cuts the accumulator set, the interleave the live `I`
   set), forms its `I_p` in locals, carries both levers' notes and the
   arm's; the other three arms carry no `SPLIT_ARM`.
3. FOUR-ARM PARITY, the issue's acceptance: literal / interleave / split /
   split+interleave, per bra record `ao.k.k_quartet` BIT FOR BIT and equal
   to each other, every class with L <= 2 at one batch and, atomized onto a
   zeroed `K`, at three; a sample through L = 4 (no shell above d) `slow`,
   milestone 13's cap. Bitwise because neither lever reassociates (their
   module docstrings): no tolerance is involved.
4. THE GOLDENS: `(p s|s s)`, one batch, per backend, the same class as
   test_k_lever_split.py's, so a diff of the two directories is the
   interleave.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import test_kernel_k as tk

from felizjueves import levers
from felizjueves.emit import cuda, hip
from felizjueves.eri.k_exchange import levers as k_levers
from felizjueves.eri.k_exchange.kernel_k import build_k_kernels, launch_shape
from felizjueves.eri.k_exchange.levers import interleave, split
from felizjueves.eri.k_exchange.levers.seams import DEFAULT_HOOKS, KSH
from felizjueves.ir import Decl, Kernel, Note, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.reference.boys import boys
from felizjueves.spec import Operator, Spec, Target

Am = tuple[int, int, int, int]

BOTH = tuple(sorted((interleave.NAME, split.NAME)))
ARMS: dict[str, tuple[str, ...]] = {
    "literal": (),
    "interleave": (interleave.NAME,),
    "split": (split.NAME,),
    "split+interleave": BOTH,
}

FAST: list[Am] = tk.FAST  # every K class with L <= 2
SLOW_SAMPLE: list[Am] = [(1, 1, 1, 0), (1, 0, 1, 1), (1, 1, 1, 1),
                         (2, 0, 2, 0), (2, 2, 0, 0), (0, 0, 2, 2),
                         (2, 1, 1, 0), (0, 2, 1, 1)]
CLASSES = ([pytest.param(am, id=tk._id(am)) for am in FAST]
           + [pytest.param(am, id=tk._id(am), marks=pytest.mark.slow)
              for am in SLOW_SAMPLE])
SPLIT3 = ([pytest.param(a, id=tk._id(a)) for a in FAST if sum(a) > 0]
          + [pytest.param(a, id=tk._id(a), marks=pytest.mark.slow)
             for a in SLOW_SAMPLE])
FAST_IDS = [pytest.param(a, id=tk._id(a)) for a in FAST]


def arm_spec(am: Am, arm: str, n_batches: int = 1) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.K, ARMS[arm], n_batches)


def _notes(kern: Kernel) -> list[str]:
    return [s.prov for s in walk(kern.body) if isinstance(s, Note)]


def _ksh_rows(kern: Kernel) -> int:
    (sh,) = [s for s in walk(kern.body)
             if isinstance(s, Decl) and s.name == KSH]
    return sh.shape[0].value


def run(kernels: tuple[Kernel, ...], am: Am, *, zero: bool) -> np.ndarray:
    """The kernels in order onto one `K`, one lane, test_kernel_k's shells
    (NaN-filled unless `zero`: then every unwritten entry would show)."""
    bra, ket = tk._pairs(am)
    args = tk.k_args(am, bra, [(ket, 0)], [tk._dcd(am[1], am[3])])
    if zero:
        args["K"] = np.zeros_like(args["K"])
    grid, block = launch_shape(args["n_bra"], 1)
    for kern in kernels:
        args["K"] = Interpreter(kern, launch=Launch(grid, block, 1),
                                boys=boys).run(**args)["K"]
    return args["K"]


# ---------------------------------------------------------------------------
# 1. The vocabulary: a composition, not a lever
# ---------------------------------------------------------------------------


def test_the_split_arm_is_a_composition_not_a_lever() -> None:
    k_names = {n for n, lv in levers.VOCABULARY.items() if "k" in lv.legal}
    assert {interleave.NAME, split.NAME} <= k_names
    # no third name for the composed behaviour, in either table
    for table in (levers.VOCABULARY, k_levers.REGISTRY):
        assert not [n for n in table if n not in (interleave.NAME, split.NAME)
                    and ("interleave" in n or "split" in n)]
    spec = arm_spec((1, 0, 1, 0), "split+interleave")
    assert spec.key() == "coulomb/d0/(1,0|1,0)/b1/k/interleave+split"
    assert spec.kernel_body_levers() == BOTH
    assert [interleave.arm(arm_spec((1, 0, 1, 0), a)) for a in ARMS] == [
        interleave.WARP, interleave.WARP, interleave.SPLIT, interleave.SPLIT]


@pytest.mark.parametrize("n_batches", [1, 3])
def test_either_application_order_is_the_same_kernels(n_batches: int) -> None:
    spec = arm_spec((1, 1, 1, 0), "split+interleave", n_batches)
    i_then_s = split.apply(interleave.apply(DEFAULT_HOOKS, spec), spec)
    s_then_i = interleave.apply(split.apply(DEFAULT_HOOKS, spec), spec)
    want = build_k_kernels(spec)  # hooks_for: key order, interleave first
    assert build_k_kernels(spec, hooks=i_then_s) == want
    assert build_k_kernels(spec, hooks=s_then_i) == want


# ---------------------------------------------------------------------------
# 2. The shape: one K slot, no I row, I_p in locals, the arm's note
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("am", FAST_IDS)
def test_each_element_kernel_holds_one_k_slot_and_no_i_row(am: Am) -> None:
    both = build_k_kernels(arm_spec(am, "split+interleave"))
    alone = build_k_kernels(arm_spec(am, "split"))
    assert [k.name for k in both] == [k.name for k in alone]
    for kb, ks in zip(both, alone, strict=True):
        assert _ksh_rows(kb) == 1  # the split's one slot, the I rows gone
        assert _ksh_rows(ks) >= 1  # the split alone keeps its I rows
        notes = _notes(kb)
        for want in (split.TRADE, interleave.TRADE, interleave.SPLIT_ARM,
                     levers.provenance(split.NAME),
                     levers.provenance(interleave.NAME)):
            assert want in notes
        assert any(isinstance(s, Decl) and s.name.startswith("I")
                   and s.init is not None for s in walk(kb.body))


@pytest.mark.parametrize("arm", ["literal", "interleave", "split"])
def test_only_the_composition_carries_the_arm_note(arm: str) -> None:
    for kern in build_k_kernels(arm_spec((1, 0, 1, 0), arm)):
        assert interleave.SPLIT_ARM not in _notes(kern)


# ---------------------------------------------------------------------------
# 3. Four-arm parity: k_quartet per bra record, and the arms against each other
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("am", CLASSES)
def test_four_arms_are_the_host_quartet_bitwise(am: Am) -> None:
    bra, ket = tk._pairs(am)
    dcd = tk._dcd(am[1], am[3])
    got = {a: run(build_k_kernels(arm_spec(am, a)), am, zero=False)
           for a in ARMS}
    for a, K in got.items():
        assert not np.isnan(K).any(), a
        for i in range(len(bra.recs)):
            assert K[i].tolist() == tk._host(tk._one(bra, i), ket, dcd, am), (
                a, i)
    assert np.any(got["literal"] != 0.0)
    want = got["literal"].tolist()
    assert all(K.tolist() == want for K in got.values())


@pytest.mark.parametrize("am", SPLIT3)
def test_four_arms_with_a_batch_split_are_bitwise_equal(am: Am) -> None:
    got = {a: run(build_k_kernels(arm_spec(am, a, 3)), am, zero=True)
           for a in ARMS}
    want = got["literal"].tolist()
    assert np.any(got["literal"] != 0.0)
    for a, K in got.items():
        assert K.tolist() == want, a
    # the composed batched cell is split's element-by-batch tiling
    both = build_k_kernels(arm_spec(am, "split+interleave", 3))
    alone = build_k_kernels(arm_spec(am, "split", 3))
    assert [k.name for k in both] == [k.name for k in alone]


# ---------------------------------------------------------------------------
# 4. The goldens: (p s|s s), split+interleave, one batch, per backend
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.spec import Spec, Operator, Target; \
#   from felizjueves.eri.k_exchange.kernel_k import build_k_kernels; \
#   from felizjueves.performance.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   [(Path(f'test/golden/cuda/k_split_interleave/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)), Path(f'test/golden/hip/k_split_interleave/' \
#   f'{k.name}.hip').write_text(hip.emit(k, GFX1200))) for k in \
#   build_k_kernels(Spec(Operator.COULOMB, 0, (1, 0, 0, 0), Target.K, \
#   ('interleave', 'split')))]"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_AM: Am = (1, 0, 0, 0)
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_split_interleave_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    kernels = build_k_kernels(arm_spec(GOLDEN_AM, "split+interleave"))
    where = GOLDEN / backend / "k_split_interleave"
    assert sorted(p.name for p in where.iterdir()) == sorted(
        f"{k.name}.{ext}" for k in kernels)
    for kern in kernels:
        assert emit(kern, arch) == (where / f"{kern.name}.{ext}").read_text()
