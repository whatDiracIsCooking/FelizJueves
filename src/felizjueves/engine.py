"""`Engine`: what BOTH ERI engines turned out to have (#292, milestone 12 G1).

    Engine                      the protocol: name, target, eri, eri_and_scale,
                                kernel
    Mmd(pq=...)                 engine #1, McMurchie-Davidson ([JEngine])
    Hgp(boys=...)               engine #2, Head-Gordon-Pople ([Fock] §3)
    engines(*, pq, boys)        both, in that order -- what a cross-engine
                                check parametrizes over
    MEMBERS                     each member, with BOTH implementations named

EXTRACTED LAST, FROM TWO IMPLEMENTATIONS (milestone 12, pin 12). The protocol
was not written before the HGP engine existed: one written in wave 0 would
have steered `hgp/` toward the MMD shape it exists to test. So this module
is the RESIDUE -- the members both engines already had when HGP was done --
and the finding is what did NOT survive, below.

THE MEMBERS, each justified by both implementations having it (`MEMBERS`
names the two callables; test/test_engine.py resolves every one):

  `name`, `target`   Each engine builds kernels for exactly one `Target`:
                     MMD's `kernel.build_pq_kernel` is the `Target.J` [p|q]
                     kernel, and HGP's `kernel_f.build_f_kernel` refuses
                     anything but `Target.F` (`hgp.recurrence.hgp_class`).
                     A caller needs the target to form a `Spec` the engine
                     will accept, and nothing else about the engine.
  `eri(bra, ket)`    `(ab|cd)` of a contracted, ordered shell quartet, rows
                     `ab`, columns `cd`, first index outer: `ao.eri.eri_block`
                     (MMD, the Hermite path) and `ao.hgp_drive.eri_block`
                     (HGP, #281, built with that signature on purpose). It
                     is exactly `ao.fock`'s injected `eri=` slot.
  `eri_and_scale`    the block and its absolute-value twin, the yardstick a
                     rounding tolerance is stated against:
                     `ao.eri.eri_block_and_scale` and
                     `ao.hgp_drive.eri_block_and_scale`. NOT in #292's
                     candidate surface, and admitted on its rule: both
                     engines have it, with one signature, and without it no
                     cross-engine check can state its tolerance generically
                     (each engine's scale sees its own cancellations: the
                     HRR's transfers are invisible to the MMD twin, #281).
  `kernel(spec)`     `Spec -> ir.Kernel`, vendor-neutral, for the engine's
                     own target: `kernel.build_pq_kernel` and
                     `kernel_f.build_f_kernel`. Its consumer is everything
                     downstream of a `Kernel` -- `emit/` spells any `Kernel`,
                     `interp` runs any `Kernel` -- so the member is the seam
                     the backends already treat as engine-free. A spec for
                     the other engine's target is refused HERE, before the
                     builder sees it (the MMD builder alone would plan a
                     `Target.K` spec as if it were J). The member promises
                     nothing about the kernel's PARAMETERS: the J kernel's
                     contract (`J_p` over Hermite targets, one warp per bra)
                     and F's (`kernel_f.pack`'s records, one thread per
                     contracted quartet, pin 7) are engine-private.

The special function is NOT a member: MMD's `eri` takes the signed Hermite
integral `pq(p, q, PQ, theta) = [p|q]` and HGP's takes the Boys function
`boys(m, T)`. They are different quantities at different depths of the two
recurrences, so each engine takes its own at CONSTRUCTION (`Mmd(pq=)`,
`Hgp(boys=)`), injected as `interp` takes `boys=` -- this module imports no
`reference/` -- and the protocol's calls take shells only.

`pair_bound` -- in #292's candidate surface -- IS NOT A MEMBER, because only
one implementation has it. The pair-factor-free Schwarz bound
(`ao.pairs.pair_bound`, `hermite_bound`, `primitive_schwarz_bound`) is the
MMD engine's Hermite `I_p = sqrt((-1)^|p| [2p]^(0))`, its `[r]^(0)`
evaluator injected. HGP has no bound of its own: `launch_f` is handed
`ao.pairs.primitive_schwarz_bounds` as `bound=`. That is sound -- a bound is
a property of the integral, which both engines compute equal (#281) -- but
it is one implementation, not two, so it stays where it is: in the AO layer,
injected into whichever plan screens.

PUSHED BACK DOWN -- the milestone's primary report (pin 12). Everything the
MMD side has that HGP could not implement is ENGINE-PRIVATE: it is reachable
from the MMD engine's own modules, and from no member of this protocol.

  1. The [JEngine] batch partition and `n_batches` (`plan/batching.py`, the
     minimax split; `kernel.build_pq_kernels`' atomic batch kernels). An
     HGP class has no Hermite targets to partition, so `Target.F` refuses
     `n_batches > 1` for its own reason (spec.py, `_F_BATCH_REFUSAL`, pin 4);
     `kernel(spec)` is the ONE-batch builder on both sides.
  2. MILD (`plan/mild.py`): the intermediate tree is chosen over the
     graded Hermite DAG; the HGP VRR's tree is fixed by eqs 25-26.
  3. The sieve (`plan/sieve.py`, Alg. 3, and its `no-sieve` lever), with the
     `tuned` lever's searched cut and seed orders (`plan/search.py`): they
     edit the MMD DAG, which HGP does not have.
  4. The graded `(r, m)` DAG (`plan/dag.py`): every edge raises `m` by one,
     and the order, the sieve and register recycling rest on it; the HGP
     VRR has same-`m` edges (#273) and its own `(e, f, m)` DAG (`hgp/vrr.py`).
  5. The `total_am`-keyed table row (`Spec.total_am` indexing [JEngine]
     Tables 1-3; `plan.dag.refuse_hgp`): an HGP class has no such row, so
     the plan layer refuses a `Target.F` spec (pin 11).
  6. `plan/cost.py`'s calibrated instrument (`PlanCost`, the quantities the
     literal board's floors are measured with). HGP counts its own FLOPs
     off its own DAGs (`hgp/cost.py`) against no published table (pin 2);
     the two counts are not one instrument and are not compared here.
  7. The thread mapping and the launch plan. Both builders have a
     `launch_shape`, but MMD's takes a warp width (one warp per bra record)
     and HGP's a block size (one block per bra pair, one thread per
     contracted quartet, pin 7): one name, two contracts, so no member. The
     J launch plan (`launch.py`'s `ClassLaunch`, the `I_q max|D_q|` ket
     sort) and the F pair stream (`launch_f.py`) are each their engine's.
  8. The pre- and post-kernels of the J chain (`Target.DQ`, `Target.JAB`,
     the `E` fragment): the MMD digestion is a Hermite back-transform; F
     digests contracted quartets in-kernel (pin 1).

THE RULE THAT NAMES NO ENGINE (test/test_layering.py, rule "engine" and
`test_nothing_above_the_engines_names_an_engine`): this module is the one
place in `src/` that imports both engines' entry points. A module ABOVE the
engines -- one that imports this protocol, or one declared engine-neutral
there (today `ao.fock`, whose ERI arrives as `eri=`) -- may not import an
engine's modules by name. Tests are witnesses, not "above": they name both
engines on purpose, to compare them.

THE ASYMMETRY, stated so it is not mistaken for symmetry: MMD's `eri` is the
host Hermite path (`ao/eri.py`, pure Python), while HGP's runs the generated
IR in the interpreter (`ao/hgp_drive.py`). The protocol promises the
integral and its layout, not how it was evaluated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from felizjueves.ao import eri as mmd_eri
from felizjueves.ao import hgp_drive
from felizjueves.ao.shell import Shell
from felizjueves.ir import Kernel
from felizjueves.kernel import build_pq_kernel
from felizjueves.kernel_f import build_f_kernel
from felizjueves.spec import Spec, Target

ShellPair = tuple[Shell, Shell]
Block = tuple[tuple[float, ...], ...]
PQ = Callable[[tuple[int, int, int], tuple[int, int, int],
               tuple[float, float, float], float], float]
Boys = Callable[[int, float], float]


@runtime_checkable
class Engine(Protocol):
    """The smallest surface both ERI engines satisfy (module docstring)."""

    name: str
    target: Target

    def eri(self, bra_pair: ShellPair, ket_pair: ShellPair) -> Block:
        """`(ab|cd)`, `ao.eri.eri_block`'s layout."""
        ...

    def eri_and_scale(self, bra_pair: ShellPair,
                      ket_pair: ShellPair) -> tuple[Block, Block]:
        """`(ab|cd)` and this engine's absolute-value twin."""
        ...

    def kernel(self, spec: Spec) -> Kernel:
        """The vendor-neutral kernel of one class of `self.target`."""
        ...


# Each member, and the callable on each side that justifies it, as
# "module:attribute" relative to `felizjueves` (resolved by test/test_engine.py).
# `name` and `target` are data: their MMD and HGP sides are the targets the
# two builders accept.
MEMBERS: dict[str, tuple[str, str]] = {
    "name": ("engine:Mmd", "engine:Hgp"),
    "target": ("kernel:build_pq_kernel", "kernel_f:build_f_kernel"),
    "eri": ("ao.eri:eri_block", "ao.hgp_drive:eri_block"),
    "eri_and_scale": ("ao.eri:eri_block_and_scale",
                      "ao.hgp_drive:eri_block_and_scale"),
    "kernel": ("kernel:build_pq_kernel", "kernel_f:build_f_kernel"),
}


def _own(engine: Engine, spec: Spec) -> Spec:
    if not isinstance(spec, Spec):
        raise TypeError(f"{engine.name}.kernel takes a Spec, not {spec!r}")
    if spec.target is not engine.target:
        raise ValueError(
            f"{spec.key()}: engine {engine.name!r} builds Target."
            f"{engine.target.name} kernels, not Target.{spec.target.name}")
    return spec


@dataclass(frozen=True)
class Mmd:
    """Engine #1, McMurchie-Davidson: the [JEngine] [p|q] path. `pq` is the
    signed Hermite integral `[p|q]` (tests pass `reference.pq.pq`)."""

    pq: PQ
    name: str = "mmd"
    target: Target = Target.J

    def eri(self, bra_pair: ShellPair, ket_pair: ShellPair) -> Block:
        return mmd_eri.eri_block(bra_pair, ket_pair, pq=self.pq)

    def eri_and_scale(self, bra_pair: ShellPair,
                      ket_pair: ShellPair) -> tuple[Block, Block]:
        return mmd_eri.eri_block_and_scale(bra_pair, ket_pair, pq=self.pq)

    def kernel(self, spec: Spec) -> Kernel:
        return build_pq_kernel(_own(self, spec))


@dataclass(frozen=True)
class Hgp:
    """Engine #2, Head-Gordon-Pople: the [Fock] §3 VRR + HRR. `boys` is
    `F_m(T)` (tests pass `reference.boys.boys`)."""

    boys: Boys
    name: str = "hgp"
    target: Target = Target.F

    def eri(self, bra_pair: ShellPair, ket_pair: ShellPair) -> Block:
        return hgp_drive.eri_block(bra_pair, ket_pair, boys=self.boys)

    def eri_and_scale(self, bra_pair: ShellPair,
                      ket_pair: ShellPair) -> tuple[Block, Block]:
        return hgp_drive.eri_block_and_scale(bra_pair, ket_pair,
                                             boys=self.boys)

    def kernel(self, spec: Spec) -> Kernel:
        return build_f_kernel(_own(self, spec))


def engines(*, pq: PQ, boys: Boys) -> tuple[Engine, ...]:
    """Every engine, MMD first: the list a cross-engine check walks instead
    of a hand-maintained pair. Each takes its own special function."""
    return (Mmd(pq=pq), Hgp(boys=boys))
