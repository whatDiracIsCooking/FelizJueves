"""The card a kernel is compiled for and run on: `Arch`, as data.

TWO MEANINGS OF "TARGET", AND WHICH ONE IS WHICH. In `spec.py`, `Target` (and
`Spec.target`) is the MATRIX (or per-pair family) being built -- `Target.J`,
the Coulomb matrix, and its siblings (`K`, the fused Fock `F`, the pair and
one-electron targets). It says nothing about hardware. The hardware is an
`Arch`, defined here: the architecture a device compiler is driven at
(`sm_86`, `gfx1200`) and the few facts about it that a backend reads. A
backend's signature is `emit(kernel, arch: Arch)`; where an older issue body
spells that parameter `target`, it means an `Arch`, never a `spec.Target`.
The word "target" is reserved for the matrix.

WHAT KEYS A `Measured` ROW: `MeasuredKey(spec, arch, toolchain)`, NOT A WIDER
`Spec`. A `Spec` is the GENERATION key -- what the plan layer and the emitter
are keyed on -- and neither depends on which card the result will be measured
on: the plan for L = 12, b = 3 is the same plan on both cards. Widening `Spec`
with an arch would make every plan record, scoreboard cell and literature join
carry a field they cannot use, and would split one plan into two keys. But a
measurement is an observation of one toolchain on one card, so an `sm_86` row
and a `gfx1200` row for the same class and batch count must not collide, and
neither must two rows from two compiler versions on the same card (a compiler
bump moving a register count is news about the compiler). Hence the measured
store is keyed on the triple, and `Spec` stays as it is. The next person to
add a card adds an `Arch` value here and touches neither `Spec` nor any plan
record.

VALUES ONLY, AND ONLY THE FACTS A BACKEND READS. The arch string each compiler
is driven at, the vendor, and the wavefront width -- the one hardware fact that
changes emitted TEXT and that the two vendors spell differently (32 on both of
this box's cards, which is exactly why it is easy to bake in as a literal and
why it is read from here instead). Spelling differences -- the `__shfl` family,
launch-bounds attributes, `atomicAdd` on `double` -- belong in the backend, not
here; a fact earns a field when a backend asks for it. (The two ceilings
below are the one exception: no backend reads them, a roofline verdict does.)

THE ROOFLINE CEILINGS (#109), AND THE REFUSAL. Each Arch also carries the two
ceilings a ridge is computed from -- `fp64_flops_per_s` and
`mem_bandwidth_bytes_per_s`, both MEASURED on this box by `ceilings.py`'s
probes behind the card gate and read here from `data/measured/ceilings.json`
(the record, with each value's probe, host, date and the card's state at probe
time). `0.0` means NOT CHARACTERISED, and an uncharacterised card has no
`ridge` and gets no `verdict`: both raise `Uncharacterised` rather than invent
a number. No datasheet peak lives here -- the published figures in the data
file are the probes' sanity bound, never a default -- because a ridge borrowed
from a datasheet scores this box's kernels against a card it is not. A probe
the gate refused records nothing, so that card stays uncharacterised and
refuses: a missing ridge refuses to score, while a low one (a contended probe
under-reports) would score wrongly and look like data.

WHAT A VERDICT RESTS ON, AND NOTHING ELSE (#141). `verdict` reads exactly the
two recorded `value`s (`recorded_ceilings`) and nothing in the record's
`published` block: no vendor figure, no per-clock bound, no ratio. Each value
is the card at the clock it ran the probe at -- gfx1200's fp64 ceiling is
~3.49 GHz of 128 flop/clk, above its 3130 MHz rated boost (the data file's
`published.gfx1200.per_clock`) -- so a verdict is a statement about this card
at that clock. The #116 assumption that RDNA4's fp32 dual-issue applies to
fp64 was a property of the published sanity bound only, never of a ridge, and
#141's cycle counter retired it (single issue: 2 fp64 FMA lanes per CU per
clock).

The ceilings are observations OF a card, not part of what names it, so they
take no part in `Arch` equality or hashing: `Arch(NVIDIA, "sm_86", 32)` is the
same card as `SM_86`, characterised or not. NOTHING IN THE LITERATURE LAYER
MAY READ THEM (test/test_layering.py): a scoreboard reading a field that a
re-probe rewrites is how a frozen observation condition is destroyed.

ONE SOURCE. `devtools/config.sh` is the shell's copy of the pins (shell cannot
import Python): `CUDA_ARCH=86`, `HIP_ARCH=gfx1200`. `from_env` reads the same
environment variables with the same defaults (and the images' `GPU_TARGETS`),
and `test/conftest.py` resolves its arch through it -- so a pin naming a card
this module does not declare fails loudly rather than drifting silently.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from felizjueves.spec import Spec


class Vendor(StrEnum):
    NVIDIA = "nvidia"  # nvcc, CUDA
    AMD = "amd"  # hipcc, ROCm


@dataclass(frozen=True)
class Arch:
    """One device architecture: what the compiler is driven at, and its facts.

    `name` is the compiler's own spelling -- `nvcc -arch=<name>`,
    `hipcc --offload-arch=<name>` -- and the string a `Measured` row records.
    """

    vendor: Vendor
    name: str
    # Lanes per warp (NVIDIA) / wavefront (AMD). gfx12 compiles wave32 by
    # default under hipcc; RDNA can run wave64, and that would be a second Arch.
    wavefront: int
    # The measured ceilings (module docstring); 0.0 = not characterised.
    fp64_flops_per_s: float = field(default=0.0, compare=False)
    mem_bandwidth_bytes_per_s: float = field(default=0.0, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.vendor, Vendor):
            raise ValueError(f"vendor {self.vendor!r} is not a Vendor")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"arch name {self.name!r} must be a non-empty string")
        if type(self.wavefront) is not int or self.wavefront <= 0:
            raise ValueError(f"wavefront {self.wavefront!r} must be a positive int")
        for q in QUANTITIES:
            v = getattr(self, q)
            if isinstance(v, bool) or not isinstance(v, int | float) or not v >= 0:
                raise ValueError(f"{q} {v!r} must be a float >= 0 (0.0: not "
                                 "characterised)")

    def __str__(self) -> str:
        return self.name

    @property
    def characterised(self) -> bool:
        return all(getattr(self, q) > 0 for q in QUANTITIES)

    @property
    def ridge(self) -> float:
        """The ridge point, flop/byte: fp64 ceiling over bandwidth, both
        measured. Raises `Uncharacterised` when either is 0.0 -- there is no
        default ridge."""
        require_characterised(self)
        return self.fp64_flops_per_s / self.mem_bandwidth_bytes_per_s


class Uncharacterised(ValueError):
    """An Arch without both measured ceilings: no ridge, and no verdict."""


QUANTITIES = ("fp64_flops_per_s", "mem_bandwidth_bytes_per_s")


def require_characterised(arch: Arch) -> Arch:
    """`arch`, if both its ceilings are measured; else refuse."""
    missing = [q for q in QUANTITIES if not getattr(arch, q) > 0]
    if missing:
        raise Uncharacterised(
            f"{arch.name} is not characterised ({', '.join(missing)} = 0.0): "
            "measure it (ceilings.py's probe, behind the card gate) -- a "
            "datasheet figure is not a substitute")
    return arch


COMPUTE_BOUND = "compute-bound"
MEMORY_BOUND = "memory-bound"


def verdict(arch: Arch, flops_per_byte: float) -> str:
    """Which roof a kernel of arithmetic intensity `flops_per_byte` sits under
    on `arch`: COMPUTE_BOUND at or past the ridge, MEMORY_BOUND below it.
    Refuses (`Uncharacterised`) for a card without both measured ceilings.
    Depends on the measured ceilings ONLY -- never on a published figure or
    bound (module docstring, #141)."""
    ridge = require_characterised(arch).ridge
    if not flops_per_byte >= 0:
        raise ValueError(f"intensity {flops_per_byte!r} must be >= 0")
    return COMPUTE_BOUND if flops_per_byte >= ridge else MEMORY_BOUND


# The record the ceilings are read from (written by ceilings.py's probes).
CEILINGS_PATH = (Path(__file__).resolve().parent.parent.parent
                 / "data" / "measured" / "ceilings.json")


def recorded_ceilings(name: str, path: Path = CEILINGS_PATH) -> dict[str, float]:
    """The recorded `{quantity: value}` for the arch named `name`; a quantity
    with no record is 0.0 (not characterised), and so is every quantity when
    the file is absent."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        data = {}
    rows = data.get("ceilings", {}).get(name, {})
    return {q: float(rows[q]["value"]) if q in rows else 0.0 for q in QUANTITIES}


# RTX 3080, CUDA 13.0 / RX 9060 XT, RDNA4, ROCm 7.2.4
SM_86 = Arch(Vendor.NVIDIA, "sm_86", wavefront=32, **recorded_ceilings("sm_86"))
GFX1200 = Arch(Vendor.AMD, "gfx1200", wavefront=32, **recorded_ceilings("gfx1200"))

# Every card this box has, and so every Arch a test or the kernel tier may name.
ARCHS: tuple[Arch, ...] = (SM_86, GFX1200)


def by_name(name: str) -> Arch:
    """The declared Arch the compiler spells `name`; ValueError if undeclared."""
    for arch in ARCHS:
        if arch.name == name:
            return arch
    raise ValueError(
        f"arch {name!r} is not declared in felizjueves.device "
        f"(declared: {', '.join(a.name for a in ARCHS)}) -- add an Arch there"
    )


def from_env(vendor: Vendor, env: Mapping[str, str] | None = None) -> Arch:
    """The Arch the environment pins for `vendor`, with config.sh's defaults.

    NVIDIA reads `CUDA_ARCH` (bare compute capability, default `86`, so
    `sm_86`); AMD reads `HIP_ARCH`, then the images' `GPU_TARGETS`, default
    `gfx1200`. A pin to an undeclared card raises rather than passing through.
    """
    env = os.environ if env is None else env
    if vendor is Vendor.NVIDIA:
        return by_name(f"sm_{env.get('CUDA_ARCH') or '86'}")
    if vendor is Vendor.AMD:
        return by_name(env.get("HIP_ARCH") or env.get("GPU_TARGETS") or "gfx1200")
    raise ValueError(f"vendor {vendor!r} is not a Vendor")


@dataclass(frozen=True)
class MeasuredKey:
    """What keys a `Measured` row: the generation key, the card, the compiler.

    See the module docstring for why this is a triple and not a wider `Spec`.
    `toolchain` is the compiler as it reports itself (`"nvcc 13.0"`,
    `"hipcc / ROCm 7.2.4"`).
    """

    spec: Spec
    arch: Arch
    toolchain: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec, Spec):
            raise ValueError(f"spec {self.spec!r} is not a Spec")
        if not isinstance(self.arch, Arch):
            raise ValueError(f"arch {self.arch!r} is not an Arch")
        if not isinstance(self.toolchain, str) or not self.toolchain:
            raise ValueError(f"toolchain {self.toolchain!r} must be non-empty")

    def key(self) -> str:
        """E.g. ``coulomb/d0/(4,4|4,4)/b1/j/literal@sm_86/nvcc 13.0``.

        Every target keys the same way: a fused Fock (`Target.F`) cell is
        ``coulomb/d0/(2,1|1,0)/b1/f/literal@sm_86/nvcc 13.0``, its class in
        AO order (#278; `measured_store`, THE F CELL).
        """
        return f"{self.spec.key()}@{self.arch.name}/{self.toolchain}"
