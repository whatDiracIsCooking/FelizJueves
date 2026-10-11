"""The card a kernel is compiled for and run on: `Arch`, as data.

TWO MEANINGS OF "TARGET", AND WHICH ONE IS WHICH. In `spec.py`, `Target` (and
`Spec.target`) is the MATRIX being built -- today only `Target.J`, the Coulomb
matrix. It says nothing about hardware. The hardware is an `Arch`, defined
here: the architecture a device compiler is driven at (`sm_86`, `gfx1200`) and
the few facts about it that a backend reads. A backend's signature is
`emit(kernel, arch: Arch)`; where an older issue body spells that parameter
`target`, it means an `Arch`, never a `spec.Target`. The word "target" is
reserved for the matrix.

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
here; a fact earns a field when a backend asks for it.

ONE SOURCE. `devtools/config.sh` is the shell's copy of the pins (shell cannot
import Python): `CUDA_ARCH=86`, `HIP_ARCH=gfx1200`. `from_env` reads the same
environment variables with the same defaults (and the images' `GPU_TARGETS`),
and `test/conftest.py` resolves its arch through it -- so a pin naming a card
this module does not declare fails loudly rather than drifting silently.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

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

    def __post_init__(self) -> None:
        if not isinstance(self.vendor, Vendor):
            raise ValueError(f"vendor {self.vendor!r} is not a Vendor")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"arch name {self.name!r} must be a non-empty string")
        if type(self.wavefront) is not int or self.wavefront <= 0:
            raise ValueError(f"wavefront {self.wavefront!r} must be a positive int")

    def __str__(self) -> str:
        return self.name


SM_86 = Arch(Vendor.NVIDIA, "sm_86", wavefront=32)  # RTX 3080, CUDA 13.0
GFX1200 = Arch(Vendor.AMD, "gfx1200", wavefront=32)  # RX 9060 XT, RDNA4, ROCm 7.2.4

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
        """E.g. ``coulomb/d0/(4,4|4,4)/b1/j/literal@sm_86/nvcc 13.0``."""
        return f"{self.spec.key()}@{self.arch.name}/{self.toolchain}"
