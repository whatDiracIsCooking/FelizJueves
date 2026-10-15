"""The kernel-body lever vocabulary, and which target may pull which lever (#337).

`Spec.levers` names departures from the literal reconstruction. Before
milestone 13 every such name belonged to a layer AROUND the kernel builder:
`no-sieve` and `tuned` to the plan layer (`plan/sieve.py`,
`plan/pipeline.py`), `recycle` to lowering (`lower.LEVERS`, applied after the
builder). Nothing could change an emitted statement. A KERNEL-BODY lever is
the missing kind: a name the builder itself reads, which changes the `Kernel`
it builds (a fused contraction, a precision, a thread mapping, ...).

THIS MODULE IS THE SEAM ONLY. It declares the vocabulary as DATA -- one
`Lever` per name in `VOCABULARY` -- and, per lever, a total legality table
over every target: legal, or refused with the REASON (the pattern of
`spec._PAIR_BATCH_REFUSAL` and `spec._F_BATCH_REFUSAL`, so a later reader
learns why, not just that). No lever's behaviour lands here: `VOCABULARY`
ships empty, and each lever issue of milestone 13 adds ONE entry here plus its
behaviour in its builder. Adding a lever is never an edit to `Spec`.

WHO READS IT.
  - `Spec.__post_init__` refuses a vocabulary lever on a target its table
    refuses, with the stated reason. A name NOT in the vocabulary is another
    layer's and passes through untouched, so its own layer refuses it as it
    always has (`plan.dag.build`: "no lever is implemented in the plan
    layer"); every pre-existing `Spec` and `Spec.key()` is unchanged.
  - The kernel builders (`kernel.build_pq_kernel(s)`,
    `kernel_k.build_k_kernel(s)`) split a spec's levers through the key
    (`Spec.kernel_body_levers`, `Spec.without_kernel_body_levers`, which
    call `split`; a builder never imports this module): the kernel-body
    ones stay with the builder, which records each in the built `Kernel` as
    a `Note` (`provenance`), and the rest go to the plan, which never sees
    a kernel lever. With no kernel lever the builders' output is
    byte-identical to before.

THE NAMES ARE ONE NAMESPACE. A kernel lever may not reuse a name another
layer owns (`OTHER_LAYERS`, restated here so this module stays a leaf; a test
pins it to `plan.sieve.NO_SIEVE`, `plan.pipeline.TUNED` and `lower.LEVERS`).

A leaf: it imports nothing from the package, so `spec` may import it. Targets
are named by their key slot (`Target.J == "j"`, a `StrEnum`), and
`TARGET_SLOTS` is pinned to `spec.Target` by a test.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

__all__ = [
    "DEFAULT_REFUSALS",
    "OTHER_LAYERS",
    "TARGET_SLOTS",
    "VOCABULARY",
    "Lever",
    "is_kernel_lever",
    "provenance",
    "refusal",
    "split",
]

# Every `spec.Target`'s key slot, in the enum's order (pinned by a test).
TARGET_SLOTS: tuple[str, ...] = ("j", "dq", "jab", "k", "s", "t", "v", "f")

# The `Spec.levers` names owned by layers other than the kernel builder.
OTHER_LAYERS: Mapping[str, str] = MappingProxyType({
    "no-sieve": "plan layer (plan/sieve.py, #162)",
    "tuned": "plan layer (plan/pipeline.py, #161)",
    "recycle": "lowering (lower.py, #55)",
})

_ONE_THREAD_PER_RECORD = (
    "has no kernel-body lever surface: it is one thread per record over an "
    "E fragment, with no [p|q] recurrence region and no batch for a lever to "
    "act on"
)

# Why each target refuses a kernel-body lever unless the lever's own table
# says otherwise. A `Lever` that is illegal on a target may give its own,
# sharper reason; these are the defaults a declaration starts from.
DEFAULT_REFUSALS: Mapping[str, str] = MappingProxyType({
    "j": "no kernel-body lever is admitted on Target.J unless its entry says so",
    "k": "no kernel-body lever is admitted on Target.K unless its entry says so",
    "dq": f"the DQ kernel {_ONE_THREAD_PER_RECORD}",
    "jab": f"the JAB kernel {_ONE_THREAD_PER_RECORD}",
    "s": f"the S kernel {_ONE_THREAD_PER_RECORD}",
    "t": f"the T kernel {_ONE_THREAD_PER_RECORD}",
    "v": (
        "the V kernel has the J recurrence but no option surface: milestone "
        "11 shipped it one batch, by decision, and a V lever is a follow-up "
        "nobody has declared"
    ),
    "f": (
        "Target.F's lever surface is #340's to declare: until it lands, no "
        "kernel-body lever reaches kernel_f"
    ),
})


@dataclass(frozen=True)
class Lever:
    """One kernel-body lever: its name, the issue that builds it, what it
    changes, and its legality on every target.

    `legal` is the set of target slots on which the builder accepts it;
    `refusals` maps EVERY other slot to the reason it is refused -- the table
    is total, checked here, so no target is ever silently unconsidered.
    `Lever.of` fills the refusals from `DEFAULT_REFUSALS`.
    """

    name: str
    issue: int
    what: str  # one line: what the lever changes in the built Kernel
    legal: frozenset[str]
    refusals: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or "+" in self.name:
            raise ValueError(f"lever name {self.name!r} must be a non-empty "
                             "string without '+' (the key's separator)")
        if self.name in OTHER_LAYERS:
            raise ValueError(f"lever name {self.name!r} is owned by the "
                             f"{OTHER_LAYERS[self.name]}")
        if type(self.issue) is not int or self.issue <= 0:
            raise ValueError(f"lever {self.name!r}: issue {self.issue!r}")
        if not isinstance(self.what, str) or not self.what:
            raise ValueError(f"lever {self.name!r} must say what it changes")
        object.__setattr__(self, "legal", frozenset(self.legal))
        object.__setattr__(self, "refusals",
                           MappingProxyType(dict(self.refusals)))
        if not self.legal:
            raise ValueError(f"lever {self.name!r} is legal on no target")
        both = self.legal & set(self.refusals)
        if both:
            raise ValueError(f"lever {self.name!r}: {sorted(both)} both legal "
                             "and refused")
        covered = self.legal | set(self.refusals)
        if covered != set(TARGET_SLOTS):
            raise ValueError(
                f"lever {self.name!r}: the legality table must cover every "
                f"target exactly; missing {sorted(set(TARGET_SLOTS) - covered)}"
                f", unknown {sorted(covered - set(TARGET_SLOTS))}")
        for slot, why in self.refusals.items():
            if not isinstance(why, str) or not why:
                raise ValueError(f"lever {self.name!r}: refusal on {slot!r} "
                                 "states no reason")

    @classmethod
    def of(cls, name: str, issue: int, what: str, legal: Iterable[str],
           **why_not: str) -> Lever:
        """A lever legal on `legal`, refused everywhere else with
        `DEFAULT_REFUSALS`' reason unless `why_not[slot]` gives its own."""
        legal = frozenset(legal)
        refusals = {s: why_not.get(s, DEFAULT_REFUSALS[s])
                    for s in TARGET_SLOTS if s not in legal}
        unknown = set(why_not) - set(refusals)
        if unknown:
            raise ValueError(f"lever {name!r}: a reason given for "
                             f"{sorted(unknown)}, which is legal or no target")
        return cls(name, issue, what, legal, refusals)


# THE VOCABULARY. Empty at #337, by design: each lever issue adds its entry,
# e.g. `"fused": Lever.of("fused", 344, "...", {"j"})`.
VOCABULARY: dict[str, Lever] = {
    "fused": Lever.of(
        "fused", 344,
        "each [r]^(0) is contracted into J_p as the recurrence forms it, one "
        "target live at a time (j_levers/fused.py)",
        {"j"},
        k=("the K kernel's contraction is staged through both E fragments "
           "per bra record, not J_p += (-1)^|q| [p+q]^(0) D_q; fusing it is "
           "a separate lever nobody has declared"),
    ),
}


def is_kernel_lever(name: str) -> bool:
    """True when `name` is a kernel-body lever (in `VOCABULARY`)."""
    return name in VOCABULARY


def refusal(target: str, name: str) -> str | None:
    """Why `name` may not be pulled on `target`, or None when it may -- or
    when `name` is not a kernel-body lever at all (another layer's to judge).
    """
    lever = VOCABULARY.get(name)
    if lever is None or target in lever.legal:
        return None
    return lever.refusals[target]


def split(levers: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`(kernel_body, rest)`, each in the given (sorted) order: what the
    builder keeps, and what it hands to the layers around it."""
    levers = tuple(levers)
    return (tuple(lv for lv in levers if is_kernel_lever(lv)),
            tuple(lv for lv in levers if not is_kernel_lever(lv)))


def provenance(name: str) -> str:
    """The `Note` a builder records for a kernel-body lever it was handed."""
    lever = VOCABULARY[name]
    return f"lever {lever.name} (#{lever.issue}): {lever.what}"
