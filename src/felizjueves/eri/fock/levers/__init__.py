"""The F levers: one module each, plugged into `kernel_f` through its seams (#368).

`seams.py` is the seam: `FHooks`, `FContext` and the frozen defaults that
reproduce the literal fused kernel byte for byte. An F lever of milestone 13
(#350-#361) lands as `f_levers/<name>.py` exposing

    apply(hooks: FHooks, spec: Spec) -> FHooks

(the hooks it moves, by `dataclasses.replace`), plus ONE line in `REGISTRY`
below and `"f"` in its `levers.VOCABULARY` entry's `legal` -- never an edit
to `kernel_f.py`.

`hooks_for(spec)` composes the registered levers a spec carries
(`Spec.kernel_body_levers()`, key order) onto `DEFAULT_HOOKS`, each lever's
`apply` seeing the previous ones' hooks -- so levers that move different
hooks compose by construction (#356's sliced HRR on #350's staged boundary),
and one that wraps a hook another moved sees that one's version. A
kernel-body lever with no entry here moves no F hook: `build_f_kernel`
records it as a `Note` and nothing else (#340's contract, unchanged).
`REGISTRY` shipped EMPTY at #368; each F lever adds its line (#351 first).
"""

from __future__ import annotations

from collections.abc import Callable

from felizjueves.eri.fock.levers import (
    batched_vrr,
    block_fab,
    etr,
    ket_break,
    scatter,
    sliced_hrr,
    sliced_site_digest,
    slim_fock,
    struct_reads,
    sub_slice,
    subdag_vrr,
)
from felizjueves.eri.fock.levers import mixed as fmixed
from felizjueves.eri.fock.levers.seams import DEFAULT_HOOKS, FHooks
from felizjueves.spec import Spec

__all__ = ["REGISTRY", "hooks_for"]

# lever name -> its module's `apply`. Empty at #368: each F lever issue adds
# its line, e.g. `"batched-vrr": batched_vrr.apply` (#350).
REGISTRY: dict[str, Callable[[FHooks, Spec], FHooks]] = {
    batched_vrr.NAME: batched_vrr.apply,  # #350: the grouped, staged boundary
    "block-fab": block_fab.apply,  # #351
    ket_break.NAME: ket_break.apply,  # #352: the monotone ket key's break
    struct_reads.NAME: struct_reads.apply,  # #353: typed pair-record reads
    sliced_hrr.NAME: sliced_hrr.apply,  # #356: one HRR per bra pair
    fmixed.NAME: fmixed.apply,  # #355: the VRR, boundary and HRR in FP32
    subdag_vrr.NAME: subdag_vrr.apply,  # #357: each group scheduled alone
    slim_fock.NAME: slim_fock.apply,  # #360: the reduced per-slice set
    sub_slice.NAME: sub_slice.apply,  # #361: each slice's HRR sub-sliced
    scatter.NAME: scatter.apply,  # #358: the atomic-free write-back
    sliced_site_digest.NAME: sliced_site_digest.apply,  # #359: per-slice digest
    etr.NAME: etr.apply,  # #354: the ETR as a second VRR, per class
}


def hooks_for(spec: Spec, base: FHooks = DEFAULT_HOOKS) -> FHooks:
    """`base` with every registered kernel-body lever of `spec` applied, in
    key order. `DEFAULT_HOOKS` itself for a spec carrying none."""
    hooks = base
    for name in spec.kernel_body_levers():
        apply = REGISTRY.get(name)
        if apply is not None:
            hooks = apply(hooks, spec)
            if not isinstance(hooks, FHooks):
                raise TypeError(f"F lever {name!r}'s apply returned "
                                f"{type(hooks).__name__}, not FHooks")
    return hooks
