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

from felizjueves.f_levers import batched_vrr, block_fab
from felizjueves.f_levers.seams import DEFAULT_HOOKS, FHooks
from felizjueves.spec import Spec

__all__ = ["REGISTRY", "hooks_for"]

# lever name -> its module's `apply`. Empty at #368: each F lever issue adds
# its line, e.g. `"batched-vrr": batched_vrr.apply` (#350).
REGISTRY: dict[str, Callable[[FHooks, Spec], FHooks]] = {
    batched_vrr.NAME: batched_vrr.apply,  # #350: the grouped, staged boundary
    "block-fab": block_fab.apply,  # #351
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
