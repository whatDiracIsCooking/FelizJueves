"""The K levers: one module each, plugged into `kernel_k` through its seams (#367).

`seams.py` is the seam: `KHooks`, `KContext` and the frozen defaults that
reproduce the literal kernel byte for byte. A K lever of milestone 13 lands as
`k_levers/<name>.py` exposing

    apply(hooks: KHooks, spec: Spec) -> KHooks

(the hooks it moves, by `dataclasses.replace`), plus ONE line in `REGISTRY`
below and its entry in `levers.VOCABULARY` -- never an edit to `kernel_k.py`.

`hooks_for(spec)` composes the registered levers a spec carries
(`Spec.kernel_body_levers()`, key order) onto `DEFAULT_HOOKS`, each lever's
`apply` seeing the previous ones' hooks -- so two levers that move different
hooks compose by construction (#348's split + interleave), and one that wraps
a hook another moved sees that one's version. A kernel-body lever with no
entry here moves no K hook: the builder records it as a `Note` and nothing
else (#337's contract, unchanged). `REGISTRY` ships EMPTY.
"""

from __future__ import annotations

from collections.abc import Callable

from felizjueves.k_levers import interleave, split
from felizjueves.k_levers.seams import DEFAULT_HOOKS, KHooks
from felizjueves.spec import Spec

__all__ = ["REGISTRY", "hooks_for"]

# lever name -> its module's `apply`. Empty at #367: each K lever issue adds
# its line, e.g. `"split": split.apply` (#341).
REGISTRY: dict[str, Callable[[KHooks, Spec], KHooks]] = {
    split.NAME: split.apply,  # #341: one kernel per K[a_i, b_j]
    interleave.NAME: interleave.apply,  # #343: interleaved I_p, warp arm
}


def hooks_for(spec: Spec, base: KHooks = DEFAULT_HOOKS) -> KHooks:
    """`base` with every registered kernel-body lever of `spec` applied, in
    key order. `DEFAULT_HOOKS` itself for a spec carrying none."""
    hooks = base
    for name in spec.kernel_body_levers():
        apply = REGISTRY.get(name)
        if apply is not None:
            hooks = apply(hooks, spec)
            if not isinstance(hooks, KHooks):
                raise TypeError(f"K lever {name!r}'s apply returned "
                                f"{type(hooks).__name__}, not KHooks")
    return hooks
