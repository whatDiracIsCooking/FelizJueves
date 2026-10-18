"""What the generator ships (#166): the LITERAL plan, by default.

`KERNEL_GEN_CMD` (`python -m felizjueves.emit`) builds every `[p|q]` kernel
from the literal plan -- the minimax cut and the sieve's fixed seed order,
`Spec.levers = ()` -- and not from the searched board's tuned plan (#161's
`tuned` lever). Here, without reading a store: the default Spec is literal
at every L and the CLI offers no lever, and `j_engine.md` says what ships
and why. The measured grounds of that decision -- the declared rule read on
the tracked rows, the `limits.json` record recomputed from the stores, and
the shipped path's text hashed against the literal rows -- are checks of
measurement artefacts: measure/measure_shipped_plan.py
(`devtools/measure.sh check`).
"""

from __future__ import annotations

from pathlib import Path

from felizjueves.emit import __main__ as emit_cli
from felizjueves.kernel import pq_spec
from felizjueves.spec import Spec

ROOT = Path(__file__).resolve().parent.parent
JENGINE_MD = ROOT / "docs" / "implementation" / "j_engine.md"


# --- the tree is self-consistent with it -------------------------------------


def test_the_default_spec_is_literal() -> None:
    assert Spec.__dataclass_fields__["levers"].default == ()
    for L in range(17):
        assert pq_spec(L).levers == ()
    assert "--lever" not in emit_cli.__doc__ and "tuned" not in emit_cli.__doc__


def test_j_engine_md_says_what_ships_and_why() -> None:
    text = " ".join(JENGINE_MD.read_text().split())
    assert "The generator ships the **literal** plan" in text
    for cite in ("tuned_may_become_the_default", "plan_vs_kernel.json",
                 "sieve_may_leave_the_default", "batch_recommendation.json"):
        assert cite in text, cite
