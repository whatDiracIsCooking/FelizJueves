"""Engine #2: the Head-Gordon-Pople recurrence of `[Fock]` §3 (milestone 12).

The second ERI engine beside the McMurchie-Davidson one, kept apart so that
neither is bent into the other's shape. `hgp.vrr` is the vertical recurrence
(`[Fock]` eqs 25-26) as a DAG over `(e, f, m)`; its target set is the
boundary set the horizontal recurrence (eq 28) reads.

No module under `hgp/` imports `plan/` or `reference/` (test/test_layering.py,
rule "hgp"): the plan layer keys on a `[JEngine]` total-am row an HGP class
does not have, and its `(r, m)` DAG is graded by `m`, which this one is not.
"""

from __future__ import annotations
