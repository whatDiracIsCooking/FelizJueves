"""MILD: the per-node reduction direction, [JEngine, v2, Alg. 2].

Reduce a Hermite node `[r]^(m)` along the smallest NON-ZERO component of `r`;
on a tie take the cyclic successor of the two tied axes; when all three are
equal take z. docs/implementation/mild.md is the specification this follows.

A PURE FUNCTION OF `r`, AND THAT IS ITS LIMIT. No graph and no state, so it is
O(1) per node and keeps Johnson, Gill & Pople's rules 1, 2, 3 and 5 -- and for
the same reason it cannot keep rule 4 (common parents), which is a property of
the graph built so far. The sieve, [JEngine, v2, Alg. 3], is the separate pass
that recovers it.

THE TIE-BREAK IS THE OPPOSITE OF THE BATCHING PASS'S, on purpose: here ties go
to the cyclic successor and all-equal goes to z; there, to the predecessor and
to x (docs/implementation/j_engine.md). Both are to be pinned by tests.

`R` and `AXES` are defined here until the plan layer's targets module exists
to own them.
"""

from __future__ import annotations

from math import inf

R = tuple[int, int, int]
AXES = ("x", "y", "z")


def mild_direction(r: R) -> str:
    """Reduce along the smallest NON-ZERO component; ties to the cyclic
    successor; all three equal -> z.

    Raises on (0, 0, 0): [0](m) is a fundamental and is never reduced, so a
    direction for it is a caller bug rather than a value to return. Raises on
    anything that is not three non-negative components, for the same reason.
    """
    if len(r) != 3 or any(c < 0 for c in r):
        raise ValueError(f"not a Hermite index: {r!r}")
    if sum(r) == 0:
        raise ValueError("[0](m) is a fundamental and has no reduction direction")

    # A zero component cannot be reduced, so it is excluded from the minimum
    # rather than winning it (Alg. 2 line 1 sets it to infinity).
    masked = [c if c else inf for c in r]
    if masked[0] == masked[1] == masked[2]:
        return "z"
    lo = min(masked)
    for i in (1, 2, 0):  # a tie goes to the successor: y over x, z over y, x over z
        if masked[i] == lo and masked[i - 1] == lo:
            return AXES[i]
    return AXES[masked.index(lo)]
