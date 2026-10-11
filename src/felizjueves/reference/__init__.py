"""Numeric ground truth whose ancestry differs from the thing under test.

`boys` is the Boys function by Gauss-Legendre quadrature, not by the downward
recurrence a kernel uses. `pq` is the McMurchie-Davidson one-centre recurrence
run directly -- a memoised recursion over (r, m), with no MILD, no DAG, no
sieve and no schedule -- so that the IR path, which does all of those, agreeing
with it means something.

float64 throughout. Read by tests and by the interpreter's pins; nothing that
emits kernel text may import this package.
"""
