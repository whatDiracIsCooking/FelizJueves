"""The plan layer: what a kernel for one class must compute, and how it is
split, decided before any kernel text exists.

Altitude, lowest first: `targets` (the Hermite targets [r](0) of a class),
`batching` ([JEngine, v2, Alg. 4]'s batch predicate and the sub-batch
partition). Everything here is host-side counting; nothing compiles or runs.
"""
