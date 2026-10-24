"""The Coulomb J chain (milestones 4, 6 and 7, moved here by #511).

`j` is the host `J_µν` by two sweeps with its seams and screened sweep,
`kernel_dq` and `kernel_jab` the ket pre-contraction and the bra
back-transform-and-scatter kernels, `launch` the whole-molecule launch plan
and its oracle, `drive` and `staged` the interpreted adapters (the chain and
the launch-plan driver), and `levers/` the `[p|q]` kernel-body levers. The
`[p|q]` builder itself is `mmd.kernel`.

Each module keeps its own layering rule (test/test_layering.py); this package
marker imports nothing.
"""
