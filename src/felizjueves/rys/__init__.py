"""Engine #3: Rys quadrature for the two-electron integrals (#468's spike,
`experimental/rys/`, brought into `src/`).

`rys.roots` is the Rys rule -- the nodes and weights of the measure
`(1/2) x^(-1/2) exp(-T x) dx` on `[0, 1]` -- in fp64 as the device computes
it (`roots_weights`, and `device_text`, the `fj_rys<N>` routine the writer
carries), and `rys.exact` an independent route in `decimal` used only to
measure it. `rys.recurrence` is the per-axis 2D vertical recurrence as IR,
whose products are the HGP boundary `[e0|f0]^(0)` that `hgp.hrr` reads, so
the fused Fock kernel on Rys (`eri/fock/kernel_frys.py`) replaces only the
VRR of `kernel_f` and keeps its HRR, staging, screen and digestion, and the
split Coulomb kernel on Rys (`eri/hjk/kernel_rj.py`) does the same to
`kernel_hj`. `rys.recurrence.quartet_rule` is the quartet scalars and the
rule both share.

No module under `rys/` imports `plan/`, `ao/` or `reference/`.
"""

from __future__ import annotations
