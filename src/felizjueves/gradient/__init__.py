"""Analytic nuclear gradients by McMurchie-Davidson (and, for `J`, `K`, `S`,
`T`, `V` and the fused `F`, by Head-Gordon-Pople too): host references and
the generated kernels
(`Spec.derivative_order = 1`).

    host.py          the references, pure Python, on the AO layer
    kernel_g1e.py    Target.S / T / V at d1: gs_, gt_, gv_{la}{lb}
    kernel_hgs.py    Target.S at d1 on Engine.HGP: hgs_{la}{lb}, the
                     overlap gradient by the HGP one-electron route
    kernel_gjab.py   Target.JAB at d1: gjab_{la}{lb}, the Coulomb energy's
                     bra derivative from J_p one Hermite order higher
    kernel_gk.py     Target.K at d1: gk_{la}{lc}_{lb}{ld}, the exchange
                     energy's bra derivative, K's plan one bra order up
    kernel_gmu.py    Target.DIPOLE / QUADRUPOLE at d1: gmu_ / gquad_{la}{lb},
                     SUM M dMU^c/dR per moment component, fixed origin
    kernel_ghj.py    Target.J on Engine.HGP at d1: ghj_{la}{lb}{lc}{ld},
                     the Coulomb energy's bra derivative per ordered
                     class on the Head-Gordon-Pople recurrence
    kernel_hgt.py    Target.T on Engine.HGP at d1: hgt_{la}{lb}, the
                     kinetic energy's gradient on kernel_hgp's route
    kernel_ghv.py    Target.V on Engine.HGP at d1: ghv_{la}{lb}, the
                     nuclear attraction's gradient on kernel_hgp's route
    kernel_gf.py     Target.F at d1, on HGP: gf_{la}{lb}{lc}{ld}, the fused
                     Fock build's dE_J - 1/2 dE_K, all four centres
    kernel_ghk.py    Target.K on Engine.HGP at d1: ghk_{la}{lc}_{lb}{ld},
                     the exchange energy's bra derivative on the same
                     recurrence, kernel_hj's nest
    host_df.py       the density-fitting references, (P|Q) and (mn|P)
    kernel_gmetric.py  Target.METRIC at d1: gmetric_{lP}{lQ}
    kernel_geri3.py    Target.ERI3 at d1: geri3_{la}{lb}{lP}
    kernel_hgeri3.py   Target.ERI3 at d1 on the HGP engine: hgeri3_{la}{lb}{lP},
                       the Cartesian rule on all three centres by VRR/HRR
    kernel_xeri3.py    Target.ERI3 at d1, uncontracted (#692): xeri3_{la}{lb}{lP},
                       the raw (nabla m n|P) of one aux block, not a contraction
    kernel_hgmetric.py Target.METRIC at d1 on the HGP engine:
                     hgmetric_{lP}{lQ}, the Cartesian rule on the HGP VRR
    rhf.py           dE_RHF/dR assembled from per-term runners (#565):
                     the signs, the half, V_nn^x, the total
    interpreted.py   the d1 kernels above as rhf.py's runners, run in
                     the interpreter (the J chain's [p|q] included)

WHAT A GRADIENT KERNEL WRITES. Not the derivative integrals -- a
`[n_atom, 3, n_ao, n_ao]` tensor -- but their contraction with a matrix,

    G[A, k] += SUM_mn M_mn dX_mn / dR_{A,k}

which is what every term of an SCF energy gradient is (`M = D` for `T`,
`V`, `J` and `K`, `M = W`, the energy-weighted density, for `S`; the sign
and the half are the caller's), and what a dipole or quadrupole
derivative at a fixed density is, one `G` per moment component
(`kernel_gmu`). The DF terms contract an aux matrix
(`SUM_PQ W_PQ d(P|Q)/dR`) and a three-index tensor (`SUM_mnP Gam_mnP
d(mn|P)/dR`) the same way (`host_df`'s module docstring has DF-J's).
THE UNCONTRACTED KERNELS ARE THE EXCEPTION (the `uncontracted` lever,
#693 and #692): `kernel_ipmetric` writes `d(P|Q)/dA` and `kernel_xeri3`
the three-center `(nabla_k m n|P)` over one aux block -- the derivative
integrals themselves, because a DF Hessian contracts them in more ways
than one kernel can fuse.
The rule is McMurchie & Davidson's, the one `experimental/mmd/gradient`
checked against libcint and finite differences:

    d/dA_k [x_A^i exp(-a x_A^2)] = 2a x_A^{i+1} exp(-a x_A^2)
                                   - i x_A^{i-1} exp(-a x_A^2)

so a derivative integral is two ordinary integrals of the SAME primitive
pair (same `zeta`, same `P`) with one Cartesian moved by one: the same `E`
tables, raised one higher on that side, and the same `[t]^(0)` (one
Hermite order higher). `V`'s operator also moves with its nucleus,
`d/dC_k [t]^(0)(P - C) = -[t + 1_k]^(0)` (Hellmann-Feynman); and the
Coulomb energy's ket derivative equals its bra derivative by the
permutational symmetry of `(ab|cd)`, so `dE_J = SUM D_ab D_cd d_bra (ab|cd)`
needs the bra side alone; so does the exchange energy's, `E_K = 1/2 SUM
D_ab D_cd (ac|bd)`, for a symmetric `D`: `dE_K = SUM D_ab D_cd d_bra
(ac|bd)` over every ordered quartet.

NO TRANSLATIONAL INVARIANCE IS USED: the derivative on both centres of a
pair (and on every nucleus, for `V`) is computed, so `SUM_A G[A] = 0` is a
check (test/test_gradient_host.py), not an assumption. Two HGP one-electron
kernels are the exceptions (`kernel_hgs` computes both centres), each
against a host that differentiates every centre on its own: `kernel_hgt`
adds `-dA` to `B`'s row (`T_ab` depends on `A - B` alone), and
`kernel_ghv` takes a nucleus's row as `-(dA + dB)` of that nucleus's own
tables (`(a|1/r_C|b)` depends on `A - C` and `B - C`).
"""
