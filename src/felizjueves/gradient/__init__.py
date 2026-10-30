"""Analytic nuclear gradients by McMurchie-Davidson (and, for `J` and `T`,
by Head-Gordon-Pople too): host references and the generated kernels
(`Spec.derivative_order = 1`).

    host.py          the references, pure Python, on the AO layer
    kernel_g1e.py    Target.S / T / V at d1: gs_, gt_, gv_{la}{lb}
    kernel_gjab.py   Target.JAB at d1: gjab_{la}{lb}, the Coulomb energy's
                     bra derivative from J_p one Hermite order higher
    kernel_gk.py     Target.K at d1: gk_{la}{lc}_{lb}{ld}, the exchange
                     energy's bra derivative, K's plan one bra order up
    kernel_ghj.py    Target.J on Engine.HGP at d1: ghj_{la}{lb}{lc}{ld},
                     the Coulomb energy's bra derivative per ordered
                     class on the Head-Gordon-Pople recurrence
    kernel_hgt.py    Target.T on Engine.HGP at d1: hgt_{la}{lb}, the
                     kinetic energy's gradient on kernel_hgp's route
    host_df.py       the density-fitting references, (P|Q) and (mn|P)
    kernel_gmetric.py  Target.METRIC at d1: gmetric_{lP}{lQ}
    kernel_geri3.py    Target.ERI3 at d1: geri3_{la}{lb}{lP}
    rhf.py           dE_RHF/dR assembled from per-term runners (#565):
                     the signs, the half, V_nn^x, the total
    interpreted.py   the d1 kernels above as rhf.py's runners, run in
                     the interpreter (the J chain's [p|q] included)

WHAT A GRADIENT KERNEL WRITES. Not the derivative integrals -- a
`[n_atom, 3, n_ao, n_ao]` tensor -- but their contraction with a matrix,

    G[A, k] += SUM_mn M_mn dX_mn / dR_{A,k}

which is what every term of an SCF energy gradient is (`M = D` for `T`,
`V`, `J` and `K`, `M = W`, the energy-weighted density, for `S`; the sign
and the half are the caller's). The DF terms contract an aux matrix
(`SUM_PQ W_PQ d(P|Q)/dR`) and a three-index tensor (`SUM_mnP Gam_mnP
d(mn|P)/dR`) the same way (`host_df`'s module docstring has DF-J's).
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
check (test/test_gradient_host.py), not an assumption.
"""
