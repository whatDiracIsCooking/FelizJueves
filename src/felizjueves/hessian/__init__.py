"""Analytic nuclear Hessians by McMurchie-Davidson: the Coulomb energy's,
the exchange energy's, the one-electron `S`, `T`, `V`, and the
density-fitting metric `(P|Q)` and three-center integral `(mn|P)`, as host
references and as generated kernels (`Spec.derivative_order = 2`).

    host.py          the Coulomb reference, pure Python, on the AO layer
    kernel_hdq.py    Target.DQ at d1: hdq_{lc}{ld}, the ket pre-contraction
                     differentiated on each ket centre
    kernel_hjab.py   Target.JAB at d2: hjab_{la}{lb}, the Coulomb energy's
                     Hessian from J_p two bra orders up (bra-bra) and from
                     J_p of the ket derivative densities (bra-ket)
    kernel_hkx.py    Target.K at d2: hkx_{la}{lc}_{lb}{ld}, the exchange
                     energy's Hessian, both terms in one kernel per K
                     class (D_ab D_cd straddles bra and ket, so there is
                     no ket derivative density to precompute)
    host_kx.py       the exchange derivative matrix K^X reference (#689),
                     pure Python: the ip1 and contracted forms and their
                     assembly, the CPHF right-hand side's exchange part
    kernel_kx.py     Target.KX1 / KXC at d1: kx1_ / kxc_{la}{lc}_{lb}{ld},
                     K^X's two forms as AO matrices (gk's rolled form,
                     scattered into X[3, n_ao, n_ao], not reduced)
    host_df.py      the DF metric reference, pure Python
    kernel_d2metric.py  Target.METRIC at d2: d2metric_{lP}{lQ},
                     H[A, k, B, l] += SUM_PQ W_PQ d2(P|Q) / dR_{A,k} dR_{B,l}
                     (the Hermite shift twice: [r] -> [r + 1_k + 1_l],
                     the other centre's blocks by sign, no raised E)
    host_1e.py       the one-electron references, pure Python
    kernel_h1e.py    Target.S / T / V at d2: d2s_, d2t_, d2v_{la}{lb},
                     H[A, k, B, l] += SUM_mn M_mn d2X_mn / dR_{A,k} dR_{B,l}
                     (`M = D` for T and V, `W` for S; V's operator moves
                     with its nucleus: centre-nucleus and nucleus-nucleus
                     terms by `d/dC_k [t]^(0) = -[t + 1_k]^(0)`)
    host_jx.py       the Coulomb derivative matrix J^X (#688) and its ip1
                     bra half J1, pure Python: the CPHF right-hand side's
                     two-electron Coulomb part, an AO matrix per (atom, axis)
    kernel_jx.py     Target.JX at d1: jx_{la}{lb}, J1[k, m, n] = SUM_ls
                     (dA_k m n|ls) D_ls -- the J chain's bra back-transform
                     differentiated and scattered into (m, n); the ket half
                     is hdq_ + the d0 [p|q] + jab_, no kernel of its own
    host_eri3.py     the three-center DF reference, pure Python
    kernel_d2eri3.py Target.ERI3 at d2: d2eri3_{la}{lb}{lP},
                     H[X, k, Y, l] += SUM_mnP Gam_mnP d2(mn|P) / dR dR
                     over the AO pair's two centres (the rule twice, per
                     axis) and the aux centre (the Hermite shift twice)

The sections below are the Coulomb kernels'; the one-electron ones apply
the same rule twice on ONE pair (no split), and `kernel_h1e`'s docstring
is their account; `kernel_d2metric`'s and `kernel_d2eri3`'s docstrings
are the DF metric's and the three-center integral's.

WHAT A HESSIAN KERNEL WRITES. As a gradient kernel writes a contraction and
not the derivative integrals (`felizjueves.gradient`), a Hessian kernel
writes

    H[A, k, B, l] += d2 E_J / dR_{A,k} dR_{B,l},
    E_J = 1/2 SUM_abcd D_ab D_cd (ab|cd)

for a fixed, symmetric `D`: the skeleton (`D` not differentiated) Coulomb
part of an SCF Hessian. The density response (CPHF) is the caller's.

THE RULE is the gradient's, applied twice (experimental/mmd/hessian):
`d/dA_k [x_A^i exp(-a x_A^2)] = 2a x_A^{i+1} exp(-a x_A^2) - i x_A^{i-1}
exp(-a x_A^2)`, so per axis

    dA dA E^{ij} = 2 alpha (dA E)^{i+1,j} - i (dA E)^{i-1,j}
    dB dA E^{ij} = 2 beta  (dA E)^{i,j+1} - j (dA E)^{i,j-1}
    dB dB E^{ij} = 2 beta  (dB E)^{i,j+1} - j (dB E)^{i,j-1}

on a twice-differentiated axis (and a product of two first derivatives on
two different axes), from the undressed `E` tables raised two higher.

THE SPLIT. With every ordered quartet summed, the ket-ket term equals the
bra-bra term and the ket-bra term the bra-ket one (rename `(ab) <-> (cd)`),
so the half of `E_J` cancels both twos:

    d2 E_J / dx dy = SUM D_ab D_cd [d_bra^x d_bra^y + d_bra^x d_ket^y] (ab|cd)

`x` always on the bra. The first is `SUM_ab D_ab SUM_t (dx dy E^{ab})_t
J_t` with `J_t` the J chain's `J_p` two Hermite orders up; the second is
`SUM_ab D_ab SUM_t (dx E^{ab})_t J^{(C,l)}_t` with `J^{(C,l)}_t = SUM_q
[t|q] D^{(C,l)}_q`, the `[p|q]` kernel run over the KET DERIVATIVE
DENSITIES `D^{(C,l)}_q = SUM_cd D_cd (dC_l E^{cd})_q` of the kets that
touch atom `C`. Summed over every quartet each term is symmetric in
`x <-> y` by itself (rename `(ab) <-> (cd)` again); one bra class's share
of the bra-ket term is not, so it is never mirrored.

THE EXCHANGE ENERGY `E_K = 1/2 SUM D_ab D_cd (ac|bd)` splits the same
way (rename `(ac) <-> (bd)`); `hessian.host`'s THE EXCHANGE HESSIAN and
`kernel_hkx` have the detail.

NO TRANSLATIONAL INVARIANCE IS USED: every centre is differentiated, so
`SUM_A H[A, k, B, l] = 0` and `H = H^T` are checks
(test/test_hessian_host.py), not assumptions.
"""
