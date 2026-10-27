"""Analytic nuclear hessians by McMurchie-Davidson: host references and the
generated kernels (`Spec.derivative_order = 2`).

    host.py          the references, pure Python, on the AO layer
    kernel_h1e.py    Target.S / T / V at d2: d2s_, d2t_, d2v_{la}{lb}

WHAT A HESSIAN KERNEL WRITES. As at d1 (`felizjueves.gradient`), not the
second-derivative integrals -- a `[n_atom, 3, n_atom, 3, n_ao, n_ao]`
tensor -- but their contraction with a matrix,

    H[A, k, B, l] += SUM_mn M_mn d2X_mn / dR_{A,k} dR_{B,l}

which is what the one-electron skeleton (fixed-density) part of an SCF
hessian is (`M = D` for `T` and `V`, `M = W` for `S`; the signs are the
caller's). The rule is the gradient's, applied twice -- the one
`experimental/mmd/hessian` checked against libcint, PySCF's RHF hessian
and finite differences:

    d/dA_k [x_A^i exp(-a x_A^2)] = 2a x_A^{i+1} exp(-a x_A^2)
                                   - i x_A^{i-1} exp(-a x_A^2)

so on one axis

    dA dA f_i  = 4a^2 f_{i+2} - 2a (2i + 1) f_i + i (i - 1) f_{i-2}
    dA dB f_ij = the rule on each index in turn

and a second derivative on two different axes is the product of the two
first derivatives. Every term is an ordinary integral of the SAME
primitive pair (same `zeta`, same `P`) with the Cartesians moved by two in
all: the same `E` tables raised two higher on each side, and the same
`[t]^(0)` two Hermite orders higher. `V`'s operator moves with its
nucleus, `d/dC_k [t]^(0)(P - C) = -[t + 1_k]^(0)`: a centre-nucleus term
is one function derivative times one shift, a nucleus-nucleus term two
shifts.

NO TRANSLATIONAL INVARIANCE IS USED: every ordered pair of centres (and
nuclei, for `V`) is computed, so `SUM_A H[A, k, B, l] = 0` is a check
(test/test_hessian_host.py), not an assumption.

Only the one-electron skeleton is built: no `J` / `K` second derivative,
no CPHF response, no assembled RHF hessian (`experimental/mmd/hessian`
has all three, s and p, in numpy).
"""
