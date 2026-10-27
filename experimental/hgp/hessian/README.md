# experimental/hgp/hessian — analytic nuclear hessians of J, K and G = J − K/2 by HGP

A standalone spike: the second nuclear derivatives of `(ab|cd)`, of `J`, `K`
and the two-electron Fock matrix `G = J − K/2` (`eri.fock.fock`'s), their
energy hessians by a fused sweep, and the closed-shell RHF hessian with every
two-electron piece computed by this repo's Head-Gordon–Pople engine
(`src/felizjueves/hgp/`, the kernels `eri/hgp_drive.py` builds). Cartesian,
any `l` (checked through d, so g classes run). It is built on
`experimental/hgp/gradient` (its `grad.py` and `lanes.py` imported unchanged,
by path) and borrows `experimental/mmd/hessian`'s dense CPHF. Nothing in
`src/felizjueves/` imports it, nothing in `test/` collects it
(`testpaths = ["test"]`), and no tier runs it. It touches no `src/` file.
Host-side: no kernel is emitted, compiled or run on a card. The oracles are
libcint and `pyscf.hessian.rhf` through PySCF, finite differences of the
gradient spike's analytic derivatives, and `experimental/mmd/hessian`'s MMD
second derivative.

```bash
uv sync --group oracle
uv run --group oracle python experimental/hgp/hessian/check.py   # ~225 s, exit 0 = PASS
```

| File | What it is |
|---|---|
| `grad2.py` | `quartet2(sa, sb, sc, sd)` → `d2(ab\|cd)` on every pair of the four centres, `[4, 3, 4, 3, na, nb, nc, nd]`; `two_electron2(shells, atoms, natm)` → `d2(ab\|cd)/dR dR` as `[natm, 3, natm, 3] + [nao]*4`, dense; `coulomb_deriv2`, `exchange_deriv2`, `fock_deriv2` (`d2J`, `d2K`, `d2G` at fixed `D`); `two_electron_hessian(shells, atoms, natm, D)` → `{"J", "K", "G"}` energy hessians by a fused sweep, no `nao⁴` tensor; `terms`, the rule as shifted classes; `FORM`, every factor of the rule and the digestion as a multiplier. |
| `rhf.py` | `rhf_hessian(shells, atoms, natm, C, e, nocc, one)` → the RHF hessian `[natm, 3, natm, 3]` and its `one, J, K, response` parts: `(ab\|cd)`, `d(ab\|cd)`, `F`, `F^x` and the skeleton `J`/`K` by HGP, the CPHF solve `experimental/mmd/hessian`'s `cphf`, the one-electron and nuclear terms the caller's (`one`). |
| `check.py` | The eight tables below; exits nonzero on a miss. |

## The route

The gradient spike's rule, `d/dA_k g_i = 2a g_{i+1_k} − i_k g_{i−1_k}`, is a
linear map `M_k = 2a U_k + L_k` from the Cartesians of `l` to those of `l ± 1`.
Applied twice:

```
same centre:   d2/dA_k dA_m g = (2a)² U_k U_m g[l+2] + 2a (U_k L_m + L_k U_m) g[l] + L_k L_m g[l−2]
two centres:   (2a U_k + L_k)_p (2b U_m + L_m)_q            four classes (l_p ± 1, l_q ± 1)
```

So every second derivative is a few ordinary contracted integrals of shifted
classes, each weighted per **primitive** by `(2a)²`, `2a`, `2a·2b`, `2b` or 1.
HGP contracts between the VRR and the HRR (`[Fock]` eq 28 holds only on
contracted values), so, as in the gradient spike, the weight multiplies the
seed `pair` lane by lane (every VRR value is linear in it), the lanes are
summed, and the unchanged HRR runs once on the weighted boundary
(`grad.block`). A term's class fixes its weight — a shift `s` on a centre
carrying `n` derivatives has `(2a)^{(s+n)/2}` — so per shell quartet there
are at most 36 blocks: 12 same-centre (four centres × `s ∈ {+2, 0, −2}`) and
24 cross-centre (six pairs × `(±1, ±1)`). Each is evaluated once and mapped
onto all nine `(k, m)` by the `U`/`L` matrices. The ten `p ≤ q` centre pairs
are computed and the other six are their transposes. Translational
invariance is **not** used, so table 3 tests it.

The sweeps are the gradient spike's. `two_electron2` evaluates each unique
shell quartet once and assigns it over the 8-fold orbit.
`two_electron_hessian` is `eri.fock`'s digestion read for a second
derivative: each distinct ordered orbit member is contracted as it forms into
`H_J = ½ Σ D_ab D_cd d2(ab|cd)` and `H_K = −¼ Σ D_ac D_bd d2(ab|cd)`, with
`H_G = H_J + H_K`, the two-electron skeleton of the RHF hessian at fixed `D`.

The RHF hessian is `experimental/mmd/hessian`'s expression:

```
H_xy = E_nuc^xy + Σ D h^xy − Σ W S^xy            (one: libcint, the caller's)
     + H_J + H_K                                 (grad2, fused, HGP)
     + Σ D^y F^x − Σ W^y S^x                      (response; F, F^x and the CPHF's (ai|bj) by HGP)
```

## The numbers (`check.py`, this box, 224 s)

Molecules: the gradient spikes' distorted water
(`O 0 0 0.1; H 0.1 -0.757 0.587; H -0.05 0.8 0.5`, no symmetry) in STO-3G
(7 AOs), 6-31G (13 AOs) and 6-31G* (19 AOs, a d shell on O, so the second
derivative runs g classes), and HCN in STO-3G (11 AOs, linear). Every SCF at
`conv_tol = 1e-14`, `conv_tol_grad = 1e-11` (`experimental/mmd/hessian`'s
reason). The gates are `experimental/mmd/hessian`'s, stated before the
numbers, plus the route's and the fused sweep's.

| Table | What | Worst | Gate |
|---|---|---|---|
| 1. the route | `grad.block` with every weight `quartet2` asks for (`(2a)²`, `2a·2b`, …) against `hgp_drive.eri_block_and_scale` in `ir.interp`, the weight on the coefficients, every shifted class of the quartet of the four highest shells | 3.29 ε of the absolute-value twin, elementwise (H₂O/STO-3G) | 16 ε |
| 2. second derivatives vs libcint | `d2(ab\|cd)` from `int2e_ipip1` / `_ipvip1` / `_ip1ip2` over all 16 ordered centre pairs; `d2J`, `d2K`, `d2G` at the RHF density from both | 1.90e-12 of max (`d2(ab\|cd)`, H₂O/6-31G) | 1e-11 |
| 3. invariance | `\|Σ_A d2(ab\|cd)/dR_A dR_B\|` | 1.06e-12 of max (HCN) | 1e-11 |
| 4. finite differences | against central differences of the gradient spike's analytic `d(ab\|cd)` and of its fused `G` gradient at fixed `D`, `h = 1e-4` bohr, STO-3G | 6.30e-09 of max (`d2(ab\|cd)`, H₂O), the `O(h²)` truncation | 1e-7 |
| 5. against MMD | `experimental/mmd/hessian`'s `d2(ab\|cd)` on the same molecule, s/p cases | 1.07e-12 of max (HCN) | 1e-11 |
| 6. energy hessians at fixed `D` | fused `J`, `K`, `G` against the dense contraction of table 2's tensor, and of libcint's | dense 7.19e-14; libcint 2.06e-12 Hartree/bohr² (`J`, H₂O/6-31G) | 1e-11 |
| 7. RHF hessian | skeleton against `partial_hess_elec + hess_nuc`; the whole against `pyscf.hessian.rhf` with its CPHF solved densely (`experimental/mmd/hessian`'s `_pyscf_dense`); `\|H − H^T\|`; `\|Σ_A H\|` | skeleton 3.48e-12, total 3.42e-12 (H₂O/6-31G*; `max\|H\|` 1.25); symmetry 8.3e-14; invariance 9.2e-12 | 1e-10 |
| 8. mutations | each `FORM` factor alone on H₂O/STO-3G: raise `2 → 1`, lower `1 → 0` (table 2), J `½ → ¼`, K `−¼ → −½` (table 7) | all four caught (smallest miss 0.72) | must exceed the gate |

**Why table 1 is gated on the twin, not on `max|X|`.** A `(2a)²` weight on
HCN's C 1s (`a ≈ 71`) makes the weighted boundary ~10⁴ times the block, and
the HRR's `CD` transfer cancels it back down. Measured against `max|X|`, the
lanes and the interpreter then differ by 5.35e-13 on `(pp|sd)`. That is the
same rounding magnified by the cancellation, and the HGP engine's own
yardstick for it is `hgp_drive`'s absolute-value twin ("THE SCALE"). Against
that twin the difference is 1.92 ε. The `max|X|` figure is still printed,
but it is not gated.

The shipped Krylov CPHF in PySCF is 6.14e-08 off on H₂O/6-31G and 4.79e-08
on 6-31G*, as `experimental/mmd/hessian` found. It is printed, not gated.

Sweep times: dense `d2(ab|cd)` 2.9 s (H₂O/STO-3G), 8.0 s (6-31G), 4.2 s (HCN),
77 s (6-31G*, where the g classes dominate); the fused hessian 0.9–24 s; the
whole RHF hessian (first derivatives, CPHF and fused skeleton) 1.1–28 s.

## What is not done

- The one-electron and nuclear-repulsion terms come from libcint and PySCF:
  `experimental/hgp/one_electron` has no derivatives yet.
- No kernel. The VRR and HRR run over numpy lanes on the host; nothing is
  emitted, compiled, run on a card or timed. A device form would raise each
  class exactly as here: the weight is one multiply on the seed per primitive
  quartet, and the 36 blocks of a quartet share every lane scalar.
- `two_electron2` is dense (`9·natm²·nao⁴`); the fused hessian is not. The
  CPHF is `experimental/mmd/hessian`'s dense solve on a dense `(ab|cd)` and
  MO transform. No screening; Cartesian only; closed-shell RHF only.
- d is the highest shell checked (6-31G*, so g in the raised class). The HGP
  DAGs and `lanes.py` are general in `l`, but nothing above d was run.
