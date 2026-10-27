# experimental/mmd/hessian — the analytic RHF nuclear hessian by McMurchie–Davidson

A standalone spike: plain Python and numpy, s and p shells, Cartesian, built
on `experimental/mmd/gradient` (its `mmd.py` and `ints.py` imported unchanged,
by path). Nothing in `src/felizjueves/` imports it, nothing in `test/`
collects it (`testpaths = ["test"]`), and no tier runs it. The oracle is
libcint and `pyscf.hessian.rhf` through PySCF, plus finite differences of the
gradient spike's analytic first derivatives and gradient.

```bash
uv sync --group oracle
uv run --group oracle python experimental/mmd/hessian/check.py   # ~130 s, exit 0 = PASS
```

| File | What it is |
|---|---|
| `ints2.py` | `one_electron2(shells, charges, coords)` → `d2S, d2T, d2V`, each `[natm, 3, natm, 3, nao, nao]` (`d2V` with its operator terms, centre–nucleus and nucleus–nucleus); `two_electron2(shells, natm)` → `d2(ab\|cd)` as `[natm, 3, natm, 3] + [nao]*4`; `nuclear_repulsion2`. |
| `hessian.py` | `cphf(C, e, nocc, eri, F0, F1, S1)` → the density responses `D1`, `W1` and `U`, every perturbation in one dense solve; `rhf_hessian(shells, charges, coords, C, e, nocc)` → the hessian `[natm, 3, natm, 3]` and its `nuc, T, V, J, K, S, response` parts. |
| `check.py` | The four tables below; exits nonzero on a miss. |

## How the second derivatives are formed

The gradient spike's rule, applied twice: `d/dA_k [x_A^i exp(-a x_A^2)] = 2a g_{i+1} - i g_{i-1}`,
so a second-derivative integral is up to four ordinary integrals with the
functions' `l` moved by two in all (`E` to `l + 2` on each side, `R` to
`L + 2`; the kinetic ket to `l + 4`), read off the same `E` and `R` tables.

- **S, T**: per primitive pair, `d2/dA dA`, `d2/dA dB`, `d2/dB dB`; the
  `(A, B)` block is written to both `[A, k, B, l]` and `[B, l, A, k]`.
- **V**: the same three, plus the moving operator,
  `d/dC_k R_tuv(P - C) = -R_{tuv+e_k}`: `d2/dA_k dC_l` is one function
  derivative times one `R` shift, `d2/dC_k dC_l` is `+R_{tuv+e_k+e_l}`.
- **(ab|cd)**: per shell pair, a 34-component Hermite stack (the pair, its
  six first derivatives, its 27 second ones, coefficients folded in); per
  bra primitive one `R`, contracted three ways (bra stack × ket pair, bra
  first × ket first, bra pair × ket stack), which covers all 16 ordered pairs
  of the four centres. One sweep over unique shell quartets, scattered over
  the 8-fold orbit with `ints.py`'s `_scatter`.
- **E_nuc**: `Z_A Z_B (3 r r^T / r^5 - I / r^3)` on the four atom blocks.

No translational invariance is used, so table 2 is a check.

## The hessian

The gradient spike's gradient, differentiated with `D` and `W` moving:

```
H_xy = E_nuc^xy + Σ D h^xy + ½ Σ D D (mn||ls)^xy − Σ W S^xy        (skeleton)
     + Σ D^y F^x − Σ W^y S^x                                        (response)
```

with `(mn||ls) = (mn|ls) − ½(ml|sn)` and `F^x = h^x + Σ (mn||ls)^x D_ls` the
skeleton Fock derivative. `C^y = C U^y`; `U_ij = −½ S^y_ij` on the occupied
block, and the CPHF equations for the virtual–occupied block,

```
(e_a − e_i) U_ai + Σ_bj [4(ai|bj) − (ab|ij) − (aj|bi)] U_bj = −(F^y_ai − e_i S^y_ai + G[D_S]_ai),
D_S = −2 C_o S^y_oo C_o^T,
```

are solved densely for all `3·natm` perturbations at once. `D^y` follows from
`C^y`, and `W^y` from `W = 2 C_o F_oo C_o^T` (`hessian.py`'s docstring has the
expression).

## The numbers (`check.py`, this box, 131 s)

The gradient spike's molecules: a distorted water
(`O 0 0 0.1; H 0.1 -0.757 0.587; H -0.05 0.8 0.5`, no symmetry) in STO-3G
(7 AOs) and 6-31G (13 AOs), and HCN in STO-3G (11 AOs). Every SCF at
`conv_tol = 1e-14`, `conv_tol_grad = 1e-11`. At the gradient spike's
`1e-12`, the residual `F_ai` alone moves the hessian by ~5e-9 (H₂O/STO-3G:
4.5e-9 against PySCF, 5.5e-12 once tightened).

| Table | What | Worst | Gate |
|---|---|---|---|
| 1. second derivatives vs libcint | `d2S` from `int1e_ipipovlp` / `_ipovlpip`; `d2T` from `int1e_ipipkin` / `_ipkinip`; `d2V` as PySCF's hessian `hcore_generator` − `d2T`; `d2(ab\|cd)` from `int2e_ipip1` / `_ipvip1` / `_ip1ip2`, permuted onto all 16 ordered centre pairs | 3.30e-12 of max (`d2T`, H₂O/STO-3G) | 1e-11 |
| 2. invariance | `\|Σ_A d2X/dR_A dR_B\|` | 3.30e-12 of max (`d2T`) | 1e-11 |
| 3. finite differences | against central differences of the gradient spike's `dS, dT, dV, d(ab\|cd)`, `h = 1e-4` bohr, STO-3G only | 1.39e-08 of max (`d2T`, HCN), the `O(h²)` truncation | 1e-7 |
| 4. RHF hessian | skeleton against `partial_hess_elec + hess_nuc`; the whole against `pyscf.hessian.rhf` with its CPHF solved densely; `\|H − H^T\|`; `\|Σ_A H\|`; FD of the gradient spike's RHF gradient, SCF re-converged per displacement, STO-3G only | total 6.86e-12 (H₂O/6-31G; `max\|H\|` 1.25); skeleton 6.69e-12; symmetry 1.0e-13; invariance 3.8e-12; FD 4.73e-08 (HCN) | 1e-10; FD 1e-6 |

**The 1e-11 integral gates** are the gradient spike's 1e-12 widened tenfold.
The worst entries are same-atom blocks that invariance makes exactly zero,
for example O 1s–O 1s differentiated twice on O. They are sums of terms
carrying `(2a)²` with `a ≈ 130`, and libcint's own `d2T` and `d2(ab|cd)`
miss invariance by up to 2.6e-12 of max on H₂O/6-31G, as this code does.

**Against PySCF's Krylov CPHF.** `Hessian.kernel()` as shipped is 6.14e-08
off on H₂O/6-31G and 2.20e-09 on HCN, even at `conv_tol_cpscf = 1e-14` and
`max_cycle = 200`. Plugged into PySCF's own response operator (`gen_vind`),
this code's `U` leaves a residual of 7.4e-15 in PySCF's CPHF equations, and
PySCF's Krylov `mo1` leaves 7.3e-7 (H₂O/6-31G). So table 4 gates against
`_pyscf_dense`: PySCF's `make_h1`, `gen_vind` and `hess_elec` with only the
linear solve made dense. It prints the shipped result beside it, not gated.

## What it is not

A reference, not a fast path: pure-Python loops, every second-derivative
tensor dense (`d2(ab|cd)` is `9·natm²·nao⁴`), the orbital hessian built from
a full MO transform and solved densely. s and p only (the gradient spike's
`shells_of`); no screening; closed-shell RHF only; no DF hessian; nothing
inferred from translational invariance (every centre pair is computed).
