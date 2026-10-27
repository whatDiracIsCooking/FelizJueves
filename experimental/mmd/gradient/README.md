# experimental/mmd/gradient — analytic S, T, V, J, K and DF gradients by McMurchie–Davidson

A standalone spike: plain Python and numpy, s and p shells, Cartesian.
Nothing in `src/felizjueves/` imports it, nothing in `test/` collects it
(`testpaths = ["test"]`), and no tier runs it. The oracle is libcint through
PySCF, plus finite differences of this module's own integrals.

```bash
uv sync --group oracle
uv run --group oracle python experimental/mmd/gradient/check.py     # ~50 s, exit 0 = PASS
uv run --group oracle python experimental/mmd/gradient/df_check.py  # ~280 s, exit 0 = PASS
```

| File | What it is |
|---|---|
| `mmd.py` | `Shell` and `shells_of(mol)` (libcint's normalisation, s and p only, refusing above), `cart(l)` in libcint's order, `e_coeffs` (per-axis `E^{ij}_t`), `Pair` (one primitive pair's `E` on three axes, `ov` and the 3D Hermite expansion `herm`), `d_centre` (the derivative rule), `boys` (series + downward recurrence below `T = 30`, `erf` + upward above) and `hermite_r` (`R_tuv`, vectorised over primitive quartets or nuclei), `displaced` (for finite differences). |
| `ints.py` | `one_electron(shells, charges, coords)` → `S, T, V, dS, dT, dV`; `two_electron(shells, natm)` → `(ab\|cd)` and `d(ab\|cd)/dR`; `coulomb`, `exchange`, `coulomb_deriv`, `exchange_deriv`; `nuclear_repulsion`, `rhf_energy`, `rhf_gradient(shells, charges, coords, D, W)` → the RHF gradient and its `nuc, T, V, J, K, S` parts. |
| `check.py` | The five tables below; exits nonzero on a miss. |
| `df.py` | Density fitting: `metric(aux, natm)` → `(P\|Q)` and `d(P\|Q)/dR`; `three_center(shells, aux, natm)` → `(mn\|P)` and `d(mn\|P)/dR`; `df_coulomb`, `df_exchange`, `df_j_gradient`, `df_k_gradient`, `df_rhf_energy`, `df_rhf_gradient` (auxiliary-basis response included). |
| `df_check.py` | The DF tables below; exits nonzero on a miss. |

## How the derivatives are formed

One rule for every centre: `d/dA_k [x_A^i exp(-a x_A^2)] = 2a g_{i+1} - i g_{i-1}`.
A derivative integral is therefore two ordinary integrals with one function's
`l` moved by one, read off the same `E` and `R` tables, so the integrals run to
d internally for a p basis (f for the kinetic ket, which also takes `l + 2`).

- **S**: `(π/p)^{3/2} E^x_{ij,0} E^y_{kl,0} E^z_{mn,0}`, differentiated on `A` and `B`.
- **T**: `-½ <a|∇²|b>` per axis, `j(j-1) S_{i,j-2} - 2b(2j+1) S_{ij} + 4b² S_{i,j+2}`;
  `d/dA` shifts the bra, `d/dB` the ket before the Laplacian is applied.
- **V**: `-Z_C (2π/p) Σ_tuv E_tuv R_tuv(p, P - C)`, `d/dA` and `d/dB` by the
  rule, and the operator term `d/dC_k R_tuv = -R_{tuv+e_k}` (Hellmann–Feynman).
- **(ab|cd)**: `2π^{5/2}/(pq√(p+q)) Σ E^{ab}_tuv (-1)^{τ+ν+φ} E^{cd}_τνφ R_{t+τ,u+ν,v+φ}(α, P - Q)`.
  Per shell pair, a 7-component Hermite stack (the pair, its `d/dA_k`, its
  `d/dB_k`, coefficients folded in); per bra primitive, one `R` for every ket
  primitive; the stack contracted against the undifferentiated other side.
  One sweep over the unique shell quartets, scattered over the 8-fold orbit
  (each derivative tensor `d(ab|cd)/dR` has the same symmetry).
- **J, K**: `dJ_mn = Σ_ls d(mn|ls)/dR D_ls`, `dK_mn = Σ_ls d(ml|sn)/dR D_ls`,
  `D` fixed.
- **RHF**: `Σ D dh + ½ Σ D dJ − ¼ Σ D dK − Σ W dS + dE_nuc`, with `D` the total
  density and `W = 2 Σ_occ ε_i C_i C_iᵀ`.

The derivative on every centre is computed; translational invariance is not
used to infer one from the others, so it is checked (table 3) rather than
assumed.

## The numbers (`check.py`, this box, 49 s)

Molecules: a distorted water (`O 0 0 0.1; H 0.1 -0.757 0.587; H -0.05 0.8 0.5`,
no symmetry) in STO-3G (7 AOs) and 6-31G (13 AOs), and HCN in STO-3G (11 AOs,
three p atoms). Densities from PySCF's RHF at `conv_tol = 1e-12`.

| Table | What | Worst | Gate |
|---|---|---|---|
| 1. integrals | `S, T, V, (ab\|cd), J, K` against `int1e_ovlp`, `int1e_kin`, `int1e_nuc`, `int2e`, `get_j`, `get_k` | 6.80e-15 of max (`(ab\|cd)`, H₂O/6-31G); `E_RHF` 4.1e-13 absolute | 1e-12; 1e-10 |
| 2. derivatives vs libcint | `dS, dT` from `int1e_ipovlp`, `int1e_ipkin`; `dV` as `hcore_generator − dT` (operator term included); `d(ab\|cd)` from `int2e_ip1` over all four centres; `dJ, dK` at the RHF density | 3.94e-14 of max (`d(ab\|cd)`, H₂O/6-31G) | 1e-12 |
| 3. invariance | `\|Σ_A dX/dR_A\|` | 3.22e-15 of max (`dT`) | 1e-12 |
| 4. finite differences | analytic against central differences of this module's integrals, `h = 1e-4` bohr, STO-3G only | 8.74e-09 of max (`dT`, HCN), the `O(h²)` truncation | 1e-7 |
| 5. RHF gradient | against `pyscf.grad.RHF`; the J+K part against its `get_veff` | 1.77e-13 Hartree/bohr (HCN; `max\|grad\|` 1.6e-2); J+K 1.95e-14 | 1e-10 |

## Density fitting (`df.py`)

An auxiliary function is a pair with a unit s partner (exponent 0,
coefficient 1, same centre), so `(P|Q) = (P s₀|Q s₀)` and
`(mn|P) = (mn|P s₀)` go through `ints.py`'s `_pair_data` and `_quartet`
unchanged, prefactor included. The partner's derivative vanishes, so the aux
side contributes its `d/dA` component alone; every centre is computed, as
above. With `γ_P = Σ (P|mn) D_mn`, `c = M⁻¹γ` and `C^P = Σ_Q M⁻¹_PQ (Q|··)`:

- **DF-J**: `dE_J = cᵀ dγ − ½ cᵀ dM c`.
- **DF-K**: `dE_K = −¼ [2 Σ_P Σ_ml d(ml|P) (D C^P D)_ml − Σ_PQ dM_PQ Tr(C^P D C^Q D)]`.
- **DF-RHF**: the RHF terms above with these two in place of `J` and `K`.

`df_check.py` (this box, 278 s): the same three molecules, auxiliary basis
`cc-pvdz-jkfit` **cut to its s and p shells** (the spike stops at p on both
bases), densities from PySCF's DF-RHF on that basis at `conv_tol = 1e-12`.

| Table | What | Worst | Gate |
|---|---|---|---|
| 1. integrals | `(P\|Q)`, `(mn\|P)` against `int2c2e`, `int3c2e`; `J_DF`, `K_DF` against `with_df.get_jk` | 2.51e-14 of max (`(mn\|P)`, HCN); `E_DF-RHF` 3.98e-13 absolute | 1e-12; 1e-10 |
| 2. derivatives vs libcint | `d(P\|Q)` from `int2c2e_ip1`; `d(mn\|P)` from `int3c2e_ip1` (on `m` and `n`) and `int3c2e_ip2` (on `P`) | 5.13e-14 of max (`d(P\|Q)`, H₂O) | 1e-12 |
| 3. invariance | `\|Σ_A dX/dR_A\|` | 1.42e-15 of max (`d(mn\|P)`, H₂O/6-31G) | 1e-12 |
| 4. finite differences | `d(P\|Q)`, `d(mn\|P)`, and `dE_J`, `dE_K` at fixed `D`, against central differences, `h = 1e-4` bohr, STO-3G only | 4.84e-09 of max (`dE_J`, H₂O) | 1e-7 |
| 5. DF-RHF gradient | against `pyscf.df.grad.rhf` (`auxbasis_response` on) | 2.56e-13 Hartree/bohr (HCN) | 1e-10 |

## What it is not

A reference, not a fast path: pure-Python loops over primitive pairs and the
Hermite indices, ~4 s for the H₂O/6-31G ERI derivatives, and every
derivative tensor dense (`natm · 3 · nao⁴`). s and p only (`shells_of`
refuses d: libcint's Cartesian d normalisation is per component and not
written here); no screening; the DF metric inverted densely, no Cholesky; closed-shell RHF only; no density-response
(CPHF) terms, which a first gradient does not need.
