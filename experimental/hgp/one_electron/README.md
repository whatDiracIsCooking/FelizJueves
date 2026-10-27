# experimental/hgp/one_electron — S, T, V and the dipole by the HGP route

A standalone spike: plain Python and numpy, s and p shells, Cartesian.
Nothing in `src/felizjueves/` imports it, nothing in `test/` collects it
(`testpaths = ["test"]`), and no tier runs it. The oracle is libcint through
PySCF, not this repo's MMD one-electron path.

```bash
uv sync --group oracle
uv run --group oracle python experimental/hgp/one_electron/check.py   # ~5 s, exit 0 = PASS
```

| File | What it is |
|---|---|
| `hgp1e.py` | `Shell`, `shells_of(mol)` (libcint's normalisation, s and p only, refusing above), `cart(l)` in libcint's order, `boys`, the two VRRs (`vrr_overlap`, `vrr_nuclear`), the contracted `hrr`, and the per-shell-pair blocks and whole matrices: `overlap(shells)`, `kinetic(shells)`, `nuclear(shells, charges, coords)`, `dipole(shells, origin)` → `<a\| r − C \|b>`, `[3, n_ao, n_ao]`, x, y, z. `FORM` is every sign as a multiplier (all 1). |
| `check.py` | The three tables below; exits nonzero on a miss. |

## The route

Head-Gordon–Pople's split on Obara–Saika's one-electron recurrences: per
primitive pair a **VRR** on the bra centre builds `(e|0)` up to the order the
target needs, the coefficients are applied and the pairs **summed**, and the
**HRR** `(a|b+1_k) = (a+1_k|b) + AB_k (a|b)` then runs once per shell pair on
contracted values (`AB` carries no exponent).

| Matrix | Formed from contracted overlap-type `(a\|b)` | VRR order |
|---|---|---|
| `S` | `(a\|b)` | `la + lb` |
| `MU_k` | `(a\|b+1_k) + (B_k − C_k)(a\|b)` | `la + lb + 1` |
| `T` | `−½ Σ_k [b_k(b_k−1)(a\|b−2_k)₀ − 2(2b_k+1)(a\|b)₁ + 4(a\|b+2_k)₂]` | `la + lb + 2` |
| `V` | `−Σ_C Z_C (a\|b)⁽⁰⁾_C`, the nuclear VRR with auxiliary index `m ≤ la + lb` | `la + lb` |

`T`'s subscript `n` is a contracted table weighted by `bⁿ` (`b` the ket
exponent): the per-primitive power of `b` is folded into the weight before the
contraction, so all three tables share one HRR. `V` sums the nuclei into the
same table, so its HRR also runs once per shell pair.

## What was checked

`check.py`, all PASS:

1. `boys` against `scipy.special.gammainc`, `m ≤ 4`, `T ∈ [0, 1e3]`: worst
   2.08e-15 relative.
2. Against libcint (`int1e_ovlp_cart`, `int1e_kin_cart`, `int1e_nuc_cart`,
   `int1e_r_cart` at origin `(0.3, −0.7, 1.1)`), `|err| / max|ref|`, tolerance
   1e-13. Every **ordered** shell pair is evaluated (no symmetry used), so the
   symmetry of each matrix is a check too:

   | molecule | nao | S | T | V | MU |
   |---|---|---|---|---|---|
   | H₂O/STO-3G | 7 | 3.3e-16 | 1.8e-16 | 7.8e-16 | 4.2e-16 |
   | H₂O/6-31G | 13 | 2.9e-16 | 4.7e-15 | 8.4e-16 | 3.1e-16 |
   | NH₃/6-31G, distorted off-axis | 15 | 3.6e-16 | 1.4e-14 | 2.2e-15 | 2.4e-16 |
   | CO, uncontracted 6-31G | 44 | 1.5e-15 | 2.8e-15 | 2.4e-16 | 8.0e-16 |

3. Each sign in `FORM` flipped alone is caught: the VRR's `PA` and `1/2p`
   terms and the HRR's `AB` in all four matrices on all four molecules, the
   nuclear VRR's two `m+1` terms in `V`, `B − C` in `MU`, and the kinetic
   middle term in `T`.

## Not done

d and above (the recurrences are general in `l`; `shells_of` refuses them
because libcint's common factor here is s/p only, and nothing above p was
checked); spherical harmonics; screening; any device kernel or generator path.
