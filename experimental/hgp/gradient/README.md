# experimental/hgp/gradient — analytic nuclear gradients of J, K and G = J − K/2 by HGP

A standalone spike: the nuclear derivatives of `(ab|cd)`, of `J`, `K` and the
two-electron Fock matrix `G = J − K/2` (`eri.fock.fock`'s), and their energy
gradients, computed by this repo's Head-Gordon–Pople engine
(`src/felizjueves/hgp/`, the kernels `eri/hgp_drive.py` builds). Cartesian,
any `l` (checked through d). Nothing in `src/felizjueves/` imports it,
nothing in `test/` collects it (`testpaths = ["test"]`), and no tier runs it.
It touches no `src/` file. Host-side: no kernel is emitted, compiled or run on
a card. The oracle is libcint through PySCF, finite differences of this
module's own integrals, and `experimental/mmd/gradient`'s MMD derivative.

```bash
uv sync --group oracle
uv run --group oracle python experimental/hgp/gradient/check.py   # ~45 s, exit 0 = PASS
```

| File | What it is |
|---|---|
| `lanes.py` | `compile_kernel(kernel)`: an `hgp.recurrence` VRR or HRR `ir.Kernel` translated statement by statement into numpy over lanes (one lane per primitive quartet) and memoised; `boys_lanes(top, T)`, `reference.boys` (its 112-node rule and asymptote) over arrays. Only the statement and expression kinds the two kernels hold are accepted; anything else is refused. |
| `grad.py` | `quartet(sa, sb, sc, sd)` → `(ab\|cd)` and its derivative on each of the four centres; `two_electron(shells, atoms, natm)` → `(ab\|cd)` and `d(ab\|cd)/dR`, dense; `coulomb`, `exchange`, `fock` and `coulomb_deriv`, `exchange_deriv`, `fock_deriv` (`dJ`, `dK`, `dG` at fixed `D`); `two_electron_gradient(shells, atoms, natm, D)` → `{"J", "K", "G"}` energy gradients by a fused sweep, no `nao⁴` tensor; `FORM`, every factor of the rule and the digestion as a multiplier. |
| `check.py` | The eight tables below; exits nonzero on a miss. |

## The route

One rule for every centre, `d/dA_k [x_A^i e^{−a x_A²}] = 2a g_{i+1} − i g_{i−1}`,
so `d(ab|cd)/dA` is two ordinary integrals of the classes `(la+1 lb|lc ld)`
and `(la−1 lb|lc ld)`. The `2a` is per **primitive**, and HGP contracts
between the VRR and the HRR (`[Fock]` eq 28 holds only on contracted values),
so it cannot be applied after the HRR. It is folded into the contraction
instead, as `experimental/hgp/one_electron` folds the kinetic ket's `bⁿ`:

```
boundary(e, f) = SUM_{ijkl} 2a_i · [e0|f0]^(0)_{ijkl}    VRR of the raised class, per primitive
d/dA (ab|cd)   = HRR(boundary; AB, CD) − i · (a−1 b|cd)  HRR once, contracted
```

Every VRR value is linear in the seed scale `pair` (`hgp_drive`, "THE
BINDING"), so the raised class runs the unchanged VRR with `pair · 2a` lane by
lane. The HRR's `AB` and `CD` carry no exponent, so it runs unchanged on the
weighted boundary. Each of the four centres is done this way: per shell
quartet, four raised classes and up to four lowered ones. Translational
invariance is **not** used to infer a centre, so table 4 tests it instead.

The kernels are `hgp.recurrence`'s, built by `hgp_drive._kernels` and memoised
per class. They are not interpreted one primitive quartet at a time
(`hgp_drive` takes ~0.15 s for an interpreted `(pp|pp)`). `lanes.py` runs the
same IR over every primitive quartet at once, as a GPU runs one lane per
thread, and table 1 holds it to the interpreter. No recurrence is restated.

The sweeps:
- `two_electron` evaluates each unique shell quartet once (`i ≤ j`, `k ≤ l`,
  `ij ≤ kl`) and assigns its value and atom-resolved derivative over the
  8-fold orbit.
- `two_electron_gradient` is the fused form, `eri.fock`'s digestion read for a
  derivative. Each distinct ordered member of each orbit is contracted as it
  forms into
  `dE_J = ½ Σ D_ab D_cd d(ab|cd)` and `dE_K = −¼ Σ D_ac D_bd d(ab|cd)`,
  with `dE_G = dE_J + dE_K` (the RHF gradient's two-electron part, `D` the
  total density).

## The numbers (`check.py`, this box, 45 s)

Molecules: `experimental/mmd/gradient`'s distorted water
(`O 0 0 0.1; H 0.1 -0.757 0.587; H -0.05 0.8 0.5`, no symmetry) in STO-3G
(7 AOs), 6-31G (13 AOs) and 6-31G* (19 AOs, a d shell on O, so the derivative
runs f classes), and HCN in STO-3G (11 AOs, linear). Densities from PySCF's
RHF at `conv_tol = 1e-12`. The gates are `experimental/mmd/gradient`'s,
stated before the numbers.

| Table | What | Worst | Gate |
|---|---|---|---|
| 1. the route | `boys_lanes` against scalar `reference.boys`, `m ≤ 17`, `T ∈ [0, 1e6]` (relative); `grad.block` against `hgp_drive.eri_block` in `ir.interp` on every raised, plain and lowered class of four quartets per case | Boys 4.43e-16; blocks 3.17e-15 of max (HCN) | 1e-15; 1e-14 |
| 2. integrals vs libcint | `(ab\|cd)`, `J`, `K`, `G` against `int2e`, `get_j`, `get_k` | 1.34e-14 of max (`(ab\|cd)`, 6-31G*) | 1e-12 |
| 3. derivatives vs libcint | `d(ab\|cd)/dR` from `int2e_ip1` over all four centres; `dJ`, `dK`, `dG` at the RHF density from both | 4.07e-14 of max (`d(ab\|cd)`, HCN) | 1e-12 |
| 4. invariance | `\|Σ_A d(ab\|cd)/dR_A\|` | 4.76e-14 of max (HCN) | 1e-12 |
| 5. finite differences | against central differences of this module's own `(ab\|cd)`, `h = 1e-4` bohr, STO-3G | 4.22e-09 of max (HCN), the `O(h²)` truncation | 1e-7 |
| 6. against MMD | `experimental/mmd/gradient`'s `d(ab\|cd)/dR` on the same molecule, STO-3G | 4.03e-14 of max (HCN) | 1e-13 |
| 7. energy gradients | fused against the dense contraction; the J and K parts against PySCF's `get_jk` gradient; the RHF total (libcint's one-electron and nuclear terms + this module's `dE_G`) against `pyscf.grad.RHF` | fused 2.31e-14; J 6.31e-14, K 1.28e-14; total 5.06e-14 Hartree/bohr (H₂O/6-31G; `max\|grad\|` 7.6e-2) | 1e-13; 1e-10 |
| 8. mutations | each `FORM` factor alone on H₂O/STO-3G: raise `2 → 1`, lower `1 → 0` (table 3), J `½ → ¼`, K `−¼ → −½` (table 7) | all four caught (smallest miss 0.33) | must exceed the gate |

Sweep times (dense, with derivatives): 0.6 s (H₂O/STO-3G), 1.3 s (6-31G),
0.8 s (HCN), 20.8 s (6-31G*, where the f classes dominate); the fused
gradient 0.2–4.6 s.

## What is not done

- The one-electron and nuclear-repulsion terms of the RHF gradient come from
  PySCF in table 7: `experimental/hgp/one_electron` has no derivatives yet.
- No kernel. The VRR and HRR run over numpy lanes on the host; nothing is
  emitted, compiled, run on a card or timed. A device form would raise the
  class per centre exactly as here: the `2a` weight is one multiply on the
  seed per primitive quartet.
- Every derivative tensor in `two_electron` is dense (`natm · 3 · nao⁴`); the
  fused gradient is not. No screening; Cartesian only; RHF closed-shell
  only; no density response (CPHF), which a first gradient does not need.
- d is the highest shell checked (6-31G*, so f in the raised class); the
  HGP DAGs and `lanes.py` are general in `l`, but nothing above d was run.
