# Analytic nuclear gradients by McMurchie–Davidson: `S`, `T`, `V`, `E_J` and the DF integrals

**Status: implemented host-side and as generated kernels, interpreted
only. Not compiled by nvcc or hipcc, and not run on either card.** This is
the derived note for the first `derivative_order = 1` keys: the contracted
nuclear gradients of the overlap, the kinetic energy, the nuclear
attraction (its Hellmann–Feynman term included) and the Coulomb energy
`E_J`. It says what is generated, what it is checked against, and what is
not done. The session that built it had no device compiler and no card;
the kernel texts passed clang 18's CUDA front end (device side, `-Wall
-Wextra`, a declaration stub for the runtime) with no diagnostic, which is
a proxy, not the kernel tier.

**Nothing here is scored.** No published table covers these gradients;
the rule is McMurchie & Davidson's (`docs/references.md` entry 46) and the
kernels are this repo's own design, as the one-electron family is. No
`claims.json` record, cell or floor was added.

The spike this ports is `experimental/mmd/gradient/` (s and p, numpy),
which checked the same rule against libcint's `int1e_ip*`, `int2e_ip1`
and `pyscf.grad.RHF` to ~1e-13. That spike is not imported anywhere.

## The keys

`Spec.derivative_order` was a field with one legal value, kept so that
widening it would not retrofit every consumer. It is now `(0, 1)`
(`spec.DERIVATIVE_ORDERS`), and order 1 is legal on
`spec.GRADIENT_TARGETS` only: `S`, `T`, `V`, `JAB`, `METRIC` and `ERI3`,
each on its native engine, MMD (the last two: "The density-fitting
gradients", below). Every other target, and HGP, refuses it with
`_GRADIENT_REFUSAL`. A gradient class is its integral's class, `am` the
same pair `(la, lb)`, so `overlap/d1/(2,1)/b1/s/literal` and
`overlap/d0/(2,1)/b1/s/literal` differ in the `d` segment alone and every
d0 key is byte-identical. Unbatched, for each target's d0 reason.

## What a d1 kernel writes

Not the derivative integrals (a `[n_atom, 3, n_ao, n_ao]` tensor) but
their contraction with a matrix, the shape every SCF gradient term has:

    G[A, k] += SUM_mn M_mn dX_mn / dR_{A,k}

`M = D` for `T`, `V` and `J`, the energy-weighted density `W` for `S`; the
signs and halves of an energy are the caller's. Per shell pair `i <= j`
the weight of Cartesian pair `(r, c)` is `M[a_r, b_c] + M[b_c, a_r]` off
the diagonal and `M[a_r, a_c]` on it, which is `SUM_mn M_mn X_mn` exactly
for any `M` (no symmetry assumed) and any `X` symmetric in `m <-> n`.

## The rule

    d/dA_k [x_A^i exp(-a x_A^2)] = 2a x_A^{i+1} exp(-a x_A^2) - i x_A^{i-1} exp(-a x_A^2)

A derivative integral is two ordinary integrals of the SAME primitive pair
(same `zeta`, same `P`) with one Cartesian moved by one. So per axis

    dA E^{ij}_t = 2 alpha E^{i+1,j}_t - i E^{i-1,j}_t
    dB E^{ij}_t = 2 beta  E^{i,j+1}_t - j E^{i,j-1}_t

on the differentiated axis, the plain `E` on the other two, from the same
undressed `E` tables raised one index higher on each side.

- `S`: the `t = 0` products with one axis differentiated.
- `T`: the kinetic bracket `K_x S_y S_z + S_x K_y S_z + S_x S_y K_z`
  (`e_fragment.kinetic_fragment`'s `K_ij`) with the differentiated axis's
  factor replaced: `dA K_ij = 2 alpha K_{i+1,j} - i K_{i-1,j}`,
  `dB K_ij = 2 beta K_{i,j+1} - j K_{i,j-1}`. The operator commutes with
  a translation, so the rule applies to the ket function before the
  Laplacian. Tables at `(la + 1, lb + 3)`.
- `V`: the record's Hermite densities, once, outside the nucleus loop,
  `H0_t = SUM_r w_r E^r_t` (`|t| <= L`) and `HA_k,t`, `HB_k,t`
  (`|t| <= L + 1`); then per nucleus `C`, with `[t]^(0)` at `P - C`,
  `theta = sqrt(zeta)`, `|t| <= L + 1`:
  `gA_k -= Z_C SUM_t HA_k,t [t]^(0)` (and `B`), and the
  Hellmann–Feynman term `G[C, k] += s Z_C SUM_t H0_t [t + 1_k]^(0)`,
  from `d/dC_k [t]^(0)(P - C) = -[t + 1_k]^(0)`.
- `E_J`: `E_J = 1/2 SUM D_ab D_cd (ab|cd)`, and its ket derivative equals
  its bra derivative by the permutational symmetry of `(ab|cd)`, so
  `dE_J/dR_{A,k} = SUM_ab D_ab SUM_t (dA_k E^{ab})_t J_t` with `J_t` the J
  chain's `J_p` of the bra record **one Hermite order higher**,
  `|t| <= la + lb + 1`. The half of `E_J` and the bra-plus-ket two cancel:
  no factor appears.

No translational invariance is used. The derivative on both centres of a
pair, and on every nucleus for `V`, is computed, so `SUM_A G[A] = 0` is a
check.

## The host reference (`gradient/host.py`)

`overlap_gradient`, `kinetic_gradient`, `potential_gradient` (`one_centre=`
injected) and `coulomb_gradient` (`one_centre=` injected: `[p|q] =
(-1)^{|q|} [p+q]^(0)`, one `[r]^(0)` table per bra–ket record pair, the
kets `ao.density.make_kets` over the unique pairs at weight 2 off the
diagonal), each over every shell pair or `pairs=`, plus
`nuclear_repulsion_gradient`, `shell_atoms` (a shell sits on the nucleus
whose centre it equals exactly) and `records` (the kernels' per-record
arrays). `atoms=` gives each shell a row of its own, which the kernel tests
use to size a sum whose two centres cancel on one atom.

## The kernels

`gradient/kernel_g1e.py` builds `gs_`, `gt_`, `gv_{la}{lb}`
(`build_g1e_kernel`); `gradient/kernel_gjab.py` builds `gjab_{la}{lb}`
(`build_gjab_kernel`). One thread per primitive-pair record. Each inlines
`e_fragment.axis_tables_fragment` one index above the class (#586's
per-axis tables, read at constant indices; no `E^{ab}` block is formed),
declares each derivative factor where it is first read, and ends in six
`AtomicAdd`s into `G[n_atom, 3]` (three more per nucleus for `gv`). `G` is
zeroed by the caller.

- `gv` runs `[t]^(0)` through the plan layer one order up,
  `kernel_v.v_plan((la, lb + 1))`, with `recurrence_body` unchanged, as
  `kernel_v` does at d0.
- `gjab` reads `Jp[n_rows, n_p1]`, `n_p1 = len(hermite_targets(la+lb+1))`,
  in `kernel_jab`'s interleaved spelling. That `J_p` is the ordinary d0
  `[p|q]` kernel of the quartet class with the bra raised by one,
  `kernel_gjab.gjab_pq_spec(la, lb, lc, ld) = (la + 1, lb | lc, ld)`, over
  the same bra records and the J chain's `DQ` kets. Nothing in the `[p|q]`
  kernel or its plan is new.

Parameters (`kernel_s`'s per-axis spelling for the one-electron three):
`n_rec n_ao n_atom`, `zeta_ab pa_* pb_* [p_*: V] s_ab ea_bra eb_ket`,
`off_a off_b atom_a atom_b`, `M[n_ao, n_ao]`, `[Z Cnuc: V]`,
`G[n_atom, 3]`. For `V` the atoms are the nuclei.

## The generator

`python -m felizjueves.emit --grad-l [LMAX]` (2 bare, accepted 0..3,
exit 2 above) adds the four families for every `la <= lb <= LMAX` beside
the J ladders. It is never in the bare default and not in
`KERNEL_GEN_CMD`. The raised `[p|q]` class `gjab` reads is written by the
class selector (`--class LA+1 LB LC LD`), not by this flag. Goldens:
`test/golden/{cuda,hip}/{gs,gt,gv,gjab}_{la}{lb}` for (ss) through (dd),
48 files, the largest 211 KB.

## Witnesses, and what ran where

| What | Against | Where | Worst |
|---|---|---|---|
| host `S`, `T`, `V`, `E_J` (`test_gradient_host.py`) | central differences, `h = 1e-4`, of `overlap_matrix`, `kinetic_matrix`, `potential_matrix`, `coulomb_matrix` (each libcint-witnessed) | H₂O/STO-3G, a two-atom s, p basis; H₂O/cc-pVDZ and H₂O/STO-3G's `E_J` `slow` | ~2e-9 to 6e-9 of `max\|G\|` (the `O(h²)` floor), gate 1e-7 |
| host invariance | `SUM_A G = 0` | same | ~1e-15 of `max\|G\|`, gate 1e-13 |
| `V`'s Hellmann–Feynman term | a nucleus with no basis function, by finite difference | two-atom s, p basis plus a bare nucleus | as above |
| `gs`, `gt`, `gv` interpreted (`test_kernel_g1e.py`) | `host`, per class | every ordered class (ss)–(ff) on HF/cc-pVTZ; whole H₂O/STO-3G | `gs`, `gt` bitwise; `gv` 2.2 eps of the split-centre scale, gate 16 eps |
| `gjab` interpreted (`test_kernel_gjab.py`) | `host`, per bra class, host `J_p` | every ordered class (ss)–(ff) on a three-atom s–f basis | gate 16 eps of the split-centre scale |
| the chain: `[p|q]` at `gjab_pq_spec` (interpreted) → `gjab` | `host.coulomb_gradient` | the two-atom s, p basis; H₂O/STO-3G `slow` | gate 64 eps of the scale |
| compile, `-Wall -Wextra` | clang 18 CUDA front end, device side, runtime stub | all 40 units through (ff) | no diagnostic (a proxy) |
| compile, nvcc / hipcc (`test_emit_grad.py`) | #92's filter | — | **not run** (no compiler) |
| on a card (`test_grad_device.py`, `test/grad_driver.py`) | `host`, `1e-13` of `max\|G\|` | — | **not run** (no card) |

## The density-fitting gradients: `(P|Q)` and `(mn|P)`

**Status: host reference and generated kernels, interpreted; not compiled
by nvcc or hipcc and not run on either card** (the session had neither;
clang 18's CUDA front end passed all 31 units, `-Wall -Wextra`, a proxy).

`Target.METRIC` and `Target.ERI3` at d1. Each kernel contracts the
derivative integrals the way the others do:

    G[A, k] += SUM_PQ  W_PQ    d(P|Q)  / dR_{A,k}      gmetric_{lP}{lQ}
    G[A, k] += SUM_mnP Gam_mnP d(mn|P) / dR_{A,k}      geri3_{la}{lb}{lP}

with any `W` (`n_aux x n_aux`) and any `Gam` (`[n_ao, n_ao, n_aux]`, the
d0 kernel's layout). A DF-J energy gradient is the two together:

    E_J = SUM D_mn (mn|P) d_P - 1/2 d (P|Q) d,   (P|Q) d = (P|mn) D_mn
    dE_J/dR = geri3 at Gam = D (x) d   +   gmetric at W = -1/2 d d^T

(`E_J` is stationary in `d`, so `d` is not differentiated).

**The rules.** An AO pair's centres `A` and `B` take the Cartesian rule on
the per-axis `E` tables one index higher, as `gs`/`gv` do. An aux
function's single-center `E` does not depend on where it sits, so moving
it moves only the Hermite Gaussians: the kernels read the tower one index
up, `[t + tau + 1_k]^(0)`, with no raised `E` at all. For `(P|Q)` that
gives `d/dB = -d/dA` exactly (the tower reads `A - B`); for `(mn|P)` the
aux centre gets `-[t + tau + 1_k]^(0)` and `A`, `B` are each computed, so
`SUM_A G[A] = 0` is a check. The tower is the plan of the `Target.J`
quartet one order up (`(lP + 1, 0 | lQ, 0)`, `(la + 1, lb | lP, 0)`), cut
to its read targets by a liveness pass.

**The host reference** (`gradient/host_df.py`) takes the Cartesian rule on
every centre, the aux side included (`2 alpha E^{a+1} - a E^{a-1}` on the
single-center `E`), so kernel against host is two routes, not one restated.

**The kernels** keep the d0 kernels' staging and mapping: one thread per
aux shell pair (`metric_classes`) or per cell (`eri3_classes`), both record
ranges serially, the weights formed once per pair/cell by `kernel_g1e`'s
rule (swapped pairs need no flag), the weighted Hermite densities once per
bra record, then per (bra, ket) record pair the tower and a dot product.
Six (`gmetric`) or nine (`geri3`) `AtomicAdd`s per pair/cell into
`G[n_atom, 3]`, zeroed by the caller. Extra arrays beyond the d0 staging:
`atom_p`, `atom_q` (`host_df.metric_records`); `ea_bra`, `eb_ket`,
`atom_a`, `atom_b`, `atom_p` (`host_df.eri3_records`).

**The generator.** `python -m felizjueves.emit --grad-df-l [LMAX]` (4
bare, 0..4, exit 2 above) adds `gmetric_{lP}{lQ}` for `lP <= lQ`,
`lP + lQ <= LMAX`, and `geri3_{la}{lb}{lP}` for `la <= lb`,
`la + lb + lP <= LMAX`: 9 + 22 units, the d0 kernels' validated cap. Never
in the bare default nor in `KERNEL_GEN_CMD`. Goldens:
`test/golden/{cuda,hip}/{gmetric,geri3}_*`, 62 files, the largest 232 KB.

| What | Against | Where | Worst |
|---|---|---|---|
| host `(P\|Q)`, `(mn\|P)` (`test_gradient_host_df.py`) | central differences, `h = 1e-4`, of `metric_matrix`, `eri3_tensor` (libcint-witnessed) | a three-atom s/p/d basis and s/p/d aux; `(mn\|P)` through d `slow` | 2.1e-9, 2.7e-9 of `max\|G\|`, gate 1e-7 |
| host DF-J `dE_J/dR` | central differences of `E_J`, the solve redone per geometry | the s, p sub-bases; H₂O/STO-3G against the JKFIT s, p shells `slow` | 3.1e-9, 2.5e-9 |
| host invariance | `SUM_A G = 0` | same | gate 1e-13 of `max\|G\|` |
| `gmetric` interpreted (`test_kernel_gmetric.py`) | `host_df`, per class | every class `lP + lQ <= 4` (`lP + lQ = 4` `slow`), swapped and off-diagonal pairs | 4.9 eps of the split-centre scale, gate 16 eps |
| `geri3` interpreted (`test_kernel_geri3.py`) | `host_df`, per class | every class `la + lb + lP <= 4` (= 4 `slow`); whole H₂O/STO-3G against JKFIT s, p `slow` | 2.6 eps of the split-centre scale, gate 16 eps |
| compile, `-Wall -Wextra` | clang 18 CUDA front end, device side, runtime stub | all 31 units | no diagnostic (a proxy) |
| compile, nvcc / hipcc (`test_emit_grad_df.py`) | #92's filter | — | **not run** (no compiler) |
| on a card (`test_grad_df_device.py`, `grad_driver`'s program) | `host_df`, `1e-13` of `max\|G\|`, DF-J of H₂O/STO-3G against JKFIT s/p/d | — | **not run** (no card) |

Not done for DF: the metric solve and `d` stay host-side; no DF-K
gradient (the kernels take any `Gam`, but nothing stages K's); nothing
above L = 4; no screen; nothing measured; Cartesian only.

## What is not done

- No compile by nvcc or hipcc and no card run, on either card. The compile
  cases and the device test are written.
- The device test feeds `gjab` the host's `J_p`: an on-card chain (`DQ`,
  the raised `[p|q]`, `gjab`) is a follow-up.
- No exchange (`K`) gradient, so no whole RHF gradient on the card. The
  host has every term but `K` (`-1/4 SUM D dK D` needs all four centres of
  `(ac|bd)` against a density on both sides). The DF-J terms are built
  (above); no DF-K.
- Nothing is measured: no registers, spills or wall-clock. `gv` and
  `gjab` are unrolled and reach ~210 KB of text at (dd); a low-stack form
  (#586's) is not built.
- No screening, Cartesian only, no Hessian.
