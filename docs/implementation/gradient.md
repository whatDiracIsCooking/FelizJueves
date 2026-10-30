# Analytic nuclear gradients by McMurchie–Davidson: `S`, `T`, `V`, `E_J`, `E_K` and the DF integrals

**Status: implemented host-side and as generated kernels, compiled by
nvcc and hipcc and run on both cards; the whole RHF gradient assembled and
held to PySCF interpreted (#565) and on the RX 9060 XT only, H₂O/STO-3G
and 6-31G, CUDA not run (#566); nothing scored or timed, the closing
`StructuralLimit` is #567's** (the RTX 3080 and the RX 9060 XT,
against the host, kernel classes at L <= 4 only, except the DF terms,
through (ff|i) and (i|i) since #613, `gs`, `gt`, `gv` through (ff)
since #614, `gk` through `(ff|ff)` since #618 and the whole `gjab` chain
(`DQ`, raised `[p|q]`, `gjab`) through (ff) since #619; which inputs, per kernel, in the tables below).
The Coulomb gradient on the HGP recurrence (`ghj_`, "The Coulomb gradient
on the HGP recurrence") is generated and interpreted only; the exchange
gradient on it (`ghk`, "The exchange gradient on HGP") is interpreted
through `(dd|dd)` and compiled by nvcc and hipcc through `(dd|pp)`, on no
card; the kinetic, overlap and nuclear-attraction gradients on HGP
(`hgt_`, `hgs_`, `ghv_`, through (ff)) are interpreted and compiled, not
run on a card; the HGP `(mn|P)` gradient `hgeri3` is interpreted and
compiled only, no card.
**The fused Fock gradient `gf_` (HGP, `Target.F` at d1) is generated,
interpreted and compiled, not run**: held to `eri.eri`'s derivative
integrals per class through `(dd|dd)`, to the host's `dE_J - 1/2 dE_K`
and to PySCF; compiled by nvcc (sm_86) and hipcc (gfx1200) through `(pp|pp)`, never on a card ("The fused Fock gradient on HGP", below). This is
the derived note for the first `derivative_order = 1` keys: the contracted
nuclear gradients of the overlap, the kinetic energy, the nuclear
attraction (its Hellmann–Feynman term included), the Coulomb energy
`E_J` and, added after, the exchange energy `E_K`. It says what is
generated, what it is checked against, and what is not done. Neither
session that built it had a device compiler or a card (the kernel texts
passed clang 18's CUDA front end, `-Wall -Wextra`, a proxy); a later
session (2026-10-27) compiled and ran them.

**Nothing here is scored.** No published table covers these gradients;
the rule is McMurchie & Davidson's (`docs/references.md` entry 46) and the
kernels are this repo's own design, as the one-electron family is. No
`claims.json` record, cell or floor was added.

The spike this ports is `experimental/mmd/gradient/` (s and p, numpy),
which checked the same rule against libcint's `int1e_ip*`, `int2e_ip1`
and `pyscf.grad.RHF` to ~1e-13. That spike is not imported anywhere.

## The keys

`Spec.derivative_order` was a field with one legal value, kept so that
widening it would not retrofit every consumer. It is now `(0, 1, 2)`
(`spec.DERIVATIVE_ORDERS`; order 2 is `hessian.md`'s), and order 1 is legal on
`spec.GRADIENT_TARGETS` only: `S`, `T`, `V`, `JAB`, `K`, `METRIC` and
`ERI3`, each on its native engine, MMD (the last two: "The density-fitting
gradients", below), and `DQ`, which only the Hessian reads (`hessian.md`).
`F` takes it on its native engine, HGP ("The fused Fock gradient on
HGP", below). Every other target refuses it with `_GRADIENT_REFUSAL`, and so does
every other engine -- except six off-native pairs,
`spec.GRADIENT_OFF_NATIVE`: `J` on HGP ("The Coulomb gradient on the HGP
recurrence", below), key `coulomb/d1/(la,lb|lc,ld)/b1/j.hgp/literal`,
`T` on HGP ("The kinetic gradient on HGP", below), key
`kinetic/d1/(la,lb)/b1/t.hgp/literal`, `ERI3` on HGP ("The
three-center gradient on HGP", below), key
`coulomb/d1/(la,lb|lP)/b1/eri3.hgp/literal`, `K` on HGP ("The exchange
gradient on HGP", below), key `coulomb/d1/(la,lc|lb,ld)/b1/k.hgp/literal`, and `S` on HGP ("The
overlap gradient on HGP", below), key
`overlap/d1/(la,lb)/b1/s.hgp/literal`, and `V` on HGP ("The
nuclear-attraction gradient on HGP", below), key
`nuclear/d1/(la,lb)/b1/v.hgp/literal`. A gradient class is its integral's class, `am` the
same pair `(la, lb)`, so `overlap/d1/(2,1)/b1/s/literal` and
`overlap/d0/(2,1)/b1/s/literal` differ in the `d` segment alone and every
d0 key is byte-identical. Unbatched, for each target's d0 reason. `K` at
d1 keeps K's quartet `am = (la, lc, lb, ld)`, the Hermite pairing, bra
first (`coulomb/d1/(2,1|1,0)/b1/k/literal`); `Spec` would admit a batch
count there as it does at d0, and `kernel_gk` refuses any but 1.

## What a d1 kernel writes

Not the derivative integrals (a `[n_atom, 3, n_ao, n_ao]` tensor) but
their contraction with a matrix, the shape every SCF gradient term has:

    G[A, k] += SUM_mn M_mn dX_mn / dR_{A,k}

`M = D` for `T`, `V`, `J` and `K`, the energy-weighted density `W` for
`S`; the signs and halves of an energy are the caller's (RHF's exchange
energy is `-1/2 E_K`). Per shell pair `i <= j`
the weight of Cartesian pair `(r, c)` is `M[a_r, b_c] + M[b_c, a_r]` off
the diagonal and `M[a_r, a_c]` on it, which is `SUM_mn M_mn X_mn` exactly
for any `M` (no symmetry assumed) and any `X` symmetric in `m <-> n`.
One exception writes the integrals themselves: the DF metric's
`uncontracted` form, `ipmetric_` (#693, "The uncontracted metric
derivative", below), for the DF Hessian.

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
- `E_K`: `E_K = 1/2 SUM D_ab D_cd (ac|bd)` (`K_ab = SUM_cd D_cd (ac|bd)`,
  K's pairing: bra record `(a, c)`, ket record `(b, d)`). Its ket
  derivative equals its bra derivative for a **symmetric** `D` (rename
  `a <-> b`, `c <-> d` in the ket term, then `D_ba = D_ab`), so
  `dE_K/dR = SUM D_ab D_cd d_bra (ac|bd)` over every **ordered** shell
  quartet, the bra's two centres `A` and `C` differentiated. Per (bra,
  ket) record pair, `I_bd[p] = SUM_q (-1)^{|q|} [p+q]^(0) s_bd E^{bd}_q`
  over `|p| <= la + lc + 1` (K's stage 1, one order up), then
  `SUM_ac D_ab D_cd SUM_p (dX_k E^{ac})_p I_bd[p]`. Both `D` blocks
  straddle the records (`a`, `c` on the bra; `b`, `d` on the ket), so
  unlike `E_J` the density does not factor into a per-bra Hermite density
  and a per-bra `J_p`: the contraction stays at the quartet. No factor:
  the half of `E_K` and the bra-plus-ket two cancel, as for `E_J`. An
  asymmetric `D` is refused.

No translational invariance is used. The derivative on both centres of a
pair, and on every nucleus for `V`, is computed, so `SUM_A G[A] = 0` is a
check.

## The host reference (`gradient/host.py`)

`overlap_gradient`, `kinetic_gradient`, `potential_gradient` (`one_centre=`
injected) and `coulomb_gradient` (`one_centre=` injected: `[p|q] =
(-1)^{|q|} [p+q]^(0)`, one `[r]^(0)` table per bra–ket record pair, the
kets `ao.density.make_kets` over the unique pairs at weight 2 off the
diagonal), each over every shell pair or `pairs=`, plus
`exchange_gradient` (`one_centre=` injected; every ordered quartet or
`quartets=`, the bra derivative differentiated before its fold,
`absolute=True` the same sum with every factor in `|.|`, the kernel
test's round-off scale), `exchange_pairs` / `class_quartets` /
`exchange_records` (the `gk` launch's records, one class at a time),
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
- **From `la + lb = 5`, `gv` defaults to a low-stack form** (#614;
  `kernel_g1e.g1e_lowstack`, `LOWSTACK_MIN_LSUM = 5`: (df), (fd) and (ff)
  through f; `lowstack=` forces either form at any `gv` class, `gs` / `gt`
  refuse `True`; name, parameters, launch and `G` contract the same). The
  unrolled `gv` keeps `H0` and the six derivative densities live across
  its nucleus loop. The low-stack one forms no density: per nucleus it
  stores `[t]^(0)` into a local `R` as it forms
  (`recurrence.store_targets`), and per block row, its weight read from
  `M`, takes `gA_k -= Z_C wt (2 alpha SUM E^{a+1_k,b} R - a_k SUM
  E^{a-1_k,b} R)` (and `gB`) and the Hellmann–Feynman `f_k += wt SUM
  E^{ab}_t R[t + 1_k]` as rolled sums over the per-axis tables
  (`e_fragment.rolled_e_sum`, `up=k`), `kernel_geri3`'s low-stack shape.
  Below the gate the text is byte-identical (the goldens through (dd) are
  unchanged).

  **Measured** (`data/measured/g1e.json`, `measure/measure_g1e.py`,
  `devtools/measure.sh g1e-compile`, a compiler and no card, its rule
  declared before any row; a separate store, not `measured_store`, as
  #606 and #613 did). Every one of the 48 unrolled units through (ff) and
  the 16 low-stack `gv` units, alone, killed at 300 s; admissible at
  <= 14,000 B of stack (#613's gradient-family budget) on both arches.
  `measure.sh check` holds the gate to the rows
  (`test_the_gate_is_what_the_measured_rows_rule`).

  | unit | `sm_86` regs / spill stores / stack / compile | `gfx1200` VGPRs / spills / scratch / compile |
  |---|---|---|
  | `gs_33` | 255 / 2,180 B / 1,592 B / 1.0 s | 192 / 767 / 1,824 B / 1.5 s |
  | `gt_33` | 255 / 9,028 B / 5,296 B / 1.7 s | 192 / 1,708 / 4,776 B / 1.9 s |
  | `gv_22` unrolled | 255 / 15,772 B / 9,752 B / 4.6 s | 192 / 2,568 / 4,812 B / 4.2 s |
  | `gv_23` unrolled | 255 / 43,748 B / **21,104 B** / 18 s | 192 / 5,004 / 8,692 B / 13 s |
  | `gv_32` unrolled | 255 / 40,460 B / **18,472 B** / 17 s | 192 / 5,052 / 8,620 B / 14 s |
  | `gv_33` unrolled | 255 / 102,540 B / **50,016 B** / 144 s | 192 / 9,880 / **15,244 B** / 38 s |
  | `gv_23` low-stack | 255 / 344 B / 4,880 B / 1.8 s | 192 / 2 / 4,528 B / 1.5 s |
  | `gv_33` low-stack | 255 / 1,392 B / 7,712 B / 2.4 s | 192 / 14 / 6,448 B / 1.5 s |

  `gs` and `gt` hold no density and fit unrolled everywhere (the largest
  is `gt_33`, 5,296 B), so they have one form. `gv_22`'s 9,752 B is over
  #586's d0 8 KB budget but under this one; under 8 KB the gate would
  move (dd), which ran on both cards in #607, so the rule is #613's. Every
  low-stack unit is at most 7,712 B of stack and 2.4 s. No clock
  (#586 found the d0 low-stack `V` up to ~1.25x slower at (df)).
- `gjab` reads `Jp[n_rows, n_p1]`, `n_p1 = len(hermite_targets(la+lb+1))`,
  in `kernel_jab`'s interleaved spelling. That `J_p` is the ordinary d0
  `[p|q]` kernel of the quartet class with the bra raised by one,
  `kernel_gjab.gjab_pq_spec(la, lb, lc, ld) = (la + 1, lb | lc, ld)`, over
  the same bra records and the J chain's `DQ` kets. Nothing in the `[p|q]`
  kernel or its plan is new.
- **`gjab` has a low-stack form, the default from `la + lb = 5`** (#619;
  `build_gjab_kernel(cls, lowstack=True)`; name, parameters, launch and
  `G` contract the same). The unrolled `gjab` holds the six derivative
  densities `HA_k`, `HB_k` over `hermite_targets(L + 1)` and the pooled
  derivative factors as straight-line locals (~210 KB of text at (dd)).
  The low-stack one forms no density: per block row, its weight read from
  `M`, it takes `gA_k += wt (2 alpha SUM E^{a+1_k,b} Jp - a_k SUM
  E^{a-1_k,b} Jp)` (and `gB`) as rolled sums over the per-axis tables
  (`e_fragment.rolled_e_sum`), reading the record's `Jp` row where it lies
  in global memory: `gv`'s low-stack shape without `R` or a nucleus loop.
  The per-row weight and derivative terms are shared with `gv`
  (`kernel_g1e.lowstack_weight` / `lowstack_derivative`, the `gv` text
  byte-identical at all 32 units). The default is
  `kernel_gjab.gjab_lowstack`, an `la + lb` floor `LOWSTACK_MIN_LSUM =
  5` read from `data/measured/gjab.json` by its declared rule
  (`measure/measure_gjab.py`, `devtools/measure.sh gjab-compile`, a
  compiler and no card; #614's rule restated, declared before any row,
  every unit compiled alone under a 300 s kill, admissible at <= 14,000 B
  of stack on both arches; `measure.sh check` holds the floor to the
  rows). The unrolled form is over on `sm_86` only: (df) / (fd) 18,976 /
  21,240 B, (ff) 46,680 B with 93,788 B of spill stores and a 115 s
  compile (`gfx1200`'s worst, (ff), 11,908 B, 15 s); through `la + lb =
  4` it fits both (worst (dd), 8,648 B on `sm_86`), so those classes stay
  unrolled, their text and goldens unchanged. Every low-stack unit fits
  both, worst (ff) 6,352 B (`sm_86`) / 5,424 B (`gfx1200`), under 3 s.

`gradient/kernel_gk.py` builds `gk_{la}{lc}_{lb}{ld}` (`build_gk_kernel`),
one per K class, on `kernel_k`'s mapping: **one warp per bra record**, the
kets strip-mined over the lanes, no screen (K's bound bounds the
integral, not its derivative), no shared memory.

- The `[r]^(0)` are the plan's for the d0 K class with the bra raised by
  one, `kernel_gk.gk_pq_spec(la, lc, lb, ld) = (la + 1, lc | lb, ld)`
  (split `(lp + 1, lq)`), `recurrence_body` unchanged, as `kernel_k`
  runs it. `E^bd` is `block_fragment(lb, ld)` inlined into a local
  `Ebd`, as `kernel_k`'s default; stage 1 is `kernel_k`'s fold, into
  straight-line locals `I{n}`.
- The derivative is taken through **raised rows**: on axis `k`,
  `(dA_k E)^{ac} = 2 alpha E^{a+1_k, c} - a_k E^{a-1_k, c}`, so with
  `V(a', c) = SUM_p E^{a'c}_p I[p]` (an ordinary stage-2 fold over a row
  of shell `la +- 1`) the A side is `2 alpha SUM W V(a+1_k, c) - SUM a_k
  W V(a-1_k, c)`, `W = D_ab D_cd`, and C the same with `c` moved. Each
  raised row is folded once per `(b,d)` row and digested into every axis
  it serves, into per-lane `uA_k`, `lA_k`, `uC_k`, `lC_k`; after the ket
  loop `gA_k = 2 alpha uA_k - lA_k`. `E^{a'c}_p` is a product of the
  per-axis tables `axis_tables_fragment(la + 1, lc + 1)`.
- Six `WarpReduce`s, then lane 0 adds `s_ac gA_k` into `G[atom_a, k]` and
  `s_ac gC_k` into `G[atom_c, k]`.
- A launch is one class: every ordered bra record of `(la, lc)` against
  every ordered ket record of `(lb, ld)` (`host.exchange_records`), so
  the class launches over a basis sum to the whole `dE_K`
  (`host.class_quartets` partitions the ordered quartets).
- Params: `n_bra n_ket n_ao n_atom`, the bra side `P XPA XPC [n_bra, 3]`,
  `zeta_ac s_ac ea_bra ec_bra`, `off_a off_c atom_a atom_c`, the ket side
  `Q XQB XQD [n_ket, 3]`, `eta s_bd`, `off_b off_d`, then `M[n_ao, n_ao]`
  (the density, `D_ab` and `D_cd` read at the records' AO offsets) and
  `G[n_atom, 3]`.
- The unrolled text grows fast: 29 KB at `(ds|ss)`, 169 KB at `(pp|pp)`,
  809 KB at `(pd|pd)`, 4.0 MB at `(dd|dd)` (cuda), so it is the default
  only below `L = 5` (next section).

### The rolled `gk` (#618)

From `L = la + lc + lb + ld >= kernel_gk.LOWSTACK_MIN_L = 5`, and for
every class with an f shell, `build_gk_kernel(cls)` emits a **rolled
form** (`gk_lowstack(cls)`; `lowstack=` forces either form at any class;
the name, parameters, launch and `G` contract are the same, so a rolled
unit is a drop-in). Below the floor the default text is byte-identical
and the 16 s / p goldens did not move. The unrolled form of a class with
an f shell is never built by default (`UNROLLED_L_MAX = 2`) and, by the
user's decision, was not compiled or measured.

The rolled form holds no `E` block and no straight-line fold. Per bra
record it forms the bra's per-axis `E` tables one index up, `(la + 1, lc
+ 1)`; per ket record the recurrence (the same plan, `gk_pq_spec`) with
each `[r]^(0)` stored into a local `R[n_t(L + 1)]` as it forms
(`recurrence.store_targets`), and the ket's per-axis tables `(lb, ld)`,
every odd-`t` entry negated once (so `(-1)^|q|` is in the table). Then,
per `(b, d)` row (`cartesian_loops`), stage 1 is `I[p] = s_bd SUM_q
E^bd_q R[p + q]` for every `|p| <= lp + 1` (`hermite_loops`), and per
`(a, c)` row, with `W = M[a, b] M[c, d]` read where it is used,

    gA_k += W (2 alpha SUM_t E^{a+1_k, c}_t I[t] - a_k SUM_t E^{a-1_k, c}_t I[t])

and `gC_k` likewise on `c`. Every `SUM` is a loop nest over the tables,
factored per axis as `e_fragment.rolled_e_sum` is. The tables are
**compact** (`kernel_gk.tables_fragment`): only the reachable `E^{ij}_t`,
`t <= i + j`, at `compact_offset(i, j, l2) + t`, 125 doubles a table at
`(4, 4)` against the dense 225. That is what brings `(ff|ff)` under the
stack budget: with the dense tables `gk_33_33` was 15,008 B of `sm_86`
stack (a hand probe), compact 11,616 B. The rolled form folds each
derivative term per `(a, c)` row, where the unrolled one folds each raised
row once and digests it into up to three axes, so the two agree to
round-off, never bitwise.

**The block mapping is not built.** The issue named a second form,
`kernel_kblock`'s one warp per `K_AB` block. By the user's decision only a
rolled / low-stack block variant was in scope, and only if cheap; the
energy K's rolled block kernels measured slower than `kb` or neutral (#571,
#583), so it was not built and nothing about it is measured.

**Measured** in `data/measured/gk.json` (`measure/measure_gk.py`,
`devtools/measure.sh gk-compile`, a compiler and no card; the rule
`declared` before any row; `measure.sh check` holds the gate to the rows):
every class through `(ff|ff)` (256) in the rolled form, and the unrolled
form of the classes with every `l <= 2` level by level in `L` through the
floor's level, each unit compiled alone and **killed at 10 s** (the user's
per-compile budget for #618; a killed unit is recorded `over_budget`).
Admissible means at most 14,000 B of stack (#613's budget) and within
10 s, on both arches.

| | `sm_86` | `gfx1200` |
|---|---|---|
| unrolled, L <= 4 (50 classes) | all admissible; worst `gk_22_00` 11,696 B, 2.1 s | all admissible; worst `gk_22_00` 10,836 B, 4.3 s |
| unrolled, L = 5 | `gk_22_01` 14,616 B and `gk_22_10` 14,880 B (8.5 s, 18,400 B spill stores): **over** | all within budget (worst 11,648 B, 7.8 s) |
| unrolled, L = 6 (`gfx1200` only, the first session's per-arch stop) | not compiled | 8 of 10 **over_budget** (killed at 10 s) |
| rolled, all 256 | worst stack `gk_33_33` 11,616 B (255 registers, 1,956 B spill stores, 2.8 s, 204 KB of source); worst compile `gk_31_21` 4.8 s; median 2.5 s, 2,632 B | worst `gk_33_33` 10,736 B (192 VGPRs, 412 spilled, 2.2 s); worst compile 2.6 s; median 1.6 s, 2,680 B |

So the floor is 5 (`sm_86`'s L = 5), no default unit is over the budget
on either arch, and no rolled unit came near the 10 s cap. At `(dd|dd)`
the rolled unit is 101 KB, 4,448 B of `sm_86` stack and 3.7 s, against the
unrolled 4.0 MB. No clock was measured.

Parameters of the pair kernels (`kernel_s`'s per-axis spelling for the
one-electron three):
`n_rec n_ao n_atom`, `zeta_ab pa_* pb_* [p_*: V] s_ab ea_bra eb_ket`,
`off_a off_b atom_a atom_b`, `M[n_ao, n_ao]`, `[Z Cnuc: V]`,
`G[n_atom, 3]`. For `V` the atoms are the nuclei.

## The exchange gradient on HGP (`gradient/kernel_ghk.py`)

`ghk_{la}{lc}_{lb}{ld}` writes the same `G` as `gk` -- `dE_K/dR`, `E_K =
1/2 SUM D_ab D_cd (ac|bd)`, the bra derivative alone over every ordered
quartet, `D` symmetric -- but on the Head-Gordon–Pople recurrence of
`[Fock]` §3.1 instead of MMD's Hermite tower. Its key is `K` at d1 on
`Engine.HGP` (`ghk_spec`), K's Hermite pairing `(la, lc | lb, ld)`, the
HGP bra pair `(a, c)` exactly as `kernel_hk`'s.

**The derivative as raised and lowered classes.** On contracted
integrals

    d/dA_k (ac|bd) = (a+1_k c|bd)_{2 alpha} - a_k (a-1_k c|bd)
    d/dC_k (ac|bd) = (a c+1_k|bd)_{2 gamma} - c_k (a c-1_k|bd)

where `_{2 alpha}` is the contraction with each primitive quartet weighted
by twice its `A` exponent. The HRR (eq 28) is linear and its coefficients
are geometry alone, so it holds for any per-primitive weight: each of the
three contractions is an ordinary HGP boundary set. Per contracted quartet
the kernel runs ONE VRR, `vrr_class(am) = (e_lo, la + lc + 1 - e_lo | lb,
ld)` with `e_lo = max(la - 1, 0)`, whose targets cover every boundary
needed; sums its ket primitives into `T(e, f)` per bra primitive pair;
folds `T` into three contracted sets after the ket loop (`2 alpha T` over
the `(la + 1, lc)` boundary, `2 gamma T` over `(la, lc + 1)`'s, `T` over
the lowered classes'), so a weight multiplies once per bra primitive pair;
then runs up to four HRRs (`hrr_classes`: `(la ± 1, lc)`, `(la, lc ± 1)`,
the lowered only where the shell has a unit to lower), each on its own
locals. The deposit is `W = M[a, b] M[c, d]` times the two derivative
rows per axis, into six thread-private sums carried over the thread's
kets, then six `AtomicAdd`s into `G[atom_a]`, `G[atom_c]`.

**The rest is `kernel_hj`'s, imported**: the nest (`hgp_nest`, its quartet
scalars in `eri/hgp_drive.py`'s operation order), the staged `|ac)`
record (`kernel_f`'s `REC`, one block per bra pair, every thread copying
it), one thread per contracted quartet, bra primitives outer. One class
mode only (`ordered`'s: an orbit representative would move the bra's two
centres), no lever, no screen (`gk`'s reason), one batch. Beside the
record it reads `bra_exp[n_bra, k_ab, 2]` (`alpha`, `gamma`) and
`bra_atom`; `gradient.host.hgp_exchange_launches` packs a class's
launches, one per `(k_ab, k_cd)` pair of contraction degrees (a side is
degree-uniform, `kernel_f.pack`'s rule).

    params  n_bra n_ket k_ab k_cd n_ao n_atom (I32)
            bra_prim[n_bra, k_ab, REC] bra_exp[n_bra, k_ab, 2] AB[n_bra, 3]
            bra_ao bra_atom [n_bra, 2] (I32)
            ket_prim[n_ket, k_cd, REC] [CD[n_ket, 3]: ld > 0]
            ket_ao[n_ket, 2] (I32)
            M[n_ao, n_ao]  G[n_atom, 3] (OUT, AtomicAdd-ed, zeroed by the caller)

**Checked** (`test_kernel_ghk.py`), interpreted against
`host.exchange_gradient` (MMD: another recurrence, another order), gate
`16 eps` of the host's `absolute=True` twin, `gk`'s gate unchanged: every
K class with each `l <= 1` on the two-atom s, p basis and every class
with a d shell through `(dd|dd)` on the two-atom s, p, d basis (L + 1 > 4
`slow`), worst 4.4 eps (`(ss|ss)`), 0.4 eps at `(dd|dd)`; the 16 s, p
classes summed to the whole `dE_K` and translationally invariant; a
mixed-degree basis (1, 2, 3 primitives) cut into several launches a
class; 1, 2 and 32 threads a block. Against libcint (`slow`): `(dd|dd)` on
CO/cc-pVTZ, all 256 ordered quartets, against the tracked
`int2e_ip1_cart` witness (`manifest.k_gradient`) within `1e-12` of the
whole `max|G|` (that test's gate).

**Emitted and compiled** (`test_emit_grad_hk.py`): `python -m
felizjueves.emit --grad-hk-l [LMAX]` (1 bare, 16 units; accepted 0..2,
exit 2 above), never in the bare default nor in `KERNEL_GEN_CMD`; goldens
`test/golden/{cuda,hip}/ghk_*` for the 16 s, p classes, 32 files. nvcc
(`sm_86`) and hipcc (`gfx1200`) compile every golden and the d classes `(ds|ss)`, `(sd|ss)`,
`(dp|ps)`, `(dd|ss)`, `(dd|pp)` (L + 1 > 4 `slow`), each alone under a
600 s cap, no warning in generated text (2026-10-30, a 4-core cloud
container, no sccache: the s, p units about 2 s each on nvcc, at most
14 s on hipcc; `ghk_22_11` 11.6 s / 44 s).
**`ghk_22_22` did not compile within 600 s** (ptxas on the straight-line
unit), so `(dd|dd)` is interpreted and libcint-witnessed but not in the
compile set.

**Not done**: no card has run it; nothing measured (registers, stack,
clock); the unit is straight-line, so its text grows with the raised
classes' HRRs and nothing rolled or low-stack is built; nothing above
`(dd|dd)` is built or checked, and `(dd|dd)` itself does not compile in
600 s (a rolled form, `gk`'s #618 move, is the follow-up); not wired into
`gradient.rhf`'s
interpreted runners (`gk` stays the default exchange term).

## The generator

`python -m felizjueves.emit --grad-l [LMAX]` (2 bare, accepted 0..3,
exit 2 above) adds the four families for every `la <= lb <= LMAX` beside
the J ladders. It is never in the bare default and not in
`KERNEL_GEN_CMD`. The raised `[p|q]` class `gjab` reads is written by the
class selector (`--class LA+1 LB LC LD`), not by this flag. Goldens:
`test/golden/{cuda,hip}/{gs,gt,gv,gjab}_{la}{lb}` for (ss) through (dd),
48 files, the largest 211 KB.

`python -m felizjueves.emit --grad-k-l [LMAX]` (1 bare, accepted 0..3
since #618, exit 2 above) adds `gk_{la}{lc}_{lb}{ld}` for every K class
with each `l <= LMAX` (16 units bare, 81 at 2, 256 at 3), each in its
default form, beside the J ladders, never in the bare default nor in
`KERNEL_GEN_CMD`. Goldens: `test/golden/{cuda,hip}/gk_*` for the 16 s / p
classes (unrolled) and the four classes with one f shell and nothing else
(`L + 1 = 4`, rolled), 40 files; every other default unit through
`(ff|ff)` (236 a backend) is pinned by size and sha256 in
`test/golden/gk_digests.json` (`slow` above `L + 1 = 4`;
`test_emit_grad_k.py`).

`python -m felizjueves.emit --hgp-grad-l [LMAX]` (1 bare, accepted 0..3,
exit 2 above) adds `hgt_{la}{lb}` and `hgs_{la}{lb}`, the kinetic and overlap
gradients on HGP, for every
`la <= lb <= LMAX`, beside the J ladders, never in the bare default nor in
`KERNEL_GEN_CMD`. Goldens: `test/golden/{cuda,hip}/hgs_{00,01,11}`, 6
files (`test_emit_hgs.py`).

`python -m felizjueves.emit --grad-f-l [LMAX]` (1 bare, accepted 0..2,
exit 2 above) adds `gf_{la}{lb}{lc}{ld}` for every F class of milestone
12's ladder with each `l <= LMAX` (6 units bare, 21 at 2), beside the J
ladders, never in the bare default nor in `KERNEL_GEN_CMD`. Golden:
`test/golden/{cuda,hip}/gf_0000` (2 files, 17 KB each); the other 20
units are pinned by size and sha256 in `test/golden/gf_digests.json`
(`slow` above `L + 1 = 4`; `test_emit_grad_f.py`, which also compiles
the six bare-ladder units under both compilers). An unrolled `gf_` is
large: `(pp|pp)`'s CUDA text is ~590 KB.

## The RHF gradient, assembled (`gradient/rhf.py`, #565)

With `D` and `W` the total density and energy-weighted density,

    dE/dR = - SUM W dS/dR + SUM D dT/dR + SUM D dV/dR
            + dE_J/dR - 1/2 dE_K/dR + dV_nn/dR

`rhf_gradient(basis, nuclei, D, W, runners)` applies those signs and the
half once (`rhf.SIGN`) to five injected runners, each `(basis, nuclei, M) ->
G` with the d1 kernels' contract, adds the host's `V_nn^x`
(`host.nuclear_repulsion_gradient`), and returns every signed term and
their sum in `rhf.TERMS` order. `V`'s runner carries the Hellmann–Feynman
term (inside `gv_`). Two sets of runners exist: `rhf.host_runners` (the
references) and `interpreted.interpreted_runners(boys=)`, every term from
the generated kernels in `ir.interp`, each in its default form, launched
class by class into one zeroed `G` as `test/grad_driver.py` packs them for
a card. J is the chain, not the host's `J_p`: the d0 `[p|q]` of
`kernel_gjab.gjab_pq_spec`'s class (the bra raised by one) through
`eri.j_coulomb.drive.KernelJP`, then `gjab_`.

**On a card (#566, `test/test_rhf_gradient_device.py`).** The device
runners feed the same `rhf_gradient`, one program per term, every kernel
in its default form, the terms reduced on the host: `gs_` / `gt_` / `gv_`
per pair class (`grad_driver.one_electron_launches`; `gv_` carries the
Hellmann–Feynman term); the J chain wholly on the card (`DQ`, the plan
keyed on its `Dmax`, every bra class raised by one by
`grad_driver.raised_plan`, the raised `[p|q]`, then `gjab_` reading `J_p`
where it lies, `grad_driver.gjab_tail`; no host `J_p`); every `gk_` class
through `gk_driver`, each unit compiled alone under its 10 s budget, the
per-class `G` summed on the host in launch order; `V_nn^x` the host's.
Inputs are #550's H₂O/STO-3G and H₂O/6-31G fixtures, both `slow`
(`tiers.by_l`: `gk_` at `(pp|pp)` and the raised `[p|q]` reach L = 5).
Bounds, set before the run: every term and the total within `1e-10`
absolute of PySCF and of the interpreted `rhf_gradient`, and run to run
within `SPREAD_TOL = 1e-13` absolute.

Run on the RX 9060 XT (gfx1200) only, 2026-10-30, `kernel-tier.sh
--backend hip --full`, 2 passed: total against PySCF 8.17e-14 (STO-3G) /
4.66e-14 (6-31G), against the interpreted gradient 1.78e-15 / 3.11e-15,
run to run 6.9e-17 / 4.4e-15; worst term `potential` against PySCF
(7.64e-14 / 4.97e-14). **CUDA was not run** (NVML failed to initialise in
that container; PR #696 is labelled `unverified-on-card`).

**Decision (the user's, #566): run-to-run bound, not bit-identity.** #566
asked for the device gradient to be bit-identical across two launches.
Because every d1 kernel `atomicAdd`s into `G` (the departure below), the
order of the additions is the hardware's and a device gradient is **not
bit-reproducible**: the first hip run's totals differed by 2.8e-17
(STO-3G) and 1.8e-15 (6-31G). The criterion was restated as `SPREAD_TOL =
1e-13` absolute on every term and the total; the test still reports
whether the totals came out bit-identical (they did not). Getting bit
reproducibility back means the plan's partials and a fixed-order
reduction, a kernel change nobody has made.

This departs from the milestone's plan where #552 and #557 closed: the
kernels `atomicAdd` into `G` (no partials and no fixed-order reduction, so
a device term is not bit-reproducible; the interpreted one is), the
Hellmann–Feynman term is `gv_`'s, with no `vhf_` kernel, and the module
lives in `gradient/`, beside the kernels, not in a `nuclear_gradient/`
package.

**The witness** is #550's gradient fixtures (`manifest.gradient_fixture`):
their converged `D` and `W` (`dme`) are the inputs, and each term is held
to the fixture's (libcint's derivative integrals contracted by PySCF's own
route), the total to `nuc_grad_method().kernel()`'s. #550 records `T + V`
as one term, `grad:h`; the test splits off `T` from the fixture's stored
`int1e_ipkin` so each is held separately. Molecules: #550's H₂O in STO-3G,
6-31G and cc-pVDZ and C1 NH₂F/6-31G, plus #565's `h2o_sto3g_d_grad`:
H₂O/STO-3G with one primitive d on O (`Case.extra_shells`), which reaches
every class cc-pVDZ does -- through `(dd|dd)`, the raised `[p|q]` at L =
9 -- at about STO-3G's cost. cc-pVDZ adds only contraction depth (O's
nine-primitive s), and its interpreted `gk_` runs for hours.

## Witnesses, and what ran where

| What | Against | Where | Worst |
|---|---|---|---|
| host `S`, `T`, `V`, `E_J` (`test_gradient_host.py`) | central differences, `h = 1e-4`, of `overlap_matrix`, `kinetic_matrix`, `potential_matrix`, `coulomb_matrix` (each libcint-witnessed) | H₂O/STO-3G, a two-atom s, p basis; H₂O/cc-pVDZ and H₂O/STO-3G's `E_J` `slow` | ~2e-9 to 6e-9 of `max\|G\|` (the `O(h²)` floor), gate 1e-7 |
| host invariance | `SUM_A G = 0` | same | ~1e-15 of `max\|G\|`, gate 1e-13 |
| `V`'s Hellmann–Feynman term | a nucleus with no basis function, by finite difference | two-atom s, p basis plus a bare nucleus | as above |
| `gs`, `gt`, `gv` interpreted (`test_kernel_g1e.py`) | `host`, per class | every ordered class (ss)–(ff) on HF/cc-pVTZ; whole H₂O/STO-3G; `gv` also in its other form at every class (#614) | `gs`, `gt` bitwise; `gv` 2.2 eps of the split-centre scale, gate 16 eps |
| `hgs` interpreted (`test_kernel_hgs.py`) | `host`, per class, and `gs` interpreted on the same pairs | every ordered class (ss)–(ff) on HF/cc-pVTZ; whole H₂O/STO-3G and CO/cc-pVTZ (`slow`), `SUM_A G = 0` | 17.3 eps of the split-centre scale at (sd), every other class <= 9.8; gate 64 eps |
| compile, nvcc / hipcc (`test_emit_hgs.py`) | #92's filter | `sm_86` and `gfx1200`, every `hgs` unit (ss)–(ff) | passed, no warning in generated text (2026-10-30, a device-less cloud container) |
| `gjab` interpreted (`test_kernel_gjab.py`) | `host`, per bra class, host `J_p` | every ordered class (ss)–(ff) on a three-atom s–f basis; also in its other form (low-stack) at every class (#619) | unrolled 1.8 eps, low-stack 5.7 eps of the split-centre scale, gate 16 eps |
| the chain: `[p|q]` at `gjab_pq_spec` (interpreted) → `gjab` | `host.coulomb_gradient` | the two-atom s, p basis; H₂O/STO-3G `slow`; each in both forms (#619) | gate 64 eps of the scale |
| host `E_K` (`test_gradient_host.py`) | central differences of `1/2 SUM D exchange_matrix` (libcint-witnessed) | the two-atom s, p basis; H₂O/STO-3G `slow` | 5.2e-9 of `max\|G\|` on the s, p basis, gate 1e-7 |
| host `E_K` invariance, partition | `SUM_A G = 0`; the class quartets cover every ordered quartet once | the s, p basis | ~5e-16 of `max\|G\|` |
| `gk` interpreted (`test_kernel_gk.py`) | `host.exchange_gradient`, per class, its `absolute=True` twin as scale | every K class with each `l <= 1` on the s, p basis; every class with a d shell through `(dd\|dd)` on a two-atom s, p, d basis (L + 1 > 4 `slow`); the 16 s, p classes summed to the whole `dE_K`; 1, 4 and 32 lanes | at most 0.7 eps of the scale, gate 16 eps |
| rolled `gk` interpreted (`test_kernel_gk.py`, #618) | the unrolled `gk` on the same records (the differential), the class's `absolute=True` twin as scale; and at f `host.exchange_gradient` | every K class with each `l <= 2` (L + 1 > 4 `slow`); f witnesses `(ff\|ff)` (bra `(f_A, f_B)` against both off-centre kets), `(fs\|fs)`, `(fd\|pd)` on a two-atom single-primitive s, p, d, f basis (`slow`) | all within the gate, 16 eps (the 107 slow cases passed, 2026-10-29) |
| libcint `gk` witness (`test_k_gradient_witness.py`, #618) | `int2e_ip1_cart` on CO/cc-pVTZ, `co_ccpvtz_j`'s density, per class and whole | the file: 256 classes, sum, invariance; the host on `(dd\|dd)`, `(pd\|pd)`, `(ds\|ds)`, `(dp\|sp)` (`slow`) | host 2.6e-16 of `max\|G\|`, gate 1e-12 |
| libcint `gjab` witness (`test_j_gradient_witness.py`, #639) | `int2e_ip1_cart` on CO/cc-pVTZ, `co_ccpvtz_j`'s density, contracted into `dE_J/dR` per `gjab` bra class and whole (`manifest.j_gradient`) | the file: 10 classes, sum, invariance; the host on every class, so the whole `host.coulomb_gradient` (`slow`); the file against PySCF today (`oracle`) | host 1.25e-15 of `max\|G\|` (worst `(ss)`), gate 1e-14; host ~37 CPU-min |
| `gk` on a card (`test_gk_device.py`, `test/gk_driver.py`, #618; on the shared runner since #657, not yet re-run on a card) | H₂O/STO-3G: the host (whole) and the other form (per class); CO/cc-pVTZ (`slow`, `high_l`): the libcint witness per class and whole, and the unrolled form on the same upload | RTX 3080 and RX 9060 XT; every unit its own TU under a 10 s kill | H₂O 3.22e-15 / 3.37e-15 of `max\|G\|`; CO 2.63e-15 / 2.94e-15 against libcint, differential over 62 / 58 classes 9.2e-17 / 1.2e-16 |
| the RHF gradient, interpreted (`test_rhf_gradient.py`, #565) | #550's gradient fixtures: each term from libcint at the recorded `D`, `W` (`grad:h` split into `T`, `V`); the total `nuc_grad_method()`'s | H₂O/STO-3G, H₂O/6-31G, C1 NH₂F/6-31G, H₂O/STO-3G + one d on O, H₂O/cc-pVDZ; one-electron and nuclear terms on s, p bases fast, J, K, the totals and every d case `slow` | H₂O/STO-3G worst term 7.5e-14 (`V`), total 7.9e-14, net 2.7e-15 (118 s interpreted); the d case 6.9e-14 / 6.9e-14 (1162 s, `gk_`'s rolled form from L = 5 the bulk); H₂O/6-31G passes; every one-electron term on all five passes; NH₂F and cc-pVDZ totals not run to the end (hours interpreted); gate `1e-10` absolute |
| the RHF gradient on a card (`test_rhf_gradient_device.py`, #566) | as the interpreted row, and the interpreted `rhf_gradient`; two runs against each other | RX 9060 XT only (CUDA not run): H₂O/STO-3G and H₂O/6-31G, both `slow`, every term from device runners (the J chain wholly on the card), 2 passed | total 8.17e-14 / 4.66e-14 of PySCF, 1.78e-15 / 3.11e-15 of interpreted; run to run 6.9e-17 / 4.4e-15, gate `SPREAD_TOL = 1e-13` (not bit-identical: the atomics) |
| compile, `-Wall -Wextra` | clang 18 CUDA front end, device side, runtime stub | all 40 pair units through (ff); the 16 `gk` goldens | no diagnostic (a proxy) |
| compile, nvcc / hipcc (`test_emit_grad.py`, `test_emit_grad_k.py`) | #92's filter | `sm_86` and `gfx1200` | passed, no warning in generated text |
| on a card (`test_grad_device.py`, `test/grad_driver.py`, on the shared runner since #659, which ran on both cards on 2026-10-30) | `host`, `1e-13` of `max\|G\|`; `gk` on H₂O/STO-3G only | RTX 3080 and RX 9060 XT: `gs`, `gt`, `gv`, `gjab` on H₂O/STO-3G and cc-pVDZ (`slow`, its (dd) classes), `gk` on H₂O/STO-3G (through `(pp\|pp)`); 9 cases per card | 3.20e-14 (CUDA), 2.36e-14 (HIP), both `gt` on cc-pVDZ; `gk` 4.1e-15 / 4.0e-15; the atomics' run-to-run spread up to 2e-14 |
| on a card through (ff) (`test_grad_device.py`, `co_ccpvtz_proton-*`, `slow`; #614) | `host`, `1e-13` of `max\|G\|` | RTX 3080 and RX 9060 XT: `gs`, `gt`, `gv` on CO/cc-pVTZ (f on both atoms, all 16 ordered classes (ss)–(ff) asserted) with a bare proton as a third nucleus, its own density as `M`, every class in its default form (`gv` (df), (fd), (ff) low-stack), each compile under a 300 s timeout; 12 passed per card with the cases above | `gt` 1.29e-14 (CUDA) / 7.64e-15 (HIP); `gv` 1.94e-15 / 2.80e-15; `gs` 6.2e-16 on both |
| the `gjab` chain on a card (`test_gjab_chain_ondevice.py`, `grad_driver.raised_plan` / `gjab_tail` on `plan_driver.driver`; #619) | `host.coulomb_gradient` (H₂, H₂O) and CO's tracked libcint witness (`manifest.j_gradient`, #639), `1e-13` of `max\|G\|` | RTX 3080 and RX 9060 XT: `DQ` on the card (read back, the plan keyed on its `Dmax`), the plan's bra classes raised by one, then `gjab` reading `J_p` where it lies, one process, no host `J_p`; H₂/STO-3G (the PR gate) and H₂O/STO-3G (`slow`) in both forms, CO/cc-pVTZ (`slow`, `high_l`, hand-run: all 16 ordered classes, (df), (fd), (ff) low-stack, raised `[p\|q]` to L = 13) in default forms; every compile under a 300 s timeout; 5 passed per card | H₂O 2.1e-15 / 2.0e-15; CO 9.0e-16 (CUDA) / 1.07e-15 (HIP); the CO cell ~17 / 20 min, almost all the host reference, before #639 moved it to the witness (not re-run on a card since) |

## The density-fitting gradients: `(P|Q)` and `(mn|P)`

**Status: host reference and generated kernels, interpreted, compiled by
nvcc and hipcc and run on both cards, through (ff|i) and (i|i) since #613**
(the first session had neither compiler, and clang 18's CUDA front end
passed all 31 L <= 4 units, a proxy; the card runs are the table's last
rows). The ladder is FreeBreakfast's DF-CASSCF one: `geri3` for `la <= lb
<= 3`, `lP <= 6` (70 classes), `gmetric` for `lP <= lQ <= 6` (28).

`Target.METRIC` and `Target.ERI3` at d1. Each kernel contracts the
derivative integrals the way the others do:

    G[A, k] += SUM_PQ  W_PQ    d(P|Q)  / dR_{A,k}      gmetric_{lP}{lQ}
    G[A, k] += SUM_mnP Gam_mnP d(mn|P) / dR_{A,k}      geri3_{la}{lb}{lP}

with any `W` (`n_aux x n_aux`) and any `Gam` (`[n_ao, n_ao, n_aux]`, the
d0 kernel's layout). A DF-J energy gradient is the two together:

    E_J = SUM D_mn (mn|P) d_P - 1/2 d (P|Q) d,   (P|Q) d = (P|mn) D_mn
    dE_J/dR = geri3 at Gam = D (x) d   +   gmetric at W = -1/2 d d^T

(`E_J` is stationary in `d`, so `d` is not differentiated). A DF-K one
(#620) is the same two kernels at the exchange weights: with `D = occ C
C^T` (`C` the occupied coefficients) and `E_K = 1/2 SUM D_mn D_ls (ml|ns)`,

    X_ijP = SUM_ml C_mi C_lj (ml|P),   (P|Q) d_ij = X_ij,
    dE_K/dR = geri3 at Gam^K_mlP = occ^2 SUM_ij C_mi C_lj d_ijP
            + gmetric at W^K_PQ = -1/2 occ^2 SUM_ij d_ijP d_ijQ,

and the closed-shell two-electron gradient is `G_J - 1/2 G_K`.
`host_df.df_j_stage` / `df_k_stage` stage the weights (pure Python, a
Cholesky solve of the metric) and return them with the energy, so a
consumer staging its own (FreeBreakfast, dense GPU linear algebra) has
them to compare against; `host_df.df_rhf_gradient` returns `G_J`, `G_K`,
each split into its `geri3` and `gmetric` halves, and `G_J - 1/2 G_K`.
Nothing stages them on a device: the solve stays host-side.

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

**The low-stack form (#613).** The unrolled kernels above hold the
weights, `H0` and the six derivative densities per aux row, for every bra
record, in registers or on the stack, and the text grows to 26 MB at
(ff|i) (whose build also overflowed Python's recursion limit: a 3,360-term
fold, now chunked by `kernel_gmetric.chunked_fold`, the same left fold
bitwise and the same text below 129 terms). `build_geri3_kernel(cls)` /
`build_gmetric_kernel(cls)` therefore emit, from an `L` floor, a form with
the SAME name, parameters, launch and `G` contract and no density or
weight held: per (bra, aux) record pair the tower stored into a local `R`
(`recurrence.store_targets`, order `L + 1`), per aux row `U[t] = SUM_tau
[t + tau]^(0) E^q_tau` (`|t| <= la + lb + 1`; `gmetric`: `|t'| <= lP + 1`
of its parity), and per element its weight read from `M` where it is used
and its terms as rolled sums over per-axis tables: for `geri3`, the bra's
`axis_tables_fragment(la + 1, lb + 1)`, `dA_k` as `2 alpha SUM E^{a+1_k,
b} U - a_k SUM E^{a-1_k, b} U` and the aux centre as `SUM E^{ab}_t U[t +
1_k]` (`e_fragment.rolled_e_sum(..., up=k)`, new; the default text
unchanged); for `gmetric`, `SUM_t E^a_t U[t + 1_k]` over the P record's
single-center tables. The forms differ only in association (round-off).

**The gate is an `L` floor set from measured rows**
(`data/measured/df_grad_lowstack.json`, `devtools/measure.sh
df-grad-lowstack-compile`, `measure/measure_df_grad_lowstack.py`, the
rule declared before any row; `measure.sh check` holds the gate to the
rows). Every class's low-stack form and the unrolled form level by level in
`L`, through the first level with an inadmissible class, each unit
compiled alone under a 300 s budget; an unrolled form is admissible at <=
14,000 B of stack and within the budget on `sm_86` AND `gfx1200`.

- `geri3`: `kernel_geri3.LOWSTACK_MIN_L = 5`. Every L <= 4 class is
  admissible (the largest unrolled frame 9,776 B, (dd|s) on `sm_86`); at
  L = 5 the unrolled (sf|d) is 14,184 B, (pd|d) 24,408 B, (pf|p) 26,896 B,
  (dd|p) 27,144 B and (df|s) 22,520 B on `sm_86` (the RX 9060 XT's first
  inadmissible level is L = 6). 49 of 70 classes gated.
- `gmetric`: `kernel_gmetric.LOWSTACK_MIN_L = 10`. Every unrolled class
  through L = 9 is admissible ((f|i) 12,432 B, (g|h) 11,520 B on `sm_86`);
  (g|i) is 15,968 B and (h|h) 18,200 B. 4 of 28 classes gated.
- No class at L <= 4 is gated, so every golden is unchanged.
- Every low-stack unit of the ladder is within both budgets: at most
  13,792 B of `sm_86` stack (`geri3_336`, (ff|i), 255 registers, 1,916 B
  of spill stores, 3.1 s, 196 KB of text) and 12,816 B of `gfx1200`
  scratch, compiled in at most 5.2 s (`sm_86`) / 2.1 s (`gfx1200`).
  `gmetric_66` (i|i) is 7,872 B / 7,296 B. (ff|i) is close to the
  ~14 KB target: `R` at order 13 (4,480 B) and the bra tables one index up
  (5,400 B, about half of them slots no row reads) are most of it.
- Before / after on `sm_86`: (df|s) unrolled 22,520 B with 46,640 B of
  spill stores in 12.9 s (468 KB of text), low-stack 5,136 B in 1.9 s
  (80 KB); (pd|d) 24,408 B against 2,912 B; (h|h) 18,200 B against
  5,312 B. Every row is in the store.
- The clock is NOT measured: #606 found its d0 low-stack forms 1.46x to
  4.58x slower, and nothing says these are not.

**The generator.** `python -m felizjueves.emit --grad-df-l [LMAX]
[--grad-df-aux-l AUXMAX]` (#613: the split cap of FreeBreakfast's
`df_kernels.py`; LMAX the orbital `l`, 1 bare, 0..3; AUXMAX the aux `l`,
2 bare or omitted, 0..6; exit 2 above either, or for `--grad-df-aux-l`
alone) adds `gmetric_{lP}{lQ}` for `lP <= lQ <= AUXMAX` and
`geri3_{la}{lb}{lP}` for `la <= lb <= LMAX`, `lP <= AUXMAX`, each in its
builder's default form. Never in the bare default nor in
`KERNEL_GEN_CMD`. Goldens: `test/golden/{cuda,hip}/{gmetric,geri3}_*`,
the 31 classes at `L <= 4`, 62 files, the largest 232 KB.

| What | Against | Where | Worst |
|---|---|---|---|
| host `(P\|Q)`, `(mn\|P)` (`test_gradient_host_df.py`) | central differences, `h = 1e-4`, of `metric_matrix`, `eri3_tensor` (libcint-witnessed) | a three-atom s/p/d basis and s/p/d aux; `(mn\|P)` through d `slow` | 2.1e-9, 2.7e-9 of `max\|G\|`, gate 1e-7 |
| host DF-J `dE_J/dR` | central differences of `E_J`, the solve redone per geometry | the s, p sub-bases; H₂O/STO-3G against the JKFIT s, p shells `slow` | 3.1e-9, 2.5e-9 |
| host DF-K `dE_K/dR` (#620) | central differences of `E_K` (numpy, the solve redone per geometry) | the s, p sub-bases (and `G_J - 1/2 G_K`); H₂O/STO-3G against the JKFIT s, p shells `slow` | 3.6e-9, 2.6e-9 |
| host invariance | `SUM_A G = 0` | same | gate 1e-13 of `max\|G\|` |
| host DF-RHF two-electron gradient (`oracle.manifest.df_rhf`, #620) | PySCF's `density_fit()` RHF gradient (`get_jk` / `get_veff`, aux response on) over the SAME aux set, at closed-form occupied coefficients | H₂O/STO-3G against cc-pVTZ-JKFIT's s, p shells (57 aux), `slow` (~15 s); `G_J`, `G_K` and `G_J - 1/2 G_K` apart, and both energies | 4.4e-13 (J), 6.2e-13 (K), 7.8e-13 (two-electron) absolute, gate `1e-10` |
| `geri3` / `gmetric` interpreted with the DF-K weights (`test_kernel_df_k_grad.py`, #620) | `host_df`, whole gradient | `Gam^K`, `W^K` and `Gam^J - 1/2 Gam^K`, `W^J - 1/2 W^K` on three-atom s, p bases (through (pp\|p), (p\|p)) | 0.37 eps of the split-centre scale, gate 16 eps |
| `gmetric` interpreted (`test_kernel_gmetric.py`) | `host_df`, per class | every class `lP + lQ <= 4` (`lP + lQ = 4` `slow`), swapped and off-diagonal pairs | 4.9 eps of the split-centre scale, gate 16 eps |
| `geri3` interpreted (`test_kernel_geri3.py`) | `host_df`, per class | every class `la + lb + lP <= 4` (= 4 `slow`); whole H₂O/STO-3G against JKFIT s, p `slow` | 2.6 eps of the split-centre scale, gate 16 eps |
| compile, `-Wall -Wextra` | clang 18 CUDA front end, device side, runtime stub | all 31 units | no diagnostic (a proxy) |
| compile, nvcc / hipcc (`test_emit_grad_df.py`) | #92's filter | `sm_86` and `gfx1200` | passed, no warning in generated text |
| on a card (`test_grad_df_device.py`, `grad_driver`'s program, on the shared runner since #659, which ran on both cards on 2026-10-30) | `host_df`, `1e-13` of `max\|G\|`, DF-J of H₂O/STO-3G against JKFIT s/p/d | RTX 3080 and RX 9060 XT, `gmetric` and `geri3` (`slow`) | 6.58e-15 (CUDA), 4.87e-15 (HIP), both `gmetric` |
| every ladder class above L = 4, interpreted, default form (`test_kernel_df_grad_ladder.py`, `slow` from L + 1 = 5) | `host_df`, per class, 16 eps of the TOWER-TWIN split-centre scale (the host on `\|Gam\|` with `test_kernel_metric.pq_tower`; the signed scale is blind to the tower's own cancellation from L ~ 9, where the unrolled form misses it as well) | 49 `geri3`, 19 `gmetric` classes on one-primitive bases with swapped and diagonal pairs; every L <= 4 class in the low-stack form forced; four classes forced unrolled above the floor | worst 0.96 eps of the tower-twin scale ((pf|s), forced low-stack), gate 16 eps |
| the DF gradient witness (`oracle.manifest.df_gradient`, #613) | libcint's `int3c2e_ip1_cart` / `int3c2e_ip2_cart`, `int2c2e_ip1_cart`, contracted per class with closed-form `Gam`, `W`, on `prim_spd_f` against the aux-i basis (every ladder class) | the host, every class (`slow`); the kernel interpreted at (df\|i), (ff\|h), (ff\|i), (h\|i), (i\|i) (`slow`) | host 3.91e-14 (`geri3`), 1.71e-14 (`gmetric`) of the witness's largest `max\|G\|`; the kernel at most 2.72e-15 ((h\|i)); gate `1e-12` |
| on a card, the ladder (`test_grad_df_device.py` Part 2, `slow` + `high_l`) | `host_df` at `1e-13` of `max\|G\|` per program, and the witness at `1e-12` | RTX 3080 and RX 9060 XT: all 70 + 28 classes in their default forms on the aux-i system, one program per aux `l`, each compile under a 300 s timeout; 16 passed per card (with Part 1) | host 1.20e-14 (CUDA) / 9.34e-15 (HIP), (`geri3`, aux i); libcint 5.49e-14 / 5.48e-14 of max; runs bit-identical on both |

`gmetric` has an HGP twin, `hgmetric_{lP}{lQ}` (`gradient/kernel_hgmetric.py`,
the Cartesian rule on the HGP VRR instead of the Hermite shift; same
parameters and `G` contract), interpreted through L = 4 and compiled for s
and p, on no card: `hgp_df.md`, "The metric's nuclear gradient on HGP".

### The uncontracted metric derivative (`gradient/kernel_ipmetric.py`, #693)

The DF Hessian needs `d(P|Q)/dR` in more contractions than `gmetric_`'s
one (`SUM_Q (P|Q)^x d_Q` for the response of `d`, `SUM_Q (P|Q)^x c_Q,ij`
inside `c^x = M^-1 (T^x - M^x c)`, the `M^x M^y` cross terms), each a gemm
once the derivative exists as a matrix. `ipmetric_{lP}{lQ}` writes it:

    M1[k, P, Q] = d(P|Q)/dA_k        A the centre of P,  [3, n_aux, n_aux]

keyed `coulomb/d1/(lP,lQ)/b1/metric/uncontracted` -- the kernel-body lever
`uncontracted` (`levers.VOCABULARY`, legal on the MMD `metric` slot -- and,
since #687, on `s`, `t` and `v`, their ip1 form, `hessian.kernel_ip1e` --
and at d1 only, by the lever's new `orders` field, `levers.order_refusal`).
`gmetric_`'s Hermite shift and tower one order up, in its low-stack rolled
form at every class (no `3 nP nQ` accumulators: 2,352 at (i|i)), each
element accumulated in place in the pair's own slots, which it zeroes
first, and plain stores, no atomics. The other centre needs no second
kernel: `(P|Q)` reads `A - B`, so `M1[k, Q, P] = -M1[k, P, Q]`, written as
the negated mirror of an off-diagonal pair (bitwise); a diagonal pair forms
its whole block. libcint's `int2c2e_ip1` differentiates the electron
coordinate, so `M1 = -int2c2e_ip1`. Atom `X`'s block is the rows on `X`
and their mirror; that slicing, and a launch of only some classes (the
kernel leaves every other element untouched), are the caller's.

| What | Against | Where | Worst |
|---|---|---|---|
| host `host_df.metric_derivative` (each centre by the Cartesian rule) | libcint `-int2c2e_ip1_cart`, the numeric golden store (`metric_ip1{x,y,z}_h2o_ccpvdz_jkfit_bse`, one axis per entry: the whole array is over the per-array budget) | cc-pVDZ-JKFIT on H₂O, every class through (f\|f) (`slow`; s, p fast) | 7.21e-14 of max, gate `1e-12` |
| host | the aux-i witness (`manifest.df_gradient`, libcint `int2c2e_ip1_cart` contracted per class), `M1` contracted with the same `W` | all 28 classes through (i\|i) (`slow`) | 1.72e-14 of the largest class `max\|G\|`, gate `1e-12` |
| kernel, interpreted (`test_kernel_ipmetric.py`) | the host, per class, every owned element; every other element untouched | all 28 classes `lP <= lQ <= 6`: L <= 4 on `test_kernel_gmetric.py`'s basis, above on two-shell bases with the absolute-value tower's scale | 3.62 eps ((d\|d)), gate 16 eps |
| kernel, interpreted | libcint: the golden over all of cc-pVDZ-JKFIT (`slow`; s, p fast), and the aux-i witness at (s\|s), (p\|p), (f\|g), (h\|i), (i\|i) | | 7.21e-14 (golden), 5.29e-15 (witness) |
| compile, nvcc / hipcc (`test_emit_ipmetric.py`) | #92's filter | all 28 classes on `sm_86` and `gfx1200`, device-less cloud container, 2026-10-30 | no warning in generated text |

Ladder `--ip-metric-l [LMAX]` (`lP <= lQ <= LMAX`, 0..6, bare 2: six
goldens), never in the bare default nor in `KERNEL_GEN_CMD`. **Not run on a
card and not measured** (no device harness: the d1 DF program of
`test_grad_df_device.py` launches `gmetric_` only). The three-center twin
is #692.

Not done for DF: the metric solve and `d` stay host-side (DF-K's `Gam^K`,
`W^K` are staged on the host only, #620; no device run of them, the
kernels having run on both cards with an arbitrary `Gam`); no screen;
no clock (the low-stack form's cost is unmeasured); Cartesian only (the
libcint witness is Cartesian `_cart` integrals, not the spherical
`int3c2e_ip1` / `int2c2e_ip1` #613 named: the kernels are Cartesian and
the spherical transform is the consumer's, as #615 decided for d0).

## The Coulomb gradient on the HGP recurrence (`gradient/kernel_ghj.py`)

**Status: generated and interpreted only -- not emitted by any ladder, not
compiled by nvcc or hipcc, not run on a card, not measured.** A second
route to `dE_J/dR`, on the Head-Gordon–Pople engine of `hgp_fock.md`
instead of the MMD J chain: `Spec(COULOMB, 1, (la, lb, lc, ld), Target.J,
engine=Engine.HGP)`, the d0 HGP `J` key with `d1` in its `d` segment, one
unit per ORDERED class, `ghj_{la}{lb}{lc}{ld}`.

What it computes is the MMD gradient's split, reached through the other
recurrence: `dE_J/dR = SUM D_ab D_cd (d/dA + d/dB)(ab|cd)` over every
ordered quartet, the bra derivative alone (the ket's equals it), the bra
derivative on `a`'s and `b`'s atoms, no half and no factor 2. Per
contracted quartet the Cartesian rule gives

    d/dA_k (ab|cd) = 2 alpha (a+1_k, b|cd) - a_k (a-1_k, b|cd)
    d/dB_k (ab|cd) = 2 beta  (a, b+1_k|cd) - b_k (a, b-1_k|cd)

and the one design point is where `2 alpha` / `2 beta` go. They belong to
the primitive pair, and the HRR (eq 28) runs on contracted values -- but
eq 28 is linear with geometry-only coefficients, so it holds for any
per-primitive weighting of its boundary. So ONE VRR (the closure of the
union of the boundaries, `|e|` from `la - 1` to `la + lb + 1`, on the
fundamental one Boys order up) feeds THREE contractions in the
`K_AB x K_CD` nest -- `C` plain, `CA` weighted `2 alpha`, `CB` weighted
`2 beta` -- and up to FOUR HRRs after it: `(la+1, lb)` on `CA`,
`(la, lb+1)` on `CB`, `(la-1, lb)` and `(la, lb-1)` on `C` (each absent
when its `l` is 0). The deposit forms `gA_k`, `gB_k` from `D_ab D_cd`
read where the ordered quartet puts them and adds six values to `G` by
atomic add.

Everything else is `kernel_hj`'s: one block per bra shell pair, one
thread per contracted quartet, `|ab)` staged in shared memory (plus its
`(alpha, beta)` per primitive pair, `bra_exp`), bra primitives outer, the
quartet scalars in `eri/hgp_drive.py`'s order, `kernel_f`'s record. Two
departures, each for a stated reason: no canonical mode (a bra derivative
breaks the 8-fold orbit a canonical kernel shares, so every ordered class
is its own unit and the key takes no lever), and no screen
(`G_ab G_cd max|D_cd|` bounds an ERI, not a derivative ERI; `gk` takes the
same position).

Host side, `gradient/interpreted.py`: `ghj_class_gradient` packs one
ordered class's every ordered bra and ket shell pair
(`walk_hjk.pair_record`, `kernel_f.pack`, one launch per pair of
contraction degrees) and `ghj_gradient` sums every class;
`interpreted_runners(coulomb="hgp")` puts it in `rhf.py`'s Coulomb slot in
place of the J chain.

| what | held to | result |
|---|---|---|
| every ordered class with each `l <= 2`, summed per unordered group `{la, lb} x {lc, ld}` (36 groups, 81 classes; three atoms, a two-primitive s), interpreted (`test_kernel_ghj.py`; raised `L + 1 > 4` `slow`) | `host.coulomb_gradient` restricted to the group's bra pairs and kets, `1e-12` of the per-shell absolute scale (`test_hgp_eri.py`'s cross-engine bound) | all 36 groups pass; worst 5.2e-15 of scale (2026-10-30, interpreted, no card) |
| the whole `dE_J` of test_gradient_host.py's s, p pair (`slow`) | the host, `1e-12` of `max\|G\|`; zero net force | 5.6e-16 of `max\|G\|` |
| the Coulomb term of H₂O/STO-3G, H₂O/STO-3G plus a d primitive, H₂O/6-31G (`slow`) | PySCF's `grad:j` (#550's fixtures), `1e-10` absolute | 4.7e-15 / 5.2e-15 / 7.5e-15 absolute |

Not done: no `--grad-*` emit ladder writes a `ghj_` unit and no golden
holds its text, so nvcc and hipcc have never seen it; no card; no
register, stack or clock measurement; nothing above `(dd|dd)` is
exercised (the builder takes any class); the device assembly of the RHF
gradient (#566) does not use it.

## The kinetic gradient on HGP (`gradient/kernel_hgt.py`)

**Status: generated, interpreted against the MMD host and libcint, and
compiled by nvcc (sm_86) and hipcc (gfx1200) through (ff); not run on a
card, nothing measured.** `hgt_{la}{lb}`, key
`kinetic/d1/(la,lb)/b1/t.hgp/literal`, writes the same `G[A, k] += SUM_mn
M_mn dT_mn/dR_{A,k}` as `gt_` on the second engine: `kernel_hgp`'s route
(a VRR per primitive pair, `hgp.vrr`'s eq 25 at `f = 0`, the contraction,
`hgp.hrr`'s eq 28 on contracted values), its mapping (one thread per
contracted shell pair) and its kinetic combination `T(a', b) = (2 lb + 3)
(a'|b)_1 - 2 SUM_d (a'|b + 2_d)_2 - 1/2 SUM_d b_d (b_d - 1) (a'|b - 2_d)_0`.
The bra derivative `2 alpha T(a + 1_k, b) - a_k T(a - 1_k, b)` is folded
into the contraction: `alpha` and `beta` are per primitive, so there are
up to six contracted tables, weighted `(2 alpha)^p beta^n`, each with its
own HRR (`p = 0` absent for an s bra, `n = 0` while `lb <= 1`). The ket
centre is `-dA` (translational invariance, no third centre), so each pair
adds `gA` to `A`'s row and `-gA` to `B`'s, six `AtomicAdd`s. Staged by
`gradient.host.pair_records` (`records`' per-record arrays plus the
per-pair `rec0` / `rec1`, offsets, atoms and `A - B`).

Witnessed interpreted (`test_kernel_hgt.py`): per class (ss) through (ff)
on HF/cc-pVTZ against `host.kinetic_gradient`, worst 37.2 eps of
`test_kernel_g1e.py`'s split-centre scale at (sd) (gate 64 eps, looser
than MMD's 16 because the kinetic combination and the HRR on contracted
values cancel; `kernel_hgp`'s d0 test allows 1e-13 of max for the same
reason); the whole gradient of H₂O/STO-3G, H₂O/6-31G and
`h2o_sto3g_d_grad` against libcint's `int1e_ipkin` contracted with the
fixtures' density, within `1e-12` absolute. Ladder `python -m
felizjueves.emit --hgp-grad-l [LMAX]` (1 bare, 0..3), never in the bare
default nor in `KERNEL_GEN_CMD`; goldens `test/golden/{cuda,hip}/hgt_{00,
01,11}`; every class through (ff) compiles warning-free under both
compilers (`test_emit_hgp_grad.py`, slowest `hgt_33`: 6.3 s nvcc, 14.6 s
hipcc, 2026-10-30). `S` and `V` on HGP are `hgs_`'s and `ghv_`'s, below.

## The overlap gradient on HGP (`gradient/kernel_hgs.py`)

`Target.S` at d1 is also legal on `Engine.HGP` (`spec.GRADIENT_OFF_NATIVE`,
beside HGP `J`, `T` and `K`), keyed `overlap/d1/(la,lb)/b1/s.hgp/literal`
and built by `kernel_hgs.build_hgs_kernel` (`hgs_{la}{lb}`, shells
through f): the same contraction as `gs_`, `G[A, k] += SUM_mn M_mn
dS_mn/dR_{A,k}`, on the recurrence of the HGP one-electron `hs_`
(`one_electron.md`, "The HGP route") instead of McMurchie & Davidson's
`E`.

The rule is the Cartesian one above, read on the integrals:

    dA_k (a|b) = 2 alpha (a+1_k|b) - a_k (a-1_k|b)
    dB_k (a|b) = 2 beta  (a|b+1_k) - b_k (a|b-1_k)

`alpha` and `beta` are per primitive, and the HGP HRR runs on contracted
values, so each exponent is folded in before the contraction (as `hs_`'s
`T` folds `beta^n`): one overlap VRR (`[Fock]` eq 25 at `f = 0`, the
`m + 1` terms dropped) to `la + lb + 1` per primitive pair feeds three
contracted tables -- unweighted (the `a-1_k`, `b-1_k` targets), times `2
alpha` (`a+1_k`) and times `2 beta` (`b+1_k`) -- each with its own HRR
(`[Fock]` eq 28's bra half), `kernel_hgp.table_dags`'s DAGs. Then per
block row `gs_`'s weight from `M` (`kernel_g1e.weights`), and six
`AtomicAdd`s into `G` per SHELL PAIR, both centres computed (no
translational invariance is used, so `SUM_A G = 0` stays a check). One
thread per contracted shell pair, `kernel_hgp`'s mapping; staged by
`host.pair_records` (the per-record arrays are `host.records`', so `gs_`
and `hgs_` read one sweep). Interpreted it is round-off, not bitwise,
against the host -- 17.3 eps of the split-centre scale at (sd), where
the contracted HRR's `AB` terms cancel, so its gate is 64 eps rather
than `gs`'s 16. It shares `hgt_`'s ladder, `--hgp-grad-l`, which writes both
per class. Compiled by nvcc and hipcc through (ff); its card cases
(`test_grad_device.py`'s `*-hs`, H₂O/STO-3G, H₂O/cc-pVDZ and CO/cc-pVTZ
with the bare proton) are written and **not yet run on a card**. Not
measured.

## The nuclear-attraction gradient on HGP (`gradient/kernel_ghv.py`)

**Status: generated, interpreted against the MMD host and PySCF, and
compiled by nvcc (sm_86) and hipcc (gfx1200) through (ff); its device test
written, not run on a card; nothing measured.** `ghv_{la}{lb}`, key
`nuclear/d1/(la,lb)/b1/v.hgp/literal`, writes what `gv_` writes, `G[X, k]
+= SUM_mn M_mn dV_mn/dR_{X,k}` with every nucleus's Hellmann–Feynman term,
on `kernel_hgp`'s `hv_` route: per nucleus, the point-charge VRR per
primitive pair to `la + lb + 1`, the contraction, the HRR on contracted
values; one thread per contracted shell pair; nothing of MMD's (its own
layering rule, `ghv_builder`, keeps `mmd`, `plan` and the witnesses out of
its ancestry; `kernel_g1e`'s weight rule is restated, not imported).

The rule is the Cartesian one, `d/dA_k |a) = 2 alpha |a + 1_k) - a_k |a -
1_k)`, with the weight folded in before the contraction as `hgt_` and
`hgs_` do: per nucleus three contracted tables, `SUM 2 alpha (e|0)` (HRR
targets `(a + 1_k|b)`), `SUM 2 beta (e|0)` (`(a|b + 1_k)`) and `SUM (e|0)`
(`(a - 1_k|b)`, `(a|b - 1_k)`; empty at (ss)), each its own HRR, all on
one VRR (`kernel_hgp.table_dags`, `vrr_dag`, `factors`). The nucleus row
is **not** a third derivative: one nucleus's integral depends on `A - C`
and `B - C` alone, so `dC = -(dA + dB)` of that nucleus's own tables,
exactly. That is why the record loop runs inside the nucleus loop: a
contracted table is one nucleus's (`hv_` sums the nuclei before its one
HRR; the gradient cannot), at the price of the seed's `exp` per record
per nucleus. `G` is `AtomicAdd`-ed, three per nucleus and six per pair.
Staged by `gradient.host.pair_records`, as `hgt_` and `hgs_`.

| What | Against | Where | Worst |
|---|---|---|---|
| `ghv` interpreted (`test_kernel_ghv.py`) | `host.potential_gradient`, per class, `M` non-symmetric, `test_kernel_g1e.py`'s split-centre scale | every ordered class (ss)–(ff) on HF/cc-pVTZ (L + 1 > 4 `slow`); the s, p classes on test_kernel_hgp.py's synthetic off-axis basis plus a fourth, bare nucleus | 6.0 eps of the scale (HF (dd)), gate 16 eps |
| `ghv` interpreted, whole | PySCF's `potential` term (`int1e_ipnuc`, `int1e_iprinv`; test_rhf_gradient.py's witness) | H₂O/STO-3G, H₂O/6-31G | 7.9e-14 / 4.9e-14 absolute, gate `1e-10` |
| compile (`test_emit_ghv.py`, `test_ghv_device.py`) | #92's filter | nvcc `sm_86`, hipcc `gfx1200`: every class through (ff), each unit alone, and the synthetic program | passed, no warning in generated text (2026-10-30, a device-less cloud container); `ghv_33` 2.5 s / 7.3 s, 6,104 B of `sm_86` stack and 4,708 B of `gfx1200` scratch, under #613's 14,000 B, so one (unrolled) form -- a reading, not a measured row |
| on a card (`test_ghv_device.py`, `test/ghv_driver.py`) | the host at `1e-13` of `max\|G\|`, PySCF at `1e-10` | written for the synthetic basis and H₂O/STO-3G, 6-31G; **not run on either card** | -- |

Ladder: `--hgp-grad-l [LMAX]` writes `ghv_` beside `hgt_` and `hgs_` per
class (class-major); goldens `test/golden/{cuda,hip}/ghv_{00,01,11}`.

## The three-center gradient on HGP (`gradient/kernel_hgeri3.py`)

`gradient/kernel_hgeri3.py`'s `build_hgeri3_kernel((la, lb, lP))` →
`hgeri3_{la}{lb}{lP}`, keyed `Spec(COULOMB, 1, (la, lb, lP), Target.ERI3,
engine=Engine.HGP)` (``coulomb/d1/(la,lb|lP)/b1/eri3.hgp/literal``): the
HGP sibling of `geri3`, as `heri3` is of `eri3` (`hgp_df.md`). The same
parameters in the same order (restated and held equal), the same staging
and records, one thread per cell, and the same `G` contract with nine
`AtomicAdd`s into a caller-zeroed `G`; `df_grad_launch.run_eri3(engine=)`
swaps one for the other.

**The rule is the Cartesian one on all three centres**, `d/dX_k (x y|q) = 2
xi (x + 1_k, y|q) - x_k (x - 1_k, y|q)`, and likewise on `Y` and on the aux
centre `C` with `2 gam`: the aux Gaussian is the HGP pair with an
exponent-0 partner, so raising its `l` is raising an ordinary Gaussian.
`geri3` takes the Hermite shift on `C` instead; here no centre is inferred
and `SUM_A G = 0` is a check. The factor `2 xi` is per primitive, so each
raised class is contracted with its own weight: per cell up to six HRR
classes in the built (higher `l` first) orientation, `(x ± 1, y | lP)`,
`(x, y ± 1 | lP)`, `(x, y | lP ± 1)`, the raised ones on accumulators
weighted `2 xi`, `2 eta`, `2 gam`, the lowered ones unweighted. The HRR is
linear with a geometry-only coefficient, so it holds on any
per-primitive-weighted contraction. **One VRR per primitive quartet** feeds
all of them: `hgp.vrr.build` of the rectangle spanning their boundaries,
cut to the union by `hgp.vrr.group_dag` (Boys top `L + 1`), the aux side's
`QC = 0` terms dropped (`kernel_hgp`'s rule). Then per staged element the
weight from `M` (`geri3`'s rule) times `up - x_k down`. No low-stack form.

Ladder `python -m felizjueves.emit --hgp-grad-eri3-l [LMAX]`: `--hgp-eri3-l`'s
classes and bound (`la <= lb <= LMAX`, `lP <= LMAX`, 0..`HGP_DF_L_MAX` = 1,
bare 1: 6 units), never in the bare default nor in `KERNEL_GEN_CMD`.
Goldens `test/golden/{cuda,hip}/hgeri3_*` (`test_emit_hgp_grad_df.py`).

| What | Against | Where | Worst |
|---|---|---|---|
| `hgeri3` interpreted (`test_kernel_hgeri3.py`) | `host_df`, per class, and `SUM_A G = 0` | every class `la + lb + lP <= 4` on `test_kernel_geri3.py`'s bases (swapped, diagonal, one- and two-centre cells; L + 1 = 5 `slow`) | 9.05 eps of the split-centre scale ((pd\|p)), gate 16 eps |
| `hgeri3` against `geri3`, both interpreted | the MMD kernel, per class | the same 22 classes | 15.2 eps ((dd\|s)), gate 32 eps |
| `hgeri3` against libcint | the DF gradient witness (`oracle.manifest.df_gradient`), per class | (ss\|p), (ss\|d), (sp\|s), (sp\|p), (sd\|p), (pp\|s), (pp\|p), (pp\|d) on `prim_spd_f` against the aux-i basis ((ss\|s) is exactly zero there) | 4.95e-15 of the witness's largest `max\|G\|`, gate `1e-12` |
| compile, nvcc / hipcc (`test_emit_hgp_grad_df.py`) | #92's filter | the 6 s/p units, `sm_86` and `gfx1200` | passed, no warning in generated text (a device-less cloud container, 2026-10-30) |

**Not on a card**: there is no device harness for it yet, and the session
that built it had no card. Not measured.

## The fused Fock gradient on HGP (`gradient/kernel_gf.py`)

`Target.F` at `derivative_order = 1`, on F's native engine:
`coulomb/d1/(la,lb|lc,ld)/b1/f/literal`, the d0 key with `d1` in place of
`d0`. One kernel per F class, `gf_{la}{lb}{lc}{ld}`, AO order, writes

    G[A, k] += dE_2e / dR_{A,k},   E_2e = 1/2 SUM D G = SUM Gam (ab|cd),
    Gam_abcd = 1/2 D_ab D_cd - 1/4 D_ac D_bd,

which is `dE_J - 1/2 dE_K` in `rhf.py`'s terms: the whole two-electron
RHF gradient from one fused kernel per class, as `kernel_f` builds the
whole `G = J - K/2`. It is the (HGP, fused) cell of `[Fock]`'s 2x2 at d1;
the (MMD, split) cell is `gjab` + `gk` above.

**The weights are derived** (`derive_gradient_weights`): the kernel visits
one representative per orbit of `kernel_f.orbit_group()`, with
`kernel_f`'s canonical half and degeneracy factor, and summing `Gam` over
the eight images gives, per component,

    w_ijkl = deg (4 D_ab D_cd - D_ac D_bd - D_ad D_bc)

(8 images x 1/2 on the Coulomb pairing; 4 x -1/4 on each exchange
pairing). Because one representative stands for members whose bra
centres are its ket centres, **all four centres are differentiated**: the
bra-only shortcut belongs to a sweep over every ordered quartet. No
translational invariance is used, so zero net force stays a check.

**The derivative by HGP.** `d(ab|cd)/dA_k = (a+1_k b|cd)^{2 alpha} - a_k
(a-1_k b|cd)`, the first a contracted integral whose primitive quartets
carry `2 alpha`. HRR's coefficients (`AB`, `CD`) are the same for every
primitive, so it transfers a weighted contracted sum exactly as an
unweighted one, and the contraction stays inside the kernel as in
`kernel_f` (pin 5). Per primitive quartet: one VRR (`hgp.vrr.reduce`'s
terms, `order_key`'s order) over the union of the boundaries of the eight
derivative classes (each centre raised by one, and lowered by one where
its `l > 0`), Boys to `L + 1`; five contracted families (`c0`
unweighted, for the lowered classes; `cA`..`cD` weighted by `2 alpha`,
`2 beta`, `2 gamma`, `2 delta`, read from two exponent fields appended to
`kernel_f`'s record); after the nest one HRR per derivative class (its
locals prefixed `ap`, `am`, ...), then per ERI the weight and twelve
multiply-adds into the four centres' sums. `AB` and `CD` are always
parameters (the B- and D-raised classes always transfer). The mapping is
`kernel_f`'s (one block per bra pair, one thread per contracted quartet,
shared staging per thread); the bra's two centres are summed over the ket
loop and flushed once, C and D per quartet: `AtomicAdd` into `G[atom,
k]`, 12 statements. **No screen** (a gradient bound needs the factor one
`l` higher, #555), so no `tau`, no `G_ab`, no `Dsh`. No lever.

**The stream** (`gradient/interpreted.py`, `gf_launches`): built from the
basis alone, nothing pruned or screened -- canonical pairs (`launch_f`'s
rule), batched by `(la, lb)` and degree, each batch pair `I <= J` one
launch, `diag` on `I == J`. `gf_gradient` runs every launch interpreted
into one `G`; `gf_run` is one launch. It does not import `launch_f` (whose
screens and pruning the kernel does not read).

**Witnesses** (`test/test_kernel_gf.py`), none of them HGP:

| Witness | Inputs | Worst | Gate |
|---|---|---|---|
| `eri.eri`'s derivative integrals (MMD, the shell raised with `2 alpha` coefficients, lowered), one quartet, `deg = 1` | every ladder class through `(dd|dd)`, as launched and transposed (36), four distinct atoms, random symmetric `D` | 1.4e-15 of the abs-value twin (L <= 6) | 1e-13 |
| `host.coulomb_gradient - 1/2 host.exchange_gradient` | three-atom synthetic bases, two s shells on one atom (so `a = b`, `c = d`, `P = Q` occur), plus an s, p or d shell | 4.8e-15 of `max|G|` | 1e-13 |
| PySCF `grad:j + grad:k` (#550's fixtures) | H2O/STO-3G; H2O/STO-3G plus one d primitive on O | 2.2e-14 / 2.1e-14 absolute | 1e-10 |

Every whole-basis `G` sums to zero within `1e-12` of its max. A wrong
Coulomb weight (`kernel_f`'s 2) fails the per-class witness. The fast
tier holds the classes with `L + 1 <= 4` and the s-only basis; the rest
are `slow` (26 cases, ~66 s on 4 workers).

**Compiled** (`test_emit_grad_f.py`, a compiler and no card): the six
units through `(pp|pp)` under nvcc 13.0 (`sm_86`) and hipcc / ROCm 7.2.4
(`gfx1200`), each alone with the kernel tier's flags, no warning in the
generated text, 12 passed (2026-10-30, a device-less cloud container).

**Not done.** Never run on a card, not timed, not measured for registers,
spill or stack; the d classes are not compiled by any test; only one
text golden. No low-stack form: the unrolled `(dd|dd)`
builds in ~6 s and its CUDA text is 11 MB (`(pp|pp)` 0.59 MB), so a
low-stack or rolled form is the obvious next step before any compile. Not wired into `rhf.py`'s
`Runners` (it replaces two of them; the assembly takes the two terms
apart). No screen, Cartesian only, nothing above `(dd|dd)` (milestone 12's
ladder).

## What is not done

- The HGP Coulomb gradient (`ghj_`, above) is interpreted only: not
  emitted, not compiled, not on a card.
- `ghk`, the exchange gradient on HGP, is interpreted through `(dd|dd)`
  and compiled by nvcc and hipcc through `(dd|pp)` only (`(dd|dd)` over
  600 s on nvcc); no card, no measurement ("The
  exchange gradient on HGP").
- `hgs` (the overlap gradient on HGP, above) has not run on a card and
  is not measured.
- The HGP `(mn|P)` gradient (`hgeri3`) has run on no card and has no
  device harness; its CLI ladder stops at s and p shells. The DF metric's
  gradient is not built on HGP.
- The fused Fock gradient `gf_` (above) is interpreted, and compiled
  through `(pp|pp)`: no d-class compile, no card, no measurement.
- On a card, nothing above L = 4 has run but the DF terms (through
  (ff|i) and (i|i), #613) and the one-electron `gs`, `gt`, `gv` (through
  (ff), #614), `gk` (through `(ff|ff)` on CO/cc-pVTZ, #618) and the
  `gjab` chain (through (ff) on CO/cc-pVTZ, #619). The
  forced-unrolled `gv` and `gjab` at (df) / (ff) are compiled
  (`g1e.json`, `gjab.json`) but not run on a card.
- The whole RHF gradient on a card has run on the RX 9060 XT only, on
  s, p bases only (H₂O/STO-3G, 6-31G; #566): not on the RTX 3080, and no
  basis with a d or f shell (H₂O/cc-pVDZ is not run whole even
  interpreted). A device gradient is not bit-reproducible (the atomics),
  held run to run to `SPREAD_TOL = 1e-13` by the user's decision. The DF
  two-electron gradient, `G_J - 1/2 G_K`, is assembled on the host and
  held to PySCF's DF RHF gradient (above, #620); the one-electron terms
  are not added to it.
- `gk`'s rolled form is built (#618) but its f classes are witnessed
  interpreted on three classes only (`(ff|ff)` on a subset of its
  records, `(fs|fs)`, `(fd|pd)`); the full 256-class interpreted sweep
  against the host was not run (hours of pure Python). The block mapping
  (`kernel_kblock`'s, one warp per `K_AB` block) is not built, and no
  `gk` clock is measured.
- `hgt_` (the kinetic gradient on HGP, above) has not run on a card and
  is not measured. Neither has `ghv_` (the nuclear-attraction gradient on
  HGP, above), whose device test is written for s and p.
- `gs`, `gt`, `gv` are measured for registers, spill, stack and compile
  time (`g1e.json`, #614), and `gjab` likewise (`gjab.json`, #619), not
  for wall-clock.
- No screening (`gk` visits every ordered quartet; a derivative bound is
  not derived), Cartesian only. The Coulomb energy's Hessian is built
  on these pieces (`hessian.md`), and so are the one-electron Hessians of
  `S`, `T` and `V` (`hessian.md`, "The one-electron Hessians"), the
  exchange Hessian (`hkx`, on `gk`'s mapping) and the DF metric and
  three-center Hessians (`d2metric`, `d2eri3`, on `gmetric`'s and
  `geri3`'s).
- Nothing is scored (no published board) and no gradient kernel is timed;
  milestone 20's limits are one record in
  `data/literature/jengine/limits.json` (#567).

### What AIMD still needs

The gradient is a piece of an ab initio MD step, not the step. Missing, by
what the tree holds today:

- **The SCF that produces `D` and `W`.** Every input density here is a
  fixture's (PySCF's converged RHF, #550); there is no eigensolver and no
  SCF loop in `src/`.
- **The per-step rebuild** of everything geometry-dependent that is built
  on the host today: the primitive-pair records and their bounds
  (`ao/pairs.py`), the J chain's plan (`plan/`, host-side only) and the
  launch packing, once per nuclear step.
- **A gradient screen proved against the derivative integrals.** `gk_`,
  `ghk` and `gf_` visit every ordered quartet: K's bound bounds the
  integral, not its derivative, and no derivative bound is derived.
- **d and f shells in the whole gradient on a card.** Each term has run
  alone through (ff) on both cards, but the assembled device gradient only
  on s, p bases, and no orbital g shell is built.
- **RI gradients end to end.** `geri3` (`(mn|P)^x`) and `gmetric`
  (`(P|Q)^x`) run on both cards, but the metric solve and the DF-K weights
  are host-side and the one-electron terms are not added to the DF
  two-electron gradient.
- **UHF.** Only `rhf.py` assembles a gradient; the exchange rule assumes
  a symmetric `D` (`host.exchange_gradient` refuses an asymmetric one),
  and no spin-resolved assembly exists.
- **Bit reproducibility, if a trajectory needs it**: the atomics make a
  device gradient vary run to run (within 1e-13 measured), so two runs of
  one step can differ in the last bits.
