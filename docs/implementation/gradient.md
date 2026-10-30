# Analytic nuclear gradients by McMurchie–Davidson: `S`, `T`, `V`, `E_J`, `E_K` and the DF integrals

**Status: implemented host-side and as generated kernels, compiled by
nvcc and hipcc and run on both cards; the whole RHF gradient assembled and
held to PySCF interpreted only (#565)** (the RTX 3080 and the RX 9060 XT,
against the host, kernel classes at L <= 4 only, except the DF terms,
through (ff|i) and (i|i) since #613, `gs`, `gt`, `gv` through (ff)
since #614, `gk` through `(ff|ff)` since #618 and the whole `gjab` chain
(`DQ`, raised `[p|q]`, `gjab`) through (ff) since #619; which inputs, per kernel, in the tables below).
The Coulomb gradient on the HGP recurrence (`ghj_`, "The Coulomb gradient
on the HGP recurrence") is generated and interpreted only. This is
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
Every other target, and HGP, refuses it with
`_GRADIENT_REFUSAL` -- except one off-native pair,
`spec.GRADIENT_OFF_NATIVE`: `J` on HGP ("The Coulomb gradient on the HGP
recurrence", below), key `coulomb/d1/(la,lb|lc,ld)/b1/j.hgp/literal`. A gradient class is its integral's class, `am` the
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
`eri.j_coulomb.drive.KernelJP`, then `gjab_`. The device runners are #566.

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
| compile, `-Wall -Wextra` | clang 18 CUDA front end, device side, runtime stub | all 40 pair units through (ff); the 16 `gk` goldens | no diagnostic (a proxy) |
| compile, nvcc / hipcc (`test_emit_grad.py`, `test_emit_grad_k.py`) | #92's filter | `sm_86` and `gfx1200` | passed, no warning in generated text |
| on a card (`test_grad_device.py`, `test/grad_driver.py`, on the shared runner since #659, not yet run on a card) | `host`, `1e-13` of `max\|G\|`; `gk` on H₂O/STO-3G only | RTX 3080 and RX 9060 XT: `gs`, `gt`, `gv`, `gjab` on H₂O/STO-3G and cc-pVDZ (`slow`, its (dd) classes), `gk` on H₂O/STO-3G (through `(pp\|pp)`); 9 cases per card | 3.20e-14 (CUDA), 2.36e-14 (HIP), both `gt` on cc-pVDZ; `gk` 4.1e-15 / 4.0e-15; the atomics' run-to-run spread up to 2e-14 |
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
| on a card (`test_grad_df_device.py`, `grad_driver`'s program, on the shared runner since #659, not yet run on a card) | `host_df`, `1e-13` of `max\|G\|`, DF-J of H₂O/STO-3G against JKFIT s/p/d | RTX 3080 and RX 9060 XT, `gmetric` and `geri3` (`slow`) | 6.58e-15 (CUDA), 4.87e-15 (HIP), both `gmetric` |
| every ladder class above L = 4, interpreted, default form (`test_kernel_df_grad_ladder.py`, `slow` from L + 1 = 5) | `host_df`, per class, 16 eps of the TOWER-TWIN split-centre scale (the host on `\|Gam\|` with `test_kernel_metric.pq_tower`; the signed scale is blind to the tower's own cancellation from L ~ 9, where the unrolled form misses it as well) | 49 `geri3`, 19 `gmetric` classes on one-primitive bases with swapped and diagonal pairs; every L <= 4 class in the low-stack form forced; four classes forced unrolled above the floor | worst 0.96 eps of the tower-twin scale ((pf|s), forced low-stack), gate 16 eps |
| the DF gradient witness (`oracle.manifest.df_gradient`, #613) | libcint's `int3c2e_ip1_cart` / `int3c2e_ip2_cart`, `int2c2e_ip1_cart`, contracted per class with closed-form `Gam`, `W`, on `prim_spd_f` against the aux-i basis (every ladder class) | the host, every class (`slow`); the kernel interpreted at (df\|i), (ff\|h), (ff\|i), (h\|i), (i\|i) (`slow`) | host 3.91e-14 (`geri3`), 1.71e-14 (`gmetric`) of the witness's largest `max\|G\|`; the kernel at most 2.72e-15 ((h\|i)); gate `1e-12` |
| on a card, the ladder (`test_grad_df_device.py` Part 2, `slow` + `high_l`) | `host_df` at `1e-13` of `max\|G\|` per program, and the witness at `1e-12` | RTX 3080 and RX 9060 XT: all 70 + 28 classes in their default forms on the aux-i system, one program per aux `l`, each compile under a 300 s timeout; 16 passed per card (with Part 1) | host 1.20e-14 (CUDA) / 9.34e-15 (HIP), (`geri3`, aux i); libcint 5.49e-14 / 5.48e-14 of max; runs bit-identical on both |

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

## What is not done

- The HGP Coulomb gradient (`ghj_`, above) is interpreted only: not
  emitted, not compiled, not on a card.
- On a card, nothing above L = 4 has run but the DF terms (through
  (ff|i) and (i|i), #613) and the one-electron `gs`, `gt`, `gv` (through
  (ff), #614), `gk` (through `(ff|ff)` on CO/cc-pVTZ, #618) and the
  `gjab` chain (through (ff) on CO/cc-pVTZ, #619). The
  forced-unrolled `gv` and `gjab` at (df) / (ff) are compiled
  (`g1e.json`, `gjab.json`) but not run on a card.
- The whole RHF gradient is assembled and held to PySCF interpreted
  only ("The RHF gradient, assembled", #565); on a card each term has run
  alone, and the device assembly is #566. The DF
  two-electron gradient, `G_J - 1/2 G_K`, is assembled on the host and
  held to PySCF's DF RHF gradient (above, #620); the one-electron terms
  are not added to it.
- `gk`'s rolled form is built (#618) but its f classes are witnessed
  interpreted on three classes only (`(ff|ff)` on a subset of its
  records, `(fs|fs)`, `(fd|pd)`); the full 256-class interpreted sweep
  against the host was not run (hours of pure Python). The block mapping
  (`kernel_kblock`'s, one warp per `K_AB` block) is not built, and no
  `gk` clock is measured.
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
