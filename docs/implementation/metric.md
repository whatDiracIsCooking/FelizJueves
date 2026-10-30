# The DF Coulomb metric `(P|Q)` on the McMurchie-Davidson recurrence

**Status: implemented, tested at `lP + lQ ≤ 4` (milestone 15, #410–#416;
PRs #425–#430), and since #606 through (i|i) against libcint on both
cards, with its low-stack form measured (see "The low-stack form"; the
milestone-15 bullets below are as they were).** This is the derived note for the
two-center Coulomb metric

    (P|Q) = ∫∫ P(r1) Q(r2) / r12

of a **Cartesian** auxiliary (density-fitting) basis. It is the contract the
host reference and the kernel were built against, and the one place that
says what is generated, what the aux record carries, what the oracle is,
where it ran and what is not done.

- **Generated:** `Target.METRIC` (`spec.py`, #410), one kernel per aux class
  `(lP, lQ)` (`kernel_metric.py`, #414), emitted by
  `python -m felizjueves.emit --metric-l [LMAX]` as `metric_{lP}{lQ}` (#415).
  The ladder is never in the bare default and is not in `KERNEL_GEN_CMD`.
- **Host reference:** `eri/density_fit/metric.py` (#412), with its `[p|q]` evaluator
  injected.
- **Oracle:** libcint's `int2c2e_cart` on PySCF's
  `df.make_auxmol(mol, 'cc-pvtz-jkfit')`, tracked as four schema-3 row
  pieces (#411).
- **Where it ran:** the RTX 3080 (`sm_86`, CUDA 13.0) (#416, PR #430).
  At PR #430 the RX 9060 XT (`gfx1200`) was not enumerated in the
  container, so its units were only compiled. #431 fixed that detection
  and closed; the card's first metric run was #606's (PR #615), Parts 1
  to 3 (see "The low-stack form").
- **Correctness is tested at `lP + lQ ≤ 4` only**, the nine classes
  `lP ≤ lQ`: (s|s) (s|p) (s|d) (s|f) (s|g) (p|p) (p|d) (p|f) (d|d). The
  builder and the emitter accept any `L`.

**Nothing here is scored.** Until #606 nothing was measured; since then
`data/measured/df_lowstack.json` holds every ladder unit's compile row and
a wall-clock row for each class where the gate is read. There is no
`measured_store` (`kernels.json`) row for any `metric_` unit and no
counters. No published board exists. `[JEngine]` (entry 35)
is J only, and the recurrence is McMurchie & Davidson's (entries 39 and 46).
The auxiliary basis is Weigend's cc-pVTZ-jkfit (entry 47), which is an
input here, not a table to score against. Milestone 15 added no
`table{1,2,3}.json` cell, scoreboard, floor or `claims.json` record.
`data/literature/jengine/limits.json` records this as a `StructuralLimit`
(#417), as #245 did for the one-electron family. A metric class's
`Spec.total_am` is `lP + lQ`. It is the `L` of the quartet the class runs,
but it indexes no row of `[JEngine]` Tables 1–3.

**Out of scope:** DF-J and DF-K themselves, spherical aux functions, and any
measurement beyond #606's. The three-center `(mn|P)`, this note's named follow-up, was
built in milestone 16 on the same aux record and `center_fragment`:
`three_center.md` is the one place with its detail.

## The family, as a key

`Target.METRIC` (`"metric"`, #410) is a **pair** target: its `am` is
`(lP, lQ)` (`AM_LENGTH` 2), on `Operator.COULOMB` (`OPERATOR_OF`), and its
key is `coulomb/d0/(lP,lQ)/b1/metric/literal`. Every pre-existing key is
unchanged, all 196 measured `J` keys among them (`test_spec.py`,
`measure/measure_measured_keys.py`).

It is unbatched **by decision** (`spec._PAIR_BATCH_REFUSAL`). The structural
pair rule does not apply, because the metric has Hermite targets, as `V` and
`SCHWARZ` do. A batched metric kernel is a follow-up nobody has filed. It
has no lever: `levers.py`'s `TARGET_SLOTS` and `DEFAULT_REFUSALS` carry
`"metric"` with a refusal.

## The aux record: one Gaussian as a Hermite "pair"

An auxiliary primitive is a single-center Gaussian. Treat it as a Hermite
"pair" with `ζ = α`, `P = A`, `X_PA = 0` and no second Gaussian, so there is
no `K_AB`. Its `E` is the single-center expansion

    x^i exp(−α x²) = Σ_t E^i_t Λ_t(x; α, A)
    E^{i+1}_t = (1/2α) E^i_{t−1} + (t+1) E^i_{t+1},   E^0_0 = 1

and `E^i_t = 0` whenever `i − t` is odd. The integral is then

    (P|Q) = Σ_{i∈P} Σ_{k∈Q} s_i s_k Σ_t E^P_t(i) Σ_τ (−1)^{|τ|} [t+τ]^(0)(A_i − B_k, θ) E^Q_τ(k)
    θ² = α_i α_k / (α_i + α_k)

**The record weight is `s = c·(π/α)^{3/2}`, with no extra `N`.** This
corrects the issue text's `c·N·(π/α)^{3/2}`. A `Shell`'s coefficients
multiply **unnormalised** primitives (`ao/shell.py`), and
`test/oracle/basis.py` already puts libcint's normalisation into `c`.
Multiplying `ao.shell.primitive_norm` in again would normalise twice.
`eri.density_fit.metric.aux_dress` is the one place a dressing is applied. Two
witnesses check it:

- the aux overlap from the same shells against the fixture's `int1e_ovlp`
  (worst 8.0e-16 of `max|S|`);
- every (s|s) element against its closed form.

The other half of the prefactor is `(2/√π) θ` in the prelude's `W0`, the
statement the `[p|q]` kernel already runs. The two halves multiply to

    (2/√π) θ (π/α_i)^{3/2} (π/α_k)^{3/2} = 2π^{5/2} / (α_i α_k √(α_i + α_k))

which is McMurchie & Davidson's `λ`, and `ao.pairs.dress`'s split with one
Gaussian on each side. `test_ao_metric.py` asserts this identity to 4 eps
relative. **No kernel and no host module forms a constant.**

The device spelling of `E` is `e_fragment.center_fragment(l)` (#410):

- the recurrence above with the `X_PA` term **absent**, not given a zero
  weight;
- the seed is `Lit(1.0)`, so there is no `exp` and no `zeta` bind;
- every odd-parity entry is a structural zero that is never declared, read
  or written;
- every other entry is one `Weighted` over `HALF_INV_ZETA` and `ONE`.

Its locals are named `C{x,y,z}_i_t`, not `E…`, so that `(mn|P)` can inline
it beside `block_fragment` without a clash; `kernel_eri3` now inlines
the two side by side (`three_center.md`). It is pinned bitwise to
`mmd.hermite.e_axis` at `X_PA = 0` for `l = 0..4` (`test_e_fragment.py`).
The host's `eri.density_fit.metric.center_e` is `mmd.hermite.e_block` at that point, the
same arithmetic.

## The host reference, `eri/density_fit/metric.py`

`metric_block` and `metric_block_and_scale` compute one ordered aux shell
pair, the second with its absolute-value twin. `metric_matrix` and
`scale_matrix` place every pair `i ≤ j`, mirrored. A diagonal shell pair
writes its lower triangle only, which is `eri.schwarz`'s rule. `ao` imports
neither `reference/` nor `plan/`: the evaluator arrives as `pq=`.

`metric_classes(aux)` is the staging. It returns one `MetricClass` per
`(lP, lQ)` with `lP ≤ lQ`, covering every aux shell pair `P ≤ Q` by index:

- A pair whose first shell has the higher `l` is staged as `(j, i)` with
  `swapped` set.
- Each class has one record table (`A`, `alpha`, `s`) holding every shell it
  touches once.
- A pair is two record ranges (`p_rec0/p_rec1`, `q_rec0/q_rec1`) plus its
  offsets `off_p` and `off_q`.
- `place_staged` scatters a swapped block transposed.

## The kernel, `kernel_metric.py`

**A metric class is a pair that runs a quartet.** The key is `(lP, lQ)`.
The recurrence is the `Target.J` quartet `(lP 0|lQ 0)`'s at `L = lP + lQ`:
`metric_plan` asks `pipeline.plan` for that spec, as
`kernel_schwarz.quartet_spec` does, and `recurrence.fundamental_prelude` and
`recurrence_body` spell its one DAG in `schedule`'s order, unchanged.
`plan/` was not edited.

**The parity-dead targets are dropped.** This corrects the issue text's
"nothing is dead". The single-center `E^a_t` vanishes whenever `a_d − t_d`
is odd on some axis, so the contraction reads `[r]^(0)` only where:

- `|r| ≡ L (mod 2)`, and
- `|r| ≥ (lP mod 2) + (lQ mod 2)`.

For example, (p|p) never reads `[0]`. The DAG still builds the whole
`|r| ≤ L` tower, so one backward liveness pass drops the unread targets and
every intermediate, fundamental and Boys weight that only they read. This
is `kernel_schwarz`'s rule restated, and a declared but unread local would
otherwise fail #92's warning filter on both compilers. `live_targets` is
the kept set. `test_kernel_metric.py` pins it independently to that formula
for seven classes, and the L = 5 build test checks the same rule there.

The kernel contract:

- **Thread mapping.** One thread per aux shell pair `(P ≤ Q)`, guarded by
  `pair < n_pair`, with `launch_shape` giving `ceil(n_pair/threads)` blocks.
  The thread runs its pair's two record ranges serially: bra outer, with the
  `P` block formed once per bra record; ket inner. It holds `nP × nQ`
  accumulators in registers.
- **`E` is `center_fragment`, inlined twice**, into `Ebra` at `lP` and
  `Eket` at `lQ`, with `half_inv_zeta = 0.5 / alpha` of the record.
- **One ket sign per class.** Every live `τ` has the parity of `lQ`, so
  `(−1)^{|τ|} = (−1)^{lQ}`. The inner sum is spelled unsigned, and the
  sign is applied once, as `acc − w v` or `acc + w v`. Negation is exact.
- **No prefactor in the kernel.** `s_i s_k` is multiplied once per
  (bra, ket), and `(2/√π) θ` is inside `[0]^(m)`. Its `fj_boys` is the
  one template every unit carries (#301), which holds on all of
  `reference.boys`' domain, as tight aux pairs need.
- **Plain mirrored store, no atomics.** Each element belongs to exactly one
  aux shell pair. An off-diagonal pair writes `(a, b)` and its mirror. A
  diagonal pair (only when `lP == lQ`) writes `a ≥ b`, mirrored, which is
  `kernel_jab`'s rule. A swapped pair needs no flag, because with the mirror
  both elements are written either way. Runs are bit-reproducible.
- **Arguments.** The record arrays are `kernel_schwarz`'s interleaved
  `[n_rec, 3]` layout for `A`, the layout every two-electron kernel takes
  (the one-electron `S`, `T`, `V` and `SX` take one array per axis). They are named after `MetricClass.arrays()`, with one
  exception: `s` is the parameter `s_aux`, because the prelude declares a
  local `s = −2θ²` (`PARAM_OF` maps the name). `n_rec` is cast to `(void)`.

`eri/density_fit/metric_drive.py` is the interpreted adapter (`metric_matrix_kernel`,
with Boys injected), after `eri/j_coulomb/drive.py`. Layering rules `metric_builder`
(off `ao`, `interp`, `reference/`, `emit` and `device`) and
`metric_adapter` keep the two apart (`test_layering.py`).

## The oracle: four schema-3 row pieces

The fixture is libcint's `int2c2e_cart` and `int1e_ovlp` on
`df.make_auxmol(mol, 'cc-pvtz-jkfit')`, with `mol` being `h2o_ccpvdz_j`'s
molecule (Cartesian, inherited). The aux basis has 45 shells and 166
functions: O reaches g and H reaches f. The fixture holds the whole
matrices, and the tests filter to `L ≤ 4`. It was generated with PySCF
2.14.0. `generate.py` checks `int2c2e_cart` bitwise against `int2c2e`
before writing.

A one-file fixture would be over the repo's 512 KiB
`check-added-large-files` hook, so by the user's decision the matrices are
split by rows into four files:

    test/oracle/fixtures/h2o_ccpvtz_jkfit_metric_rows{0_43,43_81,81_121,121_166}.json

They follow #121's precedent for HF/cc-pVTZ. **Schema 3**
(`test/oracle/manifest.py`) adds one optional key,
`metric = {set, rows, parent, converted}`. A file carrying it is a metric
piece:

- it has no `density`;
- its arrays are exactly `int2c2e_cart:rows` and `int1e_ovlp:rows`, rows
  `[r0, r1)` at full width, cut on shell boundaries;
- its aux basis is in `metric.converted`, so no consumer that walks the
  converted fixtures as AO bases picks it up.

Schema 2 is unchanged. `manifest.metric_set(name)` reassembles a set, and
refuses each of: a missing piece, a gap, an overlap, or pieces whose inputs
differ. It also refuses a reassembled matrix that is not symmetric to 1e-14
of its largest element. `max|M|` is about 96.5.

## Tolerances

Every bound is stated in its test's docstring before the run.

**Against libcint.** Each comparison is held to `1e-12 · max|M|` of the
whole fixture matrix:

| Comparison | Worst |
|---|---|
| host (`test_ao_metric.py`), and each host block also at 1e-12 of its own absolute-value scale (worst 1.3e-13) | 2.6e-14, at (d\|d) |
| interpreted kernel, every `L ≤ 4` class (`test_kernel_metric.py`) | 2.6e-14 |
| card (`test_metric_device.py`) | **2.61e-14** |

**Against the host: 16 eps over the tower's absolute-value twin.** Kernel
and host round differently. The host's `[t|τ]` is `reference.pq`'s fixed
recursion and the kernel's is the plan's MILD DAG. The host also dresses
each `E` entry, where the kernel multiplies `s_i s_k` once. So the bound is

    |M_kernel − M_host| ≤ 16 eps · scale,   eps = 2^-52

where `scale` is the element's sum with every `[t+τ]` replaced by its
**tower's** absolute-value twin. That twin is `pq_tower` in
`test_kernel_metric.py`: the same raising recursion over `|X_d|` and
`|[0]^(m)|`. Its coefficients are nonnegative, so it bounds the terms of
both recursions whichever axis each step raises.

**A per-target scale was tried first and found unsound.** It is
`metric_block_and_scale`'s twin, which takes each `[t+τ]` as one atom. It
cannot see the cancellation inside the tower: for (p|p),
`[2,0,0]^(0) = X²[0]^(2) + [0]^(1)` has terms of opposite sign. On
single-primitive (p|p) pairs about 1.8 bohr apart, an honest few-ulp
rounding read as 122.9–229.3 eps of the per-target scale. #414's PR computed
three such elements exactly with mpmath, and host and kernel were equally
far from the exact value. The failure was in the yardstick, so the scale
changed and the 16 did not.

Worst readings under the tower-twin bound:

| Where | Synthetic basis | Fixture |
|---|---|---|
| interpreted | 1.78 eps | 2.60 eps |
| card | 2.09 eps | 7.17 eps |

The device bound deliberately does **not** add `WIDE_BOYS_REL`, as
Schwarz's does: the card is held to the interpreter's yardstick unchanged.
An element whose scale is 0 (one centre, odd total parity) must be exactly
0.0, and so must every element no launched pair owns.

## Where it ran

**On the RTX 3080 (`sm_86`, nvcc 13.0), PR #430.** The command was
`devtools/kernel-tier.sh --backend cuda -j 12 -- -k metric`. It exited 0
with `device: present` and no warning in generated text, and the device
suite gave 2 passed, 0 skipped. The two tests are:

- `test_cuda_every_metric_class_through_l4_matches_the_host`: Part 1 is the
  nine classes on `test_kernel_metric.SYNTHETIC`; Part 2 is the whole
  fixture.
- `test_cuda_metric_through_l4_matches_libcint`: every pair of the nine
  classes, 962 pairs.

The driver is `test/metric_driver.py`. At PR #430 it built one program per
aux basis, after `test/schwarz_driver.py`, launched every class once into
one zeroed `n_aux × n_aux` matrix `RUNS` times behind the VRAM probe, and
compiled through `plan_driver.compile_checked`. The runs were bit-identical
and the matrix exactly symmetric. Since #654 it runs on the shared runner
(#649, `test/harness/`): one runner per class's unit, one launch per class
with the matrix carried from class to class, each launch `RUNS` times and
the runner failing unless every run is bit-identical to the first. That
port ran on both cards on 2026-10-30 (#654), Parts 1 to 3 and the HGP
twin, every figure unchanged from the `main()` it replaced: Part 1 worst
2.09 eps, Part 2 7.17 eps, Part 3 3.71 eps and 2.76e-14 of `max|M|`
against libcint, on the RTX 3080 and the RX 9060 XT alike. The results are in the tables above: 2.61e-14 of
`max|M|` against libcint, and 7.17 eps of the tower-twin scale against the
host (max abs 3.55e-14) on the fixture.

**Not on the RX 9060 XT (`gfx1200`), at PR #430.** The card was not
enumerated in the container: `/dev/kfd` existed, but `rocminfo` showed no gfx agent. No HIP
number is claimed. Only the compile-only companions ran under hipcc /
ROCm 7.2.4 for `gfx1200`, with no warning in generated text:

- the nine units, in `test_emit_metric.py` (#415);
- both device programs, in `test_metric_device.py` (#416).

**#431** fixed `devtools/kernel-tier.sh`'s HIP device check, which tested
`/dev/kfd` only and printed `device: present` with no usable card: it now
asks `lib.sh`'s `amd_gpu_present` (an enumerated kfd GPU node). It closed
on 2026-10-17 with that fix. The device tests first ran on the card in
#606's session (PR #615): "The low-stack form" below.

## The low-stack form (#606)

`build_metric_kernel` has the same low-stack form as `(mn|P)`
(`three_center.md`, "The low-stack form"; `kernel_metric.py`'s docstring).
Its rule is `L ≥ 10`, or `E` blocks over `E_DOUBLES_MAX = 1785` doubles.
That bound was eri3's (dd|g), restated. It is now the metric's own, because
#606's measurement tightened eri3's bound and not this one. On
`lP ≤ lQ ≤ 6` the form is the default for (s|i) through (i|i) and for
(h|h): 8 of 28 classes, none at `L ≤ 4`.

- **Text.** At (i|i) its local arrays are 6,768 B against 37,736 B of `E`
  blocks unrolled, and its source is 85 KB against 738 KB.
- **Interpreted** (`test_kernel_df_lowstack.py`):
  - every L ≤ 4 class forced, against the host and within `1e-14` of the
    unrolled kernel, bitwise through (p|p);
  - by default (`slow`): (s|i), (p|i), (h|h), (h|i) and (i|i). Worst
    0.94 eps of the tower-twin scale against the host, and 1.9e-16 of max
    against the unrolled kernel.
- **Measured** (`data/measured/df_lowstack.json`, `devtools/measure.sh
  df-lowstack-compile`; the method and rule are in `three_center.md`,
  "The low-stack form"):
  - all 28 classes in their default form;
  - the low-stack form of the five unrolled classes from `L = 7`, each
    unit alone, under a 300 s compile budget.

  `metric_66` (i|i) is 255 registers, 6,352 B of `sm_86` stack and 64 B
  of spill stores, in 1.8 s; on `gfx1200` it is 192 VGPRs and 5,824 B of
  scratch, in 2.6 s. The largest unrolled class the rule keeps is (g|h):
  4,944 B of `sm_86` stack and 5,320 B of `gfx1200` scratch, inside the
  ~14 KB budget. So the metric's gate does not move.
- **Against libcint through (i|i)**: the tracked aux-i set
  `prim_spd_f_auxi_metric`. Its basis has `l = 0..6` aux shells on He
  and Ne, with Ne's i contracted, for 168 Cartesian functions. It is four
  schema-3 row pieces.
  - The host matches libcint within 2.76e-14 of max. The worst class is
    (d|g); every h and i class is better.
  - Interpreted, (s|i), (h|h), (h|i) and (i|i) match within 2.2e-15 of
    max.
  - **On both cards** (`test_metric_device.py` Part 3, `slow` and
    `high_l`, #606). Every one of the 28 classes runs, 8 of them
    low-stack, with one program per `lQ`.

    | card | worst against the host | against `int2c2e_cart` |
    |---|---|---|
    | RTX 3080 | 3.71 eps of the tower-twin scale | 2.76e-14 of max |
    | RX 9060 XT | 3.71 eps | 2.76e-14 of max |

    The 2.76e-14 is the host's own, set at (d|g). Runs are bit-identical
    on both cards. The same session ran Parts 1 and 2 on the RX 9060 XT
    too: the card's first metric run, which was #431's.
- **Clock** (`devtools/measure.sh df-lowstack`): the low-stack form is
  slower wherever both forms compile.

  | class | RTX 3080 | RX 9060 XT |
  |---|---|---|
  | (g\|g) | 3.50x | 1.73x |
  | (g\|h) | 4.58x | 1.46x |

  The unrolled forms are inside the stack budget, so neither class moves
  under the gate. (i|i) low-stack takes 4.21 ms on the RTX 3080 and
  5.27 ms on the RX 9060 XT.

## What is not done

- **Measurement beyond #606's** compile and clock rows in
  `data/measured/df_lowstack.json`. There are no counters, and no
  `kernels.json` row.
- **A score.** Nothing is scored, because no published board exists.
- **Spherical comparison.** The libcint check is `int2c2e_cart`; #606
  asked for spherical, and the kernels are Cartesian.
- **Spherical aux functions.** Only Cartesian aux functions are handled.
- **DF-J and DF-K themselves**, and anything that factors or inverts the
  metric.
- **The three-center `(mn|P)`** was the named follow-up and is now built
  (milestone 16, #441–#447): it reuses the aux record (`AuxRec`, one
  primitive with `s` and no pair factor) unchanged and inlines
  `center_fragment` beside `block_fragment`. Its status,
  tests and what it does not reach are in `three_center.md`, not here.
- **An HGP twin.** `(P|Q)` on the HGP VRR (`hmetric_*`, `Target.METRIC` on
  `Engine.HGP`), the same parameters and output, is built for s and p shells,
  checked interpreted and run on the RTX 3080 only (PR #601): `hgp_df.md`.
- **A batched metric kernel and a metric lever.** Derivatives are built
  elsewhere: `gmetric_` (gradient, #603/#613, both cards; `gradient.md`)
  and `d2metric_` (Hessian, #611, interpreted only; `hessian.md`).
