# The three-center DF integral `(mn|P)` on the McMurchie-Davidson recurrence

**Status: implemented, tested at `L = la + lb + lP ≤ 4` (milestone 16,
#441–#446; PRs #452–#457), and since #606 through (ff|i) against libcint
on both cards, with its low-stack form measured (see "The low-stack form,
and the ladder through aux i"; the milestone-16 bullets below are as they
were).** This is the derived
note for the three-center Coulomb integral of density fitting

    (mn|P) = ∫∫ m(r1) n(r1) P(r2) / r12

where `m` and `n` are Cartesian AOs of an ordinary basis and `P` is a
function of a **Cartesian** auxiliary basis. It is the contract the host
reference and the kernel were built against, and the one place that says
what is generated, what the oracle is, where it ran and what is not done.
It is `metric.md`'s shape with one more index.

- **Generated:** `Target.ERI3` (`spec.py`, #441), one kernel per class
  `(la, lb, lP)` (`kernel_eri3.py`, #444), emitted by
  `python -m felizjueves.emit --eri3-l [LMAX]` as `eri3_{la}{lb}{lP}`
  (#445). The ladder is never in the bare default and is not in
  `KERNEL_GEN_CMD`.
- **Host reference:** `eri/density_fit/eri3.py` (#443), with its `[t|τ]` evaluator
  injected.
- **Oracle:** libcint's `int3c2e_cart` on `h2o_sto3g_j`'s molecule against
  PySCF's `df.make_auxmol(mol, 'cc-pvtz-jkfit')`, tracked as one schema-4
  piece (#442).
- **Where it ran:** the RTX 3080 (`sm_86`, CUDA 13.0) (#446, PR #457).
  At PR #457 kfd did not enumerate the RX 9060 XT (`gfx1200`,
  `gfx_target_version` 0), so the units and both device programs were
  only compiled there. #431 fixed that detection and closed; the card's
  first `(mn|P)` run was #606's (PR #615), Parts 1 to 3 (see "The
  low-stack form, and the ladder through aux i").
- **Correctness is tested at `la + lb + lP ≤ 4` only**, the 22 classes
  `la ≤ lb` from (ss|s) through (dd|s). The builder and the emitter accept
  any `L`.

**Nothing here is scored.** Until #606 nothing was measured; since then
`data/measured/df_lowstack.json` holds every ladder unit's compile row and
a wall-clock row for each class where the gate is read. That file is not
`measured_store`'s `kernels.json`. No published board exists. `[JEngine]` (entry 35)
is J only, and the recurrence is McMurchie & Davidson's (entries 39 and 46).
The auxiliary basis is Weigend's cc-pVTZ-jkfit (entry 47), an input here
and not a table to score against. Milestone 16 added no `table{1,2,3}.json`
cell, scoreboard, floor or `claims.json` record.
`data/literature/jengine/limits.json` records this as a `StructuralLimit`
(#447), as #245 did for the one-electron family and #417 for the metric.

**Out of scope:** DF-J and DF-K, anything that factors or inverts the
metric, spherical aux functions, screening, and any measurement beyond
#606's (see "What is not done").

## The family, as a key: a triple that runs a quartet

`Target.ERI3` (`"eri3"`, #441) is the first target whose class names
**three** shells. Its `am` is the triple `(la, lb, lP)` (`AM_LENGTH` 3),
the AO shell pair on the bra and the one aux shell on the ket, on
`Operator.COULOMB` (`OPERATOR_OF`). Its key is
`coulomb/d0/(la,lb|lP)/b1/eri3/literal`. The validator names a mis-shaped
`am` by `spec._SHAPE` (`pair`, `triple`, `quartet`).

The recurrence it runs is the `Target.J` quartet `(la lb|lP 0)`'s at
`L = la + lb + lP`, so `total_am` is that `L`. The key indexes no row of
`[JEngine]` Tables 1–3: the plan is asked for the quartet, never for this
key. The triple is **not** an AO quartet with a shell dropped. Its third
shell is an aux shell of another basis, and no fourth shell was ever there.

It is unbatched **by decision** (`spec._PAIR_BATCH_REFUSAL`, whose message
reads `for triple target eri3: ...`). The structural pair rule does not
apply, because the class has Hermite targets that could be partitioned. A
batched three-center kernel is a follow-up nobody has filed. It has no
lever: `levers.py`'s `TARGET_SLOTS` and `DEFAULT_REFUSALS` carry `"eri3"`
with a refusal, as they do `"metric"`. Every pre-existing key is
byte-identical (`test_spec.py`; `test_measured_keys.py` pins a 510-key J
digest computed before the change).

## The two sides

The ket's Hermite "pair" is one aux Gaussian, as in the metric: `ζ = α`,
`Q = A`, `X_PA = 0`, and no `K_CD`. For a bra primitive pair `ij`
(`p = ζ_i + ζ_j`, centre `P_ij`) and an aux primitive `k`,

    (mn|P) = Σ_{ij} Σ_k s_ij s_k Σ_t E^{mn}_t(ij) Σ_τ (−1)^{|τ|} [t+τ]^(0)(P_ij − A_k, θ) E^P_τ(k)
    θ² = p α_k / (p + α_k)

- **The bra is an ordinary Hermite pair**, `ao.pairs.primitive_pairs`'
  record. `ao.pairs.dress` is the only place `c_i c_j (π/p)^{3/2}` is
  multiplied in. The Gaussian product `K_AB` is **not** a dressing factor:
  it is `E`'s seed, `E^{00}_0 = exp(+ζ X_PA X_PB) = exp(−μ X_AB²)` per
  axis (`mmd.hermite.e_axis`). This corrects the issue text of #443, which
  put `K_AB` in `dress`.
- **The ket is `eri.density_fit.metric`'s aux record, unchanged**:
  `s_k = c_k (π/α_k)^{3/2}` by `eri.density_fit.metric.aux_dress`, with **no** extra
  primitive normalisation, because `c` already carries libcint's
  (`metric.md`, "The aux record"). `E^P` is `eri.density_fit.metric.center_e`.

So `(2/√π) θ (π/p)^{3/2} (π/α)^{3/2} = 2π^{5/2} / (p α √(p + α))` is
McMurchie & Davidson's `λ`, as for `(P|Q)` and `(ab|cd)`. No kernel and no
host module forms that constant: `(2/√π) θ` is the prelude's `W0`.

## The host reference, `eri/density_fit/eri3.py`

`eri3_block(sA, sB, sP, *, pq)` computes one ordered shell triple as
`[a][b][q]`; `eri3_block_and_scale` also returns its absolute-value twin.
`eri3_tensor(basis, aux, *, pq)` and `scale_tensor` return
`(n_ao, n_ao, n_aux)`. `ao` imports neither `reference/` nor `plan/`: the
evaluator arrives as `pq=`. Layering gives `eri.density_fit.eri3` the `ao` rule
(`test_layering.py`).

`eri3_tensor` places every AO shell pair `i ≤ j` and its `(m, n)` mirror. A
diagonal shell pair writes `a ≥ b` only, mirrored, so the tensor is
`(m, n)`-symmetric **exactly**. The two blocks `(ab|P)` and `(ba|P)` are
equal exactly but not always bitwise, because `E^{ab}` and `E^{ba}` round
differently.

`eri3_classes(basis, aux)` is the staging, shaped for the kernel. Every AO
shell pair `i ≤ j`, crossed with every aux shell, is a **cell** of class
`(la, lb, lP)` with `la ≤ lb`:

- A pair whose first shell has the higher `l` is staged as `(j, i)` with
  `swapped` set. `place_staged` lands its block with `m, n` transposed.
- A class holds two record tables, each entry once and contiguous: the bra
  records (`zeta_ab`, `P`, `XPA`, `XPB`, `s_ab`, `eri.schwarz`'s record) and
  the aux records (`A`, `alpha`, `s`, `eri.density_fit.metric`'s).
- A cell is two record ranges, `[ab_rec0, ab_rec1)` and `[p_rec0, p_rec1)`,
  and three offsets, `off_a`, `off_b`, `off_p`.
- `class_block` rebuilds each bra `E` from the record alone and is
  `eri3_block` of the staged shells bitwise.

**The internal twin was not built.** `eri.eri`'s quartet `(ab|c0)`, with `0`
an s shell of exponent 0, would be this integral algebraically, but
`ao.shell.Shell` refuses an exponent that is not positive, and #443 said to
drop the check rather than widen `shell`. `test_ao_eri3.py`'s
`test_the_internal_twin_is_refused_by_shell` pins the refusal, so a `Shell`
that starts accepting 0 makes the twin constructible and the file says so.

## The kernel, `kernel_eri3.py`

`eri3_spec` gives the class key, `quartet_spec` the `Target.J` quartet
`(la lb|lP 0)`, and `eri3_plan` asks `pipeline.plan` for that quartet, the
Schwarz and metric pattern. `recurrence.fundamental_prelude` and
`recurrence_body` spell its one DAG in `schedule`'s order, unchanged.
`plan/` was not edited and no new recurrence was written.

**The dead targets are dropped.** The bra block reaches every
`|t| ≤ la + lb`, and the single-center `E^P_τ` is zero whenever
`q_d − τ_d` is odd on some axis, so every live `τ` has `|τ| ≡ lP (mod 2)`.
The contraction reads `[r]^(0)` at exactly:

- every `|r| ≤ L` with `|r| ≥ lP mod 2`, and
- when the bra is (ss), only `|r| ≡ lP (mod 2)`.

`live_targets(la, lb, lP)` is that set. One backward liveness pass
(`_live`, `kernel_metric`'s rule) drops every unread target and everything
only it reads, because a declared but unread local fails #92's warning
filter on both compilers. `test_kernel_eri3.py` pins the kept set to the
formula, written independently of the builder.

The kernel contract:

- **Thread mapping.** One thread per cell (an AO shell pair crossed with an
  aux shell), guarded by `cell < n_cell`, with `launch_shape` giving the
  grid. The thread walks its cell's two record ranges serially: bra outer,
  with the `mn` block formed once per bra record; aux inner. It holds
  `nA × nB × nP` accumulators in registers. That is 540 at `(dd|g)`, so
  high classes spill; #606 measured it and gated them (the low-stack form
  below).
- **Both `E` fragments are inlined:** `e_fragment.block_fragment(la, lb)`
  into `Ebra` (`zeta_ab`, `half_inv_zeta = 0.5 / zeta`, the record's
  `XPA` / `XPB`) and `e_fragment.center_fragment(lP)` into `Eket`
  (`half_inv_zeta = 0.5 / alpha`). Only the two arrays are read
  afterwards, and every `Ebra` and `Eket` entry read is a structurally
  nonzero one the fragment writes.
- **One ket sign per class.** Every live `τ` has the parity of `lP`, so
  `(−1)^{|τ|} = (−1)^{lP}`. The inner sum is spelled unsigned and the sign
  applied once, `acc − w v` for odd `lP` and `acc + w v` otherwise.
- **One common subexpression beyond the metric.** The inner sum
  `u_tq = Σ_τ [t+τ]^(0) Eket[q, τ]` does not depend on the bra row, so it
  is declared once per `(t, q)`. The operations and their order are the
  metric's, so the rounding is too.
- **No prefactor in the kernel.** `s_ij s_k` is multiplied once per
  (bra, aux) record pair, and `(2/√π) θ` is inside `[0]^(m)`.
- **The Boys domain.** A tight aux exponent far from a bra centre puts `T`
  past where the old narrow series overflowed (near 709;
  `test_kernel_eri3.py`'s tight triple reaches `T ≈ 2667`). The one
  `fj_boys<M>` every unit carries since #301 holds there.
- **Plain mirrored store, no atomics**, into `eri3[n_ao, n_ao, n_aux]`. An
  off-diagonal pair writes `(a, b, q)` and its `(m, n)` mirror; a diagonal
  pair (only when `la == lb`) writes `a ≥ b`, mirrored, which is
  `kernel_jab`'s rule. A swapped cell needs no flag, because with the
  mirror it writes the same two elements as `place_staged`. Each element
  belongs to one cell, so runs are bit-reproducible. Elements of cells not
  launched are untouched.
- **Arguments.** The record and cell arrays are plain positional
  parameters, named after `Eri3Class.arrays()`, vectors interleaved
  `[n_rec, 3]`, the layout every two-electron kernel takes. The one
  exception is `s`, passed as `s_aux`, because the prelude declares a local
  `s = −2θ²` (`PARAM_OF`, as in the metric).

**Round-off, not bitwise**, against the host. The host forms
`((s_ab E) [t|τ]) (s E_P)` with `reference.pq`'s fixed recursion; the
kernel forms `(s_ab s_k)(E u)` on the plan's MILD DAG. No class matches the
host's term order, and `test_no_class_is_bitwise_and_why` records that
even (ss|s) is not bitwise.

`eri/density_fit/eri3_drive.py` is the interpreted adapter, after `eri/density_fit/metric_drive.py`,
with Boys injected. Layering rules `eri3_builder` (off `ao`, `interp`, `emit`,
`device` and `reference/`) and `eri3_adapter` keep the two apart (`test_layering.py`).

## Two recorded decisions

**The name is `eri3`.** The target is `Target.ERI3`, its key slot `eri3`,
and its units `eri3_{la}{lb}{lP}`, decided before the milestone started.
`kernel_eri3.eri3_name` is the one rule. The three `l`s run together while
every `l ≤ 9` (`eri3_220`), and are `_`-separated once any `l > 9`
(`eri3_0_11_0`). #445's first version had no separator, so `--eri3-l 11`
wrote `(0,11,0)` and `(0,1,10)` to the same file, `eri3_0110`, and one
silently overwrote the other. With the separator every name is unique at
any `L`, and every name whose `l`s are all at most 9 is unchanged, the 44
goldens among them. `test_emit_cli.py`
checks uniqueness through `L = 40`, checks the ladder's names against every
other ladder's, and writes `--eri3-l 11` with a stubbed builder into files
opened in exclusive mode.

**The layout is `[m][n][P]`, the aux index fastest**, decided in
`eri/density_fit/eri3.py` and honoured by the kernel, the driver and the fixture. It
makes the `(m, n)` mirror a transpose of the first two indices and keeps a
cell's writes in one short run of aux columns. **The known trade:** a
future DF-J would prefer aux-major slabs, `[P][m][n]`, so that one aux
function's `(mn|P)` is contiguous for the contraction with the density and
the fitted coefficients. Nothing consumes the tensor yet, so the trade is
recorded, not taken.

## The oracle: one schema-4 piece

The fixture is

    test/oracle/fixtures/h2o_sto3g_jkfit_3c_cols0_166.json

from `df.incore.aux_e2(mol, auxmol, intor="int3c2e_cart", aosym="s1")`,
with `mol` being `h2o_sto3g_j`'s molecule and basis (7 Cartesian AOs, s and
p) and `auxmol = df.make_auxmol(mol, 'cc-pvtz-jkfit')` (45 shells, 166
functions, O to g, H to f). It was generated with PySCF 2.14.0 and libcint
6.1.3, and `generate.py` checks the result bitwise against
`intor="int3c2e"` before writing. At 193,822 B it is under the 512 KiB
hook, so the set is one piece. The tracked fixture count rose from 27 to
28.

**Schema 4** (`test/oracle/manifest.py`, `THREE_CENTER_SCHEMA`) adds one
optional key, `three_center = {set, cols, parent, converted}`. A file
carrying it is a three-center piece:

- its `shells` and `nao` are the aux basis's, as in a metric piece, and
  the aux basis is in `three_center.converted`;
- its one array is `int3c2e_cart:<parent>`, shaped
  `(nao_parent, nao_parent, c1 − c0)`;
- it has no density.

A `three_center` key in a file below schema 4 is refused by name, and so is
a file that is both a metric and a three-center piece. AO fixtures stay
schema 2 and metric pieces schema 3, so no tracked fixture changed meaning.

`manifest.three_center_set(name)` reassembles a set (its core,
`assemble_three_center`, takes loaded pieces, so a duplicate can be
injected). It refuses no piece or a piece of another set, a duplicate or
overlapping piece, a column gap at the start, middle or end, pieces whose
inputs differ, an absent parent or one with no converted basis, and a
parent that does not match the piece. **The parent check compares what the
piece carries:** its atoms, `provenance.mol`, `provenance.cart`, and the
array's AO width as the parent's `nao`. The piece records none of the
parent's shells, so "atoms, shells, nao" in #442's text reduces to that.

**The `(m, n)` symmetry is asserted to `1e-14 · max|(mn|P)|`, not
bitwise.** #442 first asked for `(mn|P) == (nm|P)` bitwise. That is false
for the `s1` call, because libcint computes both triangles separately:
1,998 of the 8,134 entries differ from their transpose, by at most 4.4e-16
(6.2e-17 of the largest element). By the user's decision the call stays
`s1` and the symmetry is held to the metric's rule.
`test_oracle_three_center_fixture.py` pins both numbers, so a regeneration
that changed them is seen.

## Tolerances

Every bound is stated in its test's docstring before the run.
`max|(mn|P)|` of the fixture is about 7.21.

**Against libcint**, each comparison is held to `1e-12 · max|(mn|P)|` of
the whole fixture tensor:

| Comparison | Worst |
|---|---|
| host, every block of every `L ≤ 4` class, both bra orders (`test_ao_eri3.py`); each block also at 1e-12 of its own absolute-value scale (worst 6.6e-14) | 3.3e-14 |
| interpreted kernel, aux `l ≤ 1` / `l ≤ 2` (`test_kernel_eri3.py`) | 2.06e-14 / 3.31e-14 |
| card, the fixture's twelve `L ≤ 4` classes (`test_eri3_device.py`) | **3.30e-14** |

The block-scale check skips the elements that are zero by parity on one
centre. There the bra's `P_ij` equals the centre only to rounding, so the
computed element (about 1e-17) is input noise the twin cannot bound.
libcint's value is exactly 0, which is asserted, and ours is held to the
tensor-scale bound like every other element.

**Against the host: 16 eps over the tower's absolute-value twin**, the
metric's bound unchanged:

    |T_kernel − T_host| ≤ 16 eps · scale,   eps = 2^-52

where `scale` is the element's sum with every `[t+τ]` replaced by its
**tower's** absolute-value twin, `test_kernel_metric.pq_tower`. That is the
host bound because kernel and host round differently (above), and because a
per-target twin, which takes each `[t+τ]` as one atom, cannot see the
cancellation inside the tower: `metric.md`, "Tolerances", says how it was
found unsound. An element whose scale is 0 must be exactly 0.0.

| Where | Synthetic basis | Fixture |
|---|---|---|
| interpreted | 1.80 eps | within 16 eps (worst not recorded) |
| card | 4.66 eps | 4.77 eps |

The synthetic bases are `test_kernel_eri3.SYNTHETIC` × `SYNTHETIC_AUX`:
three centres, every `l` to g on both sides, with swapped, diagonal and
off-diagonal pairs and one-centre triples in every class. The device bound
deliberately does **not** add `WIDE_BOYS_REL`: the card is held to the
interpreter's yardstick unchanged, as in the metric.

## Where it ran

**On the RTX 3080 (`sm_86`, nvcc 13.0), PR #457.** The command was
`devtools/kernel-tier.sh --backend cuda -j 12 -- -k eri3`. It exited 0
with `device: present`, compiled 55 units with no warning in generated
text, and the device suite gave 2 passed, 0 skipped:

- `test_cuda_every_eri3_class_through_l4_matches_the_host`: Part 1, the 22
  classes on the synthetic bases (145 cells), worst 4.66 eps of the
  tower-twin scale (max abs 2.13e-14).
- `test_cuda_eri3_through_l4_matches_libcint`: Part 2, the fixture's twelve
  `L ≤ 4` classes (666 cells, 7,279 of 8,134 elements owned), worst
  4.77 eps against the host and 3.30e-14 of `max|(mn|P)|` against libcint.
  The fixture's three classes above the cap, (sp|g), (pp|f) and (pp|g),
  are **not launched**, and their elements are asserted to be exactly 0.0.

Both parts assert bit-identical runs and an exactly `(m, n)`-symmetric
tensor. The driver is `test/eri3_driver.py`. At PR #457 it built one
program per (AO basis, aux basis), after `test/metric_driver.py`, launched
every class once into one zeroed `(n_ao, n_ao, n_aux)` tensor `RUNS` times
behind the VRAM probe, and compiled through `plan_driver.compile_checked`,
each call built from the kernel's own parameter list. Since #657 it runs on
the shared runner (#649, `test/harness/`): one runner over every class's
unit and one sequence launching each class into the tensor, `RUNS` times,
the runner failing unless every run is bit-identical to the first. That
port has run through the real runner on the host against a scratch stub
runtime only, not yet on a card.

**Not on the RX 9060 XT (`gfx1200`), at PR #457.** The card was on the
PCI bus but kfd
did not enumerate it (`gfx_target_version` 0, the DKMS state #431
records). `kernel-tier.sh --backend hip -- -k eri3` compiled 55 units with
no warning and then exited 2, `device ABSENT -- numbers unproven`: "could
not run", not a pass. No HIP number is claimed. Only the compile-only
companions ran under hipcc / ROCm 7.2.4 for `gfx1200`, with no warning in
generated text:

- the 22 units, in `test_emit_eri3.py` (#445);
- both device programs, in `test_eri3_device.py` (#446).

By the user's decision the HIP device run was attached to **#431**, the
metric's. #431 closed on 2026-10-17 with the card's detection fixed; the
run itself came in #606's session (PR #615), below.

## The emitted text

`--eri3-l [LMAX]` (4 bare, any `LMAX ≥ 0`, a negative one exits 2) adds
`eri3_{la}{lb}{lP}` for every `la ≤ lb` with `la + lb + lP ≤ LMAX` beside
the J ladders, built as `--metric-l` is (#415). It refuses `--class`,
`--shell-ls`, `--fock-l`, `--fock-class` and `--n-batches` with exit 2.
The 22 `L ≤ 4` units are goldens per backend
(`test/golden/{cuda,hip}/eri3_*`, 44 files). The largest, `eri3_220.hip`,
is 58,818 B, far under the 512 KiB hook, so there is **no digests file**;
`test_emit_eri3.py` asserts both. `--eri3-l 5` is written, not refused
(`slow`), and never compiled or launched.

## The low-stack form, and the ladder through aux i (#606)

The kernel above holds `Ebra[nA nB][n_t(la+lb)]`, `Eket[nP][n_t(lP)]` and
`nA nB nP` accumulators per thread, and nvcc keeps the blocks whole.
FreeBreakfast measured `sm_86` frames of 13,856 B at (dd|g), 32,040 B at
(df|g) and 70,232 B at (ff|g), against `E` blocks of 14,280, 28,680 and
69,000 B. At (ff|g) and full residency that is ~7.3 GB of a 10 GB RTX 3080,
held for the context's lifetime. Unclamped aCD aux sets reach i (`l = 6`),
which makes it worse: (ff|i) has 2,800 accumulators. So `build_eri3_kernel`
has a **low-stack form** (`kernel_eri3.py`'s docstring, "THE LOW-STACK
FORM"), the default wherever `eri3_lowstack(cls)`:

- the same name, parameters, launch and `[m][n][P]` output;
- per-axis `E` tables for the bra (`e_fragment.axis_tables_fragment`) and
  the aux record (`center_tables_fragment`, new);
- the tower's live `[r]^(0)` stored into a local `R` as each one forms
  (`recurrence.store_targets`);
- per aux row, `U[t] = Σ_τ R[t+τ] E^q_τ`, then every bra row's
  `Σ_t E^{ab}_t U[t]`, all as loops (`rolled_center_sum`,
  `rolled_e_sum`);
- the result accumulated **in place** in the cell's own `eri3` slots, which
  it zeroes first, then mirrored by a plain copy. No atomics: each element
  belongs to one cell.

The rule is `L ≥ 10`, or `E` blocks larger than (pf|g)'s (1,575 doubles).
The bound was set from the measurement below. Before it, the bound was
(dd|g)'s 1,785 doubles, and that left (dd|g) unrolled. On `la ≤ lb ≤ 3`,
`lP ≤ 6` the rule gates 25 of 70 classes: every (df|·) and (ff|·), every
aux-i row, and (dd|g), (pf|h) and (dd|h). Nothing at `L ≤ 4` is gated, so
the 22 goldens are unchanged. The metric has the same form. Its bound
stays at 1,785 doubles (`metric.md`).

**From the text**, every gated class declares at most 8,800 B of local
arrays and at most 128 KiB of source (`slow`, `test_kernel_df_lowstack.py`).

**Measured** in `data/measured/df_lowstack.json`
(`measure/measure_df_lowstack.py`). Its rule (`declared`) was written into
the store before any row.

**The compile** (`devtools/measure.sh df-lowstack-compile`, no card)
covers 115 units per arch. That is every eri3 and metric class of the
ladder in its default form, plus the other form where the gate is read:

- the unrolled (df|g) and (ff|g), which are #606's before;
- the low-stack form of every eri3 class the old rule kept unrolled from
  `L = 7`;
- the low-stack form of every such metric class.

Each unit was compiled alone, one at a time, with the measured store's
flags:

- nvcc 13.0, `sm_86`, `-Xptxas -v`; the stack figure is the frame before
  device link;
- hipcc / ROCm 7.2.4, `gfx1200`, `-Rpass-analysis=kernel-resource-usage`;
  the figure is `ScratchSize`.

Each compile had a **300 s budget**. A unit still running at 300 s was
killed and recorded `over_budget`.

| unit | form | `sm_86` | `gfx1200` |
|---|---|---|---|
| `eri3_224` (dd\|g) | unrolled (before) | 255 regs, 15,992 B stack, 27,276 B spill stores, 24.8 s | 192 VGPRs, 10,516 B scratch, 25.1 s |
| | low-stack (now the default) | 162 regs, 3,280 B, no spill, 2.3 s | 144 VGPRs, 3,344 B, 2.0 s |
| `eri3_234` (df\|g) | unrolled | 255 regs, 33,992 B, 54,104 B spill stores, 51.9 s | 192 VGPRs, 18,988 B, 65.6 s |
| | low-stack | 255 regs, 4,608 B, 56 B, 2.0 s | 180 VGPRs, 4,560 B, 3.1 s |
| `eri3_334` (ff\|g) | unrolled | **over budget**, killed at 300 s (1.52 MB of source) | 192 VGPRs, 34,660 B, 242.5 s |
| | low-stack | 255 regs, 6,320 B, 56 B, 2.6 s | 192 VGPRs, 6,320 B, 3.1 s |
| `eri3_134` (pf\|g) | unrolled (kept) | 255 regs, 13,928 B, 22,540 B, 16.6 s | 192 VGPRs, 9,412 B, 25.8 s |
| `eri3_336` (ff\|i) | low-stack | 255 regs, 8,256 B, 68 B, 3.2 s | 192 VGPRs, 8,512 B, 4.7 s |

Across every low-stack unit of the ladder:

- the stack is at most 8,256 B on `sm_86` and 8,512 B of `gfx1200`
  scratch;
- spill stores on `sm_86` are at most 68 B;
- each unit compiles in at most 5.1 s.

Every eri3 class the gate keeps unrolled is at most 13,928 B, with (pf|g)
the largest.

**The decision on (dd|g).** The declared rule admits an unrolled form only
when, on both arches, its stack is at most 14,000 B and it compiles within
300 s. The unrolled (dd|g) is 15,992 B on `sm_86`, so it is not admissible
and must be gated. FreeBreakfast's 13,856 B is after device link; the rule
reads ptxas's per-unit frame.

The bound therefore tightened to (pf|g)'s blocks, the largest admissible
unrolled class. That gates (dd|g) and no other class.
`measure_df_lowstack.test_the_gate_is_what_the_measured_rows_rule` (in
`measure.sh check`) holds the gate to the rows.

**The clock** (`devtools/measure.sh df-lowstack`, behind the card gate)
covers both forms where both compile within the budget. Each class is
timed alone on a synthetic input:

- eri3: 8,704 to 9,216 cells, from 12 to 16 contracted AO shells per `l`
  against 64 aux shells;
- metric: 96 aux shells;
- protocol: 3 rounds × 7 timed launches.

| class | `sm_86` low / unrolled | ratio | `gfx1200` low / unrolled | ratio |
|---|---|---|---|---|
| (dd\|g) | 9.42 / 2.57 ms | 3.67 | 13.12 / 3.91 ms | 3.36 |
| (pf\|g) | 8.46 / 2.31 ms | 3.66 | 12.15 / 3.79 ms | 3.20 |
| (pd\|h) | 4.89 / 1.69 ms | 2.90 | 10.23 / 2.75 ms | 3.71 |
| (sf\|h) | 3.65 / 1.17 ms | 3.13 | 2.84 / 1.59 ms | 1.79 |
| (df\|g) | 18.49 / 6.41 ms | 2.89 | 29.01 / 9.91 ms | 2.93 |
| (ff\|g) | 36.46 ms / not compiled | — | 46.59 / 21.51 ms | 2.17 |
| (ff\|i) | 75.45 ms | — | 133.70 ms | — |

The low-stack form is **slower everywhere**, by 1.8x to 3.7x. So the clock
moves no admissible class under the gate. It is a stack and compile-time
trade, not a speed one. The (dd|g) move costs ~3.5x on that class's
launch. Beside the clock, the two forms' tensors agree within `1e-14` of
max.

**Against libcint through (ff|i).** The tracked aux-i set is
`prim_spd_f_auxi_3c`, five schema-4 column pieces, on the AO parent
`prim_spd_f`:

- AOs: He with s, p and d; Ne with one f contracted over two primitives;
  20 AOs in all.
- Aux: `l = 0..6` shells on both atoms, with Ne's i contracted; 168
  Cartesian functions.

Every one of the 70 ladder classes occurs. The comparison is Cartesian
(`int3c2e_cart`), as the kernels are. #606's spherical comparison is the
consumer's transform, and spherical is not handled here.

- The host matches libcint within 1.16e-14 of max over the whole tensor.
- Interpreted, (ss|i), (pf|h), (df|g), (ff|g) and (ff|i) match within
  4.6e-15 of max.
- The cards: Part 3 below.

**Interpreted** (`test/test_kernel_df_lowstack.py`):

- every L ≤ 4 class with the form forced, against the host within
  16 eps of the tower-twin scale and within `1e-14` of the unrolled kernel;
- by default (`slow`) on f AO shells against h and i aux shells:
  (ff|s), (df|p), (ss|i), (sp|i), (pp|i), (df|i), (ff|h) and (ff|i). Worst
  1.54 eps of the scale against the host, and 5.7e-16 of the class's max
  against the unrolled kernel. (ff|i) itself is 0.75 eps and 7.0e-17.

Only (ss|s) is bitwise. The rest differ by association.

**On both cards** (`test_eri3_device.py` Part 3, `slow` and `high_l`, run
by hand with `pytest -m "gpu and <backend>"`, #606). Every one of the 70
classes runs on the aux-i set in the form the gate picks; 25 of them are
low-stack. There is one program per `lP`, each compiled within the 300 s
budget.

| card | worst against the host | against `int3c2e_cart` |
|---|---|---|
| RTX 3080 | 2.75 eps of the tower-twin scale (gate 16) | 1.19e-14 of max (gate `1e-12`) |
| RX 9060 XT | 2.46 eps | 1.19e-14 of max |

Runs are bit-identical on both cards. The libcint figure is the host's
own, set by the low classes.

The same session ran Parts 1 and 2 (L ≤ 4) on the RX 9060 XT too. That
is the card's first `(mn|P)` run; it was #431's.

## The packed output (FreeBreakfast#1680)

FreeBreakfast's rebuilt density fitting stores `B` packed by kept shell
pair, the pairs' blocks side by side in class order, and streams it from
the host in aux batches. `build_eri3_kernel(cls, packed=True)` is the
kernel that writes that layout directly: `peri3_{la}{lb}{lP}`, the same
records, the same cells, the same arithmetic in either form (unrolled or
low-stack, the same `eri3_lowstack` rule), only the stores moved. The
square `eri3_*` kernels are unchanged, text and all.

**The contract.** The parameters are the square kernel's with `n_ao`
replaced by `n_col` and the three cell offsets `off_a`, `off_b`, `off_p`
replaced by two cell arrays, `dst` and `diag`
(`kernel_eri3.PACKED_CELL_ARRAYS`: `ab_rec0 ab_rec1 p_rec0 p_rec1 dst
diag`), and the output is flat, `n_col · n_aux` doubles:

    block element (a, b, q) of cell k  ->  eri3[dst[k] + n_aux (a nB + b) + q]

`a` over the staged first shell (`la <= lb`, as the square kernel stages
it), `nB = len(cartesians(lb))`, `q` over the aux shell's Cartesians. The
aux index is fastest: the output is `n_col` columns of `n_aux`, a pair's
block `nA nB` consecutive columns, so `dst = column · n_aux + off_p` for a
pair whose block opens at `column`. Everything about the layout is table
contents: the kernel reads `dst` and nothing else, so a batch of pairs
laid out from its own column 0 is just another `dst`.

**A diagonal shell pair writes its full square block** (`diag[k] = 1`,
read only where `la == lb`): `a >= b` is formed, as in the square kernel,
and each element is stored at `(a, b)` and its mirror `(b, a)`. That is
the square kernel's `[off_a + a][off_b + b][off_p + q]` bitwise, where
the square kernel, `kernel_schwarz` and the full ERI keep only the
triangle meaningful; the per-class batched spherical transform downstream
needs complete blocks. An off-diagonal pair stores its block once, with
no `(m, n)` mirror.

**Host side.** `eri.density_fit.eri3.packed_columns(classes, keep=None)`
lays the pairs out in class order from column 0 (`keep` admits a subset,
for a batch) and `packed_cells(cls, columns, n_aux)` forms the cell
arrays; `eri3_drive.launch_packed` runs the kernel interpreted.

**The witness is the square kernel, bitwise.** Every store is plain and
the arithmetic is the same statements in the same order, so there is no
rounding to bound: `test_kernel_eri3_packed.py` (interpreted, every class
at L ≤ 4 in both forms, a two-batch split with batch-relative offsets,
diagonal blocks symmetric) and `test_eri3_packed_device.py` (on a card:
Part 1 every class at L ≤ 4 in both forms; Part 2, `slow` + `high_l`,
all 70 classes through (ff|i) on the aux-i set in the default form and
the low-stack form wherever the default is unrolled) compare the packed
output with `gather_packed` of the square kernel's tensor using
`np.array_equal`. The square kernel's own witnesses (host, libcint) carry
through that equality.

**Where it ran:** CUDA only, on an RTX A6000 (`sm_86`, CUDA 13.0): Parts
1 and 2 bitwise (FreeBreakfast#1680's PR). HIP was not run: no AMD card
on that machine, and hipcc was not invoked. Every unit of the ladder
FreeBreakfast emits (the 70 classes in the default form, plus the 45
default-unrolled classes in the low-stack form) compiles alone under nvcc
(`sm_86`, the tier's `NVCC_FLAGS`, no cache) with no warning, the slowest
in 10.6 s (`peri3_134`), against the 300 s per-file budget.

## The spherical-aux packed output (#743)

FreeBreakfast's rust tree applied `U_aux` to the packed output's
Cartesian aux index afterwards, as one dense GEMM against a block-diagonal
matrix (FreeBreakfast#1736). `build_eri3_kernel(cls, packed=True,
spherical_aux=True)` writes the aux index already in real solid
harmonics: `speri3_{la}{lb}{lP}`, `(ab|Q)` with `a`, `b` Cartesian and
`2 lP + 1` columns per aux shell. Cartesian `P` is never formed.

**Hobson's theorem.** An aux primitive is single-centre, so a solid
harmonic on it is a pure degree-`lP` Hermite function:
`S_m(r_A) e^{-α r_A²} = (2α)^{-lP} S_m(∂_A) e^{-α r_A²}`. Its Hermite
expansion is `E^m_τ = (2α)^{-lP} c[m][τ]` at `|τ| = lP`, zero below.
Every Cartesian row carries lower-degree terms too, and the harmonic
combination cancels them. Per (bra record, aux record) the kernel
therefore drops the single-centre tables and the `rolled_center_sum`, and
per column `m` forms

    U[t] = Σ_{|τ| = lP} c[m][τ] [t+τ]^(0)          every |t| ≤ la + lb
    eri3[a, b, m] ±= s_ij s_k (2α)^{-lP} Σ_t E^{ab}_t U[t]

with the class's one sign `(-1)^{lP}`. The Boys values, the R recurrence
(still to order `la + lb + lP`), the bra tables, the records, the cell
arrays and the launch are unchanged.

**The table and its convention.** `c[m][τ]` is `aux_harmonics(lP)`: the
columns `m = -lP ..= lP` (PySCF/libcint order) over `cartesians(lP)`. Each
is the Schlegel-Frisch real solid harmonic, normalized under the
Gaussian-moment product and scaled by `sqrt((2 lP - 1)!!)`. That is
FreeBreakfast's `solid_harmonic_block(l)` times `sqrt((2l-1)!!)`, one
shell's block of `fb_basis::spherical_transform_matrix`, transposed. s and
p are the identity, so `p` stays `x, y, z`. The kernel spells the table as
literals: the `2 lP + 1` columns are unrolled, each `U` sum runs over that
column's nonzero `τ` only, and the `t` and `(a, b)` loops stay rolled. No
coefficient array is declared, and `(2α)^{-lP}` is folded into the
record's `w`.

**Low-stack only.** The mode exists only in the low-stack form, at every
class: it refuses `lowstack=False` and `packed=False`. The unrolled form,
its measured rule (`E_DOUBLES_MAX`) and every Cartesian kernel are
untouched. Every square and packed unit of the ladder emits the same text
as before, checked by hashing the whole ladder's text against the parent
commit. With the aux tables gone the local arrays shrink at every class
(5,200 against 5,584 B at (ff|f), 7,104 against 8,280 B at (ff|i)). The
unrolled columns grow the text instead: 173 KB against 127 KB at (ff|i).
nvcc compiles `speri3_336` in 7.6 s, against 5.9 s for the low-stack
`peri3_336`.

**Host side.** `eri.density_fit.eri3.spherical_offsets(aux)` gives each aux
shell's first spherical column and the count, which are the kernel's
`off_p` and `n_aux`. `packed_cells(..., aux_offsets=)` forms `dst` from
them.

**The witness is the host, contracted; not bitwise.** For every cell, the
spherical block is compared with the host's Cartesian `class_block`
contracted by `aux_harmonics(lP)` on the aux index. The bound per element
is `(2·TOL_ULPS + nq + lP) eps` times the contracted scale `|c|·scale`.
That covers the Cartesian kernel's own bound on each side (the spherical
kernel's terms are a subset of the `|c|`-weighted Cartesian ones), the
contraction's `nq` roundings, and the `lP` factors of `(2α)^{-lP}`. (ss|s)
is also bitwise the Cartesian low-stack packed kernel. The table is
checked separately through l = 6, with no kernel: harmonic,
`c G cᵀ = (2l-1)!! I`, the column order and signs, and the d columns in
closed form.

**Scope, by decision (#743).** The emitter takes any class. Only (ll|l),
l = 0..3, is tested, to keep compile and test time down: interpreted
(`test_kernel_eri3_spherical.py`, s and p fast, d and f `slow`) and on
both cards (`test_eri3_spherical_device.py`, (ff|f) also `high_l`). No
`la ≠ lb` class and no `lP > 3` runs through a kernel here. The
consumer's full-ladder regeneration (FreeBreakfast#1736) covers those.

**Where it ran:** both cards, in the combined container. On the RTX 3080
(`sm_86`) and the RX 9060 XT (`gfx1200`), all four classes pass, the
worst within 3.0 eps of the contracted scale (bounds 33 to 45 eps).
Interpreted, the worst is 1.74 eps.

## The fully spherical packed output (#745)

FreeBreakfast's consumer still applied the bra's Cartesian-to-spherical
transform after `speri3_`, which needs complete Cartesian blocks.
`build_eri3_kernel(cls, packed=True, spherical_aux=True,
spherical_bra=True)` does that transform in the kernel:
`ssperi3_{la}{lb}{lP}`, `(μν|Q)` with all three indices in real solid
harmonics, `(2la+1)(2lb+1)(2lP+1)` elements a cell. B shrinks by 25/36 at
(dd|·) and 49/100 at (ff|·).

**Why per column, per primitive triple.** The low-stack form has no staged
block: one thread per cell accumulates `out[a,b,m] ±= w Σ_t E^{ab}_t U[t]`
straight into global memory, and per-thread accumulators were removed on
purpose (#606). A cell's spherical region is smaller than its Cartesian
block, so reading the accumulated block back and transforming it in place
would overwrite the next cell. So inside each unrolled aux column `m`,
after `U[t]` is formed:

    V[a,b]       = Σ_t E^{ab}_t U[t]              every Cartesian (a,b), local
    T[b]         = Σ_a c^{la}[μ][a] V[a,b]        one μ at a time
    out[μ,ν,m]  ±= w Σ_b c^{lb}[ν][b] T[b]        global, spherical slots

`c^l` is `aux_harmonics(l)`, the same table and convention as the aux
index, spelled as literals over its nonzero terms (μ and ν unrolled, the
`T` row's `b` rolled). The added locals are `Vab[nA nB]` and `Tmu[nB]`:
880 B at (ff|·), 304 B at (dd|·), independent of `lP`. The global
read-modify-writes per column and primitive triple fall from `nA nB` to
`(2la+1)(2lb+1)` (100 to 49 at (ff)). A diagonal pair forms `a ≥ b` and
mirrors within `V`, accumulates `μ ≥ ν`, and is mirrored on spherical
indices at the end. Boys, the R tower, the bra tables, the aux columns,
the records, the cells and the launch are `speri3_`'s.

**s and p are the identity.** With `la, lb ≤ 1` there is no `V` and no
transform: the body is `speri3_`'s, after one leading note, and since
`2l + 1` is the Cartesian count there, the slots are the same too.

**Host side.** `packed_columns(..., spherical_bra=True)` counts
`(2la+1)(2lb+1)` columns a pair; `eri3_drive.launch_packed(...,
spherical_bra=True)` is the interpreted launcher.

**Low-stack only**, at every class, like #743: it refuses `lowstack=False`,
`packed=False` and `spherical_aux=False`. Every existing kernel's text is
unchanged: the 860 CUDA and HIP units of `eri3_`, `peri3_` and `speri3_`
over `la ≤ lb ≤ 3`, `lP ≤ 6` (default, low-stack and, at L ≤ 6, unrolled
forms) hash identically against the parent commit.

**Stack and compile time**, each unit compiled alone by nvcc 13.0 with the
tier's `NVCC_FLAGS`, no cache, on a GH200 VM's Grace CPU (faster than the
RTX 3080 box: `speri3_336` takes 2.8 s here against 7.6 s there):

| unit | text | `sm_86` frame | spill st / ld | nvcc `sm_86` |
|---|---|---|---|---|
| `speri3_222` | 46 KB | 1,432 B | 68 / 68 B | |
| `ssperi3_222` | 75 KB | 1,904 B | 228 / 284 B | |
| `speri3_333` | 93 KB | 5,264 B | 136 / 156 B | 1.6 s |
| `ssperi3_333` | 178 KB | 6,336 B | 416 / 800 B | 2.5 s |
| `speri3_336` | 173 KB | 7,072 B | 60 / 76 B | 2.8 s |
| `ssperi3_336` | 331 KB | 7,872 B | 72 / 92 B | 5.0 s |

The frame grows by 800 to 1,072 B at (ff|·), about the 880 B of `V` and
`T`; `sm_90` frames are the same to within 16 B. The text roughly doubles
at f, since the transform is unrolled inside each of the `2lP+1` columns;
a rolled transform would need a coefficient array in local memory, which
is the stack this form avoids, so it stays unrolled.

**The witness is the host, contracted on all three indices; not bitwise.**
Per cell, the host's Cartesian `class_block` contracted by `aux_harmonics`
on `a`, `b` and `q`. The bound per element is
`(2·TOL_ULPS + nq + lP + 2(nA + nB) + 1) eps` times the contracted scale
`(|c^{la}| |c^{lb}| |c^{lP}|)·scale`: #743's bound, plus the kernel's two
transform sums and its `w S` product, plus the host's own two bra
contractions. A diagonal pair's block is symmetric in `(μ, ν)` exactly,
and each cell stores exactly its `(2la+1)(2lb+1)(2lP+1)` elements (the
output starts NaN). At `a ≤ 1` the output is also bitwise `speri3_`'s.

**Scope, by decision (#745).** The emitter takes any class. Tested: the 28
classes (aa|p), a = 0..3, p = 0..6, on a two-shell AO basis per `l`
(two diagonal pairs and an off-diagonal one) and two aux shells.
Interpreted (`test_kernel_eri3_fully_spherical.py`; L ≤ 4 fast, any d or
f bra `slow`): every class passes, the worst 2.78 eps of the contracted
scale (bounds 38 to 107 eps). No `la ≠ lb` class runs through a kernel
here.

**Where it ran:** CUDA only, on an NVIDIA GH200 (`sm_90`, CUDA 13.0), the
units built `-arch=sm_86` (the tier's `CUDA_ARCH`; `sm_90` is not a
declared `Arch`) and JIT-compiled from their PTX
(`test_eri3_fully_spherical_device.py`): all 28 classes pass, the worst
2.01 eps of the contracted scale (bounds 38 to 107 eps), each run
bit-identical across the runner's repeats. HIP: not compiled
and not run, no hipcc and no AMD card on that machine.

## Fused against two passes: the clock (#747)

`devtools/measure.sh spherical-bra` (`measure/measure_spherical_bra.py`,
store `data/measured/spherical_bra.json`) times `ssperi3_` against what
FreeBreakfast runs today: `speri3_` followed by a bra transform of its
packed output. The second pass is `sbra3_{la}{lb}`
(`kernel_eri3.build_bra_transform_kernel`): one thread per (pair, aux
column), the aux column fastest, each pair's `nA nB` Cartesian columns read
once into a local block and written once as `(2la+1)(2lb+1)` spherical
ones, with the same `aux_harmonics` literals. It reads and writes each
element once, which is the traffic floor of any transform of that output,
a GEMM's included, so the two-pass arm is a lower bound on the consumer's
and a "faster" verdict for `ssperi3_` is conservative. Host side:
`eri3.bra_pairs` gives each pair's column in both layouts;
`eri3_drive.launch_bra_transform` is the interpreted launcher.
`test_kernel_bra_transform.py` holds it, interpreted at (pd|p) and (dd|p),
to numpy applying the same map to the same input, within
`2 (nA + nB) eps` of `(|c||c|)·|X|`.

The bases are #606's clock bases (`measure_df_lowstack._clock_bases`).
Before timing, each class's two outputs are compared as a sanity witness
(`WITNESS_REL = 1e-11` of the max; recorded rows at most 6e-16). The rule
(`declared`, written before any row): ratio = median fused / median
two-pass, "faster" below 0.95.

**Partial, by decision.** The sweep was stopped after 29 of the 70
classes, (ss|s) through (sf|i) and (ps|s), on a GH200 (`sm_90`, native,
nvcc 13.0). All 29 are "faster": the ratio runs from 0.69 to 0.86, and the
margin grows with the bra's `l` (0.82 at the s and p bras, 0.69 to 0.79 at
(sf|d) and above). At (sf|i) it is 0.217 ms against 0.290 ms.
`sbra3_`'s own launch is small next to `speri3_`'s; most of the margin is
`speri3_`'s own time, from its `nA nB` global read-modify-writes per column
and primitive triple, which `ssperi3_` cuts to `(2la+1)(2lb+1)`. The rest
of the ladder (the d and f bras of both shells) is not timed.
`devtools/measure.sh spherical-bra` with `FJ_SPHERICAL_BRA_ONLY` resumes
it.

**`sm_90` is a compile target** (`device.SM_90`, `device.TARGETS`): with
`CUDA_ARCH=90` the tier and the runner build natively on Hopper (the GH200
and H100 VMs) instead of running the `sm_86` build through PTX JIT. It is
not in `device.ARCHS`, this box's scored cards, so no report, ceiling or
store sweep grows a row for it, and its ceilings stay unmeasured.
`test_eri3_spherical_device.py` and `test_eri3_fully_spherical_device.py`
(the cases below `high_l`) pass natively at `-arch=sm_90`.

## The rolled spherical kernels (#748)

The spherical kernels unroll their aux columns (#743), and `ssperi3_`
unrolls its bra transform inside each of them (#745), because the IR's only
way to carry a coefficient table without stack was literals. The IR now has
a read-only table, `ir.Const` (`Kernel.consts`, `()` for every other
kernel): a name, I32 or F64, and its values, spelled once per unit at file
scope as a `__constant__` array by both backends, and bound by the
interpreter as an array no statement may store to. Every thread of a warp
reads the same entry at once, constant memory's broadcast case.

`build_eri3_kernel(..., spherical_aux=True, rolled=True)` (`rsperi3_`) and
with `spherical_bra=True` (`rssperi3_`) loop over the aux columns `m`, and
over `mu` and `nu`, reading `aux_harmonics` as CSR tables (`hc{l}_off`,
`hc{l}_idx`, `hc{l}_val`, and the Cartesian components `hc{l}_tx/ty/tz` for
the aux shift `t + tau`). Each sum runs over the same nonzero terms in the
same order as the unrolled text, as a left fold from 0 (`0 + x` is `x`), so
interpreted the two forms are **bitwise** (`test_kernel_eri3_rolled.py`, at
(pp|d), (pd|p), (dd|f) and (sf|d) in both modes). On a card they need not
be: the device compiler forms fused multiply-adds from the text it sees, and
on the GH200 (nvcc 13.0, `sm_90`) they are not, both within the host bound,
the worst equal (1.04 eps at (dd|f), 0.09 at (ff|i);
`test_eri3_rolled_device.py`, the host contraction as the oracle).

Each unit compiled alone by nvcc 13.0, the tier's flags, no cache:

| unit | text | `sm_86` frame | spill stores | nvcc |
|---|---|---|---|---|
| `speri3_333` / `rsperi3_333` | 93 / 73 KB | 5,264 / 5,168 B | 136 / 52 B | 1.6 / 0.9 s |
| `ssperi3_333` / `rssperi3_333` | 178 / 75 KB | 6,336 / 6,064 B | 416 / 60 B | 2.7 / 1.2 s |
| `speri3_336` / `rsperi3_336` | 173 / 124 KB | 7,072 / 7,072 B | 60 / 60 B | 3.1 / 1.1 s |
| `ssperi3_336` / `rssperi3_336` | 331 / 126 KB | 7,872 / 7,952 B | 72 / 60 B | 5.3 / 1.4 s |

`sm_90` frames and spills are the same to within 8 B. The rolled
`ssperi3_` text no longer grows with the bra transform (126 KB against
331 KB at (ff|i)), its compile is about 3.8x faster, and the (ff|f) spill
stores drop from 416 to 60 B; the frame moves by at most +80 B.

**Off by default, additive.** The rolled form is behind `rolled=True`,
named apart, and every existing unit's text is unchanged (the 1,000 CUDA and
HIP units of `eri3_`, `peri3_`, `speri3_` and `ssperi3_` over the ladder
hash identically against the parent). Its wall-clock against the unrolled
form was not taken (the #747 sweep was stopped by decision), so whether it
becomes the default is open.

## A warp per cell (#749)

The thread-per-cell low-stack form keeps the tower, `U` and the bra tables
in a 6 to 8 KB per-thread frame and accumulates every element in global
memory, one read-modify-write per column, element and primitive triple.
`build_warp_eri3_kernel(cls)` (`wsperi3_`) and with `spherical_bra=True`
(`wssperi3_`) run a block of one warp (`WARP_LANES` = 32) per cell
(`warp_launch_shape`), with the same parameters, cell arrays and output:

- per bra record, lane 0 forms the axis tables into SHARED memory;
- its aux records go in phases of `warp_slots(cls)` (as many towers as fit
  `WARP_SHARED_DOUBLES` = 2,048 doubles beside `U`, `Vab` and the tables,
  `kernel_d2eri3`'s budget; 32 at (ss|s), 3 at (ff|i)), lane `g` forming
  aux record `k0 + g`'s tower into its SHARED slot of `R`;
- per pair and aux column, between barriers, the lanes split `U[t]` (by
  `t`), `Vab[a, b]` (by `(a, b)`) and the output slots (by `(μ, ν)`), each
  slot accumulated in the lane's registers across every record pair
  (`ceil(slots / 32) (2lP+1)` doubles, 26 at (ff|i));
- each lane stores its slots once at the end, a diagonal pair's mirror
  too. No global read-modify-write.

Every `U`, `Vab`, transform and accumulator update is the thread-per-cell
statement (`aux_harmonics` read from `ConstTables`' CSR, as #748's rolled
form), in the same order per slot, so **interpreted, it is bitwise the
thread-per-cell kernel** (`test_kernel_eri3_warp.py`: (pp|d), (pd|p),
(dd|f), (sf|d) in both modes, and four aux records in phases of 1 and 3).
On the card (`test_eri3_warp_device.py`, (dd|f) and (ff|i), GH200,
native `sm_90`), both forms are held to the host contraction; they are not
bitwise there (each text gets its own fused multiply-adds), with the same
worst error.

**Stack.** `wssperi3_336`: 208 B of stack, 0 spill, 15,080 B of shared
memory, 145 KB of text, 1.2 s of nvcc, against `ssperi3_336`'s 7,872 B,
72 B, none, 331 KB, 5.1 s (`sm_90`); `wssperi3_333`: 144 B against
6,336 B.

**Speed, measured once and not by the sweep** (a script over the harness,
#606's clock bases, the median of 21 runs, GH200):

| class | thread per cell | warp per cell | ratio |
|---|---|---|---|
| (pp\|p) | 0.033 ms | 0.172 ms | 5.18 |
| (dd\|d) | 0.553 ms | 1.028 ms | 1.86 |
| (dd\|i) | 3.153 ms | 2.710 ms | 0.86 |
| (sf\|f) | 0.050 ms | 0.819 ms | 16.5 |
| (ff\|f) | 6.574 ms | 6.095 ms | 0.93 |
| (ff\|i) | 14.008 ms | 11.626 ms | 0.83 |

The warp form wins where a cell's contraction is heavy (both bras d or f,
high `lP`) and loses badly where it is light: there the per-pair barriers
and lane 0's serial table build dominate. On these bases a cell has one
aux record, so the phases hold one tower; deeper aux contraction would
share more. **Opt-in, the caller picks the classes**, as with
`kernel_d2eri3`'s warp-shared tower. Every existing unit's text is
unchanged. HIP is not compiled or run.

## The mixed-precision form

`build_eri3_kernel(..., spherical_aux=True, mixed=True)` (`mssperi3_` with
`spherical_bra`; `mrssperi3_` on the rolled form) is #346's FP32
recurrence inside an FP64 boundary, reusing `j_coulomb/levers/mixed.py`:

| | FP32 | FP64 |
|---|---|---|
| tower `R` (prelude, every `[r]^(m)`), column sums `U` | ✓ | |
| the contraction `Vab = Σ_t E^{ab}_t U[t]` (~95% of the flops at f) | ✓ | |
| `P − A`, `θ` | formed in FP64, narrowed once | |
| Boys `F_m(T)` | | evaluated on `TO_F64(T)`, narrowed at its read |
| bra axis tables | narrowed once per bra record | formed in FP64 |
| bra transform, `w`, the output's accumulation | | ✓ |

Every crossing is an explicit `TO_F32` / `TO_F64`, which the IR enforces.

**Accuracy** (interpreted, which rounds FP32 where the device does; the
host's FP64 contraction as the oracle, on the fully spherical test bases):

| class | worst, FP32 ulps of the contracted scale | max error / the class's max \|B\| |
|---|---|---|
| (ss\|s) | 0.85 | 4.9e-8 |
| (pp\|f) | 1.74 | 2.3e-7 |
| (dd\|d) | 1.59 | 4.4e-7 |
| (dd\|i) | 0.06 | 1.1e-6 |
| (ff\|f) | 0.54 | 4.5e-7 |
| (ff\|i) | 0.02 | 2.3e-6 |

The tests hold it to the declared FP32 tier (`F32_REL`, ~5.4e-4 of the
contracted scale; worst 3.9e-4 of the tier interpreted, 2.3e-4 on the
card). A relative error of 1e-7 to 2e-6 in B is FP32's to give, and is
the question for a consumer: it is far below DF's own fitting error
(~1e-5 relative), but an SCF converged to 1e-8 Eh needs the large
integrals in FP64. The usual answer is dynamic precision (Luehr, Ufimtsev
and Martínez, JCTC 7, 949 (2011)): FP32 only for contributions whose bound
is below a threshold, which needs the screening this tree does not have.

**Speed** (a script over the harness, #606's clock bases, median of 21,
GH200 native `sm_90`), against the same form in FP64:

| class | unrolled FP64 | unrolled mixed | ratio | rolled FP64 | rolled mixed | ratio |
|---|---|---|---|---|---|---|
| (pp\|p) | 0.033 ms | 0.034 ms | 1.02 | 0.054 | 0.055 | 1.03 |
| (dd\|d) | 0.556 | 0.463 | 0.83 | 0.941 | 0.801 | 0.85 |
| (dd\|i) | 3.134 | 2.253 | 0.72 | 4.650 | 3.524 | 0.76 |
| (sf\|f) | 0.049 | 0.048 | 0.97 | 0.248 | 0.257 | 1.04 |
| (ff\|f) | 6.590 | 5.526 | 0.84 | 8.270 | 7.140 | 0.86 |
| (ff\|i) | 13.994 | 12.879 | 0.92 | 18.020 | 16.837 | 0.93 |

Hopper's FP32 rate is only twice its FP64 rate, so this is the floor of
the lever's value; on the RTX 3080 (FP64 at 1/64 of FP32) the contraction
is the whole of the gap, and the ratio there is not measured. The table
also shows **the rolled form's runtime cost** (#748 measured only its
compile): 1.3x to 5x the unrolled form in FP64.

## What is not done

- **Spherical comparison.** #606 asks for agreement with spherical
  `int3c2e`; the kernels are Cartesian, so the check is `int3c2e_cart`,
  through (ff|i).
- **A faster low-stack form.** It is 1.8x to 3.7x slower than the unrolled
  one wherever both compile (the clock above). The gate trades that time
  for stack and compile time.
- **L > 4 on the cc-pVTZ-jkfit fixture.** That fixture's (sp|g), (pp|f)
  and (pp|g) are still not launched. The aux-i set covers those classes.
- **Spherical aux functions beyond the packed output.** The packed
  low-stack kernel has a spherical-aux mode (#743, above) and a fully
  spherical one (#745, above). The square
  kernel, the unrolled form, the metric `(P|Q)` and the DF derivative
  kernels take Cartesian aux functions only.
- **DF-J and DF-K themselves**, and anything that factors or inverts the
  metric `(P|Q)`.
- **Screening.** A `(mn|P)` Schwarz bound is unbuilt, and every cell is
  launched.
- **Measurement beyond #606's.** `data/measured/df_lowstack.json` holds
  compile rows for the whole ladder and wall-clock rows for the gate's
  classes. Nothing else is measured: there are no counters, and no
  `kernels.json` row.
- **A score.** Nothing is scored, because no published board exists.
- **A batched kernel or a lever.** The high classes take the low-stack form
  (#606, above) instead. Splitting the Hermite targets across kernels is
  unbuilt.
- **An HGP twin.** `(mn|P)` on the HGP VRR and bra HRR (`heri3_*`,
  `Target.ERI3` on `Engine.HGP`), the same parameters and output, is built
  for s and p shells, checked interpreted and run on the RTX 3080 only
  (PR #601): `hgp_df.md`.
- **A Rys twin.** `(mn|P)` by Rys quadrature (`reri3_*`, `Target.ERI3` on
  `Engine.RYS`), the same parameters and output, is built through
  `la + lb + lP <= 9`. It is checked interpreted (against libcint through
  (pp|g)) and compiled by nvcc and hipcc, and has run on no card:
  `rys_fock.md`, "The three-center `(mn|P)` on Rys".
- **An aux-major layout** for DF-J (the trade above). Derivatives are
  built elsewhere: `geri3_` (gradient, #603/#613, both cards;
  `gradient.md`) and `d2eri3_` (Hessian, #612 / #694, through (ff|i),
  interpreted and compiled by nvcc and hipcc, on no card; `hessian.md`).
