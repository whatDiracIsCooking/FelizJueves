# The three-center DF integral `(mn|P)` on the McMurchie-Davidson recurrence

**Status: implemented, tested at `L = la + lb + lP ≤ 4`, and run on the
RTX 3080 only (milestone 16, #441–#446; PRs #452–#457).** This is the derived
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
- **Host reference:** `ao/eri3.py` (#443), with its `[t|τ]` evaluator
  injected.
- **Oracle:** libcint's `int3c2e_cart` on `h2o_sto3g_j`'s molecule against
  PySCF's `df.make_auxmol(mol, 'cc-pvtz-jkfit')`, tracked as one schema-4
  piece (#442).
- **Where it ran:** the RTX 3080 (`sm_86`, CUDA 13.0) only (#446, PR #457).
  It has **not** run on the RX 9060 XT (`gfx1200`): kfd did not enumerate
  the card (`gfx_target_version` 0). On `gfx1200` the units and both device
  programs have only been compiled. By the user's decision that run is
  attached to the open follow-up #431, with the metric's.
- **Correctness is tested at `la + lb + lP ≤ 4` only**, the 22 classes
  `la ≤ lb` from (ss|s) through (dd|s). The builder and the emitter accept
  any `L`.

**Nothing here is measured and nothing is scored.** There is no
`measured_store` row for any `eri3_` unit: no registers, spills,
wall-clock or counters. No published board exists. `[JEngine]` (entry 35)
is J only, and the recurrence is McMurchie & Davidson's (entries 39 and 46).
The auxiliary basis is Weigend's cc-pVTZ-jkfit (entry 47), an input here
and not a table to score against. Milestone 16 added no `tables.json`
cell, scoreboard, floor or `claims.json` record.
`data/literature/jengine/limits.json` records this as a `StructuralLimit`
(#447), as #245 did for the one-electron family and #417 for the metric.

**Out of scope:** DF-J and DF-K, anything that factors or inverts the
metric, spherical aux functions, screening, and any measurement (see "What
is not done").

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
  axis (`ao.hermite.e_axis`). This corrects the issue text of #443, which
  put `K_AB` in `dress`.
- **The ket is `ao.metric`'s aux record, unchanged**:
  `s_k = c_k (π/α_k)^{3/2}` by `ao.metric.aux_dress`, with **no** extra
  primitive normalisation, because `c` already carries libcint's
  (`metric.md`, "The aux record"). `E^P` is `ao.metric.center_e`.

So `(2/√π) θ (π/p)^{3/2} (π/α)^{3/2} = 2π^{5/2} / (p α √(p + α))` is
McMurchie & Davidson's `λ`, as for `(P|Q)` and `(ab|cd)`. No kernel and no
host module forms that constant: `(2/√π) θ` is the prelude's `W0`.

## The host reference, `ao/eri3.py`

`eri3_block(sA, sB, sP, *, pq)` computes one ordered shell triple as
`[a][b][q]`; `eri3_block_and_scale` also returns its absolute-value twin.
`eri3_tensor(basis, aux, *, pq)` and `scale_tensor` return
`(n_ao, n_ao, n_aux)`. `ao` imports neither `reference/` nor `plan/`: the
evaluator arrives as `pq=`. Layering gives `ao.eri3` the `ao` rule
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
  records (`zeta_ab`, `P`, `XPA`, `XPB`, `s_ab`, `ao.schwarz`'s record) and
  the aux records (`A`, `alpha`, `s`, `ao.metric`'s).
- A cell is two record ranges, `[ab_rec0, ab_rec1)` and `[p_rec0, p_rec1)`,
  and three offsets, `off_a`, `off_b`, `off_p`.
- `class_block` rebuilds each bra `E` from the record alone and is
  `eri3_block` of the staged shells bitwise.

**The internal twin was not built.** `ao.eri`'s quartet `(ab|c0)`, with `0`
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
  high classes will spill; nothing is measured.
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

`ao/eri3_drive.py` is the interpreted adapter, after `ao/metric_drive.py`,
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
`ao/eri3.py` and honoured by the kernel, the driver and the fixture. It
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
tensor. The driver is `test/eri3_driver.py`, after `test/metric_driver.py`:
one program per (AO basis, aux basis), every class launched once into one
zeroed `(n_ao, n_ao, n_aux)` tensor `RUNS` times behind the VRAM probe,
compiled through `plan_driver.compile_checked`, each call built from the
kernel's own parameter list.

**Not on the RX 9060 XT (`gfx1200`).** The card was on the PCI bus but kfd
did not enumerate it (`gfx_target_version` 0, the DKMS state #431
records). `kernel-tier.sh --backend hip -- -k eri3` compiled 55 units with
no warning and then exited 2, `device ABSENT -- numbers unproven`: "could
not run", not a pass. No HIP number is claimed. Only the compile-only
companions ran under hipcc / ROCm 7.2.4 for `gfx1200`, with no warning in
generated text:

- the 22 units, in `test_emit_eri3.py` (#445);
- both device programs, in `test_eri3_device.py` (#446).

By the user's decision the HIP device run is attached to the open
follow-up **#431**, the metric's.

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

## What is not done

- **The RX 9060 XT run** (#431).
- **Above `L = 4`.** No correctness test covers `L > 4`. The builder and
  the emitter accept any `L`. The tests build `(pd|d)` at `L = 5` and emit
  `--eri3-l 5` (both `slow`); neither is interpreted, compiled or
  launched. The fixture's (sp|g), (pp|f) and (pp|g) are not launched.
- **Spherical aux functions.** Only Cartesian aux functions are handled.
- **DF-J and DF-K themselves**, and anything that factors or inverts the
  metric `(P|Q)`.
- **Screening.** A `(mn|P)` Schwarz bound is unbuilt, and every cell is
  launched.
- **Measurement.** Nothing about `(mn|P)` is measured: no `data/measured/`
  row, no registers, spills, wall-clock or counters.
- **A score.** Nothing is scored, because no published board exists.
- **A batched or blocked kernel for the high classes that spill.** One
  thread holds `nA × nB × nP` accumulators (540 at `(dd|g)`); splitting the
  accumulators or the Hermite targets is unbuilt, as is a lever.
- **An aux-major layout** for DF-J (the trade above), and derivatives.
