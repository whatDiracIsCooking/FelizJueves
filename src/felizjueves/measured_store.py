"""The measured store: `Measured` rows, their file, and what is read off them.

OBSERVATIONS OF A TOOLCHAIN, NOT CLAIMS ABOUT THIS CODE. A row is what one
compiler, driven at one arch with one flag set, said about one emitted
translation unit: registers, spill traffic, and where and when it was asked.
A compiler bump moving a number is news about the compiler. Nothing here is
compared against `[JEngine, v2, Table 4]` -- their kernels, their compiler,
their A100; data/literature/jengine/README.md is why that table is not even
transcribed.

KEYED ON `device.MeasuredKey(spec, arch, toolchain).key()`, as #37 decided:
`Spec` is the generation key and has no arch, so an `sm_86` row and a
`gfx1200` row for the same class must be told apart by the arch, and two
compiler versions on one card by the toolchain.

THE RECORD (#50). Every MEASUREMENT field is optional -- a compile-only row has
no runtime, a timing row has no register count -- but a row with all of them
None records nothing and is refused. The PROVENANCE quintet (`toolchain`,
`arch`, `host`, `gpu`, `date`) is required and non-empty, always.

THE STORE is one JSON file, `data/measured/kernels.json`, in the idiom of
`literature/tables.py`: a `"schema"` version and an `"about"` string up top,
and a refusal -- naming the path -- of a file that is malformed, of another
schema, or whose row disagrees with its own key. A missing store is an error,
not an empty one: `load` raises, so "nothing measured yet" (a valid file with
no rows) and "no store here" cannot be confused. One file for both arches,
not one per arch: rows are keyed on the arch already, `remeasure` merges
into what is there (a one-card remeasure diffs only that card's rows), and
the file's path is what the docs map points at.

THE SUBGRID (#52) is `SUBGRID`, the one declared set of `(L, n_batches)`
cells the compile-side measurement sweeps: L in {7, 10, 13, 16} x n_batches
in {1, 3, 6, 9, 12} -- `[JEngine, v2, §7]`'s own reporting range, and 12
past it (#60) because "9 batches suffice" cannot be judged from below 9.
Every cell is a
default-split `pq_spec(L, n_batches=b)` that the plan layer admits (its
partition refuses an empty batch, #64; test/test_measured.py checks every
cell). Tests parametrize over it, #53's wall-clock rows join on it, and a
cell is added here and nowhere else. Widen per experiment, never by default.

THE K CELLS (#201) are `K_SUBGRID`: `kernel.k_spec(L, n_batches=b)` over the
same `SUBGRID_LS` x `SUBGRID_BATCHES`, cell for cell the J subgrid's twin --
the same L, the same Hermite split `(lp, lq)` and the same batch count, so
the same plan and the same Tables 1-3 row, under a different key (target
slot `k`, `am` in Hermite order, `spec.py`). `SUBGRIDS` maps each target
the store declares cells for to its subgrid. Declaring the cells measures
nothing: no K row exists until one is recorded, and every row and verdict
read off `SUBGRID` is untouched (the analyses join on `Spec.key()`, which a
K cell never equals).
- THE TARGET IS ON THE ROW. `Measured.target` is the row's `Spec.target`,
  as the key's target slot spells it. `merge` stamps it from the key for
  every target but `Target.J`, and the loader refuses a row whose target
  disagrees with its key (a K row under a J key, or a J row under a K key)
  and a non-J key whose row records none. A J row may omit it: every row
  before #201 is a J row and is kept byte for byte, and a row with no
  target is written exactly as before (`as_record`) -- the #159 pattern of
  `arm_kind`.
- THE READ PATH is `rows(..., target=)`: a filter on the key's target slot,
  beside the spec, arch and toolchain ones. `target_of` reads that slot,
  past the `.<engine>` suffix a non-native engine adds (#401: `j.hgp`), and
  `engine_of` reads the suffix (the target's native engine without one). No
  stored key carries a suffix.

THE K ROWS (#205) fill K_SUBGRID on both cards, ADDED beside the J rows
(no J row is rewritten). DECLARED BEFORE ANY SPLIT K ROW EXISTED -- the
only K numbers seen first were two one-batch calibration compiles (L = 7
and 10) -- and not to be tuned after reading one.

- THE COMPILE SIDE is `python -m felizjueves.measured --remeasure --subgrid
  --target k`: `measured.cell_kernels` of a K cell is
  `kernel_k.build_k_kernels`, aggregated into one row as a J cell's batch
  kernels are (spill SUMMED, registers MAXED, `source_sha256` of the batch
  texts concatenated; `raw`'s `jsh_dynamic_bytes` is the K kernel's `Ksh`).
- THE CLOCK is measure/measure_kernel_device.py's opt-in K sweep: each K cell
  and its J twin (`j_twin`: the J subgrid cell of the same L and
  n_batches) compiled, checked and timed BACK TO BACK in one session
  behind the gate, near-T J then K, wide-T K then J (#55's alternation);
  the J twin's statistics land in the K row's raw as
  `wallclock:control:*`, and no J row is re-timed or rewritten. The J
  twin gets the same P, zeta, Q, eta (and the same n_bra x n_ket) as the
  K cell, so the [p|q] work under both is identical.
- SPILL (`k_spill_curve`), per (arch, L), the issue's question: does K's
  spill curve fall faster than J's with batch count? With s_T(b) the
  row's spill_store + spill_load (the cell's whole spill, summed over its
  batch kernels -- the store's own aggregation, as #60 judges it; bytes on
  sm_86, register counts on gfx1200, so only the within-arch reading is
  made) and r_T(b) = s_T(b) / s_T(1) (a curve with no spill at one batch
  is flat at 1 while it stays at 0, and +inf once it spills): at each
  split b, K falls FASTER iff r_K(b) < r_J(b), SLOWER iff r_K(b) > r_J(b),
  else SAME. Per (arch, L): FASTER iff FASTER at every judged split b,
  SLOWER iff SLOWER at every one, NO SPILL iff neither curve ever spills,
  otherwise MIXED. Compiler output, so no tolerance. The worst batch
  kernel's spill (`kernel_spills`) is reported beside it, never judged.
  The J rows are the store's, recompiled in the K session by
  measure/measure_store.py's rechecks (the same text, the same counts).
- WALL-CLOCK (`k_gain`), per (arch, L) and input: each target's best-median
  gain over its one-batch cell, median(b = 1) / min_b median(b), K off the
  K rows and J off the same rows' same-session controls -- the quantity
  `jengine/k-recompute-has-more-merit`'s measured half names (claims.json).
  This module reads it; #207 scores the claim.
- A CELL THAT CANNOT BE MEASURED is recorded with its mechanism, FINAL,
  never "could not measure": in `K_REFUSED[arch]` when it has no row at
  all (the compile side), in `K_UNTIMED[arch]` when it cannot be LAUNCHED
  as kernel_k pins it (its largest batch kernel's `Ksh` over the card's
  per-block shared memory, `K_SMEM_PER_BLOCK`, read off the card), so its
  row, if any, has no clock. A refused cell enters no curve
  (`KSpillCurve.refused` names it), and a refused one-batch cell leaves
  its L NOT JUDGED.

THE F CELL (#278). The store ADMITS a `Target.F` row -- the fused Fock
build of [Fock] §3 on the HGP recurrence (milestone 12) -- before anything
fills it, so the measurement PR (#288) is not a schema PR. Nothing is
measured here and no F row is tracked. What an F row is:
- ITS KEY is `MeasuredKey(Spec(COULOMB, 0, (la, lb, lc, ld), Target.F),
  arch, toolchain).key()`, e.g. ``coulomb/d0/(2,1|1,0)/b1/f/literal@sm_86/
  nvcc 13.0``: the class in AO order (NOT K's Hermite pairing, spec.py),
  one batch, target slot `f`, and no lever -- an F key with a lever or a
  batch count other than 1 is refused (`_check_f`), because the plan layer
  refuses an F spec (`plan.dag.refuse_hgp`) and no lowering lever is
  declared for an HGP kernel. The key cannot see the ordering mistake (an
  F class spelled in Hermite order is a legal F key naming another class);
  that is the writer's, on the Spec.
- ITS TARGET IS ON THE ROW, as for K (#201): `merge` stamps `"f"` and the
  loader refuses an F key whose row records none or another.
- ITS UNITS are the existing per-row convention, unchanged: nvcc reports
  spill BYTES, hipcc spilled REGISTER COUNTS, so each row's `spill_unit`
  and `raw` say which, and an F spill is compared within one arch only.
  The row's shape is the same as every other row's -- no field is added --
  so SCHEMA is not bumped.
- ITS GATE IS NOT OPTIONAL. A timed F row (`runtime_ns` present) must carry
  every `GPU_UTIL_FIELDS` key as a READING: null (unknown) is admitted only
  for the J rows timed before #93 recorded the card's state, and no F row
  predates the gate. An ungated F `runtime_ns` is absent, never unlabelled
  (#288's acceptance); `_check_f` refuses one on `merge` and on `load`.
- THE HEADER gains an `"f"` entry (`F_ABOUT`: the key, the ladder and the
  command that writes the rows) only when the store holds an F row, so the
  tracked file, which holds none, is rewritten byte for byte.

THE HGP SPLIT J / K IS REFUSED (#407). Unlike F (#278), no `j.hgp` /
`k.hgp` row is admitted ahead of its measurement: milestone 14 emits the
`hj_` / `hk_` kernels (`python -m felizjueves.emit --hgp-*`) and times
nothing, so a row now would be registers and spills with no clock beside
them and no declared cell set to belong to. `merge` and `load` refuse any
key on a non-native engine with `HGP_JK_REFUSED`, and the measured driver
refuses to build such a cell; the issue that measures the HGP J / K admits
the row, its cells and its header together.

WALL-CLOCK (#53) is `runtime_ns`, `repeats` and `spread_ns`, written by the
opt-in, `gpu`-marked sweep in measure/measure_kernel_device.py
(`test_<backend>_wallclock_over_the_subgrid`, run alone and uncontended; its
docstring is the protocol). What one cell's number IS: the host
`steady_clock` time of the cell's WHOLE launch sequence -- every batch kernel
of `build_pq_kernels(spec)`, in plan order, into one J zeroed beforehand
(outside the clock), then one device synchronize -- producing one `J_p`.
`runtime_ns` is the MEDIAN over `repeats` timed sequences after a warm-up,
and `spread_ns` the interquartile range of those same sequences, on the
sweep's wide-T input; min, max, the quartiles, the per-batch GPU-event
medians, the problem size, the box's load as sampled before the cell, and
the same statistics on a second, near-T input (`wallclock:near:*`) are in
`raw`, keyed `wallclock:...`. The card's state is in it too (#93,
`GPU_UTIL_FIELDS`): its utilisation as the gate let each input through and
just after it, or null -- unknown, never idle -- for a row timed before the
gate recorded it; the loader refuses a timed row without the keys.
The binary is built from the same emitted text with config.sh's flag set
(the report flag aside, which changes no code). Absolute times are NOT
comparable across vendors (gfx1200 rate-limits fp64): only the ranking over
`n_batches` within one arch is a claim.

ONE ROW CARRIES BOTH SIDES. A timing and a compile observation of one cell
share a `MeasuredKey`, so they are one row: `combine` keeps the other side's
measurements (and `raw`) when, and only when, both observed the same text
(`source_sha256`). A changed kernel drops the stale side rather than pairing
a register count with a time it did not produce. The row's `date`, `host`,
`gpu` and `command` are the latest observation's (a timing row records no
`command`, so the compile's is kept).

NINE AGAINST TWELVE (#60) is `nine_vs_twelve`: whether `[JEngine, v2, §7]`'s
"for L <= 16, at most 9 batches are needed" holds on OUR cards. DECLARED
BEFORE ANY 12-BATCH ROW EXISTED, and not to be tuned after reading one. For
one arch and one subgrid L, with r9 and r12 the rows of pq_spec(L, 9) and
pq_spec(L, 12) on that arch:

- SPILL: 12 reduces spill traffic iff  spill(r12) < spill(r9), spill being
  the row's spill_store + spill_load (the cell's whole spill, summed over
  its batch kernels; bytes on sm_86, register counts on gfx1200 -- only the
  within-arch comparison is made). Compiler output, so no tolerance.
- WALL-CLOCK, judged on BOTH of #53's inputs, separately: 12 improves on an
  input iff  median(r9) - median(r12) > max(IQR(r9), IQR(r12)), i.e. it is
  faster by more than the larger recorded spread. WIDE-T is runtime_ns /
  spread_ns; NEAR-T is raw `wallclock:near:median_ns` and q3 - q1. Near-T
  is the informative one (#53: there splitting wins from L = 13 and b9 is
  best at L = 16), but a win on EITHER input counts against 9.
- 9 SUFFICES at (arch, L) iff 12 does none of the three (SPILL, WIDE, NEAR);
  9 SUFFICES ON AN ARCH iff it suffices at every subgrid L. The two arches
  are judged apart and may disagree; sm_86 and gfx1200 are our cards, not
  the paper's A100, and that is the strongest form the claim can take here.

A LEVER A/B (#55) is `lever_ab`: does a declared ARM lever (`ARMS`, below;
#55 asked it of `lower.RECYCLE`, `"recycle"`) beat the literal kernel?
DECLARED BEFORE ANY LEVER ROW EXISTED, and not to be tuned after reading one.

- THE CELLS are `lever_cells(lever)`: EVERY `SUBGRID` cell with the lever
  added to `Spec.levers` -- all four L and all five batch counts, on both
  cards, so a sign that flips with L, with n_batches or with the arch is
  seen rather than generalised from one cell. A lever cell is its own key
  (`.../recycle@...`), so the literal rows, and everything that reads them
  (`kernelrank`, `nine_vs_twelve`), are untouched by it.
- THE CONTROL IS TIMED IN THE SAME SESSION. The wall-clock sweep, asked for a
  lever (`FJ_WALLCLOCK_LEVER`), compiles and checks BOTH arms of every cell
  and times them back to back (near-T literal then lever, wide-T lever then
  literal, so neither arm always runs first), writing the lever row with the
  literal arm's same-session statistics in its `raw` as `wallclock:control:*`
  (and `wallclock:control:near:*`). The literal row in the store is NOT
  rewritten: a cross-session comparison would mix the box's drift into the
  effect, and rewriting it would move #54's and #60's recorded verdicts.
- PER INPUT (wide-T and near-T, judged apart): with c the control and r the
  lever, the lever is FASTER iff  median(c) - median(r) > max(IQR(c),
  IQR(r)), SLOWER iff  median(r) - median(c) > that same max, else NEUTRAL --
  #60's rule, symmetric.
- PER CELL: WIN if faster on an input and slower on none; LOSS if slower on
  an input and faster on none; MIXED if both; NEUTRAL if neither.
- THE DEFAULT. The lever may become the default on an arch only if it WINS at
  every cell there; the default (no lever) changes only if that holds on
  BOTH arches, and then in its own commit. Anything else -- including the
  arms disagreeing in sign across L, n_batches or arch, which is itself the
  result -- leaves the lever opt-in. Spill numbers are reported beside the
  verdict (`spill`, literal row against lever row, same compiler) and never
  enter it: the clock is the verdict.

THE ARMS (#159). An A/B arm is ONE lever from the DECLARED set `ARMS`, which
maps each lever to its KIND: `LOWERING` (`lower.LEVERS`: the builder's kernel,
rewritten after it is built -- `recycle`) or `PLAN` (the plan layer's choice,
so the builder's kernel itself differs -- `no-sieve`, `plan.sieve.NO_SIEVE`).
They are different experiments on the same kernel family: a lowering arm can
come out as the very same machine code (#55's did), a plan arm cannot.

- DECLARED, NOT ANY STRING. `lever_cells` and `lever_ab` refuse a lever
  outside `ARMS`, so the store never records an arm nothing can reproduce. A
  new arm is ONE entry in `ARMS`: #161's `"tuned"`, a `PLAN` arm, is added
  there in the same change that makes the builder honour it, not before -- a
  declared arm the builder cannot build is a slot no row can fill. This
  module may not import `plan/` (test/test_layering.py), so a plan lever is
  spelled here and test/test_measured.py pins the spelling to the plan
  layer's constant.
- ONE ARM AT A TIME. A Spec carrying two declared levers is two experiments
  at once, and `arm_of` refuses it, naming both: its row could not say which
  lever moved the clock.
- THE KIND IS ON THE ROW. `Measured.arm_kind` is the arm's kind; `merge`
  stamps it from the key's Spec, and the loader refuses a row whose kind
  disagrees with its key, a kind on a row with no arm, and a PLAN arm's row
  without one. A LOWERING row may omit it: #55's recycle rows predate the
  field and are kept byte for byte (they are the control for #55's
  conclusion), and their kind is `ARMS["recycle"]` either way. A row with no
  kind is written exactly as before (`as_record`).
- THE DRIVER (`measured.cell_kernels`) strips only `lower.LEVERS` and passes
  every other lever to `build_pq_kernels`, so a plan arm reaches the plan
  layer and a lowering arm reaches `lower`; test/test_measured.py asserts it.

THE SIEVE ARM (#159, for #162's rows) is `sieve_ab`: does Alg. 3's sieve pay
on our cards, i.e. is the literal (sieved) kernel better than `no-sieve`'s?
DECLARED BEFORE ANY `no-sieve` ROW EXISTED, and not to be tuned after
reading one.

- THE CELLS are `lever_cells(NO_SIEVE)` -- every SUBGRID cell, unsieved --
  plus any cell the recording adds (#162 adds one at L = 11); the rule is
  per cell, so an added cell is judged by it unchanged. The recorded set is
  `SIEVE_CELLS` = `ab_cells(NO_SIEVE)`: the subgrid's 20, then
  `ADDED_CELLS[NO_SIEVE]`, the one-batch default class of L = 11 (#162:
  `(6,0|5,0)`; the published pattern names `[5|6]`, and the plan layer's
  counts depend on L alone, so the default split is the cell the store
  already compiles as its literal L = 11 row, and that row is its spill
  control).
- PER CELL: `lever_ab`'s verdict, with `no-sieve` as the lever and the
  sieved literal kernel as its same-session control -- #55's protocol and
  #60's symmetric rule on both inputs, unchanged. Spill is reported beside
  it (`spill`, literal row then no-sieve row) and never judged.
- PER ARCH: the sieve is SUPPORTED on an arch iff `no-sieve` is a LOSS at
  some cell and a WIN at none; CONTRADICTED iff it is a WIN at some cell and
  a LOSS at none; otherwise NOT SHOWN (NEUTRAL everywhere, a MIXED cell, or
  WINs and LOSSes both -- a sign that flips with L or n_batches, itself the
  result). The two arches are judged apart and may disagree.
- THE DEFAULT. As #55's: the literal (sieved) plan stays the default unless
  `no-sieve` WINS at every cell on BOTH arches, and then in its own commit.

THE TUNED ARM (#163, #161's lever) is `tuned_ab`: does the searched board's
plan (`plan.pipeline.TUNED`: the recorded cut and per-batch seed order of
data/scoreboards/jengine/searched_cuts.json) beat the literal plan on our
cards? DECLARED BEFORE ANY `tuned` ROW EXISTED, and not to be tuned after
reading one. The same shape as THE SIEVE ARM, so the two plan arms compare
with each other as well as with their controls.

- THE CELLS are `lever_cells(TUNED)` -- every SUBGRID cell, tuned -- less
  the cells the lever cannot build. A tuned build is ALWAYS a recorded
  search, never a fall back to the literal plan (#161), and the file records
  n_batches in {1, 3, 6, 9} only, so the four n_batches = 12 cells raise "no
  recorded choice". They are `NOT_MEASURABLE[TUNED]`, each with that reason:
  RECORDED as not measurable, never silently skipped, and never replaced by
  the literal arm (that would time a kernel against itself). The recorded
  set is `TUNED_CELLS` = `ab_cells(TUNED)`: the subgrid's other 16, no
  added cell. Where the search chose what the literal plan does, the tuned
  kernel's text is the literal kernel's (#161: (7,3), (7,6) and (10,9) of
  the subgrid); such a cell is timed and judged like any other -- it is the
  A/B's own null, as #55's identical machine code was.
- PER CELL: `lever_ab`'s verdict, with `tuned` as the lever and the literal
  kernel as its same-session control -- #55's protocol, #162's plan-arm
  sweep (discarded phases retaken and counted) and #60's symmetric rule on
  both inputs, unchanged. Spill is reported beside it (`spill`, literal row
  then tuned row) and never judged.
- PER ARCH: the search is SUPPORTED on an arch iff `tuned` is a WIN at some
  cell and a LOSS at none; CONTRADICTED iff it is a LOSS at some cell and a
  WIN at none; otherwise NOT SHOWN (NEUTRAL everywhere, a MIXED cell, or
  WINs and LOSSes both). The sign is the sieve's reversed because the
  question is: the sieve arm asks whether taking a step away hurts, this
  one whether the searched step helps. A not-measurable cell enters neither
  side. The two arches are judged apart and may disagree.
- THE DEFAULT. As #55's: the literal plan stays the default unless `tuned`
  WINS at every cell of `TUNED_CELLS` on BOTH arches, and then in its own
  commit (#166). A not-measurable cell has no tuned kernel at all, so it
  can neither block nor carry that change; it is named beside the verdict.

COUNTERS (#110) are the roofline's inputs, read off the card's own
performance counters by a profiler -- `ncu` on NVIDIA, `rocprofv3`'s counter
mode on AMD -- never off a model:

- `fp64_flops`: fp64 floating-point operations EXECUTED, an FMA counted as
  two (ncu: `smsp__sass_thread_inst_executed_op_{dadd,dmul,dfma}_pred_on.sum`,
  dadd + dmul + 2 x dfma, thread-level, so a FLOP count and not a warp
  instruction count). gfx1200 exposes NO fp64 instruction counter
  (rocprofiler-sdk's counter definitions have none for gfx12), so an AMD row
  leaves it None: unmeasured, never zero, and never borrowed from the
  NVIDIA row or from a model.
- `dram_bytes_read`, `dram_bytes_written`: bytes moved between the card's
  last counted cache level and memory. ncu: `dram__bytes_read.sum` /
  `dram__bytes_write.sum`, DRAM traffic at 32-byte-sector granularity.
  gfx1200: the GL2C <-> EA (memory fabric) requests,
  32 x `GL2C_EA_RDREQ_32B_sum` + 64 x `..._64B_sum` + 128 x `..._128B_sum`
  read, and 64 x `GL2C_EA_WRREQ_64B_sum` + 32 x (`GL2C_EA_WRREQ_sum` -
  `GL2C_EA_WRREQ_64B_sum`) written -- L2 misses, which RDNA4's Infinity
  Cache may serve before DRAM, so an UPPER bound on DRAM bytes (#140: on
  this box they all read 0 for streamed traffic, and the driver refuses
  that reading -- see UNMEASURED). The two
  vendors' numbers share a field name, not a meaning: `counter_unit` says
  which, and nothing compares them across vendors.
- `counter_ns`: the profiler's own duration of the kernel UNDER COUNTER
  COLLECTION (ncu `gpu__time_duration.sum`; rocprofv3 End - Start
  timestamps). Collection serialises and replays kernels, so this is NOT a
  timing measurement and never lands in `runtime_ns`: it is the denominator
  of `achieved_flops_per_s` and nothing else.

THE MEASURED COUNTS ARE CANONICAL; the ratios are derived, never stored:
`dram_bytes` (read + written), `arithmetic_intensity` (FLOP/byte) and
`achieved_flops_per_s` (FLOP/s) are properties, so a stored ratio cannot drift
from its inputs. `counter_tool` is the profiler as it reports itself
(`"ncu 2025.3.1.0"`), `counter_unit` what each field means on that vendor,
`command` the full invocation and `raw` every counter by its EXACT name --
a counter renamed between toolchain versions is the drift this shape exists
to surface. The fields are serialised only when present (`as_record`), so a
row without counters reads and writes back exactly as before. Nothing here
is comparable to `[JEngine, v2, Fig. 7]`: their kernels, their A100.

THE COUNTER ROW SHAPE (#110, for #112): ONE ROW PER BATCH KERNEL, in their
own store, `data/measured/roofline.json` (`load_counters` / `save_counters`),
keyed

    kernel_key(MeasuredKey(spec, arch, toolchain), batch)
      == MeasuredKey(...).key() + "#" + str(batch)

`batch` being the kernel's 0-based position in the cell's plan order
(`measured.cell_kernels(spec)`), so a cell of `n_batches` b has exactly the
rows `#0` .. `#b-1` -- a one-batch cell has the one row `#0` -- and the loader
refuses a cell with a missing or extra kernel row, or kernel rows from two
profilers, cards or vendor meanings. The CELL's counters are NOT stored:
`cell_counters` derives them (every count summed over the cell's kernels,
each launched once per class), so the per-kernel and per-cell verdicts are
read off one set of numbers and cannot disagree. Why a separate store rather
than a batch index on `MeasuredKey` or a nested field: `kernels.json`'s 40
literal rows mean one cell each, and `kernelrank`, `batchreport` and both
verdicts read them so; a per-kernel row filed beside them would change what
"the row for this key" is, while one in its own file changes nothing they
read. A counter row carries counter measurements only (a compile observation
belongs in `kernels.json`, a timing in its `runtime_ns`), and the loader
refuses anything else in it.

THE ROOFLINE ROWS (#112) are those counter rows, filled over `SUBGRID` by
measure/measure_kernel_device.py's opt-in counter sweep (FJ_ROOFLINE_STORE) behind
the card gate (`felizjueves.gate`). Each row carries the card's state around
its phase in `raw` -- ROOFLINE_STATE_FIELDS: the gate's settled reading
before, the reading after (both KNOWN: a phase whose after-reading is
non-idle or unknown was discarded and retaken), the CPU's, and how many
phases of that cell the gate discarded -- and the loader refuses a row
without them. The VERDICT is computed, never stored: `roofline_point` reads
a row's derived intensity and achieved FLOP/s and asks `device.verdict`
against the card's MEASURED ridge (#109), so `memory_bound` cannot be
written by hand, and a card without both ceilings gets no point at all
(`device.Uncharacterised`) -- nor any row: the loader refuses a counter row
for an uncharacterised arch. `kernel_roofline` is the per-batch-kernel
verdict, `cell_roofline` the cell's (its counters summed, so its intensity is
total FLOPs over total bytes and its rate total FLOPs over the summed
serialised durations); the paper's own [7|7] finding is that SOME kernels of
a split stay memory-bound, which only the per-kernel view can see. A card in
UNMEASURED has no rows by decision, with the reason recorded (the header's
`could_not_measure`), and the loader refuses one: COULD NOT MEASURE IS NOT
ZERO, and not a missing file either.

THE BUILD ROWS (#146) are the first measurement of a BUILD rather than a
kernel: the host wall-clock of one whole-molecule `J_p` from a
`launch.LaunchPlan` on one card, timed behind the card gate by
measure/measure_j_launched_ondevice.py's opt-in sweep (FJ_LAUNCH_WALLCLOCK_STORE).
They live in their own file, `data/measured/builds.json` (`BuildRow`,
`build_key`, `load_builds` / `save_builds`), because no `Spec` names a
molecule's plan and a kernels.json row is never widened; `BUILD_TIMED_REGION`
says exactly what the number contains (and what it does not: the uploads,
the host eq. 20), and `BUILD_STATE_FIELDS` are the gate's KNOWN readings
every row carries -- a contended phase is discarded, so the loader refuses a
null or over-threshold reading rather than read it as idle. A row is keyed by
fixture, tau, n_batches (or `dispatch:<input>`), the plan's `density_sort`
and an ARM label, so #147's same-session sort-key A/B files two rows that
differ in their sort only (both carry the arm `SORT_AB_ARM`).

THE SORT KEY (#147) is `sort_key_ab`: is the plan's per-record product key
(`launch.launches(density_sort=True)`, ket key `I_q * max_q |D_q|`) better
than `density_sort=False`'s one global `max|D|` (ket key `I_q * Dmax_global`,
the order by `I_q` alone)? #147 declared the expectation before any row --
"the global key is worse. If it is not, that is the result" -- and this rule,
written before any sort-key row existed, is that sentence made checkable. It
is not to be tuned after reading one.

- THE CELLS are `SORT_KEY_CELLS`: (input, tau, n_batches) = H2O/cc-pVDZ
  (`h2o_ccpvdz_j`, d shells, 25 classes, dispatched as #146's cells are) and
  an extended chain (`h2chain2_sto3g`: s only, one class, one batch), each
  at tau = 1e-3 (#146's screened cell) and tau = 1e-10 (a production SCF
  threshold). Same molecule, same tau, same batch count per pair of arms:
  the arms differ in `density_sort` and nothing else. THE CHAIN IS TWO H2,
  not #88/#122's four: test/test_ao_screening.py's `_h2_chain(2)` (bond 1.4,
  centres 6 bohr apart, its hilbert density). The four-H2 chain's sweep
  asks for T = theta^2 |PQ|^2 up to 1289 -- inside reference.boys' T <=
  2000, but past the card's `fj_boys` as it was then (the Kummer series
  alone, witnessed on [0, 500] and overflowing near T = 700; #301 later
  gave it reference.boys' whole domain); three H2 reach 615. Two H2 reach
  188, inside the device Boys' witnessed domain under either key: the
  Schwarz screen does not see the bra-ket distance, so the product key's
  screen removes no large-T pair either. The four-H2 chain's skip counts
  are still read off the launch oracle in the fast tier
  (test/test_launch_oracle.py), where no card is needed.
- BOTH ARMS IN ONE SESSION, behind the gate, per card: one `time_builds`
  call times every cell's two arms (alternating which arm runs first from
  cell to cell), and each row carries raw 'build:session', the session's
  start in integer seconds -- `sort_key_ab` refuses two arms whose sessions
  differ. Each arm has its own gate readings (BUILD_STATE_FIELDS).
- SKIP (the screen's work, the launch oracle's count on the same plan, raw
  'build:n_skipped'; deterministic, so no tolerance): the global key skips
  FEWER pairs, the SAME number, or MORE. MORE cannot happen -- the global
  value bounds every record's `Dmax`, so its screen is a superset test (the
  structural record, jengine/ket-sort-key-is-the-product, (d)) -- and a row
  pair that says so is refused, not judged.
- CLOCK (the build's `runtime_ns` median and `spread_ns` IQR, BUILD_TIMED_
  REGION): #60's rule, symmetric, as `lever_ab` applies it -- with p the
  product arm and g the global one, the global key is SLOWER iff median(g) -
  median(p) > max(IQR(p), IQR(g)), FASTER iff median(p) - median(g) > that
  same max, else NEUTRAL. The larger IQR is this box's noise floor, the one
  every same-session A/B here has used (#55, #60, #113).
- PER CELL: WORSE if the global key is SLOWER; BETTER if it is FASTER (a win
  for the global key is the result the issue anticipates as possible);
  otherwise, the clock NEUTRAL, UNRESOLVED if it skips FEWER (it does less
  screening, but the time saved is below what this box can resolve) and SAME
  if it skips the same.
- PER ARCH: WORSE iff every cell is WORSE; CONTRADICTED iff any cell is
  BETTER; otherwise NOT SHOWN. The two arches are judged apart.
- THE MEASURED CLAIM, jengine/ket-sort-key-beats-the-global-max, is
  `asserted` iff the verdict is WORSE on BOTH arches. NOT SHOWN anywhere
  lands in data/literature/jengine/limits.json (below what this box can
  resolve) with the per-cell verdicts; CONTRADICTED is a disagreement with
  [JEngine, v2, §3.3], recorded as the result.

THE LEVER ROW (#339). A kernel-body lever (`levers.VOCABULARY`, #337)
is the third arm KIND, `KERNEL`, beside `LOWERING` and `PLAN`: the builder
itself reads it, so the kernel differs. It is DECLARED BY ITS VOCABULARY
ENTRY, not here -- `kind_of` reads the vocabulary at call time, so a lever
issue adds no entry to `ARMS` and no row format of its own. Every lever row
of every kind goes through ONE recording path:

- THE CELL is `lever_arm(spec, lever)`: the literal cell `spec` (any target
  -- a SUBGRID or `K_SUBGRID` cell, a one-batch class, an F class) with the
  lever added. It refuses an undeclared lever, a cell that already carries
  one (ONE ARM AT A TIME), and, through `Spec` itself, a (target, lever) the
  lever's legality table refuses, with its reason. `lever_cells(lever,
  cells)` is that over a set of cells (SUBGRID by default, as #55 had it).
- THE KEY is the full lever-carrying `Spec.key()` in a `MeasuredKey`: the
  lever is the key's last segment, exactly as `no-sieve` and `recycle` have
  always been spelled -- no new field, no new key shape.
- THE ROW is written by `merge`, which stamps `arm_kind` = `KERNEL` (a
  `KERNEL` row without it is refused on load, as a `PLAN` row is), and an F
  key may carry a kernel-body lever (THE F CELL's "no lever" is now "no lever
  outside the vocabulary"). The driver names a kernel lever row's units
  `lever_ladder.unit_name`, so its `source_sha256` is the emitted file's.
- THE HEADER gains its "kernel_levers" entry only once the store holds a
  `KERNEL` row (`save`), as "f" did: a store without one is written byte for
  byte as before.

THE LEVER LEDGER (#362). `felizjueves.leverledger` reads, per (class, arm,
arch), a kernel lever row beside the literal row of its class; the cells it
reads are declared HERE, so the driver that records them and the report that
reads them name one set:

- THE CLASSES (`ledger_classes`): J `pq_spec(L)`, K `k_spec(L)`, one batch,
  and every class of the 21-class F ladder, at total L <= `LEDGER_MAX_L`.
  The literal classes are the baseline, recorded compile-only like every
  ledger row (no K or F literal row at these L carries or needs a clock).
- THE ARMS (`ledger_arms`): every kernel-body lever of `levers.VOCABULARY`,
  read when asked. A lever whose builder refuses it alone (`REQUIRES`) is
  recorded as its COMPOSITION (`batched-vrr+sliced-hrr`), and `COMPOSITIONS`
  adds the arms that are compositions by declaration (`interleave+split`,
  #348). A composition of kernel-body levers is ONE arm of kind KERNEL
  (`is_composition`): `lever_arm` takes its key spelling, `arm_of` returns
  it, and a row keyed by it is not "two arms at once" -- a plan or lowering
  lever still never composes with anything.
- A COMPILE-ONLY KERNEL ROW is admitted on every target, F included: #278's
  gate binds a TIMED F row (its readings may not be null), and a ledger row
  is not timed.

THE LEVER CLOCK (#363). Every ledger arm against its literal class, on the
clock, both cards. DECLARED BEFORE ANY TIMED KERNEL ROW EXISTED, and not to
be tuned after reading one: the predecessor's lesson is that a spill win is
not a wall-clock win, so the ledger's compiler readings are never the
verdict.

- THE CELLS are `lever_clock_cells()`: every ledger arm on every class of
  its target at total L <= LEDGER_MAX_L (the ledger's cells less its
  literal baselines) -- the losers are cells like the winners.
- THE ROW is the ledger row itself: the clock is ADDED to the arm's
  compile row by `combine` (the same emitted text, `source_sha256`), so no
  key, row or field changes shape and the literal rows are not rewritten.
  It carries `runtime_ns` (the arm's wide-input median), `spread_ns` (its
  IQR), `repeats`, raw `wallclock:near:*` (the near input) and, timed in
  the SAME gated phase, the literal class's statistics as
  `wallclock:control:*` / `wallclock:control:near:*` -- #55's shape, so
  `lever_ab` reads it unchanged. The contention state is on the row: every
  `GPU_UTIL_FIELDS` reading (and the control's), the CPU busy fraction
  before, the 1-minute load, and `wallclock:gate_discarded_phases`.
- THE GATE (`felizjueves.gate`): one phase per cell -- control near, arm
  near, arm wide, control wide, back to back -- behind `gate.wait_quiet`
  and closed by `gate.after_reading`; a phase the gate refuses, or whose
  after-reading is non-idle or UNKNOWN, is discarded and retaken, never
  recorded. A cell whose every take is refused has NO ROW.
- PER CELL, `lever_ab`'s rule (#60, both inputs; WIN / LOSS / NEUTRAL /
  MIXED), unchanged.
- PER (ARM, ARCH), `lever_clock_verdict`: FASTER iff the arm WINS at some
  class and LOSES or is MIXED at none; SLOWER iff it LOSES at some class
  and WINS or is MIXED at none; NOT SHOWN otherwise (neutral everywhere, or
  both signs). A cell not timed is refused (KeyError), never skipped.
- THE DEFAULT (`lever_may_become_the_default`): only an arm that WINS at
  every class on BOTH cards, and then in its own commit; nothing here
  moves a default.

THE SPLIT (#94). This module is the store and nothing else: the record, the
file, the read path, the declared subgrid, and the two verdicts read off the
rows. It knows no toolchain -- it runs no compiler, reads no config.sh,
queries no card and spells no kernel -- so the fast tier, `kernelrank` and
`batchreport` read it without one. The driver that compiles, parses what the
compilers print and fills rows is `felizjueves.measured` (`python -m
felizjueves.measured --remeasure`), which writes INTO this; the wall-clock
side is written by measure/measure_kernel_device.py's sweep. test/test_layering.py
holds the direction: the store may not import the driver, `emit/` or a
toolchain module, and the analyses may import the store but not the driver.

Nothing under `plan/` or `literature/` may import this module
(test/test_layering.py): a measurement may never feed a literature floor.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType

from felizjueves import levers as _levers
from felizjueves.device import (
    MEMORY_BOUND,
    Arch,
    MeasuredKey,
    by_name,
    require_characterised,
    verdict,
)
from felizjueves.kernel import k_spec, pq_spec
from felizjueves.lower import LEVERS
from felizjueves.spec import NATIVE_ENGINE, Engine, Operator, Spec, Target

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "data" / "measured" / "kernels.json"
ROOFLINE = REPO / "data" / "measured" / "roofline.json"  # counter rows (#110)

# THE declared subgrid (#52): see the module docstring. One edit adds a cell.
SUBGRID_LS = (7, 10, 13, 16)
SUBGRID_BATCHES = (1, 3, 6, 9, 12)
SUBGRID: tuple[Spec, ...] = tuple(
    pq_spec(L, n_batches=b) for L in SUBGRID_LS for b in SUBGRID_BATCHES)

# THE K CELLS (#201; module docstring): the J subgrid's twins, cell for cell.
K_SUBGRID: tuple[Spec, ...] = tuple(
    k_spec(L, n_batches=b) for L in SUBGRID_LS for b in SUBGRID_BATCHES)

# Each target the store declares a subgrid for, and that subgrid.
SUBGRIDS: Mapping[Target, tuple[Spec, ...]] = MappingProxyType({
    Target.J: SUBGRID, Target.K: K_SUBGRID})

# THE K CELLS THAT CANNOT BE MEASURED (#205; module docstring, THE K ROWS),
# per arch, each with its mechanism, FINAL -- never "could not measure".
# K_REFUSED: no row at all (the compile side). K_UNTIMED: no wall-clock (the
# cell cannot be LAUNCHED as kernel_k pins it); its compile row may exist.
K_SMEM_PER_BLOCK = MappingProxyType({  # read off each card (K_HARNESS limits)
    "sm_86": (49152, 101376),   # (per-block default, opt-in maximum)
    "gfx1200": (65536, 65536),  # LDS: no opt-in above the default
})
# The largest batch kernel's `Ksh` doubles per lane, per K cell of L >= 13
# (measured.jsh_slots of measured.cell_kernels; test_kernel_device.py
# rebuilds and re-reads them, `slow`).
K_MAX_SLOTS = MappingProxyType({
    (13, 1): 270, (13, 3): 266, (13, 6): 264, (13, 9): 266, (13, 12): 266,
    (16, 1): 390, (16, 3): 389, (16, 6): 389, (16, 9): 389, (16, 12): 389,
})
K_LANES = 32  # sm_86's warp, gfx1200's wave32: the pinned blockDim.y


def _untimed(arch: str) -> Mapping[Spec, str]:
    smem, optin = K_SMEM_PER_BLOCK[arch]
    out = {}
    for s in K_SUBGRID:
        slots = K_MAX_SLOTS.get((s.total_am, s.n_batches))
        if slots is None or slots * K_LANES * 8 <= smem:
            continue
        out[s] = (
            f"its largest batch kernel's Ksh is {slots} doubles x "
            f"{K_LANES} lanes x 8 = {slots * K_LANES * 8} B of dynamic "
            f"shared memory per block, over the {smem} B {arch} gives a "
            "block at launch: kernel_k's pinned launch (one warp per bra "
            "record, I and the K accumulator in one shared array, no "
            "opt-in -- the J twin needs none) cannot run it"
            + (f" (an opt-in to {optin} B exists on this card, outside the "
               "pinned launch; the J twin it is compared with never uses it)"
               if optin > smem else " (no opt-in exists on this card)"))
    return MappingProxyType(out)


K_UNTIMED: Mapping[str, Mapping[Spec, str]] = MappingProxyType(
    {a: _untimed(a) for a in K_SMEM_PER_BLOCK})
K_REFUSED: Mapping[str, Mapping[Spec, str]] = MappingProxyType({})


# THE ARMS (#159; module docstring): every lever an A/B arm may carry, and
# its kind. One entry adds an arm -- #161 adds `"tuned": PLAN` with the lever.
LOWERING, PLAN = "lowering", "plan"
NO_SIEVE = "no-sieve"  # plan.sieve.NO_SIEVE, spelled: plan/ is off-limits here
TUNED = "tuned"  # plan.pipeline.TUNED, spelled likewise
ARMS: Mapping[str, str] = MappingProxyType({
    **dict.fromkeys(sorted(LEVERS), LOWERING),  # "recycle" (#55)
    NO_SIEVE: PLAN,  # Alg. 3's sieve off (#162)
    TUNED: PLAN,  # plan.pipeline.TUNED: the searched board's plan (#161)
})


# THE LEVER ROW (#339; module docstring): a kernel-body lever's kind. Its
# declaration is its `levers.VOCABULARY` entry, read when asked, not here.
KERNEL = "kernel"


def kind_of(lever: str) -> str:
    """The arm kind of `lever`: `ARMS[lever]` (LOWERING or PLAN), or KERNEL
    for a kernel-body lever (`levers.VOCABULARY`, #337). Anything else is
    refused: the store never records an arm nothing can reproduce."""
    if lever in ARMS:
        return ARMS[lever]
    if _levers.is_kernel_lever(lever) or is_composition(
            tuple(lever.split("+"))):
        return KERNEL
    declared = sorted({*ARMS, *_levers.VOCABULARY})
    raise ValueError(f"lever {lever!r} is not a declared arm lever "
                     f"({', '.join(declared)})")


def _declared(lever: str) -> str:
    kind_of(lever)
    return lever


def is_composition(levers: tuple[str, ...]) -> bool:
    """True when `levers` is a KERNEL COMPOSITION (#362; module docstring,
    THE LEVER LEDGER): two or more levers, every one a kernel-body lever
    (`levers.VOCABULARY`). Such a set is ONE arm of kind KERNEL -- a lever
    that only builds with another (`sliced-hrr` on `batched-vrr`, #356) or a
    declared composition (`interleave+split`, #348) -- never two arms at
    once. A plan or lowering lever never composes."""
    return len(levers) > 1 and all(_levers.is_kernel_lever(lv)
                                   for lv in levers)


def arm_of(spec: Spec) -> str | None:
    """The one declared arm lever `spec` carries, or None for a literal cell.

    Refuses a lever outside `ARMS` and the kernel-body vocabulary, and a Spec
    carrying two arms at once, naming both (module docstring, THE ARMS). A
    kernel COMPOSITION (`is_composition`, #362) is one arm, returned in its
    key spelling (`"interleave+split"`)."""
    for lv in spec.levers:
        _declared(lv)
    if is_composition(spec.levers):
        return "+".join(spec.levers)
    if len(spec.levers) > 1:
        raise ValueError(f"{spec.key()}: carries {len(spec.levers)} arms at "
                         f"once ({', '.join(spec.levers)}); an A/B arm is "
                         "one lever")
    return spec.levers[0] if spec.levers else None


def arm_kind(spec: Spec) -> str | None:
    """`kind_of(arm_of(spec))`: LOWERING, PLAN or KERNEL, or None for a
    literal cell."""
    lever = arm_of(spec)
    return None if lever is None else kind_of(lever)


def lever_arm(spec: Spec, lever: str) -> Spec:
    """THE LEVER ROW's cell (#339; module docstring): the literal cell `spec`
    with the declared arm `lever` added -- the one way every lever row of
    every kind and target is named. Raises `ValueError` for an undeclared
    lever, a cell already carrying a lever, and a (target, lever) the
    lever's legality table refuses (`Spec`'s own refusal, its reason kept).
    `lever` may be a kernel COMPOSITION in its key spelling ("a+b", #362):
    one arm, its levers added sorted, as `Spec` requires.
    """
    _declared(lever)
    if spec.levers:
        raise ValueError(f"{spec.key()}: already carries "
                         f"{', '.join(spec.levers)}; an A/B arm is one lever "
                         "added to a literal cell")
    return replace(spec, levers=tuple(sorted(lever.split("+"))))


def lever_cells(lever: str,
                cells: tuple[Spec, ...] | None = None) -> tuple[Spec, ...]:
    """`lever_arm` of every cell of `cells` -- by default #55's A/B cells,
    every SUBGRID cell (module docstring, A LEVER A/B). Only a declared arm
    is accepted, of any kind."""
    return tuple(lever_arm(s, lever)
                 for s in (SUBGRID if cells is None else cells))


# Cells an A/B adds beyond the subgrid, per arm (#162; module docstring, THE
# SIEVE ARM): the sieve's one-batch L = 11 cell. `lever_cells` stays the
# subgrid's, so #55's cells and verdict are untouched.
ADDED_CELLS: Mapping[str, tuple[Spec, ...]] = MappingProxyType({
    NO_SIEVE: (pq_spec(11, levers=(NO_SIEVE,)),),
})


# Cells of `lever_cells(lever)` the arm cannot build, each with its reason
# (#163; module docstring, THE TUNED ARM): recorded, never silently skipped,
# never replaced by the literal arm. test/test_measured.py builds every
# `lever_cells(TUNED)` cell and holds this map to exactly the ones refused.
_NO_CHOICE = ("no recorded choice: data/scoreboards/jengine/searched_cuts.json "
              "records n_batches in {1, 3, 6, 9} only, and a tuned build is "
              "always a recorded search, never a fall back to the literal plan")
NOT_MEASURABLE: Mapping[str, Mapping[Spec, str]] = MappingProxyType({
    TUNED: MappingProxyType({s: _NO_CHOICE for s in lever_cells(TUNED)
                             if s.n_batches == 12}),
})


def ab_cells(lever: str) -> tuple[Spec, ...]:
    """Every cell an A/B of `lever` records: `lever_cells(lever)` less the
    arm's `NOT_MEASURABLE` cells, then the arm's `ADDED_CELLS` -- what the
    driver compiles and the wall-clock sweep times for `--lever` /
    `FJ_WALLCLOCK_LEVER`."""
    gone = NOT_MEASURABLE.get(lever, {})
    return (tuple(s for s in lever_cells(lever) if s not in gone)
            + ADDED_CELLS.get(lever, ()))


# THE LEVER LEDGER (#362; module docstring): the cells the ledger reads.
# Its classes are every target's ladder at total L <= LEDGER_MAX_L (the
# milestone's cap); its arms every kernel-body lever of `levers.VOCABULARY`,
# read when asked, closed over REQUIRES, plus the declared COMPOSITIONS.
LEDGER_MAX_L = 4

# The levers a kernel-body lever only BUILDS with (its builder refuses it
# alone, with its reason): its ledger arm is the composition. An entry whose
# lever is not in the vocabulary is inert, so a lever can be declared here
# ahead of its merge; adding a lever is one re-run of `measured --ledger`.
# test/test_lever_ledger.py builds each lever alone (refused) and composed.
REQUIRES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "sliced-hrr": ("batched-vrr",),  # #356
    "subdag-vrr": ("batched-vrr",),  # #357
    "slim-fock": ("batched-vrr", "sliced-hrr"),  # #360
    "sliced-site-digest": ("batched-vrr", "sliced-hrr"),  # #359
    "sub-slice": ("batched-vrr", "sliced-hrr", "slim-fock"),  # #361
})

# Arms that are compositions BY DECLARATION, beside each lever's own arm:
# #348's split arm of interleaved I_p is `interleave+split`, not a lever.
COMPOSITIONS: tuple[tuple[str, ...], ...] = (("interleave", "split"),)

LEDGER_TARGETS = (Target.J, Target.K, Target.F)

# Milestone 12's 21-class F ladder (`measured.F_CLASSES`, restated: the
# store names no HGP builder; test/test_lever_ledger.py holds them equal).
_F_PAIRS = tuple((a, b) for a in range(3) for b in range(a, 3))
_F_LADDER = tuple(bra + ket for i, bra in enumerate(_F_PAIRS)
                  for ket in _F_PAIRS[i:])


def _closure(lever: str) -> tuple[str, ...]:
    out, todo = set(), [lever]
    while todo:
        lv = todo.pop()
        if lv not in out:
            out.add(lv)
            todo.extend(REQUIRES.get(lv, ()))
    return tuple(sorted(out))


def ledger_levers() -> tuple[str, ...]:
    """The kernel-body levers the ledger reads, sorted: every one legal on a
    LEDGER_TARGETS slot (`j`, `k`, `f`, each on its native engine). A lever
    legal only on an engine slot (`levers.ENGINE_SLOTS`, #401) is not one."""
    slots = {t.value for t in LEDGER_TARGETS}
    return tuple(lv for lv in sorted(_levers.VOCABULARY)
                 if slots & _levers.VOCABULARY[lv].legal)


def ledger_arms() -> tuple[str, ...]:
    """Every arm the ledger reads, in key spelling: each kernel-body lever's
    own arm (`_closure` over REQUIRES, so `sliced-hrr` is
    `batched-vrr+sliced-hrr`), then the declared COMPOSITIONS whose levers
    are all in the vocabulary. Read from the vocabulary when called.

    A lever legal on no LEDGER_TARGETS slot -- one declared for an engine
    slot only, as #401's `ordered` on `j.hgp` / `k.hgp` -- has no ledger
    class to build and gets no arm (`ledger_levers`)."""
    arms = [_closure(lv) for lv in ledger_levers()]
    arms += [c for c in COMPOSITIONS
             if all(_levers.is_kernel_lever(lv) for lv in c)]
    seen: dict[tuple[str, ...], None] = {}
    for a in arms:
        if not all(_levers.is_kernel_lever(lv) for lv in a):
            raise ValueError(f"ledger arm {'+'.join(a)}: requires a lever "
                             "outside the vocabulary")
        seen.setdefault(tuple(sorted(a)), None)
    return tuple("+".join(a) for a in seen)


def lever_issue(lever: str) -> int:
    """The issue that built kernel-body lever `lever` (its vocabulary entry)."""
    return _levers.VOCABULARY[lever].issue


def declared_ahead() -> list[str]:
    """REQUIRES' levers not (yet) in the vocabulary: inert until they land."""
    return sorted(lv for lv in REQUIRES if not _levers.is_kernel_lever(lv))


def arm_target(arm: str) -> Target:
    """The ONE target `arm` is legal on: every lever's `legal` set
    intersected; anything but exactly one ledger target is refused."""
    legal = set.intersection(*(set(_levers.VOCABULARY[lv].legal)
                               for lv in arm.split("+")))
    got = [t for t in LEDGER_TARGETS if t.value in legal]
    if len(got) != 1:
        raise ValueError(f"ledger arm {arm!r} is legal on "
                         f"{sorted(legal) or 'nothing'}: the ledger needs "
                         "exactly one of j, k, f")
    return got[0]


def ledger_classes(target: Target) -> tuple[Spec, ...]:
    """The literal classes of `target` the ledger reads, total L <=
    LEDGER_MAX_L: `pq_spec(L)` (J), `k_spec(L)` (K), one batch; every class
    of the 21-class F ladder, in ladder order."""
    Ls = range(LEDGER_MAX_L + 1)
    if target is Target.J:
        return tuple(pq_spec(L) for L in Ls)
    if target is Target.K:
        return tuple(k_spec(L) for L in Ls)
    if target is Target.F:
        return tuple(Spec(Operator.COULOMB, 0, c, Target.F) for c in _F_LADDER
                     if sum(c) <= LEDGER_MAX_L)
    raise ValueError(f"the ledger reads no {target.value!r} class")


def ledger_cells() -> tuple[Spec, ...]:
    """Every cell the ledger reads a row of: each target's literal classes
    (the baseline), then every arm on each class of its target."""
    out = [c for t in LEDGER_TARGETS for c in ledger_classes(t)]
    for arm in ledger_arms():
        out += [lever_arm(c, arm) for c in ledger_classes(arm_target(arm))]
    return tuple(out)


SCHEMA = 2  # 2: a timed row carries GPU_UTIL_FIELDS, a reading or null (#93)

# THE CARD'S STATE AT TIMING (#93). Every row with a `runtime_ns` carries
# these raw keys: the card's utilisation in percent, from its own SMI, as the
# sweep's gate let the input through (`_before`) and just after its last
# timed sequence (`_after`), on each input. `None` (JSON null) means UNKNOWN
# -- not read, or not parseable -- and NEVER idle: every row timed before
# #93 records null here, whatever gate it ran behind.
GPU_UTIL_FIELDS = ("wallclock:gpu_util_before", "wallclock:gpu_util_after",
                   "wallclock:near:gpu_util_before",
                   "wallclock:near:gpu_util_after")

ABOUT = ("Measured rows (felizjueves.measured): what one device toolchain "
         "reported about one emitted translation unit on one card, keyed on "
         "device.MeasuredKey(spec, arch, toolchain).key(). Observations of a "
         "toolchain, not claims about this code. NOT comparable to [JEngine, "
         "v2, Table 4]: their kernels, their compiler, their A100.")

HEADER = {
    "what": "Measured rows: what a device toolchain reported about one "
            "emitted translation unit (felizjueves.measured).",
    "key": "Rows are keyed on felizjueves.device.MeasuredKey(spec, arch, "
           "toolchain).key() -- the generation Spec, the card's Arch, the "
           "compiler as it reports itself (#37).",
    "nature": "Observations of a toolchain, not claims about this code: a "
              "compiler bump moving a number is news about the compiler.",
    "not_compared": "Nothing here is compared against [JEngine, v2, Table 4] "
                    "(their kernels, compiler and A100); see "
                    "data/literature/jengine/README.md.",
    "units": "nvcc: registers per thread, spill BYTES (ptxas -v). hipcc: "
             "VGPRs per lane, spilled REGISTER COUNTS (VGPR + SGPR; no "
             "store/load split, so both spill fields carry it). See each "
             "row's spill_unit and raw.",
    "remeasure": "PYTHONPATH=src uv run python -m felizjueves.measured "
                 "--remeasure --backend both",
    "wallclock": "runtime_ns is the MEDIAN host wall-clock of a cell's whole "
                 "batched launch sequence (every build_pq_kernels batch into "
                 "one zeroed J, then a synchronize), over `repeats` timed "
                 "sequences after a warm-up, on a wide-T input; spread_ns is "
                 "their interquartile range; raw 'wallclock:*' has the rest, "
                 "and 'wallclock:near:*' the same on a near-T input. Taken "
                 "alone on an "
                 "uncontended box by test/test_kernel_device.py's opt-in "
                 "wall-clock sweep (FJ_WALLCLOCK_STORE). Compare only within "
                 "one arch, across n_batches -- never across vendors.",
    "gpu_util": "#93: every row with runtime_ns carries raw "
                "'wallclock:gpu_util_before' / '_after' and "
                "'wallclock:near:gpu_util_before' / '_after' (and, on a lever "
                "row timed after #93, 'wallclock:control:...' likewise): the "
                "card's utilisation in percent from its own SMI, as the "
                "sweep's gate let the input through and 1 s after its last "
                "timed sequence (one read discarded first: rocm-smi's first "
                "read after a pause averages over it, i.e. over our own run). "
                "null means UNKNOWN, never idle: the "
                ".../recycle@... rows were timed before #93 and record null "
                "(their sm_86 rows keep #55's 'wallclock:gpu_util_pct_before_"
                "max', the highest settled reading over the cell's four "
                "timed inputs). The 40 literal SUBGRID rows were re-timed "
                "behind the gate by #93, one session per arch.",
    "levers": "#55: rows keyed .../recycle@... are the recycle arm "
              "(lower.recycle, measured.lever_cells) of every SUBGRID cell, "
              "timed against a same-session literal control kept in their "
              "raw 'wallclock:control:*'; measured.lever_ab is the verdict. "
              "CONCLUSION: recycle stays opt-in. nvcc 13.0 and hipcc / ROCm "
              "7.2.4 emit IDENTICAL machine code for the two arms in every "
              "batch kernel of every cell (both scalarise the statically "
              "indexed slot array), registers and spills match exactly, and "
              "the clock is NEUTRAL at every cell on sm_86; on gfx1200 the "
              "declared rule reads 4 cells at L <= 10 as SLOWER, on that "
              "identical code -- the A/B's noise floor, not the lever.",
    "asym": "#113: rows keyed .../b<n>/j/asym@... are the asymmetric-split "
            "STUDY (felizjueves.asym; its docstring is the rule, committed "
            "before any such row existed): L in {13, 16} x n_batches in {4, "
            "5, 7, 8}, a batch list Spec and partition refuse, each timed "
            "against its two nearest legal neighbours in the same session "
            "(raw 'wallclock:control:b<n>:*'; no literal row re-timed). A "
            "phase the card was taken during was discarded and retaken (raw "
            "'wallclock:gate_discarded_phases'). CONCLUSION: every cell is "
            "WORSE than its best legal neighbour on both sm_86 and gfx1200, "
            "so jengine/symmetry-breaking-splits-are-worse is asserted.",
}


# The counter measurements (#110; module docstring, COUNTERS), and each
# field's unit -- the derived quantities' too, though they are never stored.
COUNTERS = ("fp64_flops", "dram_bytes_read", "dram_bytes_written", "counter_ns")
COUNTER_UNITS = {
    "fp64_flops": "fp64 floating-point operations executed (FMA = 2)",
    "dram_bytes_read": "bytes (vendor meaning: the row's counter_unit)",
    "dram_bytes_written": "bytes (vendor meaning: the row's counter_unit)",
    "counter_ns": "ns, the profiler's kernel duration under counter "
                  "collection -- NOT a timing",
    "arithmetic_intensity": "FLOP / byte = fp64_flops / (dram_bytes_read + "
                            "dram_bytes_written), derived",
    "achieved_flops_per_s": "FLOP / s = fp64_flops / counter_ns, derived",
}
COUNTERS_ABOUT = (
    "#110: fp64_flops, dram_bytes_read, dram_bytes_written and counter_ns are "
    "the card's own performance counters as a profiler reported them for one "
    "kernel -- ncu on NVIDIA (fp64_flops = dadd + dmul + 2 x dfma thread "
    "instructions; DRAM bytes at sector granularity), rocprofv3's counter "
    "mode on AMD (GL2C <-> EA bytes, an upper bound on DRAM; gfx1200 has no "
    "fp64 counter, so fp64_flops is None there) -- with the profiler in "
    "counter_tool, the vendor meaning in counter_unit, the invocation in "
    "command and every counter by its exact name in raw. counter_ns is the "
    "kernel's duration under counter collection (serialised, replayed): NOT "
    "a timing, never runtime_ns. The counts are canonical; "
    "arithmetic_intensity and achieved_flops_per_s are derived, never "
    "stored. Units: " + "; ".join(f"{k}: {v}" for k, v in COUNTER_UNITS.items())
    + ". Counter rows live in data/measured/roofline.json, ONE PER BATCH "
    "KERNEL, keyed MeasuredKey.key() + '#<batch>' (plan order, from 0); a "
    "cell's counters are their sum, derived (measured_store.cell_counters), "
    "never stored. NOT comparable to [JEngine, v2, Fig. 7]: their kernels, "
    "their A100.")
HEADER["counters"] = COUNTERS_ABOUT

# THE F CELL (#278; module docstring): written into the header as "f" only
# once the store holds an F row (`save`), so the tracked file is unchanged.
F_ABOUT = (
    "#278: rows keyed .../b1/f/literal@... are the fused Fock build of "
    "[Fock] §3 on the HGP recurrence (milestone 12), one row per (class, "
    "arch, toolchain): the class (la,lb|lc,ld) in AO order -- NOT the K "
    "rows' Hermite pairing -- over the 21-class ladder, every shell l <= 2, "
    "one batch, no lever. Units as above: each row's spill_unit and raw "
    "say which (spill bytes on nvcc, spilled register counts on hipcc); "
    "compare within one arch only. A timed F row carries every "
    "'wallclock:*gpu_util_*' key as a reading, never null. Written by "
    "#288's measurement (total L <= 2) and #290's (the deferred tier, L = 3 "
    "to 8): the compile side by python -m felizjueves.measured "
    "--remeasure --target f --L ..., the clock by test/test_fock_ondevice"
    ".py's opt-in sweep (FJ_WALLCLOCK_F=1, FJ_WALLCLOCK_F_L selecting the "
    "classes by total L, behind the gate; its docstring "
    "is the protocol: one off-diagonal launch of 2048 x 128 contracted "
    "quartets, runtime_ns on the wide input, wallclock:near:* on the near "
    "one); nothing here is compared "
    "against [Fock]'s A100 timings, and there is no published register "
    "table for its §3.2 kernel.")

# THE LEVER ROW (#339; module docstring): written into the header as
# "kernel_levers" only once the store holds a KERNEL row (`save`).
KERNEL_LEVERS_ABOUT = (
    "#339: a row whose arm_kind is 'kernel' is a kernel-body lever's arm "
    "(levers.VOCABULARY, #337): the cell keyed by its full lever-carrying "
    "Spec.key() (the lever in the key's last segment), built by the "
    "builder with that lever pulled -- the kernel itself differs -- and "
    "its units named <literal name>_lv_<lever> (lever_ladder.unit_name), "
    "the names python -m felizjueves.emit's lever ladder writes. Recorded "
    "through measured_store.lever_arm and merge, by python -m "
    "felizjueves.measured --remeasure --lever <name>; what the lever "
    "changes is its vocabulary entry's 'what', and its issue says why.")

# THE LEVER CLOCK (#363; module docstring): written into the header as
# "lever_clock" only once a KERNEL row carries a runtime_ns (`save`).
LEVER_CLOCK_ABOUT = (
    "#363: a KERNEL row (a ledger arm, total L <= 4) that carries runtime_ns "
    "was timed by test/test_lever_clock_device.py's opt-in sweep "
    "(FJ_WALLCLOCK_STORE + FJ_WALLCLOCK_LEVERS=1), its clock ADDED to the "
    "ledger's compile row of the same text: runtime_ns / spread_ns are the "
    "arm's median / IQR on the wide input over `repeats` timed sequences "
    "after raw 'wallclock:warmup', raw 'wallclock:near:*' the near input, "
    "and raw 'wallclock:control:*' / 'wallclock:control:near:*' the LITERAL "
    "class timed in the same gated phase (control near, arm near, arm wide, "
    "control wide); the host steady clock only, no GPU-event times in raw. "
    "Every gpu_util reading is the phase's (felizjueves.gate: "
    "wait_quiet before, after_reading after); a refused phase was discarded "
    "and retaken (raw 'wallclock:gate_discarded_phases'), never recorded. "
    "Inputs, tau = 0 (no screen fires): J 2048 x 128 (the J sweep's), K "
    "2048 x 32 (the K sweep's), F one synthetic off-diagonal launch of 2048 "
    "x 32 contracted quartets (raw 'wallclock:n_ket'). The verdict is "
    "measured_store.lever_clock_verdict (#60's IQR rule per input, per "
    "class, per arm). Compare within one arch only.")

MEASUREMENTS = ("registers", "spill_store", "spill_load", "shared_bytes",
                "compile_s", "runtime_ns", "repeats", "spread_ns", *COUNTERS)
PROVENANCE = ("toolchain", "arch", "host", "gpu", "date")
_INTS = ("registers", "spill_store", "spill_load", "shared_bytes", "repeats",
         "fp64_flops", "dram_bytes_read", "dram_bytes_written")
# Serialised only when present (`as_record`): a row without counters is
# written exactly as it was before #110.
# And so is an arm's kind (#159): a row with none is written as before it.
# And so is a target (#201): a J row records none, as before it.
_OMITTED_WHEN_ABSENT = (*COUNTERS, "counter_tool", "counter_unit", "arm_kind",
                        "target")


_TARGETS = frozenset(t.value for t in Target)


@dataclass(frozen=True, kw_only=True)
class Measured:
    """One observation of one toolchain on one card. NOT a claim about this code.

    Every measurement is optional (None: not observed by this row), but at
    least one must be present. The provenance is required.
    """

    # compile-side, from the device compiler
    registers: int | None = None
    spill_store: int | None = None
    spill_load: int | None = None
    shared_bytes: int | None = None
    compile_s: float | None = None
    # device-side (C1)
    runtime_ns: float | None = None
    repeats: int | None = None
    spread_ns: float | None = None
    # the card's performance counters, from a profiler (#110)
    fp64_flops: int | None = None
    dram_bytes_read: int | None = None
    dram_bytes_written: int | None = None
    counter_ns: float | None = None
    # provenance, all REQUIRED
    toolchain: str  # "nvcc 13.0", "hipcc / ROCm 7.2.4"
    arch: str  # "sm_86", "gfx1200"
    host: str
    gpu: str
    date: str
    # how the numbers were asked for
    kernel: str = ""
    spill_unit: str = ""
    command: str = ""  # the compiler invocation, measurement flag included
    source_sha256: str = ""  # of the emitted translation unit
    raw: Mapping[str, int | None] = field(default_factory=dict)
    counter_tool: str = ""  # the profiler as it reports itself (#110)
    counter_unit: str = ""  # what the counter fields mean on this vendor
    arm_kind: str = ""  # an arm's kind: LOWERING, PLAN (#159), KERNEL (#339)
    target: str = ""  # the key's Spec.target, required unless J (#201)

    def __post_init__(self) -> None:
        for n in MEASUREMENTS:
            v = getattr(self, n)
            if v is None:
                continue
            if n in _INTS:
                if type(v) is not int or v < 0:
                    raise ValueError(f"{n} = {v!r} must be an int >= 0 or None")
            elif type(v) not in (int, float) or v < 0:
                raise ValueError(f"{n} = {v!r} must be a number >= 0 or None")
        if self.repeats is not None and self.repeats < 1:
            raise ValueError(f"repeats = {self.repeats!r} must be >= 1")
        if all(getattr(self, n) is None for n in MEASUREMENTS):
            raise ValueError("every measurement is None: the row records nothing")
        for n in PROVENANCE:
            if not isinstance(getattr(self, n), str) or not getattr(self, n):
                raise ValueError(f"{n} must be a non-empty string")
        if self.arm_kind not in ("", LOWERING, PLAN, KERNEL):
            raise ValueError(f"arm_kind = {self.arm_kind!r} must be "
                             f"{LOWERING!r}, {PLAN!r}, {KERNEL!r} or empty")
        if self.target and self.target not in _TARGETS:
            raise ValueError(f"target = {self.target!r} must be one of "
                             f"{', '.join(sorted(_TARGETS))} or empty")

    # Derived from the stored counts, never stored themselves (#110).
    @property
    def dram_bytes(self) -> int | None:
        """Bytes read + written, or None unless both were counted."""
        if self.dram_bytes_read is None or self.dram_bytes_written is None:
            return None
        return self.dram_bytes_read + self.dram_bytes_written

    @property
    def arithmetic_intensity(self) -> float | None:
        """fp64 FLOP per DRAM byte, or None unless both sides were counted
        (and some byte moved)."""
        b = self.dram_bytes
        if self.fp64_flops is None or not b:
            return None
        return self.fp64_flops / b

    @property
    def achieved_flops_per_s(self) -> float | None:
        """fp64 FLOP/s over the kernel's duration under counter collection,
        or None unless both were counted."""
        if self.fp64_flops is None or not self.counter_ns:
            return None
        return self.fp64_flops * 1e9 / self.counter_ns


def as_record(row: Measured) -> dict:
    """`row` as the store writes it: every field, except that the counter
    fields, `arm_kind` and `target` are left out when absent -- so a row with
    no counters is written exactly as it was before #110, one with no arm
    kind as it was before #159, and a J row as it was before #201."""
    d = asdict(row)
    for n in _OMITTED_WHEN_ABSENT:
        if d[n] in (None, ""):
            del d[n]
    return d



# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


_FIELDS = frozenset(f.name for f in fields(Measured))


def split_key(key: str) -> tuple[str, str, str]:
    """`MeasuredKey.key()` back into (spec key, arch name, toolchain)."""
    spec_key, sep, rest = key.partition("@")
    arch, sep2, toolchain = rest.partition("/")
    if not (sep and sep2 and spec_key and arch and toolchain):
        raise ValueError(f"{key!r} is not a MeasuredKey.key()")
    return spec_key, arch, toolchain


def _row(path: Path, key: str, v: object) -> Measured:
    if not isinstance(v, dict):
        raise ValueError(f"{path}: row {key!r} is not an object")
    unknown = sorted(set(v) - _FIELDS)
    if unknown:
        raise ValueError(f"{path}: row {key!r} has unknown fields {unknown}")
    try:
        row = Measured(**v)
        _, arch, toolchain = split_key(key)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path}: row {key!r}: {e}") from e
    if (arch, toolchain) != (row.arch, row.toolchain):
        raise ValueError(f"{path}: row {key!r} records arch/toolchain "
                         f"{row.arch!r}/{row.toolchain!r}, not its key's")
    try:
        _check_engine(key)
        _check_arm_kind(key, row)
        _check_target(key, row)
        _check_f(key, row)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e
    if row.runtime_ns is not None:
        for f in GPU_UTIL_FIELDS:
            if f not in row.raw:
                raise ValueError(f"{path}: timed row {key!r} has no {f!r} -- "
                                 "a reading, or null for unknown")
            v = row.raw[f]
            if v is not None and (type(v) is not int or not 0 <= v <= 100):
                raise ValueError(f"{path}: row {key!r}: {f} = {v!r} is not a "
                                 "percentage or null")
    return row


def _key_arms(key: str) -> tuple[str, ...]:
    """The declared arm levers a row key's Spec carries, read off its last
    segment (`Spec.key()`'s levers): `ARMS`' and the kernel-body
    vocabulary's (#339). Anything undeclared there -- "literal", #113's
    "asym" -- is not an arm and is not this check's business."""
    spec_key, _, _ = split_key(key)
    return tuple(lv for lv in spec_key.rsplit("/", 1)[-1].split("+")
                 if lv in ARMS or _levers.is_kernel_lever(lv))


def _check_arm_kind(key: str, row: Measured) -> None:
    """THE KIND IS ON THE ROW (module docstring, THE ARMS)."""
    arms = _key_arms(key)
    if is_composition(arms):  # #362: one KERNEL arm
        arms = ("+".join(arms),)
    if len(arms) > 1:
        raise ValueError(f"row {key!r} carries {len(arms)} arms at once "
                         f"({', '.join(arms)})")
    kind = kind_of(arms[0]) if arms else ""
    if row.arm_kind and row.arm_kind != kind:
        raise ValueError(f"row {key!r} records arm_kind {row.arm_kind!r}, "
                         f"but its key's arm kind is {kind or 'none'!r}")
    if kind in (PLAN, KERNEL) and not row.arm_kind:
        raise ValueError(f"row {key!r} is a {kind} arm and records no "
                         "arm_kind")


def _slot_of(key: str) -> tuple[Target, Engine]:
    """`(target, engine)` read off a row key's target slot (the fifth segment
    of `Spec.key()`, `.../b<n>/<target>[.<engine>]/<levers>`). A slot with no
    `.<engine>` suffix names the target's native engine (#401)."""
    spec_key, _, _ = split_key(key)
    parts = spec_key.split("/")
    try:
        slot = parts[4]
        name, dot, eng = slot.partition(".")
        target = Target(name)
        if not dot:
            return target, NATIVE_ENGINE[target]
        engine = Engine(eng)
    except (IndexError, ValueError) as e:
        raise ValueError(f"{key!r} names no target") from e
    if engine is NATIVE_ENGINE[target]:
        raise ValueError(f"{key!r} spells {target.value}'s native engine "
                         f"{engine.value}, which a Spec key never does")
    return target, engine


def target_of(key: str) -> Target:
    """The `Spec.target` a row key names, read off its target slot (the fifth
    segment of `Spec.key()`, `.../b<n>/<target>/<levers>`), past any
    `.<engine>` suffix (#401)."""
    return _slot_of(key)[0]


def engine_of(key: str) -> Engine:
    """The `Spec.engine` a row key names: the suffix of its target slot, or
    the target's native engine when there is none (#401)."""
    return _slot_of(key)[1]


# THE HGP SPLIT J / K IS NOT ADMITTED (#407's decision). A key whose target
# slot carries a non-native engine (`j.hgp`, `k.hgp`; #401) is refused on
# `merge` and on `load`, with this reason, until a later issue admits it.
HGP_JK_REFUSED = (
    "no HGP split J / K row is admitted yet (#407): milestone 14 emits and "
    "checks the hj_ / hk_ kernels and times nothing, so a registers-and-spills "
    "row would have no clock beside it and no declared cell to belong to; "
    "the issue that measures them admits the row, its cell set and its "
    "header together, as #278 did for F")


def _check_engine(key: str) -> None:
    """THE HGP SPLIT J / K IS NOT ADMITTED (`HGP_JK_REFUSED`): a key on a
    non-native engine is refused. Every admitted target is on its native
    engine (`f` is HGP's own)."""
    target, engine = _slot_of(key)
    if engine is not NATIVE_ENGINE[target]:
        raise ValueError(f"row {key!r}: {HGP_JK_REFUSED}")


def _check_target(key: str, row: Measured) -> None:
    """THE TARGET IS ON THE ROW (module docstring, THE K CELLS)."""
    want = target_of(key)
    if row.target and row.target != want:
        raise ValueError(f"row {key!r} records target {row.target!r}, but "
                         f"its key's target is {want.value!r}")
    if want is not Target.J and not row.target:
        raise ValueError(f"row {key!r} is a {want.value!r} row and records "
                         "no target")


_F_KEY = re.compile(
    r"coulomb/d0/\((\d+),(\d+)\|(\d+),(\d+)\)/b(\d+)/f/(.+)\Z")


def _check_f(key: str, row: Measured) -> None:
    """THE F CELL (module docstring): an F key names a legal one-batch
    `Target.F` spec, lever-free or carrying kernel-body levers only (#339,
    each legal on F by its own table), and a timed F row carries every gate
    reading as a reading. A key of any other target is not this check's
    business."""
    if target_of(key) is not Target.F:
        return
    spec_key, _, _ = split_key(key)
    m = _F_KEY.fullmatch(spec_key)
    if not m:
        raise ValueError(f"row {key!r} is an 'f' row whose key is not "
                         "coulomb/d0/(la,lb|lc,ld)/b<n>/f/<levers>")
    *am, n, levers = m.groups()
    lv = () if levers == "literal" else tuple(levers.split("+"))
    if not all(_levers.is_kernel_lever(x) for x in lv):
        raise ValueError(f"row {key!r}: no lever is declared for an F cell "
                         "but a kernel-body one (#339; the key carries "
                         f"{levers!r})")
    try:
        spec = Spec(Operator.COULOMB, 0, tuple(map(int, am)), Target.F,
                    lv, n_batches=int(n))
    except ValueError as e:
        raise ValueError(f"row {key!r} names no legal F spec: {e}") from e
    if spec.key() != spec_key:
        raise ValueError(f"row {key!r} does not spell its F spec's key "
                         f"{spec.key()!r}")
    if row.runtime_ns is not None:
        for f in GPU_UTIL_FIELDS:
            if row.raw.get(f) is None:
                raise ValueError(
                    f"timed F row {key!r} has no {f!r} reading -- an F row "
                    "postdates the gate, so null (unknown) is refused: an "
                    "ungated runtime_ns is absent, never unlabelled")


def load(path: Path = STORE) -> dict[str, Measured]:
    """Every row in the store, by key. Refuses a malformed or foreign file.

    A missing file raises FileNotFoundError: an empty store is a valid file
    with no rows, and is a different thing from no store at all.
    """
    path = Path(path)
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not JSON ({e})") from e
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: top level is not an object")
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected {SCHEMA}")
    if not isinstance(doc.get("about"), str) or not doc["about"]:
        raise ValueError(f"{path}: no \"about\" string")
    got = doc.get("rows")
    if not isinstance(got, dict):
        raise ValueError(f"{path}: \"rows\" is not an object")
    return {k: _row(path, k, v) for k, v in got.items()}


def merge(store: Mapping[str, Measured], key: MeasuredKey,
          row: Measured) -> dict[str, Measured]:
    """`store` with `row` at `key`, REPLACING any row already there. `key`
    is a `MeasuredKey`, or anything keyed its way (`arch`, `toolchain`,
    `key()`): #113's `asym.AsymKey`, whose cell is not a Spec. A
    `MeasuredKey` on an A/B arm stamps the arm's kind on the row (#159), and
    a key of any target but J stamps the target (#201)."""
    if (key.arch.name, key.toolchain) != (row.arch, row.toolchain):
        raise ValueError(f"row is {row.arch}/{row.toolchain}, key {key.key()!r}")
    _check_engine(key.key())
    if isinstance(key, MeasuredKey):
        kind = arm_kind(key.spec)  # refuses an undeclared lever or two arms
        if kind is not None and not row.arm_kind:
            row = replace(row, arm_kind=kind)
    target = target_of(key.key())
    if target is not Target.J and not row.target:
        row = replace(row, target=target.value)
    _check_arm_kind(key.key(), row)
    _check_target(key.key(), row)
    _check_f(key.key(), row)
    return {**store, key.key(): row}


def combine(old: Measured | None, new: Measured) -> Measured:
    """`new`, with `old`'s other-side measurements kept when both saw one text.

    A field `new` leaves None is taken from `old`, and `old`'s `raw` entries
    survive under `new`'s, only when the two rows record the same arch,
    toolchain and non-empty `source_sha256`; otherwise `new` alone (module
    docstring, ONE ROW CARRIES BOTH SIDES).
    """
    if (old is None or not new.source_sha256
            or (old.arch, old.toolchain, old.source_sha256)
            != (new.arch, new.toolchain, new.source_sha256)):
        return new
    kept = {n: getattr(old, n) for n in MEASUREMENTS
            if getattr(new, n) is None and getattr(old, n) is not None}
    return replace(new, **kept, raw={**old.raw, **new.raw},
                   kernel=new.kernel or old.kernel,
                   spill_unit=new.spill_unit or old.spill_unit,
                   command=new.command or old.command)


def rows(spec: Spec | None = None, arch: Arch | str | None = None,
         toolchain: str | None = None, *,
         store: Mapping[str, Measured] | Path = STORE,
         target: Target | None = None) -> dict[str, Measured]:
    """The read path: the rows matching every filter given, by key.

    `store` is an already-loaded mapping, or a path to load (by default the
    tracked store). `target` keeps the rows whose key names that
    `Spec.target` (#201).
    """
    got = load(store) if isinstance(store, Path) else store
    arch_name = arch.name if isinstance(arch, Arch) else arch
    out = {}
    for k, r in got.items():
        s, a, t = split_key(k)
        if spec is not None and s != spec.key():
            continue
        if arch_name is not None and a != arch_name:
            continue
        if toolchain is not None and t != toolchain:
            continue
        if target is not None and target_of(k) is not target:
            continue
        out[k] = r
    return out


def save(store: Mapping[str, Measured], path: Path = STORE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # #278: the header's "f" entry is written only once the store holds an
    # F row, so a store without one is rewritten byte for byte.
    header = (HEADER if not any(target_of(k) is Target.F for k in store)
              else {**HEADER, "f": F_ABOUT})
    # #339: likewise "kernel_levers", once the store holds a KERNEL row
    if any(r.arm_kind == KERNEL for r in store.values()):
        header = {**header, "kernel_levers": KERNEL_LEVERS_ABOUT}
    # #363: likewise "lever_clock", once a KERNEL row carries a clock
    if any(r.arm_kind == KERNEL and r.runtime_ns is not None
           for r in store.values()):
        header = {**header, "lever_clock": LEVER_CLOCK_ABOUT}
    doc = {"schema": SCHEMA, "about": ABOUT, "header": header,
           "rows": {k: as_record(store[k]) for k in sorted(store, key=_order)}}
    path.write_text(json.dumps(doc, indent=2) + "\n")


def _order(key: str) -> tuple:
    m = re.search(r"\((\d+),(\d+)\|(\d+),(\d+)\)", key)
    am = tuple(int(x) for x in m.groups()) if m else ()
    return (key.split("@", 1)[-1], sum(am), am, key)


# ---------------------------------------------------------------------------
# #110: the counter store, one row per batch kernel (module docstring)
# ---------------------------------------------------------------------------

ROOFLINE_SCHEMA = 1

ROOFLINE_ABOUT = (
    "Counter rows (felizjueves.measured_store, #110): the card's own "
    "performance counters, as a profiler reported them for ONE BATCH KERNEL "
    "of one cell, keyed device.MeasuredKey(spec, arch, toolchain).key() + "
    "'#<batch>', the kernel's 0-based position in the cell's plan order. A "
    "cell's counters are derived (summed over its kernels), never stored. "
    "Observations of a profiler on one card, not claims about this code. NOT "
    "comparable to [JEngine, v2, Fig. 7]: their kernels, their A100.")

ROOFLINE_HEADER = {
    "what": "One row per batch kernel: what a profiler's counters said about "
            "one launch of one emitted kernel on one card "
            "(felizjueves.measured_store, #110).",
    "key": "MeasuredKey(spec, arch, toolchain).key() + '#<batch>': the "
           "generation Spec, the card's Arch, the COMPILER that built the "
           "kernel, and the kernel's 0-based index in the cell's plan order "
           "(measured.cell_kernels). A cell of n_batches b has exactly the "
           "rows #0 .. #b-1; the loader refuses any other set.",
    "cell": "A cell's counters are measured_store.cell_counters: every count "
            "summed over its kernels (each launched once per class). Derived, "
            "never stored, so the per-kernel and per-cell verdicts read one "
            "set of numbers.",
    "counters": COUNTERS_ABOUT,
    "not_timing": "counter_ns is the kernel's duration under counter "
                  "collection, which serialises and replays kernels: never a "
                  "timing. Wall-clock lives in kernels.json's runtime_ns, "
                  "and a row here carries none.",
    "not_compared": "Nothing here is compared against [JEngine, v2, Fig. 7] "
                    "(their kernels, their A100), nor across vendors (the "
                    "fields share a name, not a meaning: see counter_unit).",
}

# #112: the card's state around a counter phase, in every counter row's raw.
# gpu_util_*: percent, the card's own SMI (gate.card_util); both KNOWN (the
# sweep retakes a phase whose after-reading is non-idle or unknown).
# cpu_busy_permille_before: the CPU gate's settled reading. gate_discarded_
# phases: the cell's phases the gate discarded before this one was kept.
ROOFLINE_STATE_FIELDS = ("roofline:gpu_util_before", "roofline:gpu_util_after",
                         "roofline:cpu_busy_permille_before",
                         "roofline:gate_discarded_phases")

# #140: the fp64 instruction counters a roofline row needs and a card in
# UNMEASURED does not expose, and the architectures rocprofiler-sdk 1.1.0
# (ROCm 7.2.4, share/rocprofiler-sdk/counter_defs.yaml) DOES define them for
# -- CDNA only; the derived TOTAL_64_OPS and SQ_INSTS_VALU_FLOPS_FP64 are
# defined for that set too (gfx950 alone for the latter). The refusal's
# mechanism, as data: measure/measure_toolchain.py reads the card's own
# `rocprofv3 --list-avail` and fails if one of these, or any fp64 / FLOP
# counter, appears on it.
MISSING_COUNTERS = {
    "gfx1200": ("SQ_INSTS_VALU_ADD_F64", "SQ_INSTS_VALU_MUL_F64",
                "SQ_INSTS_VALU_FMA_F64"),
}
FP64_COUNTER_ARCHS = ("gfx90a", "gfx940", "gfx941", "gfx942", "gfx950")

# #112, made FINAL by #140: the cards that have NO counter rows, by decision,
# and why. The absence is recorded here -- never as a zero, never as an absent
# file -- and it is a refusal with its mechanism, not a pending measurement.
UNMEASURED = {
    "gfx1200": (
        "FINAL (#140): no roofline row on gfx1200, because the card exposes "
        "no fp64 FLOP counter. rocprofv3 1.1.0 (ROCm 7.2.4) now LOADS -- "
        "libdw.so.1 and libhsa-amd-aqlprofile64.so.1 are present and the "
        "AMD counter witness in test/test_toolchain.py dispatches -- but of "
        "the counters rocprofiler-sdk defines for gfx1200 none counts fp64 "
        "operations: " + ", ".join(MISSING_COUNTERS["gfx1200"])
        + " (and the derived TOTAL_64_OPS, SQ_INSTS_VALU_FLOPS_FP64) are "
        "defined only for " + ", ".join(FP64_COUNTER_ARCHS)
        + ", and SQ_INSTS_VALU counts every VALU instruction whatever its "
        "type. With no measured FLOP count there is no measured intensity "
        "and no roofline point, and a FLOP count from the model or the "
        "interpreter would be a model, not a counter. Nor are there bytes: "
        "on this box (amdgpu module 6.16.13, host Linux 6.8) the GL2C <-> EA "
        "byte "
        "counters all read 0 for a probe that streams 384 MiB -- as do "
        "TCP_REQ, TA_TA_BUSY and SQ_INSTS_VALU, while GRBM_GUI_ACTIVE and "
        "SQ_WAVES count -- so the driver refuses that reading rather than "
        "file it. No gfx1200 row exists; that is not a zero, and rebuilding "
        "the image does not change it -- a rocprofiler-sdk defining an fp64 "
        "counter for gfx12 would."),
}

ROOFLINE_HEADER["card_state"] = (
    "#112: every row's raw carries " + ", ".join(ROOFLINE_STATE_FIELDS)
    + ": the card's utilisation (percent, its own SMI) as the gate let the "
    "counter phase through and 1 s after it, the CPU's busy fraction "
    "(permille) before it, and how many of that cell's phases the gate "
    "discarded (after-reading non-idle or unknown) before this one was "
    "kept. The counter run is a SEPARATE run of the same checked binary as "
    "kernels.json's timed one, on the same wide-T input (raw 'roofline:"
    "n_bra' / 'roofline:n_ket' / 'roofline:lanes'); collection serialises "
    "and replays each kernel, so it may change occupancy, and its duration "
    "is counter_ns, never runtime_ns.")
ROOFLINE_HEADER["verdict"] = (
    "#112: never stored. measured_store.roofline_point derives intensity "
    "(fp64_flops / DRAM bytes) and achieved FLOP/s (fp64_flops / "
    "counter_ns) from a row and asks device.verdict against the card's "
    "MEASURED ridge (data/measured/ceilings.json, #109): memory_bound iff "
    "intensity < ridge. kernel_roofline is per batch kernel, cell_roofline "
    "per cell (counts summed). A card without both ceilings has no ridge, "
    "no verdict and no row.")
ROOFLINE_HEADER["could_not_measure"] = dict(UNMEASURED)

_COUNTER_ONLY_FORBIDDEN = tuple(n for n in MEASUREMENTS if n not in COUNTERS)
_BATCHES = re.compile(r"/b(\d+)/")


def kernel_key(key: MeasuredKey | str, batch: int) -> str:
    """A counter row's key: the cell's `MeasuredKey.key()` + '#<batch>'."""
    cell = key.key() if isinstance(key, MeasuredKey) else key
    if type(batch) is not int or batch < 0:
        raise ValueError(f"batch {batch!r} must be an int >= 0")
    split_key(cell)
    return f"{cell}#{batch}"


def split_kernel_key(key: str) -> tuple[str, int]:
    """A counter row's key back into (the cell's MeasuredKey.key(), batch)."""
    cell, sep, batch = key.rpartition("#")
    if not sep or not re.fullmatch(r"0|[1-9]\d*", batch):
        raise ValueError(f"{key!r} is not a kernel_key (no '#<batch>')")
    split_key(cell)
    return cell, int(batch)


def n_batches_of(cell: str) -> int:
    """The `n_batches` a cell key's Spec names."""
    m = _BATCHES.search(split_key(cell)[0])
    if not m:
        raise ValueError(f"{cell!r} names no n_batches")
    return int(m[1])


def _counter_row(path: Path, key: str, v: object) -> Measured:
    try:
        cell, _ = split_kernel_key(key)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e
    if isinstance(v, dict):
        unknown = sorted(set(v) - _FIELDS)
        if unknown:
            raise ValueError(f"{path}: row {key!r} has unknown fields {unknown}")
    row = _row(path, cell, v)
    other = [n for n in _COUNTER_ONLY_FORBIDDEN if getattr(row, n) is not None]
    if other:
        raise ValueError(f"{path}: counter row {key!r} records {other}: a "
                         "counter row carries counters only (a timing is "
                         "kernels.json's runtime_ns; counter_ns is not one)")
    if all(getattr(row, n) is None for n in COUNTERS):
        raise ValueError(f"{path}: counter row {key!r} records no counter")
    for n in ("counter_tool", "counter_unit", "command"):
        if not getattr(row, n):
            raise ValueError(f"{path}: counter row {key!r} has no {n}")
    if row.arch in UNMEASURED:
        raise ValueError(f"{path}: counter row {key!r} is on {row.arch}, "
                         "which has no counter rows by decision "
                         "(UNMEASURED: could not measure)")
    try:
        require_characterised(by_name(row.arch))
    except ValueError as e:
        raise ValueError(f"{path}: counter row {key!r}: {e} -- a row for an "
                         "uncharacterised card does not exist") from e
    for f in ROOFLINE_STATE_FIELDS:
        v = row.raw.get(f)
        if type(v) is not int or v < 0 or (
                "gpu_util" in f and v > 100) or ("permille" in f and v > 1000):
            raise ValueError(f"{path}: counter row {key!r}: {f} = {v!r} -- "
                             "the card's state is a KNOWN reading in every "
                             "counter row")
    return row


def _check_cells(path: Path, got: Mapping[str, Measured]) -> None:
    cells: dict[str, dict[int, Measured]] = {}
    for k, r in got.items():
        cell, b = split_kernel_key(k)
        cells.setdefault(cell, {})[b] = r
    for cell, ks in cells.items():
        want = set(range(n_batches_of(cell)))
        if set(ks) != want:
            raise ValueError(f"{path}: cell {cell!r} has kernel rows "
                             f"{sorted(ks)}, not exactly {sorted(want)}")
        first = ks[0]
        for b, r in ks.items():
            same = ("counter_tool", "counter_unit", "gpu", "host")
            bad = [n for n in same if getattr(r, n) != getattr(first, n)]
            present = {n for n in COUNTERS if getattr(r, n) is not None}
            if present != {n for n in COUNTERS
                           if getattr(first, n) is not None}:
                bad.append("the counters present")
            if bad:
                raise ValueError(f"{path}: cell {cell!r}: kernel #{b} differs "
                                 f"from #0 in {bad}")


def load_counters(path: Path = ROOFLINE) -> dict[str, Measured]:
    """Every counter row, by `kernel_key`. Refuses a malformed or foreign
    file, a non-counter measurement, and an incomplete or mixed cell.

    A missing file raises FileNotFoundError, as `load` does.
    """
    path = Path(path)
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not JSON ({e})") from e
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: top level is not an object")
    if doc.get("schema") != ROOFLINE_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected "
                         f"{ROOFLINE_SCHEMA}")
    if not isinstance(doc.get("about"), str) or not doc["about"]:
        raise ValueError(f"{path}: no \"about\" string")
    got = doc.get("rows")
    if not isinstance(got, dict):
        raise ValueError(f"{path}: \"rows\" is not an object")
    out = {k: _counter_row(path, k, v) for k, v in got.items()}
    _check_cells(path, out)
    return out


def save_counters(store: Mapping[str, Measured], path: Path = ROOFLINE) -> None:
    """Write the counter store, refusing what `load_counters` would refuse."""
    path = Path(path)
    tmp = {k: _counter_row(path, k, as_record(r)) for k, r in store.items()}
    _check_cells(path, tmp)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": ROOFLINE_SCHEMA, "about": ROOFLINE_ABOUT,
           "header": ROOFLINE_HEADER,
           "rows": {k: as_record(store[k])
                    for k in sorted(store, key=_kernel_order)}}
    path.write_text(json.dumps(doc, indent=2) + "\n")


def _kernel_order(key: str) -> tuple:
    cell, b = split_kernel_key(key)
    return (*_order(cell), b)


def kernel_counters(store: Mapping[str, Measured],
                    cell: MeasuredKey | str) -> tuple[Measured, ...]:
    """One cell's counter rows, in plan order (#0 first). KeyError if absent."""
    c = cell.key() if isinstance(cell, MeasuredKey) else cell
    n = n_batches_of(c)
    try:
        return tuple(store[kernel_key(c, b)] for b in range(n))
    except KeyError as e:
        raise KeyError(f"{c}: no counter row {e}") from None


def cell_counters(store: Mapping[str, Measured],
                  cell: MeasuredKey | str) -> Measured:
    """A cell's counters, DERIVED from its kernel rows: each count summed
    over the batch kernels (None where the kernels did not count it); the
    provenance is kernel #0's, the date the latest; `raw` is every kernel's,
    keyed '#<batch>:<counter>'."""
    ks = kernel_counters(store, cell)
    first = ks[0]
    sums = {n: (None if any(getattr(r, n) is None for r in ks)
                else sum(getattr(r, n) for r in ks)) for n in COUNTERS}
    return replace(
        first, **sums, date=max(r.date for r in ks),
        kernel=",".join(r.kernel for r in ks), source_sha256="",
        raw={f"#{b}:{k}": v for b, r in enumerate(ks) for k, v in r.raw.items()})


@dataclass(frozen=True)
class RooflinePoint:
    """One kernel's (or one cell's) place under its card's measured roofs.
    Derived from counter rows and the card's ceilings, never stored."""

    key: str
    arch: str
    intensity: float  # fp64 FLOP / DRAM byte
    achieved_flops_per_s: float  # fp64 FLOP / counter_ns
    ridge: float  # the card's measured ridge, FLOP / byte (#109)
    peak_flops_per_s: float  # the card's measured fp64 ceiling (#109)
    memory_bound: bool  # intensity < ridge, by device.verdict

    @property
    def of_peak(self) -> float:
        """Achieved FLOP/s over the card's measured fp64 ceiling."""
        return self.achieved_flops_per_s / self.peak_flops_per_s


def roofline_point(row: Measured, key: str = "") -> RooflinePoint:
    """`row`'s verdict against its card's measured ridge. Refuses
    (`device.Uncharacterised`) a card without both ceilings, and a row
    without a measured intensity or rate (ValueError): no FLOP count, no
    point -- never a zero."""
    arch = require_characterised(by_name(row.arch))
    intensity, rate = row.arithmetic_intensity, row.achieved_flops_per_s
    if intensity is None or rate is None:
        raise ValueError(f"{key or row.kernel}: no measured intensity or "
                         "FLOP rate (fp64_flops, DRAM bytes or counter_ns "
                         "uncounted): no roofline point")
    return RooflinePoint(
        key=key, arch=arch.name, intensity=intensity,
        achieved_flops_per_s=rate, ridge=arch.ridge,
        peak_flops_per_s=arch.fp64_flops_per_s,
        memory_bound=verdict(arch, intensity) == MEMORY_BOUND)


def kernel_roofline(store: Mapping[str, Measured],
                    cell: MeasuredKey | str) -> tuple[RooflinePoint, ...]:
    """The verdict per batch kernel of `cell`, in plan order."""
    c = cell.key() if isinstance(cell, MeasuredKey) else cell
    return tuple(roofline_point(r, kernel_key(c, b))
                 for b, r in enumerate(kernel_counters(store, c)))


def cell_roofline(store: Mapping[str, Measured],
                  cell: MeasuredKey | str) -> RooflinePoint:
    """The cell's verdict, off `cell_counters` (every count summed)."""
    c = cell.key() if isinstance(cell, MeasuredKey) else cell
    return roofline_point(cell_counters(store, c), c)


# ---------------------------------------------------------------------------
# #60: do 9 batches suffice? (the definition is in the module docstring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwelveVsNine:
    """What the 12-batch cell does against the 9-batch one, at one (arch, L)."""

    arch: str
    L: int
    spill: tuple[int, int]  # (spill(r9), spill(r12))
    wide: tuple[float, float, float]  # (median r9, median r12, max IQR)
    near: tuple[float, float, float]

    @staticmethod
    def _faster(t: tuple[float, float, float]) -> bool:
        return t[0] - t[1] > t[2]

    @property
    def fewer_spills(self) -> bool:
        return self.spill[1] < self.spill[0]

    @property
    def faster_wide(self) -> bool:
        return self._faster(self.wide)

    @property
    def faster_near(self) -> bool:
        return self._faster(self.near)

    @property
    def nine_suffices(self) -> bool:
        return not (self.fewer_spills or self.faster_wide or self.faster_near)


def nine_vs_twelve(store: Mapping[str, Measured], arch: str,
                   L: int) -> TwelveVsNine:
    """#60's comparison at one (arch, L), from the store's rows.

    Raises KeyError when either cell is unrecorded, uncompiled or untimed
    on `arch`.
    """
    def one(b: int) -> Measured:
        got = rows(spec=pq_spec(L, n_batches=b), arch=arch, store=store)
        if len(got) != 1:
            raise KeyError(f"L={L}, n_batches={b} @ {arch}: {len(got)} rows")
        (r,) = got.values()
        if r.runtime_ns is None or r.spread_ns is None or r.spill_store is None \
                or r.spill_load is None:
            raise KeyError(f"L={L}, n_batches={b} @ {arch}: not both compiled "
                           "and timed")
        return r

    def near(r: Measured) -> tuple[float, float]:
        return (r.raw["wallclock:near:median_ns"],
                r.raw["wallclock:near:q3_ns"] - r.raw["wallclock:near:q1_ns"])

    r9, r12 = one(9), one(12)
    n9, n12 = near(r9), near(r12)
    return TwelveVsNine(
        arch=arch, L=L,
        spill=(r9.spill_store + r9.spill_load, r12.spill_store + r12.spill_load),
        wide=(r9.runtime_ns, r12.runtime_ns, max(r9.spread_ns, r12.spread_ns)),
        near=(n9[0], n12[0], max(n9[1], n12[1])))


# ---------------------------------------------------------------------------
# #55: a kernel-layer lever against the literal kernel (module docstring)
# ---------------------------------------------------------------------------

FASTER, SLOWER, NEUTRAL = "faster", "slower", "neutral"
WIN, LOSS, MIXED = "win", "loss", "mixed"


def _sign(control: tuple[float, float], lever: tuple[float, float]) -> str:
    """FASTER / SLOWER / NEUTRAL for (median, IQR) pairs: #60's rule, both ways."""
    margin = max(control[1], lever[1])
    if control[0] - lever[0] > margin:
        return FASTER
    if lever[0] - control[0] > margin:
        return SLOWER
    return NEUTRAL


@dataclass(frozen=True)
class LeverAB:
    """One lever cell against its same-session literal control, on one arch."""

    arch: str
    L: int
    n_batches: int
    lever: str
    spill: tuple[int, int]  # (literal row, lever row): reported, not judged
    wide: tuple[float, float, float, float]  # control median, IQR; lever median, IQR
    near: tuple[float, float, float, float]

    @property
    def kind(self) -> str:
        """The arm's kind, `kind_of(lever)` (#159, #339). A property, not a
        field, so #55's recorded `LeverAB`s compare and print exactly as
        before."""
        return kind_of(self.lever)

    @property
    def wide_sign(self) -> str:
        return _sign(self.wide[:2], self.wide[2:])

    @property
    def near_sign(self) -> str:
        return _sign(self.near[:2], self.near[2:])

    @property
    def verdict(self) -> str:
        signs = {self.wide_sign, self.near_sign}
        if FASTER in signs and SLOWER in signs:
            return MIXED
        if FASTER in signs:
            return WIN
        if SLOWER in signs:
            return LOSS
        return NEUTRAL


def lever_ab(store: Mapping[str, Measured], arch: str, spec: Spec) -> LeverAB:
    """#55's comparison for one lever cell `spec` on `arch`, from the store.

    Raises KeyError when the lever row is missing, uncompiled or untimed, or
    carries no same-session control, or the literal row is missing; and
    ValueError when `spec` is not ONE declared arm (`arm_of`).
    """
    lever = arm_of(spec)  # refuses an undeclared lever, and two arms at once
    if lever is None:
        raise ValueError(f"{spec.key()}: a literal cell is a control, not an "
                         "arm")
    literal = replace(spec, levers=())

    def one(s: Spec) -> Measured:
        got = rows(spec=s, arch=arch, store=store)
        if len(got) != 1:
            raise KeyError(f"{s.key()} @ {arch}: {len(got)} rows")
        (r,) = got.values()
        return r

    r, lit = one(spec), one(literal)
    if r.runtime_ns is None or r.spread_ns is None or r.spill_store is None \
            or "wallclock:control:median_ns" not in r.raw:
        raise KeyError(f"{spec.key()} @ {arch}: not compiled and timed "
                       "against a same-session control")
    if lit.spill_store is None or lit.spill_load is None or r.spill_load is None:
        raise KeyError(f"{literal.key()} @ {arch}: not compiled")

    def stats(prefix: str) -> tuple[float, float]:
        return (r.raw[f"{prefix}median_ns"],
                r.raw[f"{prefix}q3_ns"] - r.raw[f"{prefix}q1_ns"])

    return LeverAB(
        arch=arch, L=spec.total_am, n_batches=spec.n_batches, lever=lever,
        spill=(lit.spill_store + lit.spill_load, r.spill_store + r.spill_load),
        wide=(*stats("wallclock:control:"), r.runtime_ns, r.spread_ns),
        near=(*stats("wallclock:control:near:"), *stats("wallclock:near:")))


# #159: the sieve arm's verdict, declared before any no-sieve row (module
# docstring, THE SIEVE ARM). #162 records the rows, over SIEVE_CELLS.
SUPPORTED, CONTRADICTED, NOT_SHOWN = "supported", "contradicted", "not shown"
SIEVE_CELLS: tuple[Spec, ...] = ab_cells(NO_SIEVE)


def sieve_ab(store: Mapping[str, Measured], arch: str,
             cells: tuple[Spec, ...] | None = None) -> dict[Spec, LeverAB]:
    """`lever_ab` of every `no-sieve` cell on `arch` (by default
    `SIEVE_CELLS`: the subgrid's and #162's L = 11); a cell that is not a
    `no-sieve` arm is refused.
    Raises KeyError, as `lever_ab` does, for a cell not yet recorded."""
    cells = SIEVE_CELLS if cells is None else cells
    for c in cells:
        if arm_of(c) != NO_SIEVE:
            raise ValueError(f"{c.key()}: not a {NO_SIEVE} arm")
    return {c: lever_ab(store, arch, c) for c in cells}


def sieve_verdict(store: Mapping[str, Measured], arch: str,
                  cells: tuple[Spec, ...] | None = None) -> str:
    """The sieve on `arch` (THE SIEVE ARM, PER ARCH): SUPPORTED,
    CONTRADICTED or NOT_SHOWN, read off `sieve_ab`'s per-cell verdicts of
    `no-sieve` against the sieved literal."""
    got = {ab.verdict for ab in sieve_ab(store, arch, cells).values()}
    if MIXED not in got and LOSS in got and WIN not in got:
        return SUPPORTED
    if MIXED not in got and WIN in got and LOSS not in got:
        return CONTRADICTED
    return NOT_SHOWN


def sieve_may_leave_the_default(store: Mapping[str, Measured],
                                cells: tuple[Spec, ...] | None = None,
                                arches: tuple[str, ...] = ("sm_86", "gfx1200"),
                                ) -> bool:
    """THE SIEVE ARM, THE DEFAULT: True iff `no-sieve` WINS at every cell on
    every arch in `arches` (both cards by default)."""
    return all(ab.verdict == WIN for a in arches
               for ab in sieve_ab(store, a, cells).values())


# #163: the tuned arm's verdict, declared before any tuned row (module
# docstring, THE TUNED ARM), over TUNED_CELLS; NOT_MEASURABLE[TUNED] names
# the four 12-batch cells beside it.
TUNED_CELLS: tuple[Spec, ...] = ab_cells(TUNED)


def tuned_ab(store: Mapping[str, Measured], arch: str,
             cells: tuple[Spec, ...] | None = None) -> dict[Spec, LeverAB]:
    """`lever_ab` of every `tuned` cell on `arch` (by default `TUNED_CELLS`);
    a cell that is not a `tuned` arm, or is not measurable, is refused.
    Raises KeyError, as `lever_ab` does, for a cell not yet recorded."""
    cells = TUNED_CELLS if cells is None else cells
    for c in cells:
        if arm_of(c) != TUNED:
            raise ValueError(f"{c.key()}: not a {TUNED} arm")
        if c in NOT_MEASURABLE[TUNED]:
            raise ValueError(f"{c.key()}: not measurable -- "
                             f"{NOT_MEASURABLE[TUNED][c]}")
    return {c: lever_ab(store, arch, c) for c in cells}


def tuned_verdict(store: Mapping[str, Measured], arch: str,
                  cells: tuple[Spec, ...] | None = None) -> str:
    """The search on `arch` (THE TUNED ARM, PER ARCH): SUPPORTED iff `tuned`
    WINS somewhere and LOSES nowhere, CONTRADICTED iff the reverse, else
    NOT_SHOWN (a MIXED cell, both signs, or NEUTRAL everywhere)."""
    got = {ab.verdict for ab in tuned_ab(store, arch, cells).values()}
    if MIXED not in got and WIN in got and LOSS not in got:
        return SUPPORTED
    if MIXED not in got and LOSS in got and WIN not in got:
        return CONTRADICTED
    return NOT_SHOWN


def tuned_may_become_the_default(store: Mapping[str, Measured],
                                 cells: tuple[Spec, ...] | None = None,
                                 arches: tuple[str, ...] = ("sm_86", "gfx1200"),
                                 ) -> bool:
    """THE TUNED ARM, THE DEFAULT: True iff `tuned` WINS at every measurable
    cell on every arch in `arches` (both cards by default)."""
    return all(ab.verdict == WIN for a in arches
               for ab in tuned_ab(store, a, cells).values())


# #363: every ledger arm on the clock (module docstring, THE LEVER CLOCK),
# its rule declared here before any timed KERNEL row.
FASTER_ARM, SLOWER_ARM = "faster", "slower"


def lever_clock_cells(arm: str | None = None) -> tuple[Spec, ...]:
    """Every cell #363 times: each ledger arm (or only `arm`) on every
    ledger class of its target, in `ledger_arms()` then ladder order."""
    arms = ledger_arms() if arm is None else (arm,)
    out = []
    for a in arms:
        if a not in ledger_arms():
            raise ValueError(f"{a!r} is not a ledger arm")
        out += [lever_arm(c, a) for c in ledger_classes(arm_target(a))]
    return tuple(out)


def lever_clock_ab(store: Mapping[str, Measured], arch: str,
                   arm: str) -> dict[Spec, LeverAB]:
    """`lever_ab` of every class of `arm` on `arch`. Raises KeyError, as
    `lever_ab` does, for a cell not timed against its control."""
    return {c: lever_ab(store, arch, c) for c in lever_clock_cells(arm)}


def lever_clock_verdict(store: Mapping[str, Measured], arch: str,
                        arm: str) -> str:
    """THE LEVER CLOCK, PER (ARM, ARCH): FASTER_ARM, SLOWER_ARM or
    NOT_SHOWN, read off `lever_clock_ab`'s per-class verdicts."""
    got = {ab.verdict for ab in lever_clock_ab(store, arch, arm).values()}
    if MIXED not in got and WIN in got and LOSS not in got:
        return FASTER_ARM
    if MIXED not in got and LOSS in got and WIN not in got:
        return SLOWER_ARM
    return NOT_SHOWN


def lever_may_become_the_default(store: Mapping[str, Measured], arm: str,
                                 arches: tuple[str, ...] = ("sm_86",
                                                            "gfx1200"),
                                 ) -> bool:
    """THE LEVER CLOCK, THE DEFAULT: True iff `arm` WINS at every class on
    every arch in `arches` (both cards by default)."""
    return all(ab.verdict == WIN for a in arches
               for ab in lever_clock_ab(store, a, arm).values())



# ---------------------------------------------------------------------------
# #146: the build store -- one row per timed WHOLE-MOLECULE J build
# ---------------------------------------------------------------------------
#
# WHY ITS OWN FILE. A kernels.json row is keyed on MeasuredKey(spec, arch,
# toolchain): ONE class's batch kernels. A build is a whole molecule's launch
# plan -- every (lp, lq) class of a basis, each at its own n_batches, over
# real index records under a screen -- and no Spec names it. Filing it under
# a MeasuredKey would need a fake Spec or a widened kernel row, and every
# reader of kernels.json (kernelrank, batchreport, dispatch, both verdicts)
# reads "the row for this key" as one class. So a build row is its own record
# in its own file, data/measured/builds.json, with its own provenance and
# key, and nothing that reads kernels.json sees it.

BUILDS = REPO / "data" / "measured" / "builds.json"
BUILD_SCHEMA = 1
DISPATCH_PREFIX = "dispatch:"  # n_batches of a resolved plan: "dispatch:wide"
BASELINE_ARM = "baseline"  # the plan's default; #147 labels its own arms

# The card's state around the KEPT timing phase, in every build row's raw
# (the gate's readings, felizjueves.gate; this module imports no gate, so the
# threshold the phase was judged against is recorded beside the readings):
#   build:gpu_util_before            percent, the settled reading the gate
#                                    let the phase through at
#   build:gpu_util_after             percent, gate.after_reading after it
#   build:gpu_busy_max               the gate's GPU_BUSY_MAX at the time
#   build:cpu_busy_permille_before   the CPU gate's settled reading
#   build:gate_discarded_phases      phases of this cell discarded before
#                                    this one was kept (after-reading
#                                    non-idle or unknown)
# All KNOWN integers: a contended or unreadable phase is DISCARDED, never
# recorded, so a null here is refused rather than read as idle.
BUILD_STATE_FIELDS = ("build:gpu_util_before", "build:gpu_util_after",
                      "build:gpu_busy_max", "build:cpu_busy_permille_before",
                      "build:gate_discarded_phases")

BUILD_TIMED_REGION = (
    "runtime_ns / spread_ns / repeats are the MEDIAN, the INTERQUARTILE RANGE "
    "and the COUNT of `repeats` timed builds (after raw 'build:warmup' "
    "untimed ones, all in ONE process), host std::chrono::steady_clock. ONE "
    "TIMED BUILD IS, in order: (a) one device memset zeroing each bra "
    "class's J_p buffer; (b) every batch kernel of every (lp, lq) class of "
    "the plan, launched in plan order on the default stream into its bra "
    "class's J_p (fp64 atomics); (c) one device synchronize; (d) the "
    "device-to-host copy of every bra class's J_p (one blocking copy per bra "
    "class). It starts after a device synchronize, so no earlier work is "
    "inside it. raw 'build:device_median_ns' is (a)-(c) alone and "
    "'build:readback_median_ns' is (d) alone (each the median over the same "
    "repeats); 'build:events_median_ns' is the GPU-event time from before "
    "(a) to after (b). OUTSIDE THE CLOCK: process start, reading the input "
    "file, the VRAM probe, every device allocation, and the H2D UPLOADS -- "
    "one per bra class (P zeta Ip) and one per ket class (Q eta Iq Dmax D), "
    "done ONCE per process before the first build and timed once, apart, as "
    "raw 'build:upload_ns' (a single observation, allocation included); the "
    "host-side staging that builds those buffers (Python, ao.staged); the "
    "host-side eq. 20 back-transform and J_ab scatter (Python, "
    "ao.staged._transform), which is not device code and is not timed; and "
    "compilation. So a row is the device cost of one J_p for a molecule "
    "whose index data is already resident, read back to the host ready for "
    "eq. 20 -- not a cold start and not a whole SCF iteration.")

BUILD_ABOUT = (
    "Build rows (felizjueves.measured_store, #146): the host wall-clock of "
    "ONE WHOLE-MOLECULE J_p BUILD from a launch.LaunchPlan on one card -- "
    "every class's batch kernels into one J_p per bra class -- timed behind "
    "the card gate (felizjueves.gate), keyed by build_key. Its own file: a "
    "build is no MeasuredKey's Spec, and kernels.json's rows are not "
    "widened. Observations of two cards and their toolchains, NOT comparable "
    "to [JEngine]'s A100 figures, nor across vendors.")

BUILD_HEADER = {
    "what": "One row per timed whole-molecule J_p build (launch.launches -> "
            "test/plan_driver.py's generated driver), per card, per arm.",
    "key": "build_key(fixture, tau, n_batches, density_sort, arm, arch, "
           "toolchain) = '<fixture>/tau=<repr>/b=<n>/sort=<density|global>/"
           "<arm>@<arch>/<toolchain>'. fixture is test/oracle/fixtures/"
           "<fixture>.json (the molecule, basis and density), or "
           "'h2chain2_sto3g': test/test_ao_screening.py's _h2_chain(2) "
           "(two H2, STO-3G H, hilbert density), which has no libcint "
           "fixture -- its device J is held to the launch oracle and the "
           "host sweep's J(0) plus the screen's tau * scale; n_batches is "
           "the plan's fixed count, or 'dispatch:<input>' for a plan "
           "resolved per class by dispatch.Resolver(arch, <input>); "
           "density_sort is launch.launches' sort key (density: G * Dmax per "
           "ket; global: one global Dmax, so the order is by G alone); arm "
           "labels a same-session A/B (#147) -- 'baseline' is the plan's "
           "default, timed alone; 'sort-ab' is #147's pair of arms, which "
           "differ in density_sort only.",
    "sort_key": "#147: the 'sort-ab' rows are SORT_KEY_CELLS' two arms, "
                "timed in ONE session per card (raw 'build:session', the "
                "session's start in integer seconds, equal on both arms), "
                "each with its own gate readings. The verdict is "
                "measured_store.sort_key_ab's, declared in its module "
                "docstring (THE SORT KEY) before any of these rows existed: "
                "per cell the global key is WORSE iff slower than the "
                "product by more than the larger IQR (#60's rule), BETTER "
                "iff faster by that, else UNRESOLVED (skips fewer) or SAME; "
                "WORSE on an arch iff at every cell. Computed from the rows, "
                "never stored.",
    "timed_region": BUILD_TIMED_REGION,
    "gate": "#146: every row's raw carries " + ", ".join(BUILD_STATE_FIELDS)
            + ". A cell's timing phase is one process run of the checked "
            "binary (warm-up plus repeats) between gate.wait_quiet (card at "
            "most build:gpu_busy_max percent busy by its own SMI, CPU at most "
            "gate.BUSY_MAX) and gate.after_reading; a phase whose "
            "after-reading was non-idle or unknown is DISCARDED and retaken, "
            "and counted in build:gate_discarded_phases. A contended timing "
            "is a different experiment, not a noisy one: nothing from a "
            "discarded phase is in a row, and the loader refuses a row whose "
            "card state is unknown or over the threshold.",
    "checked": "The timed binary is the checked binary: before timing, each "
               "cell's driver is run in its correctness mode and its J "
               "asserted against the launch oracle (launch.walk) and the "
               "libcint fixture (test/test_j_launched_ondevice.py's "
               "tolerances), and every timed process's first and last J_p "
               "must agree with it within the fp64-atomic run-to-run bound.",
    "plan": "raw 'build:classes', 'build:kernels' (batch-kernel launches per "
            "build), 'build:bra_classes', 'build:ket_classes', "
            "'build:n_pairs' and 'build:n_skipped' (the launch oracle's "
            "screen count on the same plan: the (bra, ket) record pairs the "
            "kernel's break skips), and 'build:refused' (classes built at 1 "
            "batch because the plan refused the asked count).",
    "not_compared": "Report, do not compare: nothing here is compared "
                    "against [JEngine]'s A100 build times, nor across vendors "
                    "(gfx1200 rate-limits fp64), nor against kernels.json's "
                    "synthetic-input class timings.",
    "rerun": "env FJ_LAUNCH_WALLCLOCK_STORE=data/measured/builds.json "
             "devtools/kernel-tier.sh --backend <cuda|hip> -- -n 0 -s "
             "-k build_wallclock (alone; one session per card).",
}

# A card with no build row by decision, and why: a session the gate refused
# for good would be recorded here, never as a zero. Empty: both cards have
# rows.
BUILD_UNMEASURED: dict[str, str] = {}
BUILD_HEADER["could_not_measure"] = dict(BUILD_UNMEASURED)

_BUILD_INTS = ("runtime_ns", "spread_ns", "repeats")


def _n_batches_word(n: int | str) -> str:
    if type(n) is int and n >= 1:
        return str(n)
    if (isinstance(n, str) and n.startswith(DISPATCH_PREFIX)
            and len(n) > len(DISPATCH_PREFIX) and "/" not in n
            and "@" not in n):
        return n
    raise ValueError(f"n_batches {n!r} is neither an int >= 1 nor "
                     f"'{DISPATCH_PREFIX}<input>'")


def build_key(fixture: str, tau: float, n_batches: int | str,
              density_sort: bool, arm: str, arch: str, toolchain: str) -> str:
    """A build row's key (BUILD_HEADER['key'])."""
    for name, v in (("fixture", fixture), ("arm", arm), ("arch", arch),
                    ("toolchain", toolchain)):
        if not isinstance(v, str) or not v or (name != "toolchain" and (
                "/" in v or "@" in v)):
            raise ValueError(f"{name} {v!r} must be a non-empty string "
                             "without '/' or '@'")
    if type(tau) is not float or not tau >= 0.0:
        raise ValueError(f"tau {tau!r} must be a float >= 0")
    if type(density_sort) is not bool:
        raise ValueError(f"density_sort {density_sort!r} must be a bool")
    sort = "density" if density_sort else "global"
    return (f"{fixture}/tau={tau!r}/b={_n_batches_word(n_batches)}/"
            f"sort={sort}/{arm}@{arch}/{toolchain}")


@dataclass(frozen=True, kw_only=True)
class BuildRow:
    """One timed whole-molecule J_p build on one card (#146). An observation,
    not a claim about the paper's numbers; BUILD_TIMED_REGION is what the
    number contains."""

    # what was built
    fixture: str
    tau: float
    n_batches: int | str
    density_sort: bool
    arm: str
    # the timing (ns, host steady_clock; BUILD_TIMED_REGION)
    runtime_ns: int
    spread_ns: int
    repeats: int
    # provenance, all REQUIRED
    toolchain: str
    arch: str
    host: str
    gpu: str
    date: str
    command: str  # the driver's compile command (paths elided)
    source_sha256: str  # of the driver text plus every unit, in launch order
    raw: Mapping[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.key()  # validates fixture / tau / n_batches / sort / arm
        for n in _BUILD_INTS:
            v = getattr(self, n)
            if type(v) is not int or v < 0:
                raise ValueError(f"{n} = {v!r} must be an int >= 0")
        if self.repeats < 1:
            raise ValueError(f"repeats = {self.repeats!r} must be >= 1")
        for n in (*PROVENANCE, "command", "source_sha256"):
            if not isinstance(getattr(self, n), str) or not getattr(self, n):
                raise ValueError(f"{n} must be a non-empty string")
        for k, v in self.raw.items():
            if not isinstance(k, str) or type(v) not in (int, float):
                raise ValueError(f"raw {k!r} = {v!r}: a number, never null "
                                 "(an unknown is not recorded as a reading)")
        for f in BUILD_STATE_FIELDS:
            v = self.raw.get(f)
            if type(v) is not int or v < 0 or (
                    "gpu_" in f and v > 100) or ("permille" in f and v > 1000):
                raise ValueError(f"raw {f} = {v!r} -- every build row carries "
                                 "the card's state as a KNOWN reading")
        cap = self.raw["build:gpu_busy_max"]
        for f in ("build:gpu_util_before", "build:gpu_util_after"):
            if self.raw[f] > cap:
                raise ValueError(f"raw {f} = {self.raw[f]}% > build:gpu_busy_"
                                 f"max = {cap}%: a contended phase is "
                                 "discarded, never recorded")

    def key(self) -> str:
        return build_key(self.fixture, self.tau, self.n_batches,
                         self.density_sort, self.arm, self.arch,
                         self.toolchain)


_BUILD_FIELDS = frozenset(f.name for f in fields(BuildRow))


def _build_row(path: Path, key: str, v: object) -> BuildRow:
    if not isinstance(v, dict):
        raise ValueError(f"{path}: row {key!r} is not an object")
    unknown = sorted(set(v) - _BUILD_FIELDS)
    if unknown:
        raise ValueError(f"{path}: row {key!r} has unknown fields {unknown}")
    try:
        row = BuildRow(**v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path}: row {key!r}: {e}") from e
    if row.key() != key:
        raise ValueError(f"{path}: row {key!r} records the key {row.key()!r}")
    if row.arch in BUILD_UNMEASURED:
        raise ValueError(f"{path}: row {key!r} is on {row.arch}, which has no "
                         "build rows by decision (BUILD_UNMEASURED)")
    return row


def load_builds(path: Path = BUILDS) -> dict[str, BuildRow]:
    """Every build row, by `build_key`. Refuses a malformed or foreign file,
    a row filed under a key not its own, and a row without a KNOWN,
    uncontended card state. A missing file raises FileNotFoundError, as
    `load` does."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not JSON ({e})") from e
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: top level is not an object")
    if doc.get("schema") != BUILD_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected "
                         f"{BUILD_SCHEMA}")
    if not isinstance(doc.get("about"), str) or not doc["about"]:
        raise ValueError(f"{path}: no \"about\" string")
    got = doc.get("rows")
    if not isinstance(got, dict):
        raise ValueError(f"{path}: \"rows\" is not an object")
    return {k: _build_row(path, k, v) for k, v in got.items()}


def merge_build(store: Mapping[str, BuildRow],
                row: BuildRow) -> dict[str, BuildRow]:
    """`store` with `row` at its own key, REPLACING any row already there."""
    return {**store, row.key(): row}


def save_builds(store: Mapping[str, BuildRow], path: Path = BUILDS) -> None:
    for k, r in store.items():
        if r.key() != k:
            raise ValueError(f"row filed at {k!r} is {r.key()!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": BUILD_SCHEMA, "about": BUILD_ABOUT,
           "header": BUILD_HEADER,
           "rows": {k: asdict(store[k])
                    for k in sorted(store, key=_build_order)}}
    path.write_text(json.dumps(doc, indent=2) + "\n")


def _build_order(key: str) -> tuple:
    cell, _, where = key.partition("@")
    return (where, cell)


# ---------------------------------------------------------------------------
# #147: the sort key, per-record product against one global max|D|
# (module docstring, THE SORT KEY -- declared before any sort-key row)
# ---------------------------------------------------------------------------

SORT_AB_ARM = "sort-ab"  # both arms' label; the pair differs in sort only
SESSION_FIELD = "build:session"  # the session's start, integer seconds
CHAIN_INPUT = "h2chain2_sto3g"  # test_ao_screening._h2_chain(2), hilbert D
SORT_KEY_TAUS = (1e-3, 1e-10)
SORT_KEY_CELLS: tuple[tuple[str, float, int | str], ...] = tuple(
    (fixture, tau, n) for fixture, n in (("h2o_ccpvdz_j", "dispatch:wide"),
                                         (CHAIN_INPUT, 1))
    for tau in SORT_KEY_TAUS)
FEWER, SAME = "fewer", "same"
WORSE, BETTER, UNRESOLVED = "worse", "better", "unresolved"
CONTRADICTED, NOT_SHOWN = "contradicted", "not shown"


@dataclass(frozen=True)
class SortKeyAB:
    """The global-`max|D|` arm against the product arm, one cell, one arch."""

    arch: str
    fixture: str
    tau: float
    n_batches: int | str
    n_pairs: int
    skipped: tuple[int, int]  # (product, global)
    clock: tuple[float, float, float, float]  # product median, IQR; global's

    def __post_init__(self) -> None:
        if self.skipped[1] > self.skipped[0]:
            raise ValueError(
                f"{self.fixture} tau={self.tau!r} @ {self.arch}: the global "
                f"key skips {self.skipped[1]} > the product's "
                f"{self.skipped[0]} -- impossible for a bounding key, so the "
                "rows are wrong, not the verdict")

    @property
    def skip_sign(self) -> str:
        return FEWER if self.skipped[1] < self.skipped[0] else SAME

    @property
    def clock_sign(self) -> str:
        """FASTER / SLOWER / NEUTRAL, of the GLOBAL arm (#60's rule)."""
        return _sign(self.clock[:2], self.clock[2:])

    @property
    def verdict(self) -> str:
        if self.clock_sign == SLOWER:
            return WORSE
        if self.clock_sign == FASTER:
            return BETTER
        return UNRESOLVED if self.skip_sign == FEWER else SAME


def _sort_row(store: Mapping[str, BuildRow], arch: str,
              cell: tuple[str, float, int | str], density_sort: bool
              ) -> BuildRow:
    fixture, tau, n = cell
    got = [r for r in store.values()
           if r.arch == arch and r.fixture == fixture and r.tau == tau
           and r.n_batches == n and r.density_sort == density_sort
           and r.arm == SORT_AB_ARM]
    if len(got) != 1:
        raise KeyError(f"{fixture} tau={tau!r} b={n} sort="
                       f"{'density' if density_sort else 'global'} "
                       f"{SORT_AB_ARM} @ {arch}: {len(got)} rows")
    return got[0]


def sort_key_ab(store: Mapping[str, BuildRow], arch: str,
                cell: tuple[str, float, int | str]) -> SortKeyAB:
    """#147's comparison for one of `SORT_KEY_CELLS` on `arch`, from the
    build store. Raises KeyError when an arm is missing, when the two arms
    were not timed in one session, or when they differ in the toolchain or
    the plan's size."""
    p = _sort_row(store, arch, cell, True)
    g = _sort_row(store, arch, cell, False)
    for f in (SESSION_FIELD, "build:n_pairs", "build:n_skipped"):
        for r in (p, g):
            if f not in r.raw:
                raise KeyError(f"{r.key()}: no raw {f}")
    if p.raw[SESSION_FIELD] != g.raw[SESSION_FIELD]:
        raise KeyError(f"{cell} @ {arch}: the arms were timed in different "
                       f"sessions ({p.raw[SESSION_FIELD]} and "
                       f"{g.raw[SESSION_FIELD]})")
    if p.toolchain != g.toolchain or (p.raw["build:n_pairs"]
                                      != g.raw["build:n_pairs"]):
        raise KeyError(f"{cell} @ {arch}: the arms differ in more than the "
                       "sort (toolchain or plan size)")
    return SortKeyAB(
        arch=arch, fixture=cell[0], tau=cell[1], n_batches=cell[2],
        n_pairs=int(p.raw["build:n_pairs"]),
        skipped=(int(p.raw["build:n_skipped"]), int(g.raw["build:n_skipped"])),
        clock=(p.runtime_ns, p.spread_ns, g.runtime_ns, g.spread_ns))


def sort_key_arch_verdict(store: Mapping[str, BuildRow], arch: str,
                          cells=SORT_KEY_CELLS) -> str:
    """WORSE, CONTRADICTED or NOT SHOWN on one arch, over every cell."""
    got = [sort_key_ab(store, arch, c).verdict for c in cells]
    if any(v == BETTER for v in got):
        return CONTRADICTED
    if all(v == WORSE for v in got):
        return WORSE
    return NOT_SHOWN


def sort_key_claim_holds(store: Mapping[str, BuildRow], arches,
                         cells=SORT_KEY_CELLS) -> bool:
    """The measured claim is `asserted` iff WORSE on every arch."""
    return all(sort_key_arch_verdict(store, a, cells) == WORSE
               for a in arches)



# ---------------------------------------------------------------------------
# #164: jengine/sieve-non-monotone-in-spills, read on OUR cards
# ---------------------------------------------------------------------------
#
# THE PUBLISHED PATTERN ([JEngine, v2, §4.1]): the sieve's effect on spills
# is non-monotone across classes -- about 4% fewer on average, but [5|6] 3%
# more and [8|8] 11% fewer. #164 asked two questions of it, written before
# #162's rows existed: does the mean fall, and is the effect non-monotone
# across classes. How each is read was fixed in #164, after the rows were
# visible, and is transcribed here, per arch, over SIEVE_CELLS (each
# `no-sieve` cell against its literal, sieved control):
#
# - a cell's SPILLS are `spill_store + spill_load` of its row, summed over
#   its batch kernels by the store already. On sm_86 that is bytes (ptxas);
#   on gfx1200 both fields carry the spilled-REGISTER count (hipcc's
#   kernel-resource-usage, VGPRs Spill + SGPRs Spill), so the sum is twice
#   it -- a ratio within one arch is unaffected, a number across arches is
#   not comparable;
# - its CHANGE is (sieved - unsieved) / unsieved: negative is fewer spills
#   with the sieve. A cell where neither arm spills has no change and is
#   left out of the mean (its sign is SAME); one where only the sieved arm
#   spills has no finite change and is refused;
# - the MEAN is the arithmetic mean of the changes over the cells that
#   spill (the paper's "on average", read per class); `total_change` is the
#   ratio of summed spills, reported, not judged;
# - NON-MONOTONE iff the sieve gives MORE spills at some cell and FEWER at
#   another;
# - the two named classes are read at L = 16, one batch, (8,0|8,0) -- which
#   is [8|8] -- and at #162's L = 11 cell, (6,0|5,0): [6|5], the paper's
#   [5|6] with bra and ket swapped (pq_spec's split), a different kernel, so
#   its reading is a stand-in and recorded as one.
#
# The published pattern holds on an arch iff all four halves hold (mean
# falls, non-monotone, the [5|6] stand-in MORE, [8|8] FEWER). What it holds
# or fails on is OUR kernels on OUR cards: never the paper's A100, and never
# compared against Table 4 (data/literature/jengine/limits.json).

MORE = "more"
EIGHT_EIGHT = (16, 1)  # (L, n_batches): pq_spec(16) is (8,0|8,0)
FIVE_SIX_STAND_IN = (11, 1)  # pq_spec(11) is (6,0|5,0): [5|6] swapped


@dataclass(frozen=True)
class SieveSpill:
    """One `no-sieve` cell's spills against its sieved control, one arch."""

    arch: str
    am: tuple[int, ...]
    L: int
    n_batches: int
    sieved: int  # the literal row's spill_store + spill_load
    unsieved: int  # the no-sieve row's

    def __post_init__(self) -> None:
        if self.sieved < 0 or self.unsieved < 0:
            raise ValueError(f"L{self.L} b{self.n_batches} @ {self.arch}: a "
                             "negative spill count")
        if self.unsieved == 0 and self.sieved > 0:
            raise ValueError(
                f"L{self.L} b{self.n_batches} @ {self.arch}: only the sieved "
                "arm spills, so the relative change from zero is not finite")

    @property
    def sign(self) -> str:
        """MORE / FEWER / SAME spills WITH the sieve."""
        if self.sieved > self.unsieved:
            return MORE
        if self.sieved < self.unsieved:
            return FEWER
        return SAME

    @property
    def change(self) -> float | None:
        """(sieved - unsieved) / unsieved, or None where neither arm spills."""
        if self.unsieved == 0:
            return None
        return (self.sieved - self.unsieved) / self.unsieved


@dataclass(frozen=True)
class SieveSpillPattern:
    """#164's reading of the sieve's spills over a cell set, one arch."""

    arch: str
    cells: tuple[SieveSpill, ...]

    @property
    def spilling(self) -> tuple[SieveSpill, ...]:
        return tuple(c for c in self.cells if c.change is not None)

    @property
    def mean_change(self) -> float:
        got = [c.change for c in self.spilling]
        if not got:
            raise ValueError(f"{self.arch}: no cell spills in either arm")
        return sum(got) / len(got)

    @property
    def total_change(self) -> float:
        s = sum(c.sieved for c in self.cells)
        u = sum(c.unsieved for c in self.cells)
        if u == 0:
            raise ValueError(f"{self.arch}: no cell spills in either arm")
        return (s - u) / u

    @property
    def mean_falls(self) -> bool:
        return self.mean_change < 0

    @property
    def non_monotone(self) -> bool:
        signs = {c.sign for c in self.cells}
        return MORE in signs and FEWER in signs

    def at(self, L: int, n_batches: int) -> SieveSpill:
        got = [c for c in self.cells if (c.L, c.n_batches) == (L, n_batches)]
        if len(got) != 1:
            raise KeyError(f"L{L} b{n_batches} @ {self.arch}: {len(got)} "
                           "cells")
        return got[0]

    def halves(self) -> dict[str, bool]:
        """The published pattern, half by half, on this arch."""
        return {
            "mean falls": self.mean_falls,
            "non-monotone": self.non_monotone,
            "[5|6] more": self.at(*FIVE_SIX_STAND_IN).sign == MORE,
            "[8|8] fewer": self.at(*EIGHT_EIGHT).sign == FEWER,
        }

    @property
    def holds(self) -> bool:
        return all(self.halves().values())


def sieve_spills(store: Mapping[str, Measured], arch: str,
                 cells: tuple[Spec, ...] | None = None) -> SieveSpillPattern:
    """#164's pattern on `arch` over `cells` (by default SIEVE_CELLS), read
    from the spill fields alone -- a cell need be compiled, not timed.

    Raises ValueError for a cell that is not a `no-sieve` arm, and KeyError
    when either row is missing or carries no spill fields."""
    cells = SIEVE_CELLS if cells is None else cells
    out = []
    for c in cells:
        if arm_of(c) != NO_SIEVE:
            raise ValueError(f"{c.key()}: not a {NO_SIEVE} arm")
        pair = []
        for s in (replace(c, levers=()), c):
            got = rows(spec=s, arch=arch, store=store)
            if len(got) != 1:
                raise KeyError(f"{s.key()} @ {arch}: {len(got)} rows")
            (r,) = got.values()
            if r.spill_store is None or r.spill_load is None:
                raise KeyError(f"{s.key()} @ {arch}: not compiled")
            pair.append(r.spill_store + r.spill_load)
        out.append(SieveSpill(arch=arch, am=tuple(c.am), L=c.total_am,
                              n_batches=c.n_batches, sieved=pair[0],
                              unsieved=pair[1]))
    return SieveSpillPattern(arch=arch, cells=tuple(out))


def sieve_spill_claim_holds(store: Mapping[str, Measured],
                            arches: tuple[str, ...] = ("sm_86", "gfx1200"),
                            cells: tuple[Spec, ...] | None = None) -> bool:
    """The published pattern holds iff it holds, all four halves, on every
    arch in `arches`."""
    return all(sieve_spills(store, a, cells).holds for a in arches)


# ---------------------------------------------------------------------------
# #205: the K rows against J's at the same L (module docstring, THE K ROWS)
# ---------------------------------------------------------------------------

SAME = "same"
NO_SPILL = "no spill"
NOT_JUDGED = "not judged"


def j_twin(spec: Spec) -> Spec:
    """The J subgrid cell a K cell is the twin of: same L, same n_batches."""
    if spec.target is not Target.K:
        raise ValueError(f"{spec.key()} is not a K cell")
    return pq_spec(spec.total_am, n_batches=spec.n_batches)


def kernel_spills(row: Measured) -> tuple[int, ...]:
    """Each batch kernel's spill (store + load, in the row's own unit), in
    the row's kernel order, read off `raw` -- they sum to the row's."""
    names = row.kernel.split(",")
    out = []
    for n in names:
        pre = "" if len(names) == 1 else f"{n}:"
        if f"{pre}spill_store_bytes" in row.raw:  # nvcc
            out.append(row.raw[f"{pre}spill_store_bytes"]
                       + row.raw[f"{pre}spill_load_bytes"])
        else:  # hipcc: one count, carried by both fields
            out.append(2 * (row.raw[f"{pre}VGPRs Spill"]
                            + row.raw.get(f"{pre}SGPRs Spill", 0)))
    return tuple(out)


def _relative(s: int, s1: int) -> float:
    """s(b) / s(1); a curve with no spill at one batch is flat while it
    stays at 0, and has risen (inf) once it spills."""
    if s1 == 0:
        return 1.0 if s == 0 else math.inf
    return s / s1


@dataclass(frozen=True)
class KSpillCurve:
    """K's spill curve against its J twin's, at one (arch, L) -- the rule is
    the module docstring's THE K ROWS, SPILL."""

    arch: str
    L: int
    batches: tuple[int, ...]  # judged: both rows recorded, 1 first
    k: tuple[int, ...]  # the cell's whole spill, per b in `batches`
    j: tuple[int, ...]
    k_worst: tuple[int, ...]  # the worst batch kernel's: reported only
    j_worst: tuple[int, ...]
    refused: tuple[int, ...]  # b with no K row (K_REFUSED)

    @property
    def signs(self) -> tuple[str, ...]:
        """Per split b of `batches`: K FASTER / SLOWER / SAME than J."""
        out = []
        for i in range(1, len(self.batches)):
            rk = _relative(self.k[i], self.k[0])
            rj = _relative(self.j[i], self.j[0])
            out.append(FASTER if rk < rj else SLOWER if rk > rj else SAME)
        return tuple(out)

    @property
    def verdict(self) -> str:
        if len(self.batches) < 2 or self.batches[0] != 1:
            return NOT_JUDGED
        if self.k[0] == 0 and self.j[0] == 0 and not any(self.k + self.j):
            return NO_SPILL
        signs = set(self.signs)
        if signs == {FASTER}:
            return FASTER
        if signs == {SLOWER}:
            return SLOWER
        return MIXED


def k_spill_curve(store: Mapping[str, Measured], arch: str, L: int,
                  refused: Mapping[Spec, str] | None = None) -> KSpillCurve:
    """THE K ROWS, SPILL, at (arch, L), from the store. A K cell in
    `refused` (default `K_REFUSED[arch]`) is named, not judged; any other
    missing or uncompiled row raises KeyError."""
    refused = K_REFUSED.get(arch, {}) if refused is None else refused
    batches, k, j, kw, jw, gone = [], [], [], [], [], []
    for spec in K_SUBGRID:
        if spec.total_am != L:
            continue
        if spec in refused:
            gone.append(spec.n_batches)
            continue
        pair = []
        for s in (spec, j_twin(spec)):
            got = rows(spec=s, arch=arch, store=store)
            if len(got) != 1:
                raise KeyError(f"{s.key()} @ {arch}: {len(got)} rows")
            (r,) = got.values()
            if r.spill_store is None or r.spill_load is None:
                raise KeyError(f"{s.key()} @ {arch}: not compiled")
            pair.append(r)
        batches.append(spec.n_batches)
        k.append(pair[0].spill_store + pair[0].spill_load)
        j.append(pair[1].spill_store + pair[1].spill_load)
        kw.append(max(kernel_spills(pair[0])))
        jw.append(max(kernel_spills(pair[1])))
    if not batches and not gone:
        raise KeyError(f"L={L}: not a K_SUBGRID L")
    return KSpillCurve(arch=arch, L=L, batches=tuple(batches), k=tuple(k),
                       j=tuple(j), k_worst=tuple(kw), j_worst=tuple(jw),
                       refused=tuple(gone))


@dataclass(frozen=True)
class KGain:
    """One input's best-median gain of the split class over its one-batch
    cell, K against its same-session J twins, at one (arch, L)."""

    arch: str
    L: int
    input: str  # "wide" or "near"
    batches: tuple[int, ...]
    k: tuple[float, ...]  # medians, ns, per b
    j: tuple[float, ...]

    @staticmethod
    def _gain(t: tuple[float, ...]) -> float:
        return t[0] / min(t)

    @property
    def k_gain(self) -> float:
        return self._gain(self.k)

    @property
    def j_gain(self) -> float:
        return self._gain(self.j)

    @property
    def k_best(self) -> int:
        return self.batches[self.k.index(min(self.k))]

    @property
    def j_best(self) -> int:
        return self.batches[self.j.index(min(self.j))]

    @property
    def k_at_least_j(self) -> bool:
        return self.k_gain >= self.j_gain


def k_gain(store: Mapping[str, Measured], arch: str, L: int,
           refused: Mapping[Spec, str] | None = None) -> dict[str, KGain]:
    """THE K ROWS, WALL-CLOCK, at (arch, L): {"wide": ..., "near": ...}.
    Raises KeyError when the one-batch K cell, or any unrefused cell, is
    unrecorded or untimed against its same-session J twin. A cell in
    `refused` (default: `K_REFUSED[arch]` and `K_UNTIMED[arch]`) is left
    out; a refused one-batch cell leaves nothing to gain over."""
    if refused is None:
        refused = {**K_REFUSED.get(arch, {}), **K_UNTIMED.get(arch, {})}
    cells = [s for s in K_SUBGRID if s.total_am == L and s not in refused]
    if not cells or cells[0].n_batches != 1:
        raise KeyError(f"L={L} @ {arch}: no one-batch K cell to gain over")
    got: dict[str, tuple[list[float], list[float]]] = {
        "wide": ([], []), "near": ([], [])}
    for s in cells:
        rs = rows(spec=s, arch=arch, store=store)
        if len(rs) != 1:
            raise KeyError(f"{s.key()} @ {arch}: {len(rs)} rows")
        (r,) = rs.values()
        if r.runtime_ns is None or "wallclock:control:median_ns" not in r.raw:
            raise KeyError(f"{s.key()} @ {arch}: not timed against its J twin")
        got["wide"][0].append(r.runtime_ns)
        got["wide"][1].append(r.raw["wallclock:control:median_ns"])
        got["near"][0].append(r.raw["wallclock:near:median_ns"])
        got["near"][1].append(r.raw["wallclock:control:near:median_ns"])
    b = tuple(s.n_batches for s in cells)
    return {i: KGain(arch=arch, L=L, input=i, batches=b, k=tuple(kv),
                     j=tuple(jv)) for i, (kv, jv) in got.items()}
