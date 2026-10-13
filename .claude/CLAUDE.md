# CLAUDE.md

## What this is

`FelizJueves` — **Python code generation of GPU kernels** for
high-angular-momentum electron repulsion integrals (ERIs) and Coulomb-matrix
(`J`) construction: the McMurchie-Davidson recurrences and sub-batching of
Palethorpe & Barca, emitted as CUDA and HIP source rather than written by hand.

**Why a generator.** One kernel per `(l₁l₂|l₃l₄)` class, and the recurrence
tree, the register and shared-memory budget, and the chosen sub-batch split are
all functions of that class. The reviewable artefact is the thing that emits the
kernels, not the hundreds of kernels — so the generator is the product, and a
kernel is its output.

**Both vendors, from one source.** This box has an RTX 3080 (sm_86, CUDA 13.0)
and an RX 9060 XT (gfx1200, RDNA4, ROCm 7.2.4), and a generated kernel has to
satisfy **both** device compilers. `hipcc` is clang and `nvcc` is not, so they
disagree about more than their flags; that disagreement is this project's main
portability risk, and `.devcontainer/combined/` exists to put both in one shell.

`docs/references.md` is the citation of record: a flat numbered list, every
entry resolving by **DOI** and no PDF tracked anywhere in the repo. Cite by
number, or by the short key where one is defined — `[Fock]` is entry 23 and
`[JEngine]` is entry 35. A version inside a key, as in `[JEngine, v2, eq. 14]`,
means the ChemRxiv v2 preprint's equation numbering and not the journal
version's, which may renumber.

## The state of the tree — read this before trusting any tier

**The generator is live for the whole `J` chain** (milestone 7): every
arithmetic step of `J_µν` is a generated kernel. They are the `[p\|q]` kernel
(`Target.J`; one batch, and, through `build_pq_kernels`, a split class
(`n_batches > 1`) as atomic batch kernels), the ket pre-contraction
(`Target.DQ`, `kernel_dq.py`: `D_q` and `Dmax`) and the bra back-transform
and scatter (`Target.JAB`, `kernel_jab.py`: `J_ab` into `J_µν`), with the
undressed `E` transform (`e_fragment.py`, an `ir.Fragment`) inlined into the
last two. `DQ` and `JAB` are this repo's own design: `[JEngine]` defers those
steps to `[Fock]`, which is not read here (`limits.json`). **Around them,
`J_µν` for a real molecule is also built host-side** (`src/felizjueves/ao/`:
the `E` transform, pair data, the ket pre-contraction, the bra back-transform
and the scatter, all pure Python). That path is the reference each kernel is
held to, bitwise when interpreted, and it is **witnessed against
PySCF/libcint** (milestone 4, `j_engine.md`, "The Coulomb matrix, witnessed
from outside"). **Since milestone 6 a whole molecule's `J` is built from a
launch plan** (`launch.py`: one `ClassLaunch` per `(lp, lq)`, bra records in
pair order, kets sorted by `I_q · max|D_q|`). On each card, every bra and ket
class is uploaded once, and one `J_p` buffer per bra class takes every batch
kernel through fp64 atomics. Since milestone 7, `DQ` produces the kets'
`D_q` on the card and `JAB` turns each `J_p` buffer into `J_µν` there (#157:
pair kernels through (ff), `J` through d, on both cards). The plan's order is
witnessed by the launch oracle (pure Python) and by libcint (`j_engine.md`,
"The launch plan"). **Still host-side:** the pair list and each record's
inputs, the Schwarz bound, the sort (it reads `Dmax` back), the staging, and
the launch code, a test-side driver run from Python. **Nothing about `DQ` or
`JAB` is measured.** `j_engine.md`, "The J chain as device code", is the one
place that says which steps are generated and on which cards they ran.
Dispatch (`dispatch.py`, #145) chooses `n_batches` per class from measured
wall-clock for a launch plan, but the generator does not follow it:
`Spec.n_batches` defaults to 1. Not built, by decision: `J` resident across
SCF iterations, a `Launch` IR, a `LaunchCost` model (`limits.json`).
Against the predecessor `jgen`, this is parity on its energy-order `J`
surface, with one difference: Table 4, declined. Which plan ships is a
decision, not a gap: `jgen` ships its searched sieve order, and this repo has
that wire too (the opt-in `tuned` lever, #161) but ships the **literal**
plan, because the searched one was measured on both cards and refused on
wall-clock (#166; `j_engine.md`, "Parity with the predecessor"). Every tier
reports what it actually touched rather
than implying more: a `gpu`-marked test skips without a card, so a run can be
**green and empty**; `kernel-tier.sh` exits **2** in that case and whenever it
could not run — read a 2 as "could not run", never as a pass.

| | State |
|---|---|
| `docker/`, `.devcontainer/`, `devtools/` | **live.** Both toolchains build; `devtools/kernel-tier.sh` runs end to end on this box. |
| `test/` | **live, and thin.** `test_toolchain.py` asserts nvcc and hipcc compile for this box's targets and that a kernel launched on each card matches a numpy reference, plus the counter witnesses (`ncu` reads a probe's FLOPs and bytes on CUDA; on HIP, since #140, `rocprofv3` loads and two witnesses pin why there is still no gfx1200 row: its byte counters read 0, refused as `measured.ROCPROF_ZERO_BYTES`, and its `--list-avail` has no fp64 counter). `test_kernel_device.py` (`gpu` + `cuda`/`hip`) runs the **generated** `[p\|q]` kernel and `fj_boys` on each card against `reference/`, plus every split cell of `measured_store.SUBGRID` as its batched launch sequence, plus #113's asymmetric study cells, plus opt-in sweeps that skip unless their store is set and run alone: wall-clock (`FJ_WALLCLOCK_STORE`; `FJ_WALLCLOCK_ASYM=1` for #113's arms) and roofline counters (`FJ_ROOFLINE_STORE`, `ncu` on CUDA; the hip test skips with `measured_store.UNMEASURED`'s reason, final since #140). `test_ceilings_device.py` (`gpu`) bounds each card's probe by the vendor's published peak (on gfx1200's fp64, per counted GPU cycle through `rocprofv3`'s `GRBM_GUI_ACTIVE`, #141) and, opt-in (`FJ_CEILINGS_STORE`), records the ceilings. `test_j_launched_ondevice.py` (`gpu` + `cuda`/`hip`; generator `test/plan_driver.py`) builds a whole molecule's `J` from the launch plan on each card against the launch oracle and libcint (H₂, H₂O/STO-3G at 1 and 3 batches and `τ` 0 and 1e-3, H₂O/cc-pVDZ dispatched), the two-H₂ chain under both sort keys, and, opt-in and alone, the whole-build wall-clock (`FJ_LAUNCH_WALLCLOCK_STORE`, #146) and #147's sort-key A/B; its compile-only companions (`cuda`/`hip`, no `gpu`) fail on a warning in generated text. `test_j_chain_ondevice.py` (`gpu` + `cuda`/`hip`, #157) runs the generated chain: part 1 compiles each `dq_{la}_{lb}` and `jab_{la}{lb}` unit the tier wrote alone into a harness and runs it over every record of its class in HF/cc-pVTZ, (ss) through (ff), against `ao.j.dq_from_host` / `jab_from_host`; part 2 builds `J` for H₂, H₂O/STO-3G (1 and 3 batches) and H₂O/cc-pVDZ with every arithmetic step on the card (a `DQ` program, `Dmax` read back and sorted on the host, milestone 6's driver extended with one `JAB` launch per bra class) against libcint and the launch oracle, under error bounds stated in its docstring; its compile-only companions fail on a warning in a pair unit. Test-side device compiles (`conftest.compile_device_source`, `plan_driver.compile_checked`) go through sccache via `test/devcache.py` (#186; `KERNEL_NO_CACHE` turns it off; `test_devcache.py`). `test_measured.py` checks the measured store and its driver (its `cuda`/`hip` cases recompile and compare; the rest is fast: the byte-for-byte round trip of `kernels.json`, `roofline.json` and `builds.json`, the counter-row and build-row shapes and their loaders' refusals, the ncu/rocprofv3 parsers on recorded or synthetic CSV, the declared verdicts, #147's sort-key verdict held cell by cell). The rest is fast-tier: the `Spec` key, the transcribed tables' structural invariants, the scoring rules on **synthetic** candidates, the batch kernels interpreted against `reference/` (`test_kernel_batched.py`), the goldens (`test_emit_*`), `test_plan_optimum.py` (the exact optimum against an unpruned brute force, the beam as an upper bound), `test_mild_vs_optimum.py` (#111's gap; beam L = 11..16 `slow`), `test_plan_recovery.py` (the widened seed-order search, `slow`), `test_eq17_witness.py` (eq. 17's `1y`, broken by the misprint), `test_ceilings.py` and `test_device.py` (the ceiling records' provenance, the gate's refusals with injected readers, the refusal to score an uncharacterised card), `test_ridge.py` (#114's rule on synthetic rows and its claim on the tracked ones), `test_kernelrank.py` and `test_batchreport.py` recomputing `data/measured/{limits,crossread,batch_recommendation}.json` from the stores, and `test_layering.py`'s import allow-list. **The J chain, interpreted** (milestone 7, fast tier unless `slow`): `test_ir.py` and `test_interp.py` (`ir.Fragment` / `Inline` and their binding contract, #149), `test_e_fragment.py` (the `E` fragment bitwise against `ao.hermite`, (ss) through (ff), and every recurrence step one `Weighted`), `test_kernel_dq.py` and `test_kernel_jab.py` (each kernel bitwise against the host for every pair class; d and f classes `slow`), `test_ao_j_chain.py` (`J` with every step generated, against libcint and the host paths; 6-31G and cc-pVDZ `slow`), `test_emit_cli.py` (the generator CLI's two ladders and the selector's `order`), and `test_spec.py` / `test_measured_keys.py` (a pair-class key never collides with a quartet key, and all 196 measured keys are unchanged). **The plan's choices and the shipped plan** (milestone 8, fast tier unless `slow`): `test_plan_determinism.py` (the plan's output byte-identical across hash seeds, and the sieve a fixed point), `test_plan_order_search.py` (`plan/order_search.py`'s frontier, pruned product and climb), `test_plan_search.py` and `test_scoreboards.py` (the searched record replayed, never believed, and the search not the best of the 58 rules), `test_sieve_spill_claim.py` (#164's spill reading of the `no-sieve` rows), `test_planrank.py` (`plan_vs_kernel.json` recomputed) and `test_shipped_plan.py` (the literal default held to the declared rule and to the literal rows' text, and #166's limit recomputed). **The launch plan** (milestone 6, fast tier, no compiler): `test_launch.py` (the plan's records, classes, stable sort and refusals; the `jengine/ket-sort-key-is-the-product` claim), `test_launch_oracle.py` (the oracle against the host screened sweep to 0, the ordered reference and libcint; both sort keys' skip counts), `test_launch_staged.py` (`ao/staged.py` interpreted against the host sweep, libcint and the oracle's ket-loop counts; H₂O against libcint `slow`) and `test_dispatch.py` (the choice from synthetic rows, its fallbacks and refusals, the tracked file recomputed). **The AO layer and its oracle** (milestone 4): `test_ao_{shell,hermite,pairs,eri,j,j_interpreted,screening}.py` test `ao/` — `E` against closed-form Hermite polynomials, the Schwarz bound proved over randomised geometry, and the ERI blocks (all 55 classes of HF/cc-pVTZ since #121, split across five fixtures by L) and `J_µν` (by `reference/` and by the interpreted generated kernel) against PySCF/libcint through **tracked fixtures** in `test/oracle/fixtures/` (`test/oracle/`: `generate.py`, `manifest.py`, `basis.py`; `test_oracle_{basis,manifest}.py`, the overlap as the cheapest witness of libcint's conventions). They run without PySCF — s/p in the fast tier, `d`+ and the larger `J` cases `slow`; only the `oracle`-marked drift tests need the `oracle` group, and they skip without it. `test_ao_j_ondevice.py` (`gpu` + `cuda`/`hip`) builds `J_µν` with `J_p` from the class selector's kernels on each card against the same fixture (it reads the `[p\|q]` slice of each cell's `order`, between its `DQ` and its `JAB`). `test_claims.py` checks `data/literature/jengine/claims.json` against the `claim`-marked tests, both ways, `coverage.json` against the records, `j_engine.md`'s status table and every claims tally a doc quotes against the file, and every count of `limits.json`'s records a doc spells. |
| `src/felizjueves/` | **importable, and small.** `spec.py` is `Spec`, the frozen key every later subsystem joins on; since #150 its `am` is a quartet for `Target.J` and a pair `(la, lb)` for `Target.DQ` and `Target.JAB` (`AM_LENGTH`), a pair target never batched. `literature/` loads published tables and scores a candidate against them: `tables`, `rules` (`Agreement`, `AtOrUnder`, `ExactGivenInputs` — which since #108 can carry a `refused` decision instead of inputs, scoring UNSET, never PASS), `scoreboard`, `limits`, `claims`. `plan/` (below) computes the Table 1–3 cells it scores. `kernel.py` builds the `[p\|q]` kernel as IR (`ir.py`, checked by `interp.py`); `emit/` spells it for each vendor (`python -m felizjueves.emit` is the generator CLI); `build_pq_kernels` builds a split class's batch kernels, each `passes.atomize`d. `ir.Fragment` / `ir.Inline` (#149) are a device-code block with a checked binding contract, one level only (no nested `Inline`); `e_fragment.py` (#152) is the undressed `E` as fragments (the block composed at build time from one axis builder; every entry one `Weighted`, the seed `exp(+ζ X_PA X_PB)`); `kernel_dq.py` (#153) and `kernel_jab.py` (#154) are the pre-contraction and bra-transform-and-scatter kernels that inline it, each with its own layering rule that keeps it off `ao`, `interp`, `emit`, `device` and `reference/`. `device.py` holds `Arch` — with each card's measured `fp64_flops_per_s` and `mem_bandwidth_bytes_per_s`, read from `data/measured/ceilings.json` (`0.0` is *not characterised*, and `Arch.ridge` / `verdict` refuse such a card) — and `MeasuredKey`. `gate.py` is the card gate every timed, probed or profiled phase runs behind (card at most `GPU_BUSY_MAX` = 30% busy and CPU at most 10% before, and a settled reading after; non-idle or unreadable refuses). `ceilings.py` is the fp64-FMA and STREAM-triad probe pair (#109). Since #94 the measured layer is two modules: `measured_store.py` is the store — `Measured` rows, `SUBGRID`, the counter fields and per-batch-kernel counter rows (#110), the roofline point and verdict (computed, never stored, #112), the verdicts declared before their rows (`nine_vs_twelve`, `lever_ab`, #147's `sort_key_ab`, and milestone 8's `sieve_verdict` / `sieve_may_leave_the_default`, `tuned_verdict` / `tuned_may_become_the_default` and `sieve_spills`, over the declared arms `ARMS`, each with its kind on the row), and `BuildRow`, the whole-build wall-clock row (#146) — and `measured.py` is the driver that compiles, parses ptxas / AMD remarks and `ncu` / `rocprofv3` output, and runs `--remeasure` (and `--asym`, `--roofline`, `--lever`, `--lever-ab`). `planrank.py` (#165) joins `PlanCost` for literal, `no-sieve` and `tuned` against the measured spills and same-session clock per card: the counts do not predict the kernel on either card. `kernelrank.py` ranks `lower.KernelCost` against those rows (#54); `batchreport.py` derives the recommended `n_batches` per `(L, arch)` from wall-clock alone (#56); `ridge.py` is #114's crossing rule, transcribed from the issue that declared it before the rows; `crossread.py` reads model, clock and counters against each other per `(L, arch)` (#115: the choice is the clock's); `asym.py` is #113's asymmetric-split study path (its own layering rule; `Spec` and `partition` still refuse those counts). `launch.py` (#138, #142) is the whole-molecule launch plan as a value — `launches()` → `LaunchPlan` of `ClassLaunch(lp, lq, n_batches, bras, kets, kernels, bra_blocks)` over primitive-pair `PairRec`s, kets sorted descending by `I_q · max|D_q|`, stably — and the launch oracle that walks it in pure Python (`walk`, `coulomb_launched`); its bound and `[p\|q]` arrive injected, and its own layering rule keeps it off `reference/`, `interp`, `emit`, `ao.drive`, `ao.staged`, `dispatch` and the measured layer. `dispatch.py` (#145) is a class's batch count from measured wall-clock only (`batchreport`'s best median per `(arch, class, input)`, the input required; 1 for a class with no row; an uncharacterised card refused), injected into `launches(resolve=)`, never imported by it. Levers: `no-sieve` and `tuned` (plan; `tuned`, #161, builds from the searched board's recorded cut and per-batch seed orders through `pipeline.tuned_plan`, and refuses a cell with no recorded choice; both A/B'd on both cards, #162 and #163, both NOT SHOWN, so neither moved the default, #166); `passes.atomize` (applied by the builder to every split class, not a choice); and `recycle` (`lower.RECYCLE`, slot recycling, threaded through `Spec.levers` and applied by `measured`'s driver, not by `KERNEL_GEN_CMD`). #55 A/B'd `recycle` over all 20 subgrid cells on both cards: nvcc 13.0 and hipcc / ROCm 7.2.4 emit identical machine code for both arms, so it is neutral — `sm_86` 20/20 neutral, `gfx1200` 16/20 neutral with 4 cells read as slower on that identical code (the protocol's noise floor). It stays opt-in; `Spec.levers` defaults to `()`. `ao/` is the AO layer (milestone 4), host-side; its `ao` modules import neither `reference/` nor `plan/` (evaluators arrive by injection): `shell` (contracted Cartesian shells; primitive normalisation in one function), `hermite` (the undressed `E`), `pairs` (primitive-pair records, `dress` — the only place a pair factor or contraction coefficient is multiplied in — and the pair-factor-free Schwarz bound, its evaluator injected), `eri` (`(ab\|cd)` blocks), `j` (`J_µν` by two sweeps, the `jp=` seam, the `dq=` and `jab=` seams with their host references `dq_from_host` / `jab_from_host` (#151), and #88's screened sweep), `drive` (`KernelJP`: `J_p` from the interpreted generated kernel; `KernelDQ` and `KernelJAB`, the same for the two new kernels; and `coulomb_matrix_chain` (#155), `J` with every step a generated kernel, interpreted; its own `adapter` rule in `test_layering.py`, which lets it reach the builder, `interp` and `plan` but not `reference/` — Boys is injected), and `staged` (#143, the same rule: `J` from a launch plan, one `J_p` per bra class, the kernels interpreted with the real screen; since #155 its `ket_data=` and `transform=` hooks take the kets and the back-transform from the chain). `j`'s `Screened` carries `n_examined`, `max_skipped` and `dropped` since #142. `reference/boys.py` is Gauss-Legendre below `T = 125` and the measured large-`T` asymptote from there to `T_MAX = 2000` (#122), refusing beyond. |
| `src/felizjueves/plan/` | **live, host-side only.** Pipeline steps 1–3 of `[JEngine, v2, §6]`: `targets`, `batching` (the minimax batch split), `mild`, `dag` (one MILD DAG per batch), `sieve` (Alg. 3, off with the `no-sieve` lever), `cost` (`PlanCost`, Tables 1–3's quantities), composed by `pipeline.plan(spec)` with a JSON per-batch report. `PYTHONPATH=src uv run python -m felizjueves.plan replay literal` scores a board (exit 0 pass, 1 FAIL, 2 could not run, 3 UNSET/no candidates — `literal` exits 3 today, Tables 1–2's `exact_given_inputs` being refused); `report L n_batches` prints one plan. It counts; it emits no kernel text and compiles nothing. `recovery` recovers the inputs behind the printed cells as rules — the cut (Table 3, 40/40) and, unrecovered, the seed order (`seed_orders(widened=True)`, 58 rules, #108); `search` is the searched board's per-cell cut and per-batch seed-order permutations, climbed by `order_search` (#170, #172: the Pareto frontier, the pruned per-batch product and a seeded hill climb from the 58 rules) and selected by #171's declared rule; `pipeline.tuned_plan` builds a `tuned` Spec from that record. `optimum` (#106) is the exact `(intermediates, FLOPs)` Pareto frontier of a one-batch class by DP over Boys layers, for `L <= EXACT_MAX_L = 5` (above it `BeyondExactCeiling`, with the measured reason), and `beam` an achievable scheme above it — an upper bound on the optimum only; nothing in the pipeline imports it. `schedule` (step 4: a topological order of one batch's DAG, not text) also lives here, and is what `recurrence.py` and `kernel.py` consume. |
| `data/` | **live.** `literature/jengine/table{1,2,3}.json` is `[JEngine]` Tables 1–3, 120 cells with provenance. `scoreboards/jengine/{literal,searched}.json` are the two boards. On `literal` every Agreement floor is set to what the plan layer measured (exact cells: Table 3 38 on the minimax reconstruction, Tables 1–2 post-sieve 7 and 15); Table 3's `exact_given_inputs` records its recovered cut and passes 40/40, and Tables 1–2's are **refused** under the rules-only doctrine (#108: the best of 58 seed-order rules reproduces 33/40 and 38/40, and a per-cell fit is not admitted), which scores UNSET. `searched` replays `plan/search.py`'s recorded cut and per-batch seed-order permutations per cell (`searched_cuts.json`, schema 3). Since #172 it PASSes `at_or_under` on Tables 1, 2 and 3 in all 40 cells: 22 land on the printed triple exactly and 18 come in under it. Each order is climbed from the 58 rules within a bounded neighbourhood, and the plan is chosen from the frontier by #171's selection rule. That is a choice the board makes, not a recovered input. `literature/jengine/limits.json` holds seventeen `StructuralLimit`s: the evidence for the two refusals, the searched board's record (the space it searched, no residual, and the earlier spaces' residual kept as `previous_residual`), `[JEngine, v2, §4.1]`'s comparison as not reproducible here (#111), what #114's ridge answer cannot see, #147's sort-key verdict (contradicted on both cards, two cells unresolved), and #148's three: the `Launch` IR and `LaunchCost` as deliberately absent, the record-versus-shell-pair screen granularity as a reading of Alg. 1 (not a defect), and what the on-card build does not reach (the device `fj_boys` witnessed on `T ∈ [0, 500]` and overflowing near 700, fixtures only to d, so the dispatched cell only ever chooses 1; the per-pair path untimed; gfx1200's slow upload unexplained), and #158's three: the `DQ` and `JAB` kernels as this repo's own design (the paper defers them to `[Fock]`, not read), the instruction form's reach (`E` fits `Weighted`; four work-arounds where the IR did not reach), and what the generated chain on the cards does not reach (`J` through d only, nothing measured, the `(23)` `DQ` program's ~7.2 GB of the 3080's VRAM), and #160's three: the seed order's spread over the 58 rules (`seed_order_spread.json`), and Alg. 3's re-expansion direction and deletion reach as reconstruction choices, and #164's: the sieve's spill pattern on our cards (non-monotone on both, but the `[5|6]` stand-in spills less on both and the mean falls on gfx1200 only, so the published pattern fails on both and cannot be the paper's witness), and #166's: the refusal to ship the tuned plan (the literal plan stays, on measured wall-clock: `tuned_may_become_the_default` is False and the search NOT SHOWN on both cards). `literature/jengine/claims.json` is the registry of `[JEngine]`'s non-numeric claims — 32 records: 18 asserted, 14 unverifiable, 0 pending; each asserted one names a test that fails when the claim is false, each unverifiable one its mechanism (`j_engine.md`, "The `[JEngine]` claims", rendered from the file and checked by `test_claims.py`) — and `coverage.json` (#120) is the unit-by-unit reading of the paper (34 units) it is the inventory of. `mild_vs_optimum.json` (#111) is MILD's gap against `plan/optimum.py`'s exact optimum: equal through L = 4, behind on both axes at L = 5 (+6.25% intermediates, +0.625% FLOPs) — a disagreement with §4.1, recorded as the result — with a beam's lower bound above. Table 4 (spills) is deliberately not transcribed; `data/literature/jengine/README.md` says why. `measured/kernels.json` holds 196 `Measured` rows — registers and spills as nvcc (`-Xptxas -v`, bytes) and hipcc (`kernel-resource-usage`, register counts) report them on `sm_86` and `gfx1200`, keyed on `MeasuredKey.key()`, rewritten by `python -m felizjueves.measured --remeasure`: 66 literal rows (one per one-batch class L = 0..16, and one per cell of `measured_store.SUBGRID`, L ∈ {7, 10, 13, 16} × `n_batches` ∈ {1, 3, 6, 9, 12}, aggregated over its batch kernels, per arch), 40 `.../recycle@...` rows (#55: the recycle arm of every subgrid cell on both arches, each with its same-session literal control in raw `wallclock:control:*`; `measured_store.lever_ab` is the verdict, the store's header `levers` the conclusion), and 16 `.../asym@...` rows (#113: L ∈ {13, 16} × n ∈ {4, 5, 7, 8} on both arches, each timed against its legal neighbours in one session — every cell WORSE), 42 `.../no-sieve@...` rows (#162: the subgrid plus L = 11, one batch, on both arches, `arm_kind` `plan`, each with its same-session literal control; `sieve_verdict` NOT SHOWN on both) and 32 `.../tuned@...` rows (#163: the 16 subgrid cells with a recorded choice on both arches, the four 12-batch cells not measurable; `tuned_verdict` NOT SHOWN on both, so the generator ships literal, #166). The 40 literal subgrid rows (20 per arch) also carry **wall-clock**: median, IQR and 31 repeats on a wide-T input (`runtime_ns`) and a near-T one (raw `wallclock:near:*`), re-timed by #93 behind the card lock and utilisation gate, one session per arch, on host `1da02cac6d4e (container)` with the RTX 3080 (`nvcc 13.0`) and the RX 9060 XT (`hipcc / ROCm 7.2.4`); no cell dropped. Each such row records the card's state in raw `wallclock:gpu_util_before` / `_after` (and `wallclock:near:...`); the `recycle` rows were timed earlier (#55, host `ddbbc02dbb08 (container)`) and carry `null` there — unknown, not idle (the store's header `gpu_util`). Outside the subgrid nothing is timed. `measured/ceilings.json` (#109) is each card's measured fp64 rate and triad bandwidth, behind the gate: `sm_86` 4.39e11 flop/s and 6.74e11 B/s (ridge 0.65 FLOP/B), `gfx1200` 4.46e11 and 3.00e11 (ridge 1.49) — achievable rates, never datasheet peaks; the vendors' published figures are a sanity bound only. `gfx1200`'s fp64 is **settled** (#141): single issue, bounded per counted GPU cycle (128 flop/clk, `GRBM_GUI_ACTIVE`; 0.40 TFLOPS at the rated 3130 MHz boost), and the recorded 4.46e11 is 111% of that because the card ran at a counted ~3.49 GHz; the ridge reads the recorded value only. #141's session also re-measured gfx1200's triad 1.7% lower (2.947e11) and did not commit it. `measured/roofline.json` (#112) holds 124 counter rows, one per batch kernel of every subgrid cell on `sm_86` (`ncu` 2025.3.1.0, the card's state in each row); `gfx1200` has **none**, and since #140 that is **final** (`measured_store.UNMEASURED`): `rocprofv3` now loads, but gfx12 defines no fp64 counter and the GL2C byte counters read 0 on this box. It is recorded as absent, never as a zero or a "no". The answer (#114): on `sm_86` batching crosses the ridge only at L = 16, at 3 batches; L = 7, 10 and 13 are compute-bound already at one batch. `measured/limits.json` records where `KernelCost` mis-ranks those rows (#54, #60: four records) and #115's four kinds of cross-read disagreement (`asserted_value.cross_read = true`); `measured/crossread.json` is #115's table per `(L, arch)`; `measured/batch_recommendation.json` is the best-median `n_batches` per `(L, arch, input)` with its spread — wide-T picks 1 everywhere, near-T splits from L = 13 and the cards disagree at L = 16 (`j_engine.md`, "Measured"), which is why the generator ships one batch by decision (its `about`, from `batchreport.ABOUT`). `measured/plan_vs_kernel.json` (#165) is `planrank`'s report: `PlanCost` against spills and same-session clock per card, every inversion named. `literature/jengine/seed_order_spread.json` (#160) is the seed order's spread over the 58 rules per cell, and Alg. 3's three open choices with the tests that pin them. `measured/builds.json` (#146, #147) holds 22 whole-build rows, 11 per card, on host `e614efd48a48 (container)`, each with its gate readings (null refused): 3 `baseline` cells (H₂O/STO-3G at τ 0; H₂O/cc-pVDZ dispatched at τ 0 and 1e-3) and #147's 8 `sort-ab` rows per card (two arms × H₂O/cc-pVDZ and the two-H₂ chain × τ 1e-3 and 1e-10, one session per card). The timed region is in its header (zero, launch, synchronize, read back; uploads, eq. 20 and the scatter outside). Every row predates milestone 7, so none times a `DQ` or `JAB` kernel. #147's verdict: the global-`max|D|` key skips no pair anywhere, yet it is faster at some cell on each card, so the expectation is CONTRADICTED on both. Observations of a toolchain and two cards, **never compared against Table 4** or the paper's A100. `ao/screening.json` is #88's `τ` ladder (error and skip fraction per rung, compact H₂O and an extended chain of four H₂ since #122 — as long as `reference.boys`' `T ≤ 2000` allows), rewritten by `test/test_ao_screening.py --write` and pinned by its `slow` test. The libcint fixtures themselves live in `test/oracle/fixtures/` (13 files, HF/cc-pVTZ's ERI blocks split over five of them by L), with provenance. |
| the generator | **live, for the whole `J` chain.** `KERNEL_GEN_CMD` in `devtools/config.sh` is `env PYTHONPATH=src python -m felizjueves.emit`, which writes **one ladder per kernel family** (#156) in the backend's spelling (`emit/cuda.py` / `emit/hip.py`). With no ladder flag it writes 37 units: the `[p\|q]` ladder (`--L`, default L = 0..16), `build_pq_kernel(L)` — the `[p\|q]` kernel of `[JEngine, v2, Alg. 1]`, default split, **one batch, no atomics**, `pq_L{L}_p{lp}_q{lq}`, byte-identical to before #156 and pinned by `kernels.json`'s `source_sha256` — and the pair ladder (`--pair-l LMAX`, default 3, accepted 0..8), `dq_{la}_{lb}` and `jab_{la}{lb}` for every `la ≤ lb ≤ LMAX` (20 units through (ff)). `--L` alone writes only `[p\|q]`, `--pair-l` alone only the pairs. The emitter casts a scalar parameter the body never reads to `(void)` (`DQ`'s `n_pair`, `JAB`'s `n_rows`), because hipcc's `-Wunused-parameter` otherwise fails #92's filter. The kernel tier generates, compiles all 37 units with nvcc or hipcc, and its device phase (`-m "gpu and <backend>"`) runs, among the rest, `test/test_kernel_device.py` (`J_p` and `fj_boys` on the card against `reference/` in fp64) and `test_j_chain_ondevice.py` (the pair units the tier wrote, and `J` through the whole chain). **Split classes** are built by `build_pq_kernels(spec)` — one kernel per batch, `J_p` by atomic add — but are **not** written by `KERNEL_GEN_CMD`: the tests emit them (`test_emit_batched.py`, goldens for L = 4, `n_batches = 3`), and the device phase runs the subgrid's split cells and #113's asymmetric cells. **The last full kernel tier on both cards was #157's (PR #210):** 96 passed / 5 skipped on CUDA (RTX 3080), 97 passed / 5 skipped on HIP (RX 9060 XT), the skips the five opt-in recordings (ceilings, `[p\|q]` wall-clock, roofline counters, build wall-clock, sort-key A/B). **The class selector** (#89) is a second, separate CLI path: `python -m felizjueves.emit --class LA LB LC LD` (or `--shell-ls l...`, with `--n-batches N...` and `--admitted-only`) writes, per cell, the whole chain of a real class `(la lb\|lc ld)` a basis reaches, with an `order` file in launch order: `dq_{lc}_{ld}`, the `[p\|q]` batch kernels, `jab_{la}{lb}` (#156); `KERNEL_GEN_CMD` does not call it. `test_ao_j_ondevice.py` compiles the `[p\|q]` slice of those cells in the device phase and builds `J_µν` for H₂ and H₂O/STO-3G on each card against libcint — #89 (PR #118): worst `5.09e-15·scale_J` against a `1e-12` tolerance. **The launch-plan build** (#144) is a third path: `test/plan_driver.py` writes a device `main()` from a `launch.LaunchPlan` that `#include`s each class's emitted units, and `test_j_launched_ondevice.py` compiles and runs it on each card (worst `1.24e-14·scale_J` against libcint over the unscreened cells, identical on both cards). With `chain=` (#157) the same driver adds one `JAB` launch per bra class after every `[p\|q]` launch, and a separate `DQ` program (`plan_driver.dq_program`) supplies its kets: `J` through d with every arithmetic step generated, worst `1.24e-14·scale_J`, the same on both cards. What is **not generated**: the pair list and each record's inputs, the Schwarz bound, the sort and its `Dmax` read-back, the staging, and the launch code (the driver is test-side). The new kernels are **not measured** (no registers, spills, wall-clock or counters). Dispatch is host-side (`dispatch.py`) and reaches only a launch plan. The shipped kernels come from the **literal** plan, by decision: the searched order reaches `build_pq_kernel` only through the opt-in `tuned` lever, which was measured on both cards and refused as the default on wall-clock (#166). One batch is a decision too (`batchreport.ABOUT`); another count goes through the selector, `--class ... --n-batches 1 3 6` (legal: 1 or a positive multiple of 3), and `--n-batches` on the ladder path is rejected with exit 2. The last device sessions per card are #162's and #163's A/B device phases (PR #222: 83 passed / 2 skipped on each card; PR #224: 99 passed / 2 skipped on each card); #225 re-runs the full tier. Set `KERNEL_GEN_CMD=` and the tier reports generate/compile DORMANT — do not read that as green. |
| `pyproject.toml` | **a virtual uv project** — no `[build-system]`, nothing installed. `src/` is importable from pytest only, via `pythonpath = ["src"]`. Dev tooling plus numpy, and a lock. |
| `docs/` | `references.md`, plus the derived notes in `implementation/`: `j_engine.md` (recurrences, the `E` transform, kernel contract, batching, the measured sections — batch count, asymmetric split, ceilings, counters, roofline rows, the ridge answer, the cross-read, and "The roofline, in one place" — pipeline, validation targets, "The `[JEngine]` claims" status table, the outside witness of `J`, the `τ` ladder, and "The launch plan": the sort key, the record granularity, the residency shape, the on-card results, the build wall-clock, the sort-key A/B, and the parity statement — and "The J chain as device code": which steps are generated, the `DQ` and `JAB` contracts, what stays host-side, and where the chain ran) and `mild.md` (the MILD heuristic, the sieve, and MILD against the computed optimum). Both are **specifications** the code is read against, and each opens with a status block saying how much of it is implemented. Batching, MILD, the sieve, the per-batch DAG, the schedule, `PlanCost` and the optimum are implemented host-side in `plan/`; the recurrences (`recurrence.py`) and the `[p\|q]` kernel and its emission (`kernel.py`, `emit/`) are implemented for one batch and, with atomic write-back and per-batch launches, for a split class. `j_engine.md`'s batch-count and cross-read tables are kept equal to their data files by `test_batchreport.py` and `test_kernelrank.py`, and its claims table by `test_claims.py`. The `E` transform, the pre-contraction, the bra back-transform and the `J_ab` scatter are implemented host-side in `ao/` and witnessed against libcint, and since milestone 7 also generated (`e_fragment.py`, `kernel_dq.py`, `kernel_jab.py`) and run on both cards. The launch plan, its oracle, the staged driver and dispatch are implemented (milestone 6). Not implemented: a device-resident `J`, launch code in the generator, any measurement of the new kernels, and dispatch inside the generator. Tracked. |

When you run something, say which of these it touched.

## Setup

```bash
uv sync                                   # creates .venv from uv.lock
pre-commit install                        # commit-time lint
pre-commit install --hook-type pre-push   # the fast test gate
devtools/doctor.sh                        # what is degraded, and the fix for each
```

`uv` is the only assumed host tool, and **`uv.lock` is the only place Python
dependency versions live** — after editing `pyproject.toml`, run `uv lock` and
commit the result; never `uv pip install`. Both images install with
`uv sync --frozen`, which fails rather than re-resolving, so a stale lock breaks
the image build instead of letting the container drift from the host.

The device toolchains live in the images, not in the repo's assumptions — though
on THIS host the CUDA 13.0 toolkit happens to be installed natively, so
`--backend cuda` works without a container and `--backend hip` does not.

## The tiers, and what each one proves

| Tier | Command | What a green run means |
|---|---|---|
| lint | `ruff check .` | `E,F,I,UP,B`, no formatter. |
| fast | `pytest` | the Python side agrees with itself — and, through tracked fixtures, with libcint. Compiles nothing, launches nothing, needs no PySCF. |
| kernel | `devtools/kernel-tier.sh --backend cuda\|hip` | generate → compile with the real device compiler (a warning located in generated text fails the compile phase, exit 1 — #92) → run on the card → compare against the host reference, and against libcint for `J_µν` (`test_ao_j_ondevice.py`). |
| oracle | `uv sync --group oracle`, then `pytest -m oracle` | the tracked fixtures in `test/oracle/fixtures/` still equal what PySCF computes today. Without the group these tests skip; a drift is news about PySCF until proven otherwise. |

**The kernel tier is the only place a numerical claim can be made.** No hosted
runner has a GPU. Two consequences worth internalising:

- **A device compiler needs no card.** nvcc and hipcc target an *architecture*,
  so the whole compile phase is green on a device-less box and proves nothing
  about a number.
- **A `gpu`-marked test SKIPS rather than fails** when no card is present, so a
  device-less run is green *and empty*. `devtools/doctor.sh` exists to make that
  visible, and `kernel-tier.sh`'s report prints `device ABSENT -- numbers
  unproven`.

`kernel-tier.sh` exits **2** for "could not run" (no compiler, no image, no
device group). Never read a 2 as a pass.

## What is gated, and what is not

- **`git push`** runs the fast pytest tier, and only when the push touches a
  `.py` (the `files: \.py$` filter on the pre-push hook).
  `FAST_TEST_ARGS` in `devtools/config.sh` excludes the `cuda`/`hip`/`gpu`
  markers, so the gate cannot start invoking a device compiler — which it
  otherwise would, inside a container where both are present — and `slow`
  and `oracle` too.
- **The card is locked by the tests, not by a wrapper.** Every `gpu` test's
  device fixture (`test/conftest.py`, #91) takes an exclusive `flock` on the
  card at `GPU_LOCK_PATH` in `devtools/config.sh` (`/tmp/fj-gpu-{backend}.lock`)
  — under xdist the controller holds it — and a harness that carries the
  VRAM probe checks free memory before its launch, failing with both figures
  rather than running out of memory. Two
  runs on one card serialise; an outer `flock` on the same path is recognised
  as an ancestor and is redundant, not harmful.
- **The kernel tier is deliberately NOT a hook.** An unrolled high-L kernel is
  slow to compile and the tier costs minutes; a gate that costs minutes is one
  people bypass with `--no-verify`. Run it before a PR and say which backend.
- **There is no CI yet.** No `.github/` in this repo. When it lands it can lint,
  run the fast tier, and compile — never run a kernel.
- **`protect-main.py` guards the primary (`main`) checkout** and is on by
  default. Set `CLAUDE_ALLOW_MAIN_EDITS=1` for a session deliberately editing it
  (repo setup, say); the guard's docstring has the rationale, and every bypass
  is logged to `.logs/protect-main.log`.

## Containers: four files, one diamond

```
                Dockerfile.base
                 /           \
  Dockerfile.cuda             Dockerfile.hip
            |                       :
  Dockerfile.combined ..............:  (reuses install-rocm.sh, not the hip image)
```

| File | What it is |
|---|---|
| `docker/Dockerfile.base` | g++ and clang, pinned CMake/Ninja/sccache/`gh`, a uv venv at `/opt/venv`. No GPU SDK. |
| `docker/Dockerfile.cuda` | `base` + the CUDA toolkit, `sm_86`, Nsight Systems. **The default.** |
| `docker/Dockerfile.hip` | `base` + ROCm's HIP compiler and profilers, `gfx1200`. No CUDA. |
| `docker/Dockerfile.combined` | `cuda` + ROCm. Both device compilers. |

The files chain by **tag**, not by stage — each child opens `FROM
${PARENT_IMAGE}` — so **build them only with `docker/build.sh
<base|cuda|hip|combined>`**, which walks the chain. Build a child by hand with
no parent tagged and docker tries to *pull* it and fails with `pull access
denied`.

**What is deliberately absent, because this tooling was adapted from a dense
linear algebra project:** no reference LAPACK, no cuTENSOR/NCCL/nvCOMP/cuGraph,
no hipCOMP, no clang-20 + libc++ module manifest — and ROCm is installed as the
**compiler plus profilers**, not as `rocm-hip-sdk`'s ~19GB of math libraries
that no generated kernel calls. **sccache is the one thing kept against that
grain**, because it caches nvcc and hipcc output and the kernel tier recompiles
the same generated translation units across runs. Add a library only with the
code that links it, and say what does.

**Claude Code is pinned in the three GPU files**, each ending with
`docker/install-claude-code.sh`. The version is therefore declared three times —
bump it with `devtools/claude-version.sh --apply`, which rewrites all three;
`doctor.sh` reports when they disagree or the pin is behind the registry.

`docker/README.md` has the build args, the compose services and the AMD
passthrough; the **devbox** skill drives it.

## Conventions

- **Python ≥3.12, `from __future__ import annotations` everywhere.** 3.12 and
  not 3.13 because that is the system interpreter on both sides (this host and
  `ubuntu:24.04`); raising it makes `uv sync` download a managed CPython in the
  image, which `UV_PYTHON_INSTALL_DIR` exists to keep reachable by uid 1000.
- **Lint is narrow (`E,F,I,UP,B`) with no formatter hook**, on purpose: the
  output that matters is generated kernel text, and a formatter arguing with the
  generator over its own string layout is noise. `ruff check .` is clean.
- **Generated kernel text that a test asserts against is a GOLDEN FILE** — it is
  tracked, reviewed, and a diff in it is the whole signal when the generator
  changes. There is no `*.cu` line in `.gitignore` for that reason; scratch
  emission goes to `.kernel-tier/`.
- **The markers separate the two axes**, and the distinction decides what a run
  proves: `cuda`/`hip` need a *compiler*, `gpu` needs a *card*. A test asks for
  exactly the fixtures it needs (`test/conftest.py`), so it skips for the right
  reason. Two more: `oracle` needs PySCF (to *regenerate* a fixture — a
  comparison against a tracked fixture carries no marker), and `claim(id)`
  ties a test to a record in `data/literature/*/claims.json`.
- **The host oracle is numpy, in the device's precision.** Computing a float32
  reference in float64 and then loosening the tolerance to match measures
  something other than the kernel.
- **No `-use_fast_math`, and nothing that reassociates arithmetic**, in
  `NVCC_FLAGS`/`HIPCC_FLAGS`. The tier's only assertion is agreement with the
  reference to a tolerance; a flag that changes the arithmetic changes the thing
  being measured. Add one per experiment, never to the defaults.
- **`devtools/config.sh` is the one file that knows the project.** The scripts
  are meant to survive being copied into the next repo unchanged — put a new
  setting there, not in a script. Two things it cannot reach and that must be
  kept in step by hand: the `.devcontainer/*/devcontainer.json` files, and the
  `FJ_` environment prefix spelled in `devcontainer.sh` and in those files.
- **Install the pre-commit hook from whichever side you commit on.** It bakes an
  absolute `INSTALL_PYTHON` into one file shared by every worktree; installed
  from the wrong side, commits fail with `No module named 'pre_commit'`. The
  **doctor** skill has the fix.

## Skills

Most of the how-to lives in `.claude/skills/`, and the detailed rationale lives
in the code it describes — script headers (`--help` prints them), `README.md`
files, and `docs/`. This file is the map; reach for the skill when you act.

| Skill | Reach for it to… |
|---|---|
| [test](skills/test/SKILL.md) | run the suites the right way and read the result honestly — the fast pytest tier, the kernel tier on each backend, the marker/fixture split, and what a green run does *not* prove. |
| [devbox](skills/devbox/SKILL.md) | drive the containers and the container-backed **sibling** worktrees — `devcontainer.sh` (up/rebuild/shell/test/down) and `worktree.sh` (add/rm/sync/gc). |
| [worktree](skills/worktree/SKILL.md) | create and clean up the lightweight `.claude/worktrees/<name>` checkouts — no container, cheap and disposable. |
| [doctor](skills/doctor/SKILL.md) | diagnose a degraded environment — `doctor.sh` plus the failures it cannot see. |
| [pr](skills/pr/SKILL.md) | ship the current work: commit → push → PR → merge, with this repo's gotchas. |
| [issue](skills/issue/SKILL.md) | resolve one GitHub issue end to end in a fresh worktree, tested on **both** cards, then `/pr full`. |
| [milestone](skills/milestone/SKILL.md) | turn a plan into a GitHub milestone plus a DAG of PR-sized issues. |
| [milestone-run](skills/milestone-run/SKILL.md) | execute a milestone with one subagent per ready issue. |
| [audit](skills/audit/SKILL.md) | verify that documentation (skills, READMEs, this file) still matches reality — extract each claim, check it, report drift. |
