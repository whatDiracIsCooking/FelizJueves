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

**The generator is live for one kernel family, and only one:** the `[p\|q]`
kernel (the `the generator` row) — one batch, and, through
`build_pq_kernels`, a split class (`n_batches > 1`) as atomic batch kernels.
**Around it, `J_µν` for a real molecule is built host-side** (`src/felizjueves/ao/`:
the `E` transform, pair data, the ket pre-contraction, the bra back-transform
and the scatter, all pure Python) and **witnessed against PySCF/libcint**, with
`J_p` from `reference/`, from the interpreted generated kernel, or from the
generated kernel compiled and launched on both cards (milestone 4,
`j_engine.md`, "The Coulomb matrix, witnessed from outside"). Only `J_p` is
generated; the `E` transform and the scatter are not device code, and dispatch
(choosing `n_batches` per class; `Spec.n_batches` defaults to 1) is still
specification. Every tier
reports what it actually touched rather
than implying more: a `gpu`-marked test skips without a card, so a run can be
**green and empty**; `kernel-tier.sh` exits **2** in that case and whenever it
could not run — read a 2 as "could not run", never as a pass.

| | State |
|---|---|
| `docker/`, `.devcontainer/`, `devtools/` | **live.** Both toolchains build; `devtools/kernel-tier.sh` runs end to end on this box. |
| `test/` | **live, and thin.** `test_toolchain.py` asserts nvcc and hipcc compile for this box's targets and that a kernel launched on each card matches a numpy reference, plus the counter witnesses (`ncu` reads a probe's FLOPs and bytes on CUDA; the `rocprofv3` one SKIPS in this container, which cannot load `libdw.so.1`). `test_kernel_device.py` (`gpu` + `cuda`/`hip`) runs the **generated** `[p\|q]` kernel and `fj_boys` on each card against `reference/`, plus every split cell of `measured_store.SUBGRID` as its batched launch sequence, plus #113's asymmetric study cells, plus opt-in sweeps that skip unless their store is set and run alone: wall-clock (`FJ_WALLCLOCK_STORE`; `FJ_WALLCLOCK_ASYM=1` for #113's arms) and roofline counters (`FJ_ROOFLINE_STORE`, `ncu` on CUDA; the hip test skips with `measured_store.UNMEASURED`'s reason). `test_ceilings_device.py` (`gpu`) bounds each card's probe by the vendor's published peak and, opt-in (`FJ_CEILINGS_STORE`), records the ceilings. `test_measured.py` checks the measured store and its driver (its `cuda`/`hip` cases recompile and compare; the rest is fast: the byte-for-byte round trip of `kernels.json` and `roofline.json`, the counter-row shape and its loader's refusals, the ncu/rocprofv3 parsers on recorded or synthetic CSV, the declared verdicts). The rest is fast-tier: the `Spec` key, the transcribed tables' structural invariants, the scoring rules on **synthetic** candidates, the batch kernels interpreted against `reference/` (`test_kernel_batched.py`), the goldens (`test_emit_*`), `test_plan_optimum.py` (the exact optimum against an unpruned brute force, the beam as an upper bound), `test_mild_vs_optimum.py` (#111's gap; beam L = 11..16 `slow`), `test_plan_recovery.py` (the widened seed-order search, `slow`), `test_eq17_witness.py` (eq. 17's `1y`, broken by the misprint), `test_ceilings.py` and `test_device.py` (the ceiling records' provenance, the gate's refusals with injected readers, the refusal to score an uncharacterised card), `test_ridge.py` (#114's rule on synthetic rows and its claim on the tracked ones), `test_kernelrank.py` and `test_batchreport.py` recomputing `data/measured/{limits,crossread,batch_recommendation}.json` from the stores, and `test_layering.py`'s import allow-list. **The AO layer and its oracle** (milestone 4): `test_ao_{shell,hermite,pairs,eri,j,j_interpreted,screening}.py` test `ao/` — `E` against closed-form Hermite polynomials, the Schwarz bound proved over randomised geometry, and the ERI blocks (all 55 classes of HF/cc-pVTZ since #121, split across five fixtures by L) and `J_µν` (by `reference/` and by the interpreted generated kernel) against PySCF/libcint through **tracked fixtures** in `test/oracle/fixtures/` (`test/oracle/`: `generate.py`, `manifest.py`, `basis.py`; `test_oracle_{basis,manifest}.py`, the overlap as the cheapest witness of libcint's conventions). They run without PySCF — s/p in the fast tier, `d`+ and the larger `J` cases `slow`; only the `oracle`-marked drift tests need the `oracle` group, and they skip without it. `test_ao_j_ondevice.py` (`gpu` + `cuda`/`hip`) builds `J_µν` with `J_p` from the class selector's kernels on each card against the same fixture. `test_claims.py` checks `data/literature/jengine/claims.json` against the `claim`-marked tests, both ways, `coverage.json` against the records, and `j_engine.md`'s status table and every claims tally a doc quotes against the file. |
| `src/felizjueves/` | **importable, and small.** `spec.py` is `Spec`, the frozen key every later subsystem joins on. `literature/` loads published tables and scores a candidate against them: `tables`, `rules` (`Agreement`, `AtOrUnder`, `ExactGivenInputs` — which since #108 can carry a `refused` decision instead of inputs, scoring UNSET, never PASS), `scoreboard`, `limits`, `claims`. `plan/` (below) computes the Table 1–3 cells it scores. `kernel.py` builds the `[p\|q]` kernel as IR (`ir.py`, checked by `interp.py`); `emit/` spells it for each vendor (`python -m felizjueves.emit` is the generator CLI); `build_pq_kernels` builds a split class's batch kernels, each `passes.atomize`d. `device.py` holds `Arch` — with each card's measured `fp64_flops_per_s` and `mem_bandwidth_bytes_per_s`, read from `data/measured/ceilings.json` (`0.0` is *not characterised*, and `Arch.ridge` / `verdict` refuse such a card) — and `MeasuredKey`. `gate.py` is the card gate every timed, probed or profiled phase runs behind (card at most `GPU_BUSY_MAX` = 30% busy and CPU at most 10% before, and a settled reading after; non-idle or unreadable refuses). `ceilings.py` is the fp64-FMA and STREAM-triad probe pair (#109). Since #94 the measured layer is two modules: `measured_store.py` is the store — `Measured` rows, `SUBGRID`, the counter fields and per-batch-kernel counter rows (#110), the roofline point and verdict (computed, never stored, #112), and the two verdicts declared before their rows (`nine_vs_twelve`, `lever_ab`) — and `measured.py` is the driver that compiles, parses ptxas / AMD remarks and `ncu` / `rocprofv3` output, and runs `--remeasure` (and `--asym`, `--roofline`). `kernelrank.py` ranks `lower.KernelCost` against those rows (#54); `batchreport.py` derives the recommended `n_batches` per `(L, arch)` from wall-clock alone (#56); `ridge.py` is #114's crossing rule, transcribed from the issue that declared it before the rows; `crossread.py` reads model, clock and counters against each other per `(L, arch)` (#115: the choice is the clock's); `asym.py` is #113's asymmetric-split study path (its own layering rule; `Spec` and `partition` still refuse those counts). Levers: `no-sieve` (plan); `passes.atomize` (applied by the builder to every split class, not a choice); and `recycle` (`lower.RECYCLE`, slot recycling, threaded through `Spec.levers` and applied by `measured`'s driver, not by `KERNEL_GEN_CMD`). #55 A/B'd `recycle` over all 20 subgrid cells on both cards: nvcc 13.0 and hipcc / ROCm 7.2.4 emit identical machine code for both arms, so it is neutral — `sm_86` 20/20 neutral, `gfx1200` 16/20 neutral with 4 cells read as slower on that identical code (the protocol's noise floor). It stays opt-in; `Spec.levers` defaults to `()`. `ao/` is the AO layer (milestone 4), host-side; its `ao` modules import neither `reference/` nor `plan/` (evaluators arrive by injection): `shell` (contracted Cartesian shells; primitive normalisation in one function), `hermite` (the undressed `E`), `pairs` (primitive-pair records, `dress` — the only place a pair factor or contraction coefficient is multiplied in — and the pair-factor-free Schwarz bound, its evaluator injected), `eri` (`(ab\|cd)` blocks), `j` (`J_µν` by two sweeps, the `jp=` seam, and #88's screened sweep), and `drive` (`KernelJP`: `J_p` from the interpreted generated kernel; its own `adapter` rule in `test_layering.py`, which lets it reach the builder, `interp` and `plan` but not `reference/` — Boys is injected). `reference/boys.py` is Gauss-Legendre below `T = 125` and the measured large-`T` asymptote from there to `T_MAX = 2000` (#122), refusing beyond. |
| `src/felizjueves/plan/` | **live, host-side only.** Pipeline steps 1–3 of `[JEngine, v2, §6]`: `targets`, `batching` (the minimax batch split), `mild`, `dag` (one MILD DAG per batch), `sieve` (Alg. 3, off with the `no-sieve` lever), `cost` (`PlanCost`, Tables 1–3's quantities), composed by `pipeline.plan(spec)` with a JSON per-batch report. `PYTHONPATH=src uv run python -m felizjueves.plan replay literal` scores a board (exit 0 pass, 1 FAIL, 2 could not run, 3 UNSET/no candidates — `literal` exits 3 today, Tables 1–2's `exact_given_inputs` being refused); `report L n_batches` prints one plan. It counts; it emits no kernel text and compiles nothing. `recovery` recovers the inputs behind the printed cells as rules — the cut (Table 3, 40/40) and, unrecovered, the seed order (`seed_orders(widened=True)`, 58 rules, #108); `search` is the searched board's per-cell cut and seed order. `optimum` (#106) is the exact `(intermediates, FLOPs)` Pareto frontier of a one-batch class by DP over Boys layers, for `L <= EXACT_MAX_L = 5` (above it `BeyondExactCeiling`, with the measured reason), and `beam` an achievable scheme above it — an upper bound on the optimum only; nothing in the pipeline imports it. `schedule` (step 4: a topological order of one batch's DAG, not text) also lives here, and is what `recurrence.py` and `kernel.py` consume. |
| `data/` | **live.** `literature/jengine/table{1,2,3}.json` is `[JEngine]` Tables 1–3, 120 cells with provenance. `scoreboards/jengine/{literal,searched}.json` are the two boards. On `literal` every Agreement floor is set to what the plan layer measured (exact cells: Table 3 38 on the minimax reconstruction, Tables 1–2 post-sieve 7 and 15); Table 3's `exact_given_inputs` records its recovered cut and passes 40/40, and Tables 1–2's are **refused** under the rules-only doctrine (#108: the best of 58 seed-order rules reproduces 33/40 and 38/40, and a per-cell fit is not admitted), which scores UNSET. `searched` replays `plan/search.py`'s recorded cut and seed order per cell (`searched_cuts.json`) and FAILs `at_or_under` on Tables 1–2 (21 and 3 cells, all within +2.04%). `literature/jengine/limits.json` holds five `StructuralLimit`s: the evidence for the two refusals, the searched residual with each cell's margin, `[JEngine, v2, §4.1]`'s comparison as not reproducible here (#111), and what #114's ridge answer cannot see. `literature/jengine/claims.json` is the registry of `[JEngine]`'s non-numeric claims — 31 records: 18 asserted, 13 unverifiable, 0 pending; each asserted one names a test that fails when the claim is false, each unverifiable one its mechanism (`j_engine.md`, "The `[JEngine]` claims", rendered from the file and checked by `test_claims.py`) — and `coverage.json` (#120) is the unit-by-unit reading of the paper (34 units) it is the inventory of. `mild_vs_optimum.json` (#111) is MILD's gap against `plan/optimum.py`'s exact optimum: equal through L = 4, behind on both axes at L = 5 (+6.25% intermediates, +0.625% FLOPs) — a disagreement with §4.1, recorded as the result — with a beam's lower bound above. Table 4 (spills) is deliberately not transcribed; `data/literature/jengine/README.md` says why. `measured/kernels.json` holds 122 `Measured` rows — registers and spills as nvcc (`-Xptxas -v`, bytes) and hipcc (`kernel-resource-usage`, register counts) report them on `sm_86` and `gfx1200`, keyed on `MeasuredKey.key()`, rewritten by `python -m felizjueves.measured --remeasure`: 66 literal rows (one per one-batch class L = 0..16, and one per cell of `measured_store.SUBGRID`, L ∈ {7, 10, 13, 16} × `n_batches` ∈ {1, 3, 6, 9, 12}, aggregated over its batch kernels, per arch), 40 `.../recycle@...` rows (#55: the recycle arm of every subgrid cell on both arches, each with its same-session literal control in raw `wallclock:control:*`; `measured_store.lever_ab` is the verdict, the store's header `levers` the conclusion), and 16 `.../asym@...` rows (#113: L ∈ {13, 16} × n ∈ {4, 5, 7, 8} on both arches, each timed against its legal neighbours in one session — every cell WORSE). The 40 literal subgrid rows (20 per arch) also carry **wall-clock**: median, IQR and 31 repeats on a wide-T input (`runtime_ns`) and a near-T one (raw `wallclock:near:*`), re-timed by #93 behind the card lock and utilisation gate, one session per arch, on host `1da02cac6d4e (container)` with the RTX 3080 (`nvcc 13.0`) and the RX 9060 XT (`hipcc / ROCm 7.2.4`); no cell dropped. Each such row records the card's state in raw `wallclock:gpu_util_before` / `_after` (and `wallclock:near:...`); the `recycle` rows were timed earlier (#55, host `ddbbc02dbb08 (container)`) and carry `null` there — unknown, not idle (the store's header `gpu_util`). Outside the subgrid nothing is timed. `measured/ceilings.json` (#109) is each card's measured fp64 rate and triad bandwidth, behind the gate: `sm_86` 4.39e11 flop/s and 6.74e11 B/s (ridge 0.65 FLOP/B), `gfx1200` 4.46e11 and 3.00e11 (ridge 1.49) — achievable rates, never datasheet peaks; the vendors' published figures are a sanity bound only, and `gfx1200`'s fp64 one assumes fp32 dual-issue applies to fp64 (its `note` says what follows if it does not). `measured/roofline.json` (#112) holds 124 counter rows, one per batch kernel of every subgrid cell on `sm_86` (`ncu` 2025.3.1.0, the card's state in each row); `gfx1200` has **none** — could not measure (`measured_store.UNMEASURED`: `rocprofv3` cannot load in this container, and gfx1200 has no fp64 counter), recorded as absent, never as a zero or a "no". The answer (#114): on `sm_86` batching crosses the ridge only at L = 16, at 3 batches; L = 7, 10 and 13 are compute-bound already at one batch. `measured/limits.json` records where `KernelCost` mis-ranks those rows (#54, #60: four records) and #115's four kinds of cross-read disagreement (`asserted_value.cross_read = true`); `measured/crossread.json` is #115's table per `(L, arch)`; `measured/batch_recommendation.json` is the best-median `n_batches` per `(L, arch, input)` with its spread — wide-T picks 1 everywhere, near-T splits from L = 13 and the cards disagree at L = 16 (`j_engine.md`, "Measured"). Observations of a toolchain and two cards, **never compared against Table 4** or the paper's A100. `ao/screening.json` is #88's `τ` ladder (error and skip fraction per rung, compact H₂O and an extended chain of four H₂ since #122 — as long as `reference.boys`' `T ≤ 2000` allows), rewritten by `test/test_ao_screening.py --write` and pinned by its `slow` test. The libcint fixtures themselves live in `test/oracle/fixtures/` (13 files, HF/cc-pVTZ's ERI blocks split over five of them by L), with provenance. |
| the generator | **live, for one kernel family.** `KERNEL_GEN_CMD` in `devtools/config.sh` is `env PYTHONPATH=src python -m felizjueves.emit`, which writes `build_pq_kernel(L)` — the `[p\|q]` kernel of `[JEngine, v2, Alg. 1]`, default split, **one batch, no atomics** — for L = 0..16 in the backend's spelling (`emit/cuda.py` / `emit/hip.py`). The kernel tier generates, compiles all 17 units with nvcc or hipcc, and its device phase (`-m "gpu and <backend>"`) runs `test/test_kernel_device.py`: `J_p` and `fj_boys` on the card against `reference/` in fp64. **Split classes** are built by `build_pq_kernels(spec)` — one kernel per batch, `J_p` by atomic add — but are **not** written by `KERNEL_GEN_CMD`: the tests emit them (`test_emit_batched.py`, goldens for L = 4, `n_batches = 3`), and the device phase runs the subgrid's split cells and #113's asymmetric cells. The last full device phase reported on both cards was #112's (PR #134): 73 passed / 3 skipped on CUDA, 72 passed / 4 skipped on HIP — the skips the three opt-in recordings (ceilings, wall-clock, roofline) and, on HIP, the `rocprofv3` witness that cannot load `libdw.so.1`. **The class selector** (#89) is a second, separate CLI path: `python -m felizjueves.emit --class LA LB LC LD` (or `--shell-ls l...`, with `--n-batches N...` and `--admitted-only`) writes the batch kernels of the real classes `(la lb\|lc ld)` a basis reaches, with an `order` file per cell; `KERNEL_GEN_CMD` does not call it and its L = 0..16 output is unchanged. `test_ao_j_ondevice.py` compiles those files in the device phase and builds `J_µν` for H₂ and H₂O/STO-3G on each card against libcint — #89 (PR #118): 63 passed / 1 skipped per backend, worst `5.09e-15·scale_J` against a `1e-12` tolerance. What is **not generated**: the `E` transform, the ket pre-contraction and the `J_ab` scatter (host-side Python in `ao/`), and dispatch (nowhere). Set `KERNEL_GEN_CMD=` and the tier reports generate/compile DORMANT — do not read that as green. |
| `pyproject.toml` | **a virtual uv project** — no `[build-system]`, nothing installed. `src/` is importable from pytest only, via `pythonpath = ["src"]`. Dev tooling plus numpy, and a lock. |
| `docs/` | `references.md`, plus the derived notes in `implementation/`: `j_engine.md` (recurrences, the `E` transform, kernel contract, batching, the measured sections — batch count, asymmetric split, ceilings, counters, roofline rows, the ridge answer, the cross-read, and "The roofline, in one place" — pipeline, validation targets, "The `[JEngine]` claims" status table, the outside witness of `J`, the `τ` ladder) and `mild.md` (the MILD heuristic, the sieve, and MILD against the computed optimum). Both are **specifications** the code is read against, and each opens with a status block saying how much of it is implemented. Batching, MILD, the sieve, the per-batch DAG, the schedule, `PlanCost` and the optimum are implemented host-side in `plan/`; the recurrences (`recurrence.py`) and the `[p\|q]` kernel and its emission (`kernel.py`, `emit/`) are implemented for one batch and, with atomic write-back and per-batch launches, for a split class. `j_engine.md`'s batch-count and cross-read tables are kept equal to their data files by `test_batchreport.py` and `test_kernelrank.py`, and its claims table by `test_claims.py`. The `E` transform, the pre-contraction, the bra back-transform and the `J_ab` scatter are implemented host-side in `ao/` and witnessed against libcint, not generated. Still specification only: dispatch. Tracked. |

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
