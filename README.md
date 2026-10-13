# FelizJueves

Python code generation for electronic structure tensors.

## What is here

**A generator of the kernels of a Coulomb-matrix build, and that build.** The
generator emits, in CUDA and HIP, the McMurchie–Davidson `[p|q]` kernel of
`[JEngine]` and, since milestone 7, the kernels around it: the `E` transform
(inlined), the ket pre-contraction and the bra back-transform with its
scatter. A host-side AO layer turns a real molecule into their inputs and
remains the reference they are held to, and the `J` they build is compared
against PySCF/libcint — a program with no shared ancestry. Every tier reports
what it touched rather than implying more.

**In three questions:**

- **What does it generate?** Every arithmetic step of `J_µν`, in CUDA and HIP:
  the `[p|q]` kernels (`J_p = Σ_q [p|q] D_q`, one per `(lp, lq)` class and
  batch count), the pre-contraction `D_q` and `Dmax` (`Target.DQ`, one kernel
  per ket pair class) and the bra back-transform and scatter into `J_µν`
  (`Target.JAB`, one per bra pair class), with the undressed `E` coefficients
  inlined into the last two. What stays on the host is what decides
  launches: the pair list and each record's inputs, the Schwarz bound, the
  sort (which reads `Dmax` back), the staging and the launch code. The
  pre-contraction and scatter kernels are this repo's own design, because
  `[JEngine]` defers those steps to another paper (`limits.json`).
- **How is a whole molecule's `J` built?** From a **launch plan**
  (`launch.py`, milestone 6). It has one entry per `(lp, lq)` class, with bra
  records in pair order and ket records sorted by `I_q · max|D_q|`
  (`[JEngine, v2, §3.3]`). On the card, each bra class and each ket class is
  uploaded once, and every batch kernel is launched with fp64 atomics into
  one `J_p` buffer per bra class. Since milestone 7 the `DQ` kernels produce
  the kets' `D_q` and the `JAB` kernels turn each `J_p` buffer into `J_µν` on
  the card, launched from Python through a test-side driver. The plan's order
  is witnessed by an independent Python walker (the launch oracle) and by
  libcint. A class's batch count can be chosen from measured wall-clock
  (`dispatch.py`).
- **On which cards has that run?** The RTX 3080 (`sm_86`) and the RX 9060 XT
  (`gfx1200`). The whole chain built `J` for H₂, H₂O/STO-3G and H₂O/cc-pVDZ
  (s, p and d shells, `[p|q]` classes up to L = 8), within `1e-12·scale_J` of
  libcint (worst 1.24e-14). The pre-contraction and scatter kernels ran alone
  for every pair class through (ff), and the `[p|q]` kernels alone to L = 16,
  each against the host. Milestone 6's build, with `[p|q]` alone on the
  card, is timed on both cards (`data/measured/builds.json`). **Nothing about the new kernels is
  measured.** `docs/implementation/j_engine.md`, "The J chain as device code",
  is the one place with the detail.

Not built: a `J` kept on the card across SCF iterations, a `Launch` IR or
`LaunchCost` model (declined, `limits.json`), dispatch inside the generator,
and anything with gradients. Against the predecessor `jgen`, this is
parity on its energy-order `J` surface, with one difference: Table 4,
declined. Which plan ships is a decision, not a gap: `jgen` ships its
searched sieve order, and this repo has that wire too (the opt-in `tuned`
lever) but ships the **literal** plan, because the searched one was measured
on both cards and refused on wall-clock (#166;
`docs/implementation/j_engine.md`, "Parity with the predecessor").

`.claude/CLAUDE.md` ("The state of the tree") has the detail; in short:

| | State |
|---|---|
| `docker/`, `.devcontainer/`, `devtools/` | **live.** Both device toolchains build, and `devtools/kernel-tier.sh` runs end to end. |
| `test/` | **live.** Fast tier: the plan layer against `[JEngine]` Tables 1–3, the kernel IR interpreted against `reference/`, the golden kernel texts, and the AO layer against **tracked libcint fixtures** (`test/oracle/`), so PySCF is not needed to run it. Fast tier also: the launch plan, its oracle and the staged interpreted driver (`test_launch*.py`), and dispatch on synthetic and tracked rows. Kernel tier (`gpu`): the generated kernels on each card against `reference/`, `J_µν` for H₂ and H₂O against libcint, the whole molecule built from the launch plan on each card against the oracle and libcint (`test_j_launched_ondevice.py`), and the generated J chain on each card: every pre-contraction and scatter kernel through (ff) against the host, and `J` through d with every arithmetic step a generated kernel against libcint (`test_j_chain_ondevice.py`). Fast tier also: the `E` fragment and the two new kernels interpreted, bitwise against the host (`test_e_fragment.py`, `test_kernel_dq.py`, `test_kernel_jab.py`), and the whole chain interpreted against libcint (`test_ao_j_chain.py`). Last full kernel tier, PR #210 (#157): CUDA, RTX 3080, 96 passed / 5 skipped; HIP, RX 9060 XT, 97 passed / 5 skipped; the skips are the five opt-in recordings (#225 re-runs the full tier). The last device sessions per card since then are the plan-lever A/Bs' device phases (`test_kernel_device.py` and `test_measured.py`): #162 (PR #222), 83 passed / 2 skipped on each card, and #163 (PR #224), 99 passed / 2 skipped on each card, the skips the opt-in wall-clock and roofline sweeps. Fast tier also: the plan's output pinned across hash seeds (`test_plan_determinism.py`), the per-batch order search (`test_plan_order_search.py`), the sieve's spill reading (`test_sieve_spill_claim.py`), the plan-layer counts against the measured kernels (`test_planrank.py`), and the shipped plan held to the literal rows (`test_shipped_plan.py`). |
| `src/felizjueves/` | **importable.** `plan/` (batching, MILD, the sieve, the per-batch counts, `optimum.py`, the exact tree-search optimum for L ≤ 5, `search.py` and `order_search.py`, the searched board's per-batch seed orders, and `pipeline.tuned_plan`, the opt-in `tuned` lever that builds from them), `planrank.py` (whether the plan-layer counts predict the measured kernel: on neither card), `kernel.py` + `emit/` (the generator), `e_fragment.py`, `kernel_dq.py` and `kernel_jab.py` (the `E` fragment and the pre-contraction and scatter kernels, milestone 7), `measured_store.py` (the measured store: registers, spills, wall-clock on both cards; roofline counters on `sm_86`) and `measured.py` (the driver that compiles, parses and profiles into it), `gate.py` (the card gate every timed or profiled phase runs behind), `ceilings.py` (each card's measured fp64 and bandwidth ceilings), `ridge.py` (whether batching crosses our cards' ridge, under the rule #114 declared before the rows), `crossread.py` (model, clock and counters read against each other), `asym.py` (the asymmetric-split study), `literature/` (published tables and claims, and the rules that score against them), `ao/` (shells, the host `E` transform, pair data and the Schwarz bound, ERI blocks, `J_µν` with its `jp=`, `dq=` and `jab=` seams, `staged.py`, `J` from a launch plan with the kernels interpreted, and `drive.py`'s `coulomb_matrix_chain`, `J` with every step a generated kernel, interpreted), `launch.py` (the whole-molecule launch plan as a value, and the launch oracle that walks it in Python) and `dispatch.py` (a class's batch count from measured wall-clock, injected into a plan). |
| `data/` | **live.** `[JEngine]` Tables 1–3 with provenance and the two scoreboards over them; the registry of the paper's non-numeric claims (`claims.json`, 32 records: 18 asserted, 14 unverifiable, 0 pending), the unit-by-unit reading it is the inventory of (`coverage.json`), and what a model cannot reach, this machine cannot reproduce, or this repo declined (`limits.json`); the measured store and the recommended batch count per `(L, arch)`; the A/Bs of the `recycle`, `asym`, `no-sieve` and `tuned` levers against the literal kernel on both cards, none of which moved the default, and `plan_vs_kernel.json` (#165); the seed order's spread over 58 rules (`seed_order_spread.json`); each card's measured roofline ceilings, roofline rows on `sm_86` and the answer they give — **batching crosses the ridge only at L = 16 (at 3 batches); L = 7, 10 and 13 are compute-bound already at one batch; `gfx1200` could not be measured** (`docs/implementation/j_engine.md`, "The roofline, in one place" and "The `[JEngine]` claims"). That refusal is **final** (#140): `rocprofv3` now loads in the ROCm image, but gfx1200 defines no fp64 counter and its byte counters read 0 on this box. **gfx1200's fp64 ceiling is settled** (#141): single issue, 128 flop per counted GPU cycle, so the recorded 4.46e11 flop/s is the card at a counted ~3.49 GHz and the ridge stays 1.49 FLOP/B. Also: the whole-build wall-clock on both cards and #147's sort-key A/B (`measured/builds.json`; the expectation that the global `max\|D\|` key is worse is **contradicted on both cards**), and the `τ`-ladder screening record (`ao/screening.json`). |
| the generator | **live, for the whole J chain.** `KERNEL_GEN_CMD` in `devtools/config.sh` runs `python -m felizjueves.emit`, which writes, in CUDA or HIP, the single-batch `[p\|q]` kernel for L = 0..16 and the pair kernels `dq_{la}_{lb}` and `jab_{la}{lb}` for `la ≤ lb ≤ 3` (37 units). Split classes (`build_pq_kernels`, atomic write-back) and, through `--class LA LB LC LD`, each real class's whole chain (`DQ`, the `[p\|q]` batches, `JAB`) are written by the tests. **Generated: every arithmetic step of `J`.** The pair list, the bound, the sort with its `Dmax` read-back, the staging and the launch code stay host-side, and nothing about the new kernels is measured. Dispatch (`dispatch.py`) chooses a batch count per class for a launch plan; `KERNEL_GEN_CMD` does not follow it, and `Spec.n_batches` defaults to 1 by decision (another count: `python -m felizjueves.emit --class LA LB LC LD --n-batches 1 3 6`). **What it ships is the literal plan, by decision:** the searched plan reaches the emitter through the opt-in `tuned` lever, and on both cards it was measured against literal and refused on wall-clock (#166). The levers measured on both cards are `recycle`, `asym`, `no-sieve` and `tuned`, and none moved the default (`docs/implementation/j_engine.md`, "What the generator ships, and why"). |
| `pyproject.toml` | **a virtual uv project** — dev tooling, numpy, and a lock; PySCF is a separate `oracle` group a bare `uv sync` does not install. Nothing is installed; pytest imports `src/` through `pythonpath`. |

When you run something here, say which of these it touched.

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

The device toolchains do not have to be on your host — they are in the images:

```bash
docker/build.sh cuda                      # base, then cuda; tags fj:cuda and fj:latest
devtools/devcontainer.sh shell            # interactive, in the CUDA container
devtools/devcontainer.sh --hip shell      # the ROCm one
```

`docker/README.md` has the four Dockerfiles, the build args and the two front
ends.

## The tiers, and what each one proves

| Tier | Command | What a green run means |
|---|---|---|
| lint | `ruff check .` | narrow by design: `E,F,I,UP,B`, no formatter. |
| fast | `pytest` | the Python side agrees with itself, and with libcint through tracked fixtures. Compiles nothing, launches nothing, needs no PySCF. This is the push gate, and `FAST_TEST_ARGS` excludes every marker that would reach a compiler, a card, PySCF (`oracle`) or minutes (`slow`). |
| kernel | `devtools/kernel-tier.sh --backend cuda\|hip` | generate → compile with the real device compiler (a warning in generated text fails it) → run on the card → compare against the host reference, and `J_µν` against libcint. |
| oracle | `uv sync --group oracle`, then `pytest -m oracle` | the tracked fixtures still equal what PySCF computes today. |

**The kernel tier is the only place a claim about a card can be made.** No
hosted CI runner has a GPU. A `gpu`-marked test *skips* on a device-less box
rather than failing, so a green run there is green and empty —
`devtools/doctor.sh` exists to make that visible. The `gpu` fixtures lock the
card themselves (`GPU_LOCK_PATH` in `devtools/config.sh`), so two runs on one
card serialise.

This box is the reference machine for both halves: an **RTX 3080** (sm_86, CUDA
13.0) and an **RX 9060 XT** (gfx1200, RDNA4, ROCm 7.2.4). A generated kernel has
to satisfy both compilers, and `hipcc` is clang while `nvcc` is not — that
disagreement is this project's main portability risk, which is what
`.devcontainer/combined/` is for.

## Layout

```
devtools/        the scripts: doctor, devcontainer, worktree, kernel tier, claude pin
docker/          four Dockerfiles (base -> cuda/hip -> combined), build.sh, compose
.devcontainer/   one variant per GPU image, driven by devtools/devcontainer.sh
docs/            references.md, and the derived notes on what gets generated
data/            published tables and claims (literature/), scoreboards, the measured store, ao/
src/felizjueves/ plan/, the kernel builder and emit/, launch.py and dispatch.py, the measured store and driver, literature/, and ao/
test/            pytest; markers separate "needs a compiler", "needs a card", "needs PySCF";
                 test/oracle/ holds the libcint fixtures and the PySCF converter
```

Depth lives where it is used: each script's header explains itself (`--help`
prints it), `docker/README.md` covers the images, and `.claude/CLAUDE.md` is the
map for working in here.
