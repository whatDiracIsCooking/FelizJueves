# FelizJueves

Python code generation for electronic structure tensors.

## What is here

**A generator of one kernel family, and a Coulomb matrix built around it.** The
generator emits the McMurchie–Davidson `[p|q]` kernel of `[JEngine]` in CUDA
and HIP; a host-side AO layer turns a real molecule into its inputs and its
output back into `J_µν`, and that `J` is compared against PySCF/libcint — a
program with no shared ancestry. Every tier reports what it touched rather
than implying more. `.claude/CLAUDE.md` ("The state of the tree") has the
detail; in short:

| | State |
|---|---|
| `docker/`, `.devcontainer/`, `devtools/` | **live.** Both device toolchains build, and `devtools/kernel-tier.sh` runs end to end. |
| `test/` | **live.** Fast tier: the plan layer against `[JEngine]` Tables 1–3, the kernel IR interpreted against `reference/`, the golden kernel texts, and the AO layer against **tracked libcint fixtures** (`test/oracle/`), so PySCF is not needed to run it. Kernel tier (`gpu`): the generated kernels on each card against `reference/`, and `J_µν` for H₂ and H₂O against libcint. |
| `src/felizjueves/` | **importable.** `plan/` (batching, MILD, the sieve, the per-batch counts, and `optimum.py`, the exact tree-search optimum for L ≤ 5), `kernel.py` + `emit/` (the generator), `measured_store.py` (the measured store: registers, spills, wall-clock on both cards; roofline counters on `sm_86`) and `measured.py` (the driver that compiles, parses and profiles into it), `gate.py` (the card gate every timed or profiled phase runs behind), `ceilings.py` (each card's measured fp64 and bandwidth ceilings), `ridge.py` (whether batching crosses our cards' ridge, under the rule #114 declared before the rows), `crossread.py` (model, clock and counters read against each other), `asym.py` (the asymmetric-split study), `literature/` (published tables and claims, and the rules that score against them), and `ao/` (shells, the `E` transform, pair data and the Schwarz bound, ERI blocks, and `J_µν`). |
| `data/` | **live.** `[JEngine]` Tables 1–3 with provenance and the two scoreboards over them; the registry of the paper's non-numeric claims (`claims.json`, 32 records: 18 asserted, 14 unverifiable, 0 pending), the unit-by-unit reading it is the inventory of (`coverage.json`), and what a model cannot reach or this machine cannot reproduce (`limits.json`); the measured store and the recommended batch count per `(L, arch)`; each card's measured roofline ceilings, roofline rows on `sm_86` and the answer they give — **batching crosses the ridge only at L = 16 (at 3 batches); L = 7, 10 and 13 are compute-bound already at one batch; `gfx1200` could not be measured** (`docs/implementation/j_engine.md`, "The roofline, in one place" and "The `[JEngine]` claims"); the `τ`-ladder screening record (`ao/screening.json`). |
| the generator | **live, for one kernel family.** `KERNEL_GEN_CMD` in `devtools/config.sh` runs `python -m felizjueves.emit`, which writes the single-batch `[p\|q]` kernel for L = 0..16 in CUDA or HIP; split classes (`build_pq_kernels`, atomic write-back) and, through `--class LA LB LC LD`, the classes a real basis reaches are written by the tests. **Generated: `J_p` only.** The `E` transform, the pre-contraction and the scatter are host-side Python; dispatch (choosing a batch count per class) is not built. |
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
src/felizjueves/ plan/, the kernel builder and emit/, the measured store and driver, literature/, and ao/
test/            pytest; markers separate "needs a compiler", "needs a card", "needs PySCF";
                 test/oracle/ holds the libcint fixtures and the PySCF converter
```

Depth lives where it is used: each script's header explains itself (`--help`
prints it), `docker/README.md` covers the images, and `.claude/CLAUDE.md` is the
map for working in here.
