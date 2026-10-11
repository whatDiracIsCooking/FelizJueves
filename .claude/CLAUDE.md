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

**The generator does not exist yet.** This repo is the dev environment and the
infrastructure, and every tier reports that rather than implying otherwise.

| | State |
|---|---|
| `docker/`, `.devcontainer/`, `devtools/` | **live.** Both toolchains build; `devtools/kernel-tier.sh` runs end to end on this box. |
| `test/` | **live, and thin.** `test_toolchain.py` asserts nvcc and hipcc compile for this box's targets and that a kernel launched on each card matches a numpy reference. The rest is fast-tier: the `Spec` key, the transcribed tables' structural invariants, and the scoring rules on **synthetic** candidates. |
| `src/felizjueves/` | **importable, and small.** `spec.py` is `Spec`, the frozen key every later subsystem joins on. `literature/` loads published tables and scores a candidate against them: `tables`, `rules` (`Agreement`, `AtOrUnder`, `ExactGivenInputs`), `scoreboard`, `limits`. `plan/` (below) computes the Table 1–3 cells it scores. |
| `src/felizjueves/plan/` | **live, host-side only.** Pipeline steps 1–3 of `[JEngine, v2, §6]`: `targets`, `batching` (the minimax batch split), `mild`, `dag` (one MILD DAG per batch), `sieve` (Alg. 3, off with the `no-sieve` lever), `cost` (`PlanCost`, Tables 1–3's quantities), composed by `pipeline.plan(spec)` with a JSON per-batch report. `PYTHONPATH=src uv run python -m felizjueves.plan replay literal` scores a board (exit 0 pass, 1 FAIL, 2 could not run, 3 UNSET/no candidates); `report L n_batches` prints one plan. It counts; it emits no kernel text and compiles nothing. |
| `data/` | **live.** `literature/jengine/table{1,2,3}.json` is `[JEngine]` Tables 1–3, 120 cells with provenance. `scoreboards/jengine/{literal,searched}.json` are the two boards. On `literal` every Agreement floor is set to what the plan layer measured (exact cells: Table 3 38, Tables 1–2 post-sieve 7 and 15); every `exact_given_inputs` is still unset (`null`). `searched` replays `plan/search.py`'s recorded cut and seed order per cell (`searched_cuts.json`) and FAILs `at_or_under` on Tables 1–2 (21 and 3 cells, all within +2.04%); `limits.json` records that residual, with each cell's margin, as a `StructuralLimit`. Table 4 (spills) is deliberately not transcribed; `data/literature/jengine/README.md` says why. |
| the generator | **absent.** `KERNEL_GEN_CMD` in `devtools/config.sh` is empty, so the kernel tier skips its generate and compile phases **and says so in its report**. Do not read such a run as a green kernel tier. |
| `pyproject.toml` | **a virtual uv project** — no `[build-system]`, nothing installed. `src/` is importable from pytest only, via `pythonpath = ["src"]`. Dev tooling plus numpy, and a lock. |
| `docs/` | `references.md`, plus the derived notes in `implementation/`: `j_engine.md` (recurrences, kernel contract, batching, pipeline) and `mild.md` (the MILD heuristic and the sieve). Both are **specifications** the code is read against: batching, MILD and the sieve are implemented (host-side, in `plan/`); the emitter and kernels they describe are not. Tracked. |

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
| fast | `pytest` | the Python side agrees with itself. Compiles nothing, launches nothing. |
| kernel | `devtools/kernel-tier.sh --backend cuda\|hip` | generate → compile with the real device compiler → run on the card → compare against the host reference. |

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
  otherwise would, inside a container where both are present.
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
  reason.
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
