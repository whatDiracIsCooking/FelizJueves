# FelizJueves

Python code generation for electronic structure tensors.

## What is here

**The dev environment and the infrastructure. Not the generator — it does not
exist yet.** Nothing in this repo emits a kernel today, and every tier reports
that rather than implying otherwise:

| | State |
|---|---|
| `docker/`, `.devcontainer/`, `devtools/` | **live.** Both device toolchains build, and `devtools/kernel-tier.sh` runs end to end. |
| `test/` | **live, and thin.** `test_toolchain.py` asserts that nvcc and hipcc compile for this box's targets and that a kernel launched on each card agrees with a numpy reference. That is the floor the rest will stand on, not the subject. |
| the generator | **absent.** `KERNEL_GEN_CMD` in `devtools/config.sh` is empty, so the kernel tier skips its generate and compile phases and says so. |
| `pyproject.toml` | **a virtual uv project** — dev tooling and a lock, no importable package yet. |

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
| fast | `pytest` | the Python side agrees with itself. Compiles nothing, launches nothing. This is the push gate, and `FAST_TEST_ARGS` excludes every marker that would reach a compiler. |
| kernel | `devtools/kernel-tier.sh --backend cuda\|hip` | generate → compile with the real device compiler → run on the card → compare against the host reference. |

**The kernel tier is the only place a numerical claim can be made.** No hosted
CI runner has a GPU, so everything else can say is that the generator agrees
with its golden files. A `gpu`-marked test *skips* on a device-less box rather
than failing, so a green run there is green and empty — `devtools/doctor.sh`
exists to make that visible.

This box is the reference machine for both halves: an **RTX 3080** (sm_86, CUDA
13.0) and an **RX 9060 XT** (gfx1200, RDNA4, ROCm 7.2.4). A generated kernel has
to satisfy both compilers, and `hipcc` is clang while `nvcc` is not — that
disagreement is this project's main portability risk, which is what
`.devcontainer/combined/` is for.

## Layout

```
devtools/        the scripts: doctor, devcontainer, worktree, kernel tier, articles
docker/          four Dockerfiles (base -> cuda/hip -> combined), build.sh, compose
.devcontainer/   one variant per GPU image, driven by devtools/devcontainer.sh
docs/            references.md, and the derived notes on what gets generated
test/            pytest; markers separate "needs a compiler" from "needs a card"
```

Depth lives where it is used: each script's header explains itself (`--help`
prints it), `docker/README.md` covers the images, and `.claude/CLAUDE.md` is the
map for working in here.
