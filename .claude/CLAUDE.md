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

This is the map. The per-family detail (which kernels exist, what they were
checked against, which card each one ran on, what was measured and what was
decided) is in the **[tree-state](skills/tree-state/SKILL.md)** skill, one
reference file per family, moved there verbatim from this file. **Read the
relevant file before you claim, change or test anything in that family.**
Each family's design is in its `docs/implementation/*.md`. Each of those docs
opens with a status block, and it is the one place with the detail.

The pattern to expect everywhere: "generated" does not mean "run on a
card", and "run on a card" does not mean "measured". Many families are
**interpreted only** or **compiled only** on one card. The RX 9060 XT runs
that are still missing have no open issue: #431 closed on 2026-10-17 when
the card's detection was fixed (below), not when they were re-run, so the
list is each family's "RTX 3080 only" in the tree-state files.

| Family | Where it stands | Reference |
|---|---|---|
| `J` chain (`[p\|q]`, `DQ`, `JAB`, launch plan) | generated and run on both cards; `J` witnessed against libcint; low-stack `DQ`/`JAB` at (ff); one batch and the **literal** plan by decision | `j-chain.md` |
| one-electron `S`, `T`, `V` | generated, run on both cards through (ff); low-stack `V` at (ff); not scored, not measured | `j-chain.md` |
| exchange `K` (milestone 10, **open**) | `kernel_k` on cards at L ≤ 2 only; block forms `kb` / `kbr` / `kbrc` (through d `kb`, at f `kbr`, at g `kbrc`); `kbd`, `kbrt` measured and not chosen | `k-exchange.md` |
| HGP engine, fused `F`, split HGP `J`/`K` | `F` on both cards and timed; split HGP on the RTX 3080 only; nothing above L = 4 | `fock-hgp.md` |
| Schwarz `Q`, cross overlap `SX`, DF metric, `(mn\|P)` | generated; DF through (ff\|i) / (i\|i) on both cards with low-stack forms; HGP DF twins on the RTX 3080 only; Rys metric through (g\|h) compiled by nvcc and hipcc, on no card | `schwarz-overlap-df.md` |
| dipole `MU`, quadrupole; one-electron on HGP and Rys | dipole on both cards at L ≤ 4, through (ff) host/interpreted/compiled only; quadrupole (raw second moments) interpreted only; HGP one-e on the RTX 3080 only; Rys one-e interpreted only | `one-electron-extra.md` |
| gradients (d1) | `gs`/`gt`/`gv` and the on-card `gjab` chain (`DQ` → raised `[p\|q]` → `gjab`) on both cards through (ff); `gk`, DF-J terms on both cards; DF-K weights and the DF two-electron gradient host-side, held to PySCF; the whole RHF gradient assembled and interpreted, held to PySCF, not on a card; `J` on HGP (`ghj_`) interpreted only, not emitted; `T` and `S` on HGP (`hgt_`, `hgs`) interpreted and compiled through (ff), not on a card; `ghk` (`K` d1 on HGP) interpreted through (dd\|dd), nvcc- and hipcc-compiled through (dd\|pp), no card | `gradients.md` |
| Hessians (d2) | `hjab`/`hdq` and unrolled `d2s`/`d2t`/`d2v` on both cards through (dd); `hkx`, `d2metric`, `d2eri3` interpreted only | `hessians.md` |
| Rys `F` / `J` / `K`, device Boys | Rys `F` / `J` / `K` interpreted and emitted, never compiled by nvcc/hipcc (the Rys metric is, above); `fj_boys<M>`'s move left `kernels.json` pins **stale** | `rys-and-boys.md` |

Against the predecessor `jgen`, this is parity on its energy-order `J`
surface, with one difference: Table 4, declined. Which plan ships is a
decision, not a gap. The opt-in `tuned` lever exists but the **literal** plan
ships, because the searched one was measured on both cards and refused on
wall-clock (#166). The searched board PASSes `at_or_under` on Tables 1, 2 and
3 in all 40 cells: 22 land on the printed triple exactly and 18 come in under
it.

Every tier reports what it actually touched rather than implying more: a
`gpu`-marked test skips without a card, so a run can be **green and empty**;
`kernel-tier.sh` exits **2** in that case and whenever it could not run — read
a 2 as "could not run", never as a pass.

| | State |
|---|---|
| `docker/`, `.devcontainer/`, `devtools/` | **live.** Both toolchains build; `devtools/kernel-tier.sh` runs end to end on this box; `devtools/gate.sh` (the PR gate, #656) has run device-less only. |
| `test/` | **live, and thin — and it measures nothing.** Timing, counters and stored-row checks live in `measure/`. Markers: `cuda`/`hip` need a compiler, `gpu` a card, `slow` is L above `FAST_L_MAX`, a large input, or a compile-only case after the first of its kind (`tiers.sample_compiles`), `l_cut` beside `slow` when only those two cuts put it there (the gate's sampler draws from them), `oracle` PySCF, `claim(id)` a claims record. Per-family test files are listed in `test-dir.md`. `test/registry/` is the case registry (#648): every family's `(engine, quantity, deriv, L)` cases, the tiers and cards each reaches, and the rule mapping each device test's cases onto them, held to the collection both ways by `test_registry.py`; each case's `cost_s` (compile and `gpu` seconds per backend, measured or modelled) is `data/measured/registry_cost.json`, refreshed by `devtools/measure.sh registry-cost` (#652). `test/golden/numeric/` is the numerical golden store (#655): libcint/PySCF arrays as plain `.npy`, keyed by registry case, each over a `data/` basis file, plus an index of the `test/oracle/fixtures/` arrays it does not copy; `oracle.numeric.compare` is the one helper a test calls, and `oracle.generate --numeric` regenerates it bit for bit. It is not the **text** goldens beside it. `test/registry/sample.py` is the opt-in `--fj-sample` plugin over it (#653): every L ≤ 2 case (`FJ_MIN_L_MAX`), a seeded, budgeted, change-weighted draw above that never exceeds a band's budget; only `devtools/gate.sh` passes it. |
| `measure/` | **live, never a gate.** `devtools/measure.sh list` names the sweeps; `check` recomputes the tracked stores (no compiler, no card). A kernel-text change that moves a stored `source_sha256` fails `check`, not the PR gate. See `measure-dir.md`. |
| `src/felizjueves/` | **importable.** Root holds only `__init__`, `spec`, `engine`, `levers` (`test_root_layout.py` enforces it); everything else is in a package: `ao/`, `mmd/`, `hgp/`, `rys/`, `plan/`, `ir/`, `emit/`, `eri/` (`j_coulomb/`, `k_exchange/`, `fock/`, `hjk/`, `density_fit/`), `one_electron/`, `gradient/`, `hessian/`, `performance/`, `literature/`, `reference/`. `spec.Spec` is the key every subsystem joins on. See `src-dir.md`. |
| `src/felizjueves/plan/` | **live, host-side only.** Steps 1–3 of `[JEngine, v2, §6]` plus the schedule; counts, emits nothing. See `plan-dir.md`. |
| `data/` | **live.** Transcribed `[JEngine]` / `[Fock]` tables, claims and limits; the measured stores in `data/measured/`; FreeBreakfast's basis sets, verbatim, in `data/basis/` and `data/aux_basis/` (#650); the libcint fixtures in `test/oracle/fixtures/`. See `data-dir.md`. |
| the generator | **live.** `KERNEL_GEN_CMD` (`devtools/config.sh`) writes the PR gate's 47 units; every other family has its own opt-in `python -m felizjueves.emit --<family>-l` ladder, never in the bare default nor in `KERNEL_GEN_CMD`. See `generator.md`. |
| `pyproject.toml` | **a virtual uv project** — no `[build-system]`, nothing installed. `src/` is importable from pytest only, via `pythonpath = ["src"]`. Dev tooling plus numpy, and a lock. |
| `docs/` | `references.md` and the specifications in `implementation/`, each opening with a status block. See `docs-dir.md`. |

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
| **gate** | `devtools/gate.sh` | **the one check that blocks a merge** (#656; the `pr` skill runs it and pastes its `summary.md` -- seed, budgets, every case drawn -- into the PR). lint + fast + the **compile sample** (nvcc and hipcc) everywhere, plus the **gpu sample** through `kernel-tier.sh` on each card `lib.sh`'s `card_present` enumerates. The two samples select `not slow or l_cut` (`GATE_*_TEST_ARGS`; `l_cut` is what `tiers.by_l` and `tiers.sample_compiles` add beside `slow`) and `--fj-sample` draws above the `min` band: L ≤ 2 on a card, L = 0 for the compile sample (`GATE_COMPILE_MIN_L_MAX`). With no card it ends after the compile sample, exits 0 and names cuda and hip unverified on card (`unverified-on-card`, #651). Needs both compilers in one shell: exit **2** on this box's host (no hipcc), so run it in the combined container. Device-less, 4-core cloud container, 2026-10-30: lint + fast + compile, exit 0; there the compile sample took 7m57s (371 cases) where the `not slow` selection below took 7m02s (272) back to back -- that box is about twice as slow as the one that timed the row below. The gpu sample is **not yet timed on a card**. |
| lint | `ruff check .` | `E,F,I,UP,B`, no formatter. |
| fast | `pytest -m "not cuda and not hip and not gpu and not slow and not oracle"` (`FAST_TEST_ARGS`; a bare `pytest` applies no marker filter; 3m27s on `JOBS=8` with `XDIST_DIST=worksteal`, 2026-10-29) | the Python side agrees with itself — and, through tracked fixtures, with libcint — **for classes with L ≤ 4** (`test/tiers.py`'s `FAST_L_MAX`, #419): anything that builds, emits, plans or integrates a class above it is `slow`, and a parametrized test is split at the cut, so the `[JEngine]` board replays and the plan search (every cell L 7–16) are slow-tier only; reads of stored rows and keys stay fast. Compiles nothing, launches nothing, needs no PySCF. |
| kernel | `devtools/kernel-tier.sh --backend cuda\|hip` | generate → compile with the real device compiler (a warning located in generated text fails the compile phase, exit 1 — #92) → run on the card → compare against the host reference, and against libcint for `J_µν` (`test_ao_j_ondevice.py`). The device phase is `gpu and <backend> and not slow` (`KERNEL_TEST_ARGS`): L ≤ `FAST_L_MAX` on small inputs (H₂, two H₂, H₂O/STO-3G), the PR gate's card phase before `gate.sh` sampled it, 3m59s on the RTX 3080 and 3m54s on the RX 9060 XT, each with `--no-cache` (every compile cold), 2026-10-29. |
| kernel, full | `devtools/kernel-tier.sh --backend cuda\|hip --full` | the same, every `gpu` test with **L ≤ 8** (`KERNEL_FULL_TEST_ARGS`): the L = 7 J and K subgrid cells, #259's L = 8 K check, the `[p\|q]` ladder through L = 8, the 6-31G / cc-pVDZ / cc-pVTZ builds, the per-class F lever checks. For a milestone's close or a kernel-text change above the cut. **Since #301 both kernel selections exclude `high_l`**, every case above `tiers.DEVICE_L_MAX = 8` (the L = 10, 13 and 16 J/K subgrid and asymmetric cells, the ladder's L = 9..16): one L = 10 K unit's compile alone outlasted an hour. They are kept: `pytest -m "gpu and <backend> and high_l"` runs them by hand. |
| compile | `pytest -n 4 --dist worksteal -m "(cuda or hip) and not gpu and not slow"` (3m24s and 3m56s in two cold runs on a 4-core Claude cloud container, no sccache, 2026-10-29) | every compile-only test function's emitted text survives nvcc **and** hipcc, needing no card — for the first case of each kind of unit it emits (`tiers.sample_compiles`: 276 of 2,551 cases, all 179 functions); the rest are `slow`, which the 28m38s full set in `test/compile-only-timings.xml` was. Proves no number. |
| measure | `devtools/measure.sh <sweep>` | **not a correctness tier**: timing, counters, ceilings, the recompile-against-stored-rows check and the stored reports recomputed (`check`). Nothing here blocks a PR. |
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

**A card is present when the kernel enumerates it, not when its device node
exists** (#431). `/dev/kfd` is there whenever `amdgpu` is loaded, bound card or
not — the module binds by PCI id, and kernel 6.8's in-tree one does not know
the RX 9060 XT's `1002:7590`, so this box takes `amdgpu` from DKMS and a kernel
upgrade it was not rebuilt for drops the card while leaving CUDA untouched. So
AMD presence is **an enumerated GPU node in kfd's topology** (a nonzero
`gfx_target_version`), asked in one rule by `lib.sh`'s `amd_gpu_present`,
`conftest._has_amd_device` and `doctor.sh`'s enumeration — no ROCm needed, and
the same answer inside a container as on the host. Before that, HIP reported
`device: present` with no usable card and every `gpu` test failed instead of
skipping, which from inside a container reads as broken passthrough. When HIP
goes dark while CUDA works, run **doctor** first: it names the unbound device
and the `dkms autoinstall` that fixes it.

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
- **A PR from a session without both cards may merge, labelled** (#651):
  after `devtools/gate.sh` is green there (lint, fast, compile sample), carrying `unverified-on-card` and naming
  the backend(s) no card ran. `lib.sh`'s `card_present` / `card_absent` decide
  it, and `doctor.sh`'s `card verdict` line prints the same answer; the `pr`
  skill has the rule.
- **The kernel tier is deliberately NOT a hook.** An unrolled high-L kernel is
  slow to compile and the tier costs minutes; a gate that costs minutes is one
  people bypass with `--no-verify`. `devtools/gate.sh` runs it (sampled) on
  each enumerated card before a PR merges, and its summary says which.
- **What blocks a PR, and its budget: under five minutes each.** One
  script, `devtools/gate.sh`: the fast tier (3m27s), the compile sample, and
  the gpu sample per card (the kernel tier's own `not slow` selection took
  ~4m cold on either card; the sample's draw is budgeted to stay near it). Both test correctness only. A test that would push either
  past five minutes is `slow` (by L above `FAST_L_MAX`, or by a large input),
  never deleted: `--full` and a bare `pytest` still run it. Measurement is
  never a test: it goes in `measure/`, behind `devtools/measure.sh`.
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

**Two agents are pinned in the three GPU files**, as the last two installs in
each — `docker/install-claude-code.sh`, then `docker/install-agy.sh`, with only
the `USER ubuntu` switch and a toolchain check after them. So each
version is declared three times — bump Claude Code with
`devtools/claude-version.sh --apply`, which rewrites all three, and `agy` with
`devtools/agy-version.sh --apply`; `doctor.sh` reports when either disagrees
across the files or has fallen behind upstream.

`agy`'s pin is **two lines per file** (`ARG AGY_RELEASE` plus `ARG AGY_SHA512`)
because upstream's manifest serves only the current release: a version alone is
not addressable, so the pin is a release id — version plus an opaque build id —
and the sha512 of that exact tarball, which `install-agy.sh` verifies rather
than records. `agy-version.sh` rewrites the pair as one unit and reports a file
whose hash drifted from its release as `sha-differs`; `--verify` re-hashes the
pinned tarball (~62MB) without touching an image. `agy` is installed in the
images but **not logged in** — its state lives in `$HOME/.gemini`, which
nothing here binds.

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
| [tree-state](skills/tree-state/SKILL.md) | read the detailed state of a kernel family (what is generated, checked, run on which card, measured, decided) before claiming or changing anything in it; keep it updated in the same PR. |
| [test](skills/test/SKILL.md) | run the suites the right way and read the result honestly — the fast pytest tier, the kernel tier on each backend, the marker/fixture split, and what a green run does *not* prove. |
| [devbox](skills/devbox/SKILL.md) | drive the containers and the container-backed **sibling** worktrees — `devcontainer.sh` (up/rebuild/shell/test/down) and `worktree.sh` (add/rm/sync/gc). |
| [worktree](skills/worktree/SKILL.md) | create and clean up the lightweight `.claude/worktrees/<name>` checkouts — no container, cheap and disposable. |
| [doctor](skills/doctor/SKILL.md) | diagnose a degraded environment — `doctor.sh` plus the failures it cannot see. |
| [pr](skills/pr/SKILL.md) | ship the current work: commit → push → PR → merge, with this repo's gotchas. |
| [issue](skills/issue/SKILL.md) | resolve one GitHub issue end to end in a fresh worktree, tested on **both** cards, then `/pr full`. |
| [milestone](skills/milestone/SKILL.md) | turn a plan into a GitHub milestone plus a DAG of PR-sized issues. |
| [milestone-run](skills/milestone-run/SKILL.md) | execute a milestone with one subagent per ready issue. |
| [audit](skills/audit/SKILL.md) | verify that documentation (skills, READMEs, this file) still matches reality — extract each claim, check it, report drift. |
