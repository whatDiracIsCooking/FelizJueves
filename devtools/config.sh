#!/usr/bin/env bash
# Per-project settings for everything in devtools/. EDIT THIS FILE, not the
# scripts -- the scripts are meant to survive being copied into the next repo
# unchanged, and this is the one place that knows what the project is called
# and how its tests run.
#
# Sourced (never executed) by devcontainer.sh, worktree.sh, doctor.sh,
# kernel-tier.sh and prepush-tests.sh. Every value uses ${VAR:-default} so an
# environment variable still wins for a one-off:
#
#   JOBS=16 devtools/prepush-tests.sh
#
# TWO THINGS THIS FILE CANNOT REACH, both of which must be kept in step by
# hand:
#   * .devcontainer/<variant>/devcontainer.json -- JSON cannot source shell, so
#     the volume names, the container name and the .git bind path are spelled
#     out there too. PROJECT_NAME below must match the `<name>-pytest-tmp-`
#     style prefixes used there, or `worktree.sh rm` and `worktree.sh gc` will
#     not recognise this project's volumes as its own. doctor.sh checks that
#     the two agree and warns when they have drifted.
#   * the `FJ_` ENVIRONMENT PREFIX. devcontainer.sh exports FJ_GIT_DIR,
#     FJ_SCCACHE_DIR and one FJ_<GROUP>_GID per device group, and the
#     `${localEnv:...}` references in the json and in docker/compose.yaml read
#     those names literally. Renaming the project means renaming those too --
#     they are not derived from PROJECT_NAME, because a JSON file cannot
#     interpolate one.

# --- identity -------------------------------------------------------------

# Prefix for this project's docker volumes and per-worktree build images, and
# the name of the devcontainer. Must match devcontainer.json (see above).
#
# It ends up in docker resource names, so keep it lowercase and short: the repo
# is FelizJueves and this is its docker spelling. Short matters more than
# descriptive here -- this box already carries images from two other projects,
# and the `devcontainer.project` label built from this value is what scopes
# `worktree.sh gc` so it cannot delete theirs.
PROJECT_NAME=${PROJECT_NAME:-fj}

# Which devcontainer.json devcontainer.sh drives. Relative paths resolve
# against the repo root, so this works from any cwd and from any worktree.
# There is one per GPU file of docker/ (Dockerfile.cuda / .hip / .combined):
#
#   .devcontainer/cuda/devcontainer.json     THE development container: the
#                                            CUDA toolkit (nvcc) for sm_86.
#                                            Needs the NVIDIA card and
#                                            nvidia-container-toolkit.
#   .devcontainer/hip/devcontainer.json      the same toolchain with ROCm and
#                                            no CUDA at all. Needs the AMD card
#                                            and the amdgpu kernel driver. It
#                                            pins KERNEL_BACKEND=hip in
#                                            containerEnv, so kernel-tier.sh
#                                            needs no flag in there.
#   .devcontainer/combined/devcontainer.json both SDKs. The one to open when
#                                            the question is whether a
#                                            generated kernel compiles under
#                                            BOTH device compilers -- which is
#                                            this project's main portability
#                                            risk, since hipcc is clang and
#                                            nvcc is not.
#
# This is only the FALLBACK. For one call use the variant flag --
# `devtools/devcontainer.sh --hip up` -- and for a whole shell session export
# the variable. Precedence is flag, then variable, then this.
#
# Containers are keyed on the workspace folder AND this path, so the variants
# of one worktree are separate containers and can be up at once. They share
# that worktree's volumes, so run the suite in one at a time.
DEVCONTAINER_CONFIG=${DEVCONTAINER_CONFIG:-.devcontainer/cuda/devcontainer.json}

# Where the workspace is mounted INSIDE the container -- the `workspaceFolder`
# every .devcontainer/*/devcontainer.json sets, which is not the host path.
#
# One of the two things this file cannot reach and that must be kept in step by
# hand (the other is the FJ_ prefix; see the note at the top of the file).
# devcontainer.sh needs it to run this repo's own devtools/lib.sh INSIDE the
# container -- which is how the GPU probe stays ONE rule asked in two places
# rather than a snippet duplicated into a docker exec.
DEVCONTAINER_WORKSPACE_FOLDER=${DEVCONTAINER_WORKSPACE_FOLDER:-/workspace}

# --- the kernel tier ------------------------------------------------------
#
# devtools/kernel-tier.sh is the local gate for anything touching device
# behaviour: generate the kernels, hand each translation unit to the device
# compiler, run them on a card, compare against the host reference. It is the
# counterpart of the fast pytest tier, which touches no compiler and no device.
#
# NOTHING ELSE IN THE REPO PROVES A KERNEL WORKS. No hosted CI runner has a
# GPU, so this script on this box is the only place a numerical claim can be
# made at all -- which is why it is a script with phases rather than a line in
# a workflow file.

# Which backend the tier drives when no --backend flag is given. The CUDA image
# is the default container, so this matches it; the hip devcontainer overrides
# it in containerEnv.
KERNEL_BACKEND=${KERNEL_BACKEND:-cuda}

# The device compilers, and the architecture each targets. These are the same
# values the images bake in (docker/Dockerfile.cuda's CUDA_ARCH,
# docker/Dockerfile.hip's GPU_TARGETS), read from the environment when present
# so a container's own pin wins and a bare host can still be told.
#
# sm_86 is the RTX 3080 in this box; gfx1200 is the RX 9060 XT (RDNA4). Neither
# is a guess about the next machine: widen them per call rather than here.
NVCC=${NVCC:-nvcc}
HIPCC=${HIPCC:-hipcc}
CUDA_ARCH=${CUDA_ARCH:-86}
HIP_ARCH=${HIP_ARCH:-gfx1200}

# Flags every generated translation unit is compiled with, per backend.
#
# -lineinfo is NOT a debug flag here: it is what makes compute-sanitizer name a
# LINE of generated kernel text instead of an address, and a generated kernel
# is exactly the thing you cannot read a bare address in. Cheap enough to keep
# on by default.
#
# NOTE a deliberate absence: no -use_fast_math, and nothing that relaxes
# floating point. The tier's only assertion is "this agrees with the host
# reference to a tolerance", and a flag that reassociates arithmetic changes
# the thing being measured. Add one per experiment, never here.
#
# THE WARNING FLAGS ARE A RATCHET (#92, after #62's unused declaration): the
# compile phase of kernel-tier.sh FAILS on any warning whose location is a
# generated translation unit, naming the kernel and the line. nvcc's front end
# warns on its own (#177-D and friends); `-Xcompiler -Wall` adds the host
# compiler's view of the same text. hipcc is clang, and `-Wall -Wextra` is its
# whole policy. The two disagree about what warns -- that is expected, and the
# fix for a warning on one side is in the emitter, never a per-vendor
# `-diag-suppress` / `-Wno-` here.
#
# Deliberately NOT -Werror: that would also fail on a warning inside a vendor
# header (hip_runtime.h, the CUDA runtime), which the generator cannot fix.
# kernel-tier.sh filters by FILE instead; a warning outside the generated
# unit is reported and does not fail the tier.
NVCC_FLAGS=${NVCC_FLAGS:--O3 -lineinfo --std=c++17 -Xcompiler -Wall}
HIPCC_FLAGS=${HIPCC_FLAGS:--O3 -g1 -std=c++17 -Wall -Wextra}

# How the kernels get generated. The tier runs this, from the repo root, with
# the project venv first on PATH, and expects the emitted sources in
# $KERNEL_TIER_OUT (it also exports KERNEL_TIER_BACKEND and KERNEL_TIER_ARCH;
# kernel-tier.sh's header has the contract). Word-split, not eval'd: no
# quoting, no `VAR=value` prefix -- hence `env`.
#
# felizjueves.emit's CLI reads those three variables as its defaults and
# writes one translation unit per class in the backend's own spelling: the
# `[p|q]` ladder at `--L 0..8` (tiers.DEVICE_L_MAX: no test either tier
# selection runs reads a unit above it), the DQ / JAB pair ladder through
# (ff) (`--pair-l 3`, its default), and `--one-e-l 2` ADDS the one-electron
# ladder (#242): `s_`, `t_` and `v_` for every pair class `la <= lb <= 2`,
# 18 units, through (dd) -- the classes test/test_one_electron_device.py
# holds to the host on each card. 47 units. PYTHONPATH because this is a
# virtual uv project: nothing is installed, src/ is put on the path by hand.
#
# KERNEL_FULL_GEN_CMD is what `--full` generates instead: the bare command's
# whole J chain (37 units, `[p|q]` L = 0..16, whose `source_sha256`
# data/measured/kernels.json pins) plus the same one-electron ladder, 55
# units. L = 9..16 are only compiled, never run by either selection, and cold
# they are most of the compile phase (`pq_L16` alone ~115 s of nvcc on this
# box, `pq_L15` ~60 s), which is why the PR gate leaves them to `--full`.
#
# A KERNEL_GEN_CMD set by the caller wins in both modes (`--full` included),
# so `KERNEL_GEN_CMD="..." kernel-tier.sh --full` runs the caller's ladder.
# Set it EMPTY (KERNEL_GEN_CMD= devtools/kernel-tier.sh ...) and the tier
# reports its generate and compile phases DORMANT rather than pretending.
# shellcheck disable=SC2034  # read by kernel-tier.sh's --full
KERNEL_GEN_CMD_SET_BY_CALLER=${KERNEL_GEN_CMD+yes}
KERNEL_GEN_CMD=${KERNEL_GEN_CMD-env PYTHONPATH=src python -m felizjueves.emit --L 0 1 2 3 4 5 6 7 8 --pair-l 3 --one-e-l 2}
KERNEL_FULL_GEN_CMD=${KERNEL_FULL_GEN_CMD-env PYTHONPATH=src python -m felizjueves.emit --one-e-l 2}

# Where generated sources and their binaries land. Gitignored (/.kernel-tier/),
# and wiped by --fresh.
KERNEL_BUILD_DIR=${KERNEL_BUILD_DIR:-.kernel-tier}

# The compiler cache's off switch for TEST-SIDE compiles (#186). Non-empty
# means test/devcache.py prepends no sccache launcher, so every compile a test
# issues (conftest.compile_device_source, plan_driver.compile_checked) is cold.
# kernel-tier.sh --no-cache exports it for the device phase, and the same
# variable works when pytest runs alone:
#
#   KERNEL_NO_CACHE=1 pytest -m "gpu and cuda"
#
# Test compiles run in a content-addressed directory under KERNEL_BUILD_DIR
# (test-cache/<key>/), so --fresh clears those too; the sccache store
# (SCCACHE_DIR) is never touched by this repo's scripts.
KERNEL_NO_CACHE=${KERNEL_NO_CACHE:-}

# A per-unit compile cap in seconds (#618), empty for none (the default).
# Non-empty, kernel-tier.sh's compile phase kills a unit still compiling
# after it (a failed compile, never a retry) and exports it for the device
# phase, where test/devcache.py caps every test-side compile at it too
# (`compile_unit`'s timeout is the smaller of the two):
#
#   KERNEL_UNIT_TIMEOUT=10 devtools/kernel-tier.sh --backend cuda
KERNEL_UNIT_TIMEOUT=${KERNEL_UNIT_TIMEOUT:-}

# Where kernel-tier.sh writes its run logs. Relative paths resolve against the
# repo root. Gitignored; `.gitignore` and `.dockerignore` both name this path,
# so keep them in step if you rename it.
KERNEL_TIER_REPORTS=${KERNEL_TIER_REPORTS:-.kernel-tier-reports}

# Extra pytest arguments for the tier's device phase. `-m gpu` is the marker
# pyproject.toml declares for "launches a kernel", as opposed to `cuda`/`hip`,
# which only need a compiler. kernel-tier.sh replaces `{backend}` with the
# backend it runs, and this project's compiler markers ARE the backend names,
# so `--backend cuda` runs the gpu tests that launch on the NVIDIA card and
# not the AMD ones (in a combined container both compilers are present, and
# without this one backend's run would compile and launch the other's too).
# Single quotes load-bearing, as for FAST_TEST_ARGS below.
#
# TWO SELECTIONS, because the tier is a PR gate and a gate that costs an hour
# is one nobody runs. KERNEL_TEST_ARGS, the default, excludes `slow`: classes
# with L <= FAST_L_MAX (test/tiers.py) on small inputs (H2, two H2,
# H2O/STO-3G), under five minutes per card. `--full` takes
# KERNEL_FULL_TEST_ARGS instead -- every `gpu` test of the backend, the high-L
# cells and the 6-31G / cc-pVDZ / cc-pVTZ builds included (72 min on the RTX
# 3080 before the cut, most of it the K subgrid) -- for a milestone's close or
# a change to kernel text above the cut. Neither selection measures anything:
# timing, counters and the stored rows are devtools/measure.sh's.
# Both exclude `high_l`, every case with L > DEVICE_L_MAX = 8 (test/tiers.py,
# #301): above it one K unit's compile alone outlasted an hour. Run those
# by hand with -m "gpu and {backend} and high_l".
KERNEL_TEST_ARGS=${KERNEL_TEST_ARGS:-'-m "gpu and {backend} and not slow and not high_l"'}
KERNEL_FULL_TEST_ARGS=${KERNEL_FULL_TEST_ARGS:-'-m "gpu and {backend} and not high_l"'}

# The registry-sampled selection (#653, test/registry/sample.py): a pytest
# plugin that, under `--fj-sample` or FJ_SAMPLE=on, deselects the compile and
# `gpu` cases its draw does not keep. Every case with L <= 2 runs; the `med`
# (2 < L < its ladder's top) and `max` (the top) bands of each pool -- a
# (tier, backend): `gpu/cuda`, `compile/hip`, ... -- are drawn until their
# budget, in seconds of the registry's `cost_s`, is spent. Opt in: only
# devtools/gate.sh passes the flag (GATE_* below). FJ_SAMPLE=all keeps every
# case even under the flag (empty, `on` and `all` are the three values).
#   FJ_SAMPLE_SEED     the seed; empty is a hash of the merge-base and HEAD
#   FJ_SAMPLE_BASE     the ref the merge-base (and the changed paths) is from
#   FJ_BUDGET_MED/MAX  seconds per pool and band; empty is no limit
#   FJ_K_MED/MAX       a count cap per pool and band; empty is none
#   FJ_MIN_L_MAX       the `min` band's top L: every case at or below it runs
#   FJ_DEFAULT_COST_S  a case's cost while its keys' cost_s is unset (#652)
#   FJ_TOUCH_WEIGHT    how much likelier a family the diff touches is drawn
#   FJ_SAMPLE_REPORT   the JSON report (seed, budgets, cases drawn); relative
#                      to the repo root
FJ_SAMPLE=${FJ_SAMPLE:-}
FJ_SAMPLE_SEED=${FJ_SAMPLE_SEED:-}
FJ_SAMPLE_BASE=${FJ_SAMPLE_BASE:-origin/main}
FJ_BUDGET_MED=${FJ_BUDGET_MED-120}
FJ_BUDGET_MAX=${FJ_BUDGET_MAX-60}
FJ_K_MED=${FJ_K_MED:-}
FJ_K_MAX=${FJ_K_MAX:-}
FJ_MIN_L_MAX=${FJ_MIN_L_MAX:-2}
FJ_DEFAULT_COST_S=${FJ_DEFAULT_COST_S:-10}
FJ_TOUCH_WEIGHT=${FJ_TOUCH_WEIGHT:-4}
FJ_SAMPLE_REPORT=${FJ_SAMPLE_REPORT:-${KERNEL_TIER_REPORTS}/sample.json}

# --- the PR gate (devtools/gate.sh, #656) ----------------------------------
#
# The one check that blocks a merge, chosen by what this box has: lint, the
# fast tier, the compile sample under nvcc AND hipcc, and -- for each backend
# whose card is enumerated (lib.sh's card_present) -- the gpu sample on that
# card through kernel-tier.sh. With no card it stops after the compile sample
# and says which backends a PR from here carries as `unverified-on-card`
# (#651). Both sampled phases pass `--fj-sample` over these selections:
#
# `not slow or l_cut` (test/tiers.py): a case that is `slow` ONLY because its
# class is above FAST_L_MAX, or because tiers.sample_compiles dropped it, is
# a candidate, and the registry sampler draws which of them run; a case slow
# for any other reason (a large basis, a function decorated `slow`) stays
# out, as `high_l` does. The fast tier keeps FAST_TEST_ARGS, its L cut
# included; kernel-tier.sh's own selections above are unchanged.
#
# `{backend}` as in KERNEL_TEST_ARGS. Single quotes load-bearing, as for
# FAST_TEST_ARGS below.
GATE_COMPILE_TEST_ARGS=${GATE_COMPILE_TEST_ARGS:-'-m "(cuda or hip) and not gpu and not high_l and (not slow or l_cut)"'}
GATE_GPU_TEST_ARGS=${GATE_GPU_TEST_ARGS:-'-m "gpu and {backend} and not high_l and (not slow or l_cut)"'}

# The compile sample's `min` band (FJ_MIN_L_MAX for that phase only): every
# compile case at L <= this runs, the rest are drawn. 0, not the sampler's 2,
# because the compile cases at L <= 2 are ~420 per compiler, which measured
# most of half an hour on a 4-core cloud box (the gpu sample keeps 2: those
# are ~110 per card, and the card, not the compiler, is the cost).
# At 0, and the sampler's budgets, the phase took 7m57s (371 cases) on a
# 4-core cloud box where the sample_compiles selection it replaces took
# 7m02s (272 cases), back to back, 2026-10-30.
GATE_COMPILE_MIN_L_MAX=${GATE_COMPILE_MIN_L_MAX:-0}

# --- measurement ------------------------------------------------------------
#
# devtools/measure.sh runs everything that MEASURES rather than checks: the
# timed and profiled sweeps, the probes, the recompiles against the stored
# rows, and the checks that data/measured/ is what its rows recompute to.
# Those live in MEASURE_DIR, which pyproject's `testpaths` does not name, so
# no correctness run (bare pytest, the push gate, the kernel tier) collects
# them. Files there are `measure_*.py`; measure.sh passes MEASURE_PYTEST_ARGS
# so pytest collects that spelling.
MEASURE_DIR=${MEASURE_DIR:-measure}
MEASURE_PYTEST_ARGS=${MEASURE_PYTEST_ARGS:-'-c pyproject.toml -o "python_files=measure_*.py"'}

# Where each timed sweep writes, when its variable is not already set: the
# TRACKED stores, because a sweep run through measure.sh is a deliberate
# re-measurement (git shows what it moved). Point one elsewhere for a dry run:
#   FJ_WALLCLOCK_STORE=/tmp/k.json devtools/measure.sh wallclock --backend cuda
# (a store must exist before a sweep merges into it: copy the tracked one).
MEASURE_KERNELS_STORE=${MEASURE_KERNELS_STORE:-data/measured/kernels.json}
MEASURE_ROOFLINE_STORE=${MEASURE_ROOFLINE_STORE:-data/measured/roofline.json}
MEASURE_BUILDS_STORE=${MEASURE_BUILDS_STORE:-data/measured/builds.json}
MEASURE_CEILINGS_STORE=${MEASURE_CEILINGS_STORE:-data/measured/ceilings.json}
MEASURE_KBLOCK_STORE=${MEASURE_KBLOCK_STORE:-data/measured/kblock.json}
MEASURE_SCHEME_AB_STORE=${MEASURE_SCHEME_AB_STORE:-data/measured/scheme_ab.json}
MEASURE_KBLOCK_SPLIT_STORE=${MEASURE_KBLOCK_SPLIT_STORE:-data/measured/kblock_split.json}
MEASURE_KBLOCK_ROLLED_STORE=${MEASURE_KBLOCK_ROLLED_STORE:-data/measured/kblock_rolled.json}
MEASURE_DF_LOWSTACK_STORE=${MEASURE_DF_LOWSTACK_STORE:-data/measured/df_lowstack.json}
MEASURE_DF_GRAD_LOWSTACK_STORE=${MEASURE_DF_GRAD_LOWSTACK_STORE:-data/measured/df_grad_lowstack.json}
MEASURE_D2_DF_LOWSTACK_STORE=${MEASURE_D2_DF_LOWSTACK_STORE:-data/measured/d2_df_lowstack.json}
MEASURE_G1E_STORE=${MEASURE_G1E_STORE:-data/measured/g1e.json}
MEASURE_GJAB_STORE=${MEASURE_GJAB_STORE:-data/measured/gjab.json}
MEASURE_GK_STORE=${MEASURE_GK_STORE:-data/measured/gk.json}
MEASURE_KX_STORE=${MEASURE_KX_STORE:-data/measured/kx.json}
MEASURE_HKX_STORE=${MEASURE_HKX_STORE:-data/measured/hkx.json}
MEASURE_HJAB_STORE=${MEASURE_HJAB_STORE:-data/measured/hjab.json}
MEASURE_REGISTRY_COST_STORE=${MEASURE_REGISTRY_COST_STORE:-data/measured/registry_cost.json}

# How a timed or profiled sweep runs: ALONE (no xdist -- a worker compiling
# beside a timing makes it a different experiment), its prints shown, its
# skips named. Single quotes load-bearing, as for FAST_TEST_ARGS below.
MEASURE_TIMED_ARGS=${MEASURE_TIMED_ARGS:-'-p no:xdist -s -rs'}

# --- the AMD device passthrough -------------------------------------------
#
# Getting this right is fiddly and its failure mode is indistinguishable from
# having no card at all, which is why the values live here rather than in each
# caller. NVIDIA needs none of it: `--gpus all` and the container runtime do
# the equivalent.

# The device nodes ROCm needs, one per line. /dev/kfd is the compute driver
# interface and /dev/dri carries the render nodes; both are required.
ROCM_DEVICES=${ROCM_DEVICES:-"/dev/kfd
/dev/dri"}

# The HOST groups owning those nodes, one per line. Both container paths
# resolve these to NUMERIC gids with getent before passing them to
# --group-add, and that matters more than it looks: `--group-add render`
# resolves the name INSIDE the container, where docker/install-rocm.sh made
# `render` gid 110 while the bind-mounted nodes keep their host ownership (109
# on this box). The container user then joins a group the devices do not grant,
# the container starts clean, and the first device call fails with
# "hipErrorNoDevice (no ROCm-capable device is detected)" -- which reads
# exactly like a box with no GPU, and has been misread as one.
#
# kernel-tier.sh --backend hip builds its own --group-add args from this list
# and fails loudly when a name does not resolve. devcontainer.sh instead
# exports one FJ_<NAME>_GID per entry (render -> FJ_RENDER_GID) for the
# ${localEnv:...} references in the hip/combined runArgs, and only warns: the
# json carries a fallback gid so that opening it WITHOUT this script still
# passes something. Adding a name here reaches the first automatically; the
# second also needs the reference written into the json, since JSON cannot
# loop.
ROCM_GROUPS=${ROCM_GROUPS:-"render
video"}

# The image kernel-tier.sh re-execs into when the host has no hipcc. `:hip`
# rather than `:combined` because it is the smaller of the two and answers the
# HIP tier completely -- `combined` carries the CUDA toolkit this run would not
# touch. Point it at `${PROJECT_NAME}:combined` for a box where only that one
# is built.
ROCM_IMAGE=${ROCM_IMAGE:-${PROJECT_NAME}:hip}

# --- tests ----------------------------------------------------------------

# How the test suite is invoked, as an array so arguments with spaces survive.
# Anything goes here as long as it takes extra arguments on the end and exits
# non-zero on failure. Scripts that need a pytest specifically (prepush)
# resolve it through VENV_PATHS below instead.
TEST_CMD=${TEST_CMD:-pytest}

# Default xdist worker count. 8 rather than `auto` on purpose: several worktree
# containers can be up at once, and `auto` in each would claim every core on
# the box. Set to empty to drop the -n flag entirely.
JOBS=${JOBS:-8}

# The xdist scheduler, passed as `--dist` beside JOBS; empty = xdist's own
# default (`load`). `worksteal` because the suite's cost is lopsided -- a few
# tests take 10-20 s among ~11,000 that take milliseconds -- and `load`
# hands out work up front, so one worker finishes its queue long after the
# rest are idle: the fast tier took 241 s under `load` and 210 s under
# `worksteal` on this box (JOBS=8).
XDIST_DIST=${XDIST_DIST:-worksteal}

# Host CPU bounds for a devcontainer, applied by devcontainer.sh on `up` and
# `rebuild`. Both empty = unbounded, and nothing is applied.
#
#   CPUSET  pin the container to specific host cores, e.g. 0-11
#   CPUS    cap it at n cores' worth of CPU quota, e.g. 8
#
# These bound EVERYTHING the container runs -- a device compiler, a test
# runner, anything a shell inside it starts -- where JOBS above only bounds the
# test runner. Two worktree containers pinned to disjoint cores (CPUSET=0-11
# here, CPUSET=12-23 there) physically cannot contend, which is what makes a
# TIMING measurement in one of them mean anything -- and this project's whole
# claim is about kernel throughput, so a benchmark run without these is a
# number not worth recording.
CPUSET=${CPUSET:-}
CPUS=${CPUS:-}

# Extra marker expression for the default run -- and for the push gate, which
# passes this before PREPUSH_PATHS.
#
# IT EXCLUDES EVERY MARKER THAT DRIVES A COMPILER OR A CARD, which is what
# makes the gate's promise true rather than aspirational: `cuda`/`hip` invoke
# nvcc or hipcc, and inside a container those are PRESENT, so without this the
# gate would quietly start compiling device code and cost seconds-to-minutes
# depending on which side you pushed from. The kernel tier is where that
# belongs (devtools/kernel-tier.sh), and it selects the same markers from the
# other direction (KERNEL_TEST_ARGS above). It also excludes `slow`, which
# pyproject.toml's marker list promises is "kept off the pre-push gate": a
# slow test re-runs an exhaustive search (minutes), and is run by hand or
# with plain `pytest`.
#
# NB THE SINGLE QUOTES ARE LOAD-BEARING: inside ${VAR:-default} bash strips one
# level of quoting, so an unquoted `-m "not gpu"` would reach pytest as the
# three words `-m`, `not`, `gpu` -- a marker expression of `not` and a path
# called `gpu`, which collects nothing and says only "no tests ran".
FAST_TEST_ARGS=${FAST_TEST_ARGS:-'-m "not cuda and not hip and not gpu and not slow and not oracle"'}

# What the pre-push gate runs. Deliberately a curated list rather than the
# whole suite: the gate should cost seconds, so anything driving a device
# compiler, a GPU or the network belongs in the kernel tier or CI instead. The
# value is passed to pytest verbatim after FAST_TEST_ARGS, so paths, node ids
# and -k expressions all work. Empty means "whatever pytest collects by
# default", which is right while the suite is small.
PREPUSH_PATHS=${PREPUSH_PATHS:-}

# --- environment ----------------------------------------------------------

# Where to look for the project's virtualenv, in order; the first one that
# WORKS wins. `.venv` is what `uv sync` makes on the host, `/opt/venv` is what
# the Dockerfiles bake into the images.
#
# The order is host-first on purpose, and it is safe in a container even though
# `.venv` is visible there through the workspace mount: lib.sh checks that a
# venv's interpreter actually runs before taking tools from it, and a
# host-built `.venv` points at a host interpreter the image does not have. See
# `_venv_runnable` there for the failure that check exists to prevent.
VENV_PATHS=${VENV_PATHS:-.venv /opt/venv}

# Optional tools doctor.sh reports on, one `name:what is lost without it` per
# line. A missing one is a WARN (a slice of the suite will skip), never a FAIL.
#
# Most of this list is absent on the HOST and present in a container, which is
# the point: run doctor.sh on both sides and the warnings tell you which half
# of the workflow you are on. A host with neither device compiler is a
# perfectly good place to write the generator, lint it, and run the fast tier.
DOCTOR_OPTIONAL_TOOLS=${DOCTOR_OPTIONAL_TOOLS:-"docker:the devcontainer, the compose services and the worktree containers
gh:the PR flow (also needs GH_TOKEN)
nvcc:compiling a generated kernel for NVIDIA -- the whole CUDA half of the kernel tier
hipcc:compiling a generated kernel for AMD; present only in the \`hip\`/\`combined\` image
clang-format:formatting generated kernel text so a golden file stays reviewable
sccache:warm recompiles of generated kernels; without it every kernel-tier run is a cold compile
compute-sanitizer:the GPU memcheck/racecheck sweep (docker compose run --rm compute-sanitizer)
nsys:Nsight Systems timelines (launch order, overlap, host gaps) -- a timeline, NOT counters: it gives no FLOP or DRAM-byte count, so no arithmetic intensity
ncu:Nsight Compute counters on NVIDIA -- fp64 FLOPs and DRAM bytes per kernel (felizjueves.performance.measured.profile), i.e. every sm_86 counter row and its arithmetic intensity; devtools/measure.sh roofline cannot run and the cuda counter witness of measure.sh probes (measure/measure_toolchain.py) SKIPS
rocprofv3:counter mode on AMD -- the two hip counter witnesses of devtools/measure.sh probes (measure/measure_toolchain.py) and its per-clock check of gfx1200's fp64 ceiling (GRBM_GUI_ACTIVE, #141) SKIP; loaded, it still gives gfx1200 no counter row (gfx12 defines no fp64 counter, and the byte counters read 0 on this box: measured_store.UNMEASURED, final since #140)
cmake:building a reference implementation from source to compare against
ninja:the generator cmake drives for that
shellcheck:the devtools/ and docker/ lint pass (pre-commit brings its own on the host)"}

# Which GPU vendors doctor.sh probes for a LIVE DEVICE, space-separated:
# `nvidia`, `amd`, or empty to drop the section entirely.
#
# This is about the DEVICE, not the toolchain -- the two are independent and
# both matter. A device-less box cannot run a kernel, so every suite that
# compares a generated kernel against the host reference is lost there, while a
# card with no toolchain is the normal host state. Both vendors are listed
# because this box has one of each, and a generated kernel has to satisfy both.
DOCTOR_GPU_VENDORS=${DOCTOR_GPU_VENDORS:-"nvidia amd"}

# The card's lock: one file per backend, `{backend}` replaced by `cuda` or
# `hip`. test/conftest.py's device fixtures take it -- `flock` on a descriptor
# the kernel closes when the run exits, Ctrl-C included -- so every `gpu` test
# of one pytest run is exclusive on its card and a second run waits, saying for
# whom; doctor.sh's card-state row reports whether it is held. It is only as
# shared as the directory: /tmp inside a container is not the host's /tmp, so
# runs on the two sides of a container boundary do not see each other's lock.
# Single quotes load-bearing, as for KERNEL_TEST_ARGS: unquoted, the `}` of
# `{backend}` closes the ${...}.
GPU_LOCK_PATH=${GPU_LOCK_PATH:-'/tmp/fj-gpu-{backend}.lock'}

# Paths that must exist for some slice of the suite to run, one
# `path:what skips without it` per line. A RELATIVE path is resolved against
# the repo root.
#
# EMPTY, and the section disappears from doctor's output when it is. The
# obvious candidate was a local copy of the reference papers, and it was
# deliberately dropped: nothing in this repo reads a PDF, so their absence
# degrades no tier, and docs/references.md cites every paper by DOI rather than
# by a path on one machine. Add an entry here for a fixture tree or a submodule
# some slice of the suite genuinely needs.
DOCTOR_REQUIRED_PATHS=${DOCTOR_REQUIRED_PATHS:-}

# --- the agent in the images -----------------------------------------------

# Where the Claude Code version pin lives, one path per line, relative to the
# repo root: every file carrying an `ARG CLAUDE_CODE_VERSION=` line. These are
# the GPU Dockerfiles, each of which runs docker/install-claude-code.sh as a
# leaf layer -- NOT Dockerfile.base, because a bump there would rebuild the
# CUDA toolkit and the ROCm install downstream of it. See AGY_PIN_FILES below
# for the other agent, pinned the same way in the same three files.
#
# devtools/claude-version.sh reads, compares and rewrites exactly these, and
# doctor.sh warns when they disagree or have fallen behind the npm registry.
# Set it empty to drop both checks; a path listed here that does not exist is
# an error rather than a skip, because a Dockerfile renamed out of this list IS
# the drift.
CLAUDE_PIN_FILES=${CLAUDE_PIN_FILES:-"docker/Dockerfile.cuda
docker/Dockerfile.hip
docker/Dockerfile.combined"}

# The npm package those files install, and how long to wait on the registry
# before giving up and reporting the pin alone.
CLAUDE_NPM_PACKAGE=${CLAUDE_NPM_PACKAGE:-@anthropic-ai/claude-code}
CLAUDE_REGISTRY_TIMEOUT=${CLAUDE_REGISTRY_TIMEOUT:-8}

# --- the OTHER agent in the images ------------------------------------------

# Where the Antigravity CLI pin lives, one path per line: every file carrying
# an `ARG AGY_RELEASE=` line. The same three GPU Dockerfiles as
# CLAUDE_PIN_FILES above, and for the same reason -- install-agy.sh is a leaf
# layer, not a Dockerfile.base one, so a bump never invalidates the toolkits.
#
# devtools/agy-version.sh reads, compares and rewrites exactly these, and
# doctor.sh warns when they disagree or have fallen behind upstream. Set it
# empty to drop both checks; a listed path that does not exist is an error.
AGY_PIN_FILES=${AGY_PIN_FILES:-"docker/Dockerfile.cuda
docker/Dockerfile.hip
docker/Dockerfile.combined"}

# TWO pinned lines per file, not one: the sha512 belongs to the exact tarball
# the release names. agy-version.sh treats them as one unit and refuses to
# rewrite either alone, because a release bumped without its hash is a build
# that fails its own checksum -- which reads like a compromised download.
AGY_RELEASE_ARG=${AGY_RELEASE_ARG:-AGY_RELEASE}
AGY_SHA512_ARG=${AGY_SHA512_ARG:-AGY_SHA512}

# Upstream serves ONE release -- the current one -- from a per-platform
# manifest carrying its version, tarball URL and sha512 together. That triple
# arriving from a single document is what makes `--apply` safe: the hash is
# never paired with a release it does not belong to.
#
# TWO platform spellings, and they are not interchangeable: the manifest is
# keyed `linux_amd64` (upstream's installer's own naming), while the storage
# path segment and tarball are spelled `linux-x64`. install-agy.sh reads the
# second as AGY_PLATFORM; this is the first.
AGY_MANIFEST_BASE=${AGY_MANIFEST_BASE:-https://antigravity-cli-auto-updater-974169037036.us-central1.run.app/manifests}
AGY_MANIFEST_PLATFORM=${AGY_MANIFEST_PLATFORM:-linux_amd64}
AGY_REGISTRY_TIMEOUT=${AGY_REGISTRY_TIMEOUT:-8}

# How long `agy-version.sh --verify` may spend pulling the pinned tarball. It
# is ~62MB compressed (a 214MB binary), so this is a download timeout rather
# than an API one -- nothing on the report path uses it.
AGY_DOWNLOAD_TIMEOUT=${AGY_DOWNLOAD_TIMEOUT:-900}
