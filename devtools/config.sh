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
NVCC_FLAGS=${NVCC_FLAGS:--O3 -lineinfo --std=c++17}
HIPCC_FLAGS=${HIPCC_FLAGS:--O3 -g1 -std=c++17}

# How the kernels get generated. The tier runs this, from the repo root, with
# the project venv first on PATH, and expects the emitted sources in
# $KERNEL_TIER_OUT (it also exports KERNEL_TIER_BACKEND and KERNEL_TIER_ARCH;
# kernel-tier.sh's header has the contract). Word-split, not eval'd: no
# quoting, no `VAR=value` prefix -- hence `env`.
#
# felizjueves.emit's CLI reads those three variables as its defaults and
# writes one translation unit per class, L = 0..16 (`build_pq_kernel(L)`),
# in the backend's own spelling. PYTHONPATH because this is a virtual uv
# project: nothing is installed, src/ is put on the path by hand.
#
# Set it EMPTY (KERNEL_GEN_CMD= devtools/kernel-tier.sh ...) and the tier
# reports its generate and compile phases DORMANT rather than pretending.
KERNEL_GEN_CMD=${KERNEL_GEN_CMD-env PYTHONPATH=src python -m felizjueves.emit}

# Where generated sources and their binaries land. Gitignored (/.kernel-tier/),
# and wiped by --fresh.
KERNEL_BUILD_DIR=${KERNEL_BUILD_DIR:-.kernel-tier}

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
KERNEL_TEST_ARGS=${KERNEL_TEST_ARGS:-'-m "gpu and {backend}"'}

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
FAST_TEST_ARGS=${FAST_TEST_ARGS:-'-m "not cuda and not hip and not gpu and not slow"'}

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
nsys:Nsight Systems profiling -- the memory-bound vs compute-bound measurement this project is chasing
rocprofv3:the AMD counterpart of nsys
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
# the GPU Dockerfiles, each of which runs docker/install-claude-code.sh as its
# last layer -- NOT Dockerfile.base, because a bump there would rebuild the
# CUDA toolkit and the ROCm install downstream of it.
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
