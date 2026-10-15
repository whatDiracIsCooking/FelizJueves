# syntax=docker/dockerfile:1.7
# ============================================================================
# hip -- base + the ROCm/HIP toolchain, with no CUDA toolkit at all
# ============================================================================
# A HIP build needs no CUDA, which is the point of this being a SIBLING of
# Dockerfile.cuda rather than a layer on it: neither backend should pay for the
# other's SDK.
#
#   docker/build.sh hip       # builds base first, then this; tags :hip
#
# By hand, and the parent must already be tagged (see Dockerfile.base, "WHY THE
# FOUR FILES CHAIN BY TAG"):
#
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.base -t fj:base .
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.hip  -t fj:hip .
#
# Both from the REPO ROOT: the trailing `.` is the context, and this file
# bind-mounts docker/install-rocm.sh from it.
#
# This box has a real AMD card (an RX 9060 XT, RDNA4/Navi 44, LLVM target
# gfx1200), so this image is not a compile-only check the way it would be on a
# single-vendor machine -- kernels can actually be dispatched in here, and
# `devtools/kernel-tier.sh --hip` is the run that proves a generated kernel
# agrees with the reference on that hardware.
ARG PROJECT_NAME=fj
ARG PARENT_IMAGE=${PROJECT_NAME}:base
FROM ${PARENT_IMAGE}

ARG ROCM_VERSION=7.2.4
USER root

# Re-stamped rather than merely inherited from the parent, so that the
# PROJECT_NAME the devcontainer passes to THIS build is consumed where it is
# declared. See Dockerfile.base for what the label is for.
ARG PROJECT_NAME
LABEL devcontainer.project="${PROJECT_NAME}"

# The LLVM target this image compiles device code for. gfx1200 is RDNA4 (Radeon
# RX 9060 XT), the card in this box. Override for another:
#   --build-arg GPU_TARGETS=gfx1100
#
# Declared BEFORE the install because install-rocm.sh's probe compiles for it:
# a target whose device bitcode is missing fails the BUILD rather than the
# first kernel-tier run. Exported as both spellings because a CMake project
# built in here (a reference implementation from docs/references.md) may read
# either.
ARG GPU_TARGETS=gfx1200
ENV GPU_TARGETS=${GPU_TARGETS} \
    AMDGPU_TARGETS=${GPU_TARGETS} \
    HIP_ARCH=${GPU_TARGETS} \
    ROCM_PATH=/opt/rocm

# The package set is the COMPILER AND THE PROFILERS, not AMD's full math SDK --
# install-rocm.sh's header has the list, why ~19GB of hipBLAS/rocBLAS/CK is not
# in it, and the one build arg that puts it back.
ARG ROCM_PACKAGES=""
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=bind,source=docker/install-rocm.sh,target=/tmp/install-rocm.sh \
    ROCM_VERSION="${ROCM_VERSION}" GPU_TARGETS="${GPU_TARGETS}" \
    ${ROCM_PACKAGES:+ROCM_PACKAGES="${ROCM_PACKAGES}"} \
    bash /tmp/install-rocm.sh

# /opt/rocm/bin goes LAST, after /usr/bin, deliberately: hipcc, hipconfig and
# llvm-objdump are then on PATH by name, while the SYSTEM clang++ still wins a
# bare `clang++` -- ROCm ships its own clang, and a host compile that silently
# used it would not be the host compiler the CUDA image uses.
ENV PATH=${PATH}:/opt/rocm/bin

# Claude Code, PINNED -- the last root layer in this file on purpose: a version
# bump re-runs only this and the two lines after it, never the SDK above it.
# docker/install-claude-code.sh carries the rest. The same ARG is declared by
# Dockerfile.cuda, .hip and .combined, so bump it with
# `devtools/claude-version.sh --apply`, which rewrites all three at once --
# doctor.sh warns when they have drifted apart.
ARG CLAUDE_CODE_VERSION=2.1.300
ENV CLAUDE_CODE_VERSION=${CLAUDE_CODE_VERSION}
RUN --mount=type=bind,source=docker/install-claude-code.sh,target=/tmp/install-claude-code.sh \
    bash /tmp/install-claude-code.sh

USER ubuntu
RUN hipconfig --version && hipcc --version | head -1
