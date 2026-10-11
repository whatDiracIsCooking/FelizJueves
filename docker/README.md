# docker/

Four Dockerfiles, one diamond, and two front ends onto them.

```
                Dockerfile.base
                 /           \
  Dockerfile.cuda             Dockerfile.hip
            |                       :
  Dockerfile.combined ..............:   (reuses install-rocm.sh, not the hip image)
```

| File | What it is |
|---|---|
| `Dockerfile.base` | The vendor-neutral toolchain: g++ and clang, CMake and Ninja (pinned), sccache, `gh`, and a uv-managed venv at `/opt/venv`. No GPU SDK. |
| `Dockerfile.cuda` | `base` + the CUDA toolkit (`nvcc`), `sm_86`, Nsight Systems. **The default**, and what the devcontainer and compose build. |
| `Dockerfile.hip` | `base` + ROCm's HIP compiler and profilers, `gfx1200`. No CUDA at all. |
| `Dockerfile.combined` | `cuda` + ROCm. Both device compilers in one container. |

## Building

The files chain by **tag**, not by stage: each child opens `FROM
${PARENT_IMAGE}`, so the parent has to exist and be tagged first. `build.sh`
walks the chain — **use it rather than `docker build`**:

```bash
docker/build.sh base          # fj:base
docker/build.sh cuda          # base, then cuda -> fj:cuda AND fj:latest
docker/build.sh hip           # base, then hip  -> fj:hip
docker/build.sh combined      # base, cuda, then combined -> fj:combined
docker/build.sh cuda --no-cache   # extra flags pass through to every step
```

Build a child by hand with no parent tagged and docker does not fall back to
building it — it tries to **pull**, and fails with `pull access denied for fj`,
which reads like a registry problem rather than a missing local build.

`BUILD_DRY_RUN=1` prints every command and runs none of them.

## Build args

Defaults live in the Dockerfiles, never in `build.sh`; it forwards only names
set in the environment, so `docker/build.sh` with none reproduces the pinned
build.

| Arg | Default | Where | Notes |
|---|---|---|---|
| `CUDA_VERSION` | `13-0` | `Dockerfile.cuda` | **Pick against the host driver, not the newest CUDA.** A driver whose ceiling is lower makes the container refuse to start: `unsatisfied condition: cuda>=X.Y`. This box is driver 580.126.20, ceiling 13.0. |
| `CUDA_ARCH` | `86` | `Dockerfile.cuda` | RTX 3080. Widen for a cluster: `--build-arg CUDA_ARCH="80;86;90"`. CUDA 13's nvcc floor is `compute_75`. |
| `ROCM_VERSION` | `7.2.4` | `Dockerfile.hip`, `.combined` | Declared in both — bump together. |
| `GPU_TARGETS` | `gfx1200` | `Dockerfile.hip`, `.combined` | RX 9060 XT (RDNA4). `install-rocm.sh` compiles a probe for it, so a target with no device bitcode fails the **build**. |
| `ROCM_PACKAGES` | *(the compiler + profilers)* | `Dockerfile.hip`, `.combined` | See below. |
| `CLAUDE_CODE_VERSION` | pinned | all three GPU files | Bump with `devtools/claude-version.sh --apply`, which rewrites all three. |
| `UBUNTU_TAG`, `CMAKE_VERSION`, `NINJA_VERSION`, `SCCACHE_VERSION` | pinned | `Dockerfile.base` | |
| `GH_VERSION` | `2.102.0` | `Dockerfile.base` | The GitHub CLI, for the PR / issue / milestone skills. Upstream tarball, not noble's 2.45.0. Bumping it reparents every GPU child, so the CUDA and ROCm installs rebuild — bump it when a flow needs the newer `gh`. |

### What is deliberately *not* installed

These images are much smaller than the linear-algebra project this tooling came
from, and the omissions are the point — **add a library only with the code that
links it, and say what does:**

- **No reference LAPACK, no cuTENSOR, NCCL, nvCOMP, cuGraph, hipCOMP.** Nothing
  here links them. The host-side oracle is numpy.
- **No clang-20 + libc++ module manifest.** There is no C++23 module build.
- **ROCm is the compiler, not the SDK.** `rocm-hip-sdk` resolves to ~19GB of
  hipBLAS/rocBLAS/rocFFT/composable-kernel that no generated kernel calls, so
  `install-rocm.sh` installs `rocm-hip-runtime-dev` plus `rocminfo`, the two SMI
  libraries, roctracer, rocprofiler-sdk and rocm-gdb instead. This replaced an
  earlier `ROCM_PRUNE` flag that installed the SDK and deleted ~12GB
  afterwards — which shrinks nothing, since docker layers are additive. To get
  the math libraries back: `--build-arg ROCM_PACKAGES="rocm-hip-sdk
  rocm-hip-runtime-dev"`.
- **sccache stays, and it is the one thing kept against that grain.** It caches
  nvcc and hipcc output, and the kernel tier recompiles the same generated
  translation units across runs.

## The two front ends

Both run the same images, so a result in one means the same as in the other.

**`.devcontainer/` + `devtools/devcontainer.sh`** — interactive and
agent-facing:

```bash
devtools/devcontainer.sh up|rebuild|shell|test|down
devtools/devcontainer.sh --hip shell        # or --combined
```

Use `devcontainer.sh`, not VS Code's "Reopen in Container": it resolves
`FJ_GIT_DIR`, `FJ_SCCACHE_DIR` and the device group ids before the CLI runs. The
literals in the json are only a fallback, and they are this box's values.

**`compose.yaml`** — one-shot batch runs that tee to `.log/` and exit:

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)      # required, not defaulted
docker compose -f docker/compose.yaml run --rm test
docker compose -f docker/compose.yaml run --rm kernel        # nvcc + the card
docker compose -f docker/compose.yaml run --rm kernel-hip    # hipcc + the card
docker compose -f docker/compose.yaml run --rm compute-sanitizer
docker compose -f docker/compose.yaml run --rm shell
```

`HOST_UID`/`HOST_GID` are required rather than defaulted because a default of
1000 on a host where you are not uid 1000 writes root-owned files into the bind
mount, and that surfaces much later. The compose project is named `fj`
explicitly — the default would be the directory name, `docker`, which another
repo on this box would share.

| Variable | Default | What it selects |
|---|---|---|
| `FJ_IMAGE` | `fj:latest` | the CUDA image for most services |
| `FJ_HIP_IMAGE` | `fj:hip` | `kernel-hip` — must be an image with ROCm (`:hip` or `:combined`) |
| `FJ_SCCACHE_HOST_DIR` | `../.sccache` | point at the primary checkout's store from a worktree |
| `FJ_RENDER_GID`, `FJ_VIDEO_GID` | `109`, `44` | this box's `getent group render video` |
| `COMPUTE_SANITIZER_TOOL` | `memcheck` | `racecheck` for the other sweep |

## The AMD passthrough, and its one failure mode

NVIDIA needs `--gpus all` and the container runtime does the rest. AMD has no
equivalent: `/dev/kfd` and `/dev/dri` are passed in directly, and the container
user must be in the **host** groups that own them.

**Always pass a numeric gid, never a group name.** `--group-add render` resolves
`render` *inside* the container, where `install-rocm.sh` created it as gid 110,
while the bind-mounted nodes keep their host ownership (109 here). The container
then starts cleanly and the first device call fails with
`hipErrorNoDevice (no ROCm-capable device is detected)` — which reads exactly
like a machine with no AMD card, and has been misread as one.
`devtools/devcontainer.sh` and `devtools/kernel-tier.sh` both resolve these with
`getent`; `compose.yaml` reads `FJ_RENDER_GID`/`FJ_VIDEO_GID`.

## Files

| | |
|---|---|
| `build.sh` | walks the chain; `IMAGE_TAG_SUFFIX`, `IMAGE_REGISTRY`, `BUILD_PUSH` for a published variant |
| `install-cuda.sh` | the CUDA toolkit from apt, then an `nvcc -cubin` probe for `CUDA_ARCH` |
| `install-rocm.sh` | the ROCm repo and keyring, the package set above, the loader path, the device groups, then a `hipcc` compile-and-link probe for `GPU_TARGETS` |
| `install-claude-code.sh` | Node 22 and Claude Code at an explicit version, with a check that what landed is what was asked for |
| `compose.yaml` | the batch services |
