#!/usr/bin/env bash
# Install the ROCm/HIP toolchain from AMD's apt repository, then prove hipcc
# can compile and link device code for the target this image is built for.
#
# Called by docker/Dockerfile.hip, and AGAIN by docker/Dockerfile.combined on
# top of the CUDA image. A script rather than an inline RUN so the block is
# written once -- Dockerfiles have no include, and `combined` cannot inherit
# from two parents, so it reuses THIS instead of the hip image (see that file's
# header).
#
# ---------------------------------------------------------------------------
# THE PACKAGE SET IS THE COMPILER, NOT THE SDK -- and that is the deliberate
# part of this file
# ---------------------------------------------------------------------------
# `rocm-hip-sdk` is AMD's meta-package for "the HIP SDK": one name that tracks
# whatever AMD decides belongs in it, and it resolves to ~19GB across ~60
# packages -- hipBLAS, hipBLASLt, hipSOLVER, hipSPARSE, hipFFT, rocBLAS,
# rocSOLVER, rocFFT, composable_kernel, rocWMMA, hipTensor, RCCL and the rest.
# This project links NONE of it: the kernels here are generated from
# recurrence relations and call no vendor math library. So the install is the
# compiler plus the tools, and the surplus is simply not downloaded -- which is
# also why there is no ROCM_PRUNE in this file any more. Pruning existed to
# delete ~12GB that a full SDK install had already pulled; not installing it is
# strictly better than installing and deleting, since docker layers are
# additive and an `rm` in a later layer shrinks nothing.
#
# WHAT EACH NAME IS FOR:
#   rocm-hip-runtime-dev  hipcc, rocm-llvm, the HIP headers and lib/cmake/hip
#                         -- the compiler itself, and the only package here
#                         without which nothing works. Named explicitly rather
#                         than reached through a meta-package: the ROCm 7.1
#                         series dropped it from rocm-hip-sdk's dependencies,
#                         and the omission surfaces only downstream as CMake's
#                         "Failed to find ROCm root directory".
#   rocminfo              what the device actually is (gfx target, CU count).
#                         The AMD counterpart of reading compute capability.
#   rocm-smi-lib          rocm-smi: clocks, power, VRAM, the older half of the
#   amd-smi-lib           nvml analogue; amd-smi is the modern half.
#   roctracer-dev         the cupti analogue, and
#   rocprofiler-sdk       rocprofv3, its successor -- a J-engine kernel's
#                         occupancy and memory traffic is the measurement this
#                         project exists to move (docs/references.md), so the
#                         profiler is not optional tooling here.
#   rocm-gdb              parity with the cuda-gdb the CUDA toolkit carries.
#
# TO GET THE MATH LIBRARIES BACK, e.g. to benchmark against a hipBLAS-based
# reference: pass ROCM_PACKAGES="rocm-hip-sdk rocm-hip-runtime-dev" as a build
# arg. Say in the commit what links them.
set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:?ROCM_VERSION must be set, e.g. 7.2.4}"
GPU_TARGETS="${GPU_TARGETS:?GPU_TARGETS must be set, e.g. gfx1200}"
ROCM_PACKAGES="${ROCM_PACKAGES:-rocm-hip-runtime-dev rocminfo rocm-smi-lib amd-smi-lib roctracer-dev rocprofiler-sdk rocm-gdb}"

# ONE TARGET, NOT A LIST. hipcc takes a repeated `--offload-arch` flag rather
# than a CMake-style `gfx1200;gfx1100` list, so a list here reaches the compiler
# as one bogus target name and fails with a message about the whole string.
# Rejected up front instead, where the fix is obvious.
case "$GPU_TARGETS" in
  *[';,']*)
    echo "install-rocm.sh: GPU_TARGETS must name ONE target, not '$GPU_TARGETS'." >&2
    echo "  hipcc takes repeated --offload-arch flags, not a list; build one" >&2
    echo "  image per target, or extend the probe below deliberately." >&2
    exit 1 ;;
esac

# AMD spells an X.Y.0 release's repo directory X.Y. Listing
# repo.radeon.com/rocm/apt/ shows 7.0, 7.0.1, 7.0.2, 7.1, 7.2, 7.2.1, 7.2.4 --
# every .0 patch appears WITHOUT its trailing component, and there is no 7.0.0
# directory at all, so a ROCM_VERSION of 7.0.0 404s on the Release file unless
# the .0 comes off here. ONLY the URL is rewritten: ROCM_VERSION stays the full
# MAJOR.MINOR.PATCH everywhere else, because build.sh's tag and doctor's
# comparison against /opt/rocm/.info/version both want the semver.
case "$ROCM_VERSION" in
  *.*.0) ROCM_APT_DIR="${ROCM_VERSION%.0}" ;;
  *)     ROCM_APT_DIR="$ROCM_VERSION" ;;
esac

# A signed-by keyring, not `apt-key add`: apt-key is deprecated in 24.04 and
# removed in 25.04.
#
# Only repo.radeon.com/rocm is added, NOT repo.radeon.com/amdgpu. That second
# repo ships the DKMS kernel driver and AMD's libdrm fork, and neither belongs
# in a container -- the kernel driver comes from the host.
#
# The pin is load-bearing, not decoration. repo.radeon.com republishes its own
# builds of packages that also exist in Ubuntu; at equal priority apt is free
# to prefer either, and which one it picks can change between base-image
# refreshes. Priority 600 makes it deterministic -- ROCm's repo wins for
# anything it publishes. `o=repo.radeon.com` is the Origin its Release file
# declares.
install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
  | gpg --batch --yes --dearmor -o /etc/apt/keyrings/rocm.gpg
chmod 0644 /etc/apt/keyrings/rocm.gpg

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_APT_DIR} noble main" \
  > /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
  > /etc/apt/preferences.d/rocm-pin-600

apt-get update
# shellcheck disable=SC2086  # the word splitting is the point: a package list
apt-get install -y --no-install-recommends $ROCM_PACKAGES

# ROCm unpacks to /opt/rocm-<version> and rocm-core provides the /opt/rocm
# symlink. Nothing in the deb set registers that prefix with the dynamic
# loader, so without this every hipcc-linked binary -- the probe below
# included, and everything the kernel tier builds -- would need an explicit
# RPATH. One conf file settles it image-wide.
echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf
ldconfig
test -L /opt/rocm

# Device access. /dev/kfd (the compute node) and /dev/dri/renderD* are
# root:render; /dev/dri/card* is root:video. Both groups have to exist INSIDE
# the container for `ubuntu` (uid 1000, and devcontainer.json's remoteUser) to
# open those nodes once they are passed in with `--device=/dev/kfd
# --device=/dev/dri`.
#
# THE GID HERE IS NOT THE HOST'S, and must not be made to match one: the
# bind-mounted nodes keep their HOST ownership (render is gid 109 on this box,
# video 44), so what grants access is `--group-add <host-gid>` at run time --
# which is what the devcontainer runArgs and devtools/devcontainer.sh resolve
# with getent. Passing `--group-add render` by NAME instead resolves the name
# in HERE, joins a group the devices do not grant, and the first device call
# fails with hipErrorNoDevice -- indistinguishable from a box with no AMD card.
# Never rebuild the image to match a host.
(getent group render >/dev/null || groupadd -g 110 render || groupadd render)
(getent group video >/dev/null || groupadd video)
usermod -aG render,video ubuntu

# --- validation: compile AND LINK real device code for this target ----------
#
# Deliberately a compile rather than a `test -f` sweep. It proves the headers
# agree with each other under hipcc, that rocm-device-libs carries bitcode for
# ${GPU_TARGETS} SPECIFICALLY (a missing target fails here, at build time,
# rather than on the kernel tier's first run), and that libamdhip64 resolves at
# link time through the ld.so.conf entry above. No GPU access is needed: like
# nvcc, hipcc targets an architecture, not the card present.
#
# It links the HIP runtime and nothing else, which is the same statement the
# package list above makes -- when a math library joins ROCM_PACKAGES because
# something links it, add it here too, or this probe stops covering the image.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf '%s\n' \
  '#include <hip/hip_runtime.h>' \
  '__global__ void probe(float* p) { p[threadIdx.x] = 1.0f; }' \
  'int main() {' \
  '  float* p{};' \
  '  hipMalloc(&p, sizeof(float) * 32);' \
  '  probe<<<dim3(1), dim3(32), 0, nullptr>>>(p);' \
  '  return 0;' \
  '}' \
  > "$tmp/probe.hip"
/opt/rocm/bin/hipcc -std=c++17 --offload-arch="${GPU_TARGETS}" \
  "$tmp/probe.hip" -o "$tmp/probe"
test -x "$tmp/probe"

/opt/rocm/bin/hipconfig --version
/opt/rocm/bin/hipcc --version | head -3
test -x /opt/rocm/bin/rocminfo
test -x /opt/rocm/bin/rocm-smi
echo "=== ROCm ${ROCM_VERSION}: HIP compiles and links for ${GPU_TARGETS} ==="
