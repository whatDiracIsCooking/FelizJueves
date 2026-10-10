#!/usr/bin/env bash
# Install the CUDA toolkit from NVIDIA's apt repository.
#
# Called by docker/Dockerfile.cuda, and inherited (already installed) by
# docker/Dockerfile.combined, which builds on the CUDA image. Still a script
# rather than an inline RUN, to match install-rocm.sh -- which has to be one,
# since `combined` runs it a second time.
#
# Why apt and not the nvidia/cuda base image: this repo builds a HIP image too,
# and the shared docker/Dockerfile.base cannot be both nvidia/cuda and
# rocm/dev-ubuntu at once. Installing the toolkit on top of plain Ubuntu is
# what lets `base` stay vendor-neutral. The nvidia/cuda images are themselves
# Ubuntu plus these same packages, so nothing is lost -- except the NVIDIA_*
# runtime env those images set, which Dockerfile.cuda sets explicitly instead.
#
# THE TOOLKIT AND NOTHING ELSE. NCCL, cuTENSOR and nvCOMP used to be installed
# here, and came out with the rest of the linear-algebra stack: no kernel this
# project generates links any of them, and cuTENSOR alone was ~1GB. Add one
# back only alongside the code that links it.
set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:?CUDA_VERSION must be set, e.g. 13-0}"
TARGETARCH="${TARGETARCH:-amd64}"

# NVIDIA publishes one repo directory per CPU architecture, under names that do
# not match Docker's TARGETARCH spelling.
case "$TARGETARCH" in
  amd64) repo_arch=x86_64 ;;
  arm64) repo_arch=sbsa ;;
  *) echo "unsupported TARGETARCH: '$TARGETARCH'" >&2; exit 1 ;;
esac

repo="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${repo_arch}"

# The keyring package registers both the signing key and the repo definition,
# which is why no sources.list line is written by hand here.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
wget -q -O "$tmp/cuda-keyring.deb" "${repo}/cuda-keyring_1.1-1_all.deb"
dpkg -i "$tmp/cuda-keyring.deb"

apt-get update
# The VERSIONED meta-package, not `cuda` or `cuda-toolkit`: those pull the
# driver, which must come from the host through the NVIDIA container runtime.
# A driver installed inside the image conflicts with the injected one.
apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA_VERSION}"

# /usr/local/cuda is a symlink the packages maintain, so a 13.0 -> 13.1 bump
# does not leave anything pointing at a directory that no longer exists.
test -x /usr/local/cuda/bin/nvcc
/usr/local/cuda/bin/nvcc --version | grep release

# Prove the compiler can produce device code for the architecture this image
# targets, rather than merely that the binary exists. A missing or mismatched
# nvcc fails HERE, at build time, instead of on the kernel tier's first
# generated translation unit -- and `-cubin` needs no card, since nvcc targets
# an architecture and not the device present.
# CUDA_ARCH may be a CMake-style LIST -- Dockerfile.cuda documents
# `--build-arg CUDA_ARCH="80;86;90"` for a cluster build -- and `sm_80;86;90`
# is not a thing nvcc accepts. The probe only needs one real architecture to
# prove the toolchain works, so take the first entry.
cuda_arch="${CUDA_ARCH:-86}"
cuda_arch="${cuda_arch%%;*}"
printf '%s\n' \
  '__global__ void probe(float* p) { p[threadIdx.x] = 1.0f; }' \
  > "$tmp/probe.cu"
/usr/local/cuda/bin/nvcc -arch="sm_${cuda_arch}" -cubin \
  -o "$tmp/probe.cubin" "$tmp/probe.cu"
test -s "$tmp/probe.cubin"
echo "install-cuda.sh: nvcc compiles device code for sm_${cuda_arch}"
