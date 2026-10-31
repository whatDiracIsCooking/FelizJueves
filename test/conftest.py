"""Shared fixtures: find the device toolchains, and compile-and-run through them.

WHAT LIVES HERE AND WHY. Every numerical claim this project will make has one
shape -- emit a kernel, compile it with the vendor's real compiler, run it on
the card, compare against a host reference -- so the machinery for that shape
belongs in one place rather than in each test. `compile_and_run` is that
machinery; the kernel text is the test's business.

THE TWO AXES ARE INDEPENDENT, which is the thing to keep straight: a device
COMPILER targets an architecture and needs no card, while RUNNING needs the
card and not the compiler's target. So `nvcc`/`hipcc` skip on a missing
compiler and `cuda_device`/`hip_device` skip on a missing device, and a test
asks for exactly the ones it needs.

THE CARD GATE (#91) lives in `cuda_device` / `hip_device`, after the presence
probe, and is split by what a test needs -- the two are not the same gate:

- CORRECTNESS (every `gpu` test): EXCLUSIVITY and enough FREE VRAM. The
  fixture takes the card's lock -- `flock` on a descriptor this process holds,
  at devtools/config.sh's GPU_LOCK_PATH -- for the session, so two pytest runs
  serialise instead of both launching, and the waiting one says whom it waits
  for. The kernel drops the lock when that descriptor closes, which every exit
  does (Ctrl-C and SIGKILL included), so a dead run never strands the next. A
  harness that splices in VRAM_PROBE (below) is run once in probe mode first
  (`run_device_binary(..., vram=backend)`), and reports its launch's own
  requirement -- its buffers, plus the largest per-thread stack of its kernels
  times every thread the device can keep resident, which is what the driver
  reserves -- against the free memory its own context sees. Short, and the
  test FAILS with both figures and whatever the vendor tool lists on the card,
  instead of the launch running out of memory. Utilisation is deliberately NOT
  a reason to refuse: a correct answer does not care how fast it arrives, and
  a gate on it would make every correctness test hostage to an outside
  process.
- TIMING (every sweep under measure/, run by devtools/measure.sh -- the
  wall-clock sweeps, the counter sweep, the ceiling probes' recording run):
  the same lock, plus the utilisation wait of `felizjueves.performance.gate` (#109; it
  lived in test_kernel_device.py and here until a second measurement needed
  it). Nothing under test/ times anything.

UNDER XDIST THE CONTROLLER HOLDS THE LOCK and the workers run under it. A
session fixture in each worker cannot hold it: the first worker's lock would
block its siblings until its own session ended, which is the end of the run.
So `pytest_configure` takes the lock in the controller, for each backend whose
card is present and whose `-m` expression can select a `gpu` test of it; a
worker's fixture then finds the lock held by an ANCESTOR (the controller, or a
`flock` wrapped around the whole run) and proceeds without taking it. A holder
that is not an ancestor is another run, and is waited for.

`claim_marked_tests` is the other half of `test/test_claims.py`'s registry
check: every `@pytest.mark.claim(id)` under test/ and measure/, read from
source so that a narrowed run cannot hide one.

measure/conftest.py loads this file by path and re-exports all of it, so the
measurement sweeps (devtools/measure.sh) get these very fixtures, hooks and
card lock.
"""

from __future__ import annotations

import ast
import fcntl
import glob
import itertools
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import devcache
import pytest
import registry
import tiers
from registry import sample

from felizjueves.performance.device import Vendor, from_env

# Architectures the toolchains are driven at, resolved through
# felizjueves.performance.device -- the single Python source, which reads the same
# environment variables and defaults as devtools/config.sh (so a container's
# own pin wins; the images export CUDA_ARCH and GPU_TARGETS) and raises on a
# pin to a card it does not declare. CUDA_ARCH stays the bare compute
# capability ("86") because Toolchain.compile_flags spells the `sm_` itself.
CUDA_ARCH = from_env(Vendor.NVIDIA).name.removeprefix("sm_")
HIP_ARCH = from_env(Vendor.AMD).name


@dataclass(frozen=True)
class Toolchain:
    """A device compiler, and how to drive it for one translation unit."""

    name: str
    path: str
    arch: str
    ext: str

    def compile_flags(self) -> list[str]:
        if self.name == "nvcc":
            return ["-O2", "-std=c++17", f"-arch=sm_{self.arch}"]
        return ["-O2", "-std=c++17", f"--offload-arch={self.arch}"]


def _find(tool: str) -> str | None:
    """Locate a device compiler, including where the SDKs put it off PATH."""
    found = shutil.which(tool)
    if found:
        return found
    for candidate in (f"/usr/local/cuda/bin/{tool}", f"/opt/rocm/bin/{tool}"):
        if os.access(candidate, os.X_OK):
            return candidate
    return None


NVIDIA_GPUS = Path("/proc/driver/nvidia/gpus")


def _iterdir(d: Path) -> list[Path]:
    """A directory's entries, sorted; empty when it is absent or unreadable --
    which is how a device-less box and a container without the mount look."""
    try:
        return sorted(d.iterdir())
    except OSError:
        return []


def _nvidia_enumerated() -> bool:
    # An ENUMERATED GPU, not `/dev/nvidiactl` exists -- which is what this
    # asked before, and it is #431's bug with the vendors swapped. That node is
    # there whenever the nvidia module is LOADED, card or not: a driver
    # installed for a card that has since been pulled, or a module loaded on a
    # box that never had one, leaves it in place. `hip_device` had exactly this
    # shape and every `gpu` test FAILED instead of skipping, against this
    # repo's rule that a missing card skips.
    #
    # /proc/driver/nvidia/gpus/ holds one directory per GPU the driver bound,
    # and the container runtime mounts it under --gpus, so the answer is the
    # same on the host and inside a container and needs nothing installed --
    # the same property that makes amdkfd's topology the right probe for AMD.
    # This is the question lib.sh's nvidia_enumerated asks, which is the point:
    # the shell tiers and the fixtures must not disagree about what is here.
    for entry in _iterdir(NVIDIA_GPUS):
        if entry.is_dir():
            return True
    # The same fallback lib.sh's nvidia_enumerated carries, for a container
    # handed the tool but not that procfs view. Only reached when the view is
    # empty, which on a device-less box is a `which` that finds nothing.
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return False
    try:
        out = subprocess.run([tool, "-L"], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(ln.startswith("GPU ") for ln in out.splitlines())


KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def _amd_enumerated() -> bool:
    # An enumerated GPU node in amdkfd's topology, NOT `/dev/kfd` exists --
    # which is what this asked before #431. That node is there whenever the
    # amdgpu/amdkfd module is LOADED, bound card or not: the module binds by
    # PCI id, so an amdgpu that predates the card (kernel 6.8's in-tree one
    # does not know the RX 9060 XT's 1002:7590) loads and claims nothing while
    # /dev/kfd stays put. `hip_device` then did not skip, so every `gpu` test
    # FAILED on a machine with no usable card -- against this repo's rule that
    # a missing card skips -- and inside a container it read as broken
    # passthrough rather than as an unbound driver on the host.
    #
    # Node 0 is the CPU and reports gfx_target_version 0; a GPU reports its
    # ISA there (gfx1200 -> 120000). Same rule, same field, as doctor.sh and
    # lib.sh's amd_enumerated. Read through the kernel's own topology so the
    # answer needs no ROCm installed and is identical on the host and inside a
    # container given --device=/dev/kfd -- the same reason _nvidia_enumerated
    # reads the driver's procfs instead of calling nvidia-smi.
    try:
        nodes = sorted(KFD_NODES.iterdir())
    except OSError:
        return False
    for node in nodes:
        try:
            props = (node / "properties").read_text()
        except OSError:
            continue
        for line in props.splitlines():
            field, _, value = line.partition(" ")
            if field == "gfx_target_version":
                if value.strip() not in ("", "0"):
                    return True
                break
    return False


@pytest.fixture(scope="session")
def nvcc() -> Toolchain:
    path = _find("nvcc")
    if path is None:
        pytest.skip("nvcc not found -- it lives in the fj:cuda image")
    return Toolchain("nvcc", path, CUDA_ARCH, "cu")


@pytest.fixture(scope="session")
def hipcc() -> Toolchain:
    path = _find("hipcc")
    if path is None:
        pytest.skip("hipcc not found -- it lives in the fj:hip / fj:combined image")
    return Toolchain("hipcc", path, HIP_ARCH, "hip")


# -- the card gate (#91) ------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
# FelizJueves's own headers: what a `gpu`-backend unit's
# `#include "felizjueves/runtime.h"` resolves against, as a consumer's -I.
INCLUDE = REPO / "include"
ENUMERATED = {"cuda": _nvidia_enumerated, "hip": _amd_enumerated}

# The device nodes each backend's runtime must be able to OPEN, as globs.
# devtools/lib.sh's gpu_nodes is the same list; these two are the shell and
# Python readings of one rule, and the reason it is written twice is that the
# fixtures cannot source a shell file per test.
#
# The AMD render nodes are NOT a `/dev/dri/renderD*` glob, and the difference
# matters on a two-vendor box: this one has renderD128 on amdgpu and renderD129
# on nvidia, ROCm never touches the second, and probing it would let an
# NVIDIA-side permissions quirk read as a revoked AMD card. _amd_render_nodes
# picks them by the driver that owns each one.
GPU_NODE_GLOBS = {
    "cuda": ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia[0-9]*"),
    "hip": ("/dev/kfd",),
}
DRM_CLASS = Path("/sys/class/drm")


def _amd_render_nodes() -> list[str]:
    """The DRM render nodes amdgpu owns, by sysfs vendor id (0x1002) -- the
    same field doctor.sh picks the amdgpu card by for its VRAM reading.

    A container with no readable sysfs here yields nothing, and the check then
    probes /dev/kfd alone. That is the right way to degrade: kfd is where the
    AMD-side permissions fault actually bites.
    """
    nodes = []
    for entry in _iterdir(DRM_CLASS):
        if not entry.name.startswith("renderD"):
            continue
        try:
            vendor = (entry / "device" / "vendor").read_text().strip()
        except OSError:
            continue
        if vendor == "0x1002":
            nodes.append(f"/dev/dri/{entry.name}")
    return nodes


def _denied_nodes(backend: str) -> list[str]:
    """The nodes this process may not open -- empty when the card is usable.

    ENUMERATED asks whether the kernel sees a card. This asks the question that
    is not the same one: whether we are still allowed to touch it. A
    container's device grant is lost mid-life -- `--gpus all` lands in
    HostConfig.DeviceRequests and is injected by an OCI hook at container
    START, so a host driver reload, a `systemctl daemon-reload` or a
    `docker update` regenerates the device cgroup without it -- and every node
    stays VISIBLE inside while none of them opens. Enumeration still says yes,
    so without this the card looks present and every launch fails.

    open(O_RDWR) is the probe because it is the exact operation the cgroup
    filter adjudicates, and it needs no CUDA or ROCm installed; cuInit/hipInit
    would answer too, but on a box without the SDK "cannot ask" and "denied"
    become the same answer. Only EXISTING char devices are probed -- a glob
    that matches nothing contributes nothing, which is what keeps a device-less
    box reporting absent rather than denied.

    The Python half of devtools/lib.sh's gpu_denied_nodes, and
    test_device_grant.py holds the two to the same node list.
    """
    nodes = [n for pattern in GPU_NODE_GLOBS[backend]
             for n in sorted(glob.glob(pattern))]
    if backend == "hip":
        nodes += _amd_render_nodes()
    denied = []
    for node in nodes:
        if not Path(node).is_char_device():
            continue
        try:
            os.close(os.open(node, os.O_RDWR))
        except OSError:
            denied.append(node)
    return denied


def _present(backend: str) -> bool:
    """Is there a card here this process can launch on? ENUMERATED AND USABLE.

    The one rule the fixtures ask, and devtools/lib.sh's card_present is its
    shell twin (#712). Folding usability into presence is what makes a revoked
    card SKIP like an absent one instead of failing every `gpu` test: the gate
    then labels the PR `unverified-on-card` and merges, exactly as it does on a
    device-less runner, while doctor.sh and devcontainer.sh still FAIL loudly
    on it -- so nothing is silently unverified, and the suite does not report
    an environment fault as a numerical one. Same move #431 made for AMD.
    """
    return ENUMERATED[backend]() and not _denied_nodes(backend)


# What PRESENT used to be, kept as the name the rest of this file reads.
PRESENT = {b: (lambda b=b: _present(b)) for b in ENUMERATED}


def _no_card_reason(backend: str, absent: str) -> str:
    """The skip message, which has to say WHICH of two things is wrong.

    A revoked card and an absent one both skip, and they need completely
    different things done about them -- so the message that reaches `-rs` names
    the denied nodes and the recreate when that is what happened, rather than
    claiming the machine has no card.
    """
    denied = _denied_nodes(backend) if ENUMERATED[backend]() else []
    if not denied:
        return absent
    return (
        f"{backend}: the card is HERE but this process cannot open "
        f"{', '.join(denied)} -- the device cgroup does not grant it. A "
        "mid-session grant revocation (a host driver reload, a systemd "
        "daemon-reload, or a `docker update` after start), not a missing "
        "card: every node is still visible, which is why the vendor tools "
        "blame the driver. In a container, recreate it -- "
        "`devtools/devcontainer.sh rebuild`; a plain `up` reuses it and "
        "cannot restore the grant. devtools/doctor.sh's `card usable` row "
        "FAILs on this, and `devcontainer.sh gpu` exits 1."
    )
LOCK_REPORT_EVERY_S = 60.0


@cache
def lock_path(backend: str) -> Path:
    """The card's lock file: devtools/config.sh's GPU_LOCK_PATH with
    `{backend}` replaced -- read by sourcing config.sh, so the environment
    still wins, as it does for every other setting there."""
    done = subprocess.run(
        ["bash", "-c", '. devtools/config.sh && printf %s "$GPU_LOCK_PATH"'],
        cwd=REPO, capture_output=True, text=True, check=True)
    spelled = done.stdout.strip()
    assert spelled, "GPU_LOCK_PATH is empty in devtools/config.sh"
    return Path(spelled.replace("{backend}", backend))


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return "?"
    return raw.replace(b"\0", b" ").decode(errors="replace").strip() or "?"


def _ppid(pid: int) -> int:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return 0
    m = re.search(r"^PPid:\s*(\d+)", status, re.MULTILINE)
    return int(m[1]) if m else 0


def _ancestors() -> set[int]:
    seen, pid = set(), os.getppid()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        pid = _ppid(pid)
    return seen


def lock_holders(path: Path) -> list[int]:
    """Pids holding a flock on `path`, from /proc/locks (0 or less: a process
    outside this pid namespace). Waiters (`->` lines) are not holders."""
    try:
        st = path.stat()
        table = Path("/proc/locks").read_text()
    except OSError:
        return []
    want = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    pids = []
    for line in table.splitlines():
        f = line.split()
        if len(f) >= 6 and f[1] == "FLOCK" and f[5] == want:
            pids.append(int(f[4]))
    return pids


def describe_holders(path: Path) -> str:
    pids = lock_holders(path)
    if not pids:
        return "no holder visible in /proc/locks"
    return "; ".join("a process outside this pid namespace" if p <= 0
                     else f"pid {p} ({_cmdline(p)})" for p in pids)


def _say(config: pytest.Config, line: str) -> None:
    """Print past pytest's capture, so a wait is visible while it happens."""
    cap = config.pluginmanager.getplugin("capturemanager")
    if cap is None:
        print(line, file=sys.stderr, flush=True)
        return
    with cap.global_and_fixture_disabled():
        print(line, file=sys.stderr, flush=True)


def take_card_lock(config: pytest.Config, backend: str) -> int | None:
    """Block until this process holds the card's lock; return its descriptor.

    None when an ANCESTOR already holds it (the xdist controller, or `flock`
    wrapped around the run): the lock is this run's, and there is nothing to
    take. The lock is `flock` on an open descriptor, never a file anyone
    removes -- the kernel drops it when the descriptor closes, which every exit
    of this process does, Ctrl-C and SIGKILL included.
    """
    path = lock_path(backend)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    t0 = time.monotonic()
    reported = None
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass
            holders = lock_holders(path)
            if set(holders) & _ancestors():
                os.close(fd)
                return None
            if os.environ.get("PYTEST_XDIST_WORKER") and any(
                    _ppid(h) == os.getppid() for h in holders):
                pytest.fail(
                    f"{path} is held by a sibling xdist worker: the controller "
                    "should have taken it for the whole run (pytest_configure "
                    "in test/conftest.py) -- was this conftest loaded there?")
            waited = time.monotonic() - t0
            if reported is None or waited - reported >= LOCK_REPORT_EVERY_S:
                _say(config, f"card lock: {backend} -- waiting for {path} "
                     f"({waited:.0f} s so far), held by "
                     f"{describe_holders(path)}")
                reported = waited
            time.sleep(1.0)
    except BaseException:
        os.close(fd)
        raise
    os.ftruncate(fd, 0)  # who holds it, for a human; the lock is the flock
    os.write(fd, f"pid {os.getpid()} since {time.ctime()}: "
                 f"{_cmdline(os.getpid())}\n".encode())
    if reported is not None:
        _say(config, f"card lock: {backend} -- took {path} after "
             f"{time.monotonic() - t0:.0f} s")
    return fd


def _may_select(markexpr: str, backend: str) -> bool:
    """Can `-m markexpr` select a test marked `gpu` and `backend`? True iff
    some assignment of the expression's other names satisfies it, with those
    two true and the other backend false (no test launches on both cards) --
    exact for a marker expression; anything this cannot read answers True,
    since over-locking is a wait and under-locking an OOM."""
    if not markexpr.strip():
        return True
    try:
        from _pytest.mark.expression import Expression

        expr = Expression.compile(markexpr)
    except Exception:
        return True
    names = sorted(set(re.findall(r"[A-Za-z_]\w*", markexpr))
                   - {"and", "or", "not", "gpu", *PRESENT})
    if len(names) > 12:
        return True
    for bits in itertools.product((False, True), repeat=len(names)):
        on = {"gpu", backend, *(n for n, b in zip(names, bits, strict=True) if b)}
        if expr.evaluate(lambda name, /, **kw: name in on):  # noqa: B023
            return True
    return False


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Before `-m` reads the marks: `tiers.sample_compiles` and
    `tiers.settle_l_cut`, on test/'s own items only -- measure/ re-exports
    this hook, and its compile sweeps are not sampled. Then, with
    `FJ_REGISTRY_DUMP` set, the device cases resolved against the case
    registry (#648, test/test_registry.py)."""
    here = Path(__file__).parent
    ours = [i for i in items if i.path.is_relative_to(here)]
    tiers.sample_compiles(ours)
    tiers.settle_l_cut(ours)
    registry.dump_if_asked(ours)


_CONTROLLER_LOCKS = pytest.StashKey[list[int]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    sample.addoption(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Register the registry sampler (#653, test/registry/sample.py) in every
    process; then, under xdist, take each card's lock in the CONTROLLER
    (module docstring)."""
    if not config.pluginmanager.has_plugin("fj-sample"):
        config.pluginmanager.register(sample.Plugin(config), "fj-sample")
    if hasattr(config, "workerinput"):
        return
    dist = getattr(config.option, "dist", "no")
    if not dist or dist == "no":
        return
    markexpr = getattr(config.option, "markexpr", "") or ""
    fds = []
    for backend, present in PRESENT.items():
        if present() and _may_select(markexpr, backend):
            fd = take_card_lock(config, backend)
            if fd is not None:
                fds.append(fd)
    config.stash[_CONTROLLER_LOCKS] = fds


def release_card_lock(fd: int) -> None:
    """Blank the human-readable holder line, then close: the close is the
    release (an exit without this function releases just the same)."""
    try:
        os.ftruncate(fd, 0)
    finally:
        os.close(fd)


def pytest_unconfigure(config: pytest.Config) -> None:
    for fd in config.stash.get(_CONTROLLER_LOCKS, []):
        release_card_lock(fd)


def _card(request: pytest.FixtureRequest, backend: str, why: str):
    # One rule: no card you can launch on means SKIP (#712). `why` is the
    # absent message; _no_card_reason replaces it when the card is really here
    # and merely denied, because those two need different things done.
    if not PRESENT[backend]():
        pytest.skip(_no_card_reason(backend, why))
    fd = take_card_lock(request.config, backend)
    yield
    if fd is not None:
        release_card_lock(fd)


@pytest.fixture(scope="session")
def cuda_device(request: pytest.FixtureRequest):
    """Presence (skip), then the card's lock for the session (module docstring)."""
    yield from _card(request, "cuda", "no NVIDIA device on this machine")


def _no_amd_reason() -> str:
    """Why there is no AMD card -- the two causes read very differently.

    `/dev/kfd` missing is a passthrough question (a container started without
    `--device=/dev/kfd --device=/dev/dri`). `/dev/kfd` present with an empty
    topology is a HOST driver question, and saying so is the point: that is
    the case that sent agents hunting through devcontainer.json for a mount
    bug when amdgpu had simply bound no card (#431).
    """
    if not Path("/dev/kfd").exists():
        return ("no AMD device on this machine (/dev/kfd absent -- a container "
                "needs --device=/dev/kfd --device=/dev/dri)")
    return ("no AMD device on this machine: /dev/kfd is here but amdkfd lists "
            "no GPU node, so amdgpu bound no card on the HOST (not a container "
            "passthrough problem -- see `dkms status` and devtools/doctor.sh)")


@pytest.fixture(scope="session")
def hip_device(request: pytest.FixtureRequest):
    """Presence (skip), then the card's lock for the session (module docstring)."""
    yield from _card(request, "hip", _no_amd_reason())


def card_processes(backend: str) -> str:
    """What the vendor tool lists on the card -- which, from inside a
    container, misses every process outside it."""
    if backend == "cuda":
        cmd = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
               "--format=csv,noheader"]
    else:
        cmd = ["rocm-smi", "--showpids"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    rows = [ln.strip() for ln in out.splitlines() if ln.strip()
            and not ln.lstrip().startswith("=") and "KFD PIDs" not in ln
            and "PID" not in ln.split()[:1]]
    return "; ".join(rows) or ("none visible from here -- a process outside "
                               "this container is invisible to the vendor tool")


# The VRAM probe, spliced into a harness after its CHECK / dev* prelude. A
# harness calls `if (fj_vram_probe(bytes, {(const void*)k, ...})) return 0;`
# after reading its input and BEFORE allocating: with FJ_VRAM_PROBE set it
# prints one `fj-vram` line and launches nothing. The stack term is what the
# driver reserves for a launch -- the kernel's per-thread local (CUDA) /
# private (HIP) size times every thread the device can keep resident -- so an
# L = 16 kernel's requirement is its own, read from the launch, not a constant.
VRAM_PROBE = r"""
#include <initializer_list>
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define devMemGetInfo         hipMemGetInfo
#define devFuncAttributes     hipFuncAttributes
#define devFuncGetAttributes  hipFuncGetAttributes
#define devDeviceGetAttribute hipDeviceGetAttribute
#define devAttrSMs            hipDeviceAttributeMultiprocessorCount
#define devAttrThreadsPerSM   hipDeviceAttributeMaxThreadsPerMultiProcessor
#else
#define devMemGetInfo         cudaMemGetInfo
#define devFuncAttributes     cudaFuncAttributes
#define devFuncGetAttributes  cudaFuncGetAttributes
#define devDeviceGetAttribute cudaDeviceGetAttribute
#define devAttrSMs            cudaDevAttrMultiProcessorCount
#define devAttrThreadsPerSM   cudaDevAttrMaxThreadsPerMultiProcessor
#endif
static bool fj_vram_probe(size_t buffers, std::initializer_list<const void*> kernels) {
    if (!std::getenv("FJ_VRAM_PROBE")) return false;
    size_t free_b = 0, total_b = 0, stack = 0;
    int sms = 0, per_sm = 0;
    devError_t e = devMemGetInfo(&free_b, &total_b);
    if (e == devSuccess) e = devDeviceGetAttribute(&sms, devAttrSMs, 0);
    if (e == devSuccess) e = devDeviceGetAttribute(&per_sm, devAttrThreadsPerSM, 0);
    for (const void* k : kernels) {
        devFuncAttributes a;
        if (e == devSuccess) e = devFuncGetAttributes(&a, k);
        if (e == devSuccess && a.localSizeBytes > stack) stack = a.localSizeBytes;
    }
    if (e != devSuccess) {
        std::fprintf(stderr, "fj_vram_probe: %s\n", devGetErrorString(e));
        std::exit(1);
    }
    const size_t threads = (size_t)sms * (size_t)per_sm;
    std::printf("fj-vram free %zu total %zu need %zu buffers %zu stack %zu "
                "threads %zu\n", free_b, total_b, buffers + stack * threads,
                buffers, stack, threads);
    return true;
}
"""

_VRAM_LINE = re.compile(r"^fj-vram free (\d+) total (\d+) need (\d+) buffers "
                        r"(\d+) stack (\d+) threads (\d+)$", re.MULTILINE)
MIB = 1 << 20


def check_vram(binary: Path, args: tuple[str, ...], backend: str) -> dict[str, int]:
    """Run `binary` in probe mode; FAIL if its launch needs more than is free."""
    env = {**os.environ, "FJ_VRAM_PROBE": "1"}
    done = subprocess.run([str(binary), *args], capture_output=True, text=True,
                          timeout=300, env=env)
    m = _VRAM_LINE.search(done.stdout)
    assert done.returncode == 0 and m, (
        f"{binary.name}: VRAM probe exited {done.returncode} without an fj-vram "
        f"line\nstdout: {done.stdout}\nstderr: {done.stderr}")
    got = dict(zip(("free", "total", "need", "buffers", "stack", "threads"),
                   (int(x) for x in m.groups()), strict=True))
    if got["need"] > got["free"]:
        pytest.fail(
            f"{binary.name} on {backend}: not enough free VRAM to launch -- need "
            f"{got['need'] / MIB:.0f} MiB ({got['buffers'] / MIB:.1f} MiB "
            f"buffers + {got['stack']} B stack x {got['threads']} resident "
            f"threads), free {got['free'] / MIB:.0f} of "
            f"{got['total'] / MIB:.0f} MiB. Holding the card: "
            f"{card_processes(backend)}. The card lock is this run's "
            f"({lock_path(backend)}): the memory is held outside it.",
            pytrace=False)
    print(f"vram {binary.name} on {backend}: need {got['need'] / MIB:.0f} MiB "
          f"(stack {got['stack']} B x {got['threads']} threads), free "
          f"{got['free'] / MIB:.0f} of {got['total'] / MIB:.0f} MiB")
    return got


@pytest.fixture
def compile_device_source(tmp_path: Path):
    """Compile one translation unit of device source; return the binary's path.

    `link=False` stops at device code (nvcc -cubin / hipcc -c), which is the
    check that needs no card. Timeouts are generous on purpose: an unrolled
    high-angular-momentum kernel is slow to compile, and that is the point of
    the whole batching exercise rather than a sign something hung.

    Through the compiler cache (#186, test/devcache.py): every quoted
    `#include` -- relative to `tmp_path`, else to `INCLUDE`, which a `gpu`
    unit's header resolves against -- is carried into a content-addressed
    directory and the compile runs there through sccache, so an unchanged
    unit is a hit on a re-run; the binary is copied to `tmp_path/probe` (or
    `probe.o`).
    """

    def _compile(toolchain: Toolchain, source: str, *, link: bool = True) -> Path:
        text, units = devcache.localize(source, tmp_path, (INCLUDE,))
        out = tmp_path / ("probe" if link else "probe.o")
        mode = "link" if link else ("-cubin" if toolchain.name == "nvcc" else "-c")
        got = devcache.compile_unit(
            toolchain.path, toolchain.compile_flags(), f"probe.{toolchain.ext}",
            text, units, out, mode=mode, timeout=600)
        assert got.returncode == 0, (
            f"{toolchain.name} failed in {got.workdir}:\n"
            + "\n".join(" ".join(c) for c in got.commands) + f"\n{got.diagnostics}"
        )
        assert out.exists() and out.stat().st_size > 0, f"{out} is empty"
        return out

    return _compile


@pytest.fixture
def run_device_binary():
    """Run a compiled binary and return its stdout, failing loudly on nonzero.

    `vram=` names the backend of a harness that splices in VRAM_PROBE: it is
    run in probe mode first (`check_vram`), so a short card fails the test
    with its figures instead of launching into an out-of-memory error.
    """

    def _run(binary: Path, *args: str, vram: str | None = None) -> str:
        if vram is not None:
            check_vram(binary, args, vram)
        done = subprocess.run(
            [str(binary), *args], capture_output=True, text=True, timeout=300
        )
        assert done.returncode == 0, (
            f"{binary.name} exited {done.returncode}\n"
            f"stdout: {done.stdout}\nstderr: {done.stderr}"
        )
        return done.stdout

    return _run


# -- the claim registry ------------------------------------------------------

TEST_DIR = Path(__file__).resolve().parent


def _is_claim_mark(node: ast.AST) -> bool:
    """`pytest.mark.claim(...)` (or any `<x>.mark.claim(...)`)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "claim"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "mark"
    )


def scan_claim_marks(*test_dirs: Path) -> dict[str, str]:
    """Map each claim id to the node id of the one test marked with it.

    Every directory given is scanned, each node id spelled relative to its
    directory's parent (`test/...`, `measure/...`): a claim may be
    discharged by a correctness test or by a check of the measured rows
    (measure/), and one id marked in both trees is the same error as one id
    marked twice in one.

    Read from the SOURCE, not from the session's collected items, on purpose:
    a run narrowed by `-k`, `-m` or a path collects a subset, and a registry
    check that only saw that subset would pass or fail by accident. A mark
    this scanner cannot pin to one test -- a non-literal id, a module-level
    `pytestmark`, a mark on a class, the same id on two tests -- is an error,
    not something to skip, since the point is that nothing drifts silently.
    """
    found: dict[str, str] = {}
    paths = [(d.parent, p) for d in test_dirs for p in sorted(d.rglob("*.py"))]
    for root, path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = path.relative_to(root).as_posix()
        placed = list(_marks_on_tests(tree.body, f"{rel}::"))
        for where, deco in placed:
            args = deco.args
            if (
                len(args) != 1
                or deco.keywords
                or not isinstance(args[0], ast.Constant)
                or not isinstance(args[0].value, str)
            ):
                raise ValueError(f"{where}: claim mark takes one literal claim id")
            cid = args[0].value
            if cid in found:
                raise ValueError(
                    f"claim {cid!r} marked on both {found[cid]} and {where}"
                )
            found[cid] = where
        on_tests = {id(deco) for _, deco in placed}
        stray = [
            n.lineno
            for n in ast.walk(tree)
            if _is_claim_mark(n) and id(n) not in on_tests
        ]
        if stray:
            raise ValueError(
                f"{rel}: claim mark not on a test function (lines {stray}); "
                "a claim is discharged by exactly one named test"
            )
    return found


def _marks_on_tests(body: list[ast.stmt], prefix: str):
    """Yield (node id, claim-mark decorator) for functions, into classes."""
    for stmt in body:
        if isinstance(stmt, ast.ClassDef):
            yield from _marks_on_tests(stmt.body, f"{prefix}{stmt.name}::")
        elif isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            for deco in stmt.decorator_list:
                if _is_claim_mark(deco):
                    yield f"{prefix}{stmt.name}", deco


# measure/ (devtools/measure.sh): the claims a stored row discharges -- a
# verdict read off data/measured/ -- are marked there, beside the checks that
# recompute those rows; the registry check itself is correctness and stays.
MEASURE_DIR = REPO / "measure"


@pytest.fixture(scope="session")
def claim_marked_tests() -> dict[str, str]:
    """Every `@pytest.mark.claim(id)` under test/ and measure/, as
    {claim id: node id}."""
    return scan_claim_marks(TEST_DIR, MEASURE_DIR)
