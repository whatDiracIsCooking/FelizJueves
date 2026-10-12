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
- TIMING (the opt-in wall-clock sweep in test_kernel_device.py, and the
  ceiling probes' recording run in test_ceilings_device.py): the same lock,
  plus the utilisation wait of `felizjueves.gate` (#109; it lived in
  test_kernel_device.py and here until a second measurement needed it).

UNDER XDIST THE CONTROLLER HOLDS THE LOCK and the workers run under it. A
session fixture in each worker cannot hold it: the first worker's lock would
block its siblings until its own session ended, which is the end of the run.
So `pytest_configure` takes the lock in the controller, for each backend whose
card is present and whose `-m` expression can select a `gpu` test of it; a
worker's fixture then finds the lock held by an ANCESTOR (the controller, or a
`flock` wrapped around the whole run) and proceeds without taking it. A holder
that is not an ancestor is another run, and is waited for.

`claim_marked_tests` is the other half of `test/test_claims.py`'s registry
check: every `@pytest.mark.claim(id)` under test/, read from source so that a
narrowed run cannot hide one.
"""

from __future__ import annotations

import ast
import fcntl
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

from felizjueves.device import Vendor, from_env

# Architectures the toolchains are driven at, resolved through
# felizjueves.device -- the single Python source, which reads the same
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


def _has_nvidia_device() -> bool:
    # Probed through the kernel rather than nvidia-smi, so the answer is the
    # same on the host and inside a container whether or not the vendor tool
    # was installed.
    return Path("/dev/nvidiactl").exists() or Path("/proc/driver/nvidia/gpus").is_dir()


def _has_amd_device() -> bool:
    return Path("/dev/kfd").exists()


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
PRESENT = {"cuda": _has_nvidia_device, "hip": _has_amd_device}
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


_CONTROLLER_LOCKS = pytest.StashKey[list[int]]()


def pytest_configure(config: pytest.Config) -> None:
    """Under xdist, take each card's lock in the CONTROLLER (module docstring)."""
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
    if not PRESENT[backend]():
        pytest.skip(why)
    fd = take_card_lock(request.config, backend)
    yield
    if fd is not None:
        release_card_lock(fd)


@pytest.fixture(scope="session")
def cuda_device(request: pytest.FixtureRequest):
    """Presence (skip), then the card's lock for the session (module docstring)."""
    yield from _card(request, "cuda", "no NVIDIA device on this machine")


@pytest.fixture(scope="session")
def hip_device(request: pytest.FixtureRequest):
    """Presence (skip), then the card's lock for the session (module docstring)."""
    yield from _card(request, "hip", "no AMD device on this machine (/dev/kfd absent)")


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
    `#include` is carried into a content-addressed directory and the compile
    runs there through sccache, so an unchanged unit is a hit on a re-run;
    the binary is copied to `tmp_path/probe` (or `probe.o`).
    """

    def _compile(toolchain: Toolchain, source: str, *, link: bool = True) -> Path:
        text, units = devcache.localize(source, tmp_path)
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


def scan_claim_marks(test_dir: Path) -> dict[str, str]:
    """Map each claim id to the node id of the one test marked with it.

    Read from the SOURCE, not from the session's collected items, on purpose:
    a run narrowed by `-k`, `-m` or a path collects a subset, and a registry
    check that only saw that subset would pass or fail by accident. A mark
    this scanner cannot pin to one test -- a non-literal id, a module-level
    `pytestmark`, a mark on a class, the same id on two tests -- is an error,
    not something to skip, since the point is that nothing drifts silently.
    """
    found: dict[str, str] = {}
    root = test_dir.parent
    for path in sorted(test_dir.rglob("*.py")):
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


@pytest.fixture(scope="session")
def claim_marked_tests() -> dict[str, str]:
    """Every `@pytest.mark.claim(id)` under test/, as {claim id: node id}."""
    return scan_claim_marks(TEST_DIR)
