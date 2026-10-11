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

`claim_marked_tests` is the other half of `test/test_claims.py`'s registry
check: every `@pytest.mark.claim(id)` under test/, read from source so that a
narrowed run cannot hide one.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

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


@pytest.fixture(scope="session")
def cuda_device() -> None:
    if not _has_nvidia_device():
        pytest.skip("no NVIDIA device on this machine")


@pytest.fixture(scope="session")
def hip_device() -> None:
    if not _has_amd_device():
        pytest.skip("no AMD device on this machine (/dev/kfd absent)")


@pytest.fixture
def compile_device_source(tmp_path: Path):
    """Compile one translation unit of device source; return the binary's path.

    `link=False` stops at device code (nvcc -cubin / hipcc -c), which is the
    check that needs no card. Timeouts are generous on purpose: an unrolled
    high-angular-momentum kernel is slow to compile, and that is the point of
    the whole batching exercise rather than a sign something hung.
    """

    def _compile(toolchain: Toolchain, source: str, *, link: bool = True) -> Path:
        src = tmp_path / f"probe.{toolchain.ext}"
        src.write_text(source)
        out = tmp_path / ("probe" if link else "probe.o")
        cmd = [toolchain.path, *toolchain.compile_flags()]
        if not link:
            cmd += ["-cubin" if toolchain.name == "nvcc" else "-c"]
        cmd += [str(src), "-o", str(out)]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        assert done.returncode == 0, (
            f"{toolchain.name} failed:\n{' '.join(cmd)}\n{done.stderr}"
        )
        assert out.exists() and out.stat().st_size > 0, f"{out} is empty"
        return out

    return _compile


@pytest.fixture
def run_device_binary():
    """Run a compiled binary and return its stdout, failing loudly on nonzero."""

    def _run(binary: Path) -> str:
        done = subprocess.run(
            [str(binary)], capture_output=True, text=True, timeout=300
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
