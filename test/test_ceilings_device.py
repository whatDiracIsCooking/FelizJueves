"""Each card's roofline ceilings, probed on the card behind the card gate (#109).

Two tests per backend, both `gpu` plus `cuda`/`hip`, so they skip without the
card and the compiler -- the recorded ceilings themselves are data
(`data/measured/ceilings.json`), and test/test_ceilings.py checks them, and the
gate's refusal, in the fast tier.

- `test_*_probe_is_bounded_by_the_published_peak` runs in the kernel tier:
  it compiles `ceilings.PROBE_SOURCE` with devtools/config.sh's flag set, runs
  both probes (the triad checks every element on the device) and asserts each
  configuration's best rate is at or under the vendor's published peak.
  Contention can only LOWER a rate, so this check is sound on a busy card; it
  records nothing.

- `test_*_record_ceilings` is the RECORDING run, opt-in exactly as the
  wall-clock sweep is (test_kernel_device.py's section comment): it skips
  unless FJ_CEILINGS_STORE names the store, and it must run alone, uncontended.
  Each probe phase goes through `gate.gated` -- the card's settled reading
  before, the reading after -- and a phase the gate does not let through is
  REFUSED: that quantity is not recorded, and the arch stays uncharacterised
  (device.py: no ridge, no verdict). The command, per backend:

    env FJ_CEILINGS_STORE=data/measured/ceilings.json uv run pytest \\
        test/test_ceilings_device.py -p no:xdist -s -rs -m 'gpu and cuda' \\
        -k record

  (`ceilings.COMMAND`, which each record carries.) The card's lock is the
  device fixture's (test/conftest.py, #91).
"""

from __future__ import annotations

import datetime
import hashlib
import os
import shlex
import subprocess
from pathlib import Path

import pytest
from conftest import VRAM_PROBE

from felizjueves import ceilings, measured
from felizjueves.device import GFX1200, SM_86

STORE = "FJ_CEILINGS_STORE"
ARCH = {"cuda": SM_86, "hip": GFX1200}


def _toolchain(backend: str, request: pytest.FixtureRequest):
    tc = request.getfixturevalue("nvcc" if backend == "cuda" else "hipcc")
    request.getfixturevalue("cuda_device" if backend == "cuda" else "hip_device")
    return tc


def _build(backend: str, request, tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Compile the probe with config.sh's flags; return it and its provenance."""
    tc = _toolchain(backend, request)
    b = measured.BACKENDS[backend]
    flags = shlex.split(measured.config_value(b.flags_var))
    flags += ([f"-arch=sm_{tc.arch}"] if tc.name == "nvcc"
              else [f"--offload-arch={tc.arch}"])
    source = ceilings.probe_source(VRAM_PROBE)
    src = tmp_path / f"fj_ceilings.{tc.ext}"
    src.write_text(source)
    binary = tmp_path / "fj_ceilings"
    cmd = [tc.path, *flags, str(src), "-o", str(binary)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"{' '.join(cmd)}\n{done.stderr}"
    provenance = {
        "probe": "ceilings.PROBE_SOURCE (fj_fma<C>, fj_triad)",
        "compile": " ".join([Path(tc.path).name, *flags,
                             f"<fj_ceilings.{tc.ext}>"]),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "toolchain": b.toolchain(tc.path),
        "gpu": b.gpu(),
        "date": datetime.date.today().isoformat(),
    }
    return binary, provenance


def _runner(binary: Path, mode: str, backend: str, run_device_binary):
    def run() -> str:
        return run_device_binary(binary, mode, vram=backend)
    return run


def _bounded(backend, request, run_device_binary, tmp_path) -> None:
    arch = ARCH[backend]
    binary, _ = _build(backend, request, tmp_path)
    published = ceilings.load()["published"][arch.name]
    for quantity, mode in ceilings.PROBE_OF.items():
        runs = ceilings.parse_probe(_runner(binary, mode, backend,
                                            run_device_binary)())
        for r in runs:
            best = max(r.rates())
            ratio = ceilings.check_published(quantity, best, published)
            print(f"\n{arch.name} {quantity}: {r.header} -> best {best:.4g} "
                  f"{ceilings.UNIT_OF[quantity]}, {ratio:.1%} of published")


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_probe_is_bounded_by_the_published_peak(request, run_device_binary,
                                                     tmp_path) -> None:
    _bounded("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_probe_is_bounded_by_the_published_peak(request, run_device_binary,
                                                    tmp_path) -> None:
    _bounded("hip", request, run_device_binary, tmp_path)


def _record(backend, request, run_device_binary, tmp_path) -> None:
    store_path = os.environ.get(STORE)
    if not store_path:
        pytest.skip(f"recording ceilings is opt-in and must run alone, "
                    f"uncontended: set {STORE} (see the module docstring)")
    arch = ARCH[backend]
    binary, provenance = _build(backend, request, tmp_path)
    store = ceilings.load(Path(store_path))
    lines = []
    for quantity, mode in ceilings.PROBE_OF.items():
        store, ceiling, why = ceilings.record(
            store, arch, quantity, _runner(binary, mode, backend,
                                           run_device_binary),
            backend=backend, provenance=provenance)
        print(why, flush=True)
        lines.append(why)
        if ceiling is not None:
            print(f"  gate: card {ceiling.raw['gpu_util_before']}% -> "
                  f"{ceiling.raw['gpu_util_after']}%, CPU busy "
                  f"{ceiling.raw['cpu_busy_permille_before'] / 10:.1f}%; "
                  f"best {ceiling.raw['best']:.4g}, IQR "
                  f"[{ceiling.raw['q1']:.4g}, {ceiling.raw['q3']:.4g}]",
                  flush=True)
    ceilings.save(store, Path(store_path))
    refused = [ln for ln in lines if ln.startswith("REFUSED")]
    if refused:
        # The designed outcome on a contended card: nothing recorded for it.
        pytest.skip("; ".join(refused))


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_record_ceilings(request, run_device_binary, tmp_path) -> None:
    """The RTX 3080's fp64 and bandwidth ceilings, into FJ_CEILINGS_STORE."""
    _record("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_record_ceilings(request, run_device_binary, tmp_path) -> None:
    """The RX 9060 XT's fp64 and bandwidth ceilings, into FJ_CEILINGS_STORE."""
    _record("hip", request, run_device_binary, tmp_path)
