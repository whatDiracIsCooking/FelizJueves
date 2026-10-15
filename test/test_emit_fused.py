"""The `fused` lever's kernels as text (#344): goldens and a compile, no card.

THE GOLDENS are the fused arm of (1,0|1,0), L = 2 -- the milestone's per-PR
kernel cap -- one batch and `n_batches = 3`, per backend, in their own
directory (`test/golden/{cuda,hip}/fused/`) so no literal golden's glob sees
them. There is no literal golden at L = 2, so the control each is compared
against is the literal kernel emitted here. A golden says only that the text
has not moved; test/test_kernel_fused.py is what says the kernels are right.
Regenerate (never hand-edit), then review:

    PYTHONPATH=src uv run python -c "from pathlib import Path; \
from felizjueves.kernel import build_pq_kernels, pq_spec; \
from felizjueves.device import SM_86, GFX1200; from felizjueves.emit import \
cuda, hip; [(Path(f'test/golden/cuda/fused/{k.name}.cu').write_text(cuda.emit(\
k, SM_86)), Path(f'test/golden/hip/fused/{k.name}.hip').write_text(hip.emit(\
k, GFX1200))) for n in (1, 3) for k in build_pq_kernels(pq_spec(2, \
('fused',), n))]"

THE COMPILES (`cuda` / `hip`, no `gpu`) put every fused unit L = 0..2, one
batch, and the L = 2 split cell through the real device compiler with the
tier's warning flags, failing on a warning located in the generated file
(#92's rule): a device compiler needs no card, so a green compile proves
nothing about a number.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.j_levers.fused import FUSED
from felizjueves.kernel import build_pq_kernel, build_pq_kernels, pq_spec

GOLDEN = Path(__file__).resolve().parent / "golden"
BACKENDS = {
    "cuda": (cuda.emit, SM_86, "cu"),
    "hip": (hip.emit, GFX1200, "hip"),
}
# The tier's warning flags (devtools/config.sh NVCC_FLAGS / HIPCC_FLAGS).
WARN_FLAGS = {"cuda": ["-Xcompiler", "-Wall"], "hip": ["-Wall", "-Wextra"]}
NAMES = ("pq_L2_p1_q1", "pq_L2_p1_q1_x0_2", "pq_L2_p1_q1_y0_2",
         "pq_L2_p1_q1_z0_2")


def _kernels(levers: tuple[str, ...] = (FUSED,)):
    return (*build_pq_kernels(pq_spec(2, levers, 1)),
            *build_pq_kernels(pq_spec(2, levers, 3)))


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_the_fused_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = BACKENDS[backend]
    ks = _kernels()
    assert tuple(k.name for k in ks) == NAMES
    for k in ks:
        assert emit(k, arch) == (
            GOLDEN / backend / FUSED / f"{k.name}.{ext}").read_text(), k.name
    found = sorted(p.stem for p in (GOLDEN / backend / FUSED).glob(f"*.{ext}"))
    assert found == sorted(NAMES)


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_each_fused_golden_differs_from_its_literal_control(backend: str) -> None:
    emit, arch, ext = BACKENDS[backend]
    for k in _kernels(()):
        fused = (GOLDEN / backend / FUSED / f"{k.name}.{ext}").read_text()
        literal = emit(k, arch)
        assert fused != literal
        assert "lever fused (#344)" in fused
        assert "lever fused" not in literal


def _units():
    return [*(build_pq_kernel(pq_spec(L, (FUSED,))) for L in range(3)),
            *build_pq_kernels(pq_spec(2, (FUSED,), 3))]


def _compile_clean(tc, backend: str, tmp_path: Path) -> None:
    """Each unit alone, `-c` with the tier's flags: no error, and no warning
    located in the generated file (#92's rule; a vendor header's is not
    ours)."""
    emit, arch, ext = BACKENDS[backend]
    warn = WARN_FLAGS[backend]
    for k in _units():
        src = tmp_path / f"{k.name}.{ext}"
        src.write_text(emit(k, arch))
        got = subprocess.run(
            [tc.path, *tc.compile_flags(), *warn, "-c", str(src), "-o",
             str(src.with_suffix(".o"))],
            capture_output=True, text=True, timeout=600)
        diag = got.stdout + got.stderr
        assert got.returncode == 0, f"{k.name}:\n{diag}"
        ours = [ln for ln in diag.splitlines()
                if src.name in ln and "warning" in ln]
        assert not ours, f"{k.name}: warnings in generated text:\n" + \
            "\n".join(ours)


@pytest.mark.cuda
def test_the_fused_units_compile_with_nvcc(nvcc, tmp_path) -> None:
    _compile_clean(nvcc, "cuda", tmp_path)


@pytest.mark.hip
def test_the_fused_units_compile_with_hipcc(hipcc, tmp_path) -> None:
    _compile_clean(hipcc, "hip", tmp_path)
