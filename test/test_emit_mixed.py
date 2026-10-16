"""The `mixed` lever's kernels as text (#346): goldens and a compile, no card.

THE GOLDENS are the mixed arm of the class the literal goldens pin, (2,0|2,0)
at L = 4, one batch and `n_batches = 3`, per backend, in their own directory
(`test/golden/{cuda,hip}/mixed/`) so no literal golden's glob sees them; each
sits beside its literal control and the diff between them is the lever: F32
`float` declarations from the narrowed `PQ_d` and `theta` down to every
target, `(double)` widenings in the contraction, and nothing else. A golden
says only that the text has not moved; test/test_kernel_mixed.py is what
says the kernels are right. Regenerate (never hand-edit), then review:

    PYTHONPATH=src uv run python -c "from pathlib import Path; \
from felizjueves.kernel import build_pq_kernels, pq_spec; \
from felizjueves.device import SM_86, GFX1200; from felizjueves.emit import \
cuda, hip; [(Path(f'test/golden/cuda/mixed/{k.name}.cu').write_text(cuda.emit(\
k, SM_86)), Path(f'test/golden/hip/mixed/{k.name}.hip').write_text(hip.emit(\
k, GFX1200))) for n in (1, 3) for k in build_pq_kernels(pq_spec(4, \
('mixed',), n))]"

THE COMPILES (`cuda` / `hip`, no `gpu`) put every mixed unit L = 0..4, one
batch, and the L = 4 split cell -- each alone and composed with `fused` --
through the real device compiler under #92's warning filter: a device
compiler needs no card, so a green compile proves nothing about a number.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.j_levers.fused import FUSED
from felizjueves.j_levers.mixed import MIXED
from felizjueves.kernel import build_pq_kernel, build_pq_kernels, pq_spec

GOLDEN = Path(__file__).resolve().parent / "golden"
BACKENDS = {
    "cuda": (cuda.emit, SM_86, "cu"),
    "hip": (hip.emit, GFX1200, "hip"),
}
# The tier's warning flags (devtools/config.sh NVCC_FLAGS / HIPCC_FLAGS).
WARN_FLAGS = {"cuda": ["-Xcompiler", "-Wall"], "hip": ["-Wall", "-Wextra"]}
NAMES = ("pq_L4_p2_q2", "pq_L4_p2_q2_x0_4", "pq_L4_p2_q2_y0_4",
         "pq_L4_p2_q2_z0_4")


def _kernels():
    return (*build_pq_kernels(pq_spec(4, (MIXED,), 1)),
            *build_pq_kernels(pq_spec(4, (MIXED,), 3)))


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_the_mixed_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = BACKENDS[backend]
    ks = _kernels()
    assert tuple(k.name for k in ks) == NAMES
    for k in ks:
        assert emit(k, arch) == (
            GOLDEN / backend / MIXED / f"{k.name}.{ext}").read_text(), k.name
    found = sorted(p.stem for p in (GOLDEN / backend / MIXED).glob(f"*.{ext}"))
    assert found == sorted(NAMES)


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_each_mixed_golden_differs_from_its_literal_control(backend: str) -> None:
    emit, arch, ext = BACKENDS[backend]
    for name in NAMES:
        mixed = (GOLDEN / backend / MIXED / f"{name}.{ext}").read_text()
        literal = (GOLDEN / backend / f"{name}.{ext}").read_text()
        assert mixed != literal
        assert "lever mixed (#346)" in mixed and "float " in mixed
        assert "lever mixed" not in literal and "float " not in literal
        # the boundary in the text: the parameters are the literal ones
        def params(text: str) -> str:
            return text[text.index("__global__"):text.index("{")]

        assert params(mixed) == params(literal)
        assert "fp32 and fp64 as declared" in mixed
        assert "fp64 throughout" in literal


def _units():
    out = []
    for lv in ((MIXED,), (FUSED, MIXED)):
        out += [*(build_pq_kernel(pq_spec(L, lv)) for L in range(5)),
                *build_pq_kernels(pq_spec(4, lv, 3))]
    return out


def _compile_clean(tc, backend: str, tmp_path: Path) -> None:
    """Each unit alone, `-c` with the tier's flags: no error, and no warning
    located in the generated file (#92's rule)."""
    emit, arch, ext = BACKENDS[backend]
    warn = WARN_FLAGS[backend]
    for i, k in enumerate(_units()):
        src = tmp_path / f"u{i}_{k.name}.{ext}"
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
def test_the_mixed_units_compile_with_nvcc(nvcc, tmp_path) -> None:
    _compile_clean(nvcc, "cuda", tmp_path)


@pytest.mark.hip
def test_the_mixed_units_compile_with_hipcc(hipcc, tmp_path) -> None:
    _compile_clean(hipcc, "hip", tmp_path)
