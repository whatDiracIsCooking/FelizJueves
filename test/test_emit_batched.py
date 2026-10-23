"""The batched kernels as text: `build_pq_kernels` through both backends (#51).

THE GOLDEN FILES are one class at one batch count, on purpose: (2,0|2,0),
L = 4 -- the class `test/golden/{cuda,hip}/pq_L4_p2_q2.*` already pin in one
batch -- at `n_batches = 3`, three kernels per backend, named by #49's
convention (`pq_L4_p2_q2_{dir}{a}_{b}`), so each sits beside its one-batch
control and the diff between them is readable by eye. Golden count grows as
classes x batch counts; add another only because something needs it.

The text is emitted exactly as `test_emit_cuda.py` / `test_emit_hip.py` emit
theirs -- no formatter, the writer's own layout -- so the two generations are
comparable line for line. After an intended change, regenerate (never
hand-edit) and review the diff:

    PYTHONPATH=src uv run python -c "from pathlib import Path; \
from felizjueves.mmd.kernel import build_pq_kernels, pq_spec; \
from felizjueves.device import SM_86, GFX1200; from felizjueves.emit import \
cuda, hip; [(Path(f'test/golden/cuda/{k.name}.cu').write_text(cuda.emit(k, \
SM_86)), Path(f'test/golden/hip/{k.name}.hip').write_text(hip.emit(k, \
GFX1200))) for k in build_pq_kernels(pq_spec(4, n_batches=3))]"

A golden says only that the text has not moved; #49's interpreted pin
(test_kernel_batched.py) is what says the kernels are right. Nothing here
compiles or launches anything.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import pytest

from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.ir import Kernel, Raw, walk
from felizjueves.mmd.kernel import build_pq_kernel, build_pq_kernels, pq_spec

L, N_BATCHES = 4, 3
GOLDEN = Path(__file__).resolve().parent / "golden"
NAMES = ("pq_L4_p2_q2_x0_4", "pq_L4_p2_q2_y0_4", "pq_L4_p2_q2_z0_4")
BACKENDS = {
    "cuda": (cuda.emit, SM_86, "cu"),
    "hip": (hip.emit, GFX1200, "hip"),
}

PLAIN_J_WRITE = re.compile(r"^\s*J\[[^\n]*\] = ", re.MULTILINE)
ATOMIC_J_WRITE = re.compile(r"^\s*atomicAdd\(&J\[", re.MULTILINE)


@cache
def _kernels() -> tuple[Kernel, ...]:
    return build_pq_kernels(pq_spec(L, n_batches=N_BATCHES))


def _emit(backend: str, k: Kernel) -> str:
    emit, arch, _ = BACKENDS[backend]
    return emit(k, arch)


def _golden(backend: str, name: str) -> Path:
    return GOLDEN / backend / f"{name}.{BACKENDS[backend][2]}"


def test_three_kernels_named_by_the_batch_convention() -> None:
    assert tuple(k.name for k in _kernels()) == NAMES


@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize("i", range(N_BATCHES))
def test_the_batched_golden_is_unchanged(backend: str, i: int) -> None:
    k = _kernels()[i]
    assert _emit(backend, k) == _golden(backend, k.name).read_text()


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_exactly_these_goldens_exist_for_the_class(backend: str) -> None:
    # No stray batched golden left behind by a renamed batch.
    ext = BACKENDS[backend][2]
    found = sorted(p.stem for p in (GOLDEN / backend).glob(f"pq_L4_p2_q2*.{ext}"))
    assert found == ["pq_L4_p2_q2", *NAMES]


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_batched_text_is_atomic_and_one_batch_text_is_plain(backend: str) -> None:
    """The pair is the point: a split class accumulates, one batch assigns."""
    for k in _kernels():
        text = _emit(backend, k)
        assert ATOMIC_J_WRITE.search(text), k.name
        assert not PLAIN_J_WRITE.search(text), k.name
    control = _golden(backend, "pq_L4_p2_q2").read_text()
    assert PLAIN_J_WRITE.search(control)
    assert "atomicAdd" not in control


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_one_batch_through_build_pq_kernels_is_the_existing_golden(
    backend: str,
) -> None:
    (k,) = build_pq_kernels(pq_spec(L, n_batches=1))
    want = _golden(backend, "pq_L4_p2_q2").read_text()
    assert _emit(backend, k) == want == _emit(backend, build_pq_kernel(L))


@pytest.mark.parametrize("i", range(N_BATCHES))
def test_the_two_texts_differ_only_in_the_vendor_hooks(i: int) -> None:
    """test_emit_hip.py's seam assertion, on a batched kernel: the atomic
    write-back adds no vendor difference (both spell `atomicAdd`)."""
    k = _kernels()[i]
    c = cuda.emit(k, SM_86).splitlines()
    h = hip.emit(k, GFX1200).splitlines()
    assert h[0] == c[0].replace("emit.cuda", "emit.hip")
    assert h[1] == c[1].replace(SM_86.name, GFX1200.name)
    assert h[2] == "#include <hip/hip_runtime.h>"
    body_c, body_h = c[2:], h[3:]
    assert len(body_c) == len(body_h)
    changed = {(a.strip(), b.strip())
               for a, b in zip(body_c, body_h, strict=True) if a != b}
    assert changed == {
        ("fj_v += __shfl_xor_sync(0xffffffffu, fj_v, fj_o, 32);",
         "fj_v += __shfl_xor_sync(0xffffffffull, fj_v, fj_o, 32);"),
    }
    atomics_c = [x for x in body_c if "atomicAdd" in x]
    atomics_h = [x for x in body_h if "atomicAdd" in x]
    assert atomics_c and atomics_c == atomics_h


def test_no_raw_in_the_batched_kernels() -> None:
    for k in _kernels():
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name
