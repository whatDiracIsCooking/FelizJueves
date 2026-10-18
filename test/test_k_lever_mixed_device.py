"""Mixed-precision K (#347) through the device compilers and on the cards.

TWO PARTS, the issue's acceptance on the device side:

1. COMPILE-ONLY (`cuda` / `hip`, no card): a set of K classes with L <= 4
   (milestone 13's cap), each built three ways -- the literal kernel, the
   compute-only arm `kmixed` and the fully narrowed arm `kmixed-de` -- each
   batch kernel its own translation unit, compiled with the tier's flags
   plus the measurement flag (`measured.Backend.command`: `-Xptxas -v` on
   nvcc, the AMD `kernel-resource-usage` remark on hipcc). A warning in
   generated text fails (#92's rule). The figures -- registers and spill
   bytes (nvcc), VGPRs and spilled registers (hipcc), worst batch kernel per
   arm -- are PRINTED (run with `-s`) as observations: compiler readings,
   never store rows, no threshold. The clock is X2's (#362), and the
   issue's expectation -- compute-only REGRESSES, full narrowing WINS (the
   predecessor's +71% / -73% at `(dp|dp)`) -- is a PREDICTION to be tested
   there, not asserted here.
2. ON THE CARD (`gpu` + `cuda` / `hip`): `k_spec(L)`'s class for L = 0..4,
   at one batch and, where the plan admits it, three, each arm: the cell's
   launch sequence through test_kernel_device.py's K harness (`_k_build`,
   its `K_CASES` inputs and host staging `_k_case`: `ao.k._k_quartet` on
   `reference.pq` in fp64, with its |.| twin `S`) held to THE fp32 tier,
   `|K - X| <= F32_REL * S` elementwise (test/oracle/tolerance.py, the one
   declared fp32 tolerance; not the fp64 `K_REL`, and nothing looser), and
   shown to MISS the fp64 tier (the card really ran fp32). `kmixed` runs on
   the harness as it is (its signature is the literal one); `kmixed-de`'s
   five `const float*` parameters need the harness's signature, uploads and
   launch argument list changed, which `_full_harness` does textually (each
   replacement asserted to land), and its inputs are packed by the host as
   `ao.pairs.narrowed` copies, uploaded as float (the C conversion of an
   exactly-float32 double is exact). The reference is computed from the
   UNNARROWED inputs, so the packing error is inside the budget. Nothing is
   recorded.

Run part 2 as (each backend; the generator capped at L = 4):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- -k k_lever_mixed
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from oracle.tolerance import F32_REL, F64_REL

from felizjueves import measured
from felizjueves.ao.pairs import narrowed
from felizjueves.ir import is_f32
from felizjueves.k_levers import mixed
from felizjueves.kernel import k_spec
from felizjueves.kernel_k import build_k_kernels
from felizjueves.spec import Operator, Spec, Target

ARMS = mixed.ARMS


def _id(am) -> str:
    return "({}{}|{}{})".format(*("spdfg"[x] for x in am))


# ---------------------------------------------------------------------------
# 1. Compile-only: literal vs kmixed vs kmixed-de
# ---------------------------------------------------------------------------

COMPILE_CELLS = ([(k_spec(L).am, 1) for L in range(1, 5)]
                 + [((2, 0, 2, 0), 1), ((2, 1, 1, 0), 1),
                    ((1, 0, 1, 0), 3), ((1, 1, 1, 1), 3)])
COMPILE_PARAMS = [pytest.param(am, n, id=f"{_id(am)}-b{n}")
                  for am, n in COMPILE_CELLS]


def _spec(am, levers: tuple[str, ...] = (), n_batches: int = 1) -> Spec:
    return Spec(Operator.COULOMB, 0, tuple(am), Target.K, levers, n_batches)


def _figures(backend: str, compiler: str, am, n: int, tmp_path: Path) -> None:
    import test_k_lever_split_device as sd

    arms = {"literal": build_k_kernels(_spec(am, (), n))}
    for arm in ARMS:
        tag = arm.replace("-", "_")
        arms[arm] = tuple(dataclasses.replace(k, name=f"{k.name}_lv_{tag}")
                          for k in build_k_kernels(_spec(am, (arm,), n)))
    kernels = [k for ks in arms.values() for k in ks]
    with ThreadPoolExecutor(max_workers=6) as pool:
        got = dict(zip((k.name for k in kernels), pool.map(
            lambda k: sd._compile(backend, compiler, k, tmp_path), kernels),
            strict=True))
    unit = ("registers/spill-store bytes" if backend == "cuda"
            else "VGPRs/spilled registers")
    line = "; ".join(
        f"{name} worst {sd._worst([k.name for k in ks], [got[k.name] for k in ks])}"
        for name, ks in arms.items())
    print(f"\n{backend} {_id(am)} b{n} ({unit}): {line}", flush=True)
    assert len(got) == len(kernels)


@pytest.mark.cuda
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_cuda_k_mixed_kernels_compile_with_their_figures(am, n, nvcc,
                                                         tmp_path) -> None:
    _figures("cuda", nvcc.path, am, n, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_hip_k_mixed_kernels_compile_with_their_figures(am, n, hipcc,
                                                        tmp_path) -> None:
    _figures("hip", hipcc.path, am, n, tmp_path)


# ---------------------------------------------------------------------------
# 2. On the card: each arm against the host staging, at the fp32 tier
# ---------------------------------------------------------------------------


def _device_cells() -> list:
    out = []
    for arm in ARMS:
        for L in range(5):
            for b in (1, 3):
                spec = dataclasses.replace(k_spec(L, n_batches=b),
                                           levers=(arm,))
                try:
                    build_k_kernels(spec)
                except ValueError:
                    continue  # L = 0 at 3 batches: an empty batch, refused
                out.append(pytest.param(spec, id=f"L{L}-b{b}-k-{arm}"))
    return out


DEVICE_CELLS = _device_cells()

# kmixed-de's operands: the host input each narrows (`_k_inputs`' key, one
# `[n, 3]` vector or `Dcd`) and its harness upload index (K_HARNESS's
# `sizes[]` order). A vector's three per-axis parameters (#421) share it.
_FLOAT_SLOTS = {"XPA": 2, "XPC": 3, "XQB": 9, "XQD": 10, "Dcd": 14}

_UPLOAD_FLOATS = r"""
// kmixed-de (#347): an operand the host packed as float32 (ao.pairs.narrowed)
// arrives here as exactly-float32 doubles, so this conversion is exact.
static float* upload_floats(const double* h, size_t n) {{
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (float)h[i];
    float* d = nullptr;
    if (devMalloc(&d, n * sizeof(float) + 8) != devSuccess) return nullptr;
    if (n && devMemcpy(d, v.data(), n * sizeof(float), devMemcpyH2D) != devSuccess)
        return nullptr;
    return d;
}}

// usage: `limits`"""


def _swap(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"the harness changed: {old!r}"
    return text.replace(old, new)


def _full_harness(tkd) -> dict[str, str]:
    """K_HARNESS with `kmixed-de`'s float uploads. The declaration and the
    launch need no rewrite: both are derived from the kernel, a parameter
    it declares F32 read from `df[...]` (`tkd.k_launch_args`)."""
    tkd_regions = {r: n for r, n in enumerate(tkd.K_REGIONS)}
    assert {n for i in _FLOAT_SLOTS.values() for n in tkd_regions[i]} == set(
        mixed.NARROWED_PARAMS)
    floats = " || ".join(f"i == {i}" for i in sorted(_FLOAT_SLOTS.values()))
    harness = _swap(tkd.K_HARNESS, "\n// usage: `limits`", _UPLOAD_FLOATS)
    harness = _swap(harness, "    int* di[15] = {{nullptr}};\n",
                    "    int* di[15] = {{nullptr}};\n"
                    "    float* df[15] = {{nullptr}};\n")
    harness = _swap(
        harness, "        if (i == 6 || i == 13) {{",
        f"        if ({floats}) {{{{\n"
        "            df[i] = upload_floats(h, sizes[i]);\n"
        "            if (!df[i]) {{ std::fprintf(stderr, \"upload %d failed\\n\", i);"
        " return 1; }}\n"
        "            d[i] = nullptr;\n"
        "        }} else if (i == 6 || i == 13) {{")
    return {"K_HARNESS": harness}


def _packed(k: dict) -> dict:
    """`k` with the five operands replaced by the host's narrowed copies."""
    out = dict(k)
    for name in _FLOAT_SLOTS:
        name_in = name
        a = np.asarray(k[name_in], dtype=np.float64)
        out[name_in] = np.array(narrowed(a.ravel())).reshape(a.shape)
        assert all(is_f32(float(x)) for x in out[name_in].ravel())
    return out


def _run(backend: str, spec: Spec, request, compile_device_source,
         run_device_binary, tmp_path: Path, monkeypatch) -> None:
    import test_kernel_device as tkd

    arch = tkd.BACKEND[backend][1]
    base = tkd._toolchain(backend, request)
    tc = tkd._Configured(base.name, base.path, base.arch, base.ext)
    full = mixed.FULL in spec.levers
    if full:
        for attr, text in _full_harness(tkd).items():
            monkeypatch.setattr(tkd, attr, text)
    cell = tkd._k_build(backend, spec, tc, tmp_path / "build")
    smem, optin = tkd._k_limits(cell.binary)
    need = tkd.k_shared_need(cell, arch.wavefront)
    assert need <= smem, f"{spec.key()} asks {need} B of shared, card {smem}"
    out = []
    for case in tkd.K_CASES:
        k, want, scale, t_max = tkd._k_case(spec.total_am, case)
        src, dst = tmp_path / f"k-{case}.in", tmp_path / f"k-{case}.out"
        tkd._k_pack(_packed(k) if full else k, arch.wavefront).tofile(src)
        run_device_binary(cell.binary, str(src), str(dst), "0", "1",
                          vram=backend)
        got = np.fromfile(dst, dtype=np.float64).reshape(2, k["n_bra"], -1)[0]
        assert got.shape == want.shape
        assert np.all(np.isfinite(got)), f"{case}: non-finite K"
        rel = np.abs(got - want) / np.where(scale > 0, scale, 1.0)
        err = float(np.max(rel))
        assert np.all(np.abs(got - want) <= F32_REL * scale), (
            f"{spec.key()} {case} on {arch.name}: max |K - X| / S = "
            f"{err:.3e} > F32_REL {F32_REL:.3e} (T up to {t_max:.1f})")
        assert err > F64_REL, (
            f"{spec.key()} {case}: {err:.3e} meets the fp64 tier -- the "
            "card did not run fp32")
        out.append((case, err, t_max))
    print(f"\n{backend} {arch.name} {spec.key()}: " + ", ".join(
        f"{c} err/S {e:.2e} (T<={t:.0f})" for c, e, t in out), flush=True)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_cuda_k_lever_mixed_matches_the_host(spec, request,
                                             compile_device_source,
                                             run_device_binary, tmp_path,
                                             monkeypatch) -> None:
    """The arm's K cell on the RTX 3080, at the fp32 tier."""
    _run("cuda", spec, request, compile_device_source, run_device_binary,
         tmp_path, monkeypatch)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_hip_k_lever_mixed_matches_the_host(spec, request,
                                            compile_device_source,
                                            run_device_binary, tmp_path,
                                            monkeypatch) -> None:
    """The arm's K cell on the RX 9060 XT, at the fp32 tier."""
    _run("hip", spec, request, compile_device_source, run_device_binary,
         tmp_path, monkeypatch)


# ---------------------------------------------------------------------------
# Fast: the cells, the driver's naming and the harness rewrite
# ---------------------------------------------------------------------------


def test_the_device_cells_are_l_at_most_4() -> None:
    per_arm = [0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert [s.values[0].total_am for s in DEVICE_CELLS] == per_arm * 2
    assert all(sum(am) <= 4 for am, _ in COMPILE_CELLS)


def test_the_driver_builds_an_arm_as_build_k_kernels() -> None:
    # #339: `measured.cell_kernels` passes a kernel-body K lever to the
    # builder, so the harness compiles the arm's kernels as they are, only
    # renamed `<name>_lv_<arm>` (non-identifier characters as `_`).
    for arm in ARMS:
        for spec in (k_spec(2), k_spec(4, n_batches=3)):
            spec = dataclasses.replace(spec, levers=(arm,))
            want = build_k_kernels(spec)
            got = measured.cell_kernels(spec)
            tag = arm.replace("-", "_")
            assert [k.name for k in got] == [f"{k.name}_lv_{tag}"
                                             for k in want]
            assert [dataclasses.replace(k, name=w.name)
                    for k, w in zip(got, want, strict=True)] == list(want)


def test_the_full_harness_matches_the_kernel_signature() -> None:
    """The launch passes every F32 parameter of `kmixed-de` from a float
    upload (`df[...]`), every other array from a double or int one, and the
    declaration is the unit's own signature, `float` exactly where the
    kernel declares F32 (#421: derived, never restated)."""
    import test_kernel_device as tkd

    from felizjueves.device import SM_86
    from felizjueves.emit import cuda
    from felizjueves.ir import Scalar

    (kern,) = build_k_kernels(dataclasses.replace(k_spec(2),
                                                  levers=(mixed.FULL,)))
    args = tkd.k_launch_args(kern)
    f32 = [p.name for p in kern.params if p.array and p.type is Scalar.F32]
    assert sorted(f32) == sorted(mixed.NARROWED_PARAMS)
    loose = [a for a in args if "PrimPairSoA" not in a]
    assert sum(a.startswith("df[") for a in loose) == len(f32)
    assert all("df[" not in a for a in args if "PrimPairSoA" in a)
    decl = tkd.k_declaration(cuda.emit(kern, SM_86), kern.name)
    assert decl.count("const float*") == len(f32)
    assert decl.count("PrimPairSoA ") == 2
