"""The shared C++ harness and its generic runner (#649).

THE KERNEL is `axpy_parts`, an axpy taken apart so that every output is
EXACT in IEEE arithmetic and no multiply feeds an add (no FMA can form on
either compiler, contraction left at each one's default): `sum = x + y`,
`scaled = a * x`, `acc += y` (an OUT array the kernel reads -- uploaded,
updated, read back), `twice = u + u` in float32, and `idx = i` in int32. So
the card must equal numpy BIT FOR BIT, and the runner's output files must
equal `np.save` of numpy's arrays BYTE FOR BYTE (test/harness/npy.h writes
numpy's own header). It is launched with a ragged tail (`N` not a multiple of
the block), so the guard is exercised too.

TIERS. The fast tier (no compiler) holds the Python half: the spec built
from the IR parameter list and its refusals, the runner's source, the
harness headers' own rules, and the kernel interpreted against numpy. The
`cuda` / `hip` cases build the runner under nvcc and hipcc with the tier's
flags, failing on any warning in the unit or a harness header (#92), and run
the binary on specs that contradict the kernel's signature -- refused before
the first runtime call, so no card is needed. The `gpu` cases run it on each
card:

    devtools/kernel-tier.sh --backend cuda -- -k harness
    devtools/kernel-tier.sh --backend hip -- -k harness
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from harness_driver import (
    HARNESS,
    HEADERS,
    build_runner,
    launch_spec,
    run_case,
    runner_program,
    write_case,
)

from felizjueves.emit import cuda, hip
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Fn,
    If,
    Index,
    Intent,
    Kernel,
    Param,
    Scalar,
    Var,
)
from felizjueves.ir.interp import Launch, interpret
from felizjueves.performance.device import GFX1200, SM_86

BACKEND = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}
N = 1000  # not a multiple of BLOCK: the last block is ragged
BLOCK = 128
GRID = -(-N // BLOCK)
A = 2.5


def _c(fn: Fn, *args) -> Call:
    return Call(fn, tuple(args))


def axpy_kernel() -> Kernel:
    i, n = Var("i"), Var("n")

    def at(name: str) -> Index:
        return Index(name, (i,))

    out = Intent.OUT
    params = (
        Param("n", Scalar.I32),
        Param("a", Scalar.F64),
        Param("x", Scalar.F64, array=True),
        Param("y", Scalar.F64, array=True),
        Param("u", Scalar.F32, array=True),
        Param("sum", Scalar.F64, array=True, intent=out),
        Param("scaled", Scalar.F64, array=True, intent=out),
        Param("acc", Scalar.F64, array=True, intent=out),
        Param("twice", Scalar.F32, array=True, intent=out),
        Param("idx", Scalar.I32, array=True, intent=out),
    )
    body = (
        Decl("i", Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X),
                                      _c(Fn.MUL, Call(Fn.BLOCK_X),
                                         Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, i, n), (
            Assign(at("sum"), _c(Fn.ADD, at("x"), at("y"))),
            Assign(at("scaled"), _c(Fn.MUL, Var("a"), at("x"))),
            Assign(at("acc"), _c(Fn.ADD, at("acc"), at("y"))),
            Assign(at("twice"), _c(Fn.ADD, at("u"), at("u"))),
            Assign(at("idx"), i),
        )),
    )
    return Kernel("axpy_parts", params, body)


def inputs(seed: int = 649) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"x": rng.uniform(-2.0, 2.0, N), "y": rng.uniform(-2.0, 2.0, N),
            "u": rng.uniform(-2.0, 2.0, N).astype(np.float32),
            "acc": rng.uniform(-2.0, 2.0, N)}


def args(data: dict[str, np.ndarray]) -> dict[str, object]:
    return {"n": N, "a": A, **data, "sum": (N,), "scaled": (N,), "twice": (N,),
            "idx": (N,)}


def numpy_reference(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    x, y, u = data["x"], data["y"], data["u"]
    return {"sum": x + y, "scaled": A * x, "acc": data["acc"] + y,
            "twice": u + u, "idx": np.arange(N, dtype=np.int32)}


def _bitwise(a: np.ndarray, b: np.ndarray) -> bool:
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


# -- the fast tier: no compiler ----------------------------------------------


def test_spec_follows_the_parameter_list():
    data = inputs()
    spec, arrays = launch_spec(axpy_kernel(), GRID, BLOCK, args(data))
    assert spec["kernel"] == "axpy_parts"
    assert spec["grid"] == [GRID] and spec["block"] == [BLOCK]
    assert spec["shared_bytes"] == 0
    assert [a["name"] for a in spec["args"]] == [
        p.name for p in axpy_kernel().params]
    by = {a["name"]: a for a in spec["args"]}
    assert by["n"] == {"name": "n", "type": "i32", "value": N}
    assert by["a"] == {"name": "a", "type": "f64", "value": A}
    assert by["x"] == {"name": "x", "type": "f64", "array": True, "in": "x.in.npy"}
    assert by["u"]["type"] == "f32"
    assert by["acc"] == {"name": "acc", "type": "f64", "array": True,
                         "in": "acc.in.npy", "out": "acc.out.npy"}
    assert by["idx"] == {"name": "idx", "type": "i32", "array": True,
                         "shape": [N], "out": "idx.out.npy"}
    assert sorted(arrays) == ["acc.in.npy", "u.in.npy", "x.in.npy", "y.in.npy"]
    assert _bitwise(arrays["x.in.npy"], data["x"])


def test_write_case_round_trips(tmp_path):
    data = inputs()
    spec, arrays = launch_spec(axpy_kernel(), (GRID, 1), (BLOCK, 1, 1), args(data))
    path = write_case(tmp_path, spec, arrays)
    assert json.loads(path.read_text()) == spec
    for fname, a in arrays.items():
        assert _bitwise(np.load(tmp_path / fname), a)


@pytest.mark.parametrize("change, match", [
    ({"n": None}, "missing"),
    ({"bogus": 1}, "unknown"),
    ({"x": np.zeros(N, dtype=np.float32)}, "dtype"),
    ({"u": np.zeros(N)}, "dtype"),
    ({"x": (N,)}, "IN array takes an ndarray"),
    ({"n": 1.5}, "integer"),
    ({"n": True}, "integer"),
    ({"a": np.zeros(2)}, "scalar"),
    ({"sum": (N, -1)}, "shape"),
], ids=["missing", "extra", "x-f32", "u-f64", "in-shape", "n-float",
        "n-bool", "a-array", "neg-extent"])
def test_spec_refuses(change, match):
    a = args(inputs())
    for k, v in change.items():
        if v is None:
            del a[k]
        else:
            a[k] = v
    with pytest.raises((TypeError, ValueError), match=match):
        launch_spec(axpy_kernel(), GRID, BLOCK, a)


@pytest.mark.parametrize("grid, block", [(0, BLOCK), (GRID, (1, 1, 1, 1)),
                                         (GRID, ())])
def test_spec_refuses_bad_dims(grid, block):
    with pytest.raises(ValueError, match="extents"):
        launch_spec(axpy_kernel(), grid, block, args(inputs()))


def test_f32_scalar_is_rounded_to_float32():
    k = Kernel("f", (Param("s", Scalar.F32), Param("o", Scalar.F32, array=True,
                                                  intent=Intent.OUT)),
               (Assign(Index("o", (Call(Fn.THREAD_X),)), Var("s")),))
    spec, _ = launch_spec(k, 1, 1, {"s": 0.1, "o": (1,)})
    assert spec["args"][0]["value"] == float(np.float32(0.1)) != 0.1


def test_spec_carries_runs_and_warmup_only_when_asked():
    spec, _ = launch_spec(axpy_kernel(), GRID, BLOCK, args(inputs()))
    assert "runs" not in spec and "warmup" not in spec
    spec, _ = launch_spec(axpy_kernel(), GRID, BLOCK, args(inputs()), runs=3,
                          warmup=2)
    assert spec["runs"] == 3 and spec["warmup"] == 2
    for bad in ({"runs": 0}, {"runs": 1.5}, {"warmup": -1}, {"runs": True}):
        with pytest.raises(ValueError, match="integer"):
            launch_spec(axpy_kernel(), GRID, BLOCK, args(inputs()), **bad)


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_runner_source(backend, tmp_path):
    emit, arch, ext = BACKEND[backend]
    unit = tmp_path / f"axpy_parts.{ext}"
    unit.write_text(emit(axpy_kernel(), arch))
    prog = runner_program(backend, unit, ["axpy_parts"], tmp_path / "run")
    assert prog.path == tmp_path / "run" / f"runner_axpy_parts.{ext}"
    assert prog.path.read_text() == prog.source
    assert f'#include "{HARNESS / "runner.h"}"' in prog.source
    assert f'#include "{unit.resolve()}"' in prog.source
    assert "fj_kernels[] = {FJ_KERNEL(axpy_parts)};" in prog.source
    assert prog.units == (unit.resolve(), *HEADERS)
    # The harness is included before the unit, so the vendor runtime lands at
    # file scope first (include/felizjueves/runtime.h's rule).
    assert prog.source.index("runner.h") < prog.source.index(unit.name + '"')


def test_runner_source_refuses(tmp_path):
    unit = tmp_path / "k.cu"
    unit.write_text("")
    with pytest.raises(ValueError, match="cuda or hip"):
        runner_program("gpu", unit, ["k"], tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        runner_program("cuda", unit, [], tmp_path)
    with pytest.raises(ValueError, match="not a kernel name"):
        runner_program("cuda", unit, ["k); evil("], tmp_path)


def test_headers_are_listed_and_plain_cxx():
    """Every header is in HEADERS (so #92's filter covers it), and none uses
    a C++23 module, `import std`, or a FreeBreakfast macro; only fj_gpu.h
    spells a vendor runtime call."""
    assert sorted(HARNESS.glob("*")) == sorted(HEADERS)
    vendor = re.compile(r"\b(?:cuda|hip)[A-Z]\w*\s*\(")
    for h in HEADERS:
        text = h.read_text()
        assert text.splitlines()[0].startswith(f"// {h.name} -- ")
        assert "#pragma once" in text
        assert not re.search(r"^\s*(?:export\s+)?(?:module|import)\b", text, re.M), h
        assert "FBF_" not in text and "free_breakfast" not in text, h
        if h.name != "fj_gpu.h":
            assert not vendor.search(text), f"{h.name} spells a vendor call"


def test_unit_signature_is_the_parameter_list():
    """The runner passes arguments in `kernel.params` order: the emitted
    signature must be that order, one line per parameter."""
    k = axpy_kernel()
    for backend, (emit, arch, _) in BACKEND.items():
        text = emit(k, arch)
        sig = text[text.index(f"void {k.name}("):text.index(")\n{")]
        names = [line.strip().rstrip(",").split()[-1].lstrip("*")
                 for line in sig.splitlines()[1:]]
        assert names == [p.name for p in k.params], backend


def test_interpreted_kernel_matches_numpy_bitwise():
    """The host witness: the IR kernel interpreted equals numpy bit for bit,
    so a card that equals numpy equals the interpreter too."""
    data = inputs()
    want = numpy_reference(data)
    outs = {"sum": np.zeros(N), "scaled": np.zeros(N), "acc": data["acc"].copy(),
            "twice": np.zeros(N, dtype=np.float32),
            "idx": np.zeros(N, dtype=np.int32)}
    got = interpret(axpy_kernel(), launch=Launch(grid=GRID, block=(BLOCK, 1),
                                                lanes=32),
                    n=N, a=A, x=data["x"], y=data["y"], u=data["u"], **outs)
    for name, w in want.items():
        assert _bitwise(np.asarray(got[name]), w), name


# -- compile only: a compiler, no card ---------------------------------------


def _build(tc, backend: str, tmp_path: Path) -> Path:
    emit, arch, ext = BACKEND[backend]
    unit = tmp_path / f"axpy_parts.{ext}"
    unit.write_text(emit(axpy_kernel(), arch))
    prog = runner_program(backend, unit, ["axpy_parts"], tmp_path)
    return build_runner(tc, prog, tmp_path / "runner")


@pytest.mark.cuda
def test_cuda_runner_compiles_without_warnings(nvcc, tmp_path):
    _build(nvcc, "cuda", tmp_path)


@pytest.mark.hip
def test_hip_runner_compiles_without_warnings(hipcc, tmp_path):
    _build(hipcc, "hip", tmp_path)


def _refuses(tc, backend: str, tmp_path: Path) -> None:
    """The runner checks the spec against the compiled signature: a spec
    that lies about a type, the arity or a const pointer exits 1, naming it,
    and writes nothing. All before the first runtime call, so this RUNS the
    binary but needs no card: a compiler's case, not a `gpu` one."""
    binary = _build(tc, backend, tmp_path)
    spec, arrays = launch_spec(axpy_kernel(), GRID, BLOCK, args(inputs()))

    def lie(i: int, **change) -> dict:
        s = json.loads(json.dumps(spec))
        s["args"][i].update(change)
        return s

    cases = {
        "a-f32": (lie(1, type="f32"), "does not match the kernel"),
        "n-array": (lie(0, array=True, shape=[1], out="n.npy"), "takes a scalar"),
        "x-out": (lie(2, out="x.out.npy"), "const"),
        "arity": ({**spec, "args": spec["args"][:-1]}, "takes 10 arguments"),
        "kernel": ({**spec, "kernel": "nope"}, "no kernel 'nope'"),
        "runs": ({**spec, "runs": 0}, "runs must be an integer"),
        "warmup": ({**spec, "warmup": -1}, "warmup must be an integer"),
    }
    for name, (bad, match) in cases.items():
        work = tmp_path / name
        path = write_case(work, bad, arrays)
        done = subprocess.run([str(binary), str(path)], capture_output=True,
                              text=True, timeout=300)
        assert done.returncode == 1 and match in done.stderr, (name, done.stderr)
        assert not list(work.glob("*.out.npy")), name
    done = subprocess.run([str(binary)], capture_output=True, text=True)
    assert done.returncode == 2 and "usage" in done.stderr


@pytest.mark.cuda
def test_cuda_runner_refuses_a_spec_its_kernel_contradicts(nvcc, tmp_path):
    _refuses(nvcc, "cuda", tmp_path)


@pytest.mark.hip
def test_hip_runner_refuses_a_spec_its_kernel_contradicts(hipcc, tmp_path):
    _refuses(hipcc, "hip", tmp_path)


# -- on the card --------------------------------------------------------------


def _on_card(backend: str, request, tmp_path: Path) -> None:
    tc = request.getfixturevalue("nvcc" if backend == "cuda" else "hipcc")
    request.getfixturevalue(f"{backend}_device")
    binary = _build(tc, backend, tmp_path)
    data = inputs()
    want = numpy_reference(data)
    # Three runs: `acc` is read and updated, so this also witnesses that every
    # launch starts from the restored input (the runner fails otherwise).
    outs, times = run_case(binary, axpy_kernel(), tmp_path / "case", GRID,
                           BLOCK, args(data), runs=3, warmup=1)
    assert len(times) == 3 and all(t >= 0 for t in times), times
    assert sorted(outs) == sorted(want)
    for name, w in want.items():
        got = np.load(outs[name])
        assert _bitwise(got, w), (
            f"{backend} {name}: {np.count_nonzero(got != w)} of {N} differ")
        ref = tmp_path / f"ref_{name}.npy"
        np.save(ref, w)
        assert outs[name].read_bytes() == ref.read_bytes(), (
            f"{backend} {name}: the runner's .npy is not np.save's bytes")


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_runner_axpy_bitwise(request, tmp_path):
    _on_card("cuda", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_runner_axpy_bitwise(request, tmp_path):
    _on_card("hip", request, tmp_path)
