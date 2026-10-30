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
card.

THE SEQUENCE (#657) is two units in one runner (`axpy_parts` and its twin
under another name) and two launches over shared buffers: the second reads
the first's `sum` at an offset and adds into the same `acc`, so carrying,
offsets and the restore before each run are all witnessed bit for bit:

    devtools/kernel-tier.sh --backend cuda -- -k harness
    devtools/kernel-tier.sh --backend hip -- -k harness
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from harness_driver import (
    HARNESS,
    HEADERS,
    Ref,
    Step,
    Zeros,
    build_runner,
    launch_spec,
    run_case,
    run_sequence,
    runner_program,
    sequence_spec,
    units_runner,
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


# -- sequences (#657) ---------------------------------------------------------

SHIFT = 8  # the second launch reads `sum` this many elements in


def twin_kernel() -> Kernel:
    k = axpy_kernel()
    return Kernel("axpy_twin", k.params, k.body)


def sequence(data: dict[str, np.ndarray]) -> tuple[dict, list[Step]]:
    """Launch 1 is `axpy_parts` on the inputs; launch 2 is its twin on
    `x = sum[SHIFT:]`, over `N - SHIFT` elements, adding into the same `acc`."""
    bufs = {**data, "sum": Zeros((N,)), "scaled": Zeros((N,)),
            "twice": Zeros((N,), Scalar.F32), "idx": Zeros((N,), Scalar.I32),
            "sum2": Zeros((N,)), "scaled2": Zeros((N,)),
            "twice2": Zeros((N,), Scalar.F32), "idx2": Zeros((N,), Scalar.I32)}
    one = {"n": N, "a": A, "x": "x", "y": "y", "u": "u", "sum": "sum",
           "scaled": "scaled", "acc": "acc", "twice": "twice", "idx": "idx"}
    two = {**one, "n": N - SHIFT, "x": Ref("sum", SHIFT), "sum": "sum2",
           "scaled": "scaled2", "twice": "twice2", "idx": "idx2"}
    return bufs, [Step(axpy_kernel(), GRID, BLOCK, one),
                  Step(twin_kernel(), GRID, BLOCK, two)]


def sequence_reference(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    x, y, u, m = data["x"], data["y"], data["u"], N - SHIFT
    s = x + y
    acc = data["acc"] + y
    acc[:m] = acc[:m] + y[:m]

    def head(a, fill):
        out = np.full_like(fill, 0)
        out[:m] = a[:m]
        return out

    return {"sum": s, "acc": acc, "idx": np.arange(N, dtype=np.int32),
            "sum2": head(s[SHIFT:] + y[:m], s),
            "scaled2": head(A * s[SHIFT:], s),
            "twice2": head(u + u, u),
            "idx2": head(np.arange(N, dtype=np.int32), np.zeros(N, np.int32))}


SEQ_OUTS = ("sum", "acc", "idx", "sum2", "scaled2", "twice2", "idx2")


def test_sequence_spec_carries_the_clock_switches_only_when_asked():
    bufs, steps = sequence(inputs())
    spec, _ = sequence_spec(bufs, steps, outs=SEQ_OUTS)
    assert "timed_readback" not in spec and "launch_events" not in spec
    spec, _ = sequence_spec(bufs, steps, outs=SEQ_OUTS, timed_readback=True,
                            launch_events=False)
    assert spec["timed_readback"] is True and spec["launch_events"] is False


def test_sequence_reference_is_the_interpreted_sequence():
    """The two launches interpreted, the second on `sum[SHIFT:]` and the
    first's `acc`, equal `sequence_reference` bit for bit."""
    data = inputs()
    launch = Launch(grid=GRID, block=(BLOCK, 1), lanes=32)
    zeros = {"sum": np.zeros(N), "scaled": np.zeros(N),
             "twice": np.zeros(N, dtype=np.float32),
             "idx": np.zeros(N, dtype=np.int32)}
    one = interpret(axpy_kernel(), launch=launch, n=N, a=A, x=data["x"],
                    y=data["y"], u=data["u"], acc=data["acc"].copy(),
                    **{k: v.copy() for k, v in zeros.items()})
    x2 = np.concatenate([np.asarray(one["sum"])[SHIFT:], np.zeros(SHIFT)])
    two = interpret(twin_kernel(), launch=launch, n=N - SHIFT, a=A, x=x2,
                    y=data["y"], u=data["u"], acc=np.asarray(one["acc"]).copy(),
                    **{k: v.copy() for k, v in zeros.items()})
    got = {"sum": one["sum"], "idx": one["idx"], "acc": two["acc"],
           **{f"{k}2": two[k] for k in ("sum", "scaled", "twice", "idx")}}
    for name, w in sequence_reference(data).items():
        assert _bitwise(np.asarray(got[name]), w), name


def test_sequence_spec_names_buffers_once():
    data = inputs()
    bufs, steps = sequence(data)
    spec, arrays = sequence_spec(bufs, steps, outs=SEQ_OUTS, runs=2,
                                 witness="each")
    assert sorted(arrays) == ["acc.in.npy", "u.in.npy", "x.in.npy", "y.in.npy"]
    by = {b["name"]: b for b in spec["buffers"]}
    assert by["sum"] == {"name": "sum", "type": "f64", "shape": [N],
                         "out": "sum.out.npy"}
    assert by["idx2"]["type"] == "i32" and by["x"]["in"] == "x.in.npy"
    assert "out" not in by["scaled"]
    one, two = spec["launches"]
    assert one["kernel"] == "axpy_parts" and two["kernel"] == "axpy_twin"
    assert [a["name"] for a in two["args"]] == [p.name for p in
                                                axpy_kernel().params]
    x = two["args"][2]
    assert x == {"name": "x", "type": "f64", "array": True, "buffer": "sum",
                 "offset": SHIFT}
    assert "offset" not in one["args"][2] and "zero" not in one
    assert spec["runs"] == 2 and spec["witness"] == "each"
    assert "warmup" not in spec


@pytest.mark.parametrize("change, error, match", [
    ({"outs": ("nope",)}, ValueError, "name no buffer"),
    ({"witness": "close"}, ValueError, "witness"),
    ({"runs": 0}, ValueError, "runs"),
    ({"warmup": -1}, ValueError, "warmup"),
    ({"steps": []}, ValueError, "at least one step"),
    ({"arg": ("x", "nope")}, ValueError, "no buffer 'nope'"),
    ({"arg": ("x", "u")}, TypeError, "float32"),
    ({"arg": ("x", Ref("sum", N + 1))}, ValueError, "outside"),
    ({"arg": ("x", np.zeros(N))}, TypeError, "names a buffer"),
    ({"arg": ("n", "x")}, TypeError, "scalar"),
    ({"drop": "a"}, ValueError, r"missing \['a'\]"),
    ({"zero": ("nope",)}, ValueError, "zero names no buffer"),
    ({"buffer": ("bad", [1.0])}, TypeError, "ndarray or Zeros"),
    ({"buffer": ("bad", np.zeros(3, np.int16))}, TypeError, "dtype int16"),
])
def test_sequence_spec_refuses(change, error, match):
    bufs, steps = sequence(inputs())
    outs = change.get("outs", SEQ_OUTS)
    one = dict(steps[0].args)
    if "arg" in change:
        one[change["arg"][0]] = change["arg"][1]
    if "drop" in change:
        del one[change["drop"]]
    if "buffer" in change:
        bufs[change["buffer"][0]] = change["buffer"][1]
    steps = [Step(steps[0].kernel, GRID, BLOCK, one,
                  zero=change.get("zero", ())), steps[1]]
    steps = change.get("steps", steps)
    kw = {k: change[k] for k in ("witness", "runs", "warmup") if k in change}
    with pytest.raises(error, match=match):
        sequence_spec(bufs, steps, outs=outs, **kw)


def _units(backend: str, tmp_path: Path) -> list[tuple[Path, list[str]]]:
    emit, arch, ext = BACKEND[backend]
    out = []
    for k in (axpy_kernel(), twin_kernel()):
        unit = tmp_path / f"{k.name}.{ext}"
        unit.write_text(emit(k, arch))
        out.append((unit, [k.name]))
    return out


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_units_runner_source(backend, tmp_path):
    units = _units(backend, tmp_path)
    prog = units_runner(backend, units, tmp_path / "run", "pair")
    assert prog.path == tmp_path / "run" / f"pair.{BACKEND[backend][2]}"
    assert prog.units == (*(u.resolve() for u, _ in units), *HEADERS)
    for i, (unit, (k,)) in enumerate(units):
        assert f'namespace fj_u{i} {{\n#include "{unit.resolve()}"\n}}' in prog.source
        assert f'::fj::kernel_entry("{k}", &fj_u{i}::{k})' in prog.source
    assert prog.source.index("runner.h") < prog.source.index("namespace fj_u0")


def test_units_runner_refuses(tmp_path):
    units = _units("cuda", tmp_path)
    (a, _), (b, _) = units
    for bad, match in (
            ({"backend": "gpu"}, "cuda or hip"),
            ({"units": []}, "at least one unit"),
            ({"units": [(a, [])]}, "no kernel"),
            ({"units": [(a, ["axpy_parts"]), (b, ["axpy_parts"])]}, "two units"),
            ({"units": [(a, ["k); evil("])]}, "not a kernel name"),
            ({"name": "axpy_parts"}, "overwrite"),
            ({"name": "a b"}, "not a runner name")):
        kw = {"backend": "cuda", "units": units, "name": "pair", **bad}
        with pytest.raises(ValueError, match=match):
            units_runner(kw["backend"], kw["units"], tmp_path / "r", kw["name"])


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


def _build_pair(tc, backend: str, tmp_path: Path) -> Path:
    prog = units_runner(backend, _units(backend, tmp_path), tmp_path, "pair")
    return build_runner(tc, prog, tmp_path / "pair_runner")


def _sequence_refuses(tc, backend: str, tmp_path: Path) -> None:
    """A sequence the runner must refuse before the card: an unknown
    buffer, a type its buffer contradicts, an offset past its end, both
    forms at once, an unknown witness. Run with no card, as `_refuses`."""
    binary = _build_pair(tc, backend, tmp_path)
    bufs, steps = sequence(inputs())
    spec, arrays = sequence_spec(bufs, steps, outs=SEQ_OUTS)

    def arg(i: int, **change) -> dict:
        s = json.loads(json.dumps(spec))
        s["launches"][1]["args"][i].update(change)
        return s

    cases = {
        "buffer": (arg(2, buffer="nope"), "no buffer 'nope'"),
        "type": (arg(2, buffer="u"), "buffer u is f32"),
        "offset": (arg(2, offset=N + 1), "outside buffer sum"),
        "both": ({**spec, "kernel": "axpy_parts"}, "not both"),
        "witness": ({**spec, "witness": "close"}, "witness must be"),
        "empty": ({**spec, "launches": []}, "non-empty"),
        "twice": ({**spec, "buffers": spec["buffers"] + spec["buffers"][:1]},
                  "listed twice"),
        "zero": ({**spec, "launches": [{**spec["launches"][0],
                                        "zero": ["nope"]}]},
                 "`zero` names no buffer"),
    }
    for name, (bad, match) in cases.items():
        work = tmp_path / name
        path = write_case(work, bad, arrays)
        done = subprocess.run([str(binary), str(path)], capture_output=True,
                              text=True, timeout=300)
        assert done.returncode == 1 and match in done.stderr, (name, done.stderr)
        assert not list(work.glob("*.out*.npy")), name


@pytest.mark.cuda
def test_cuda_sequence_runner_refuses_what_it_cannot_launch(nvcc, tmp_path):
    _sequence_refuses(nvcc, "cuda", tmp_path)


@pytest.mark.hip
def test_hip_sequence_runner_refuses_what_it_cannot_launch(hipcc, tmp_path):
    _sequence_refuses(hipcc, "hip", tmp_path)


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


def _sequence_on_card(backend: str, request, tmp_path: Path) -> None:
    tc = request.getfixturevalue("nvcc" if backend == "cuda" else "hipcc")
    request.getfixturevalue(f"{backend}_device")
    binary = _build_pair(tc, backend, tmp_path)
    data = inputs()
    want = sequence_reference(data)
    bufs, steps = sequence(data)
    # Identical: three runs after a warmup, `acc` restored before each.
    got = run_sequence(binary, tmp_path / "same", bufs, steps, outs=SEQ_OUTS,
                       runs=3, warmup=1)
    assert len(got.events_ns) == 3 and len(got.wall_ns) == 3
    assert all(len(r) == 2 for r in got.launch_ns), got.launch_ns
    for name, w in want.items():
        (path,) = got.outs[name]
        assert _bitwise(np.load(path), w), (backend, name)
    # Each: both runs kept, both the reference.
    each = run_sequence(binary, tmp_path / "each", bufs, steps, outs=SEQ_OUTS,
                        runs=2, witness="each")
    for name, w in want.items():
        runs = each.load(name)
        assert len(runs) == 2 and all(_bitwise(r, w) for r in runs), name
    # #659's whole-build clock: the read-back inside each run, only the run's
    # two events, and buffers zeroed by the first launch (never uploaded or
    # restored -- three identical runs witness the zero is their state).
    zeroed = [replace(steps[0], zero=("sum2", "idx2")), steps[1]]
    timed = run_sequence(binary, tmp_path / "timed", bufs, zeroed,
                         outs=SEQ_OUTS, runs=3, warmup=1, timed_readback=True,
                         launch_events=False)
    assert len(timed.readback_ns) == 3 and all(r > 0 for r in timed.readback_ns)
    assert timed.launch_ns == [[], [], []] and timed.upload_ns > 0
    assert timed.launched == [axpy_kernel().name, twin_kernel().name]
    for name, w in want.items():
        (path,) = timed.outs[name]
        assert _bitwise(np.load(path), w), (backend, "timed", name)


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_sequence_carries_buffers_bitwise(request, tmp_path):
    _sequence_on_card("cuda", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_sequence_carries_buffers_bitwise(request, tmp_path):
    _sequence_on_card("hip", request, tmp_path)
