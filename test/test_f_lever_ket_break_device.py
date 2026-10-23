"""#352's `ket-break` lever ON THE CARDS: the G-sorted stream's launches,
built with the early break, against the interpreted walks -- on the RTX 3080
(`sm_86`) and the RX 9060 XT (`gfx1200`).

THE LAUNCHES ARE WALK_F'S, RECORDED (test/fock_driver.py's `record`, as
test_fock_ondevice.py's part 2): `launch_f.f_launches(..., ket_break=True)`
walked with a recording `run`, so the card replays the very records, `diag`,
`Dsh`, `D` and per-bra `M_ab` the interpreted build is handed, at the
driver's 32 threads. The program is fock_driver's, with ONE extension made
here rather than in the shared harness: the lever's extra double array
`M_ab[n_bra]` is carried at the END of each launch's double buffer
(`_lever_program` / `_lever_stage` below), and the signature check accepts
it exactly once, right after `G_ab`.

THE CHECK, its tolerances stated before any run (test_fock_ondevice.py's,
not tuned): the card's closed `F` against (a) the interpreted LEVER walk of
the same stream and (b) the interpreted LITERAL walk of the same stream
(the proof says those two are bitwise equal; test_f_lever_ket_break.py
holds that), each to `2 TOL` of `scale_G` element by element -- the card is
one more operation order (FMA contraction and the fp64 atomics' order) --
and two runs of one binary to `2 TOL`. The break's decisions are the same
IEEE products on both sides (`Gb (G_cd Mb)`, no addition to contract), as
the six-block screen's are. Molecules: H2O/STO-3G at `tau = 0.1`, `n = 4`
(the PR tier's) and, `slow` because it is not a small input, H2O/6-31G at
`tau = 1e-3`, `n = 4` -- every class they reach has L <= 4 (asserted), the
milestone's cap.

COMPILE-ONLY COMPANIONS (`cuda` / `hip`, no `gpu`) build the same programs
and one program per F class with L <= 4 with the lever, failing on a warning
in generated text (#92).

Run on a card (nothing above L = 4; without `--full` only H2O/STO-3G,
nothing `slow`):

    devtools/kernel-tier.sh --backend cuda --full -j 6 -- -k ket_break
    devtools/kernel-tier.sh --backend hip  --full -j 6 -- -k ket_break
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import cache
from pathlib import Path

import fock_driver as FD
import numpy as np
import pytest
from plan_driver import compile_checked
from test_f_lever_ket_break import _walks
from test_fock_build import BOUND, TOL, molecule, sweep
from test_kernel_device import BACKEND, _toolchain

from felizjueves.eri.fock import kernel_f as KF
from felizjueves.eri.fock.launch_f import f_launches
from felizjueves.eri.fock.levers import ket_break
from felizjueves.hgp import vrr
from felizjueves.ir import Kernel
from felizjueves.spec import Operator, Spec, Target

NAME, M = ket_break.NAME, ket_break.BOUND
CLASSES = [c for c in vrr.CLASSES if sum(c) <= 4]
MOLECULES = [
    pytest.param("h2o_sto3g", 0.1, 4, id="h2o_sto3g-tau0.1"),
    pytest.param("h2o_631g", 1e-3, 4, id="h2o_631g-tau1e-3",
                 marks=pytest.mark.slow),  # not a small input: the PR tier's cut
]

_SIZES = {"bra_prim": f"(size_t)nb * kab * {len(KF.REC)}",
          "AB": "(size_t)nb * 3", "G_ab": "(size_t)nb",
          "ket_prim": f"(size_t)nk * kcd * {len(KF.REC)}",
          "CD": "(size_t)nk * 3", "G_cd": "(size_t)nk"}


def _check_signature(k: Kernel) -> None:
    names = [p.name for p in k.params]
    if names.count(M) != 1 or names[names.index("G_ab") + 1] != M:
        raise ValueError(f"{k.name}: not a {NAME} kernel ({names})")
    want = [n for n in (*FD.SCALARS, "bra_prim", "AB", "G_ab", M, "bra_sh",
                        "bra_ao", "ket_prim", "CD", "G_cd", "ket_sh",
                        "ket_ao", *FD.SHARED, KF.FOCK) if n in names]
    if names != want:
        raise ValueError(f"{k.name}: signature {names}")


def _case(base):
    def case(i: int, k: Kernel, ns: str) -> str:
        """fock_driver's arm, plus `a_M_ab` carved off the END of the
        launch's double buffer (after every array fock_driver carves)."""
        has = {p.name for p in k.params}
        at = " + ".join(["0", *(_SIZES[n] for n in FD.DOUBLES if n in has)])
        lines = base(i, k, ns).split("\n")
        call = next(j for j, ln in enumerate(lines) if "<<<" in ln)
        lines.insert(call, f"            const double* a_{M} = pd[l] + {at};")
        return "\n".join(lines)
    return case


@contextmanager
def _lever_harness(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(FD, "_check_signature", _check_signature)
        m.setattr(FD, "_case", _case(FD._case))
        yield


def _lever_program(kernels, backend, workdir, name, monkeypatch):
    with _lever_harness(monkeypatch):
        return FD.fock_program(kernels, backend, workdir, name=name)


def _lever_stage(recs, kernels, monkeypatch) -> np.ndarray:
    """fock_driver's staging with `M_ab` appended to each launch's doubles
    (the order `_case` carves it in)."""
    with monkeypatch.context() as m:
        m.setattr(FD, "DOUBLES", (*FD.DOUBLES, M))
        return FD.fock_stage(recs, kernels)


@cache
def _recorded(name: str, tau: float, n: int):
    basis, D, _ = molecule(name)
    plan = f_launches(basis, D, bound=BOUND, tau=tau, n=n, ket_break=True)
    recs, w = FD.record(plan, basis, D)
    out: dict[str, Kernel] = {}
    for r in recs:
        out.setdefault(r.name, r.kernel)
    return tuple(recs), tuple(out.values()), w


def _build(backend, tc, kernels, tmp_path: Path, name: str, monkeypatch):
    name = f"prog_{name}"
    prog = _lever_program(kernels, backend, tmp_path, name, monkeypatch)
    return compile_checked(tc, prog, tmp_path / f"{name}_bin")


def _worst(got, want, scale) -> float:
    return float(np.max(np.abs(got - want) / np.where(scale > 0, scale, 1.0)))


def _on_card(backend, name, tau, n, request, run_device_binary, tmp_path,
             monkeypatch) -> None:
    tc = _toolchain(backend, request)
    recs, kernels, w32 = _recorded(name, tau, n)
    binary = _build(backend, tc, list(kernels), tmp_path, f"kb_{name}",
                    monkeypatch)
    basis, _, _ = molecule(name)
    Fs, _ = FD.run_on_card(binary, _lever_stage(recs, kernels, monkeypatch),
                           basis.n_ao, tmp_path, run_device_binary,
                           backend=backend)
    got = [np.array(KF.close(F)) for F in Fs]
    wb, wl, _ = _walks(name, tau, n)  # interpreted: lever, literal
    scale = sweep(name, block="scale")
    w = {"lever": _worst(got[0], np.array(wb.F), scale),
         "literal": _worst(got[0], np.array(wl.F), scale),
         "runs": _worst(got[0], got[1], scale)}
    assert w32.kernel_broken, "the break never fired at 32 threads"
    for k, v in w.items():
        assert v <= 2 * TOL, (backend, name, k, w)
    print(f"\n{backend} {BACKEND[backend][1].name} {name} tau={tau} n={n} "
          f"{NAME}: {len(recs)} launches of {[k.name for k in kernels]}; "
          f"{len(w32.kernel_broken)} quartets broken at 32 threads; worst / "
          "scale " + ", ".join(f"{k} {v:.2e}" for k, v in w.items())
          + f"; runs bit-identical {bool(np.array_equal(Fs[0], Fs[1]))}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_cuda_ket_break_matches_the_interpreted_walks(
        name, tau, n, request, run_device_binary, tmp_path,
        monkeypatch) -> None:
    _on_card("cuda", name, tau, n, request, run_device_binary, tmp_path,
             monkeypatch)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_hip_ket_break_matches_the_interpreted_walks(
        name, tau, n, request, run_device_binary, tmp_path,
        monkeypatch) -> None:
    _on_card("hip", name, tau, n, request, run_device_binary, tmp_path,
             monkeypatch)


# ---------------------------------------------------------------------------
# Compile-only companions
# ---------------------------------------------------------------------------

COMPILE_CELLS = [
    *(pytest.param(("class", c), id=f"class-{''.join(map(str, c))}")
      for c in CLASSES),
    *(pytest.param(("molecule", *p.values), id=p.id, marks=p.marks)
      for p in MOLECULES),
]


def _compiles(backend, tc, cell, tmp_path, monkeypatch) -> None:
    if cell[0] == "class":
        kernels = [KF.build_f_kernel(Spec(Operator.COULOMB, 0, cell[1],
                                          Target.F, levers=(NAME,)))]
        name = "kb_" + "".join(map(str, cell[1]))
    else:
        _, kernels, _ = _recorded(*cell[1:])
        kernels, name = list(kernels), f"kb_{cell[1]}"
    _build(backend, tc, kernels, tmp_path, name, monkeypatch)


@pytest.mark.cuda
@pytest.mark.parametrize("cell", COMPILE_CELLS)
def test_cuda_ket_break_programs_compile(cell, nvcc, tmp_path,
                                         monkeypatch) -> None:
    _compiles("cuda", nvcc, cell, tmp_path, monkeypatch)


@pytest.mark.hip
@pytest.mark.parametrize("cell", COMPILE_CELLS)
def test_hip_ket_break_programs_compile(cell, hipcc, tmp_path,
                                        monkeypatch) -> None:
    _compiles("hip", hipcc, cell, tmp_path, monkeypatch)


# ---------------------------------------------------------------------------
# Fast tier: the harness extension and the cells
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_the_device_molecules_reach_only_l_le_4_and_break(name, tau, n
                                                         ) -> None:
    recs, kernels, w = _recorded(name, tau, n)
    assert recs and kernels
    for r in recs:
        assert M in r.args and len(r.args[M]) == r.args["n_bra"]
        _check_signature(r.kernel)
    basis, D, _ = molecule(name)
    plan = f_launches(basis, D, bound=BOUND, tau=tau, n=n, ket_break=True)
    assert all(sum(L.spec.am) <= 4 and L.spec.levers == (NAME,)
               for L in plan.launches)
    assert w.kernel_broken


def test_the_lever_program_carves_m_ab_last(tmp_path, monkeypatch) -> None:
    """The arm carves `a_M_ab` after every double fock_driver carves, and
    the staged buffer puts `M_ab` there: the interpreter, fed the unstaged
    arrays, reproduces the recorded arguments."""
    recs, kernels, _ = _recorded("h2o_sto3g", 0.1, 4)
    prog = _lever_program(list(kernels), "cuda", tmp_path, "prog_kb",
                          monkeypatch)
    for k in kernels:
        has = {p.name for p in k.params}
        at = " + ".join(["0", *(_SIZES[x] for x in FD.DOUBLES if x in has)])
        assert f"const double* a_{M} = pd[l] + {at};" in prog.source
    data = _lever_stage(recs, kernels, monkeypatch)
    # the M_ab values sit at the end of each launch's doubles
    head = FD.HEADER + FD.PER_LAUNCH * len(recs)
    basis, _, _ = molecule("h2o_sto3g")
    at = head + len(basis.shells) ** 2 + basis.n_ao ** 2
    for i, r in enumerate(recs):
        nd = int(data[FD.HEADER + FD.PER_LAUNCH * i + 7])
        ni = int(data[FD.HEADER + FD.PER_LAUNCH * i + 8])
        doubles = data[at:at + nd]
        assert np.array_equal(doubles[-r.args["n_bra"]:],
                              np.asarray(r.args[M], dtype=float))
        at += nd + ni
    assert at == data.size
