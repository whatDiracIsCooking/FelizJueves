"""The fused Fock kernel on the cards (#288): per-class parity against the
host, and the whole-molecule `F` against `walk_f`'s interpreted build and
the libcint fixture pair -- on the RTX 3080 (`sm_86`) and the RX 9060 XT
(`gfx1200`).

THE LAUNCHES ARE WALK_F'S (test/fock_driver.py, `record`): `launch_f`'s plan
(#285) walked by `walk_f` (#286) with a run that records each launch, so the
card replays exactly the launches -- packed records, `diag`, `Dsh`, `D` --
the interpreted build runs, through `kernel_f`'s kernels (#282, #283)
emitted by `emit/` unchanged (#287). One program per kernel set, driven by
its input; `F` is closed on the host (`kernel_f.close`), as walk_f closes
it. Every program is compiled with devtools/config.sh's flags and FAILS on a
warning located in generated text (#92, `plan_driver.compile_checked`). No
F unit is written by KERNEL_GEN_CMD (the Fock path is separate, #287), so
the driver emits every unit itself, the same text `--fock-class` writes.

PART 1, PER CLASS. H2O/cc-pVDZ's plan at `tau = 0` (`n = 1`: nothing
screened, nothing pruned) reaches all 21 classes of milestone 12's ladder;
for each class, the plan's launches of that class -- every batch pair whose
kernel it is -- run on the card into one `F`, against (a) those launches
interpreted (`fock_driver.interpret`: walk_f's executor) and (b) `ao.fock`'s
every-ordered-quartet sweep with its ERIs from `ao.eri` (MMD, the other
engine) restricted to the ordered members of the quartets those launches
meet. The default tier runs the classes of total `L = la + lb + lc + ld <=
2` -- the issue's `ss|ss`, `sp|ss`, `sp|sp`, `sd|ss`, `pp|ss`, which on this
repo's canonical ladder (`hgp.vrr.CLASSES`, `la <= lb`, `lc <= ld`, bra
pair `<=` ket pair) are `(00|00)`, `(00|01)`, `(01|01)`, `(00|02)`,
`(00|11)`; the other 16 are `slow` (F5, the deferred tier #290 runs them).

PART 2, PER MOLECULE. H2O/STO-3G at `tau = 0` and at `tau = 0.1, n = 4`
(the screens fire: test_fock_build.py's case), and H2O/6-31G at `tau = 0`,
every launch of the plan in order into one `F`, against walk_f's
interpreted build of the same plan (`test_fock_build.walk`) and, at `tau =
0`, against the full sweep and libcint's `vj - vk/2` (`manifest.fock_pair`,
which asserts the `_j` and `_k` fixtures share density, shells and
provenance; milestone 12, pin 10). H2O/cc-pVDZ (all 21 classes, `(dd|dd)`
among them) is `slow`.

THE TOLERANCES, stated before any run and taken from the files that state
them, not tuned: `TOL = 1e-13` of `scale_G` element by element against the
sweep -- test_kernel_f.py's bound for one ERI in two operation orders,
carried through the digestion's sums by the triangle inequality, and the
card is one more operation order (FMA contraction and the fp64 atomics'
order only re-associate); `2 TOL` against the interpreted build (the
triangle inequality through the sweep, both sides within `TOL` of it); and
`LIBCINT_TOL = 1e-12` of the same scale against libcint (test_ao_fock.py's).
`scale_G` is the same sweep on `|D|` with `ao.eri`'s absolute-value twin, so
an element no computed quartet touches has scale 0 and must come back
exactly 0.0. The screened case's skip decisions are the same IEEE products
on both sides (`G_ab (G_cd dmax)`, no addition to contract), so the card
and the interpreter compute the same quartets; the full-plan scale bounds
both. Two runs of one binary differ only in the order of the atomics: held
to `2 TOL` of scale too, and whether they repeat bit for bit is printed.

THE MEASURED ROWS (milestone 12; #278's F cell) are not this file's: the
compile side is `python -m felizjueves.measured --remeasure --target f --L
...`, and the clock is measure/measure_fock_ondevice.py's opt-in sweep
(`devtools/measure.sh wallclock-f`), which times the binary part 1 below
checks. The timing inputs' cubes and the narrow `fj_boys`'s witnessed range
(NEAR_SIDE, WIDE_SIDE, BOYS_T_WITNESSED) and the sweep's class selector's
variable (WALLCLOCK_F_L) stay here, where the lever sweeps that reuse them
already read them.

COMPILE-ONLY COMPANIONS (`cuda` / `hip`, no `gpu`) build every program the
device tests run, so a vendor spelling problem shows without the card. The
device tests are `gpu` plus `cuda` or `hip`, behind the card lock and the
VRAM probe: on a box without the card they skip, and kernel-tier.sh exits 2.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from fock_driver import (
    FRec,
    fock_program,
    fock_stage,
    interpret,
    record,
    run_on_card,
)
from plan_driver import compile_checked
from test_fock_build import BOUND, LIBCINT_TOL, TOL, members, molecule, sweep, walk
from test_kernel_device import (
    BACKEND,
    _toolchain,
)

from felizjueves import kernel_f as KF
from felizjueves.ao import fock
from felizjueves.ao.eri import eri_block_and_scale
from felizjueves.hgp.vrr import CLASSES
from felizjueves.ir import Kernel
from felizjueves.launch_f import f_launches
from felizjueves.reference.pq import pq
from felizjueves.walk_f import quartets

PART1 = "h2o_ccpvdz"  # its plan reaches all 21 classes
DEFAULT_L = 2  # the default tier's ceiling on la + lb + lc + ld (the issue)
DEFAULT = tuple(c for c in CLASSES if sum(c) <= DEFAULT_L)
# the issue's names, on this repo's canonical ladder (module docstring)
ISSUE_NAMES = {"ss|ss": (0, 0, 0, 0), "sp|ss": (0, 0, 0, 1),
               "sp|sp": (0, 1, 0, 1), "sd|ss": (0, 0, 0, 2),
               "pp|ss": (0, 0, 1, 1)}
CLASS_PARAMS = [pytest.param(c, id="".join(map(str, c)),
                             marks=() if c in DEFAULT else (pytest.mark.slow,))
                for c in CLASSES]
MOLECULES = [
    pytest.param("h2o_sto3g", 0.0, 1, id="h2o_sto3g-tau0"),
    pytest.param("h2o_sto3g", 0.1, 4, id="h2o_sto3g-tau0.1"),
    pytest.param("h2o_631g", 0.0, 1, id="h2o_631g-tau0",
                 marks=pytest.mark.slow),  # off the PR gate, as the levers' is
    pytest.param("h2o_ccpvdz", 0.0, 1, id="h2o_ccpvdz-tau0",
                 marks=pytest.mark.slow),
]
# #290: the sweep's classes by total L (space-separated); unset, the default
# tier's L <= 2 -- #288's five cells, exactly as before
WALLCLOCK_F_L = "FJ_WALLCLOCK_F_L"
NEAR_SIDE, WIDE_SIDE = 0.25, 5.0  # bohr: the timing inputs' cubes
BOYS_T_WITNESSED = 1e6  # fj_boys's witnessed range (#301, test_kernel_device)


def _worst(got, want, scale) -> float:
    return float(np.max(np.abs(got - want) / np.where(scale > 0, scale, 1.0)))


def _within(got, want, scale, tol) -> bool:
    return bool(np.all(np.abs(got - want) <= tol * scale))


# ---------------------------------------------------------------------------
# The launches, recorded; the classes' references
# ---------------------------------------------------------------------------


@cache
def _plan(name: str, tau: float, n: int) -> tuple[tuple[FRec, ...], object]:
    basis, D, _ = molecule(name)
    plan = f_launches(basis, D, bound=BOUND, tau=tau, n=n)
    recs, _ = record(plan, basis, D)
    return tuple(recs), plan


def _kernels(recs) -> list[Kernel]:
    out: dict[str, Kernel] = {}
    for r in recs:
        out.setdefault(r.name, r.kernel)
    return list(out.values())


def _class_name(cls) -> str:
    return "f_" + "".join(map(str, cls))


@cache
def _class_recs(cls) -> tuple[FRec, ...]:
    recs, _ = _plan(PART1, 0.0, 1)
    got = tuple(r for r in recs if r.name == _class_name(cls))
    assert got, f"{PART1}'s plan launches no {cls}"
    return got


_ERI: dict = {}


def _block(name: str, q):
    """`ao.eri`'s block and scale of the ordered quartet `q`, evaluated
    once per process (no permutation of another)."""
    if (name, q) not in _ERI:
        basis, _, _ = molecule(name)
        sh = basis.shells
        blk, scale = eri_block_and_scale((sh[q[0]], sh[q[1]]),
                                         (sh[q[2]], sh[q[3]]), pq=pq)
        _ERI[name, q] = (np.array(blk), np.array(scale))
    return _ERI[name, q]


def restricted_sweep(name: str, keep) -> tuple[np.ndarray, np.ndarray]:
    """`ao.fock`'s sweep over every ordered quartet, its ERIs zero outside
    the ordered members `keep`: `(G, scale_G)` -- `J - K/2` on `D`, and
    `J + K/2` on `|D|` with `ao.eri`'s absolute-value twin."""
    basis, D, _ = molecule(name)
    idx = {id(s): i for i, s in enumerate(basis.shells)}

    def eri(which):
        def f(bra, ket):
            q = (idx[id(bra[0])], idx[id(bra[1])], idx[id(ket[0])],
                 idx[id(ket[1])])
            if q not in keep:
                return [[0.0] * (ket[0].n_cart * ket[1].n_cart)
                        for _ in range(bra[0].n_cart * bra[1].n_cart)]
            return _block(name, q)[which].tolist()
        return f

    J, K = fock.coulomb_exchange_reference(basis, D.tolist(), eri=eri(0))
    Js, Ks = fock.coulomb_exchange_reference(basis, np.abs(D).tolist(),
                                             eri=eri(1))
    f = fock.EXCHANGE_FRACTION
    return (np.array(J) - f * np.array(K), np.array(Js) + f * np.array(Ks))


def _met(recs) -> frozenset:
    """Every ordered member of the unique quartets the launches meet."""
    qs = []
    for r in recs:
        qs += quartets(r.pairs("bra"), r.pairs("ket"),
                       diagonal=bool(r.args["diag"]))
    assert len(qs) == len(set(qs)), "a quartet met twice"
    return members(qs)


@cache
def _class_reference(cls):
    """(interpreted F closed, sweep G, scale_G) of the class's launches."""
    recs = _class_recs(cls)
    basis, _, _ = molecule(PART1)
    F = np.array(KF.close(interpret(recs, basis.n_ao)))
    G, scale = restricted_sweep(PART1, _met(recs))
    assert np.max(np.abs(G)) > 0  # the comparison is not of zeros
    return F, G, scale


def _check_class(backend: str, cls, Fs: np.ndarray) -> dict[str, float]:
    """Part 1's assertions on the card's runs `Fs` (not closed)."""
    F_int, G, scale = _class_reference(cls)
    got = [np.array(KF.close(F)) for F in Fs]
    w = {"sweep": _worst(got[0], G, scale),
         "interp": _worst(got[0], F_int, scale),
         "runs": _worst(got[0], got[1], scale)}
    assert _within(got[0], G, scale, TOL), (backend, cls, w)
    assert _within(got[0], F_int, scale, 2 * TOL), (backend, cls, w)
    assert _within(got[1], got[0], scale, 2 * TOL), (backend, cls, w)
    w["bitwise_runs"] = float(np.array_equal(Fs[0], Fs[1]))
    return w


def _build(backend: str, tc, kernels, tmp_path: Path, name: str) -> Path:
    name = f"prog_{name}"  # never a unit's own file name
    prog = fock_program(kernels, backend, tmp_path, name=name)
    return compile_checked(tc, prog, tmp_path / f"{name}_bin")


# ---------------------------------------------------------------------------
# Part 1: per class, against the interpreted launches and the sweep
# ---------------------------------------------------------------------------


def _part1(backend: str, cls, request, run_device_binary,
           tmp_path: Path) -> Path:
    tc = _toolchain(backend, request)
    recs = _class_recs(cls)
    kernels = [recs[0].kernel]
    binary = _build(backend, tc, kernels, tmp_path, _class_name(cls))
    basis, _, _ = molecule(PART1)
    Fs, _ = run_on_card(binary, fock_stage(recs, kernels), basis.n_ao,
                        tmp_path, run_device_binary, backend=backend)
    w = _check_class(backend, cls, Fs)
    n_q = sum(r.args["n_bra"] * r.args["n_ket"] for r in recs)
    print(f"\n{backend} {BACKEND[backend][1].name} {cls} on {PART1}: "
          f"{len(recs)} launches, {n_q} bra x ket slots; worst |F - sweep| / "
          f"scale {w['sweep']:.2e} (tol {TOL:g}), |F - interp| / scale "
          f"{w['interp']:.2e} (tol {2 * TOL:g}), runs {w['runs']:.2e}, "
          f"bit-identical {bool(w['bitwise_runs'])}")
    return binary


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_each_class_matches_the_host(cls, request, run_device_binary,
                                          tmp_path) -> None:
    """Part 1 on the RTX 3080."""
    _part1("cuda", cls, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_each_class_matches_the_host(cls, request, run_device_binary,
                                         tmp_path) -> None:
    """Part 1 on the RX 9060 XT."""
    _part1("hip", cls, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Part 2: the whole molecule, against walk_f's interpreted build and libcint
# ---------------------------------------------------------------------------


def _part2(backend: str, name: str, tau: float, n: int, request,
           run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    recs, plan = _plan(name, tau, n)
    kernels = _kernels(recs)
    binary = _build(backend, tc, kernels, tmp_path, f"fock_{name}")
    basis, _, libcint = molecule(name)
    Fs, _ = run_on_card(binary, fock_stage(recs, kernels), basis.n_ao,
                        tmp_path, run_device_binary, backend=backend)
    got = [np.array(KF.close(F)) for F in Fs]
    w_int, _ = walk(name, tau, n)  # test_fock_build's interpreted walk
    F_int = np.array(w_int.F)
    scale = sweep(name, block="scale")
    w = {"interp": _worst(got[0], F_int, scale),
         "runs": _worst(got[0], got[1], scale)}
    assert _within(got[0], F_int, scale, 2 * TOL), (backend, name, tau, w)
    assert _within(got[1], got[0], scale, 2 * TOL), (backend, name, tau, w)
    if tau == 0.0:
        assert not w_int.skipped
        w["sweep"] = _worst(got[0], sweep(name), scale)
        w["libcint"] = _worst(got[0], libcint, scale)
        assert _within(got[0], sweep(name), scale, TOL), (backend, name, w)
        assert _within(got[0], libcint, scale, LIBCINT_TOL), (backend, name, w)
    else:
        assert w_int.kernel_screened and w_int.prelaunch_screened
    print(f"\n{backend} {BACKEND[backend][1].name} {name} tau={tau} n={n}: "
          f"{len(recs)} launches of {len(kernels)} classes "
          f"{[k.name for k in kernels]}; worst / scale "
          + ", ".join(f"{k} {v:.2e}" for k, v in w.items())
          + f"; runs bit-identical {bool(np.array_equal(Fs[0], Fs[1]))}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_cuda_whole_molecule_matches_walk_and_libcint(name, tau, n, request,
                                                      run_device_binary,
                                                      tmp_path) -> None:
    """Part 2 on the RTX 3080."""
    _part2("cuda", name, tau, n, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_hip_whole_molecule_matches_walk_and_libcint(name, tau, n, request,
                                                     run_device_binary,
                                                     tmp_path) -> None:
    """Part 2 on the RX 9060 XT."""
    _part2("hip", name, tau, n, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions (`cuda` / `hip`, no `gpu`): no card needed
# ---------------------------------------------------------------------------


@cache
def _molecule_kernels(name: str, tau: float, n: int) -> tuple[Kernel, ...]:
    """The plan's classes' kernels in first-launch order -- what `_part2`
    compiles -- without recording the launches."""
    basis, D, _ = molecule(name)
    plan = f_launches(basis, D, bound=BOUND, tau=tau, n=n)
    out: dict[str, Kernel] = {}
    for L in plan:
        k = KF.build_f_kernel(L.spec)
        out.setdefault(k.name, k)
    return tuple(out.values())


COMPILE_CELLS = [
    *(pytest.param(("class", c), id=f"class-{''.join(map(str, c))}",
                   marks=() if c in DEFAULT else (pytest.mark.slow,))
      for c in CLASSES),
    *(pytest.param(("molecule", *p.values), id=p.id, marks=p.marks)
      for p in MOLECULES),
]


def _compiles(backend: str, tc, cell, tmp_path: Path) -> None:
    if cell[0] == "class":
        kernels, name = [KF.build_f_kernel(cell[1])], _class_name(cell[1])
    else:
        kernels, name = list(_molecule_kernels(*cell[1:])), f"fock_{cell[1]}"
    _build(backend, tc, kernels, tmp_path, name)


@pytest.mark.cuda
@pytest.mark.parametrize("cell", COMPILE_CELLS)
def test_cuda_fock_programs_compile(cell, nvcc, tmp_path) -> None:
    _compiles("cuda", nvcc, cell, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cell", COMPILE_CELLS)
def test_hip_fock_programs_compile(cell, hipcc, tmp_path) -> None:
    _compiles("hip", hipcc, cell, tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the cells, the programs' text and the staged data
# ---------------------------------------------------------------------------


def test_the_default_classes_are_the_issues_five() -> None:
    assert set(DEFAULT) == set(ISSUE_NAMES.values())
    assert len(DEFAULT) == 5 and len(CLASSES) == 21


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py): walks every class's kernel
def test_part1s_plan_reaches_every_class() -> None:
    recs, _ = _plan(PART1, 0.0, 1)
    assert {r.name for r in recs} == {_class_name(c) for c in CLASSES}


def test_part2s_molecules_kernels_are_the_recorded_launches() -> None:
    for name, tau, n in (("h2o_sto3g", 0.0, 1), ("h2o_sto3g", 0.1, 4)):
        recs, plan = _plan(name, tau, n)
        assert len(recs) == len(plan.launches)
        assert [k.name for k in _kernels(recs)] == [
            k.name for k in _molecule_kernels(name, tau, n)]


def test_the_program_reads_the_kernels_parameters(tmp_path) -> None:
    """Every case passes one argument per kernel parameter, `AB` and `CD`
    only where the kernel has them."""
    kernels = list(_molecule_kernels("h2o_631g", 0.0, 1))
    prog = fock_program(kernels, "hip", tmp_path)
    for k in kernels:
        at = prog.source.index(">>>(", prog.source.index(f"::{k.name}<<<"))
        call = prog.source[at:prog.source.index(");", at)]
        assert call.count(",") == len(k.params) - 1, k.name
        assert ("a_AB" in call) == ("AB" in KF.params_of(
            tuple(int(x) for x in k.name[2:])))


def _unstage(data: np.ndarray, kernels) -> list[dict]:
    """The input sliced as the program slices it, back into each launch's
    arguments (the program's layout, restated)."""
    nl, n_sh, n_ao, tau = int(data[0]), int(data[1]), int(data[2]), data[3]
    heads = data[4:4 + 9 * nl].reshape(nl, 9)
    at = 4 + 9 * nl
    Dsh = data[at:at + n_sh * n_sh]
    at += n_sh * n_sh
    D = data[at:at + n_ao * n_ao]
    at += n_ao * n_ao
    out = []
    for h in heads:
        ki, nb, nk, kab, kcd, dg, th, nd, ni = (int(x) for x in h)
        k = kernels[ki]
        has = {p.name for p in k.params}
        sizes = {"bra_prim": nb * kab * len(KF.REC), "AB": 3 * nb,
                 "G_ab": nb, "ket_prim": nk * kcd * len(KF.REC), "CD": 3 * nk,
                 "G_cd": nk, "bra_sh": 2 * nb, "bra_ao": 2 * nb,
                 "ket_sh": 2 * nk, "ket_ao": 2 * nk}
        args = {"n_bra": nb, "n_ket": nk, "k_ab": kab, "k_cd": kcd,
                "n_sh": n_sh, "n_ao": n_ao, "diag": dg, "tau": tau,
                "Dsh": Dsh, "D": D}
        start = at
        for n in ("bra_prim", "AB", "G_ab", "ket_prim", "CD", "G_cd",
                  "bra_sh", "bra_ao", "ket_sh", "ket_ao"):
            if n in has:
                args[n] = data[at:at + sizes[n]]
                at += sizes[n]
        assert at - start == nd + ni
        out.append((k, nb, th, args))
    assert at == data.size
    return out


def test_the_staged_data_through_the_interpreter_is_the_walk() -> None:
    """The input the card receives, sliced back as the program slices it
    and interpreted launch by launch, gives walk_f's interpreted `F`
    within `2 TOL` of scale (another block width): the staging, checked
    without a card."""
    name = "h2o_sto3g"
    recs, _ = _plan(name, 0.1, 4)
    kernels = _kernels(recs)
    data = fock_stage(recs, kernels)
    basis, _, _ = molecule(name)
    again = []
    for k, nb, th, args in _unstage(data, kernels):
        a = {p.name: args[p.name] for p in k.params if p.name != KF.FOCK}
        again.append(FRec(k, nb, th, {**a, KF.FOCK: None}))
    F = np.array(KF.close(interpret(again, basis.n_ao)))
    w, _ = walk(name, 0.1, 4)
    scale = sweep(name, block="scale")
    assert _within(F, np.array(w.F), scale, 2 * TOL)
    assert np.array_equal(F, np.array(KF.close(interpret(recs, basis.n_ao))))
