"""The fused Fock kernel on the cards (#288): per-class parity against the
host, the whole-molecule `F` against `walk_f`'s interpreted build and the
libcint fixture pair, and the measured rows' wall-clock -- on the RTX 3080
(`sm_86`) and the RX 9060 XT (`gfx1200`).

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

THE MEASURED ROWS (milestone 12; #278's F cell). One row per (class, arch,
toolchain): #288 recorded the five classes the default tier runs, #290 (the
deferred tier) the sixteen above them. The compile side -- registers,
spill, static shared bytes -- is `python -m felizjueves.measured
--remeasure --target f --L ...` (`measured.py`, THE F CELLS). The clock
is this file's opt-in sweep, `test_<backend>_f_wallclock`: it skips unless
FJ_WALLCLOCK_STORE names the store and FJ_WALLCLOCK_F=1, and must run
ALONE (a contended card is a different experiment). FJ_WALLCLOCK_F_L
selects its classes by total L (unset: L <= 2, #288's five):

    FJ_WALLCLOCK_STORE=data/measured/kernels.json FJ_WALLCLOCK_F=1 \\
        FJ_WALLCLOCK_F_L="3 4 5 6 7 8" \\
        devtools/kernel-tier.sh --backend cuda -- -k f_wallclock

1. COMPILE each class's program, then CHECK it by part 1 -- the timed
   binary is the checked binary -- and require its compile row (the same
   emitted text, `source_sha256`).
2. TIME each class on two inputs (`fock_driver.synthetic_launch`: one
   off-diagonal launch, `TIMING_N_BRA x TIMING_N_KET` = 2048 x 128
   contracted quartets of 3-primitive shells, `K_AB = K_CD = 9`, `tau = 0`),
   NEAR (centres in a 0.25-bohr cube: Boys `T` below ~1) then WIDE (a
   5-bohr cube: `T` up to ~250, inside the narrow `fj_boys`'s witnessed
   `[0, 500]`, asserted): `WARMUP` untimed and `REPEATS` timed sequences,
   each the launch and its synchronize under the host's steady clock, the
   zeroing outside it. WIDE is the row's `runtime_ns` (median) and
   `spread_ns` (IQR); NEAR is raw `wallclock:near:*`.
3. THE GATE (`felizjueves.gate`): before each input the card at most
   GPU_BUSY_MAX busy and the CPU at most BUSY_MAX; one reading GPU_SETTLE_S
   after. A phase the gate refuses, or whose after-reading is non-idle or
   UNKNOWN, is discarded and retaken (up to STUDY_RETAKES, counted in raw
   `wallclock:gate_discarded_phases`), never recorded: an F row's four
   `GPU_UTIL_FIELDS` must be readings (#278), so an ungated `runtime_ns`
   is absent, never unlabelled.
4. WRITE each row as it is timed, `combine`d into its compile row.
Absolute times are not comparable across vendors, and nothing is compared
against `[Fock]`'s A100 timings (milestone 12, pin 2).

COMPILE-ONLY COMPANIONS (`cuda` / `hip`, no `gpu`) build every program the
device tests run, so a vendor spelling problem shows without the card. The
device tests are `gpu` plus `cuda` or `hip`, behind the card lock and the
VRAM probe: on a box without the card they skip, and kernel-tier.sh exits 2.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import shlex
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from fock_driver import (
    TIMING_N_BRA,
    TIMING_N_KET,
    FRec,
    fock_program,
    fock_stage,
    interpret,
    max_boys_T,
    record,
    reps,
    run_on_card,
    synthetic_launch,
)
from plan_driver import compile_checked
from test_fock_build import BOUND, LIBCINT_TOL, TOL, members, molecule, sweep, walk
from test_kernel_device import (
    BACKEND,
    GPU_BUSY_MAX,
    REPEATS,
    STUDY_RETAKES,
    WALLCLOCK_RESUME,
    WALLCLOCK_STORE,
    WARMUP,
    _cpu_busy,
    _cpu_list,
    _stats,
    _toolchain,
    check_vram,
)
from tiers import is_fast

from felizjueves import gate, measured
from felizjueves import kernel_f as KF
from felizjueves.ao import fock
from felizjueves.ao.eri import eri_block_and_scale
from felizjueves.device import Arch, MeasuredKey
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
    pytest.param("h2o_631g", 0.0, 1, id="h2o_631g-tau0"),
    pytest.param("h2o_ccpvdz", 0.0, 1, id="h2o_ccpvdz-tau0",
                 marks=pytest.mark.slow),
]
WALLCLOCK_F = "FJ_WALLCLOCK_F"
# #290: the sweep's classes by total L (space-separated); unset, the default
# tier's L <= 2 -- #288's five cells, exactly as before
WALLCLOCK_F_L = "FJ_WALLCLOCK_F_L"
NEAR_SIDE, WIDE_SIDE = 0.25, 5.0  # bohr: the timing inputs' cubes
BOYS_T_WITNESSED = 500.0  # the narrow fj_boys's witnessed range (limits.json)


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
# The measured rows' clock: opt-in, run alone (module docstring)
# ---------------------------------------------------------------------------


def _source_sha(backend: str, k: Kernel) -> str:
    """`measured.cell_sha256` of the class's one unit: its emitted text."""
    emit, arch, _ = BACKEND[backend]
    return hashlib.sha256(emit(k, arch).encode()).hexdigest()


def _wallclock_classes(env=os.environ) -> tuple:
    """The F sweep's classes, in ladder order: those of total L in
    FJ_WALLCLOCK_F_L (#290), or the default tier's when it is unset."""
    text = env.get(WALLCLOCK_F_L, "").strip()
    if not text:
        return DEFAULT
    Ls = {int(x) for x in text.split()}
    got = tuple(c for c in CLASSES if sum(c) in Ls)
    assert got, f"{WALLCLOCK_F_L}={text!r} selects no class"
    return got


def _f_wallclock(backend: str, request, run_device_binary,
                 tmp_path: Path) -> None:
    store_path = os.environ.get(WALLCLOCK_STORE)
    if not store_path or not os.environ.get(WALLCLOCK_F, "").strip():
        pytest.skip(f"the F wall-clock is opt-in and must run alone, "
                    f"uncontended: set {WALLCLOCK_STORE} and {WALLCLOCK_F}=1 "
                    "(see the module docstring)")
    tc = _toolchain(backend, request)
    arch: Arch = BACKEND[backend][1]
    b = measured.BACKENDS[backend]
    toolchain, gpu = b.toolchain(b.compiler()), b.gpu()
    store = measured.load(Path(store_path))
    lanes = arch.wavefront

    cpuset, cpus = measured.config_value("CPUSET"), measured.config_value("CPUS")
    if cpuset:
        os.sched_setaffinity(0, _cpu_list(cpuset))
    print(f"\nCPUSET={cpuset or '(empty)'} CPUS={cpus or '(empty)'}"
          + (" -- CPUS is a container quota, not applied here" if cpus else ""))

    def key(cls) -> MeasuredKey:
        return MeasuredKey(KF.f_class_spec(*cls), arch, toolchain)

    sweep_cells = list(_wallclock_classes())
    if os.environ.get(WALLCLOCK_RESUME, "").strip():
        def done(cls) -> bool:
            r = store.get(key(cls).key())
            return r is not None and r.runtime_ns is not None
        print(f"{WALLCLOCK_RESUME}: {sum(map(done, sweep_cells))} cells "
              "already timed, skipped")
        sweep_cells = [c for c in sweep_cells if not done(c)]
        if not sweep_cells:
            pytest.skip(f"{WALLCLOCK_RESUME}: every cell already timed")

    # 1. compile, check (the timed binary is the checked binary), and the
    #    compile row of the same text must be there first
    cells = []
    for cls in sweep_cells:
        work = tmp_path / _class_name(cls)
        binary = _part1(backend, cls, request, run_device_binary, work)
        k = _class_recs(cls)[0].kernel
        sha = _source_sha(backend, k)
        old = store.get(key(cls).key())
        assert old is not None and old.source_sha256 == sha, (
            f"{key(cls).key()}: no compile row of this text -- run "
            "`python -m felizjueves.measured --remeasure --target f --L "
            f"{sum(cls)}` first")
        inputs = {}
        for label, side, seed in (("near", NEAR_SIDE, 288_000),
                                  ("wide", WIDE_SIDE, 288_100)):
            rec, basis = synthetic_launch(cls, side=side, seed=seed + sum(cls))
            assert max_boys_T(basis, side) <= BOYS_T_WITNESSED
            inputs[label] = (fock_stage([rec], [rec.kernel]), basis.n_ao,
                             rec, len(basis.shells))
            assert rec.kernel == k
        cells.append((cls, binary, sha, inputs))
    command = " ".join([Path(tc.path).name,
                        *shlex.split(measured.config_value(b.flags_var))])

    # 2-3. time, one cell at a time, near then wide, behind the gate
    discarded = [0]

    def timed(binary: Path, data: np.ndarray, n_ao: int, label: str):
        work = tmp_path / "timing"
        work.mkdir(exist_ok=True)
        src = work / "timing.in"
        data.tofile(src)
        check_vram(binary, (str(src), str(work / "timing.out")), backend)
        for take in range(1 + STUDY_RETAKES):
            try:
                busy, util = gate.wait_quiet(
                    backend, say=lambda line: print(line, flush=True))
            except gate.Contended as e:
                discarded[0] += 1
                print(f"  DISCARDED {label}: {e} (take {take + 1})", flush=True)
                continue
            Fs, out = run_on_card(binary, data, n_ao, work, run_device_binary,
                                  backend=backend, timing=(WARMUP, REPEATS),
                                  tag="timing")
            card = (util, gate.after_reading(backend))
            if card[0] is not None and card[1] is not None \
                    and card[1] <= GPU_BUSY_MAX:
                break
            discarded[0] += 1
            print(f"  DISCARDED {label}: card util {card[0]} -> {card[1]}% "
                  f"(take {take + 1})", flush=True)
        else:
            pytest.fail(f"{label}: the box or the card stayed contended "
                        f"through {STUDY_RETAKES + 1} takes -- not recorded")
        wall, events = reps(out)
        assert len(wall) == len(events) == REPEATS, out
        assert np.all(np.isfinite(Fs)), f"{label}: non-finite F"
        return wall, busy, card, bool(np.array_equal(Fs[0], Fs[1])), events

    for cls, binary, sha, inputs in cells:
        discarded[0] = 0
        load1 = os.getloadavg()[0]
        tag = key(cls).key()
        near = timed(binary, inputs["near"][0], inputs["near"][1],
                     f"{tag} near")
        wide = timed(binary, inputs["wide"][0], inputs["wide"][1],
                     f"{tag} wide")
        busy_after = _cpu_busy(0.5)
        wall, busy, card, rep, events = wide
        q1, med, q3 = (float(x) for x in np.percentile(wall, [25, 50, 75]))
        rec = inputs["wide"][2]
        raw = {
            **_stats("wallclock:near:", near[0], near[1], near[2]),
            "wallclock:warmup": WARMUP,
            "wallclock:n_bra": TIMING_N_BRA,
            "wallclock:n_ket": TIMING_N_KET,
            "wallclock:threads": rec.threads,
            "wallclock:lanes": lanes,
            "wallclock:k_ab": rec.args["k_ab"],
            "wallclock:k_cd": rec.args["k_cd"],
            # #290: which synthetic basis (fock_driver's SYNTHETIC_COUNTS,
            # 104 shells, or DEFERRED_COUNTS, 198) -- same launch shape
            "wallclock:n_shells": inputs["wide"][3],
            "wallclock:near_side_milli_bohr": round(1000 * NEAR_SIDE),
            "wallclock:wide_side_milli_bohr": round(1000 * WIDE_SIDE),
            "wallclock:min_ns": int(min(wall)),
            "wallclock:q1_ns": round(q1),
            "wallclock:q3_ns": round(q3),
            "wallclock:max_ns": int(max(wall)),
            "wallclock:events_median_ns": round(float(np.median(events))),
            f"wallclock:{_class_name(cls)}:events_median_ns": round(
                float(np.median(events))),
            "wallclock:near:events_median_ns": round(
                float(np.median(near[4]))),
            "wallclock:cpu_busy_permille_before": round(1000 * busy),
            "wallclock:cpu_busy_permille_after": round(1000 * busy_after),
            "wallclock:loadavg1_milli": round(1000 * load1),
            "wallclock:gpu_util_before": card[0],
            "wallclock:gpu_util_after": card[1],
            "wallclock:F_bitwise_repeatable": int(rep),
            "wallclock:near:F_bitwise_repeatable": int(near[3]),
            "wallclock:gate_discarded_phases": discarded[0],
        }
        row = measured.Measured(
            runtime_ns=round(med), repeats=REPEATS, spread_ns=round(q3 - q1),
            toolchain=toolchain, arch=arch.name, host=measured.host(), gpu=gpu,
            date=datetime.date.today().isoformat(), kernel=_class_name(cls),
            source_sha256=sha, raw=raw)
        k = key(cls)
        store = measured.merge(store, k, measured.combine(store.get(k.key()),
                                                          row))
        measured.save(store, Path(store_path))  # each cell as it is timed
        nq1, nmed, nq3 = np.percentile(near[0], [25, 50, 75])
        print(f"{tag}: wide median {med / 1e6:.3f} ms, IQR "
              f"{(q3 - q1) / 1e6:.3f} ms; near median {nmed / 1e6:.3f} ms, IQR "
              f"{(nq3 - nq1) / 1e6:.3f} ms; {REPEATS} timed (+{WARMUP} "
              f"warm-up) each; card {near[2][0]}->{near[2][1]}% / "
              f"{card[0]}->{card[1]}%; discarded {discarded[0]}; F "
              f"bit-repeatable near {near[3]} wide {rep}", flush=True)
    print(f"\n{backend} {arch.name} ({gpu}), compiled `{command}`")


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_f_wallclock(request, run_device_binary, tmp_path) -> None:
    """The F rows' clock on the RTX 3080 (opt-in, alone)."""
    _f_wallclock("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_f_wallclock(request, run_device_binary, tmp_path) -> None:
    """The F rows' clock on the RX 9060 XT (opt-in, alone)."""
    _f_wallclock("hip", request, run_device_binary, tmp_path)


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


def test_the_wallclock_sweep_selects_its_classes_by_total_L() -> None:
    # #290: unset, #288's five; "3 4 5 6 7 8", the deferred sixteen
    assert _wallclock_classes({}) == DEFAULT
    deferred = _wallclock_classes({WALLCLOCK_F_L: "3 4 5 6 7 8"})
    assert len(deferred) == 16 and set(deferred).isdisjoint(DEFAULT)
    assert deferred[-1] == (2, 2, 2, 2)
    with pytest.raises(AssertionError, match="selects no class"):
        _wallclock_classes({WALLCLOCK_F_L: "9"})


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


def test_the_timing_inputs_stay_in_the_witnessed_boys_range() -> None:
    # #290: every class of the ladder, the deferred sixteen included -- those
    # above L = FAST_L_MAX (tiers.py) in the `slow` twin below
    _timing_inputs(c for c in CLASSES if is_fast(sum(c)))


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_the_timing_inputs_stay_in_the_witnessed_boys_range_above_l4(
        ) -> None:
    _timing_inputs(c for c in CLASSES if not is_fast(sum(c)))


def _timing_inputs(classes) -> None:
    for cls in classes:
        for side in (NEAR_SIDE, WIDE_SIDE):
            rec, basis = synthetic_launch(cls, side=side, seed=288_000)
            assert max_boys_T(basis, side) <= BOYS_T_WITNESSED
            assert rec.args["n_bra"] == TIMING_N_BRA
            assert rec.args["n_ket"] == TIMING_N_KET
            assert rec.args["diag"] == 0 and rec.args["tau"] == 0.0
            if cls in DEFAULT:  # #288's input, unchanged by #290
                assert len(basis.shells) == 104
            assert rec.args["k_ab"] == rec.args["k_cd"] == 9
