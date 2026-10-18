"""The fused Fock kernel on the clock (#288, #290): the F rows' wall-clock,
on the RTX 3080 (`sm_86`) and the RX 9060 XT (`gfx1200`).

Moved from test/test_fock_ondevice.py, whose part 1 (per-class parity
against the host) checks every binary this sweep times -- `_part1` is
imported, not copied -- so the timed binary is the checked binary. Nothing
here is a correctness claim, and the two no-card cases below hold the
sweep's own class selector and its inputs, not a kernel.

THE MEASURED ROWS (milestone 12; #278's F cell). One row per (class, arch,
toolchain): #288 recorded the five classes the default tier runs, #290 (the
deferred tier) the sixteen above them. The compile side -- registers,
spill, static shared bytes -- is `python -m felizjueves.measured
--remeasure --target f --L ...` (`measured.py`, THE F CELLS). The clock
is this file's opt-in sweep, `test_<backend>_f_wallclock`: it skips unless
FJ_WALLCLOCK_STORE names the store and FJ_WALLCLOCK_F=1, and must run
ALONE (a contended card is a different experiment). FJ_WALLCLOCK_F_L
selects its classes by total L (unset: L <= 2, #288's five):

    FJ_WALLCLOCK_F_L="3 4 5 6 7 8" devtools/measure.sh wallclock-f \\
        --backend cuda

(which sets FJ_WALLCLOCK_F=1 and, unless already set, FJ_WALLCLOCK_STORE
to the tracked data/measured/kernels.json).

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
"""

from __future__ import annotations

import datetime
import hashlib
import os
import shlex
from pathlib import Path

import numpy as np
import pytest
from fock_driver import (
    TIMING_N_BRA,
    TIMING_N_KET,
    fock_stage,
    max_boys_T,
    reps,
    run_on_card,
    synthetic_launch,
)
from measure_kernel_device import (
    GPU_BUSY_MAX,
    REPEATS,
    STUDY_RETAKES,
    WALLCLOCK_RESUME,
    WALLCLOCK_STORE,
    WARMUP,
    _cpu_busy,
    _cpu_list,
    _stats,
)
from test_fock_ondevice import (
    BOYS_T_WITNESSED,
    DEFAULT,
    NEAR_SIDE,
    WALLCLOCK_F_L,
    WIDE_SIDE,
    _class_name,
    _class_recs,
    _part1,
)
from test_kernel_device import (
    BACKEND,
    _toolchain,
    check_vram,
)
from tiers import is_fast

from felizjueves import gate, measured
from felizjueves import kernel_f as KF
from felizjueves.device import Arch, MeasuredKey
from felizjueves.hgp.vrr import CLASSES
from felizjueves.ir import Kernel

WALLCLOCK_F = "FJ_WALLCLOCK_F"


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


def test_the_wallclock_sweep_selects_its_classes_by_total_L() -> None:
    # #290: unset, #288's five; "3 4 5 6 7 8", the deferred sixteen
    assert _wallclock_classes({}) == DEFAULT
    deferred = _wallclock_classes({WALLCLOCK_F_L: "3 4 5 6 7 8"})
    assert len(deferred) == 16 and set(deferred).isdisjoint(DEFAULT)
    assert deferred[-1] == (2, 2, 2, 2)
    with pytest.raises(AssertionError, match="selects no class"):
        _wallclock_classes({WALLCLOCK_F_L: "9"})


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
