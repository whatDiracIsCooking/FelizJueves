"""The atomic-free F write-back's tail (#358), MEASURED -- not a test.

Moved out of test/test_f_lever_scatter_device.py (its part 4), because the
test suite measures correctness only and this is a clock: what is left
there checks the scatter lever on the card; this times it.

THE WRITE-BACK TAIL, MEASURED BEFORE THE VERDICT (the issue's order; ALONE,
behind the gate): per class with L <= 4, one synthetic off-diagonal launch
(`fock_driver.synthetic_launch`, 2048 x 128 contracted quartets, the F
rows' wide input) timed under four programs -- the literal kernel,
block-fab's (#351, H7), the scatter lever's F kernel ALONE (`f_only`: W
zeroed and written, no scatter) and the whole scatter path -- and the
fractions printed: block-fab's atomic tail `(t_H7 - t_Fonly) / t_H7` (what
plain stores save over its atomics, the zeroing of W charged to the
stores), the scatter kernel's share of the path `(t_path - t_Fonly) /
t_path`, and the net `t_path / t_H7 - 1`. Printed only, never a store row
(#362 is the ledger), no threshold. Every take waits for a quiet card
(`gate.wait_quiet`) and is discarded when the card was busy after it, up to
`STUDY_RETAKES` retakes. The path's two timed runs are also checked
bit-identical, the lever's determinism on this input; the correctness
tier's own witness of it is test_f_lever_scatter_device.py's per-molecule
part (H2O/STO-3G, every run compared bit for bit).

Run as (each backend; `FJ_WALLCLOCK_F_L` narrows the classes by total L,
unset all of L = 0..4):

    [FJ_WALLCLOCK_F_L="2 3 4"] devtools/measure.sh scatter-tail \\
        --backend cuda|hip

The harness (`measure/conftest.py`) provides test/conftest.py's fixtures
and puts `test/` on the path, so the scatter device file's helpers and
`fock_driver` import as they do there.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import test_f_lever_scatter_device as tsd
import test_fock_ondevice as tfo
from conftest import check_vram
from fock_driver import (
    FRec,
    fock_program,
    fock_scatter_stage,
    fock_stage,
    max_boys_T,
    reps,
    run_on_card,
    scatter_map_np,
    synthetic_launch,
)
from measure_kernel_device import GPU_BUSY_MAX, REPEATS, STUDY_RETAKES, WARMUP
from plan_driver import compile_checked
from test_kernel_device import BACKEND

from felizjueves.eri.fock.levers import block_fab
from felizjueves.performance import gate


def _tail_classes(env=os.environ) -> tuple:
    text = env.get(tfo.WALLCLOCK_F_L, "").strip()
    Ls = {int(x) for x in text.split()} if text else {0, 1, 2, 3, 4}
    got = tuple(c for c in tsd.L4 if sum(c) in Ls)
    assert got, f"{tfo.WALLCLOCK_F_L}={text!r} selects no class with L <= 4"
    return got


def _timed(backend: str, binary: Path, data: np.ndarray, n_ao: int,
           label: str, run_device_binary, tmp_path: Path):
    work = tmp_path / "timing"
    work.mkdir(exist_ok=True)
    src = work / "timing.in"
    data.tofile(src)
    check_vram(binary, (str(src), str(work / "timing.out")), backend)
    for take in range(1 + STUDY_RETAKES):
        try:
            gate.wait_quiet(backend, say=lambda line: print(line, flush=True))
        except gate.Contended as e:
            print(f"  DISCARDED {label}: {e} (take {take + 1})", flush=True)
            continue
        Fs, out = run_on_card(binary, data, n_ao, work, run_device_binary,
                              backend=backend, timing=(WARMUP, REPEATS),
                              tag="timing")
        after = gate.after_reading(backend)
        if after is not None and after <= GPU_BUSY_MAX:
            break
        print(f"  DISCARDED {label}: card {after}% after (take {take + 1})",
              flush=True)
    else:
        pytest.fail(f"{label}: contended through {STUDY_RETAKES + 1} takes")
    wall, events = reps(out)
    assert len(events) == REPEATS and np.all(np.isfinite(Fs)), label
    return float(np.median(events)), float(np.median(wall)), Fs


def _tail(backend: str, request, run_device_binary, tmp_path: Path) -> None:
    tc = tfo._toolchain(backend, request)
    for cls in _tail_classes():
        rec, basis = synthetic_launch(cls, side=tfo.WIDE_SIDE,
                                      seed=358_000 + sum(cls))
        assert max_boys_T(basis, tfo.WIDE_SIDE) <= tfo.BOYS_T_WITNESSED
        work = tmp_path / "".join(map(str, cls))
        arms = {}
        for arm, levs in (("literal", ()), ("block-fab", (block_fab.NAME,))):
            k = tsd._named(levs, cls)
            r = FRec(k, rec.grid, rec.threads, rec.args)
            prog = fock_program([k], backend, work / arm, name=f"prog_{arm}")
            binary = compile_checked(tc, prog, work / arm / "bin")
            arms[arm] = (binary, fock_stage([r], [k]))
        (r,) = tsd.as_lever([rec])
        data = fock_scatter_stage([r], [r.kernel], mapper=scatter_map_np)
        for arm, f_only in (("f-only", True), ("path", False)):
            binary = tsd._build(backend, tc, [r.kernel], work / arm, arm,
                                f_only=f_only)
            arms[arm] = (binary, data)
        t, bitwise = {}, None
        for arm, (binary, staged) in arms.items():
            ev, wall, Fs = _timed(backend, binary, staged, basis.n_ao,
                                  f"{cls} {arm}", run_device_binary, work)
            t[arm] = ev
            if arm == "path":
                bitwise = bool(np.array_equal(Fs[0], Fs[1]))
        print(f"\n{backend} {BACKEND[backend][1].name} {cls} tail, 2048 x 128 "
              f"wide, median events ms: " + ", ".join(
                  f"{a} {v / 1e6:.3f}" for a, v in t.items())
              + f"; block-fab's atomic tail (t_H7 - t_Fonly) / t_H7 "
              f"{(t['block-fab'] - t['f-only']) / t['block-fab']:+.4f}; "
              f"scatter share {(t['path'] - t['f-only']) / t['path']:.4f}; "
              f"net vs H7 {t['path'] / t['block-fab'] - 1:+.4f}; "
              f"path bit-identical runs {bitwise}", flush=True)
        assert bitwise, f"{cls}: the scatter path is not deterministic"


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_f_lever_scatter_tail(request, run_device_binary,
                                   tmp_path) -> None:
    """The write-back tail on the RTX 3080 (alone)."""
    _tail("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_f_lever_scatter_tail(request, run_device_binary,
                                  tmp_path) -> None:
    """The write-back tail on the RX 9060 XT (alone)."""
    _tail("hip", request, run_device_binary, tmp_path)


def test_the_tail_classes_are_l_at_most_4_and_refuse_an_empty_pick() -> None:
    assert _tail_classes({}) == tsd.L4
    assert all(sum(c) <= 4 for c in _tail_classes({tfo.WALLCLOCK_F_L: "3 4"}))
    with pytest.raises(AssertionError, match="selects no class"):
        _tail_classes({tfo.WALLCLOCK_F_L: "5"})
