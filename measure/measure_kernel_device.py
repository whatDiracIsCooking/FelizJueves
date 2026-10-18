"""The [p|q] and K kernels on the card, MEASURED: the wall-clock sweep over
the J subgrid (#53), its lever and asymmetric arms (#55, #113, #162, #163),
the roofline counter sweep (#112) and the K wall-clock beside each cell's J
twin (#205).

These moved here from test/test_kernel_device.py, whose correctness checks
they reuse unchanged: every binary timed or profiled here is built by
`_batched_source` / `_k_build` and passed through `_check_batched` /
`_k_check` first -- the TIMED binary is the CHECKED binary -- so this file
adds a clock, a profiler and a store write, never a second build. Nothing
here asserts that a kernel is right; test/ does that, and a correctness run
never times anything.

RUN THROUGH devtools/measure.sh, one sweep per invocation, ALONE on the box
(no xdist; the card's lock is the device fixture's, measure/conftest.py):

    devtools/measure.sh wallclock   --backend cuda   # FJ_WALLCLOCK_STORE
    devtools/measure.sh roofline    --backend cuda   # FJ_ROOFLINE_STORE
    devtools/measure.sh wallclock-k --backend cuda   # + FJ_WALLCLOCK_K=1

Each sweep skips unless its store variable names the store to write; the
script defaults it to the tracked file (devtools/config.sh). The section
comments below are the protocols, and the optional selectors
(FJ_WALLCLOCK_BATCHES, _LEVER, _ASYM, _RESUME, FJ_ROOFLINE_BATCHES,
FJ_WALLCLOCK_LS) pass through the environment unchanged.
"""
from __future__ import annotations

import datetime
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import check_vram
from test_kernel_device import (
    BACKEND,
    K_SUBGRID,
    SUBGRID,
    _batched_source,
    _check_batched,
    _Configured,
    _inputs,
    _k_build,
    _k_check,
    _k_inputs,
    _k_limits,
    _k_pack,
    _pack,
    _toolchain,
    k_shared_need,
)

from felizjueves import asym, gate, measured, measured_store
from felizjueves.device import Arch
from felizjueves.kernel import (
    k_spec,
)
from felizjueves.spec import Spec

# --- the wall-clock sweep -----------------------------------------------------
#
# OPT-IN, AND RUN ALONE. It skips unless FJ_WALLCLOCK_STORE names the store to
# write (normally data/measured/kernels.json), because inside the kernel tier
# it would share the box with xdist workers compiling other tests -- and a
# contended timing is garbage, not a noisy measurement. Run it by itself, with
# nothing else compile-bound on the box; the card's lock is the device
# fixture's (test/conftest.py, GPU_LOCK_PATH), so no `flock` wrapper is needed
# -- one is harmless, the fixture recognising an ancestor's hold as its own:
#
#   devtools/measure.sh wallclock --backend cuda
#
# (FJ_WALLCLOCK_STORE defaults there to data/measured/kernels.json.)
#
# FJ_WALLCLOCK_BATCHES (optional, e.g. "9,12") restricts the sweep to the
# SUBGRID cells of those n_batches -- every other row is left as it is. #60
# used it to time 9 and 12 in ONE session: its verdict compares the two
# against their IQR, and a cross-session comparison would mix the box's
# drift into the effect.
#
# FJ_WALLCLOCK_LEVER (optional, e.g. "recycle") sweeps the lever's cells,
# `measured.ab_cells(lever)` (the subgrid, plus L = 11 for "no-sieve", #162),
# instead: each cell's literal arm is compiled,
# checked and timed BESIDE it as a same-session control (near-T control then
# lever, wide-T lever then control) and lands in the LEVER row's raw as
# `wallclock:control:*`; the literal rows are not rewritten. measured.py's
# "A LEVER A/B" is the protocol's rationale and `measured.lever_ab` the
# verdict (#55). The compile rows of the lever cells must exist first:
#   python -m felizjueves.measured --remeasure --subgrid --lever recycle
# A PLAN arm's sweep (#162: "no-sieve", measured_store's THE ARMS) discards a
# phase the card was taken during and retakes it, as #113's study does
# (STUDY_RETAKES), counting the discards in the lever row's raw
# `wallclock:gate_discarded_phases`; #55's recycle sweep is recorded as it was.
# A plan arm's gate that stays shut past its deadline is a discarded phase
# too (counted, retaken), each cell is written as soon as it is timed, and
# FJ_WALLCLOCK_RESUME=1 skips the cells whose lever row already carries its
# same-session control -- so a session the gate stopped resumes, and every
# A/B pair is still timed back to back. #163's "tuned" is a plan arm swept
# the same way over `ab_cells("tuned")`; its NOT_MEASURABLE cells (no
# recorded choice) are printed with their reason, never timed as literal.
#
# FJ_WALLCLOCK_ASYM=1 (optional) sweeps #113's asymmetric study cells,
# `asym.ASYM_CELLS`, instead: each cell's two nearest legal neighbours
# (literal SUBGRID Specs) are compiled, checked and timed BESIDE it as
# same-session controls (near-T lo, asym, hi; wide-T hi, asym, lo) and land
# in the ASYMMETRIC row's raw as `wallclock:control:b<n>:*`; no literal row
# is re-timed or rewritten. `asym`'s docstring is the rule, declared before
# any asymmetric row existed, and `asym.asym_vs_neighbours` the verdict. A
# study phase the card was taken during is retaken (STUDY_RETAKES). The
# compile rows must exist first:
#   python -m felizjueves.measured --remeasure --asym
#
# THE PROTOCOL, in order, all inside one test so the phases cannot overlap:
#
# 1. CPUSET from devtools/config.sh, if set, pins this process (and so the
#    harness) with sched_setaffinity; CPUS is a container quota and cannot be
#    applied to a bare process -- a set CPUS is reported, not silently
#    honoured. Both are empty by default.
# 2. COMPILE every SUBGRID cell's harness, one after another, with config.sh's
#    NVCC_FLAGS / HIPCC_FLAGS (the flag set the measured rows' registers came
#    from). No compile runs after this point.
# 3. CHECK every compiled binary against the oracle on CASES (`_check_batched`)
#    -- the TIMED binary is the CHECKED binary.
# 4. TIME each cell: wait for the box to be quiet (CPU busy at most BUSY_MAX
#    over a 1 s sample, for up to QUIET_WAIT_S; otherwise the test FAILS --
#    the run is discarded, not averaged in) AND for the card to be (its own
#    SMI's utilization at most GPU_BUSY_MAX, for up to GPU_QUIET_WAIT_S --
#    added by #55, after a process outside the container held the RTX 3080;
#    #93 records it per input: raw `wallclock:gpu_util_before` is the settled
#    reading the gate let through, `wallclock:gpu_util_after` one taken
#    GPU_SETTLE_S after the last timed sequence, `wallclock:near:*` and
#    `wallclock:control:*` likewise -- `null` where the SMI said nothing
#    parseable, which means UNKNOWN and never idle),
#    then WARMUP untimed sequences and
#    REPEATS timed ones, on each of two inputs -- no screening (tau = 0) so
#    every cell does all its work, n_bra x n_ket = TIMING_N_BRA x
#    TIMING_N_KET, and the same input for every n_batches of one L:
#      WIDE (CASES' wide regime, T up to ~430): THE ROW's number. The median
#        and IQR of the host wall-clock are runtime_ns / spread_ns.
#      NEAR (T < ~2): `wallclock:near:*` in raw. Each batch recomputes its
#        fundamentals (kernel.py), so the Boys series length -- long at
#        large T -- weighs on the batch-count ranking; timing both regimes
#        shows whether the ranking depends on it, instead of hiding it.
# 5. WRITE one row per cell, `measured_store.combine`d into the existing
#    compile row (same text, one row: measured_store.py's docstring, ONE ROW
#    CARRIES BOTH SIDES -- the store left measured.py in #94).

WALLCLOCK_STORE = "FJ_WALLCLOCK_STORE"
WALLCLOCK_BATCHES = "FJ_WALLCLOCK_BATCHES"
WALLCLOCK_LEVER = "FJ_WALLCLOCK_LEVER"
WALLCLOCK_ASYM = "FJ_WALLCLOCK_ASYM"
WALLCLOCK_RESUME = "FJ_WALLCLOCK_RESUME"
# #113 only: a phase whose after-reading is non-idle or unknown was shared
# with something outside this process (the first study session saw 100%
# after-readings with bimodal times), so it is DISCARDED and retaken, up to
# this many times, and the row counts its discards in raw
# `wallclock:gate_discarded_phases`. The literal and lever sweeps record the
# after-reading instead, as they always have.
STUDY_RETAKES = 5
WARMUP = 5
REPEATS = 31
TIMING_N_BRA = 2048  # one warp each: several waves on either card
TIMING_N_KET = 128   # four kets per lane at wavefront 32
# THE TIMING GATE, on top of the correctness gate the device fixtures already
# hold (test/conftest.py: the card's lock and the free-VRAM probe), is
# `felizjueves.gate` -- lifted there by #109 so the ceiling probes (and #112's
# counter sweep) read the same code; its docstring has the readings and why
# the after-reading is taken the way it is. Utilisation is a reason to WAIT
# here and nowhere else: #55's timings came out bimodal while a process
# outside the container held the RTX 3080 at 100%, and a timing on a busy
# card is garbage where a correct answer is still correct. The names below
# are the gate's, re-exported so this file's protocol reads as it did.
BUSY_MAX = gate.BUSY_MAX
QUIET_WAIT_S = gate.QUIET_WAIT_S
GPU_BUSY_MAX = gate.GPU_BUSY_MAX
GPU_QUIET_WAIT_S = gate.GPU_QUIET_WAIT_S
GPU_SETTLE_S = gate.GPU_SETTLE_S
_cpu_busy = gate.cpu_busy
_card_util = gate.card_util


def test_the_wallclock_path_reads_the_same_gate() -> None:
    # One helper, not a copy (moved from test/test_ceilings.py with the
    # sweep): this file's names ARE the gate's. No card, no compiler.
    assert GPU_BUSY_MAX is gate.GPU_BUSY_MAX
    assert GPU_SETTLE_S is gate.GPU_SETTLE_S
    assert _card_util is gate.card_util and _cpu_busy is gate.cpu_busy
    assert (GPU_BUSY_MAX, GPU_QUIET_WAIT_S, GPU_SETTLE_S, BUSY_MAX,
            QUIET_WAIT_S) == (30, 900.0, 1.0, 0.10, 120.0)


def _quiet(backend: str | None = None, seen: list[int] | None = None) -> float:
    """Block until the box is quiet -- the CPU and, given `backend`, the card
    (`gate.wait_quiet`) -- and return the CPU busy fraction it settled at; the
    card's settled utilization is appended to `seen`. Past the gate's
    deadline the test FAILS: the timing is discarded, not averaged in."""
    try:
        busy, util = gate.wait_quiet(
            backend, say=lambda line: print(line, flush=True))
    except gate.Contended as e:
        pytest.fail(f"{e} -- timing DISCARDED, not averaged in")
    if backend is not None and seen is not None:
        seen.append(util)
    return busy


def _cpu_list(spec: str) -> set[int]:
    """`0-11,16` -> {0..11, 16}."""
    out: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def _parse_reps(out: str, n_batches: int):
    wall, events, batches = [], [], []
    for line in out.splitlines():
        f = line.split()
        if not f or f[0] != "rep":
            continue
        assert (f[2], f[4], f[6]) == ("wall_ns", "events_ns", "batch_ns"), line
        wall.append(int(f[3]))
        events.append(float(f[5]))
        batches.append([float(x) for x in f[7:]])
        assert len(batches[-1]) == n_batches, line
    return wall, events, batches


def _wallclock(backend: str, request, compile_device_source,
               run_device_binary, tmp_path: Path) -> None:
    store_path = os.environ.get(WALLCLOCK_STORE)
    if not store_path:
        pytest.skip(f"wall-clock is opt-in and must run alone, uncontended: set "
                    f"{WALLCLOCK_STORE} (see the section comment)")
    base = _toolchain(backend, request)
    tc = _Configured(base.name, base.path, base.arch, base.ext)
    arch: Arch = BACKEND[backend][1]
    b = measured.BACKENDS[backend]
    toolchain, gpu = b.toolchain(b.compiler()), b.gpu()
    store = measured.load(Path(store_path))

    # 1. CPU bounds
    cpuset, cpus = measured.config_value("CPUSET"), measured.config_value("CPUS")
    if cpuset:
        os.sched_setaffinity(0, _cpu_list(cpuset))
    print(f"\nCPUSET={cpuset or '(empty)'} CPUS={cpus or '(empty)'}"
          + (" -- CPUS is a container quota, not applied here" if cpus else ""))

    # 2. compile everything first
    only = os.environ.get(WALLCLOCK_BATCHES, "").strip()
    wanted = {int(b) for b in only.split(",")} if only else None
    lever = os.environ.get(WALLCLOCK_LEVER, "").strip()
    study = bool(os.environ.get(WALLCLOCK_ASYM, "").strip())
    assert not (lever and study), f"{WALLCLOCK_LEVER} or {WALLCLOCK_ASYM}"
    grid = (asym.ASYM_CELLS if study
            else measured.ab_cells(lever) if lever else SUBGRID)
    # a phase shared with something else is discarded and retaken: #113's
    # study, and a plan arm's A/B (#162); #55's lowering sweep is as it was
    plan_arm = bool(lever) and measured_store.ARMS[lever] == measured_store.PLAN
    retake = study or plan_arm
    sweep = [s for s in grid if wanted is None or s.n_batches in wanted]
    assert sweep, f"{WALLCLOCK_BATCHES}={only!r} selects no cell"
    if plan_arm and os.environ.get(WALLCLOCK_RESUME, "").strip():
        # #162: a plan arm's sweep writes each cell as it is timed, so a
        # session the gate stopped is resumed, not redone: skip every cell
        # whose lever row already carries its same-session control
        def done(s: Spec) -> bool:
            r = store.get(measured.measured_key(s, arch, toolchain).key())
            return (r is not None and r.runtime_ns is not None
                    and "wallclock:control:median_ns" in r.raw
                    and "wallclock:gate_discarded_phases" in r.raw)
        skipped = [s for s in sweep if done(s)]
        sweep = [s for s in sweep if not done(s)]
        print(f"{WALLCLOCK_RESUME}: {len(skipped)} cells already timed, "
              f"skipped")
        if not sweep:
            pytest.skip(f"{WALLCLOCK_RESUME}: every cell already timed")
    for s, why in (measured_store.NOT_MEASURABLE.get(lever, {}).items()
                   if lever else ()):
        print(f"{s.key()}: NOT MEASURABLE -- {why}")
    print(f"{WALLCLOCK_BATCHES}={only or '(all)'} {WALLCLOCK_LEVER}="
          f"{lever or '(none)'} {WALLCLOCK_ASYM}={int(study)}: "
          f"{len(sweep)} cells")
    cells = []
    for spec in sweep:
        work = tmp_path / f"L{spec.total_am}-b{spec.n_batches}"
        work.mkdir()
        src, names, sha = _batched_source(backend, spec, work)
        binary = compile_device_source(tc, src).rename(work / "cell")
        # Same-session controls, as (raw prefix, Spec, binary): the literal
        # arm of a lever cell (measured_store, A LEVER A/B), or an asymmetric
        # cell's two nearest legal neighbours, lo then hi (asym, #113).
        arms = ([("wallclock:control:", replace(spec, levers=()))] if lever
                else [(asym.control_prefix(n), n) for n in spec.neighbours]
                if study else [])
        controls = []
        for prefix, cspec in arms:
            cwork = work / f"control-b{cspec.n_batches}"
            cwork.mkdir()
            csrc, _, _ = _batched_source(backend, cspec, cwork)
            controls.append((prefix, cspec, compile_device_source(
                tc, csrc).rename(cwork / "cell")))
        cells.append((spec, binary, names, sha, controls))
        print(f"compiled {spec.key()}" + "".join(
            f" and control {c[1].key()}" for c in controls), flush=True)
    command = " ".join([Path(tc.path).name, *tc.compile_flags()])

    # 3. the timed binary is the checked binary -- every arm of an A/B
    for spec, binary, _, _, controls in cells:
        _check_batched(backend, spec, binary, run_device_binary, tmp_path)
        for _, cspec, cbinary in controls:
            _check_batched(backend, cspec, cbinary, run_device_binary, tmp_path)

    # 4. time, one cell at a time
    lanes = arch.wavefront
    discarded = [0]  # the study's gate-discarded phases, per cell (below)

    def timed(spec: Spec, binary: Path, wide: bool):
        L = spec.total_am
        a = _inputs(L, wide=wide, tau_quantile=None, seed=53_000 + L,
                    n_bra=TIMING_N_BRA, n_ket=TIMING_N_KET)
        src, dst = tmp_path / "timing.in", tmp_path / "timing.out"
        _pack(L, a, lanes).tofile(src)
        args = (str(src), str(dst), str(WARMUP), str(REPEATS))
        check_vram(binary, args, backend)  # before the wait, off the clock
        for take in range(1 + (STUDY_RETAKES if retake else 0)):
            seen: list[int] = []
            if not plan_arm:
                busy = _quiet(backend, seen)
            else:
                # #162: a gate that stays shut past its deadline (the box or
                # the card busy before the phase) is a discarded phase too,
                # counted and retaken -- never a timing taken anyway
                try:
                    busy, util = gate.wait_quiet(
                        backend, say=lambda line: print(line, flush=True))
                except gate.Contended as e:
                    discarded[0] += 1
                    print(f"  DISCARDED {spec.key()} "
                          f"{'wide' if wide else 'near'}: {e} (take "
                          f"{take + 1})", flush=True)
                    continue
                seen.append(util)
            out = run_device_binary(binary, *args)
            # the gate's reading, and one after (gate.after_reading: one SMI
            # read discarded, GPU_SETTLE_S, then the reading -- our own run
            # excluded)
            card = (seen[-1], gate.after_reading(backend))
            if not retake or (card[1] is not None and card[1] <= GPU_BUSY_MAX):
                break
            # #113 / #162: the card was taken DURING the phase (after-reading
            # non-idle or unknown): the phase is discarded and retaken
            discarded[0] += 1
            print(f"  DISCARDED {spec.key()} {'wide' if wide else 'near'}: "
                  f"card util {card[0]} -> {card[1]}% (take {take + 1})",
                  flush=True)
        else:
            pytest.fail(f"{spec.key()}: the box or the card stayed contended "
                        f"through {STUDY_RETAKES + 1} takes -- this cell not "
                        "recorded")
        wall, events, batches = _parse_reps(out, spec.n_batches)
        assert len(wall) == REPEATS, out
        J = np.fromfile(dst, dtype=np.float64).reshape(2, TIMING_N_BRA, -1)
        assert np.all(np.isfinite(J)), f"{spec.key()}: non-finite J (timing input)"
        q1, med, q3 = (float(x) for x in np.percentile(wall, [25, 50, 75]))
        return wall, events, batches, J, (q1, med, q3), busy, card

    def gpu_fields(prefix: str, card: tuple[int, int | None]) -> dict:
        # measured.GPU_UTIL_FIELDS: the gate's settled reading before the
        # input's warm-up, and one GPU_SETTLE_S after its last timed sequence
        return {f"{prefix}gpu_util_before": card[0],
                f"{prefix}gpu_util_after": card[1]}

    table = []
    def control_stats(prefix: str, got) -> dict:
        w, _, _, _, stats, b_, g = got
        return {f"{prefix}q1_ns": round(stats[0]),
                f"{prefix}median_ns": round(stats[1]),
                f"{prefix}q3_ns": round(stats[2]),
                f"{prefix}min_ns": int(min(w)),
                f"{prefix}max_ns": int(max(w)),
                f"{prefix}cpu_busy_permille_before": round(1000 * b_),
                **gpu_fields(prefix, g)}

    for spec, binary, names, sha, controls in cells:
        L = spec.total_am
        load1 = os.getloadavg()[0]
        discarded[0] = 0
        ctrl: dict[str, int | None] = {}
        # Arm order, so no arm always runs first: near-T the first control
        # before the cell and the rest after it; wide-T the reverse (a lever:
        # near control-lever, wide lever-control; asym: near lo-asym-hi, wide
        # hi-asym-lo).
        for prefix, cspec, cbinary in controls[:1]:
            ctrl |= control_stats(prefix + "near:", timed(cspec, cbinary, False))
        # the secondary input first: near-T (T < ~2, short Boys series)
        n_wall, _, _, _, (n_q1, n_med, n_q3), n_busy, n_gpu = timed(
            spec, binary, False)
        for prefix, cspec, cbinary in controls[1:]:
            ctrl |= control_stats(prefix + "near:", timed(cspec, cbinary, False))
        for prefix, cspec, cbinary in reversed(controls[1:]):
            ctrl |= control_stats(prefix, timed(cspec, cbinary, True))
        # the row's number: wide-T (T up to ~430, the Boys series long)
        wall, events, batches, J, (q1, med, q3), busy, card = timed(
            spec, binary, True)
        for prefix, cspec, cbinary in controls[:1]:
            ctrl |= control_stats(prefix, timed(cspec, cbinary, True))
        busy_after = _cpu_busy(0.5)
        raw = {
            "wallclock:near:median_ns": round(n_med),
            "wallclock:near:q1_ns": round(n_q1),
            "wallclock:near:q3_ns": round(n_q3),
            "wallclock:near:min_ns": int(min(n_wall)),
            "wallclock:near:max_ns": int(max(n_wall)),
            "wallclock:near:cpu_busy_permille_before": round(1000 * n_busy),
            **gpu_fields("wallclock:near:", n_gpu),
            "wallclock:warmup": WARMUP,
            "wallclock:n_bra": TIMING_N_BRA,
            "wallclock:n_ket": TIMING_N_KET,
            "wallclock:lanes": lanes,
            "wallclock:min_ns": int(min(wall)),
            "wallclock:q1_ns": round(q1),
            "wallclock:q3_ns": round(q3),
            "wallclock:max_ns": int(max(wall)),
            "wallclock:events_median_ns": round(float(np.median(events))),
            "wallclock:cpu_busy_permille_before": round(1000 * busy),
            "wallclock:cpu_busy_permille_after": round(1000 * busy_after),
            "wallclock:loadavg1_milli": round(1000 * load1),
            **gpu_fields("wallclock:", card),
            "wallclock:J_bitwise_repeatable": int(np.array_equal(J[0], J[1])),
            **ctrl,
        }
        if retake:
            raw["wallclock:gate_discarded_phases"] = discarded[0]
        per_batch = np.median(np.array(batches), axis=0)
        for name, t in zip(names, per_batch, strict=True):
            raw[f"wallclock:{name}:events_median_ns"] = round(float(t))
        row = measured.Measured(
            runtime_ns=round(med), repeats=REPEATS, spread_ns=round(q3 - q1),
            toolchain=toolchain, arch=arch.name, host=measured.host(), gpu=gpu,
            date=datetime.date.today().isoformat(), kernel=",".join(names),
            source_sha256=sha, raw=raw)
        key = measured.measured_key(spec, arch, toolchain)
        old = store.get(key.key())
        assert old is None or old.source_sha256 == sha, (
            f"{key.key()}: the store's compile row is of other text -- "
            "remeasure the compile side first")
        store = measured.merge(store, key, measured.combine(old, row))
        if plan_arm:  # #162: each cell written as it is timed (resumable)
            measured.save(store, Path(store_path))
        table.append((L, spec.n_batches, med, q3 - q1, n_med, n_q3 - n_q1))
        print(f"{key.key()}: wide median {med / 1e6:.3f} ms, IQR "
              f"{(q3 - q1) / 1e6:.3f} ms, range [{min(wall) / 1e6:.3f}, "
              f"{max(wall) / 1e6:.3f}] ms; near median {n_med / 1e6:.3f} ms, "
              f"IQR {(n_q3 - n_q1) / 1e6:.3f} ms; {REPEATS} timed (+{WARMUP} "
              f"warm-up) each; CPU busy {n_busy:.1%}/{busy:.1%} -> "
              f"{busy_after:.1%}, load1 {load1:.2f}; card util {n_gpu[0]}->"
              f"{n_gpu[1]}% / {card[0]}->{card[1]}%; J bit-repeatable "
              f"{raw['wallclock:J_bitwise_repeatable']}", flush=True)
        for prefix, cspec, _ in controls if study else ():
            print(f"  control {cspec.key()} (same session): wide median "
                  f"{ctrl[prefix + 'median_ns'] / 1e6:.3f} ms, near median "
                  f"{ctrl[prefix + 'near:median_ns'] / 1e6:.3f} ms", flush=True)
        if lever:
            print(f"  control (literal, same session): wide median "
                  f"{ctrl['wallclock:control:median_ns'] / 1e6:.3f} ms, near "
                  f"median {ctrl['wallclock:control:near:median_ns'] / 1e6:.3f}"
                  " ms", flush=True)

    # 5. one write, after every cell timed
    measured.save(store, Path(store_path))
    if lever:
        print(f"\n#55 A/B, {lever} against its same-session literal control:")
        for spec, *_ in cells:
            c = measured.lever_ab(store, arch.name, spec)
            print(f"  L={c.L} b{c.n_batches}: wide {c.wide_sign}, near "
                  f"{c.near_sign} -> {c.verdict.upper()} (spill {c.spill[0]} "
                  f"-> {c.spill[1]})")
        if lever == measured_store.NO_SIEVE:  # #159's declared verdict, this arch
            print(f"  {arch.name}: the sieve "
                  f"{measured_store.sieve_verdict(store, arch.name).upper()}")
        if lever == measured_store.TUNED:  # #163's declared verdict, this arch
            try:
                print(f"  {arch.name}: the search "
                      f"{measured_store.tuned_verdict(store, arch.name).upper()}")
            except KeyError as e:  # a resumed sweep, not every cell timed yet
                print(f"  {arch.name}: the search cannot be judged yet ({e})")
    if study:
        print("\n#113, the asymmetric split against its same-session "
              "neighbours:")
        judged = [spec for spec, *_ in cells]
        for spec in judged:
            c = asym.asym_vs_neighbours(store, arch.name, spec)
            print(f"  L={c.L} b{c.n_batches} {c.ks}: wide {c.wide_sign} vs "
                  f"b{c.wide_best}, near {c.near_sign} vs b{c.near_best} -> "
                  f"{c.verdict.upper()} (spill {c.spill}, regs {c.registers})")
        print(f"  {arch.name}: "
              f"{asym.arch_verdict(store, arch.name, judged).upper()}")
    print(f"\n{backend} {arch.name} ({gpu}), compiled `{command}`")
    for col, label in ((2, "wide (the row)"), (4, "near (raw)")):
        print(f" {label}: median+-IQR, ms")
        for L in sorted({t[0] for t in table}):
            at_L = [t for t in table if t[0] == L]
            best = min(at_L, key=lambda t: t[col])
            print(f"  L={L}: " + "  ".join(
                f"b{t[1]} {t[col] / 1e6:.3f}+-{t[col + 1] / 1e6:.3f}"
                for t in at_L) + f"   -> lowest median at n_batches={best[1]}")


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_wallclock_over_the_subgrid(request, compile_device_source,
                                         run_device_binary, tmp_path) -> None:
    """Wall-clock of every SUBGRID cell's launch sequence on the RTX 3080."""
    _wallclock("cuda", request, compile_device_source, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_wallclock_over_the_subgrid(request, compile_device_source,
                                        run_device_binary, tmp_path) -> None:
    """Wall-clock of every SUBGRID cell's launch sequence on the RX 9060 XT."""
    _wallclock("hip", request, compile_device_source, run_device_binary, tmp_path)


# --- the roofline sweep (#112) ------------------------------------------------
#
# OPT-IN, AND RUN ALONE, like the wall-clock sweep and for the same reason: a
# counter row taken beside the tier's compiles, or on a card something else
# holds, is not a noisy row but a different experiment. It skips unless
# FJ_ROOFLINE_STORE names the counter store to write (normally
# data/measured/roofline.json):
#
#   devtools/measure.sh roofline --backend cuda
#
# FJ_ROOFLINE_BATCHES (optional, e.g. "9,12") restricts it to the SUBGRID
# cells of those n_batches; every other row is left as it is.
#
# THE PROTOCOL, in order, inside one test:
# 1. COMPILE every SUBGRID cell's batched harness (`_batched_source`, the
#    wall-clock sweep's, with config.sh's NVCC_FLAGS) and refuse a cell whose
#    emitted text is not the text the measured store's compile row hashed.
# 2. CHECK every binary against the oracle (`_check_batched`, J_REL): the
#    profiled binary is the checked binary, and the same one the wall-clock
#    sweep times -- a SEPARATE run of it (counter collection serialises and
#    replays each kernel, so it may change occupancy; its duration is
#    counter_ns, never runtime_ns, and no wall-clock cell is touched).
# 3. PROFILE each cell, one launch sequence (no warm-up, one rep: each batch
#    kernel launched exactly once) on the wide-T timing input -- kernels.json's
#    row input, TIMING_N_BRA x TIMING_N_KET, tau = 0 -- under
#    `measured.profile` (ncu), behind the card gate (`felizjueves.gate`, the
#    shared helper): wait for a quiet card and CPU (a deadline FAILS the test,
#    nothing written), profile, then the after-reading. A phase whose
#    after-reading is non-idle or unknown was shared with something outside
#    this process (#113's lesson on this box): it is DISCARDED and retaken, up
#    to ROOFLINE_RETAKES times, else the test fails and nothing is written.
#    The cell's discard count lands in every one of its rows.
# 4. WRITE one row per batch kernel (`measured.counter_rows`, #110's shape)
#    with the card's state (measured_store.ROOFLINE_STATE_FIELDS) and the
#    input in raw, once, after every cell -- then print each kernel's and
#    each cell's verdict against the card's measured ridge.
#
# gfx1200 is measured_store.UNMEASURED: its test skips with the recorded,
# FINAL reason (#140: rocprofv3 loads, but the card has no fp64 FLOP counter,
# so no roofline row -- not a zero).

ROOFLINE_STORE = "FJ_ROOFLINE_STORE"
ROOFLINE_BATCHES = "FJ_ROOFLINE_BATCHES"
ROOFLINE_RETAKES = 5


def _roofline(backend: str, request, compile_device_source,
              run_device_binary, tmp_path: Path) -> None:
    store_path = os.environ.get(ROOFLINE_STORE)
    if not store_path:
        pytest.skip(f"the counter sweep is opt-in and must run alone, "
                    f"uncontended: set {ROOFLINE_STORE} (see the section "
                    "comment)")
    arch: Arch = BACKEND[backend][1]
    if arch.name in measured.UNMEASURED:
        pytest.skip(measured.UNMEASURED[arch.name])
    base = _toolchain(backend, request)
    tc = _Configured(base.name, base.path, base.arch, base.ext)
    b = measured.BACKENDS[backend]
    toolchain, gpu = b.toolchain(b.compiler()), b.gpu()
    compile_rows = measured.load()
    store = measured.load_counters(Path(store_path))

    # 1. compile everything first, against the measured store's text
    only = os.environ.get(ROOFLINE_BATCHES, "").strip()
    wanted = {int(n) for n in only.split(",")} if only else None
    sweep = [s for s in SUBGRID if wanted is None or s.n_batches in wanted]
    assert sweep, f"{ROOFLINE_BATCHES}={only!r} selects no cell"
    cells = []
    for spec in sweep:
        work = tmp_path / f"L{spec.total_am}-b{spec.n_batches}"
        work.mkdir()
        src, names, sha = _batched_source(backend, spec, work)
        units = b.units(spec)
        assert [u.name for u in units] == names
        assert measured.cell_sha256(units) == sha
        key = measured.MeasuredKey(spec, arch, toolchain)
        old = compile_rows.get(key.key())
        assert old is not None and old.source_sha256 == sha, (
            f"{key.key()}: the measured store's compile row is of other text "
            "(or absent) -- remeasure the compile side first")
        binary = compile_device_source(tc, src).rename(work / "cell")
        cells.append((spec, binary, units, work))
        print(f"compiled {spec.key()}", flush=True)

    # 2. the profiled binary is the checked binary
    for spec, binary, _, _ in cells:
        _check_batched(backend, spec, binary, run_device_binary, tmp_path)

    # 3. profile, one cell at a time, behind the gate
    lanes = arch.wavefront
    new: dict[str, measured.Measured] = {}
    discarded_total = 0
    t_sweep = datetime.datetime.now()
    for spec, binary, units, work in cells:
        L = spec.total_am
        a = _inputs(L, wide=True, tau_quantile=None, seed=53_000 + L,
                    n_bra=TIMING_N_BRA, n_ket=TIMING_N_KET)
        src, dst = work / "counter.in", work / "counter.out"
        _pack(L, a, lanes).tofile(src)
        args = (str(src), str(dst), "0", "1")
        check_vram(binary, args, backend)  # before the wait, off the gate
        regex = "^(" + "|".join(u.name for u in units) + ")$"
        discarded = 0
        for take in range(1 + ROOFLINE_RETAKES):
            try:
                busy, before = gate.wait_quiet(
                    backend, say=lambda line: print(line, flush=True))
            except gate.Contended as e:
                pytest.fail(f"{e} -- counter sweep DISCARDED, nothing written")
            outdir = work / f"take{take}"
            outdir.mkdir()
            try:
                launches, tool, command = measured.profile(
                    backend, [str(binary), *args], kernel_regex=regex,
                    workdir=outdir)
            except measured.CouldNotMeasure as e:
                pytest.skip(f"{e} -- nothing written")
            after = gate.after_reading(backend)
            if gate.is_idle(after):
                break
            discarded += 1
            print(f"  DISCARDED {spec.key()}: card util {before} -> {after}% "
                  f"(take {take + 1})", flush=True)
        else:
            pytest.fail(f"{spec.key()}: the card stayed contended through "
                        f"{ROOFLINE_RETAKES + 1} takes -- nothing written")
        discarded_total += discarded
        command = command.replace(str(src), "<counter.in>").replace(
            str(dst), "<counter.out>")
        rows = measured.counter_rows(b, spec, units, launches,
                                     toolchain=toolchain, gpu=gpu, tool=tool,
                                     command=command)
        state = {"roofline:gpu_util_before": before,
                 "roofline:gpu_util_after": after,
                 "roofline:cpu_busy_permille_before": round(1000 * busy),
                 "roofline:gate_discarded_phases": discarded,
                 "roofline:n_bra": TIMING_N_BRA,
                 "roofline:n_ket": TIMING_N_KET,
                 "roofline:lanes": lanes}
        for k, r in rows.items():
            assert all(m in r.raw for m in measured.NCU_METRICS), k
            assert r.runtime_ns is None, k
            new[k] = replace(r, raw={**r.raw, **state})
        cell = measured.cell_roofline(new, measured.MeasuredKey(
            spec, arch, toolchain))
        print(f"{spec.key()}: card {before}->{after}%, discarded {discarded};"
              f" I = {cell.intensity:.2f} FLOP/B (ridge {cell.ridge:.2f}), "
              f"{cell.achieved_flops_per_s / 1e9:.1f} GFLOP/s "
              f"({cell.of_peak:.0%} of peak) -> "
              f"{'MEMORY' if cell.memory_bound else 'COMPUTE'}-bound",
              flush=True)

    # 4. one write, after every cell profiled
    elapsed = (datetime.datetime.now() - t_sweep).total_seconds()
    merged = {**store, **new}
    measured.save_counters(merged, Path(store_path))
    print(f"\n{backend} {arch.name} ({gpu}): {len(cells)} cells, {len(new)} "
          f"kernel rows, {discarded_total} phases discarded by the gate, "
          f"profiling took {elapsed:.0f} s")
    for spec, *_ in cells:
        key = measured.MeasuredKey(spec, arch, toolchain)
        ks = measured.kernel_roofline(merged, key)
        c = measured.cell_roofline(merged, key)
        print(f"  L={spec.total_am} b{spec.n_batches}: cell I {c.intensity:.2f}"
              f" {'MEM' if c.memory_bound else 'CMP'}; kernels "
              + " ".join(f"{p.intensity:.2f}{'m' if p.memory_bound else 'c'}"
                         for p in ks))


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_roofline_over_the_subgrid(request, compile_device_source,
                                        run_device_binary, tmp_path) -> None:
    """ncu counter rows of every SUBGRID batch kernel on the RTX 3080."""
    _roofline("cuda", request, compile_device_source, run_device_binary,
              tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_roofline_over_the_subgrid(request, compile_device_source,
                                       run_device_binary, tmp_path) -> None:
    """gfx1200: no fp64 FLOP counter, a final refusal (#140,
    measured_store.UNMEASURED) -- skips with the recorded reason, which
    test_toolchain.py pins to the card's own counter list; no row, never a
    zero."""
    _roofline("hip", request, compile_device_source, run_device_binary,
              tmp_path)


# --- the K wall-clock sweep (#205) --------------------------------------------
#
# OPT-IN, AND RUN ALONE, as the J sweep: it runs only with FJ_WALLCLOCK_STORE
# set AND FJ_WALLCLOCK_K=1, and its protocol is the J sweep's, steps 1-5,
# with these differences (measured_store's THE K ROWS is the rule):
#
#   devtools/measure.sh wallclock-k --backend cuda
#
# - THE CELLS are K_SUBGRID less `K_REFUSED[arch]` and `K_UNTIMED[arch]`
#   (named, never timed);
#   FJ_WALLCLOCK_LS (e.g. "7,10") restricts them by L -- never by batch
#   count, since a curve is read within one session.
# - THE CONTROL is each cell's J twin (`measured_store.j_twin`), compiled,
#   checked and timed beside it -- near-T J then K, wide-T K then J -- and
#   written into the K row's raw as `wallclock:control:*`; no J row is
#   rewritten. Both arms get the SAME P, zeta, Q, eta, at K_TIMING_N_BRA x
#   K_TIMING_N_KET: one ket per lane, because a K quartet at L = 16 costs
#   about a thousand times a J one (#202's count) and 128 kets per bra
#   would make one sequence minutes long. tau = 0: no ket is screened.
# - A PHASE THE CARD WAS TAKEN DURING IS DISCARDED AND RETAKEN (#113, #162):
#   a gate that stays shut past its deadline, or an after-reading non-idle
#   or unknown, up to STUDY_RETAKES times, counted in the K row's raw
#   `wallclock:gate_discarded_phases`; each cell is written as soon as it
#   is timed, and FJ_WALLCLOCK_RESUME=1 skips the cells already timed.

WALLCLOCK_K = "FJ_WALLCLOCK_K"
WALLCLOCK_LS = "FJ_WALLCLOCK_LS"
K_TIMING_N_BRA = TIMING_N_BRA
K_TIMING_N_KET = 32


def _stats(prefix: str, wall, busy: float, card) -> dict:
    q1, med, q3 = (float(x) for x in np.percentile(wall, [25, 50, 75]))
    return {f"{prefix}median_ns": round(med), f"{prefix}q1_ns": round(q1),
            f"{prefix}q3_ns": round(q3), f"{prefix}min_ns": int(min(wall)),
            f"{prefix}max_ns": int(max(wall)),
            f"{prefix}cpu_busy_permille_before": round(1000 * busy),
            f"{prefix}gpu_util_before": card[0],
            f"{prefix}gpu_util_after": card[1]}


def _k_wallclock(backend: str, request, compile_device_source,
                 run_device_binary, tmp_path: Path) -> None:

    store_path = os.environ.get(WALLCLOCK_STORE)
    if not store_path or not os.environ.get(WALLCLOCK_K, "").strip():
        pytest.skip(f"the K wall-clock is opt-in and must run alone, "
                    f"uncontended: set {WALLCLOCK_STORE} and {WALLCLOCK_K}=1 "
                    "(see the section comment)")
    base = _toolchain(backend, request)
    tc = _Configured(base.name, base.path, base.arch, base.ext)
    arch: Arch = BACKEND[backend][1]
    b = measured.BACKENDS[backend]
    toolchain, gpu = b.toolchain(b.compiler()), b.gpu()
    store = measured.load(Path(store_path))
    lanes = arch.wavefront

    # 1. CPU bounds
    cpuset, cpus = measured.config_value("CPUSET"), measured.config_value("CPUS")
    if cpuset:
        os.sched_setaffinity(0, _cpu_list(cpuset))
    print(f"\nCPUSET={cpuset or '(empty)'} CPUS={cpus or '(empty)'}"
          + (" -- CPUS is a container quota, not applied here" if cpus else ""))

    refused = {**measured_store.K_REFUSED.get(arch.name, {}),
               **measured_store.K_UNTIMED.get(arch.name, {})}
    for s, why in refused.items():
        print(f"{s.key()} @ {arch.name}: NOT TIMED -- {why}")
    only = os.environ.get(WALLCLOCK_LS, "").strip()
    ls = {int(x) for x in only.split(",")} if only else None
    sweep = [s for s in K_SUBGRID if s not in refused
             and (ls is None or s.total_am in ls)]
    if os.environ.get(WALLCLOCK_RESUME, "").strip():
        def done(s: Spec) -> bool:
            r = store.get(measured.measured_key(s, arch, toolchain).key())
            return (r is not None and r.runtime_ns is not None
                    and "wallclock:control:median_ns" in r.raw)
        print(f"{WALLCLOCK_RESUME}: {sum(map(done, sweep))} cells already "
              "timed, skipped")
        sweep = [s for s in sweep if not done(s)]
        if not sweep:
            pytest.skip(f"{WALLCLOCK_RESUME}: every cell already timed")
    assert sweep, f"{WALLCLOCK_LS}={only!r} selects no cell"

    # 2. compile everything first: the K cells, then their J twins
    cells = []
    for spec in sweep:
        work = tmp_path / f"k-L{spec.total_am}-b{spec.n_batches}"
        cell = _k_build(backend, spec, tc, work)
        smem, optin = _k_limits(cell.binary)
        need = k_shared_need(cell, lanes)
        assert need <= smem, (
            f"{spec.key()}: {need} B shared > the card's {smem} (opt-in "
            f"{optin}), and the cell is not in K_REFUSED")
        twin = measured_store.j_twin(spec)
        jwork = work / "twin"
        jwork.mkdir()
        jsrc, jnames, _ = _batched_source(backend, twin, jwork)
        jbin = compile_device_source(tc, jsrc).rename(jwork / "cell")
        cells.append((cell, twin, jbin, jnames))
        print(f"compiled {spec.key()} ({cell.compile_s:.0f} s, {need} B "
              f"shared of {smem}) and its J twin", flush=True)
    command = " ".join([Path(tc.path).name, *tc.compile_flags()])

    # 3. the timed binary is the checked binary -- both arms
    for cell, twin, jbin, _ in cells:
        _k_check(backend, cell, run_device_binary, tmp_path)
        _check_batched(backend, twin, jbin, run_device_binary, tmp_path)

    # 4. time, one cell at a time, both arms back to back
    discarded = [0]

    def timed(binary: Path, packed: np.ndarray, nb: int, label: str):
        src, dst = tmp_path / "timing.in", tmp_path / "timing.out"
        packed.tofile(src)
        args = (str(src), str(dst), str(WARMUP), str(REPEATS))
        check_vram(binary, args, backend)  # before the wait, off the clock
        for take in range(1 + STUDY_RETAKES):
            try:
                busy, util = gate.wait_quiet(
                    backend, say=lambda line: print(line, flush=True))
            except gate.Contended as e:
                discarded[0] += 1
                print(f"  DISCARDED {label}: {e} (take {take + 1})", flush=True)
                continue
            out = run_device_binary(binary, *args)
            card = (util, gate.after_reading(backend))
            if card[1] is not None and card[1] <= GPU_BUSY_MAX:
                break
            discarded[0] += 1
            print(f"  DISCARDED {label}: card util {card[0]} -> {card[1]}% "
                  f"(take {take + 1})", flush=True)
        else:
            pytest.fail(f"{label}: the box or the card stayed contended "
                        f"through {STUDY_RETAKES + 1} takes -- not recorded")
        wall, events, batches = _parse_reps(out, nb)
        assert len(wall) == REPEATS, out
        res = np.fromfile(dst, dtype=np.float64)
        assert np.all(np.isfinite(res)), f"{label}: non-finite result"
        half = res.size // 2
        return wall, events, batches, busy, card, bool(
            np.array_equal(res[:half], res[half:]))

    for cell, _twin, jbin, _jnames in cells:
        spec, L = cell.spec, cell.spec.total_am
        discarded[0] = 0
        load1 = os.getloadavg()[0]
        packs = {}
        for wide in (False, True):
            a = _inputs(L, wide=wide, tau_quantile=None, seed=53_000 + L,
                        n_bra=K_TIMING_N_BRA, n_ket=K_TIMING_N_KET)
            k = _k_inputs(k_spec(L), a, seed=54_000 + L + int(wide))
            packs[wide] = (_pack(L, a, lanes), _k_pack(k, lanes))
        nbt = spec.n_batches
        tag = spec.key()
        jn = timed(jbin, packs[False][0], nbt, f"{tag} J twin near")
        kn = timed(cell.binary, packs[False][1], nbt, f"{tag} near")
        kw = timed(cell.binary, packs[True][1], nbt, f"{tag} wide")
        jw = timed(jbin, packs[True][0], nbt, f"{tag} J twin wide")
        busy_after = _cpu_busy(0.5)
        wall, events, batches, busy, card, rep = kw
        q1, med, q3 = (float(x) for x in np.percentile(wall, [25, 50, 75]))
        raw = {
            **_stats("wallclock:near:", kn[0], kn[3], kn[4]),
            "wallclock:warmup": WARMUP,
            "wallclock:n_bra": K_TIMING_N_BRA,
            "wallclock:n_ket": K_TIMING_N_KET,
            "wallclock:lanes": lanes,
            "wallclock:min_ns": int(min(wall)),
            "wallclock:q1_ns": round(q1),
            "wallclock:q3_ns": round(q3),
            "wallclock:max_ns": int(max(wall)),
            "wallclock:events_median_ns": round(float(np.median(events))),
            "wallclock:cpu_busy_permille_before": round(1000 * busy),
            "wallclock:cpu_busy_permille_after": round(1000 * busy_after),
            "wallclock:loadavg1_milli": round(1000 * load1),
            "wallclock:gpu_util_before": card[0],
            "wallclock:gpu_util_after": card[1],
            "wallclock:J_bitwise_repeatable": int(rep),
            "wallclock:gate_discarded_phases": discarded[0],
            **_stats("wallclock:control:near:", jn[0], jn[3], jn[4]),
            **_stats("wallclock:control:", jw[0], jw[3], jw[4]),
        }
        per_batch = np.median(np.array(batches), axis=0)
        for name, t in zip(cell.names, per_batch, strict=True):
            raw[f"wallclock:{name}:events_median_ns"] = round(float(t))
        row = measured.Measured(
            runtime_ns=round(med), repeats=REPEATS, spread_ns=round(q3 - q1),
            toolchain=toolchain, arch=arch.name, host=measured.host(),
            gpu=gpu, date=datetime.date.today().isoformat(),
            kernel=",".join(cell.names), source_sha256=cell.sha, raw=raw)
        key = measured.measured_key(spec, arch, toolchain)
        old = store.get(key.key())
        assert old is not None and old.source_sha256 == cell.sha, (
            f"{key.key()}: no compile row of this text -- run --remeasure "
            "--subgrid --target k first")
        store = measured.merge(store, key, measured.combine(old, row))
        measured.save(store, Path(store_path))  # resumable, cell by cell
        print(f"{key.key()}: wide K {med / 1e6:.3f}+-{(q3 - q1) / 1e6:.3f} ms"
              f" vs J {raw['wallclock:control:median_ns'] / 1e6:.3f} ms; near"
              f" K {raw['wallclock:near:median_ns'] / 1e6:.3f} ms vs J "
              f"{raw['wallclock:control:near:median_ns'] / 1e6:.3f} ms; "
              f"card {card[0]}->{card[1]}%, discarded {discarded[0]}",
              flush=True)

    # 5. the reading, per L whose whole curve this session timed
    print(f"\n{backend} {arch.name} ({gpu}), compiled `{command}`")
    for L in sorted({c[0].spec.total_am for c in cells}):
        try:
            g = measured_store.k_gain(store, arch.name, L)
        except KeyError as e:
            print(f"  L={L}: gain not read ({e})")
            continue
        for i, x in g.items():
            print(f"  L={L} {i}: K gain {x.k_gain:.3f} (best b{x.k_best}), J "
                  f"gain {x.j_gain:.3f} (best b{x.j_best}) -> K >= J "
                  f"{x.k_at_least_j}")


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_k_wallclock_over_the_k_subgrid(request, compile_device_source,
                                             run_device_binary,
                                             tmp_path) -> None:
    """Wall-clock of every K_SUBGRID cell beside its J twin, RTX 3080."""
    _k_wallclock("cuda", request, compile_device_source, run_device_binary,
                 tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_k_wallclock_over_the_k_subgrid(request, compile_device_source,
                                            run_device_binary,
                                            tmp_path) -> None:
    """Wall-clock of every K_SUBGRID cell beside its J twin, RX 9060 XT."""
    _k_wallclock("hip", request, compile_device_source, run_device_binary,
                 tmp_path)
