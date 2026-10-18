"""Every lever arm on the clock, both cards, behind the card gate (#363).

THE CLOCK ARBITRATES. #362's ledger reads each kernel-body lever's arm as
the compilers see it (registers, spill); the predecessor's most repeated
lesson is that a spill win is not a wall-clock win, so this file times
every arm against its literal class on each card and records the clock on
the arm's own ledger row. measured_store.py's THE LEVER CLOCK is the rule,
declared (with `lever_clock_verdict`) before any timed KERNEL row existed.

THE CELLS are `measured_store.lever_clock_cells()`: the 22 ledger arms
(compositions included) on every class of their target at total L <= 4 --
J `pq_spec(L)`, K `k_spec(L)`, the 13 F ladder classes: 206 cells per card.

THE HARNESSES ARE THE EXISTING ONES, unchanged, driven per cell:
- J: test_kernel_device.py's `BATCHED_HARNESS` (`_batched_source`), the J
  sweep's input (`_inputs` / `_pack`, TIMING_N_BRA x TIMING_N_KET =
  2048 x 128).
- K: test_kernel_device.py's `K_HARNESS` (`_k_build`), the K sweep's input
  (2048 x K_TIMING_N_KET = 32); `kmixed-de` through
  test_k_lever_mixed_device.py's harness rewrite (its five operands
  uploaded as float) on the host's narrowed operands -- the literal control
  of that cell gets the SAME narrowed input.
- F: test/fock_driver.py's program around one synthetic off-diagonal launch
  (`synthetic_launch`, the F rows' NEAR and WIDE cubes, 2048 bras x
  F_N_KET = 32 kets, 3-primitive shells, 32 threads); `ket-break` through
  test_f_lever_ket_break_device.py's program extension with its `M_ab`
  formed by `walk_f.dominating_factors`; `block-fab` at 32 threads
  (`block_fab.launch_shape`); `scatter` as its whole path, the F kernel
  AND the scatter kernel (`fock_scatter_program`, #358's tail's `path`).
  F_N_KET is a quarter of the F rows' 128 so a card's session fits ~1.5 h
  (156 F cells, four timed sequences each); the comparison is same-session,
  so it is not compared with the F rows' clock.
Every input has tau = 0: no screen fires, every arm does all its work. So
`ket-break` (#352) and every screen-saving lever are timed for their COST
on unscreened work, not for what they save under a screen -- that is the
protocol's reach, said in the verdict, never hidden.

THE PROTOCOL, per card, in one test so no phase overlaps another:
1. CPUSET (devtools/config.sh) pins the process, as the J sweep does.
2. BUILD every cell's arm binary and each class's literal control (once per
   class), compiles through the compiler cache; a warning in generated text
   fails (F, `compile_checked`). Each arm's `source_sha256` must equal its
   ledger row's (`python -m felizjueves.measured --remeasure --ledger` first).
3. TIME each cell in ONE GATED PHASE (`felizjueves.gate`): `wait_quiet`
   (the card at most GPU_BUSY_MAX busy, the CPU at most BUSY_MAX), then
   control near, arm near, arm wide, control wide -- WARMUP untimed and
   REPEATS timed sequences each (the host steady clock around launch and
   synchronize) -- then `after_reading`. A phase the gate refuses before,
   or whose after-reading is non-idle or UNKNOWN, is DISCARDED and retaken,
   up to STUDY_RETAKES, counted in raw `wallclock:gate_discarded_phases`;
   a cell whose every take is refused gets NO ROW (and fails the test once
   the rest are written).
4. AGREE: the arm's result on each input is held to the literal control's
   from the same phase, `max |arm - literal| <= AGREE_REL * max |literal|`
   (AGREE_REL[fp32] for the fp32 levers). A gross-error witness only -- each
   lever's accuracy against the oracle is its own device test's -- so a
   fast wrong kernel cannot win; a cell that disagrees gets NO ROW.
5. WRITE the cell's row as soon as it is timed (`combine`d into the arm's
   compile row; resumable with FJ_WALLCLOCK_RESUME=1), then print every
   arm's per-class verdict and `lever_clock_verdict`, losers included.

OPT-IN, AND RUN ALONE, in the foreground, with nothing else on the box:

    REQUIRE_GPU=1 devtools/measure.sh levers --backend cuda

(`hip` for the RX 9060 XT), which sets FJ_WALLCLOCK_STORE (the tracked
data/measured/kernels.json unless already set) and FJ_WALLCLOCK_LEVERS=1
and selects `-k "lever_clock and gpu"` here, without xdist.
FJ_WALLCLOCK_ARMS="a,b" restricts the arms. REQUIRE_GPU=1 turns a missing
card (a skip) into a failure.

MOVED FROM test/ (test_lever_clock_device.py): nothing here is a correctness
claim. The J and K harnesses, inputs and checks are test_kernel_device.py's
and the timing constants measure_kernel_device.py's, imported, not copied.
The declared verdict rule on synthetic rows stayed behind, as
test/test_lever_clock.py.

COMPILE-ONLY COMPANIONS (`cuda` / `hip` + `slow`, no card) build every
cell's programs, so a spelling problem shows without the card and the
compiler cache is warm for the session (`devtools/measure.sh recheck
--full`). The no-compiler cases below hold the cell list, the fp32 set, the
sweep's own phase and row helpers, and the harness's hashes and the recorded
verdicts against the tracked ledger rows (`devtools/measure.sh check`).
"""

from __future__ import annotations

import datetime
import hashlib
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import fock_driver as FD
import measure_kernel_device as mkd
import numpy as np
import pytest
import test_f_lever_ket_break_device as kbd
import test_k_lever_mixed_device as kmd
import test_kernel_device as tkd
from plan_driver import compile_checked

from felizjueves import gate, measured, measured_store
from felizjueves import kernel_f as KF
from felizjueves.device import Arch
from felizjueves.f_levers import block_fab, ket_break, scatter
from felizjueves.f_levers import mixed as fmixed
from felizjueves.j_levers import mixed as jmixed
from felizjueves.k_levers import mixed as kmixed
from felizjueves.spec import Spec, Target
from felizjueves.walk_f import dominating_factors

LEVERS_ENV = "FJ_WALLCLOCK_LEVERS"
ARMS_ENV = "FJ_WALLCLOCK_ARMS"
REQUIRE_GPU = "REQUIRE_GPU"
WARMUP, REPEATS = mkd.WARMUP, mkd.REPEATS  # 5 and 31, the sweeps'
F_N_KET = 32
F_THREADS = FD.THREADS  # 32: the literal F launch's block
NEAR_SIDE, WIDE_SIDE = 0.25, 5.0  # test_fock_ondevice's cubes (asserted)
# The arms that compute in fp32 by design (#346, #347, #355): the agreement
# witness reads them at the fp32 tier's order, everything else at fp64's.
FP32_LEVERS = frozenset({jmixed.MIXED, kmixed.COMPUTE, kmixed.FULL,
                         fmixed.NAME})
AGREE_REL = {False: 1e-10, True: 1e-3}


def is_fp32(spec: Spec) -> bool:
    return bool(FP32_LEVERS & set(spec.levers))


def literal(spec: Spec) -> Spec:
    return replace(spec, levers=())


def selected_cells(env=os.environ) -> tuple[Spec, ...]:
    """The cells this session times: every lever clock cell, or those of
    the arms FJ_WALLCLOCK_ARMS names."""
    text = env.get(ARMS_ENV, "").strip()
    if not text:
        return measured_store.lever_clock_cells()
    out: list[Spec] = []
    for arm in (a.strip() for a in text.split(",") if a.strip()):
        out += measured_store.lever_clock_cells(arm)
    assert out, f"{ARMS_ENV}={text!r} selects no cell"
    return tuple(out)


def cell_sha(backend: str, spec: Spec) -> str:
    """`measured.cell_sha256` of the cell's units as the harness emits them:
    the ledger row's `source_sha256` when the text is today's."""
    emit, arch, _ = tkd.BACKEND[backend]
    return hashlib.sha256("".join(
        emit(k, arch) for k in measured.cell_kernels(spec)).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# One runnable arm: a binary, how to stage an input for it, how to read it
# ---------------------------------------------------------------------------


@dataclass
class Runnable:
    """One built program (an arm, or a class's literal control)."""

    spec: Spec
    binary: Path
    names: tuple[str, ...]
    sha: str
    # input label -> (staged file, output file, how to read the result)
    inputs: dict[str, tuple[Path, Path]] = field(default_factory=dict)
    reshape: Callable[[np.ndarray], np.ndarray] = lambda a: a


def _f_kets(rec: FD.FRec) -> tuple[list, list]:
    return rec.pairs("bra"), rec.pairs("ket")


@dataclass
class _Builder:
    backend: str
    tc: object  # conftest Toolchain (F) -- the J/K paths wrap it
    compile_device_source: Callable
    monkeypatch: pytest.MonkeyPatch
    work: Path

    def __post_init__(self) -> None:
        self.arch: Arch = tkd.BACKEND[self.backend][1]
        self.cfg = tkd._Configured(self.tc.name, self.tc.path, self.tc.arch,
                                   self.tc.ext)
        self.n = 0

    def _dir(self, spec: Spec) -> Path:
        self.n += 1
        d = self.work / f"c{self.n:03d}"
        d.mkdir(parents=True)
        return d

    # -- J ------------------------------------------------------------------
    def j(self, spec: Spec) -> Runnable:
        d = self._dir(spec)
        src, names, sha = tkd._batched_source(self.backend, spec, d)
        binary = self.compile_device_source(self.cfg, src).rename(d / "cell")
        return Runnable(spec, binary, tuple(names), sha)

    # -- K ------------------------------------------------------------------
    def k(self, spec: Spec) -> Runnable:
        d = self._dir(spec)
        with self.monkeypatch.context() as m:
            if kmixed.FULL in spec.levers:
                for attr, text in kmd._full_harness(tkd).items():
                    m.setattr(tkd, attr, text)
            cell = tkd._k_build(self.backend, spec, self.cfg, d)
        return Runnable(spec, cell.binary, cell.names, cell.sha)

    # -- F ------------------------------------------------------------------
    def f(self, spec: Spec) -> Runnable:
        d = self._dir(spec)
        (k,) = measured.cell_kernels(spec)
        name = f"prog_{k.name}"
        if scatter.NAME in spec.levers:
            prog = FD.fock_scatter_program([k], self.backend, d, name=name)
        elif ket_break.NAME in spec.levers:
            prog = kbd._lever_program([k], self.backend, d, name,
                                      self.monkeypatch)
        else:
            prog = FD.fock_program([k], self.backend, d, name=name)
        binary = compile_checked(self.tc, prog, d / f"{name}_bin")
        emit, arch, _ = tkd.BACKEND[self.backend]
        sha = hashlib.sha256(emit(k, arch).encode()).hexdigest()
        return Runnable(spec, binary, (k.name,), sha)

    def build(self, spec: Spec) -> Runnable:
        return {Target.J: self.j, Target.K: self.k,
                Target.F: self.f}[spec.target](spec)


def build_cells(builder: _Builder, cells) -> tuple[dict, dict]:
    """Every cell's arm and every class's literal control, built."""
    controls: dict[Spec, Runnable] = {}
    arms: dict[Spec, Runnable] = {}
    for spec in cells:
        lit = literal(spec)
        if lit not in controls:
            t0 = time.perf_counter()
            controls[lit] = builder.build(lit)
            print(f"built control {lit.key()} ({time.perf_counter() - t0:.0f}"
                  " s)", flush=True)
        t0 = time.perf_counter()
        arms[spec] = builder.build(spec)
        print(f"built {spec.key()} ({time.perf_counter() - t0:.0f} s)",
              flush=True)
    return controls, arms


# ---------------------------------------------------------------------------
# The inputs, staged to files before any gate (off the clock)
# ---------------------------------------------------------------------------


def _j_input(spec: Spec, wide: bool, lanes: int) -> tuple[np.ndarray, int]:
    L = spec.total_am
    a = tkd._inputs(L, wide=wide, tau_quantile=None, seed=53_000 + L,
                    n_bra=mkd.TIMING_N_BRA, n_ket=mkd.TIMING_N_KET)
    return tkd._pack(L, a, lanes), mkd.TIMING_N_KET


def _k_input(spec: Spec, wide: bool, lanes: int) -> tuple[np.ndarray, int]:
    L = spec.total_am
    a = tkd._inputs(L, wide=wide, tau_quantile=None, seed=53_000 + L,
                    n_bra=mkd.K_TIMING_N_BRA, n_ket=mkd.K_TIMING_N_KET)
    k = tkd._k_inputs(literal(spec), a, seed=54_000 + L + int(wide))
    if kmixed.FULL in spec.levers:  # both arms on the narrowed operands
        k = kmd._packed(k)
    return tkd._k_pack(k, lanes), mkd.K_TIMING_N_KET


def f_launch(spec: Spec, wide: bool) -> tuple[FD.FRec, int]:
    """The synthetic launch of `spec`'s class, as `spec`'s arm launches it
    (its kernel, block width and extra argument), and the basis's n_ao."""
    cls = tuple(spec.am)
    side = WIDE_SIDE if wide else NEAR_SIDE
    rec, basis = FD.synthetic_launch(cls, side=side,
                                     seed=363_000 + 100 * wide + sum(cls),
                                     n_ket=F_N_KET)
    (k,) = measured.cell_kernels(spec)
    args = dict(rec.args)
    threads = rec.threads
    if ket_break.NAME in spec.levers:
        bras, kets = _f_kets(rec)
        args[ket_break.BOUND] = dominating_factors(
            [float(x) for x in np.ravel(args["Dsh"])], int(args["n_sh"]),
            bras, kets)
    if block_fab.NAME in spec.levers or scatter.NAME in spec.levers:
        _, (threads, _) = block_fab.launch_shape(rec.grid, F_THREADS, 32)
    return FD.FRec(k, rec.grid, threads, args), basis.n_ao


def _f_stage(spec: Spec, wide: bool, monkeypatch) -> tuple[np.ndarray, int]:
    rec, n_ao = f_launch(spec, wide)
    if scatter.NAME in spec.levers:
        return FD.fock_scatter_stage([rec], [rec.kernel],
                                     mapper=FD.scatter_map_np), n_ao
    if ket_break.NAME in spec.levers:
        return kbd._lever_stage([rec], [rec.kernel], monkeypatch), n_ao
    return FD.fock_stage([rec], [rec.kernel]), n_ao


def stage(r: Runnable, cell: Spec, lanes: int, monkeypatch,
          workdir: Path) -> None:
    """Write `r`'s near and wide inputs for `cell` (the arm whose phase it
    runs in: a `kmixed-de` cell's control reads the narrowed input too)."""
    workdir.mkdir(parents=True, exist_ok=True)
    for label, wide in (("near", False), ("wide", True)):
        if cell.target is Target.J:
            data, _ = _j_input(cell, wide, lanes)
            r.reshape = lambda a: a.reshape(2, mkd.TIMING_N_BRA, -1)[0]
        elif cell.target is Target.K:
            data, _ = _k_input(cell, wide, lanes)
            r.reshape = lambda a: a.reshape(2, mkd.K_TIMING_N_BRA, -1)[0]
        else:
            data, n_ao = _f_stage(r.spec, wide, monkeypatch)
            r.reshape = (lambda n: lambda a: a.reshape(2, n, n)[0])(n_ao)
        src, dst = workdir / f"{label}.in", workdir / f"{label}.out"
        data.tofile(src)
        r.inputs[label] = (src, dst)


# ---------------------------------------------------------------------------
# The phase
# ---------------------------------------------------------------------------


def _run(r: Runnable, label: str, run_device_binary) -> tuple:
    src, dst = r.inputs[label]
    out = run_device_binary(r.binary, str(src), str(dst), str(WARMUP),
                            str(REPEATS))
    if r.spec.target is Target.F:
        wall, _ = FD.reps(out)
    else:
        wall, _, _ = mkd._parse_reps(out, len(r.names))
    assert len(wall) == REPEATS, out
    res = r.reshape(np.fromfile(dst, dtype=np.float64))
    assert np.all(np.isfinite(res)), f"{r.spec.key()} {label}: non-finite"
    return wall, res


def gated_phase(backend: str, arm: Runnable, ctl: Runnable,
                run_device_binary, retakes: int = mkd.STUDY_RETAKES):
    """The cell's one gated phase (module docstring, step 3): (CPU busy
    before, (card before, card after), discarded, {(who, input): (wall,
    result)}), or (None, None, discarded, None) when every take was refused."""
    discarded = 0
    order = (("control", "near"), ("arm", "near"), ("arm", "wide"),
             ("control", "wide"))
    who = {"arm": arm, "control": ctl}
    for take in range(1 + retakes):
        try:
            busy, util = gate.wait_quiet(
                backend, say=lambda line: print(line, flush=True))
        except gate.Contended as e:
            discarded += 1
            print(f"  DISCARDED {arm.spec.key()}: {e} (take {take + 1})",
                  flush=True)
            continue
        got = {(w, i): _run(who[w], i, run_device_binary) for w, i in order}
        after = gate.after_reading(backend)
        if gate.is_idle(after):
            return busy, (util, after), discarded, got
        discarded += 1
        print(f"  DISCARDED {arm.spec.key()}: card {util} -> {after}% "
              f"(take {take + 1})", flush=True)
    return None, None, discarded, None


def agree(spec: Spec, got: dict) -> dict[str, float]:
    """Step 4: the arm's result against the same phase's control, per
    input; returns max |arm - literal| / max |literal| per input."""
    out = {}
    for label in ("near", "wide"):
        a, c = got[("arm", label)][1], got[("control", label)][1]
        assert a.shape == c.shape, (spec.key(), a.shape, c.shape)
        top = float(np.max(np.abs(c)))
        out[label] = float(np.max(np.abs(a - c))) / (top or 1.0)
    return out


def _stats(prefix: str, wall) -> dict:
    q1, med, q3 = (float(x) for x in np.percentile(wall, [25, 50, 75]))
    return {f"{prefix}median_ns": round(med), f"{prefix}q1_ns": round(q1),
            f"{prefix}q3_ns": round(q3), f"{prefix}min_ns": int(min(wall)),
            f"{prefix}max_ns": int(max(wall))}


def row_raw(spec: Spec, got: dict, busy: float, card: tuple, discarded: int,
            load1: float, session: int, lanes: int) -> dict:
    """The arm row's raw: both inputs of both arms, and the phase's state
    on every prefix (module docstring, steps 3 and 5)."""
    raw: dict = {}
    for prefix, key in (("wallclock:", ("arm", "wide")),
                        ("wallclock:near:", ("arm", "near")),
                        ("wallclock:control:", ("control", "wide")),
                        ("wallclock:control:near:", ("control", "near"))):
        s = _stats(prefix, got[key][0])
        if prefix == "wallclock:":
            s = {k: v for k, v in s.items() if k != "wallclock:median_ns"}
        raw |= s
        raw[f"{prefix}gpu_util_before"] = card[0]
        raw[f"{prefix}gpu_util_after"] = card[1]
        raw[f"{prefix}cpu_busy_permille_before"] = round(1000 * busy)
    n_ket = {Target.J: mkd.TIMING_N_KET, Target.K: mkd.K_TIMING_N_KET,
             Target.F: F_N_KET}[spec.target]
    n_bra = {Target.J: mkd.TIMING_N_BRA, Target.K: mkd.K_TIMING_N_BRA,
             Target.F: FD.TIMING_N_BRA}[spec.target]
    raw |= {"wallclock:warmup": WARMUP, "wallclock:n_bra": n_bra,
            "wallclock:n_ket": n_ket, "wallclock:lanes": lanes,
            "wallclock:loadavg1_milli": round(1000 * load1),
            "wallclock:gate_discarded_phases": discarded,
            "wallclock:session": session}
    return raw


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def _toolchain_or_fail(backend: str, request):
    try:
        return tkd._toolchain(backend, request)
    except pytest.skip.Exception as e:
        if os.environ.get(REQUIRE_GPU, "").strip() == "1":
            pytest.fail(f"{REQUIRE_GPU}=1 and no {backend} card: {e}")
        raise


def _done(store, key: str) -> bool:
    r = store.get(key)
    return (r is not None and r.runtime_ns is not None
            and "wallclock:control:median_ns" in r.raw)


def _lever_clock(backend: str, request, compile_device_source,
                 run_device_binary, monkeypatch, tmp_path: Path) -> None:
    store_path = os.environ.get(mkd.WALLCLOCK_STORE)
    if not store_path or not os.environ.get(LEVERS_ENV, "").strip():
        pytest.skip(f"the lever clock is opt-in and must run alone, "
                    f"uncontended: set {mkd.WALLCLOCK_STORE} and "
                    f"{LEVERS_ENV}=1 (see the module docstring)")
    tc = _toolchain_or_fail(backend, request)
    arch: Arch = tkd.BACKEND[backend][1]
    b = measured.BACKENDS[backend]
    toolchain, gpu = b.toolchain(b.compiler()), b.gpu()
    store = measured.load(Path(store_path))
    lanes = arch.wavefront
    session = int(time.time())

    # 1. CPU bounds
    cpuset = measured.config_value("CPUSET")
    if cpuset:
        os.sched_setaffinity(0, mkd._cpu_list(cpuset))
    print(f"\nCPUSET={cpuset or '(empty)'}; {REQUIRE_GPU}="
          f"{os.environ.get(REQUIRE_GPU, '')!r}", flush=True)

    def key(spec: Spec) -> str:
        return measured.measured_key(spec, arch, toolchain).key()

    cells = list(selected_cells())
    if os.environ.get(mkd.WALLCLOCK_RESUME, "").strip():
        n0 = len(cells)
        cells = [s for s in cells if not _done(store, key(s))]
        print(f"{mkd.WALLCLOCK_RESUME}: {n0 - len(cells)} cells already "
              "timed, skipped", flush=True)
        if not cells:
            pytest.skip(f"{mkd.WALLCLOCK_RESUME}: every cell already timed")
    for s in cells:  # the ledger's compile rows must exist, of this text
        for c in (s, literal(s)):
            assert c in measured_store.ledger_cells(), c.key()
            assert key(c) in store, (
                f"{key(c)}: no ledger row -- run `python -m "
                "felizjueves.measured --remeasure --ledger` first")

    # 2. build everything first
    t0 = time.perf_counter()
    builder = _Builder(backend, tc, compile_device_source, monkeypatch,
                       tmp_path / "build")
    controls, arms = build_cells(builder, cells)
    for s, r in arms.items():
        assert store[key(s)].source_sha256 == r.sha, (
            f"{key(s)}: the ledger row is of other text -- remeasure first")
    for r in [*controls.values(), *arms.values()]:  # K: the launch fits
        if r.spec.target is Target.K:
            smem, _ = tkd._k_limits(r.binary)
            need = max(measured.jsh_slots(k) for k in
                       measured.cell_kernels(r.spec)) * lanes * 8
            assert need <= smem, (r.spec.key(), need, smem)
    print(f"built {len(arms)} arms and {len(controls)} controls in "
          f"{time.perf_counter() - t0:.0f} s", flush=True)

    # 3-5. one gated phase per cell, written as it is timed
    unrecorded: list[str] = []
    t_sweep = time.perf_counter()
    for i, spec in enumerate(cells):
        arm, ctl = arms[spec], controls[literal(spec)]
        cdir = tmp_path / "inputs" / f"c{i:03d}"
        stage(arm, spec, lanes, monkeypatch, cdir / "arm")
        stage(ctl, spec, lanes, monkeypatch, cdir / "control")
        for r in (arm, ctl):  # the VRAM probe, off the clock
            for src, dst in r.inputs.values():
                tkd.check_vram(r.binary, (str(src), str(dst), str(WARMUP),
                                          str(REPEATS)), backend)
        load1 = os.getloadavg()[0]
        busy, card, discarded, got = gated_phase(backend, arm, ctl,
                                                 run_device_binary)
        if got is None:
            unrecorded.append(f"{spec.key()}: the gate refused every take "
                              f"({discarded})")
            print(f"NOT RECORDED {unrecorded[-1]}", flush=True)
            continue
        err = agree(spec, got)
        tol = AGREE_REL[is_fp32(spec)]
        if max(err.values()) > tol:
            unrecorded.append(f"{spec.key()}: arm disagrees with the literal "
                              f"control ({err}, tol {tol:g})")
            print(f"NOT RECORDED {unrecorded[-1]}", flush=True)
            continue
        wall = got[("arm", "wide")][0]
        q1, med, q3 = (float(x) for x in np.percentile(wall, [25, 50, 75]))
        row = measured.Measured(
            runtime_ns=round(med), repeats=REPEATS, spread_ns=round(q3 - q1),
            toolchain=toolchain, arch=arch.name, host=measured.host(), gpu=gpu,
            date=datetime.date.today().isoformat(), kernel=",".join(arm.names),
            source_sha256=arm.sha,
            raw=row_raw(spec, got, busy, card, discarded, load1, session,
                        lanes))
        mk = measured.measured_key(spec, arch, toolchain)
        old = store.get(mk.key())
        assert old is not None and old.source_sha256 == arm.sha
        store = measured.merge(store, mk, measured.combine(old, row))
        measured.save(store, Path(store_path))
        ab = measured_store.lever_ab(store, arch.name, spec)
        print(f"[{i + 1}/{len(cells)} {time.perf_counter() - t_sweep:.0f} s] "
              f"{spec.key()}: wide {ab.wide[2] / 1e6:.3f}+-"
              f"{ab.wide[3] / 1e6:.3f} vs literal {ab.wide[0] / 1e6:.3f}+-"
              f"{ab.wide[1] / 1e6:.3f} ms ({ab.wide_sign}); near "
              f"{ab.near[2] / 1e6:.3f} vs {ab.near[0] / 1e6:.3f} ms "
              f"({ab.near_sign}) -> {ab.verdict.upper()}; card {card[0]}->"
              f"{card[1]}%, discarded {discarded}; agree "
              f"{max(err.values()):.1e}", flush=True)

    # the verdicts, every arm, losers included
    print(f"\n#363 lever clock on {arch.name} ({gpu}, {toolchain}):")
    for a in measured_store.ledger_arms():
        try:
            abs_ = measured_store.lever_clock_ab(store, arch.name, a)
        except KeyError as e:
            print(f"  {a}: not judged ({e})")
            continue
        counts = {v: sum(x.verdict == v for x in abs_.values())
                  for v in (measured_store.WIN, measured_store.LOSS,
                            measured_store.NEUTRAL, measured_store.MIXED)}
        verdict = measured_store.lever_clock_verdict(store, arch.name, a)
        print(f"  {a}: {verdict.upper()} "
              f"({', '.join(f'{k} {v}' for k, v in counts.items())})")
    assert not unrecorded, ("cells with NO ROW:\n  "
                            + "\n  ".join(unrecorded))


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_lever_clock(request, compile_device_source, run_device_binary,
                          monkeypatch, tmp_path) -> None:
    """Every lever arm against its literal class on the RTX 3080 (opt-in)."""
    _lever_clock("cuda", request, compile_device_source, run_device_binary,
                 monkeypatch, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_lever_clock(request, compile_device_source, run_device_binary,
                         monkeypatch, tmp_path) -> None:
    """Every lever arm against its literal class on the RX 9060 XT (opt-in)."""
    _lever_clock("hip", request, compile_device_source, run_device_binary,
                 monkeypatch, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions (no card): every cell's programs build
# ---------------------------------------------------------------------------

ARM_PARAMS = [pytest.param(a, id=a) for a in measured_store.ledger_arms()]


def _compiles(backend: str, tc, arm: str, compile_device_source,
              monkeypatch, tmp_path: Path) -> None:
    builder = _Builder(backend, tc, compile_device_source, monkeypatch,
                       tmp_path)
    cells = measured_store.lever_clock_cells(arm)
    controls, arms = build_cells(builder, cells)
    for s, r in arms.items():
        assert r.sha == cell_sha(backend, s), s.key()
        stage(r, s, 32, monkeypatch, tmp_path / "in" / r.binary.parent.name)
        stage(controls[literal(s)], s, 32, monkeypatch,
              tmp_path / "in" / f"ctl-{r.binary.parent.name}")


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize("arm", ARM_PARAMS)
def test_cuda_lever_clock_programs_compile(arm, nvcc, compile_device_source,
                                           monkeypatch, tmp_path) -> None:
    _compiles("cuda", nvcc, arm, compile_device_source, monkeypatch, tmp_path)


@pytest.mark.hip
@pytest.mark.slow
@pytest.mark.parametrize("arm", ARM_PARAMS)
def test_hip_lever_clock_programs_compile(arm, hipcc, compile_device_source,
                                          monkeypatch, tmp_path) -> None:
    _compiles("hip", hipcc, arm, compile_device_source, monkeypatch, tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the cells, the fp32 set, the inputs, the hashes
# ---------------------------------------------------------------------------


def test_the_cells_are_every_ledger_arm_at_l_at_most_4() -> None:
    cells = selected_cells({})
    assert len(cells) == len(set(cells)) == 206
    assert all(c.total_am <= 4 and c.levers for c in cells)
    got = {measured_store.arm_of(c) for c in cells}
    assert got == set(measured_store.ledger_arms()) and len(got) == 22
    assert set(map(literal, cells)) <= set(measured_store.ledger_cells())
    assert selected_cells({ARMS_ENV: "fused, split"}) == (
        measured_store.lever_clock_cells("fused")
        + measured_store.lever_clock_cells("split"))
    with pytest.raises(ValueError, match="not a ledger arm"):
        selected_cells({ARMS_ENV: "recycle"})


def test_the_fp32_levers_are_vocabulary_levers() -> None:
    from felizjueves import levers
    assert FP32_LEVERS <= set(levers.VOCABULARY)
    arms = measured_store.ledger_arms()
    assert {a for a in arms if FP32_LEVERS & set(a.split("+"))} == {
        "mixed", "kmixed", "kmixed-de", "fmixed"}


def test_the_f_inputs_are_the_f_rows_cubes_and_stay_witnessed() -> None:
    import test_fock_ondevice as tfo
    assert (NEAR_SIDE, WIDE_SIDE) == (tfo.NEAR_SIDE, tfo.WIDE_SIDE)
    for cls in [(0, 0, 0, 0), (0, 2, 1, 1), (1, 1, 1, 1)]:
        spec = KF.f_class_spec(*cls)
        for wide in (False, True):
            rec, n_ao = f_launch(spec, wide)
            assert (rec.args["n_bra"], rec.args["n_ket"]) == (
                FD.TIMING_N_BRA, F_N_KET)
            assert rec.args["tau"] == 0.0 and rec.threads == F_THREADS
        _, basis = FD.synthetic_launch(cls, side=WIDE_SIDE,
                                       seed=363_100 + sum(cls), n_ket=F_N_KET)
        assert FD.max_boys_T(basis, WIDE_SIDE) <= tfo.BOYS_T_WITNESSED


def test_the_ket_break_launch_carries_its_dominating_factor() -> None:
    spec = KF.f_class_spec(0, 1, 0, 1, (ket_break.NAME,))
    rec, _ = f_launch(spec, True)
    assert len(rec.args[ket_break.BOUND]) == rec.args["n_bra"]
    G = [float(x) for x in rec.args["G_cd"]]
    assert G == sorted(G, reverse=True)  # the break's sorted stream


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_the_harness_emits_the_ledger_rows_text(backend) -> None:
    """A sample of cells, one per target and the special harnesses: the
    hash the sweep asserts against each ledger row matches it today."""
    store = measured_store.load()
    _, arch, _ = tkd.BACKEND[backend]
    for a in ("fused", "split", "kmixed-de", "ket-break", "scatter",
              "block-fab"):
        spec = measured_store.lever_clock_cells(a)[-1]
        (r,) = measured_store.rows(spec, arch.name, store=store).values()
        assert r.source_sha256 == cell_sha(backend, spec), spec.key()


def test_the_agreement_witness_reads_the_arm_against_the_control() -> None:
    spec = measured_store.lever_clock_cells("fused")[0]
    c = np.array([[1.0, -2.0], [0.5, 0.0]])
    got = {("control", "near"): (None, c), ("arm", "near"): (None, c),
           ("control", "wide"): (None, c),
           ("arm", "wide"): (None, c + np.array([[0, 2e-9], [0, 0]]))}
    err = agree(spec, got)
    assert err["near"] == 0.0 and err["wide"] == pytest.approx(1e-9)
    assert err["wide"] > AGREE_REL[False] and err["wide"] < AGREE_REL[True]


def test_the_phase_discards_a_refused_take_and_records_nothing(
        monkeypatch) -> None:
    """The gate's refusals, injected: a contended before-reading and a busy
    after-reading are both discarded; every take refused gives no result."""
    calls = []

    def wait(backend, say=None):
        calls.append("wait")
        if len(calls) == 1:
            raise gate.Contended("card 99% busy")
        return 0.01, 3

    after = iter([95, 2])
    monkeypatch.setattr(gate, "wait_quiet", wait)
    monkeypatch.setattr(gate, "after_reading", lambda b: next(after))
    monkeypatch.setattr(sys.modules[__name__], "_run",
                        lambda r, label, run: ([1, 2, 3], np.zeros(1)))
    r = Runnable(KF.f_class_spec(0, 0, 0, 0), Path("x"), ("x",), "")
    busy, card, discarded, got = gated_phase("cuda", r, r, None)
    assert (busy, card, discarded) == (0.01, (3, 2), 2) and len(got) == 4
    monkeypatch.setattr(gate, "after_reading", lambda b: None)
    calls.clear()
    calls.append("skip-contended")
    busy, card, discarded, got = gated_phase("cuda", r, r, None, retakes=2)
    assert got is None and card is None and discarded == 3


def test_the_row_raw_carries_both_arms_and_the_contention_state() -> None:
    spec = measured_store.lever_clock_cells("fmixed")[3]
    w = list(range(100, 131))
    got = {(x, i): (w, None) for x in ("arm", "control")
           for i in ("near", "wide")}
    raw = row_raw(spec, got, 0.02, (4, 1), 1, 0.5, 7, 32)
    for f in measured_store.GPU_UTIL_FIELDS:
        assert raw[f] is not None
    for p in ("wallclock:control:", "wallclock:control:near:",
              "wallclock:near:"):
        assert {f"{p}median_ns", f"{p}q1_ns", f"{p}q3_ns"} <= set(raw)
    assert "wallclock:median_ns" not in raw  # it is the row's runtime_ns
    assert raw["wallclock:gate_discarded_phases"] == 1
    assert raw["wallclock:n_ket"] == F_N_KET
    assert all(type(v) is int for v in raw.values())


# The recorded result (#363's two sessions, one per card): every arm's
# verdict per arch, read off the tracked store by the declared rule.
_F, _S, _N = (measured_store.FASTER_ARM, measured_store.SLOWER_ARM,
              measured_store.NOT_SHOWN)
RECORDED = {  # arm: (sm_86, gfx1200)
    "batched-vrr": (_S, _S), "block-fab": (_F, _N), "coalesced": (_N, _N),
    "etr": (_N, _S), "fmixed": (_F, _F), "fused": (_F, _N),
    "interleave": (_F, _F), "ket-break": (_N, _N), "kmixed": (_F, _N),
    "kmixed-de": (_F, _F), "mixed": (_F, _S), "scatter": (_N, _S),
    "shuffle": (_S, _N), "batched-vrr+sliced-hrr": (_S, _S),
    "batched-vrr+sliced-hrr+sliced-site-digest": (_S, _S),
    "batched-vrr+sliced-hrr+slim-fock": (_S, _S), "split": (_S, _S),
    "stage-eac": (_S, _S), "struct-reads": (_N, _S),
    "batched-vrr+sliced-hrr+slim-fock+sub-slice": (_S, _S),
    "batched-vrr+subdag-vrr": (_S, _S), "interleave+split": (_S, _S),
}


def test_every_cell_is_timed_gated_on_both_cards_and_the_verdicts_hold(
        ) -> None:
    """The rows prove both cards (not a green run): every lever clock cell
    has a gated, same-session-controlled row on sm_86 and gfx1200, and the
    declared rule reads RECORDED off them -- losers included."""
    store = measured_store.load()
    assert set(RECORDED) == set(measured_store.ledger_arms())
    for arch in ("sm_86", "gfx1200"):
        for spec in measured_store.lever_clock_cells():
            (r,) = measured_store.rows(spec, arch, store=store).values()
            assert r.runtime_ns is not None and r.repeats == REPEATS
            assert "wallclock:control:median_ns" in r.raw
            assert "wallclock:gate_discarded_phases" in r.raw
            for f in measured_store.GPU_UTIL_FIELDS:
                assert r.raw[f] is not None and r.raw[f] <= gate.GPU_BUSY_MAX
    for arm, want in RECORDED.items():
        got = tuple(measured_store.lever_clock_verdict(store, a, arm)
                    for a in ("sm_86", "gfx1200"))
        assert got == want, arm
    assert [a for a in RECORDED if measured_store.lever_may_become_the_default(
        store, a)] == ["fmixed"]
