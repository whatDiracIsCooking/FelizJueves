"""The DF kernels' two forms (#606): compiled across the aux ladder, timed.

    devtools/measure.sh df-lowstack-compile [--backend cuda|hip]   a compiler
    devtools/measure.sh df-lowstack [--backend cuda|hip]           timed

OPT-IN: each test skips unless FJ_DF_LOWSTACK_STORE names the store
(measure.sh sets it to config.sh's MEASURE_DF_LOWSTACK_STORE, the tracked
data/measured/df_lowstack.json). The clock runs ALONE.

THE COMPILE (no card). Every class of the ladder FreeBreakfast emits -- eri3
`la <= lb <= 3`, `lP <= 6` (70 classes) and the metric `lP <= lQ <= 6` (28)
-- in its DEFAULT form (`eri3_lowstack` / `metric_lowstack`), and beside it
the OTHER form of the classes the gate is read on (`OTHER_FORM`): the
unrolled (df|g) and (ff|g), the before of #606's after, and the low-stack
form of every eri3 class the default keeps unrolled from `L = 7`. Each unit
is compiled ALONE, one at a time, with the measured store's flags
(`measured.BACKENDS`: nvcc `sm_86` `-Xptxas -v`, hipcc `gfx1200`
`-Rpass-analysis=kernel-resource-usage`), through `os.wait4` for the
compiler's peak RSS: source bytes, compile wall-clock, peak RSS, registers,
spill, stack (nvcc's per-kernel stack frame before device link; hipcc's
ScratchSize). A unit that does not finish within `COMPILE_BUDGET_S = 300`
s is killed and recorded `over_budget` with no figures (the user's
per-compile budget for #606). One row per (arch, toolchain, unit, form).

THE CLOCK (a card). Per class of `CLOCK_CLASSES`, both forms where both
compile within the budget, on `_clock_bases(cls)`: 8 shells of each bra
`l` and 32 aux shells on a grid, two primitives each (so every cell walks
4 bra records), and the class's cells only. `ROUNDS` rounds, each running
the low-stack then the unrolled program (WARMUP untimed launches, REPEATS
timed, one launch per event pair), each phase behind `gate.wait_quiet` with
its after-reading, a contended phase retaken up to RETAKES times and then
the row refused. The two forms' tensors are first held to each other
(1e-14 of max, `test_kernel_df_lowstack.REL`).

THE RULE is the store's `declared`, written before any row (DECLARED).
"""

from __future__ import annotations

import datetime
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import eri3_driver
import metric_driver
import numpy as np
import pytest
from test_kernel_device import BACKEND, _toolchain

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.density_fit import kernel_eri3 as ke
from felizjueves.eri.density_fit import kernel_metric as km
from felizjueves.eri.density_fit.eri3 import eri3_classes
from felizjueves.eri.density_fit.metric import metric_classes
from felizjueves.performance import gate, measured
from felizjueves.performance.gate import GPU_BUSY_MAX

STORE = "FJ_DF_LOWSTACK_STORE"
SCHEMA = 1
COMPILE_BUDGET_S = 300.0  # the user's per-compile budget (#606)
STACK_BUDGET_B = 14_000  # #606's "~14 KB, i.e. (dd|g)'s today" (13,856 B)
ROUNDS, WARMUP, REPEATS, RETAKES = 3, 2, 7, 3
BAND = 0.05

ERI3_LADDER = [(a, b, p) for a in range(4) for b in range(a, 4)
               for p in range(7)]
METRIC_LADDER = [(p, q) for p in range(7) for q in range(p, 7)]

DECLARED = {
    "issue": 606,
    "question": "which DF classes the low-stack gate must cover; the open "
                "one is the unrolled (dd|g), 15,992 B of sm_86 stack by "
                "hand, over #606's ~14 KB",
    "rule": "per class and form, over both arches: a class's UNROLLED form "
            "is admissible when its stack is at most STACK_BUDGET_B = "
            "14,000 B on sm_86 and on gfx1200 and its compile finishes "
            "within COMPILE_BUDGET_S = 300 s on both. Every class whose "
            "unrolled form is not admissible must be under the gate (the "
            "acceptance: every class through (ff|i) / (i|i) at <= ~14 KB, "
            "and the user's 5-minute compile budget). A class whose "
            "unrolled form IS admissible stays unrolled unless the clock "
            "says the low-stack form is 'faster' (median low / unrolled "
            "below 1 - BAND = 0.95) on both cards. The gate is then the "
            "smallest of the module's rule shapes (an L floor, an E-block "
            "bound) that covers every class so marked and no admissible "
            "class the clock does not move; where none does, the shape is "
            "widened, never the budget",
    "expected": "the unrolled (dd|g) is not admissible on sm_86 (15,992 B "
                "by hand), so the E-block arm tightens below (dd|g)'s; the "
                "low-stack form is slower than the unrolled one where both "
                "fit, as V's and DQ's were at the mixed f classes (#586)",
    "clock": "per class: median over every timed launch of both forms in "
             "the same rounds, 'faster' / 'slower' / 'neutral' at BAND; a "
             "record, ruled on only as above",
    "declared": "2026-10-28, before any row",
}


def _store_path() -> Path:
    p = os.environ.get(STORE)
    if not p:
        pytest.skip(f"the DF low-stack records are opt-in: set {STORE} "
                    "(devtools/measure.sh df-lowstack[-compile])")
    return Path(p)


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "issue": 606,
            "about": __doc__.split("\n\n")[1], "compile": [], "clock": []}


def _declare(path: Path, store: dict) -> None:
    declared = json.loads(json.dumps(DECLARED))
    if "declared" not in store:
        store["declared"] = declared
        path.write_text(json.dumps(store, indent=1) + "\n")
    assert store["declared"] == declared, (
        "the store's declared rule is not this file's: its rows would be "
        "read under another rule")


def _save(path: Path, store: dict, section: str, row: dict,
          key: tuple[str, ...]) -> None:
    rows = [r for r in store[section]
            if tuple(r[k] for k in key) != tuple(row[k] for k in key)]
    rows.append(row)
    rows.sort(key=lambda r: tuple(str(r[k]) for k in key))
    store[section] = rows
    path.write_text(json.dumps(store, indent=1) + "\n")


# --- what is compiled -----------------------------------------------------------


def _default_low(target: str, cls) -> bool:
    return (ke.eri3_lowstack(cls) if target == "eri3"
            else km.metric_lowstack(cls))


def _other_form() -> list[tuple[str, tuple[int, ...]]]:
    """The classes compiled in their non-default form too."""
    out = [("eri3", (2, 3, 4)), ("eri3", (3, 3, 4))]  # #606's before
    out += [("eri3", c) for c in ERI3_LADDER
            if sum(c) >= 7 and not ke.eri3_lowstack(c)]
    out += [("metric", c) for c in METRIC_LADDER
            if sum(c) >= 7 and not km.metric_lowstack(c)]
    return out


OTHER_FORM = _other_form()


def compile_units() -> list[tuple[str, tuple[int, ...], bool]]:
    """(target, class, lowstack) of every unit the compile sweep builds."""
    units = [("eri3", c, ke.eri3_lowstack(c)) for c in ERI3_LADDER]
    units += [("metric", c, km.metric_lowstack(c)) for c in METRIC_LADDER]
    units += [(t, c, not _default_low(t, c)) for t, c in OTHER_FORM]
    return units


def _build(target: str, cls, lowstack: bool):
    if target == "eri3":
        return ke.build_eri3_kernel(cls, lowstack=lowstack)
    return km.build_metric_kernel(cls, lowstack=lowstack)


def _compile(backend: str, kern, workdir: Path, *,
             budget: float = COMPILE_BUDGET_S) -> dict:
    """One unit through `os.wait4`, killed at `budget` s (COMPILE_BUDGET_S
    unless the caller has its own: #618's 10 s)."""
    b = measured.BACKENDS[backend]
    text = BACKEND[backend][0](kern, b.arch)
    src = workdir / f"{kern.name}.{b.ext}"
    obj = workdir / f"{kern.name}.o"
    src.write_text(text)
    cmd = b.command(b.compiler(), src, obj)
    log = workdir / f"{kern.name}.log"
    t0 = time.perf_counter()
    killed = threading.Event()

    def kill(pid: int) -> None:
        killed.set()
        os.killpg(pid, signal.SIGKILL)

    with log.open("w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        timer = threading.Timer(budget, kill, (proc.pid,))
        timer.start()
        _, status, ru = os.wait4(proc.pid, 0)
        timer.cancel()
    wall = time.perf_counter() - t0
    out = log.read_text()
    for f in (src, obj, log):
        f.unlink(missing_ok=True)
    row = {"unit": kern.name, "source_bytes": len(text)}
    if killed.is_set():
        return {**row, "over_budget": True, "compile_s": None,
                "killed_at_s": round(wall, 1), "peak_rss_kb": ru.ru_maxrss}
    assert os.waitstatus_to_exitcode(status) == 0, out[-4000:]
    raw = (measured.parse_ptxas(out, kern.name) if backend == "cuda"
           else measured.parse_amd_remarks(out, kern.name))
    regs, store, load, _ = measured._numbers(b, raw)
    return {**row, "over_budget": False, "compile_s": round(wall, 2),
            "peak_rss_kb": ru.ru_maxrss, "registers": regs,
            "spill_store": store, "spill_load": load,
            "stack_bytes": raw.get("stack_frame_bytes",
                                   raw.get("ScratchSize"))}


def _compile_rows(backend: str, request, tmp_path: Path) -> None:
    path = _store_path()
    _toolchain(backend, request)
    b = measured.BACKENDS[backend]
    store = _load(path)
    _declare(path, store)
    head = {"arch": b.arch.name, "toolchain": b.toolchain(b.compiler()),
            "host": measured.host(),
            "date": datetime.date.today().isoformat()}
    only = os.environ.get("FJ_DF_LOWSTACK_ONLY")  # e.g. "eri3_334,metric_66"
    for target, cls, low in compile_units():
        kern = _build(target, cls, low)
        if only and kern.name not in only.split(","):
            continue
        r = _compile(backend, kern, tmp_path)
        row = {**head, "target": target, "am": list(cls), "L": sum(cls),
               "form": "lowstack" if low else "unrolled",
               "default": low == _default_low(target, cls), **r}
        store = _load(path)
        _save(path, store, "compile", row,
              ("arch", "toolchain", "unit", "form"))
        print(f"{backend} {r['unit']} {row['form']}: "
              + ("OVER BUDGET" if r["over_budget"] else
                 f"{r['compile_s']} s, r{r['registers']} spill "
                 f"{r['spill_store']} stack {r['stack_bytes']}"),
              flush=True)


# --- the clock ------------------------------------------------------------------

# (target, class): the gate's question classes in both forms (the unrolled
# kept ones the rule reads, #606's (df|g) and (ff|g) before and after), and
# the top of each ladder in the default form alone
CLOCK_CLASSES = (("eri3", (2, 2, 4)), ("eri3", (1, 3, 4)),
                 ("eri3", (1, 2, 5)), ("eri3", (0, 3, 5)),
                 ("eri3", (2, 3, 4)), ("eri3", (3, 3, 4)),
                 ("eri3", (3, 3, 6)),
                 ("metric", (4, 4)), ("metric", (4, 5)), ("metric", (6, 6)))
N_AO_SHELLS, N_AUX_SHELLS, N_METRIC_SHELLS = 12, 64, 96


def _grid(i: int, step: float) -> tuple[float, float, float]:
    return (step * (i % 4), step * ((i // 4) % 4), 0.7 * (i // 16)
            + 0.13 * (i % 3))


def _clock_bases(target: str, cls) -> tuple[Basis, Basis | None]:
    """eri3: `N_AO_SHELLS` shells of each bra `l` (one set when `la = lb`,
    16 then), two primitives each, and `N_AUX_SHELLS` aux shells of `lP`,
    one primitive, on 4 x 4 grids. metric: `N_METRIC_SHELLS` aux shells of
    each `l`."""
    if target == "metric":
        lP, lQ = cls
        ls = [lP] if lP == lQ else [lP, lQ]
        n = N_METRIC_SHELLS if lP == lQ else N_METRIC_SHELLS // 2
        aux = Basis(tuple(Shell(lx, _grid(i, 1.1), (1.3 - 0.01 * (i % 7),),
                                (1.0,)) for lx in ls for i in range(n)))
        return aux, None
    la, lb, lP = cls
    ls = [la] if la == lb else [la, lb]
    n = 16 if la == lb else N_AO_SHELLS
    basis = Basis(tuple(Shell(lx, _grid(i, 1.5), (1.4, 0.45), (0.6, 0.5))
                        for lx in ls for i in range(n)))
    aux = Basis(tuple(Shell(lP, _grid(i, 1.2), (1.1 - 0.01 * (i % 5),),
                            (1.0,)) for i in range(N_AUX_SHELLS)))
    return basis, aux


def _stats(xs) -> dict[str, float]:
    q1, med, q3 = (float(x) for x in np.percentile(xs, [25, 50, 75]))
    return {"median_ns": round(med), "q1_ns": round(q1), "q3_ns": round(q3),
            "n": len(xs)}


def _verdict(ratio: float) -> str:
    return ("faster" if ratio < 1 - BAND else
            "slower" if ratio > 1 + BAND else "neutral")


def _compiles_in_budget(store: dict, arch: str, unit: str, form: str
                        ) -> bool | None:
    """The compile sweep's answer for this unit and form (None: no row)."""
    for r in store["compile"]:
        if (r["arch"], r["unit"], r["form"]) == (arch, unit, form):
            return not r["over_budget"]
    return None


def _clock(backend: str, request, run_device_binary, tmp_path: Path) -> None:
    path = _store_path()
    tc = _toolchain(backend, request)
    b = measured.BACKENDS[backend]
    store = _load(path)
    _declare(path, store)
    only = os.environ.get("FJ_DF_LOWSTACK_ONLY")
    for target, cls in CLOCK_CLASSES:
        kern = _build(target, cls, _default_low(target, cls))
        if only and kern.name not in only.split(","):
            continue
        forms = []
        for low in (True, False):
            form = "lowstack" if low else "unrolled"
            ok = _compiles_in_budget(store, b.arch.name, kern.name, form)
            if ok:
                forms.append(low)
            else:
                print(f"{kern.name} {form}: not timed ("
                      + ("over the compile budget" if ok is False
                         else "no compile row") + ")", flush=True)
        if not forms:
            continue
        basis, aux = _clock_bases(target, cls)
        work = tmp_path / kern.name
        work.mkdir()
        if target == "eri3":
            classes = [c for c in eri3_classes(basis, aux)
                       if (c.la, c.lb, c.lP) == cls]
            n_ao, n_aux = basis.n_ao, aux.n_ao
            units = {low: eri3_driver.compile_timed(
                classes, n_ao, n_aux, backend, tc, work / str(low),
                lowstack=low, timeout=COMPILE_BUDGET_S) for low in forms}

            def run(low, warmup, reps, classes=classes, n_ao=n_ao,
                    n_aux=n_aux, units=units, work=work):
                return eri3_driver.run_timed(
                    units[low], classes, n_ao, n_aux, backend,
                    work / str(low), run_device_binary, warmup=warmup,
                    reps=reps)
            size = {"n_ao": n_ao, "n_aux": n_aux,
                    "n_cell": sum(c.n_cell for c in classes),
                    "n_bra_rec": sum(c.n_bra_rec for c in classes)}
        else:
            classes = [c for c in metric_classes(basis)
                       if (c.lP, c.lQ) == cls]
            n_aux = basis.n_ao
            units = {low: metric_driver.compile_timed(
                classes, n_aux, backend, tc, work / str(low), lowstack=low,
                timeout=COMPILE_BUDGET_S) for low in forms}

            def run(low, warmup, reps, classes=classes, n_aux=n_aux,
                    units=units, work=work):
                return metric_driver.run_timed(
                    units[low], classes, n_aux, work / str(low),
                    lowstack=low, warmup=warmup, reps=reps)
            size = {"n_aux": n_aux,
                    "n_pair": sum(c.n_pair for c in classes)}
        assert len(classes) == 1, (kern.name, len(classes))
        outs = {low: run(low, 0, 1)[0] for low in forms}
        rel = None
        if len(forms) == 2:
            ref = outs[False]
            s = float(np.max(np.abs(ref)))
            assert s > 0.0, kern.name
            rel = float(np.max(np.abs(outs[True] - ref))) / s
            assert rel <= 1e-14, f"{kern.name}: forms differ by {rel:.2e}"
        per: dict[bool, list[float]] = {low: [] for low in forms}
        readings, discarded = [], 0
        for _ in range(ROUNDS):
            for low in forms:
                for take in range(RETAKES + 1):
                    try:
                        busy, util = gate.wait_quiet(
                            backend, say=lambda s: print(s, flush=True))
                    except gate.Contended as e:
                        discarded += 1
                        print(f"  DISCARDED {kern.name}: {e} "
                              f"(take {take + 1})")
                        continue
                    _, t = run(low, WARMUP, REPEATS)
                    after = gate.after_reading(backend)
                    if util is not None and after is not None \
                            and after <= GPU_BUSY_MAX:
                        break
                    discarded += 1
                    print(f"  DISCARDED {kern.name}: card {util} -> {after}%")
                else:
                    pytest.fail(f"{kern.name}: contended through "
                                f"{RETAKES + 1} takes -- not recorded")
                readings.append({"form": "lowstack" if low else "unrolled",
                                 "cpu_busy_permille": round(1000 * busy),
                                 "gpu_util_before": util,
                                 "gpu_util_after": after})
                per[low].extend(t["0"])
        arms = {("lowstack" if low else "unrolled"): _stats(per[low])
                for low in forms}
        row = {"arch": b.arch.name, "toolchain": b.toolchain(b.compiler()),
               "gpu": b.gpu(), "host": measured.host(),
               "date": datetime.date.today().isoformat(),
               "target": target, "am": list(cls), "unit": kern.name,
               "default": "lowstack" if _default_low(target, cls)
               else "unrolled", **size, "rounds": ROUNDS, "warmup": WARMUP,
               "repeats": REPEATS, "gate": readings,
               "gate_discarded_runs": discarded, "arms": arms,
               "lowstack_vs_unrolled_rel": rel}
        if len(forms) == 2:
            ratio = (arms["lowstack"]["median_ns"]
                     / arms["unrolled"]["median_ns"])
            row["lowstack_over_unrolled"] = round(ratio, 4)
            row["verdict"] = _verdict(ratio)
        store = _load(path)
        _save(path, store, "clock", row, ("arch", "toolchain", "unit"))
        print(f"{backend} {kern.name}: " + ", ".join(
            f"{a} {v['median_ns'] / 1e6:.3f} ms" for a, v in arms.items())
            + (f" -> {row['verdict']} ({row['lowstack_over_unrolled']})"
               if "verdict" in row else ""), flush=True)


# --- the rule, read back from the tracked rows (`measure.sh check`) -------------

TRACKED = (Path(__file__).resolve().parent.parent
           / "data/measured/df_lowstack.json")
ARCHES = ("sm_86", "gfx1200")


def admissible(store: dict, unit: str) -> bool | None:
    """DECLARED's rule on the unrolled form of `unit`, over both arches
    (None: not compiled unrolled on both)."""
    rows = {r["arch"]: r for r in store["compile"]
            if r["unit"] == unit and r["form"] == "unrolled"}
    if not all(a in rows for a in ARCHES):
        return None
    return all(not r["over_budget"] and r["stack_bytes"] <= STACK_BUDGET_B
               for r in rows.values())


def clock_moves(store: dict, unit: str) -> bool:
    """The low-stack form 'faster' on both cards."""
    rows = [r for r in store.get("clock", []) if r["unit"] == unit]
    return (len(rows) == len(ARCHES)
            and all(r.get("verdict") == "faster" for r in rows))


def test_the_gate_is_what_the_measured_rows_rule():
    store = json.loads(TRACKED.read_text())
    assert store["declared"] == json.loads(json.dumps(DECLARED))
    names = {}
    for c in ERI3_LADDER:
        names[ke.build_eri3_kernel(c, lowstack=False).name] = (
            ke.eri3_lowstack(c))
    for c in METRIC_LADDER:
        names[km.build_metric_kernel(c, lowstack=False).name] = (
            km.metric_lowstack(c))
    seen = 0
    for unit, gated in names.items():
        ok = admissible(store, unit)
        if ok is None:
            continue
        seen += 1
        want = (not ok) or clock_moves(store, unit)
        assert gated == want, (unit, gated, ok)
    assert seen >= 60, seen
    # every low-stack row of the ladder is inside both budgets
    for r in store["compile"]:
        if r["form"] == "lowstack":
            assert not r["over_budget"], r["unit"]
            assert r["stack_bytes"] <= STACK_BUDGET_B, r["unit"]


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_df_lowstack_clock(request, run_device_binary, tmp_path) -> None:
    _clock("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_df_lowstack_clock(request, run_device_binary, tmp_path) -> None:
    _clock("hip", request, run_device_binary, tmp_path)


@pytest.mark.cuda
def test_cuda_df_lowstack_compile(request, tmp_path) -> None:
    _compile_rows("cuda", request, tmp_path)


@pytest.mark.hip
def test_hip_df_lowstack_compile(request, tmp_path) -> None:
    _compile_rows("hip", request, tmp_path)
