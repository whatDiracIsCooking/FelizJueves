"""Fused HGP `F` against split MMD `J` + `K`, per class, on polyglycine.

    devtools/measure.sh scheme-ab [--backend cuda|hip]   (FJ_SCHEME_AB_STORE)

THE QUESTION. [Fock] times two whole Fock builds that differ in three ways
at once -- the scheme (fused, full 8-fold symmetry, a six-block density
screen: opt-Brc; or split `J` and `K` over every ordered quartet, a
one-block screen: opt-UM), the recurrence (HGP or MMD) and the batching --
and finds opt-Brc faster below ~3,000 basis functions on double-zeta sets
([Fock, §6.1]). It times no class against another. Is the fused HGP `F`
faster than MMD's `J` + `K` for the low-angular-momentum classes that
dominate a build ([Fock, §6.4.2]: the six s/p classes are over 60% of
opt-Brc's time on (H2O)_100/cc-pVDZ)? This sweep asks it for (ss|ss) and
(pp|pp), on [Fock, §6]'s own polyglycines, gly_n = H(C2H3NO)_n OH, n = 5,
10, 15 in Cartesian 6-31G (223, 433, 643 AOs). It does not separate the
scheme from the recurrence: that is the HGP split cell's arm (milestone 14),
which `measured_store` does not admit yet (`HGP_JK_REFUSED`).

ONE CLASS, BY ITS SHELLS. 6-31G is s and p only, so the (ss|ss) part of
`G = J - K/2` is exactly `G` of the sub-basis of the s shells (and the
density's s-s block), and (pp|pp)'s exactly that of the p shells: every
quartet of the sub-basis is of the class, and every quartet of the class is
in it. Each arm builds its own launch plan of that sub-basis, so each
scheme's screen sees exactly the class's quartets and density. (`kb`'s
block factor is `max|D|` of the sub-density, not of the molecule's: the
one place the restriction changes a screen input, recorded, not hidden.)

THE ARMS, one gated process each, every timed region GPU events around the
arm's launches (`timed_region` in the store):

    f            kernel_f literal, launch_f's plan (eq. 29, `N_MAG`
                 magnitudes), recorded by `fock_driver.record_plan`
    f_block_fab  the same launches with `block-fab` (#351: one atomic per
                 F element per block, fp64, exact) -- the cheapest exact
                 lever, because the literal F atomic flush is 2-4x of the
                 class's time in kernels.json; reported, never the verdict
    dq           MMD J step 2: every ket pair class's `Target.DQ` launch
    j            MMD J steps 3-4: every `[p|q]` launch (literal, one
                 batch) and every `JAB` launch, on #157's chain driver, the
                 plan keyed on the DQ kernel's own `Dmax`
    k            MMD K: kernel_kblock (#481, opt-UM's mapping [Fock, §4.3])
                 over `kb_launches`' plan, max|D| per block

    mmd = dq + j + k, summed per timed run (run r of each process; the
    processes run back to back within a round).

THE SAME tau FOR EVERY ARM, `TAU` (each scheme's own screen against it).

CORRECTNESS FIRST, on the timed binaries' own outputs. Per cell: the card's
closed `F` against the card's `J - K/2` (two engines, two recurrences, no
host reference: walk_f and walk_kb are out of reach at this size), within
`screen_bound` -- the two schemes skip different quartets, each skipped
contribution under `tau` -- recorded as measured. Once, at gly5 and
`tau = 0` (nothing skipped by either), the same comparison within
`CHECK_TOL` of `max|G|`: the arithmetic agrees before anything is timed.

THE VERDICT, DECLARED BEFORE ANY ROW (`DECLARED`): per (molecule, class,
card) `fused` when the literal `f`'s third quartile is under `mmd`'s first,
`split` when `mmd`'s third quartile is under `f`'s first, else `unresolved`.
The hypothesis -- "fused HGP F is faster than MMD J + K through (pp|pp)" --
holds on a card when every one of its six cells reads `fused`. Below
[Fock]'s ~3,000-function crossover throughout, so `fused` is expected; the
trend in n (the ratio `mmd / f`) is the informative part.

NOT A CORRECTNESS TEST and not a gate: measure/, behind the card gate,
alone. Its inputs are untracked (`oracle.polyglycine`: a 643-AO density is
~9 MB of JSON) and every row pins the file it read by sha256 and E_RHF.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path

import numpy as np
import pytest
from fock_driver import fock_program, fock_stage, record_plan, run_on_card
from kblock_driver import class_binaries, run_classes, timings
from oracle import polyglycine
from plan_driver import (
    RUNS,
    compile_checked,
    dq_classes,
    dq_program,
    dq_read_back,
    dq_stage,
    driver,
    read_back_j,
    reps,
    stage,
)
from test_fock_build import BOUND as F_BOUND
from test_j_chain_ondevice import _ket_data, _rows_of
from test_j_launched_ondevice import BOUND as J_BOUND
from test_kernel_device import BACKEND, _toolchain

from felizjueves.ao.density import epair
from felizjueves.ao.drive import _chain_plan, _unique_pairs
from felizjueves.ao.shell import Basis
from felizjueves.eri.fock import kernel_f as KF
from felizjueves.eri.fock.launch_f import f_launches
from felizjueves.eri.fock.levers import block_fab
from felizjueves.eri.k_exchange.k import record_bounds
from felizjueves.eri.k_exchange.kernel_kblock import build_kblock_kernel
from felizjueves.eri.k_exchange.launch_kblock import kb_launches
from felizjueves.launch import bound_value
from felizjueves.performance import gate, measured
from felizjueves.performance.gate import GPU_BUSY_MAX
from felizjueves.reference.pq import pq

STORE = "FJ_SCHEME_AB_STORE"
SCHEMA = 1
MOLECULES = (5, 10, 15)
CLASSES = {"ss": 0, "pp": 1}  # the class -> the shells' l
TAU = 1e-10
N_MAG = 4  # launch_f's magnitudes: test_fock_ondevice's whole-molecule n
CHECK_TOL = 1e-12  # gly5, tau = 0: of max|G| (test_ao_fock's libcint tol)
CHECK_N = 5
ROUNDS, WARMUP, REPEATS, RETAKES = 3, 3, 11, 3
ARMS = ("f", "f_block_fab", "dq", "j", "k")
MMD = ("dq", "j", "k")

DECLARED = {
    "issue": "measure-mmd-vs-hgp",
    "hypothesis": "fused HGP F (kernel_f, literal) is faster than MMD J + K "
                  "(DQ + [p|q] + JAB, literal, one batch; kernel_kblock) for "
                  "(ss|ss) and (pp|pp) on gly5, gly10 and gly15 / 6-31G",
    "rule": "per (molecule, class, card): 'fused' when q3(f) < q1(mmd), "
            "'split' when q3(mmd) < q1(f), else 'unresolved'; the hypothesis "
            "holds on a card when all six cells read 'fused'. f_block_fab is "
            "reported beside it and decides nothing.",
    "expected": "fused everywhere: every molecule is under [Fock, §6.1]'s "
                "~3,000-function crossover; the ratio mmd/f across n is the "
                "reading",
    "declared": "2026-10-23, before any row",
}
TIMED_REGION = {
    "f": "GPU events around every F launch; the zeroing of F and a "
         "synchronize outside (fock_driver's timing mode)",
    "f_block_fab": "as f",
    "dq": "GPU events around every DQ launch; the outputs' poisoning "
          "outside (plan_driver.DQ_PROGRAM's timing mode)",
    "j": "GPU events from before the J_p and J_mn memsets to after the last "
         "JAB launch (plan_driver.DRIVER, chain=)",
    "k": "GPU events from before K's memset to after the last kb launch "
         "(kblock_driver, mode kb)",
    "outside_every_arm": "uploads, plan building, the D_q read-back and "
                         "the host sort between DQ and J, F's closing F + F^T",
}


def _stats(xs) -> dict[str, float]:
    q1, med, q3 = (float(x) for x in np.percentile(xs, [25, 50, 75]))
    return {"median_ns": round(med), "q1_ns": round(q1), "q3_ns": round(q3),
            "n": len(xs)}


def verdict(f: dict, mmd: dict) -> str:
    """`DECLARED`'s rule on two `_stats`."""
    if f["q3_ns"] < mmd["q1_ns"]:
        return "fused"
    if mmd["q3_ns"] < f["q1_ns"]:
        return "split"
    return "unresolved"


def screen_bound(n_shells: int, tau: float) -> float:
    """What two screens that each skip contributions under `tau` may leave
    between their `G`s, per element: every shell pair's skipped quartets
    under `tau` on each side, `J`'s and `K/2`'s and `F`'s (3 terms), the
    shell pairs counted once (`n (n + 1) / 2`). A bug-catcher with a
    stated reach, not a proved bound; the measured value is recorded."""
    return 3.0 * n_shells * (n_shells + 1) / 2.0 * tau


@cache
def sub(n: int, l: int):  # noqa: E741
    """gly{n}'s shells of angular momentum `l`, the density's block on them,
    and the cache file's metadata."""
    inp = polyglycine.load(n)
    idx = [i for i, s in enumerate(inp.basis.shells) if s.l == l]
    off = inp.basis.offsets
    ao = np.concatenate([np.arange(off[i], off[i] + inp.basis.shells[i].n_cart)
                         for i in idx])
    basis = Basis(tuple(inp.basis.shells[i] for i in idx))
    D = np.ascontiguousarray(inp.density[np.ix_(ao, ao)])
    return basis, D, {"sha256": inp.sha256, "e_rhf": inp.meta["e_rhf"],
                      "nao": inp.meta["nao"], "pyscf": inp.meta["pyscf"],
                      "geometry_sha256": inp.meta["geometry_sha256"]}


def _plain(binary: Path, *args: str, vram: str | None = None) -> str:
    """A runner for an already-probed, gated binary (no VRAM probe)."""
    done = subprocess.run([str(binary), *args], capture_output=True,
                          text=True, timeout=1800)
    assert done.returncode == 0, (f"{binary.name} exited {done.returncode}\n"
                                  f"{done.stdout}\n{done.stderr}")
    return done.stdout


@dataclass
class Cell:
    """One (molecule, class, tau)'s compiled arms, staged."""

    n: int
    cls: str
    tau: float
    basis: Basis
    D: np.ndarray
    f: dict  # arm -> (binary, data, recs)
    dq: tuple  # (binary, staged, classes)
    j: tuple  # (binary, staged, drv, plan)
    k: list  # kblock ClassBinary
    wd: Path


def build(tc, backend: str, n: int, cls: str, tau: float, run,
          wd: Path) -> Cell:
    """Plan, record, compile and stage every arm of one cell. The DQ step
    runs once here (correctness mode): the J plan is keyed on its `Dmax`."""
    basis, D, _ = sub(n, CLASSES[cls])
    wd.mkdir(parents=True, exist_ok=True)

    fplan = f_launches(basis, D, bound=F_BOUND, tau=tau, n=N_MAG)
    f = {}
    for arm, levers in (("f", ()), ("f_block_fab", (block_fab.NAME,))):
        recs = record_plan(fplan, basis, D, levers=levers)
        kernels = list({r.name: r.kernel for r in recs}.values())
        prog = fock_program(kernels, backend, wd / arm, name=arm)
        binary = compile_checked(tc, prog, wd / f"{arm}_bin")
        f[arm] = (binary, fock_stage(recs, kernels), recs)

    classes = dq_classes(basis, D)
    dqp = dq_program(classes, backend, wd / "dq")
    dq_bin = compile_checked(tc, dqp, wd / "dq_bin")
    dq_in = wd / "dq.in"
    dq_stage(classes).tofile(dq_in)
    run(dq_bin, str(dq_in), str(wd / "dq.out"), vram=backend)
    dq_runs = dq_read_back(classes, np.fromfile(wd / "dq.out"))
    rows = _rows_of(classes, dq_runs[0])
    pairs = _unique_pairs(basis)
    gs = {(a, b): [bound_value(J_BOUND, zeta,
                               basis.shells[a].l + basis.shells[b].l)
                   for _, zeta, _ in epair(basis.shells[a],
                                           basis.shells[b]).recs]
          for a, b in pairs}
    jplan = _chain_plan(basis, pairs, gs, rows, n_batches=1, tau=tau,
                        strict=False, resolve=None)
    drv = driver(jplan, backend, wd / "jgen", chain=basis)
    j_bin = compile_checked(tc, drv, wd / "j_bin")
    j_in = wd / "j.in"
    stage(jplan, basis, D, drv, ket_data=_ket_data(basis, rows)).tofile(j_in)

    kplan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq),
                        tau=tau, build=build_kblock_kernel)
    assert [c.am for c in kplan] == [(CLASSES[cls],) * 4], kplan
    parts = class_binaries(tc, kplan, basis, D, backend, wd / "k")
    return Cell(n, cls, tau, basis, D, f, (dq_bin, dq_in, classes),
                (j_bin, j_in, drv, jplan), parts, wd)


def run_arm(c: Cell, arm: str, backend: str, run, *, warmup: int = 0,
            reps_: int = RUNS):
    """One arm's process: `(output matrix of its first run, ns per timed
    run)`. warmup = 0 and reps_ = RUNS is the correctness mode's count."""
    timing = (warmup, reps_)
    if arm in ("f", "f_block_fab"):
        binary, data, _ = c.f[arm]
        F, out = run_on_card(binary, data, c.basis.n_ao, c.wd, run,
                             backend=backend, timing=timing, tag=arm)
        _, ev = _f_reps(out)
        return np.array(KF.close(F[0].tolist())), ev
    if arm == "dq":
        binary, staged, _ = c.dq
        out = run(binary, str(staged), str(c.wd / "dqt.out"),
                  str(warmup), str(reps_), vram=backend)
        return None, _dq_reps(out)
    if arm == "j":
        binary, staged, drv, _ = c.j
        dst = c.wd / "j.out"
        out = run(binary, str(staged), str(dst), str(warmup), str(reps_),
                  vram=backend)
        J = read_back_j(drv, np.fromfile(dst, dtype=np.float64))[0]
        return J, [e for *_, e in reps(out)]
    if arm == "k":
        Ks, outs, _ = run_classes(c.k, c.basis, "kb", run, backend,
                                  warmup=warmup, reps=reps_)
        (out,) = outs
        return Ks[0], timings(out)["total"]
    raise ValueError(arm)


def _f_reps(out: str):
    from fock_driver import reps as f_reps

    return f_reps(out)


def _dq_reps(out: str) -> list[float]:
    got = [line.split() for line in out.splitlines()
           if line.startswith("rep ") and " dq events_ns " in line]
    assert [int(g[1]) for g in got] == list(range(len(got))), out
    return [float(g[4]) for g in got]


def agreement(c: Cell, backend: str, run) -> dict:
    """The card's `F` against the card's `J - K/2`, both closed and full."""
    F, _ = run_arm(c, "f", backend, run, reps_=2)
    Fb, _ = run_arm(c, "f_block_fab", backend, run, reps_=2)
    J, _ = run_arm(c, "j", backend, run, reps_=2)
    K, _ = run_arm(c, "k", backend, run, reps_=2)
    G = J - 0.5 * K
    for name, x in (("F", F), ("F block-fab", Fb), ("J", J), ("K", K)):
        assert np.all(np.isfinite(x)), f"non-finite {name}"
    scale = float(np.max(np.abs(G)))
    return {"max_abs_G": scale,
            "f_vs_mmd_abs": float(np.max(np.abs(F - G))),
            "f_block_fab_vs_f_abs": float(np.max(np.abs(Fb - F))),
            "max_abs_J": float(np.max(np.abs(J))),
            "max_abs_K": float(np.max(np.abs(K)))}


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "about": __doc__.split("\n\n")[1],
            "declared": DECLARED, "timed_region": TIMED_REGION, "rows": []}


def _resources(backend: str, kernels: dict, workdir: Path) -> dict:
    """Registers and spills of each kernel, keyed by `arm:name` (measured's
    parsers): the two F arms emit kernels of the same name."""
    b = measured.BACKENDS[backend]
    emit = BACKEND[backend][0]
    out = {}
    for label, k in kernels.items():
        unit = measured.Unit(k.name, emit(k, b.arch), 0)
        wd = workdir / label.replace(":", "_")
        wd.mkdir(parents=True, exist_ok=True)
        raw, _, _ = measured.compile_unit(b, unit, compiler=b.compiler(),
                                          workdir=wd)
        regs, store, load, _ = measured._numbers(b, raw)
        out[label] = {"registers": regs, "spill_store": store,
                      "spill_load": load}
    return out


def kernels_of(c: Cell) -> dict:
    """Every distinct kernel a cell launches, `arm:name` -> Kernel."""
    out = {}
    for arm in ("f", "f_block_fab"):
        for r in c.f[arm][2]:
            out[f"{arm}:{r.name}"] = r.kernel
    for cl in c.j[3]:
        for k in cl.kernels:
            out[f"j:{k.name}"] = k
    for p in c.k:
        out[f"k:{p.drv.classes[0].kb.name}"] = p.drv.classes[0].kb
    return out


def _timed(c: Cell, backend: str) -> tuple[dict, list, int]:
    """ROUNDS rounds, every arm gated in each: per arm the timed runs, and
    per round the MMD runs summed index by index."""
    per = {a: [] for a in ARMS}
    mmd: list[float] = []
    readings, discarded = [], 0
    for rnd in range(ROUNDS):
        this = {}
        for arm in ARMS:
            for take in range(RETAKES + 1):
                try:
                    busy, util = gate.wait_quiet(
                        backend, say=lambda s: print(s, flush=True))
                except gate.Contended as e:
                    discarded += 1
                    print(f"  DISCARDED {arm}: {e} (take {take + 1})")
                    continue
                _, ns = run_arm(c, arm, backend, _plain, warmup=WARMUP,
                                reps_=REPEATS)
                after = gate.after_reading(backend)
                if util is not None and after is not None \
                        and after <= GPU_BUSY_MAX:
                    break
                discarded += 1
                print(f"  DISCARDED {arm}: card {util} -> {after}%")
            else:
                pytest.fail(f"{arm}: contended through {RETAKES + 1} takes "
                            "-- not recorded")
            assert len(ns) == REPEATS, (arm, ns)
            this[arm] = list(map(float, ns))
            per[arm] += this[arm]
            readings.append({"round": rnd, "arm": arm,
                             "cpu_busy_permille": round(1000 * busy),
                             "gpu_util_before": util,
                             "gpu_util_after": after})
        mmd += list(np.sum([this[a] for a in MMD], axis=0))
    return {**{a: _stats(per[a]) for a in ARMS}, "mmd": _stats(mmd)}, \
        readings, discarded


def _measure(backend: str, request, run_device_binary, tmp_path: Path):
    store_path = os.environ.get(STORE)
    if not store_path:
        pytest.skip(f"the scheme A/B clock is opt-in and must run alone: set "
                    f"{STORE} (devtools/measure.sh scheme-ab)")
    try:
        for n in MOLECULES:
            polyglycine.load(n)
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(f"the polyglycine inputs are not written: {e}")
    tc = _toolchain(backend, request)
    b = measured.BACKENDS[backend]
    compiler = b.compiler()
    run = run_device_binary

    # the arithmetic first: gly5 at tau = 0, nothing skipped by either side
    for cls in CLASSES:
        c0 = build(tc, backend, CHECK_N, cls, 0.0, run,
                   tmp_path / f"check_{cls}")
        a = agreement(c0, backend, run)
        rel = a["f_vs_mmd_abs"] / a["max_abs_G"]
        print(f"\n{backend} gly{CHECK_N} ({cls}|{cls}) tau=0: |F - (J - K/2)|"
              f" / max|G| = {rel:.2e} (tol {CHECK_TOL:g})", flush=True)
        assert rel <= CHECK_TOL, f"({cls}|{cls}): F vs J - K/2 at tau = 0"

    path = Path(store_path)
    for n in MOLECULES:
        for cls in CLASSES:
            c = build(tc, backend, n, cls, TAU, run, tmp_path / f"g{n}_{cls}")
            _, _, inputs = sub(n, CLASSES[cls])
            a = agreement(c, backend, run)
            bound = screen_bound(len(c.basis.shells), TAU)
            assert a["f_vs_mmd_abs"] <= bound, (
                f"gly{n} ({cls}|{cls}): |F - (J - K/2)| "
                f"{a['f_vs_mmd_abs']:.3e} over the screens' reach {bound:.3e}")
            stats, readings, discarded = _timed(c, backend)
            work = c.wd / "res"
            work.mkdir()
            row = {
                "arch": b.arch.name, "toolchain": b.toolchain(compiler),
                "gpu": b.gpu(), "host": measured.host(),
                "date": datetime.date.today().isoformat(),
                "molecule": f"gly{n}", "basis": "6-31g (cartesian)",
                "class": f"({cls}|{cls})", "tau": TAU, "n_mag": N_MAG,
                "inputs": inputs, "n_shells": len(c.basis.shells),
                "n_ao": c.basis.n_ao,
                "counts": {
                    "f_launches": len(c.f["f"][2]),
                    "dq_launches": len(c.dq[2]),
                    "pq_launches": len(c.j[2].launches),
                    "jab_launches": len(c.j[2].jab),
                    "kb_blocks": sum(len(cl.blocks)
                                     for p in c.k for cl in p.plan),
                },
                "agreement": {**a, "screen_bound_abs": bound},
                "rounds": ROUNDS, "warmup": WARMUP, "repeats": REPEATS,
                "gate": readings, "gate_discarded_runs": discarded,
                "arms": stats,
                "mmd_over_f": stats["mmd"]["median_ns"]
                / stats["f"]["median_ns"],
                "mmd_over_f_block_fab": stats["mmd"]["median_ns"]
                / stats["f_block_fab"]["median_ns"],
                "verdict": verdict(stats["f"], stats["mmd"]),
                "resources": _resources(backend, kernels_of(c), work),
            }
            store = _load(path)
            key = (row["arch"], row["toolchain"], row["molecule"],
                   row["class"])
            store["rows"] = [r for r in store["rows"]
                             if (r["arch"], r["toolchain"], r["molecule"],
                                 r["class"]) != key]
            store["rows"].append(row)
            store["rows"].sort(key=lambda r: (r["arch"], r["toolchain"],
                                              r["class"],
                                              int(r["molecule"][3:])))
            path.write_text(json.dumps(store, indent=1) + "\n")
            s = stats
            print(f"{backend} gly{n} ({cls}|{cls}) {c.basis.n_ao} AOs: "
                  f"f {s['f']['median_ns'] / 1e6:.3f} ms, f_block_fab "
                  f"{s['f_block_fab']['median_ns'] / 1e6:.3f}, mmd "
                  f"{s['mmd']['median_ns'] / 1e6:.3f} (dq "
                  f"{s['dq']['median_ns'] / 1e6:.3f} j "
                  f"{s['j']['median_ns'] / 1e6:.3f} k "
                  f"{s['k']['median_ns'] / 1e6:.3f}); mmd/f "
                  f"{row['mmd_over_f']:.2f} -> {row['verdict']}; "
                  f"|F - G| {a['f_vs_mmd_abs']:.2e}", flush=True)


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_scheme_ab_clock(request, run_device_binary, tmp_path) -> None:
    _measure("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_scheme_ab_clock(request, run_device_binary, tmp_path) -> None:
    _measure("hip", request, run_device_binary, tmp_path)


def test_scheme_ab_store_reads_as_declared() -> None:
    """`devtools/measure.sh check`: the tracked store's header is `DECLARED`
    and `TIMED_REGION`, and every row's verdict and ratios are what its own
    arms recompute to. No card, no compiler."""
    path = Path(__file__).resolve().parents[1] / "data/measured/scheme_ab.json"
    if not path.is_file():
        pytest.skip(f"{path.name} is not written")
    store = json.loads(path.read_text())
    assert store["declared"] == DECLARED
    assert store["timed_region"] == TIMED_REGION
    assert store["rows"], "an empty store"
    for r in store["rows"]:
        a = r["arms"]
        assert set(a) == {*ARMS, "mmd"}
        assert r["verdict"] == verdict(a["f"], a["mmd"]), r["molecule"]
        assert r["mmd_over_f"] == a["mmd"]["median_ns"] / a["f"]["median_ns"]
        assert r["mmd_over_f_block_fab"] == (a["mmd"]["median_ns"]
                                             / a["f_block_fab"]["median_ns"])
        assert r["tau"] == TAU and r["n_mag"] == N_MAG
        assert (r["agreement"]["f_vs_mmd_abs"]
                <= r["agreement"]["screen_bound_abs"])
        assert a["mmd"]["n"] == ROUNDS * REPEATS == a["f"]["n"]
