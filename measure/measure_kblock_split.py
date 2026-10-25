"""The split block K kernel's compiler figures beside the block kernel's (#548).

    devtools/measure.sh kblock-split [--backend cuda|hip]

OPT-IN: each test skips unless FJ_KBLOCK_SPLIT_STORE names the store
(measure.sh sets it to config.sh's MEASURE_KBLOCK_SPLIT_STORE, the tracked
data/measured/kblock_split.json). A COMPILER, NO CARD: nothing here
launches, so nothing here is a clock. It writes one row per (arch,
toolchain).

WHAT IS MEASURED. Per class of `CLASSES`, per granularity of
`kernel_kblock_split` (`block`, `column`, `element`) and per stack `R` of
`STACKS`, every kernel of the split set compiled as its own unit with the
measured store's flags and parsers (`measured.compile_unit`): registers and
spill (nvcc `sm_86`, `-Xptxas -v`, bytes; hipcc `gfx1200`, VGPRs and
spilled VGPRs, a count), and per set the WORST kernel (most spill, then most
registers) reported beside `kernel_kblock`'s own kernel (`kb`). `block` at
`R = 1` is the block kernel's extent with the stack's parameter shapes, so
it is `kb`'s control.

`CLASSES` is `(dd|dd)` only, by the user's decision for #548 (the issue
asks for every class through `(dd|dd)`; widen `CLASSES` to take them).

THE VERDICT RULE is the store's `declared`, written before any row: this
trades arithmetic for shared memory, so a per-class SPILL win is expected,
a per-class CLOCK win is not, and the row that decides is the aggregate at
`R > 1`, which needs the card A/B (#548's follow-ups) and is not here.
"""

from __future__ import annotations

import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pytest
from test_kernel_device import BACKEND, _toolchain

from felizjueves.eri.k_exchange.kernel_kblock import build_kblock_kernel
from felizjueves.eri.k_exchange.kernel_kblock_split import (
    GRANULARITIES,
    accumulators_per_density,
    build_kblock_split_kernels,
    shared_bytes,
)
from felizjueves.performance import measured

STORE = "FJ_KBLOCK_SPLIT_STORE"
SCHEMA = 1
CLASSES = ((2, 2, 2, 2),)
STACKS = (1, 4)  # 4: the block kernel's largest stack at wave32

DECLARED = {
    "issue": 548,
    "hypothesis": "splitting kernel_kblock's K_AB accumulators per column "
                  "or per element relieves register pressure: the worst "
                  "kernel of a split set spills less than the block kernel",
    "rule": "per (class, granularity, R, arch): 'spill win' when the worst "
            "split kernel's spill_store is below the block kernel's (kb, "
            "R = 1) and below the same granularity's block set at that R, "
            "'no win' otherwise. A spill win is NOT a clock win: the split "
            "recomputes the recurrence and the E blocks per kernel, so a "
            "per-class clock win is not expected at R = 1, and the row that "
            "decides the split is the aggregate wall-clock at R > 1 against "
            "R block-kernel sweeps, measured on a card (#548's follow-ups), "
            "not here",
    "expected": "spill win for column and element at R = 1 on sm_86; the "
                "split's register count stays at the ceiling (the "
                "recurrence's working set is unchanged)",
    "declared": "2026-10-25, before any row",
}


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "issue": 548,
            "about": __doc__.split("\n\n")[1], "declared": DECLARED,
            "rows": []}


def _resources(backend: str, kern, workdir: Path) -> dict[str, int]:
    b = measured.BACKENDS[backend]
    emit = BACKEND[backend][0]
    unit = measured.Unit(kern.name, emit(kern, b.arch), 0)
    raw, secs, _ = measured.compile_unit(b, unit, compiler=b.compiler(),
                                         workdir=workdir)
    regs, store, load, smem = measured._numbers(b, raw)
    return {"kernel": kern.name, "registers": regs, "spill_store": store,
            "spill_load": load, "compile_s": round(secs, 1)}


def _worst(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: (r["spill_store"], r["registers"]))


def verdict(worst: dict, kb: dict, block_at_r: dict) -> str:
    """`DECLARED`'s rule."""
    win = (worst["spill_store"] < kb["spill_store"]
           and worst["spill_store"] < block_at_r["spill_store"])
    return "spill win" if win else "no win"


def _measure(backend: str, request, tmp_path: Path) -> None:
    store_path = os.environ.get(STORE)
    if not store_path:
        pytest.skip(f"the split's compile figures are opt-in: set {STORE} "
                    "(devtools/measure.sh kblock-split)")
    _toolchain(backend, request)
    b = measured.BACKENDS[backend]
    lanes = b.arch.wavefront
    jobs = max(1, (os.cpu_count() or 2) // 2)
    classes = []
    for am in CLASSES:
        sets = {(g, R): build_kblock_split_kernels(am, g, R=R, lanes=lanes)
                for g in GRANULARITIES for R in STACKS}
        kerns = [build_kblock_kernel(am)] + [k for ks in sets.values()
                                             for k in ks]
        work = tmp_path / "{}{}{}{}".format(*am)
        work.mkdir()
        with ThreadPoolExecutor(jobs) as pool:
            res = list(pool.map(partial(_resources, backend, workdir=work),
                                kerns))
        by = {r["kernel"]: r for r in res}
        kb = by[kerns[0].name]
        entry = {"am": list(am), "L": sum(am), "kb": kb, "sets": []}
        for (g, R), ks in sets.items():
            rows = [by[k.name] for k in ks]
            worst = _worst(rows)
            block_at_r = _worst([by[k.name] for k in sets["block", R]])
            acc = accumulators_per_density(am, g)
            entry["sets"].append({
                "granularity": g, "R": R, "n_kernels": len(ks),
                "accumulators_per_density": acc,
                "shared_bytes": shared_bytes(am, acc, R, lanes),
                "worst": worst,
                "spill_store_sum": sum(r["spill_store"] for r in rows),
                "verdict": verdict(worst, kb, block_at_r),
                "kernels": rows,
            })
        classes.append(entry)
    compiler = b.compiler()
    row = {"arch": b.arch.name, "toolchain": b.toolchain(compiler),
           "host": measured.host(), "date": datetime.date.today().isoformat(),
           "lanes": lanes, "spill_unit": (measured.NV_SPILL_UNIT
                                          if backend == "cuda"
                                          else measured.AMD_SPILL_UNIT),
           "classes": classes}
    path = Path(store_path)
    store = _load(path)
    store["rows"] = [r for r in store["rows"]
                     if (r["arch"], r["toolchain"]) != (row["arch"],
                                                        row["toolchain"])]
    store["rows"].append(row)
    store["rows"].sort(key=lambda r: (r["arch"], r["toolchain"]))
    path.write_text(json.dumps(store, indent=1) + "\n")
    print(f"\n{backend} {b.arch.name} -> {path}")
    for c in classes:
        kb = c["kb"]
        print("  ({}{}|{}{})".format(*c["am"]) + f" kb r{kb['registers']} "
              f"s{kb['spill_store']}/{kb['spill_load']}")
        for s in c["sets"]:
            w = s["worst"]
            print(f"    {s['granularity']:>7} R={s['R']}: {s['n_kernels']} "
                  f"kernels, worst {w['kernel']} r{w['registers']} "
                  f"s{w['spill_store']}/{w['spill_load']}, "
                  f"{s['shared_bytes']} B smem -- {s['verdict']}")


@pytest.mark.cuda
def test_cuda_kblock_split_compile(request, tmp_path) -> None:
    _measure("cuda", request, tmp_path)


@pytest.mark.hip
def test_hip_kblock_split_compile(request, tmp_path) -> None:
    _measure("hip", request, tmp_path)
