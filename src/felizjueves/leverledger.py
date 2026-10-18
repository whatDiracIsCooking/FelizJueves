"""The lever ledger: per-class register and spill deltas, both cards (#362).

    python -m felizjueves.leverledger            # the ledger, as JSON
    python -m felizjueves.leverledger --write    # rewrite LEDGER
    python -m felizjueves.leverledger --table    # one line per row

(with `PYTHONPATH=src uv run` in front, as for every entry point here.) The
rows it reads are written by `python -m felizjueves.measured --remeasure
--ledger`, which also rewrites LEDGER.

WHAT IT IS. For every `(class, arm, arch)` -- a class of J, K or F at total
L <= `measured_store.LEDGER_MAX_L`, an arm of `measured_store.ledger_arms()`
legal on its target, an arch of `device.ARCHS` -- the compiler's reading of
the arm's emitted unit(s) beside the LITERAL reconstruction of the same
class under the same toolchain, and the difference, SIGN INCLUDED: nvcc's
registers per thread, spill store and load BYTES and stack frame; hipcc's
VGPRs, spilled REGISTER COUNTS (VGPR + SGPR, one number in both spill
fields) and ScratchSize; the static shared bytes / LDS either reports; and
the unit count (a lever such as K's `split` emits several kernels per
class: `registers` is then their maximum, the spills their sum, as every
multi-unit row of the store is aggregated). Compare within one arch only:
the two toolchains' units differ.

THE BASELINE. Every row's `delta` is against the literal class -- the
store's row for `arm_spec.without the levers`, same arch, same toolchain.
An arm that is a COMPOSITION (a lever that only builds with another, e.g.
`batched-vrr+sliced-hrr`, or a declared one, `interleave+split`) is also
read against each of its proper sub-arms the ledger holds, under `vs`
(`vs["batched-vrr"]` is sliced-hrr's own increment on top of #350's
staging). The row says which baseline each delta is against.

A REPORT, NOT A RATCHET (the issue's guardrail). There is no tolerance, no
floor and no threshold: a regression (a positive delta), a null (no
change) and a field one compiler did not report (None, so its delta is
None) are all rows, never filtered. `flips` lists every (class, arm, field)
whose delta has OPPOSITE signs on the two arches -- a first-class result,
not an error.

A MISSING ROW FAILS. `ledger` raises `MissingRows` naming every lever,
arm, class and arch without a row (or without the literal row it is read
against): a vocabulary lever with no rows is never silently absent. A
regression never fails it.

ROWS COME FROM THE STORE (`measured_store`), never from a model, a file the
generator wrote or a compiler: measure/measure_lever_ledger.py recomputes LEDGER
from data/measured/kernels.json with no device and no toolchain, and
test/test_layering.py holds this module to the store and `device`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from felizjueves import measured_store as store
from felizjueves.device import ARCHS, MeasuredKey, by_name
from felizjueves.spec import Spec

LEDGER = store.REPO / "data" / "measured" / "lever_ledger.json"
SCHEMA = 1

FIELDS = ("registers", "spill_store", "spill_load", "scratch",
          "shared_bytes", "units")
UNITS = {
    "registers": "nvcc: registers per thread (ptxas -v); hipcc: VGPRs per "
                 "lane (kernel-resource-usage). Over a multi-unit class, the "
                 "maximum.",
    "spill_store": "nvcc: spill store BYTES; hipcc: spilled REGISTER COUNT "
                   "(VGPRs Spill + SGPRs Spill). Over units, the sum.",
    "spill_load": "nvcc: spill load BYTES; hipcc: the same count as "
                  "spill_store (no store/load split). Over units, the sum.",
    "scratch": "nvcc: stack frame bytes (raw stack_frame_bytes); hipcc: "
               "ScratchSize bytes. Over units, the maximum; None when the "
               "row's raw does not carry it.",
    "shared_bytes": "static shared bytes (nvcc smem) / LDS Size (hipcc); "
                    "None when the compiler printed none.",
    "units": "the class's emitted translation units (kernels).",
}
_SCRATCH = {"sm_86": "stack_frame_bytes", "gfx1200": "ScratchSize"}

ABOUT = (
    "The lever ledger (felizjueves.leverledger, #362): per (class, arm, "
    "arch), the compiler's registers, spills, scratch and shared bytes of "
    "the arm's emitted units beside the LITERAL class under the same "
    "toolchain, and the delta (arm - literal), sign included. Classes: J "
    "pq_spec(L), K k_spec(L), and every F ladder class, at total L <= 4. "
    "Arms: every kernel-body lever of levers.VOCABULARY, a lever that only "
    "builds with another as that composition (measured_store.REQUIRES), "
    "and the declared compositions (interleave+split); a composition is "
    "also read against its sub-arms under 'vs'. Compile-only: nvcc -Xptxas "
    "-v and hipcc kernel-resource-usage, no card, no clock. A REPORT, NOT A "
    "RATCHET: no tolerance, no floor, no threshold; regressions, nulls and "
    "the cross-toolchain direction flips ('flips') are first-class rows. "
    "Recomputed from data/measured/kernels.json by "
    "test/test_lever_ledger.py; a vocabulary lever with no rows fails it. "
    "Compare within one arch only. NOT comparable to the predecessor's "
    "ledgers or [JEngine, v2, Table 4]: other kernels.")


class MissingRows(ValueError):
    """A ledger cell has no row (or no literal row to be read against)."""


def _literal(spec: Spec) -> Spec:
    return replace(spec, levers=())


def scratch(row: store.Measured) -> int | None:
    """The row's scratch reading: the maximum of its raw stack frame
    (nvcc) / ScratchSize (hipcc) over its units, or None if absent."""
    name = _SCRATCH.get(row.arch)
    got = [v for k, v in row.raw.items()
           if name and (k == name or k.endswith(":" + name))
           and isinstance(v, int)]
    return max(got) if got else None


def readings(row: store.Measured) -> dict[str, int | None]:
    """The ledger's FIELDS off one stored row."""
    return {"registers": row.registers, "spill_store": row.spill_store,
            "spill_load": row.spill_load, "scratch": scratch(row),
            "shared_bytes": row.shared_bytes,
            "units": len(row.kernel.split(",")) if row.kernel else None}


def _delta(a: Mapping[str, int | None],
           b: Mapping[str, int | None]) -> dict[str, int | None]:
    """`a - b` per field; None where either side is None."""
    return {f: (None if a[f] is None or b[f] is None else a[f] - b[f])
            for f in FIELDS}


def sign(d: int | None) -> str:
    return "null" if d is None else "+" if d > 0 else "-" if d < 0 else "0"


def _by_cell(rows: Mapping[str, store.Measured]
             ) -> dict[tuple[str, str], dict[str, store.Measured]]:
    """`{(spec key, arch): {toolchain: row}}`."""
    out: dict[tuple[str, str], dict[str, store.Measured]] = {}
    for k, r in rows.items():
        s, a, t = store.split_key(k)
        out.setdefault((s, a), {})[t] = r
    return out


def ledger(rows: Mapping[str, store.Measured]) -> dict:
    """The ledger document from the store's rows. Raises `MissingRows`
    naming every ledger cell without a row or without its literal row."""
    cells = _by_cell(rows)
    arms = store.ledger_arms()
    arch_names = tuple(a.name for a in ARCHS)
    out, missing = [], []
    for arm in arms:
        target = store.arm_target(arm)
        subs = [s for s in arms if s != arm
                and set(s.split("+")) < set(arm.split("+"))]
        for cls in store.ledger_classes(target):
            spec = store.lever_arm(cls, arm)
            for arch in arch_names:
                got = cells.get((spec.key(), arch), {})
                if not got:
                    missing.append(f"{arm} {spec.key()} @{arch}: no row")
                    continue
                for tool in sorted(got):
                    lit = cells.get((cls.key(), arch), {}).get(tool)
                    if lit is None:
                        missing.append(f"{arm} {spec.key()} @{arch}/{tool}: "
                                       f"no literal row {cls.key()}")
                        continue
                    mine = readings(got[tool])
                    row = {
                        "target": target.value, "class": cls.key(),
                        "arm": arm, "arch": arch, "toolchain": tool,
                        "key": MeasuredKey(spec, by_name(arch), tool).key(),
                        "baseline": MeasuredKey(cls, by_name(arch),
                                                tool).key(),
                        "spill_unit": got[tool].spill_unit,
                        "arm_readings": mine,
                        "literal_readings": readings(lit),
                        "delta": _delta(mine, readings(lit)),
                    }
                    vs = {}
                    for s in subs:
                        sub = cells.get((store.lever_arm(cls, s).key(),
                                         arch), {}).get(tool)
                        if sub is not None:
                            vs[s] = _delta(mine, readings(sub))
                    if vs:
                        row["vs"] = vs
                    out.append(row)
    if missing:
        levers = sorted({lv for m in missing
                         for lv in m.split(" ", 1)[0].split("+")})
        raise MissingRows(
            f"{len(missing)} lever-ledger cells have no row (levers "
            f"{', '.join(levers)}); record them with python -m "
            "felizjueves.measured --remeasure --ledger:\n"
            + "\n".join(missing))
    return {
        "schema": SCHEMA,
        "about": ABOUT,
        "baseline": "delta = arm - literal: the store's row of the same "
                    "class with no lever, same arch and toolchain ('baseline' "
                    "names its key). 'vs' = arm - each proper sub-arm the "
                    "ledger holds, for a composition.",
        "units": UNITS,
        "max_L": store.LEDGER_MAX_L,
        "arms": {a: {"target": store.arm_target(a).value,
                     "levers": a.split("+"),
                     "issues": [store.lever_issue(lv)
                                for lv in a.split("+")]}
                 for a in arms},
        # REQUIRES entries for levers not yet in the vocabulary (in review)
        "declared_ahead": store.declared_ahead(),
        "rows": out,
        "flips": flips(out),
    }


def flips(rows: Sequence[Mapping]) -> list[dict]:
    """Every (class, arm, field) whose delta is + on one arch and - on the
    other: the toolchains disagree on the lever's direction."""
    seen: dict[tuple[str, str, str], dict[str, str]] = {}
    for r in rows:
        for f in FIELDS:
            seen.setdefault((r["class"], r["arm"], f), {})[r["arch"]] = \
                sign(r["delta"][f])
    return [{"class": c, "arm": a, "field": f, **signs}
            for (c, a, f), signs in seen.items()
            if {"+", "-"} <= set(signs.values())]


def render(doc: Mapping) -> str:
    return json.dumps(doc, indent=2) + "\n"


def write(rows: Mapping[str, store.Measured], path: Path = LEDGER) -> None:
    path.write_text(render(ledger(rows)))


def table(doc: Mapping) -> str:
    lines = []
    for r in doc["rows"]:
        d = r["delta"]
        lines.append(
            f"{r['arm']:28} {r['class']:32} {r['arch']:8} "
            + " ".join(f"{f}={r['arm_readings'][f]}({sign(d[f])}"
                       f"{'' if d[f] in (None, 0) else abs(d[f])})"
                       for f in ("registers", "spill_store", "scratch")))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m felizjueves.leverledger")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--store", type=Path, default=store.STORE)
    ap.add_argument("--out", type=Path, default=LEDGER)
    args = ap.parse_args(argv)
    try:
        doc = ledger(store.load(args.store))
    except (OSError, ValueError) as e:
        print(f"leverledger: {e}", file=sys.stderr)
        return 2
    if args.write:
        args.out.write_text(render(doc))
    elif args.table:
        print(table(doc))
    else:
        sys.stdout.write(render(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
