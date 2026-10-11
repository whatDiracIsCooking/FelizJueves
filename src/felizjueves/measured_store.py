"""The measured store: `Measured` rows, their file, and what is read off them.

OBSERVATIONS OF A TOOLCHAIN, NOT CLAIMS ABOUT THIS CODE. A row is what one
compiler, driven at one arch with one flag set, said about one emitted
translation unit: registers, spill traffic, and where and when it was asked.
A compiler bump moving a number is news about the compiler. Nothing here is
compared against `[JEngine, v2, Table 4]` -- their kernels, their compiler,
their A100; data/literature/jengine/README.md is why that table is not even
transcribed.

KEYED ON `device.MeasuredKey(spec, arch, toolchain).key()`, as #37 decided:
`Spec` is the generation key and has no arch, so an `sm_86` row and a
`gfx1200` row for the same class must be told apart by the arch, and two
compiler versions on one card by the toolchain.

THE RECORD (#50). Every MEASUREMENT field is optional -- a compile-only row has
no runtime, a timing row has no register count -- but a row with all of them
None records nothing and is refused. The PROVENANCE quintet (`toolchain`,
`arch`, `host`, `gpu`, `date`) is required and non-empty, always.

THE STORE is one JSON file, `data/measured/kernels.json`, in the idiom of
`literature/tables.py`: a `"schema"` version and an `"about"` string up top,
and a refusal -- naming the path -- of a file that is malformed, of another
schema, or whose row disagrees with its own key. A missing store is an error,
not an empty one: `load` raises, so "nothing measured yet" (a valid file with
no rows) and "no store here" cannot be confused. One file for both arches,
not one per arch: rows are keyed on the arch already, `remeasure` merges
into what is there (a one-card remeasure diffs only that card's rows), and
the file's path is what the docs map points at.

THE SUBGRID (#52) is `SUBGRID`, the one declared set of `(L, n_batches)`
cells the compile-side measurement sweeps: L in {7, 10, 13, 16} x n_batches
in {1, 3, 6, 9, 12} -- `[JEngine, v2, §7]`'s own reporting range, and 12
past it (#60) because "9 batches suffice" cannot be judged from below 9.
Every cell is a
default-split `pq_spec(L, n_batches=b)` that the plan layer admits (its
partition refuses an empty batch, #64; test/test_measured.py checks every
cell). Tests parametrize over it, #53's wall-clock rows join on it, and a
cell is added here and nowhere else. Widen per experiment, never by default.

WALL-CLOCK (#53) is `runtime_ns`, `repeats` and `spread_ns`, written by the
opt-in, `gpu`-marked sweep in test/test_kernel_device.py
(`test_<backend>_wallclock_over_the_subgrid`, run alone and uncontended; its
docstring is the protocol). What one cell's number IS: the host
`steady_clock` time of the cell's WHOLE launch sequence -- every batch kernel
of `build_pq_kernels(spec)`, in plan order, into one J zeroed beforehand
(outside the clock), then one device synchronize -- producing one `J_p`.
`runtime_ns` is the MEDIAN over `repeats` timed sequences after a warm-up,
and `spread_ns` the interquartile range of those same sequences, on the
sweep's wide-T input; min, max, the quartiles, the per-batch GPU-event
medians, the problem size, the box's load as sampled before the cell, and
the same statistics on a second, near-T input (`wallclock:near:*`) are in
`raw`, keyed `wallclock:...`. The card's state is in it too (#93,
`GPU_UTIL_FIELDS`): its utilisation as the gate let each input through and
just after it, or null -- unknown, never idle -- for a row timed before the
gate recorded it; the loader refuses a timed row without the keys.
The binary is built from the same emitted text with config.sh's flag set
(the report flag aside, which changes no code). Absolute times are NOT
comparable across vendors (gfx1200 rate-limits fp64): only the ranking over
`n_batches` within one arch is a claim.

ONE ROW CARRIES BOTH SIDES. A timing and a compile observation of one cell
share a `MeasuredKey`, so they are one row: `combine` keeps the other side's
measurements (and `raw`) when, and only when, both observed the same text
(`source_sha256`). A changed kernel drops the stale side rather than pairing
a register count with a time it did not produce. The row's `date`, `host`,
`gpu` and `command` are the latest observation's (a timing row records no
`command`, so the compile's is kept).

NINE AGAINST TWELVE (#60) is `nine_vs_twelve`: whether `[JEngine, v2, §7]`'s
"for L <= 16, at most 9 batches are needed" holds on OUR cards. DECLARED
BEFORE ANY 12-BATCH ROW EXISTED, and not to be tuned after reading one. For
one arch and one subgrid L, with r9 and r12 the rows of pq_spec(L, 9) and
pq_spec(L, 12) on that arch:

- SPILL: 12 reduces spill traffic iff  spill(r12) < spill(r9), spill being
  the row's spill_store + spill_load (the cell's whole spill, summed over
  its batch kernels; bytes on sm_86, register counts on gfx1200 -- only the
  within-arch comparison is made). Compiler output, so no tolerance.
- WALL-CLOCK, judged on BOTH of #53's inputs, separately: 12 improves on an
  input iff  median(r9) - median(r12) > max(IQR(r9), IQR(r12)), i.e. it is
  faster by more than the larger recorded spread. WIDE-T is runtime_ns /
  spread_ns; NEAR-T is raw `wallclock:near:median_ns` and q3 - q1. Near-T
  is the informative one (#53: there splitting wins from L = 13 and b9 is
  best at L = 16), but a win on EITHER input counts against 9.
- 9 SUFFICES at (arch, L) iff 12 does none of the three (SPILL, WIDE, NEAR);
  9 SUFFICES ON AN ARCH iff it suffices at every subgrid L. The two arches
  are judged apart and may disagree; sm_86 and gfx1200 are our cards, not
  the paper's A100, and that is the strongest form the claim can take here.

A LEVER A/B (#55) is `lever_ab`: does a kernel-layer lever (today only
`lower.RECYCLE`, `"recycle"`) beat the literal kernel? DECLARED BEFORE ANY
LEVER ROW EXISTED, and not to be tuned after reading one.

- THE CELLS are `lever_cells(lever)`: EVERY `SUBGRID` cell with the lever
  added to `Spec.levers` -- all four L and all five batch counts, on both
  cards, so a sign that flips with L, with n_batches or with the arch is
  seen rather than generalised from one cell. A lever cell is its own key
  (`.../recycle@...`), so the literal rows, and everything that reads them
  (`kernelrank`, `nine_vs_twelve`), are untouched by it.
- THE CONTROL IS TIMED IN THE SAME SESSION. The wall-clock sweep, asked for a
  lever (`FJ_WALLCLOCK_LEVER`), compiles and checks BOTH arms of every cell
  and times them back to back (near-T literal then lever, wide-T lever then
  literal, so neither arm always runs first), writing the lever row with the
  literal arm's same-session statistics in its `raw` as `wallclock:control:*`
  (and `wallclock:control:near:*`). The literal row in the store is NOT
  rewritten: a cross-session comparison would mix the box's drift into the
  effect, and rewriting it would move #54's and #60's recorded verdicts.
- PER INPUT (wide-T and near-T, judged apart): with c the control and r the
  lever, the lever is FASTER iff  median(c) - median(r) > max(IQR(c),
  IQR(r)), SLOWER iff  median(r) - median(c) > that same max, else NEUTRAL --
  #60's rule, symmetric.
- PER CELL: WIN if faster on an input and slower on none; LOSS if slower on
  an input and faster on none; MIXED if both; NEUTRAL if neither.
- THE DEFAULT. The lever may become the default on an arch only if it WINS at
  every cell there; the default (no lever) changes only if that holds on
  BOTH arches, and then in its own commit. Anything else -- including the
  arms disagreeing in sign across L, n_batches or arch, which is itself the
  result -- leaves the lever opt-in. Spill numbers are reported beside the
  verdict (`spill`, literal row against lever row, same compiler) and never
  enter it: the clock is the verdict.

THE SPLIT (#94). This module is the store and nothing else: the record, the
file, the read path, the declared subgrid, and the two verdicts read off the
rows. It knows no toolchain -- it runs no compiler, reads no config.sh,
queries no card and spells no kernel -- so the fast tier, `kernelrank` and
`batchreport` read it without one. The driver that compiles, parses what the
compilers print and fills rows is `felizjueves.measured` (`python -m
felizjueves.measured --remeasure`), which writes INTO this; the wall-clock
side is written by test/test_kernel_device.py's sweep. test/test_layering.py
holds the direction: the store may not import the driver, `emit/` or a
toolchain module, and the analyses may import the store but not the driver.

Nothing under `plan/` or `literature/` may import this module
(test/test_layering.py): a measurement may never feed a literature floor.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from felizjueves.device import Arch, MeasuredKey
from felizjueves.kernel import pq_spec
from felizjueves.lower import LEVERS
from felizjueves.spec import Spec

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "data" / "measured" / "kernels.json"

# THE declared subgrid (#52): see the module docstring. One edit adds a cell.
SUBGRID_LS = (7, 10, 13, 16)
SUBGRID: tuple[Spec, ...] = tuple(
    pq_spec(L, n_batches=b) for L in SUBGRID_LS for b in (1, 3, 6, 9, 12))


def lever_cells(lever: str) -> tuple[Spec, ...]:
    """#55's A/B cells for `lever`: every SUBGRID cell with the lever added
    (module docstring, A LEVER A/B). Only a lowering lever is accepted."""
    if lever not in LEVERS:
        raise ValueError(f"lever {lever!r} is not a kernel-layer lever "
                         f"({', '.join(sorted(LEVERS))})")
    return tuple(replace(s, levers=tuple(sorted({*s.levers, lever})))
                 for s in SUBGRID)


SCHEMA = 2  # 2: a timed row carries GPU_UTIL_FIELDS, a reading or null (#93)

# THE CARD'S STATE AT TIMING (#93). Every row with a `runtime_ns` carries
# these raw keys: the card's utilisation in percent, from its own SMI, as the
# sweep's gate let the input through (`_before`) and just after its last
# timed sequence (`_after`), on each input. `None` (JSON null) means UNKNOWN
# -- not read, or not parseable -- and NEVER idle: every row timed before
# #93 records null here, whatever gate it ran behind.
GPU_UTIL_FIELDS = ("wallclock:gpu_util_before", "wallclock:gpu_util_after",
                   "wallclock:near:gpu_util_before",
                   "wallclock:near:gpu_util_after")

ABOUT = ("Measured rows (felizjueves.measured): what one device toolchain "
         "reported about one emitted translation unit on one card, keyed on "
         "device.MeasuredKey(spec, arch, toolchain).key(). Observations of a "
         "toolchain, not claims about this code. NOT comparable to [JEngine, "
         "v2, Table 4]: their kernels, their compiler, their A100.")

HEADER = {
    "what": "Measured rows: what a device toolchain reported about one "
            "emitted translation unit (felizjueves.measured).",
    "key": "Rows are keyed on felizjueves.device.MeasuredKey(spec, arch, "
           "toolchain).key() -- the generation Spec, the card's Arch, the "
           "compiler as it reports itself (#37).",
    "nature": "Observations of a toolchain, not claims about this code: a "
              "compiler bump moving a number is news about the compiler.",
    "not_compared": "Nothing here is compared against [JEngine, v2, Table 4] "
                    "(their kernels, compiler and A100); see "
                    "data/literature/jengine/README.md.",
    "units": "nvcc: registers per thread, spill BYTES (ptxas -v). hipcc: "
             "VGPRs per lane, spilled REGISTER COUNTS (VGPR + SGPR; no "
             "store/load split, so both spill fields carry it). See each "
             "row's spill_unit and raw.",
    "remeasure": "PYTHONPATH=src uv run python -m felizjueves.measured "
                 "--remeasure --backend both",
    "wallclock": "runtime_ns is the MEDIAN host wall-clock of a cell's whole "
                 "batched launch sequence (every build_pq_kernels batch into "
                 "one zeroed J, then a synchronize), over `repeats` timed "
                 "sequences after a warm-up, on a wide-T input; spread_ns is "
                 "their interquartile range; raw 'wallclock:*' has the rest, "
                 "and 'wallclock:near:*' the same on a near-T input. Taken "
                 "alone on an "
                 "uncontended box by test/test_kernel_device.py's opt-in "
                 "wall-clock sweep (FJ_WALLCLOCK_STORE). Compare only within "
                 "one arch, across n_batches -- never across vendors.",
    "gpu_util": "#93: every row with runtime_ns carries raw "
                "'wallclock:gpu_util_before' / '_after' and "
                "'wallclock:near:gpu_util_before' / '_after' (and, on a lever "
                "row timed after #93, 'wallclock:control:...' likewise): the "
                "card's utilisation in percent from its own SMI, as the "
                "sweep's gate let the input through and 1 s after its last "
                "timed sequence (one read discarded first: rocm-smi's first "
                "read after a pause averages over it, i.e. over our own run). "
                "null means UNKNOWN, never idle: the "
                ".../recycle@... rows were timed before #93 and record null "
                "(their sm_86 rows keep #55's 'wallclock:gpu_util_pct_before_"
                "max', the highest settled reading over the cell's four "
                "timed inputs). The 40 literal SUBGRID rows were re-timed "
                "behind the gate by #93, one session per arch.",
    "levers": "#55: rows keyed .../recycle@... are the recycle arm "
              "(lower.recycle, measured.lever_cells) of every SUBGRID cell, "
              "timed against a same-session literal control kept in their "
              "raw 'wallclock:control:*'; measured.lever_ab is the verdict. "
              "CONCLUSION: recycle stays opt-in. nvcc 13.0 and hipcc / ROCm "
              "7.2.4 emit IDENTICAL machine code for the two arms in every "
              "batch kernel of every cell (both scalarise the statically "
              "indexed slot array), registers and spills match exactly, and "
              "the clock is NEUTRAL at every cell on sm_86; on gfx1200 the "
              "declared rule reads 4 cells at L <= 10 as SLOWER, on that "
              "identical code -- the A/B's noise floor, not the lever.",
}


MEASUREMENTS = ("registers", "spill_store", "spill_load", "shared_bytes",
                "compile_s", "runtime_ns", "repeats", "spread_ns")
PROVENANCE = ("toolchain", "arch", "host", "gpu", "date")
_INTS = ("registers", "spill_store", "spill_load", "shared_bytes", "repeats")


@dataclass(frozen=True, kw_only=True)
class Measured:
    """One observation of one toolchain on one card. NOT a claim about this code.

    Every measurement is optional (None: not observed by this row), but at
    least one must be present. The provenance is required.
    """

    # compile-side, from the device compiler
    registers: int | None = None
    spill_store: int | None = None
    spill_load: int | None = None
    shared_bytes: int | None = None
    compile_s: float | None = None
    # device-side (C1)
    runtime_ns: float | None = None
    repeats: int | None = None
    spread_ns: float | None = None
    # provenance, all REQUIRED
    toolchain: str  # "nvcc 13.0", "hipcc / ROCm 7.2.4"
    arch: str  # "sm_86", "gfx1200"
    host: str
    gpu: str
    date: str
    # how the numbers were asked for
    kernel: str = ""
    spill_unit: str = ""
    command: str = ""  # the compiler invocation, measurement flag included
    source_sha256: str = ""  # of the emitted translation unit
    raw: Mapping[str, int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for n in MEASUREMENTS:
            v = getattr(self, n)
            if v is None:
                continue
            if n in _INTS:
                if type(v) is not int or v < 0:
                    raise ValueError(f"{n} = {v!r} must be an int >= 0 or None")
            elif type(v) not in (int, float) or v < 0:
                raise ValueError(f"{n} = {v!r} must be a number >= 0 or None")
        if self.repeats is not None and self.repeats < 1:
            raise ValueError(f"repeats = {self.repeats!r} must be >= 1")
        if all(getattr(self, n) is None for n in MEASUREMENTS):
            raise ValueError("every measurement is None: the row records nothing")
        for n in PROVENANCE:
            if not isinstance(getattr(self, n), str) or not getattr(self, n):
                raise ValueError(f"{n} must be a non-empty string")



# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


_FIELDS = frozenset(f.name for f in fields(Measured))


def split_key(key: str) -> tuple[str, str, str]:
    """`MeasuredKey.key()` back into (spec key, arch name, toolchain)."""
    spec_key, sep, rest = key.partition("@")
    arch, sep2, toolchain = rest.partition("/")
    if not (sep and sep2 and spec_key and arch and toolchain):
        raise ValueError(f"{key!r} is not a MeasuredKey.key()")
    return spec_key, arch, toolchain


def _row(path: Path, key: str, v: object) -> Measured:
    if not isinstance(v, dict):
        raise ValueError(f"{path}: row {key!r} is not an object")
    unknown = sorted(set(v) - _FIELDS)
    if unknown:
        raise ValueError(f"{path}: row {key!r} has unknown fields {unknown}")
    try:
        row = Measured(**v)
        _, arch, toolchain = split_key(key)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path}: row {key!r}: {e}") from e
    if (arch, toolchain) != (row.arch, row.toolchain):
        raise ValueError(f"{path}: row {key!r} records arch/toolchain "
                         f"{row.arch!r}/{row.toolchain!r}, not its key's")
    if row.runtime_ns is not None:
        for f in GPU_UTIL_FIELDS:
            if f not in row.raw:
                raise ValueError(f"{path}: timed row {key!r} has no {f!r} -- "
                                 "a reading, or null for unknown")
            v = row.raw[f]
            if v is not None and (type(v) is not int or not 0 <= v <= 100):
                raise ValueError(f"{path}: row {key!r}: {f} = {v!r} is not a "
                                 "percentage or null")
    return row


def load(path: Path = STORE) -> dict[str, Measured]:
    """Every row in the store, by key. Refuses a malformed or foreign file.

    A missing file raises FileNotFoundError: an empty store is a valid file
    with no rows, and is a different thing from no store at all.
    """
    path = Path(path)
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not JSON ({e})") from e
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: top level is not an object")
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected {SCHEMA}")
    if not isinstance(doc.get("about"), str) or not doc["about"]:
        raise ValueError(f"{path}: no \"about\" string")
    got = doc.get("rows")
    if not isinstance(got, dict):
        raise ValueError(f"{path}: \"rows\" is not an object")
    return {k: _row(path, k, v) for k, v in got.items()}


def merge(store: Mapping[str, Measured], key: MeasuredKey,
          row: Measured) -> dict[str, Measured]:
    """`store` with `row` at `key`, REPLACING any row already there."""
    if (key.arch.name, key.toolchain) != (row.arch, row.toolchain):
        raise ValueError(f"row is {row.arch}/{row.toolchain}, key {key.key()!r}")
    return {**store, key.key(): row}


def combine(old: Measured | None, new: Measured) -> Measured:
    """`new`, with `old`'s other-side measurements kept when both saw one text.

    A field `new` leaves None is taken from `old`, and `old`'s `raw` entries
    survive under `new`'s, only when the two rows record the same arch,
    toolchain and non-empty `source_sha256`; otherwise `new` alone (module
    docstring, ONE ROW CARRIES BOTH SIDES).
    """
    if (old is None or not new.source_sha256
            or (old.arch, old.toolchain, old.source_sha256)
            != (new.arch, new.toolchain, new.source_sha256)):
        return new
    kept = {n: getattr(old, n) for n in MEASUREMENTS
            if getattr(new, n) is None and getattr(old, n) is not None}
    return replace(new, **kept, raw={**old.raw, **new.raw},
                   kernel=new.kernel or old.kernel,
                   spill_unit=new.spill_unit or old.spill_unit,
                   command=new.command or old.command)


def rows(spec: Spec | None = None, arch: Arch | str | None = None,
         toolchain: str | None = None, *,
         store: Mapping[str, Measured] | Path = STORE) -> dict[str, Measured]:
    """The read path: the rows matching every filter given, by key.

    `store` is an already-loaded mapping, or a path to load (by default the
    tracked store).
    """
    got = load(store) if isinstance(store, Path) else store
    arch_name = arch.name if isinstance(arch, Arch) else arch
    out = {}
    for k, r in got.items():
        s, a, t = split_key(k)
        if spec is not None and s != spec.key():
            continue
        if arch_name is not None and a != arch_name:
            continue
        if toolchain is not None and t != toolchain:
            continue
        out[k] = r
    return out


def save(store: Mapping[str, Measured], path: Path = STORE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": SCHEMA, "about": ABOUT, "header": HEADER,
           "rows": {k: asdict(store[k]) for k in sorted(store, key=_order)}}
    path.write_text(json.dumps(doc, indent=2) + "\n")


def _order(key: str) -> tuple:
    m = re.search(r"\((\d+),(\d+)\|(\d+),(\d+)\)", key)
    am = tuple(int(x) for x in m.groups()) if m else ()
    return (key.split("@", 1)[-1], sum(am), am, key)


# ---------------------------------------------------------------------------
# #60: do 9 batches suffice? (the definition is in the module docstring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwelveVsNine:
    """What the 12-batch cell does against the 9-batch one, at one (arch, L)."""

    arch: str
    L: int
    spill: tuple[int, int]  # (spill(r9), spill(r12))
    wide: tuple[float, float, float]  # (median r9, median r12, max IQR)
    near: tuple[float, float, float]

    @staticmethod
    def _faster(t: tuple[float, float, float]) -> bool:
        return t[0] - t[1] > t[2]

    @property
    def fewer_spills(self) -> bool:
        return self.spill[1] < self.spill[0]

    @property
    def faster_wide(self) -> bool:
        return self._faster(self.wide)

    @property
    def faster_near(self) -> bool:
        return self._faster(self.near)

    @property
    def nine_suffices(self) -> bool:
        return not (self.fewer_spills or self.faster_wide or self.faster_near)


def nine_vs_twelve(store: Mapping[str, Measured], arch: str,
                   L: int) -> TwelveVsNine:
    """#60's comparison at one (arch, L), from the store's rows.

    Raises KeyError when either cell is unrecorded, uncompiled or untimed
    on `arch`.
    """
    def one(b: int) -> Measured:
        got = rows(spec=pq_spec(L, n_batches=b), arch=arch, store=store)
        if len(got) != 1:
            raise KeyError(f"L={L}, n_batches={b} @ {arch}: {len(got)} rows")
        (r,) = got.values()
        if r.runtime_ns is None or r.spread_ns is None or r.spill_store is None \
                or r.spill_load is None:
            raise KeyError(f"L={L}, n_batches={b} @ {arch}: not both compiled "
                           "and timed")
        return r

    def near(r: Measured) -> tuple[float, float]:
        return (r.raw["wallclock:near:median_ns"],
                r.raw["wallclock:near:q3_ns"] - r.raw["wallclock:near:q1_ns"])

    r9, r12 = one(9), one(12)
    n9, n12 = near(r9), near(r12)
    return TwelveVsNine(
        arch=arch, L=L,
        spill=(r9.spill_store + r9.spill_load, r12.spill_store + r12.spill_load),
        wide=(r9.runtime_ns, r12.runtime_ns, max(r9.spread_ns, r12.spread_ns)),
        near=(n9[0], n12[0], max(n9[1], n12[1])))


# ---------------------------------------------------------------------------
# #55: a kernel-layer lever against the literal kernel (module docstring)
# ---------------------------------------------------------------------------

FASTER, SLOWER, NEUTRAL = "faster", "slower", "neutral"
WIN, LOSS, MIXED = "win", "loss", "mixed"


def _sign(control: tuple[float, float], lever: tuple[float, float]) -> str:
    """FASTER / SLOWER / NEUTRAL for (median, IQR) pairs: #60's rule, both ways."""
    margin = max(control[1], lever[1])
    if control[0] - lever[0] > margin:
        return FASTER
    if lever[0] - control[0] > margin:
        return SLOWER
    return NEUTRAL


@dataclass(frozen=True)
class LeverAB:
    """One lever cell against its same-session literal control, on one arch."""

    arch: str
    L: int
    n_batches: int
    lever: str
    spill: tuple[int, int]  # (literal row, lever row): reported, not judged
    wide: tuple[float, float, float, float]  # control median, IQR; lever median, IQR
    near: tuple[float, float, float, float]

    @property
    def wide_sign(self) -> str:
        return _sign(self.wide[:2], self.wide[2:])

    @property
    def near_sign(self) -> str:
        return _sign(self.near[:2], self.near[2:])

    @property
    def verdict(self) -> str:
        signs = {self.wide_sign, self.near_sign}
        if FASTER in signs and SLOWER in signs:
            return MIXED
        if FASTER in signs:
            return WIN
        if SLOWER in signs:
            return LOSS
        return NEUTRAL


def lever_ab(store: Mapping[str, Measured], arch: str, spec: Spec) -> LeverAB:
    """#55's comparison for one lever cell `spec` on `arch`, from the store.

    Raises KeyError when the lever row is missing, uncompiled or untimed, or
    carries no same-session control, or the literal row is missing.
    """
    (lever,) = [lv for lv in spec.levers if lv in LEVERS]
    literal = replace(spec, levers=tuple(lv for lv in spec.levers if lv != lever))

    def one(s: Spec) -> Measured:
        got = rows(spec=s, arch=arch, store=store)
        if len(got) != 1:
            raise KeyError(f"{s.key()} @ {arch}: {len(got)} rows")
        (r,) = got.values()
        return r

    r, lit = one(spec), one(literal)
    if r.runtime_ns is None or r.spread_ns is None or r.spill_store is None \
            or "wallclock:control:median_ns" not in r.raw:
        raise KeyError(f"{spec.key()} @ {arch}: not compiled and timed "
                       "against a same-session control")
    if lit.spill_store is None or lit.spill_load is None or r.spill_load is None:
        raise KeyError(f"{literal.key()} @ {arch}: not compiled")

    def stats(prefix: str) -> tuple[float, float]:
        return (r.raw[f"{prefix}median_ns"],
                r.raw[f"{prefix}q3_ns"] - r.raw[f"{prefix}q1_ns"])

    return LeverAB(
        arch=arch, L=spec.total_am, n_batches=spec.n_batches, lever=lever,
        spill=(lit.spill_store + lit.spill_load, r.spill_store + r.spill_load),
        wide=(*stats("wallclock:control:"), r.runtime_ns, r.spread_ns),
        near=(*stats("wallclock:control:near:"), *stats("wallclock:near:")))
