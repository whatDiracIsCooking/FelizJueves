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

COUNTERS (#110) are the roofline's inputs, read off the card's own
performance counters by a profiler -- `ncu` on NVIDIA, `rocprofv3`'s counter
mode on AMD -- never off a model:

- `fp64_flops`: fp64 floating-point operations EXECUTED, an FMA counted as
  two (ncu: `smsp__sass_thread_inst_executed_op_{dadd,dmul,dfma}_pred_on.sum`,
  dadd + dmul + 2 x dfma, thread-level, so a FLOP count and not a warp
  instruction count). gfx1200 exposes NO fp64 instruction counter
  (rocprofiler-sdk's counter definitions have none for gfx12), so an AMD row
  leaves it None: unmeasured, never zero, and never borrowed from the
  NVIDIA row or from a model.
- `dram_bytes_read`, `dram_bytes_written`: bytes moved between the card's
  last counted cache level and memory. ncu: `dram__bytes_read.sum` /
  `dram__bytes_write.sum`, DRAM traffic at 32-byte-sector granularity.
  gfx1200: the GL2C <-> EA (memory fabric) requests,
  32 x `GL2C_EA_RDREQ_32B_sum` + 64 x `..._64B_sum` + 128 x `..._128B_sum`
  read, and 64 x `GL2C_EA_WRREQ_64B_sum` + 32 x (`GL2C_EA_WRREQ_sum` -
  `GL2C_EA_WRREQ_64B_sum`) written -- L2 misses, which RDNA4's Infinity
  Cache may serve before DRAM, so an UPPER bound on DRAM bytes (#140: on
  this box they all read 0 for streamed traffic, and the driver refuses
  that reading -- see UNMEASURED). The two
  vendors' numbers share a field name, not a meaning: `counter_unit` says
  which, and nothing compares them across vendors.
- `counter_ns`: the profiler's own duration of the kernel UNDER COUNTER
  COLLECTION (ncu `gpu__time_duration.sum`; rocprofv3 End - Start
  timestamps). Collection serialises and replays kernels, so this is NOT a
  timing measurement and never lands in `runtime_ns`: it is the denominator
  of `achieved_flops_per_s` and nothing else.

THE MEASURED COUNTS ARE CANONICAL; the ratios are derived, never stored:
`dram_bytes` (read + written), `arithmetic_intensity` (FLOP/byte) and
`achieved_flops_per_s` (FLOP/s) are properties, so a stored ratio cannot drift
from its inputs. `counter_tool` is the profiler as it reports itself
(`"ncu 2025.3.1.0"`), `counter_unit` what each field means on that vendor,
`command` the full invocation and `raw` every counter by its EXACT name --
a counter renamed between toolchain versions is the drift this shape exists
to surface. The fields are serialised only when present (`as_record`), so a
row without counters reads and writes back exactly as before. Nothing here
is comparable to `[JEngine, v2, Fig. 7]`: their kernels, their A100.

THE COUNTER ROW SHAPE (#110, for #112): ONE ROW PER BATCH KERNEL, in their
own store, `data/measured/roofline.json` (`load_counters` / `save_counters`),
keyed

    kernel_key(MeasuredKey(spec, arch, toolchain), batch)
      == MeasuredKey(...).key() + "#" + str(batch)

`batch` being the kernel's 0-based position in the cell's plan order
(`measured.cell_kernels(spec)`), so a cell of `n_batches` b has exactly the
rows `#0` .. `#b-1` -- a one-batch cell has the one row `#0` -- and the loader
refuses a cell with a missing or extra kernel row, or kernel rows from two
profilers, cards or vendor meanings. The CELL's counters are NOT stored:
`cell_counters` derives them (every count summed over the cell's kernels,
each launched once per class), so the per-kernel and per-cell verdicts are
read off one set of numbers and cannot disagree. Why a separate store rather
than a batch index on `MeasuredKey` or a nested field: `kernels.json`'s 40
literal rows mean one cell each, and `kernelrank`, `batchreport` and both
verdicts read them so; a per-kernel row filed beside them would change what
"the row for this key" is, while one in its own file changes nothing they
read. A counter row carries counter measurements only (a compile observation
belongs in `kernels.json`, a timing in its `runtime_ns`), and the loader
refuses anything else in it.

THE ROOFLINE ROWS (#112) are those counter rows, filled over `SUBGRID` by
test/test_kernel_device.py's opt-in counter sweep (FJ_ROOFLINE_STORE) behind
the card gate (`felizjueves.gate`). Each row carries the card's state around
its phase in `raw` -- ROOFLINE_STATE_FIELDS: the gate's settled reading
before, the reading after (both KNOWN: a phase whose after-reading is
non-idle or unknown was discarded and retaken), the CPU's, and how many
phases of that cell the gate discarded -- and the loader refuses a row
without them. The VERDICT is computed, never stored: `roofline_point` reads
a row's derived intensity and achieved FLOP/s and asks `device.verdict`
against the card's MEASURED ridge (#109), so `memory_bound` cannot be
written by hand, and a card without both ceilings gets no point at all
(`device.Uncharacterised`) -- nor any row: the loader refuses a counter row
for an uncharacterised arch. `kernel_roofline` is the per-batch-kernel
verdict, `cell_roofline` the cell's (its counters summed, so its intensity is
total FLOPs over total bytes and its rate total FLOPs over the summed
serialised durations); the paper's own [7|7] finding is that SOME kernels of
a split stay memory-bound, which only the per-kernel view can see. A card in
UNMEASURED has no rows by decision, with the reason recorded (the header's
`could_not_measure`), and the loader refuses one: COULD NOT MEASURE IS NOT
ZERO, and not a missing file either.

THE BUILD ROWS (#146) are the first measurement of a BUILD rather than a
kernel: the host wall-clock of one whole-molecule `J_p` from a
`launch.LaunchPlan` on one card, timed behind the card gate by
test/test_j_launched_ondevice.py's opt-in sweep (FJ_LAUNCH_WALLCLOCK_STORE).
They live in their own file, `data/measured/builds.json` (`BuildRow`,
`build_key`, `load_builds` / `save_builds`), because no `Spec` names a
molecule's plan and a kernels.json row is never widened; `BUILD_TIMED_REGION`
says exactly what the number contains (and what it does not: the uploads,
the host eq. 20), and `BUILD_STATE_FIELDS` are the gate's KNOWN readings
every row carries -- a contended phase is discarded, so the loader refuses a
null or over-threshold reading rather than read it as idle. A row is keyed by
fixture, tau, n_batches (or `dispatch:<input>`), the plan's `density_sort`
and an ARM label, so #147's same-session sort-key A/B files two rows that
differ in their arm and sort only.

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

from felizjueves.device import (
    MEMORY_BOUND,
    Arch,
    MeasuredKey,
    by_name,
    require_characterised,
    verdict,
)
from felizjueves.kernel import pq_spec
from felizjueves.lower import LEVERS
from felizjueves.spec import Spec

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "data" / "measured" / "kernels.json"
ROOFLINE = REPO / "data" / "measured" / "roofline.json"  # counter rows (#110)

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
    "asym": "#113: rows keyed .../b<n>/j/asym@... are the asymmetric-split "
            "STUDY (felizjueves.asym; its docstring is the rule, committed "
            "before any such row existed): L in {13, 16} x n_batches in {4, "
            "5, 7, 8}, a batch list Spec and partition refuse, each timed "
            "against its two nearest legal neighbours in the same session "
            "(raw 'wallclock:control:b<n>:*'; no literal row re-timed). A "
            "phase the card was taken during was discarded and retaken (raw "
            "'wallclock:gate_discarded_phases'). CONCLUSION: every cell is "
            "WORSE than its best legal neighbour on both sm_86 and gfx1200, "
            "so jengine/symmetry-breaking-splits-are-worse is asserted.",
}


# The counter measurements (#110; module docstring, COUNTERS), and each
# field's unit -- the derived quantities' too, though they are never stored.
COUNTERS = ("fp64_flops", "dram_bytes_read", "dram_bytes_written", "counter_ns")
COUNTER_UNITS = {
    "fp64_flops": "fp64 floating-point operations executed (FMA = 2)",
    "dram_bytes_read": "bytes (vendor meaning: the row's counter_unit)",
    "dram_bytes_written": "bytes (vendor meaning: the row's counter_unit)",
    "counter_ns": "ns, the profiler's kernel duration under counter "
                  "collection -- NOT a timing",
    "arithmetic_intensity": "FLOP / byte = fp64_flops / (dram_bytes_read + "
                            "dram_bytes_written), derived",
    "achieved_flops_per_s": "FLOP / s = fp64_flops / counter_ns, derived",
}
COUNTERS_ABOUT = (
    "#110: fp64_flops, dram_bytes_read, dram_bytes_written and counter_ns are "
    "the card's own performance counters as a profiler reported them for one "
    "kernel -- ncu on NVIDIA (fp64_flops = dadd + dmul + 2 x dfma thread "
    "instructions; DRAM bytes at sector granularity), rocprofv3's counter "
    "mode on AMD (GL2C <-> EA bytes, an upper bound on DRAM; gfx1200 has no "
    "fp64 counter, so fp64_flops is None there) -- with the profiler in "
    "counter_tool, the vendor meaning in counter_unit, the invocation in "
    "command and every counter by its exact name in raw. counter_ns is the "
    "kernel's duration under counter collection (serialised, replayed): NOT "
    "a timing, never runtime_ns. The counts are canonical; "
    "arithmetic_intensity and achieved_flops_per_s are derived, never "
    "stored. Units: " + "; ".join(f"{k}: {v}" for k, v in COUNTER_UNITS.items())
    + ". Counter rows live in data/measured/roofline.json, ONE PER BATCH "
    "KERNEL, keyed MeasuredKey.key() + '#<batch>' (plan order, from 0); a "
    "cell's counters are their sum, derived (measured_store.cell_counters), "
    "never stored. NOT comparable to [JEngine, v2, Fig. 7]: their kernels, "
    "their A100.")
HEADER["counters"] = COUNTERS_ABOUT

MEASUREMENTS = ("registers", "spill_store", "spill_load", "shared_bytes",
                "compile_s", "runtime_ns", "repeats", "spread_ns", *COUNTERS)
PROVENANCE = ("toolchain", "arch", "host", "gpu", "date")
_INTS = ("registers", "spill_store", "spill_load", "shared_bytes", "repeats",
         "fp64_flops", "dram_bytes_read", "dram_bytes_written")
# Serialised only when present (`as_record`): a row without counters is
# written exactly as it was before #110.
_OMITTED_WHEN_ABSENT = (*COUNTERS, "counter_tool", "counter_unit")


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
    # the card's performance counters, from a profiler (#110)
    fp64_flops: int | None = None
    dram_bytes_read: int | None = None
    dram_bytes_written: int | None = None
    counter_ns: float | None = None
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
    counter_tool: str = ""  # the profiler as it reports itself (#110)
    counter_unit: str = ""  # what the counter fields mean on this vendor

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

    # Derived from the stored counts, never stored themselves (#110).
    @property
    def dram_bytes(self) -> int | None:
        """Bytes read + written, or None unless both were counted."""
        if self.dram_bytes_read is None or self.dram_bytes_written is None:
            return None
        return self.dram_bytes_read + self.dram_bytes_written

    @property
    def arithmetic_intensity(self) -> float | None:
        """fp64 FLOP per DRAM byte, or None unless both sides were counted
        (and some byte moved)."""
        b = self.dram_bytes
        if self.fp64_flops is None or not b:
            return None
        return self.fp64_flops / b

    @property
    def achieved_flops_per_s(self) -> float | None:
        """fp64 FLOP/s over the kernel's duration under counter collection,
        or None unless both were counted."""
        if self.fp64_flops is None or not self.counter_ns:
            return None
        return self.fp64_flops * 1e9 / self.counter_ns


def as_record(row: Measured) -> dict:
    """`row` as the store writes it: every field, except that the counter
    fields are left out when absent -- so a row with no counters is written
    exactly as it was before #110."""
    d = asdict(row)
    for n in _OMITTED_WHEN_ABSENT:
        if d[n] in (None, ""):
            del d[n]
    return d



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
    """`store` with `row` at `key`, REPLACING any row already there. `key`
    is a `MeasuredKey`, or anything keyed its way (`arch`, `toolchain`,
    `key()`): #113's `asym.AsymKey`, whose cell is not a Spec."""
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
           "rows": {k: as_record(store[k]) for k in sorted(store, key=_order)}}
    path.write_text(json.dumps(doc, indent=2) + "\n")


def _order(key: str) -> tuple:
    m = re.search(r"\((\d+),(\d+)\|(\d+),(\d+)\)", key)
    am = tuple(int(x) for x in m.groups()) if m else ()
    return (key.split("@", 1)[-1], sum(am), am, key)


# ---------------------------------------------------------------------------
# #110: the counter store, one row per batch kernel (module docstring)
# ---------------------------------------------------------------------------

ROOFLINE_SCHEMA = 1

ROOFLINE_ABOUT = (
    "Counter rows (felizjueves.measured_store, #110): the card's own "
    "performance counters, as a profiler reported them for ONE BATCH KERNEL "
    "of one cell, keyed device.MeasuredKey(spec, arch, toolchain).key() + "
    "'#<batch>', the kernel's 0-based position in the cell's plan order. A "
    "cell's counters are derived (summed over its kernels), never stored. "
    "Observations of a profiler on one card, not claims about this code. NOT "
    "comparable to [JEngine, v2, Fig. 7]: their kernels, their A100.")

ROOFLINE_HEADER = {
    "what": "One row per batch kernel: what a profiler's counters said about "
            "one launch of one emitted kernel on one card "
            "(felizjueves.measured_store, #110).",
    "key": "MeasuredKey(spec, arch, toolchain).key() + '#<batch>': the "
           "generation Spec, the card's Arch, the COMPILER that built the "
           "kernel, and the kernel's 0-based index in the cell's plan order "
           "(measured.cell_kernels). A cell of n_batches b has exactly the "
           "rows #0 .. #b-1; the loader refuses any other set.",
    "cell": "A cell's counters are measured_store.cell_counters: every count "
            "summed over its kernels (each launched once per class). Derived, "
            "never stored, so the per-kernel and per-cell verdicts read one "
            "set of numbers.",
    "counters": COUNTERS_ABOUT,
    "not_timing": "counter_ns is the kernel's duration under counter "
                  "collection, which serialises and replays kernels: never a "
                  "timing. Wall-clock lives in kernels.json's runtime_ns, "
                  "and a row here carries none.",
    "not_compared": "Nothing here is compared against [JEngine, v2, Fig. 7] "
                    "(their kernels, their A100), nor across vendors (the "
                    "fields share a name, not a meaning: see counter_unit).",
}

# #112: the card's state around a counter phase, in every counter row's raw.
# gpu_util_*: percent, the card's own SMI (gate.card_util); both KNOWN (the
# sweep retakes a phase whose after-reading is non-idle or unknown).
# cpu_busy_permille_before: the CPU gate's settled reading. gate_discarded_
# phases: the cell's phases the gate discarded before this one was kept.
ROOFLINE_STATE_FIELDS = ("roofline:gpu_util_before", "roofline:gpu_util_after",
                         "roofline:cpu_busy_permille_before",
                         "roofline:gate_discarded_phases")

# #140: the fp64 instruction counters a roofline row needs and a card in
# UNMEASURED does not expose, and the architectures rocprofiler-sdk 1.1.0
# (ROCm 7.2.4, share/rocprofiler-sdk/counter_defs.yaml) DOES define them for
# -- CDNA only; the derived TOTAL_64_OPS and SQ_INSTS_VALU_FLOPS_FP64 are
# defined for that set too (gfx950 alone for the latter). The refusal's
# mechanism, as data: test/test_toolchain.py reads the card's own
# `rocprofv3 --list-avail` and fails if one of these, or any fp64 / FLOP
# counter, appears on it.
MISSING_COUNTERS = {
    "gfx1200": ("SQ_INSTS_VALU_ADD_F64", "SQ_INSTS_VALU_MUL_F64",
                "SQ_INSTS_VALU_FMA_F64"),
}
FP64_COUNTER_ARCHS = ("gfx90a", "gfx940", "gfx941", "gfx942", "gfx950")

# #112, made FINAL by #140: the cards that have NO counter rows, by decision,
# and why. The absence is recorded here -- never as a zero, never as an absent
# file -- and it is a refusal with its mechanism, not a pending measurement.
UNMEASURED = {
    "gfx1200": (
        "FINAL (#140): no roofline row on gfx1200, because the card exposes "
        "no fp64 FLOP counter. rocprofv3 1.1.0 (ROCm 7.2.4) now LOADS -- "
        "libdw.so.1 and libhsa-amd-aqlprofile64.so.1 are present and the "
        "AMD counter witness in test/test_toolchain.py dispatches -- but of "
        "the counters rocprofiler-sdk defines for gfx1200 none counts fp64 "
        "operations: " + ", ".join(MISSING_COUNTERS["gfx1200"])
        + " (and the derived TOTAL_64_OPS, SQ_INSTS_VALU_FLOPS_FP64) are "
        "defined only for " + ", ".join(FP64_COUNTER_ARCHS)
        + ", and SQ_INSTS_VALU counts every VALU instruction whatever its "
        "type. With no measured FLOP count there is no measured intensity "
        "and no roofline point, and a FLOP count from the model or the "
        "interpreter would be a model, not a counter. Nor are there bytes: "
        "on this box (amdgpu module 6.16.13, host Linux 6.8) the GL2C <-> EA "
        "byte "
        "counters all read 0 for a probe that streams 384 MiB -- as do "
        "TCP_REQ, TA_TA_BUSY and SQ_INSTS_VALU, while GRBM_GUI_ACTIVE and "
        "SQ_WAVES count -- so the driver refuses that reading rather than "
        "file it. No gfx1200 row exists; that is not a zero, and rebuilding "
        "the image does not change it -- a rocprofiler-sdk defining an fp64 "
        "counter for gfx12 would."),
}

ROOFLINE_HEADER["card_state"] = (
    "#112: every row's raw carries " + ", ".join(ROOFLINE_STATE_FIELDS)
    + ": the card's utilisation (percent, its own SMI) as the gate let the "
    "counter phase through and 1 s after it, the CPU's busy fraction "
    "(permille) before it, and how many of that cell's phases the gate "
    "discarded (after-reading non-idle or unknown) before this one was "
    "kept. The counter run is a SEPARATE run of the same checked binary as "
    "kernels.json's timed one, on the same wide-T input (raw 'roofline:"
    "n_bra' / 'roofline:n_ket' / 'roofline:lanes'); collection serialises "
    "and replays each kernel, so it may change occupancy, and its duration "
    "is counter_ns, never runtime_ns.")
ROOFLINE_HEADER["verdict"] = (
    "#112: never stored. measured_store.roofline_point derives intensity "
    "(fp64_flops / DRAM bytes) and achieved FLOP/s (fp64_flops / "
    "counter_ns) from a row and asks device.verdict against the card's "
    "MEASURED ridge (data/measured/ceilings.json, #109): memory_bound iff "
    "intensity < ridge. kernel_roofline is per batch kernel, cell_roofline "
    "per cell (counts summed). A card without both ceilings has no ridge, "
    "no verdict and no row.")
ROOFLINE_HEADER["could_not_measure"] = dict(UNMEASURED)

_COUNTER_ONLY_FORBIDDEN = tuple(n for n in MEASUREMENTS if n not in COUNTERS)
_BATCHES = re.compile(r"/b(\d+)/")


def kernel_key(key: MeasuredKey | str, batch: int) -> str:
    """A counter row's key: the cell's `MeasuredKey.key()` + '#<batch>'."""
    cell = key.key() if isinstance(key, MeasuredKey) else key
    if type(batch) is not int or batch < 0:
        raise ValueError(f"batch {batch!r} must be an int >= 0")
    split_key(cell)
    return f"{cell}#{batch}"


def split_kernel_key(key: str) -> tuple[str, int]:
    """A counter row's key back into (the cell's MeasuredKey.key(), batch)."""
    cell, sep, batch = key.rpartition("#")
    if not sep or not re.fullmatch(r"0|[1-9]\d*", batch):
        raise ValueError(f"{key!r} is not a kernel_key (no '#<batch>')")
    split_key(cell)
    return cell, int(batch)


def n_batches_of(cell: str) -> int:
    """The `n_batches` a cell key's Spec names."""
    m = _BATCHES.search(split_key(cell)[0])
    if not m:
        raise ValueError(f"{cell!r} names no n_batches")
    return int(m[1])


def _counter_row(path: Path, key: str, v: object) -> Measured:
    try:
        cell, _ = split_kernel_key(key)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e
    if isinstance(v, dict):
        unknown = sorted(set(v) - _FIELDS)
        if unknown:
            raise ValueError(f"{path}: row {key!r} has unknown fields {unknown}")
    row = _row(path, cell, v)
    other = [n for n in _COUNTER_ONLY_FORBIDDEN if getattr(row, n) is not None]
    if other:
        raise ValueError(f"{path}: counter row {key!r} records {other}: a "
                         "counter row carries counters only (a timing is "
                         "kernels.json's runtime_ns; counter_ns is not one)")
    if all(getattr(row, n) is None for n in COUNTERS):
        raise ValueError(f"{path}: counter row {key!r} records no counter")
    for n in ("counter_tool", "counter_unit", "command"):
        if not getattr(row, n):
            raise ValueError(f"{path}: counter row {key!r} has no {n}")
    if row.arch in UNMEASURED:
        raise ValueError(f"{path}: counter row {key!r} is on {row.arch}, "
                         "which has no counter rows by decision "
                         "(UNMEASURED: could not measure)")
    try:
        require_characterised(by_name(row.arch))
    except ValueError as e:
        raise ValueError(f"{path}: counter row {key!r}: {e} -- a row for an "
                         "uncharacterised card does not exist") from e
    for f in ROOFLINE_STATE_FIELDS:
        v = row.raw.get(f)
        if type(v) is not int or v < 0 or (
                "gpu_util" in f and v > 100) or ("permille" in f and v > 1000):
            raise ValueError(f"{path}: counter row {key!r}: {f} = {v!r} -- "
                             "the card's state is a KNOWN reading in every "
                             "counter row")
    return row


def _check_cells(path: Path, got: Mapping[str, Measured]) -> None:
    cells: dict[str, dict[int, Measured]] = {}
    for k, r in got.items():
        cell, b = split_kernel_key(k)
        cells.setdefault(cell, {})[b] = r
    for cell, ks in cells.items():
        want = set(range(n_batches_of(cell)))
        if set(ks) != want:
            raise ValueError(f"{path}: cell {cell!r} has kernel rows "
                             f"{sorted(ks)}, not exactly {sorted(want)}")
        first = ks[0]
        for b, r in ks.items():
            same = ("counter_tool", "counter_unit", "gpu", "host")
            bad = [n for n in same if getattr(r, n) != getattr(first, n)]
            present = {n for n in COUNTERS if getattr(r, n) is not None}
            if present != {n for n in COUNTERS
                           if getattr(first, n) is not None}:
                bad.append("the counters present")
            if bad:
                raise ValueError(f"{path}: cell {cell!r}: kernel #{b} differs "
                                 f"from #0 in {bad}")


def load_counters(path: Path = ROOFLINE) -> dict[str, Measured]:
    """Every counter row, by `kernel_key`. Refuses a malformed or foreign
    file, a non-counter measurement, and an incomplete or mixed cell.

    A missing file raises FileNotFoundError, as `load` does.
    """
    path = Path(path)
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not JSON ({e})") from e
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: top level is not an object")
    if doc.get("schema") != ROOFLINE_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected "
                         f"{ROOFLINE_SCHEMA}")
    if not isinstance(doc.get("about"), str) or not doc["about"]:
        raise ValueError(f"{path}: no \"about\" string")
    got = doc.get("rows")
    if not isinstance(got, dict):
        raise ValueError(f"{path}: \"rows\" is not an object")
    out = {k: _counter_row(path, k, v) for k, v in got.items()}
    _check_cells(path, out)
    return out


def save_counters(store: Mapping[str, Measured], path: Path = ROOFLINE) -> None:
    """Write the counter store, refusing what `load_counters` would refuse."""
    path = Path(path)
    tmp = {k: _counter_row(path, k, as_record(r)) for k, r in store.items()}
    _check_cells(path, tmp)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": ROOFLINE_SCHEMA, "about": ROOFLINE_ABOUT,
           "header": ROOFLINE_HEADER,
           "rows": {k: as_record(store[k])
                    for k in sorted(store, key=_kernel_order)}}
    path.write_text(json.dumps(doc, indent=2) + "\n")


def _kernel_order(key: str) -> tuple:
    cell, b = split_kernel_key(key)
    return (*_order(cell), b)


def kernel_counters(store: Mapping[str, Measured],
                    cell: MeasuredKey | str) -> tuple[Measured, ...]:
    """One cell's counter rows, in plan order (#0 first). KeyError if absent."""
    c = cell.key() if isinstance(cell, MeasuredKey) else cell
    n = n_batches_of(c)
    try:
        return tuple(store[kernel_key(c, b)] for b in range(n))
    except KeyError as e:
        raise KeyError(f"{c}: no counter row {e}") from None


def cell_counters(store: Mapping[str, Measured],
                  cell: MeasuredKey | str) -> Measured:
    """A cell's counters, DERIVED from its kernel rows: each count summed
    over the batch kernels (None where the kernels did not count it); the
    provenance is kernel #0's, the date the latest; `raw` is every kernel's,
    keyed '#<batch>:<counter>'."""
    ks = kernel_counters(store, cell)
    first = ks[0]
    sums = {n: (None if any(getattr(r, n) is None for r in ks)
                else sum(getattr(r, n) for r in ks)) for n in COUNTERS}
    return replace(
        first, **sums, date=max(r.date for r in ks),
        kernel=",".join(r.kernel for r in ks), source_sha256="",
        raw={f"#{b}:{k}": v for b, r in enumerate(ks) for k, v in r.raw.items()})


@dataclass(frozen=True)
class RooflinePoint:
    """One kernel's (or one cell's) place under its card's measured roofs.
    Derived from counter rows and the card's ceilings, never stored."""

    key: str
    arch: str
    intensity: float  # fp64 FLOP / DRAM byte
    achieved_flops_per_s: float  # fp64 FLOP / counter_ns
    ridge: float  # the card's measured ridge, FLOP / byte (#109)
    peak_flops_per_s: float  # the card's measured fp64 ceiling (#109)
    memory_bound: bool  # intensity < ridge, by device.verdict

    @property
    def of_peak(self) -> float:
        """Achieved FLOP/s over the card's measured fp64 ceiling."""
        return self.achieved_flops_per_s / self.peak_flops_per_s


def roofline_point(row: Measured, key: str = "") -> RooflinePoint:
    """`row`'s verdict against its card's measured ridge. Refuses
    (`device.Uncharacterised`) a card without both ceilings, and a row
    without a measured intensity or rate (ValueError): no FLOP count, no
    point -- never a zero."""
    arch = require_characterised(by_name(row.arch))
    intensity, rate = row.arithmetic_intensity, row.achieved_flops_per_s
    if intensity is None or rate is None:
        raise ValueError(f"{key or row.kernel}: no measured intensity or "
                         "FLOP rate (fp64_flops, DRAM bytes or counter_ns "
                         "uncounted): no roofline point")
    return RooflinePoint(
        key=key, arch=arch.name, intensity=intensity,
        achieved_flops_per_s=rate, ridge=arch.ridge,
        peak_flops_per_s=arch.fp64_flops_per_s,
        memory_bound=verdict(arch, intensity) == MEMORY_BOUND)


def kernel_roofline(store: Mapping[str, Measured],
                    cell: MeasuredKey | str) -> tuple[RooflinePoint, ...]:
    """The verdict per batch kernel of `cell`, in plan order."""
    c = cell.key() if isinstance(cell, MeasuredKey) else cell
    return tuple(roofline_point(r, kernel_key(c, b))
                 for b, r in enumerate(kernel_counters(store, c)))


def cell_roofline(store: Mapping[str, Measured],
                  cell: MeasuredKey | str) -> RooflinePoint:
    """The cell's verdict, off `cell_counters` (every count summed)."""
    c = cell.key() if isinstance(cell, MeasuredKey) else cell
    return roofline_point(cell_counters(store, c), c)


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



# ---------------------------------------------------------------------------
# #146: the build store -- one row per timed WHOLE-MOLECULE J build
# ---------------------------------------------------------------------------
#
# WHY ITS OWN FILE. A kernels.json row is keyed on MeasuredKey(spec, arch,
# toolchain): ONE class's batch kernels. A build is a whole molecule's launch
# plan -- every (lp, lq) class of a basis, each at its own n_batches, over
# real index records under a screen -- and no Spec names it. Filing it under
# a MeasuredKey would need a fake Spec or a widened kernel row, and every
# reader of kernels.json (kernelrank, batchreport, dispatch, both verdicts)
# reads "the row for this key" as one class. So a build row is its own record
# in its own file, data/measured/builds.json, with its own provenance and
# key, and nothing that reads kernels.json sees it.

BUILDS = REPO / "data" / "measured" / "builds.json"
BUILD_SCHEMA = 1
DISPATCH_PREFIX = "dispatch:"  # n_batches of a resolved plan: "dispatch:wide"
BASELINE_ARM = "baseline"  # the plan's default; #147 labels its own arms

# The card's state around the KEPT timing phase, in every build row's raw
# (the gate's readings, felizjueves.gate; this module imports no gate, so the
# threshold the phase was judged against is recorded beside the readings):
#   build:gpu_util_before            percent, the settled reading the gate
#                                    let the phase through at
#   build:gpu_util_after             percent, gate.after_reading after it
#   build:gpu_busy_max               the gate's GPU_BUSY_MAX at the time
#   build:cpu_busy_permille_before   the CPU gate's settled reading
#   build:gate_discarded_phases      phases of this cell discarded before
#                                    this one was kept (after-reading
#                                    non-idle or unknown)
# All KNOWN integers: a contended or unreadable phase is DISCARDED, never
# recorded, so a null here is refused rather than read as idle.
BUILD_STATE_FIELDS = ("build:gpu_util_before", "build:gpu_util_after",
                      "build:gpu_busy_max", "build:cpu_busy_permille_before",
                      "build:gate_discarded_phases")

BUILD_TIMED_REGION = (
    "runtime_ns / spread_ns / repeats are the MEDIAN, the INTERQUARTILE RANGE "
    "and the COUNT of `repeats` timed builds (after raw 'build:warmup' "
    "untimed ones, all in ONE process), host std::chrono::steady_clock. ONE "
    "TIMED BUILD IS, in order: (a) one device memset zeroing each bra "
    "class's J_p buffer; (b) every batch kernel of every (lp, lq) class of "
    "the plan, launched in plan order on the default stream into its bra "
    "class's J_p (fp64 atomics); (c) one device synchronize; (d) the "
    "device-to-host copy of every bra class's J_p (one blocking copy per bra "
    "class). It starts after a device synchronize, so no earlier work is "
    "inside it. raw 'build:device_median_ns' is (a)-(c) alone and "
    "'build:readback_median_ns' is (d) alone (each the median over the same "
    "repeats); 'build:events_median_ns' is the GPU-event time from before "
    "(a) to after (b). OUTSIDE THE CLOCK: process start, reading the input "
    "file, the VRAM probe, every device allocation, and the H2D UPLOADS -- "
    "one per bra class (P zeta Ip) and one per ket class (Q eta Iq Dmax D), "
    "done ONCE per process before the first build and timed once, apart, as "
    "raw 'build:upload_ns' (a single observation, allocation included); the "
    "host-side staging that builds those buffers (Python, ao.staged); the "
    "host-side eq. 20 back-transform and J_ab scatter (Python, "
    "ao.staged._transform), which is not device code and is not timed; and "
    "compilation. So a row is the device cost of one J_p for a molecule "
    "whose index data is already resident, read back to the host ready for "
    "eq. 20 -- not a cold start and not a whole SCF iteration.")

BUILD_ABOUT = (
    "Build rows (felizjueves.measured_store, #146): the host wall-clock of "
    "ONE WHOLE-MOLECULE J_p BUILD from a launch.LaunchPlan on one card -- "
    "every class's batch kernels into one J_p per bra class -- timed behind "
    "the card gate (felizjueves.gate), keyed by build_key. Its own file: a "
    "build is no MeasuredKey's Spec, and kernels.json's rows are not "
    "widened. Observations of two cards and their toolchains, NOT comparable "
    "to [JEngine]'s A100 figures, nor across vendors.")

BUILD_HEADER = {
    "what": "One row per timed whole-molecule J_p build (launch.launches -> "
            "test/plan_driver.py's generated driver), per card, per arm.",
    "key": "build_key(fixture, tau, n_batches, density_sort, arm, arch, "
           "toolchain) = '<fixture>/tau=<repr>/b=<n>/sort=<density|global>/"
           "<arm>@<arch>/<toolchain>'. fixture is test/oracle/fixtures/"
           "<fixture>.json (the molecule, basis and density); n_batches is "
           "the plan's fixed count, or 'dispatch:<input>' for a plan "
           "resolved per class by dispatch.Resolver(arch, <input>); "
           "density_sort is launch.launches' sort key (density: G * Dmax per "
           "ket; global: one global Dmax, so the order is by G alone); arm "
           "labels a same-session A/B (#147) -- 'baseline' is the plan's "
           "default, timed alone.",
    "timed_region": BUILD_TIMED_REGION,
    "gate": "#146: every row's raw carries " + ", ".join(BUILD_STATE_FIELDS)
            + ". A cell's timing phase is one process run of the checked "
            "binary (warm-up plus repeats) between gate.wait_quiet (card at "
            "most build:gpu_busy_max percent busy by its own SMI, CPU at most "
            "gate.BUSY_MAX) and gate.after_reading; a phase whose "
            "after-reading was non-idle or unknown is DISCARDED and retaken, "
            "and counted in build:gate_discarded_phases. A contended timing "
            "is a different experiment, not a noisy one: nothing from a "
            "discarded phase is in a row, and the loader refuses a row whose "
            "card state is unknown or over the threshold.",
    "checked": "The timed binary is the checked binary: before timing, each "
               "cell's driver is run in its correctness mode and its J "
               "asserted against the launch oracle (launch.walk) and the "
               "libcint fixture (test/test_j_launched_ondevice.py's "
               "tolerances), and every timed process's first and last J_p "
               "must agree with it within the fp64-atomic run-to-run bound.",
    "plan": "raw 'build:classes', 'build:kernels' (batch-kernel launches per "
            "build), 'build:bra_classes', 'build:ket_classes', "
            "'build:n_pairs' and 'build:n_skipped' (the launch oracle's "
            "screen count on the same plan: the (bra, ket) record pairs the "
            "kernel's break skips), and 'build:refused' (classes built at 1 "
            "batch because the plan refused the asked count).",
    "not_compared": "Report, do not compare: nothing here is compared "
                    "against [JEngine]'s A100 build times, nor across vendors "
                    "(gfx1200 rate-limits fp64), nor against kernels.json's "
                    "synthetic-input class timings.",
    "rerun": "env FJ_LAUNCH_WALLCLOCK_STORE=data/measured/builds.json "
             "devtools/kernel-tier.sh --backend <cuda|hip> -- -n 0 -s "
             "-k build_wallclock (alone; one session per card).",
}

# A card with no build row by decision, and why: a session the gate refused
# for good would be recorded here, never as a zero. Empty: both cards have
# rows.
BUILD_UNMEASURED: dict[str, str] = {}
BUILD_HEADER["could_not_measure"] = dict(BUILD_UNMEASURED)

_BUILD_INTS = ("runtime_ns", "spread_ns", "repeats")


def _n_batches_word(n: int | str) -> str:
    if type(n) is int and n >= 1:
        return str(n)
    if (isinstance(n, str) and n.startswith(DISPATCH_PREFIX)
            and len(n) > len(DISPATCH_PREFIX) and "/" not in n
            and "@" not in n):
        return n
    raise ValueError(f"n_batches {n!r} is neither an int >= 1 nor "
                     f"'{DISPATCH_PREFIX}<input>'")


def build_key(fixture: str, tau: float, n_batches: int | str,
              density_sort: bool, arm: str, arch: str, toolchain: str) -> str:
    """A build row's key (BUILD_HEADER['key'])."""
    for name, v in (("fixture", fixture), ("arm", arm), ("arch", arch),
                    ("toolchain", toolchain)):
        if not isinstance(v, str) or not v or (name != "toolchain" and (
                "/" in v or "@" in v)):
            raise ValueError(f"{name} {v!r} must be a non-empty string "
                             "without '/' or '@'")
    if type(tau) is not float or not tau >= 0.0:
        raise ValueError(f"tau {tau!r} must be a float >= 0")
    if type(density_sort) is not bool:
        raise ValueError(f"density_sort {density_sort!r} must be a bool")
    sort = "density" if density_sort else "global"
    return (f"{fixture}/tau={tau!r}/b={_n_batches_word(n_batches)}/"
            f"sort={sort}/{arm}@{arch}/{toolchain}")


@dataclass(frozen=True, kw_only=True)
class BuildRow:
    """One timed whole-molecule J_p build on one card (#146). An observation,
    not a claim about the paper's numbers; BUILD_TIMED_REGION is what the
    number contains."""

    # what was built
    fixture: str
    tau: float
    n_batches: int | str
    density_sort: bool
    arm: str
    # the timing (ns, host steady_clock; BUILD_TIMED_REGION)
    runtime_ns: int
    spread_ns: int
    repeats: int
    # provenance, all REQUIRED
    toolchain: str
    arch: str
    host: str
    gpu: str
    date: str
    command: str  # the driver's compile command (paths elided)
    source_sha256: str  # of the driver text plus every unit, in launch order
    raw: Mapping[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.key()  # validates fixture / tau / n_batches / sort / arm
        for n in _BUILD_INTS:
            v = getattr(self, n)
            if type(v) is not int or v < 0:
                raise ValueError(f"{n} = {v!r} must be an int >= 0")
        if self.repeats < 1:
            raise ValueError(f"repeats = {self.repeats!r} must be >= 1")
        for n in (*PROVENANCE, "command", "source_sha256"):
            if not isinstance(getattr(self, n), str) or not getattr(self, n):
                raise ValueError(f"{n} must be a non-empty string")
        for k, v in self.raw.items():
            if not isinstance(k, str) or type(v) not in (int, float):
                raise ValueError(f"raw {k!r} = {v!r}: a number, never null "
                                 "(an unknown is not recorded as a reading)")
        for f in BUILD_STATE_FIELDS:
            v = self.raw.get(f)
            if type(v) is not int or v < 0 or (
                    "gpu_" in f and v > 100) or ("permille" in f and v > 1000):
                raise ValueError(f"raw {f} = {v!r} -- every build row carries "
                                 "the card's state as a KNOWN reading")
        cap = self.raw["build:gpu_busy_max"]
        for f in ("build:gpu_util_before", "build:gpu_util_after"):
            if self.raw[f] > cap:
                raise ValueError(f"raw {f} = {self.raw[f]}% > build:gpu_busy_"
                                 f"max = {cap}%: a contended phase is "
                                 "discarded, never recorded")

    def key(self) -> str:
        return build_key(self.fixture, self.tau, self.n_batches,
                         self.density_sort, self.arm, self.arch,
                         self.toolchain)


_BUILD_FIELDS = frozenset(f.name for f in fields(BuildRow))


def _build_row(path: Path, key: str, v: object) -> BuildRow:
    if not isinstance(v, dict):
        raise ValueError(f"{path}: row {key!r} is not an object")
    unknown = sorted(set(v) - _BUILD_FIELDS)
    if unknown:
        raise ValueError(f"{path}: row {key!r} has unknown fields {unknown}")
    try:
        row = BuildRow(**v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path}: row {key!r}: {e}") from e
    if row.key() != key:
        raise ValueError(f"{path}: row {key!r} records the key {row.key()!r}")
    if row.arch in BUILD_UNMEASURED:
        raise ValueError(f"{path}: row {key!r} is on {row.arch}, which has no "
                         "build rows by decision (BUILD_UNMEASURED)")
    return row


def load_builds(path: Path = BUILDS) -> dict[str, BuildRow]:
    """Every build row, by `build_key`. Refuses a malformed or foreign file,
    a row filed under a key not its own, and a row without a KNOWN,
    uncontended card state. A missing file raises FileNotFoundError, as
    `load` does."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not JSON ({e})") from e
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: top level is not an object")
    if doc.get("schema") != BUILD_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected "
                         f"{BUILD_SCHEMA}")
    if not isinstance(doc.get("about"), str) or not doc["about"]:
        raise ValueError(f"{path}: no \"about\" string")
    got = doc.get("rows")
    if not isinstance(got, dict):
        raise ValueError(f"{path}: \"rows\" is not an object")
    return {k: _build_row(path, k, v) for k, v in got.items()}


def merge_build(store: Mapping[str, BuildRow],
                row: BuildRow) -> dict[str, BuildRow]:
    """`store` with `row` at its own key, REPLACING any row already there."""
    return {**store, row.key(): row}


def save_builds(store: Mapping[str, BuildRow], path: Path = BUILDS) -> None:
    for k, r in store.items():
        if r.key() != k:
            raise ValueError(f"row filed at {k!r} is {r.key()!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": BUILD_SCHEMA, "about": BUILD_ABOUT,
           "header": BUILD_HEADER,
           "rows": {k: asdict(store[k])
                    for k in sorted(store, key=_build_order)}}
    path.write_text(json.dumps(doc, indent=2) + "\n")


def _build_order(key: str) -> tuple:
    cell, _, where = key.partition("@")
    return (where, cell)
