"""The fused Fock kernel ON A CARD (#288) -- harness.

    record(plan, basis, D, *, threads=)            -> (list[FRec], FWalk)
    fock_program(kernels, backend, workdir)        -> Program (the runner)
    fock_stage(recs, kernels)                      -> Staged (the sequence)
    run_on_card(binary, staged, workdir, *, timing=) -> ([run] F, Sequenced)
    reps(ran)                                      -> (wall_ns, events_ns)
    interpret(recs, n_ao)                          -> F (not closed)
    synthetic_launch(cls, *, side, seed)           -> (FRec, basis)
    fock_scatter_program / fock_scatter_stage / interpret_scatter
                                     the #358 scatter path (end of file)

THE LAUNCHES ARE `walk_f`'s, RECORDED. `record` walks `launch_f`'s plan with
`walk_f` (#286) and an injected `run` that only RECORDS each call -- the
kernel, the grid, the block and every argument by name, exactly what the
interpreted build is handed -- and returns the running `F` unchanged. So
the card replays the very launches the interpreted walk runs (the same
packed records, the same `diag`, the same `Dsh` and `D`), and nothing about
the plan, the packing or the screen is decided here. `interpret` runs those
recorded launches through `interp` (Boys from `reference/`, injected), one
after another into one `F`: walk_f's own executor, restated, so a subset of
the launches (one class) has a host reference too.

THE PROGRAM is the shared runner (#649; ported by #657) over a set of
emitted F units (`kernel_f.build_f_kernel`, spelt by `emit/cuda.py` /
`emit/hip.py` unchanged, each in its own namespace,
`harness_driver.units_runner`), driven at RUN time by a SEQUENCE: any number
of launches of any of its kernels, in order, into ONE `F`. The per-class
parity runs and the wall-clock sweep therefore run the SAME binary on
different sequences -- the timed binary is the checked binary. `stage`
builds the sequence from the records: every parameter read off the
launch's own kernel (so a lever's extra array is staged with no change
here), `Dsh` and `D` one buffer shared by every launch, every other array
the launch's own buffer, and a launch is `grid = n_bra`, `block =
(threads, 1)` (`kernel_f.launch_shape`).

TWO MODES. Correctness: `RUNS` runs, each from a zeroed `F`, every run's
`F` kept -- NOT closed (the host's `kernel_f.close` is the caller's, as in
walk_f) -- since the six blocks add by atomics, whose order is the card's:
the caller compares the runs at its tolerance. Timing (`timing=(WARMUP,
REPEATS)`): the same sequence WARMUP times untimed and REPEATS times timed,
each from a zeroed `F` (restored, and synchronized, OUTSIDE the clock),
each timed run's host `steady_clock` across every launch and the closing
synchronize and its GPU events around the launches (`reps`), and the last
two `F`s returned.

THE SYNTHETIC LAUNCH (`synthetic_launch`) is the wall-clock sweep's input:
one OFF-diagonal launch of the class (`diag = 0`), `TIMING_N_BRA` bra pairs
by `TIMING_N_KET` ket pairs -- test/test_kernel_device.py's sizes for the
J rows -- of a basis of 3-primitive shells (`K_AB = K_CD = 9`) at random
centres in a cube of side `side`, `tau = 0` so no quartet is screened, a
random symmetric density. The bra and ket pairs are disjoint sets of the
class's canonical pairs; nothing is closed or compared but its finiteness
(the binary was checked on real launches first). The basis is
`SYNTHETIC_COUNTS` shells of `l = 0, 1, 2` when that fills the launch, and
`DEFERRED_COUNTS` otherwise (#290: twelve of the sixteen classes above L =
2); the launch shape, the exponents and the cubes are the same either way.

Harness, in `test/`: not a second emitter, no measurement decided here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from harness_driver import Ref, Sequenced, Step, Zeros, run_sequence, units_runner
from plan_driver import Program, _unit

from felizjueves.ao.density import checked_density
from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.fock import kernel_f as KF
from felizjueves.eri.fock import kernel_f_scatter as FS
from felizjueves.eri.fock.levers import scatter as SC
from felizjueves.eri.fock.walk_f import KET_BREAK, FWalk, pair_record, walk_f
from felizjueves.hgp import vrr
from felizjueves.ir import Intent, Kernel, Scalar
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.reference.boys import boys

THREADS = 32  # block width: kets strided over 32 lanes; not numerical
RUNS = 2  # runs per check-mode process: the run-to-run witness
SCALARS = ("n_bra", "n_ket", "k_ab", "k_cd", "n_sh", "n_ao", "diag", "tau")
SHARED = ("Dsh", "D")
TIMING_N_BRA = 2048  # test_kernel_device.py's TIMING_N_BRA
TIMING_N_KET = 128  # and TIMING_N_KET


@dataclass(frozen=True)
class FRec:
    """One recorded launch: the kernel, its grid and block width, and every
    argument by name (arrays flat, row-major), as walk_f hands `run`."""

    kernel: Kernel
    grid: int
    threads: int
    args: Mapping[str, object]

    @property
    def name(self) -> str:
        return self.kernel.name

    def pairs(self, side: str) -> list[tuple[int, int]]:
        sh = [int(x) for x in self.args[f"{side}_sh"]]
        return list(zip(sh[::2], sh[1::2], strict=True))


def record(plan, basis: Basis, D, *, threads: int = THREADS
           ) -> tuple[list[FRec], FWalk]:
    """walk_f over `plan` with a run that records every launch and returns
    `F` unchanged (module docstring, THE LAUNCHES)."""
    recs: list[FRec] = []

    def run(kernel, grid, block, args):
        recs.append(FRec(kernel, grid, block[0], dict(args)))
        return {KF.FOCK: args[KF.FOCK]}

    w = walk_f(plan, basis, D, run=run, threads=threads)
    return recs, w


def record_plan(plan, basis: Basis, D, *, threads: int = THREADS,
                levers: tuple[str, ...] = ()) -> list[FRec]:
    """The launches `record` would return, packed straight from `plan`:
    walk_f's per-launch arguments (`pair_record`, `kernel_f.pack`, `Dsh`,
    `D`) with none of its walk -- no partition of the unique quartets and no
    six-block replay, which at polyglycine's size are ~1e9 Python set
    members. Every launch's kernel is built with `levers` added to its own;
    a plan whose launches name `ket-break` or `scatter` (launch arguments
    walk_f adds) is refused. `test_fock_record_plan.py` holds it to
    `record` launch by launch."""
    rows = checked_density(basis, D)
    sh, n_sh, n_ao = basis.shells, len(basis.shells), basis.n_ao
    Dsh = KF.density_blocks(rows, basis.offsets, [s.n_cart for s in sh])
    D_flat = tuple(x for r in rows for x in r)
    records = [[pair_record(basis, p.a, p.b, G=p.G, prims=p.prims)
                for p in b.pairs] for b in plan.batches]
    kernels: dict[object, Kernel] = {}
    out = []
    for L in plan.launches:
        own = tuple(L.spec.levers)
        if set(own) & {KET_BREAK, SC.NAME}:
            raise ValueError(f"record_plan: launch ({L.bra}, {L.ket}) names "
                             f"{own}; walk_f's `record` stages those")
        spec = replace(L.spec, levers=tuple(sorted({*own, *levers})))
        if spec not in kernels:
            kernels[spec] = KF.build_f_kernel(spec)
        k = kernels[spec]
        pb, pk = KF.pack(records[L.bra]), KF.pack(records[L.ket])
        args = {
            "n_bra": pb.n, "n_ket": pk.n, "k_ab": pb.degree,
            "k_cd": pk.degree, "n_sh": n_sh, "n_ao": n_ao,
            "diag": int(L.bra == L.ket), "tau": float(plan.tau),
            "bra_prim": pb.prim, "AB": pb.sep, "G_ab": pb.G,
            "bra_sh": pb.sh, "bra_ao": pb.ao, "ket_prim": pk.prim,
            "CD": pk.sep, "G_cd": pk.G, "ket_sh": pk.sh, "ket_ao": pk.ao,
            "Dsh": Dsh, "D": D_flat,
        }
        grid, block = KF.launch_shape(pb.n, threads)
        out.append(FRec(k, grid, block[0],
                        {p.name: args[p.name] for p in k.params
                         if p.name != KF.FOCK}))
    return out


def interpret(recs: Sequence[FRec], n_ao: int) -> np.ndarray:
    """The recorded launches through the interpreter, in order, into one
    zeroed `F` (`n_ao x n_ao`, NOT closed)."""
    F = np.zeros(n_ao * n_ao)
    for r in recs:
        a = {}
        for p in r.kernel.params:
            v = F if p.name == KF.FOCK else r.args[p.name]
            if p.array:
                v = np.array(v, dtype=np.int32 if p.type is Scalar.I32
                             else float)
            a[p.name] = v
        F = Interpreter(r.kernel, launch=Launch(r.grid, (r.threads, 1), 32),
                        boys=boys).run(**a)[KF.FOCK]
    return np.asarray(F, dtype=float).reshape(n_ao, n_ao)


def _check_signature(k: Kernel) -> None:
    names = [p.name for p in k.params]
    want = [n for n in (*SCALARS, "bra_prim", "AB", "G_ab", "bra_sh",
                        "bra_ao", "ket_prim", "CD", "G_cd", "ket_sh",
                        "ket_ao", *SHARED, KF.FOCK) if n in names]
    if names != want or [p.name for p in k.params
                         if p.intent is Intent.OUT] != [KF.FOCK]:
        raise ValueError(f"{k.name}: signature {names} is not the fused F "
                         "kernel's (kernel_f.params_of, no tap)")


def fock_program(kernels: Sequence[Kernel], backend: str, workdir: Path, *,
                 units: Mapping[str, Path] | None = None,
                 name: str = "fock") -> Program:
    """The runner over `kernels`' units (module docstring, THE PROGRAM)."""
    if not kernels or len({k.name for k in kernels}) != len(kernels):
        raise ValueError("fock_program takes distinct kernels")
    if name in {k.name for k in kernels}:
        raise ValueError(f"the program {name!r} would overwrite its unit")
    for k in kernels:
        _check_signature(k)
    return runner_of(kernels, backend, workdir, units=units, name=name)


def runner_of(kernels: Sequence[Kernel], backend: str, workdir: Path, *,
              units: Mapping[str, Path] | None = None,
              name: str) -> Program:
    """One runner over every kernel's unit, emitted into `workdir` unless
    `units` has it (hjk_driver's too)."""
    workdir.mkdir(parents=True, exist_ok=True)
    return units_runner(backend, [(_unit(k, backend, workdir, units), [k.name])
                                  for k in kernels], workdir, name)


def _global(recs: Sequence, key: str):
    vals = {tuple(np.ravel(np.asarray(r.args[key], dtype=float)))
            for r in recs}
    if len(vals) != 1:
        raise ValueError(f"the launches disagree on {key}")
    return np.array(next(iter(vals)))


_NP = {Scalar.F64: np.float64, Scalar.I32: np.int32}


@dataclass(frozen=True)
class Staged:
    """A sequence for the runner: its buffers, its launches, the output
    buffer read back and its side."""

    buffers: Mapping[str, object]
    steps: tuple[Step, ...]
    out: str
    n_ao: int


def stage(recs: Sequence, kernels: Sequence[Kernel], *, out: str,
          shared: Sequence[str] = SHARED) -> Staged:
    """`recs` (FRec or hjk_driver.HRec), each launching one of `kernels`,
    as a sequence into ONE zeroed `n_ao x n_ao` buffer named `out`: every
    parameter read off the launch's own kernel -- a scalar from its
    arguments, `out` the output, a `shared` array one buffer for every
    launch (they must agree), and every other array the launch's own
    buffer `l<i>.<name>`. One launch per record, `grid` blocks of
    `(threads, 1)`."""
    if not recs:
        raise ValueError("no launch to stage")
    index = {k.name: k for k in kernels}
    n_ao = int(_global(recs, "n_ao")[0])
    buffers: dict[str, object] = {out: Zeros((n_ao, n_ao))}
    for n in shared:
        if any(n in {p.name for p in r.kernel.params} for r in recs):
            buffers[n] = _global(recs, n).astype(np.float64)
    steps = []
    for i, r in enumerate(recs):
        if index.get(r.name) != r.kernel:
            raise ValueError(f"{r.name} is not one of the program's kernels")
        assert r.grid == r.args["n_bra"], (r.grid, r.args["n_bra"])
        args: dict[str, object] = {}
        for p in r.kernel.params:
            if p.name == out:
                args[p.name] = out
            elif not p.array:
                args[p.name] = r.args[p.name]
            elif p.name in shared:
                args[p.name] = p.name
            else:
                buf = f"l{i}.{p.name}"
                buffers[buf] = np.ravel(np.asarray(r.args[p.name],
                                                   dtype=_NP[p.type]))
                args[p.name] = buf
        steps.append(Step(r.kernel, r.grid, (r.threads, 1), args))
    return Staged(buffers, tuple(steps), out, n_ao)


def fock_stage(recs: Sequence[FRec], kernels: Sequence[Kernel]) -> Staged:
    """The sequence of `recs`, each launching one of `kernels`, into `F`
    (module docstring, THE PROGRAM)."""
    return stage(recs, kernels, out=KF.FOCK)


def unstage(staged: Staged) -> list[tuple[Kernel, int, int, dict]]:
    """Each launch's kernel, grid, block width and arguments as the runner
    is handed them -- every array read back out of its buffer, the output
    `None` (for the tests that interpret the staged sequence)."""
    out = []
    for st in staged.steps:
        args = {}
        for p in st.kernel.params:
            v = st.args[p.name]
            if p.name == staged.out:
                v = None
            elif p.array:
                ref = v if isinstance(v, Ref) else Ref(v)
                v = np.ravel(np.asarray(staged.buffers[ref.buffer]))[
                    ref.offset:]
            args[p.name] = v
        grid = st.grid if isinstance(st.grid, int) else st.grid[0]
        out.append((st.kernel, int(grid), int(st.block[0]), args))
    return out


def reps(ran: Sequenced) -> tuple[list[int], list[int]]:
    """The timed runs' host wall-clock and GPU-event times, in order."""
    return list(ran.wall_ns), [int(round(e)) for e in ran.events_ns]


def run_on_card(binary: Path, staged: Staged, workdir: Path, *,
                timing: tuple[int, int] | None = None,
                tag: str = "fock") -> tuple[np.ndarray, Sequenced]:
    """Run the sequence: `[RUNS][n_ao, n_ao]` (not closed), every run kept
    (the kernels add by atomics, so the caller compares the runs at its
    tolerance), and the runner's result. `timing=(warmup, repeats)`: the
    same sequence `warmup` times untimed and `repeats` (>= 2) timed, each
    from a zeroed output, and the last two runs' outputs returned; the
    clocks are `reps(...)`. The caller holds the card (its device
    fixture)."""
    if timing is None:
        warmup, runs = 0, RUNS
    else:
        warmup, runs = timing
        if warmup < 0 or runs < 2:
            raise ValueError("warmup >= 0 and repeats >= 2")
    ran = run_sequence(binary, workdir / tag, staged.buffers, staged.steps,
                       outs=[staged.out], runs=runs, warmup=warmup,
                       witness="each")
    Xs = ran.load(staged.out)[-2:] if timing else ran.load(staged.out)
    for X in Xs:
        assert X.shape == (staged.n_ao, staged.n_ao), X.shape
    return np.stack(Xs), ran


# --- the wall-clock sweep's input -------------------------------------------

_EXPS = (3.0, 0.8, 0.25)  # 3 primitives per shell: K_AB = K_CD = 9
_COEFS = (0.15, 0.55, 0.45)


def _canonical_pairs(idx: Mapping[int, list[int]], la: int, lb: int
                     ) -> list[tuple[int, int]]:
    """Every canonical shell pair of class `(la, lb)`, `la <= lb`; on a tie
    of `l` the lower index first."""
    if la == lb:
        s = idx[la]
        return [(s[i], s[j]) for i in range(len(s)) for j in range(i, len(s))]
    return [(a, b) for a in idx[la] for b in idx[lb]]


# #288's basis: fills every class of total L <= 2, and (00|12), (01|02),
# (01|11) at L = 3 and (01|12) at L = 4. #290: a class it cannot fill
# (fewer than TIMING_N_BRA bra pairs, or TIMING_N_KET ket pairs disjoint from
# them -- (ss|dd) has 10 ket pairs from 4 d shells) draws the same launch
# shape from DEFERRED_COUNTS, which fills every class of the ladder
# ((dd|dd): 2211 canonical pairs).
SYNTHETIC_COUNTS = (66, 34, 4)
DEFERRED_COUNTS = (66, 66, 66)


def synthetic_basis(*, side: float, seed: int,
                    counts=SYNTHETIC_COUNTS) -> Basis:
    """Shells of `l = 0, 1, 2` (`counts` of each, 3 primitives, exponents
    jittered by up to 10%) at uniform random centres in a cube of `side`."""
    rng = np.random.default_rng(seed)
    shells = []
    for l, n in enumerate(counts):  # noqa: E741
        for _ in range(n):
            c = tuple(float(x) for x in rng.uniform(0.0, side, 3))
            e = tuple(float(x * (1.0 + 0.1 * rng.uniform())) for x in _EXPS)
            shells.append(Shell(l, c, e, _COEFS))
    return Basis(tuple(shells))


def max_boys_T(basis: Basis, side: float) -> float:
    """An upper bound on the Boys argument `T = rho |PQ|^2` of any primitive
    quartet of `basis` (centres in a cube of `side`): `rho <= p_max / 2`."""
    p_max = 2.0 * max(e for s in basis.shells for e in s.exponents)
    return p_max / 2.0 * 3.0 * side * side


def synthetic_launch(cls: tuple[int, int, int, int], *, side: float,
                     seed: int, n_ket: int = TIMING_N_KET
                     ) -> tuple[FRec, Basis]:
    """The timing launch of `cls` (module docstring, THE SYNTHETIC LAUNCH);
    `n_ket` ket pairs, the last of the canonical list (#363's lever clock
    asks fewer than the F rows' TIMING_N_KET)."""
    la, lb, lc, ld = cls
    for counts in (SYNTHETIC_COUNTS, DEFERRED_COUNTS):
        basis = synthetic_basis(side=side, seed=seed, counts=counts)
        idx: dict[int, list[int]] = {}
        for i, s in enumerate(basis.shells):
            idx.setdefault(s.l, []).append(i)
        bra_all = _canonical_pairs(idx, la, lb)
        ket_all = _canonical_pairs(idx, lc, ld)
        bras = bra_all[:TIMING_N_BRA]
        kets = ket_all[-n_ket:]
        if len(bras) == TIMING_N_BRA and len(kets) == n_ket \
                and not set(bras) & set(kets):
            break
    assert len(bras) == TIMING_N_BRA and len(kets) == n_ket
    assert not set(bras) & set(kets), "the bra and ket pairs must differ"

    def rec(a: int, b: int) -> KF.PairRec:
        n = len(basis.shells[a].exponents) * len(basis.shells[b].exponents)
        return pair_record(basis, a, b, G=1.0, prims=range(n))

    pb, pk = KF.pack([rec(*p) for p in bras]), KF.pack([rec(*p) for p in kets])
    rng = np.random.default_rng(seed + 1)
    r = rng.uniform(-1.0, 1.0, (basis.n_ao, basis.n_ao))
    D = (r + r.T) / 2.0
    Dsh = KF.density_blocks(D, basis.offsets,
                            [s.n_cart for s in basis.shells])
    k = KF.build_f_kernel(cls)
    every = {
        "n_bra": pb.n, "n_ket": pk.n, "k_ab": pb.degree, "k_cd": pk.degree,
        "n_sh": len(basis.shells), "n_ao": basis.n_ao, "diag": 0,
        "tau": 0.0, "bra_prim": pb.prim, "AB": pb.sep, "G_ab": pb.G,
        "bra_sh": pb.sh, "bra_ao": pb.ao, "ket_prim": pk.prim, "CD": pk.sep,
        "G_cd": pk.G, "ket_sh": pk.sh, "ket_ao": pk.ao, "Dsh": Dsh,
        "D": tuple(D.ravel()), KF.FOCK: (0.0,) * (basis.n_ao ** 2)}
    args = {n: every[n] for n in KF.params_of(cls)}
    grid, (threads, _) = KF.launch_shape(pb.n, THREADS)
    return FRec(k, grid, threads, args), basis


# --- #358: the atomic-free write-back's launch path --------------------------
#
# ADDITIVE: everything above is the default path, unchanged. A `scatter`
# lever kernel (eri/fock/levers/scatter.py) writes no F: it stores into a slot
# buffer W, and the class-agnostic `f_scatter` kernel (kernel_f_scatter.py)
# sums W into F. So each launch of the scatter path is THREE device steps,
# in stream order: zero W (the step's `zero`, inside the sequence and so
# inside the clock -- it is the lever's cost), the F kernel, then the
# scatter over that launch's map (`el`, `ptr`, `src`, the launch's own
# buffers). The runner carries the scatter unit beside the F units.
#
# W is one buffer the size of the largest launch's n_slot. `f_only=True`
# stages the same sequence without the scatter launches (W written, F left
# zero): the tail measurement's arm, never checked against F.

SCATTER_THREADS = 128  # the scatter kernel's block width; not numerical


def _cls_of(k: Kernel) -> tuple[int, int, int, int]:
    """The F class of an F unit, from its name (`f_<la><lb><lc><ld>...`)."""
    digits = k.name.split("_")[1]
    if not (k.name.startswith("f_") and len(digits) == 4
            and digits.isdigit()):
        raise ValueError(f"{k.name} is not an F unit's name")
    return tuple(int(c) for c in digits)  # type: ignore[return-value]


def _check_scatter_signature(k: Kernel) -> None:
    names = [p.name for p in k.params]
    want = [n for n in (*SCALARS, "bra_prim", "AB", "G_ab", "bra_sh",
                        "bra_ao", "ket_prim", "CD", "G_cd", "ket_sh",
                        "ket_ao", *SHARED, FS.SLOTS) if n in names]
    if names != want or [p.name for p in k.params
                         if p.intent is Intent.OUT] != [FS.SLOTS]:
        raise ValueError(f"{k.name}: signature {names} is not the scatter "
                         "lever's F kernel (F replaced by W, no tap)")


def scatter_map_np(cls, n_ket: int, bra_ao, ket_ao) -> FS.ScatterMap:
    """`kernel_f_scatter.scatter_map`, vectorised for the timing launches
    (millions of slots); held equal to it on small launches by
    test_f_lever_scatter_device.py's fast tier."""
    n = [len(vrr.cartesians(x)) for x in cls]
    bao = np.asarray(bra_ao, dtype=np.int64).reshape(-1, 2)
    kao = np.asarray(ket_ao, dtype=np.int64).reshape(-1, 2)
    nb, nk = len(bao), len(kao)
    assert nk == n_ket
    n_slot = FS.n_slot(tuple(cls), nb, nk)
    row_w = SC.row_width(tuple(cls), nk)
    rows = np.empty(n_slot, dtype=np.int64)
    cols = np.empty(n_slot, dtype=np.int64)
    base = np.arange(nb, dtype=np.int64) * row_w
    ab = KF.BLOCKS[0]
    for i in range(n[ab[0]]):
        for j in range(n[ab[1]]):
            s = base + i * n[ab[1]] + j
            rows[s], cols[s] = bao[:, 0] + i, bao[:, 1] + j
    off = SC.ket_offsets(tuple(cls))
    o = [bao[:, 0][:, None], bao[:, 1][:, None], kao[:, 0][None, :],
         kao[:, 1][None, :]]
    k = np.arange(nk, dtype=np.int64)[None, :]
    for b in KF.BLOCKS[1:]:
        for i in range(n[b[0]]):
            for j in range(n[b[1]]):
                s = (base[:, None] + SC.n_ab(tuple(cls))
                     + (off[b] + i * n[b[1]] + j) * nk + k)
                rows[s] = np.broadcast_to(o[b[0]] + i, s.shape)
                cols[s] = np.broadcast_to(o[b[1]] + j, s.shape)
    width = int(max(rows.max(), cols.max())) + 1
    key = rows * width + cols
    src = np.argsort(key, kind="stable")  # ascending slots within an element
    uniq, counts = np.unique(key[src], return_counts=True)
    ptr = np.concatenate([[0], np.cumsum(counts)])
    el = np.stack([uniq // width, uniq % width], axis=1).ravel()
    # numpy arrays in the map's tuple fields: what staging reads is the same
    return FS.ScatterMap(len(uniq), n_slot, el, ptr, src)


def interpret_scatter(recs: Sequence[FRec], n_ao: int, *,
                      threads: int = SCATTER_THREADS) -> np.ndarray:
    """The recorded `scatter` launches through the interpreter, each F
    launch then its scatter, in order, into one zeroed `F` (NOT closed)."""
    F = np.zeros(n_ao * n_ao)
    sk = FS.build_scatter_kernel()
    for r in recs:
        cls = _cls_of(r.kernel)
        n_slot = FS.n_slot(cls, r.args["n_bra"], r.args["n_ket"])
        a = {}
        for p in r.kernel.params:
            v = np.zeros(n_slot) if p.name == FS.SLOTS else r.args[p.name]
            if p.array:
                v = np.array(v, dtype=np.int32 if p.type is Scalar.I32
                             else float)
            a[p.name] = v
        W = Interpreter(r.kernel, launch=Launch(r.grid, (r.threads, 1), 32),
                        boys=boys).run(**a)[FS.SLOTS]
        m = FS.scatter_map(cls, r.args["n_ket"], r.args["bra_ao"],
                           r.args["ket_ao"])
        grid, block = FS.launch_shape(m.n_el, threads)
        sa = {n: (np.array(v, dtype=np.int32) if n in ("el", "ptr", "src")
                  else v) for n, v in m.args(W, F, n_ao).items()}
        F = Interpreter(sk, launch=Launch(grid, block, 32)).run(**sa)[KF.FOCK]
    return np.asarray(F, dtype=float).reshape(n_ao, n_ao)


def fock_scatter_program(kernels: Sequence[Kernel], backend: str,
                         workdir: Path, *,
                         units: Mapping[str, Path] | None = None,
                         name: str = "fock_scatter") -> Program:
    """The runner over `kernels` (each a `scatter` lever F kernel) and the
    `f_scatter` unit (the block comment above)."""
    if not kernels or len({k.name for k in kernels}) != len(kernels):
        raise ValueError("fock_scatter_program takes distinct kernels")
    if name in {k.name for k in kernels} | {FS.NAME}:
        raise ValueError(f"the program {name!r} would overwrite its unit")
    for k in kernels:
        _check_scatter_signature(k)
    return runner_of([*kernels, FS.build_scatter_kernel()], backend, workdir,
                     units=units, name=name)


def fock_scatter_stage(recs: Sequence[FRec], kernels: Sequence[Kernel], *,
                       mapper=None, f_only: bool = False) -> Staged:
    """The scatter path's sequence for `recs` (each a `scatter` lever F
    launch of one of `kernels`): per launch `W` zeroed, the F kernel into
    `W`, then -- unless `f_only` -- `f_scatter` over the launch's map
    (computed here by `mapper`, `kernel_f_scatter.scatter_map` unless
    given) into `F`."""
    if not recs:
        raise ValueError("no launch to stage")
    mapper = FS.scatter_map if mapper is None else mapper
    index = {k.name: k for k in kernels}
    n_ao = int(_global(recs, "n_ao")[0])
    buffers: dict[str, object] = {KF.FOCK: Zeros((n_ao, n_ao))}
    for n in SHARED:
        buffers[n] = _global(recs, n).astype(np.float64)
    sk = FS.build_scatter_kernel()
    steps, n_slot = [], 0
    for i, r in enumerate(recs):
        if index.get(r.name) != r.kernel:
            raise ValueError(f"{r.name} is not one of the program's kernels")
        a = r.args
        cls = _cls_of(r.kernel)
        m = mapper(cls, a["n_ket"], a["bra_ao"], a["ket_ao"])
        assert m.n_slot == FS.n_slot(cls, a["n_bra"], a["n_ket"])
        n_slot = max(n_slot, m.n_slot)
        args: dict[str, object] = {}
        for p in r.kernel.params:
            if p.name == FS.SLOTS:
                args[p.name] = FS.SLOTS
            elif not p.array:
                args[p.name] = a[p.name]
            elif p.name in SHARED:
                args[p.name] = p.name
            else:
                buf = f"l{i}.{p.name}"
                buffers[buf] = np.ravel(np.asarray(a[p.name],
                                                   dtype=_NP[p.type]))
                args[p.name] = buf
        assert r.grid == a["n_bra"], (r.grid, a["n_bra"])
        steps.append(Step(r.kernel, r.grid, (r.threads, 1), args,
                          zero=(FS.SLOTS,)))
        if f_only:
            continue
        sa: dict[str, object] = {}
        for n, v in m.args(FS.SLOTS, KF.FOCK, n_ao).items():
            if n in ("el", "ptr", "src"):
                buf = f"l{i}.{n}"
                buffers[buf] = np.asarray(v, dtype=np.int32)
                v = buf
            sa[n] = v
        grid, block = FS.launch_shape(m.n_el, SCATTER_THREADS)
        steps.append(Step(sk, grid, block, sa))
    buffers[FS.SLOTS] = Zeros((n_slot,))
    return Staged(buffers, tuple(steps), KF.FOCK, n_ao)
