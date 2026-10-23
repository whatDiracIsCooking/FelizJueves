"""The J chain on the card (#157): every new pair kernel through (ff), and
`J_mn` with EVERY arithmetic step a generated kernel, through d, on the RTX
3080 and the RX 9060 XT.

TWO PARTS, TWO COMPILE COSTS -- reach is bought with the cheap kernels, not
the expensive driver:

PART 1, PER PAIR CLASS, (ss) THROUGH (ff). For each pair class `(la, lb)`,
`la <= lb <= 3`, of KERNEL_GEN_CMD's pair ladder (#156: `dq_{la}_{lb}` and
`jab_{la}{lb}`, 10 classes, 20 units), each unit is compiled ALONE into a
small harness (`plan_driver.dq_program` / `jab_program`: one translation
unit, no `[p|q]`, no Boys) and run on the card over every record of every
unique shell pair of that class in the tracked HF/cc-pVTZ basis (the one
fixture whose shells reach f; s-f on F, s-d on H), against the host on the
SAME inputs: `ao.j.dq_from_host` for `D_q` / `Dmax`, and `ao.j.jab_from_host`
applied pair by pair for the scatter over a seeded `J_p`. Under
devtools/kernel-tier.sh the unit compiled is THE FILE THE GENERATE PHASE
WROTE (`KERNEL_TIER_OUT`, already compiled by the tier's compile phase, #92's
filter): the test asserts it is byte-identical to `emit(...)` and includes
that file, as test_kernel_device.py does for `[p|q]`. The witness chain for
f: `E` is held to the closed-form Hermite identity (#152) and, through
`ao/eri.py`, to libcint's ERI blocks up to (ff|ff) (test_ao_eri.py); the host
`D_q` and scatter are built from that `E`. No f `J` fixture is needed.

PART 2, PER MOLECULE, THROUGH d. `J_mn` for H2/STO-3G, H2O/STO-3G (at 1 and
3 batches) and H2O/cc-pVDZ, every step on the card, on MILESTONE 6'S
DRIVER (#144, `plan_driver.driver`), not a new one:

  1. DQ: every ket pair class's `Target.DQ` kernel (`plan_driver.dq_program`,
     one launch per class); `D_q` and `Dmax` READ BACK -- the chain's one
     round-trip (j_engine.md, "The round-trip");
  2. the host keys each ket record `G * Dmax` with the KERNEL's `Dmax` and
     sorts (`ao.drive._chain_plan`, #155's plan builder: `launch`'s own
     stable descending sort, no host pre-contraction);
  3. `[p|q]`: milestone 6's driver, its kets staged from the kernel's `D_q` /
     `Dmax` through `ao.staged`'s `ket_data=` hook (the plan-key check, bit
     for bit, intact), one `J_p` buffer per bra class;
  4. JAB: in the SAME process, after every `[p|q]` launch on the same stream,
     one `Target.JAB` launch per bra pair class reads its `J_p` buffer on the
     card and adds into one zeroed `J_mn`, read back.

No `[p|q]` above L = 8 enters a driver (asserted): the reach stays
milestone 6's. f in a whole-molecule driver would need a new f `J` fixture
and `[p|q]` up to L = 12, a separate decision.

THE TOLERANCES, stated where they are asserted, with `u = 2^-53`,
`gamma_n = n u / (1 - n u)`, `L = la + lb` (or `lc + ld`):

  E per side. Each axis' `E^{ij}_t` is the recurrence of `ao.hermite` from
  `E^{00}_0 = exp(zeta X_PA X_PB)`, whose ARGUMENT is formed in the same
  order from the same host-supplied `X_PA`, `X_PB` on both sides (bitwise
  equal) and whose `exp` is within 1 ulp (2u) on each vendor's libm and the
  host's. Each raise adds at most 3 roundings (a product and two adds), and
  the 3D product 2, so a computed entry is within `gamma_{3L+8}` of the
  ABSOLUTE-VALUE recurrence `Eabs` (the same recurrence over `|X_PA|`,
  `|X_PB|`, which bounds `|E|`): `N_E = 4L + 8` is taken, above that count.
  FMA contraction only removes roundings.

  DQ (part 1, and per class in part 2): per side `D_q = w SUM_r x_r (s E_rq)`
  is within `gamma_{N_DQ} S_q` of exact, `N_DQ = N_E + n_cd + 3`,
  `S_q = |w| SUM_r |x_r| |s| Eabs_rq`; so
      |D_dev - D_host| <= 2 gamma_{N_DQ} S_q,   element by element,
  and `|Dmax_dev - Dmax_host| <= max_q` of that (max is 1-Lipschitz). The
  kernel has no atomic and no cross-thread reduction, so its two runs in one
  process are asserted BIT-IDENTICAL, and `Dmax_dev == max_q |D_dev|` exactly.

  JAB device vs host on the same `J_p`: each record's `v_r = SUM_p (s E_rp)
  Jp_p` is within `gamma_{N_E + n_p + 2}` of exact relative to
  `SUM_p |s| Eabs_rp |Jp_p|`, and an element of `J` is the sum of the
  `K` records of the one shell pair that owns it (the scatter rule: every
  element is written by exactly one bra shell pair), `K` adds more; so
      |J_dev - J_host| <= 2 gamma_{N_JAB} T,  N_JAB = N_E + n_p + 2 + K,
  `T = SUM_k SUM_p |s_k| Eabs_k,rp |Jp_k,p|` over the owning pair's records.

  THE fp64-ATOMIC RUN-TO-RUN TOLERANCE (JAB; `_r2r_bound`). Two runs of the
  same binary compute every `c = fl(s E)` identically (deterministic code,
  same inputs). Where the `J_p` rows differ between the runs (part 2: they
  come out of `[p|q]`'s own atomics), each `v` differs by at most
  `SUM_p |c| |dJp| + gamma_{n_p+1} SUM_p |c| (|Jp1| + |Jp2|)`, and the `K`
  atomic adds into an element, unordered, are each within `gamma_{K-1}` of
  their exact sum: with `M = max(|Jp1|, |Jp2|)`, `Delta = SUM |s| Eabs |dJp|`,
  `T = SUM |s| Eabs M` and `|c| <= (1 + gamma_{N_E+1}) |s| Eabs`,
      |J(run 1) - J(run 2)| <= (1 + gamma_{N_E+1}) [Delta + 2 gamma_{n_p+1} T
                                + 2 gamma_{K-1} (1 + gamma_{n_p+1}) T].
  When the `J_p` the runs read are bit-identical (part 1: one host buffer),
  `v` is too and only the atomic term is left:
  `2 gamma_{K-1} (1 + gamma_{N_E + n_p + 2}) T`. A pair with one record
  (`K = 1`) therefore must repeat bit for bit. Part 2 also asserts
  milestone 6's `J_p` run-to-run bound (`test_j_launched_ondevice.py`'s
  module docstring), with `A_p` over the DQ kernel's staged `D_q`.

  Part 2's outside witnesses, inherited, not re-derived and not loosened:
  libcint `|J_dev - J_pyscf| <= 1e-12 scale_J` (test_ao_j.py's JK_TOL) and
  the launch oracle `launch.walk` over the same plan (host `D_q`,
  `reference.pq`, fsum) `<= 1e-13 max|J|` (ORACLE_TOL); and, to localise
  step 4, the host scatter (`ao.staged._transform`) of the card's own `J_p`
  within the JAB bound above.

COMPILE-ONLY COMPANIONS (`cuda` / `hip`, no `gpu`) build both parts' programs
with devtools/config.sh's NVCC_FLAGS / HIPCC_FLAGS, where a warning located
in generated text fails (#92; a planted warning shows it fires on a pair
unit), so a vendor spelling problem can be fixed without the card. Part 2's
companion plans the driver from the HOST `D_q` (no card to run DQ on): the
driver's text depends on the per-class record counts and kernels, not on the
ket order a key sorts.

WHAT SKIPS: the device tests are `gpu` plus `cuda` or `hip`, behind the card
lock and the VRAM probe, so on a box without the card they skip and the run is
green and empty (kernel-tier.sh exits 2). Nothing here is a measurement:
no registers, spills, wall-clock or counters of the new kernels.
"""

from __future__ import annotations

import math
import os
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from plan_driver import (
    RUNS,
    DQClass,
    chain_counts,
    compile_checked,
    counts,
    dq_classes,
    dq_program,
    dq_read_back,
    dq_stage,
    driver,
    jab_program,
    jab_records,
    jab_rows,
    jab_stage,
    read_back,
    read_back_j,
    stage,
    transform,
)
from test_j_launched_ondevice import (
    BOUND,
    JK_TOL,
    ORACLE_TOL,
    _abs_jp,
    _check_counts,
    _gamma,
)
from test_kernel_device import BACKEND, _toolchain
from tiers import FAST_L_MAX, by_l, is_fast

from felizjueves.ao.density import epair
from felizjueves.ao.drive import _chain_plan, _unique_pairs
from felizjueves.ao.j import (
    cart_pairs,
    coulomb_matrix_and_scale,
    dq_from_host,
    jab_from_host,
)
from felizjueves.ao.staged import KetData
from felizjueves.emit import __main__ as emit_cli
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.kernel_dq import build_dq_kernel, dq_shape
from felizjueves.kernel_dq import launch_shape as dq_launch_shape
from felizjueves.kernel_jab import build_jab_kernel, jab_launch_shape
from felizjueves.launch import LaunchPlan, bound_value, walk
from felizjueves.mmd.hermite import _raise, e_block, hermite_order
from felizjueves.reference.pq import pq as ref_pq

PART1 = "hf_ccpvtz"
PAIR_CLASSES = tuple(emit_cli.pair_classes(emit_cli.DEFAULT_PAIR_L))
# A pair class's L is la + lb (tiers.py): (ss) through (pd), (dd) included,
# are the PR gate; (df) and (ff) are `slow`.
PAIR_PARAMS = [by_l(c, sum(c), id=f"{c[0]}{c[1]}") for c in PAIR_CLASSES]
# (fixture, n_batches, slow): part 2, through d. H2O/cc-pVDZ's whole chain is
# ~110 s on the RTX 3080 -- a third of the gate on its own -- so it is `slow`;
# H2O/STO-3G reaches every class it runs below L = 4.
CHAIN_CELLS = (
    ("h2_sto3g_j", 1, False),
    ("h2o_sto3g_j", 1, False),
    ("h2o_sto3g_j", 3, False),
    ("h2o_ccpvdz_j", 1, True),
)
CHAIN_PARAMS = [pytest.param(name, n, id=f"{name}-b{n}",
                             marks=(pytest.mark.slow,) if slow else ())
                for name, n, slow in CHAIN_CELLS]
L_MAX_DRIVER = 8  # milestone 6's reach: no [p|q] above it enters a driver


# ---------------------------------------------------------------------------
# Inputs, and the bounds' scales
# ---------------------------------------------------------------------------


@cache
def _case(name: str):
    """(basis, D) of a tracked fixture."""
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return ob.from_fixture(fx), np.array(fx.density)


@cache
def _dq_classes(name: str) -> dict[tuple[int, int], DQClass]:
    basis, D = _case(name)
    return {dc.cls: dc for dc in dq_classes(basis, D)}


def _pairs_of(basis, cls) -> list[tuple[int, int]]:
    """Unique shell pairs `a >= b` of class `cls`, in sweep order."""
    sh = basis.shells
    return [(a, b) for a in range(len(sh)) for b in range(a + 1)
            if (sh[a].l, sh[b].l) == cls]


def _n_e(L: int) -> int:
    return 4 * L + 8  # above E's 3L + 8 roundings (module docstring)


def _e_abs(la: int, lb: int, zeta: float, xpa, xpb) -> np.ndarray:
    """`Eabs[r, t]`: the recurrence over `|X_PA|`, `|X_PB|` from the same
    `E^{00}_0`, rows `cart_pairs(la, lb)`, columns `hermite_order(la + lb)`;
    `|E| <= Eabs` entry by entry (positive coefficients throughout)."""
    h = 0.5 / zeta
    axes: dict[tuple[int, int, int], list[float]] = {}

    def axis(d: int, i: int, j: int) -> list[float]:
        if (d, i, j) not in axes:
            e = [math.exp(zeta * xpa[d] * xpb[d])]
            for _ in range(i):
                e = _raise(e, abs(xpa[d]), h)
            for _ in range(j):
                e = _raise(e, abs(xpb[d]), h)
            axes[(d, i, j)] = e
        return axes[(d, i, j)]

    ts = hermite_order(la + lb)
    out = np.zeros((len(cart_pairs(la, lb)), len(ts)))
    for r, (a, b) in enumerate(cart_pairs(la, lb)):
        ex = [axis(d, a[d], b[d]) for d in range(3)]
        for c, t in enumerate(ts):
            if all(t[d] <= a[d] + b[d] for d in range(3)):
                out[r, c] = ex[0][t[0]] * ex[1][t[1]] * ex[2][t[2]]
    return out


def _scatter(M: np.ndarray, oa: int, ob_: int, na: int, nb: int,
             vals: np.ndarray) -> None:
    """Put a shell pair's per-Cartesian-pair values where its scatter writes
    (kernel_jab's rule): the block and its mirror, the diagonal pair's lower
    triangle mirrored. Every element is owned by one pair, so it is SET."""
    for i in range(na):
        for j in range(nb):
            if oa == ob_ and i < j:
                continue
            M[oa + i, ob_ + j] = M[ob_ + j, oa + i] = vals[i * nb + j]


def _jab_scales(la: int, lb: int, data: np.ndarray, ints: np.ndarray,
                jp1: np.ndarray, jp2: np.ndarray, n_ao: int, into=None):
    """`(T, Delta, K)` per element of `J` for one JAB launch's records
    (module docstring): `T = SUM |s| Eabs max(|Jp1|, |Jp2|)`, `Delta = SUM
    |s| Eabs |Jp1 - Jp2|` over the owning shell pair's records, `K` their
    count. `into` accumulates several launches (disjoint elements)."""
    T, Dl, K = into if into is not None else (
        np.zeros((n_ao, n_ao)), np.zeros((n_ao, n_ao)), np.zeros((n_ao, n_ao)))
    na, nb = (la + 1) * (la + 2) // 2, (lb + 1) * (lb + 2) // 2
    pairs: dict[tuple[int, int], list[int]] = {}
    for i, (oa, ob_, _) in enumerate(ints.astype(int)):
        pairs.setdefault((int(oa), int(ob_)), []).append(i)
    for (oa, ob_), idx in pairs.items():
        t = np.zeros(na * nb)
        dl = np.zeros(na * nb)
        for i in idx:
            row = int(ints[i, 2])
            E = abs(data[i, 7]) * _e_abs(la, lb, data[i, 0], data[i, 1:4],
                                         data[i, 4:7])
            t += E @ np.maximum(np.abs(jp1[row]), np.abs(jp2[row]))
            dl += E @ np.abs(jp1[row] - jp2[row])
        _scatter(T, oa, ob_, na, nb, t)
        _scatter(Dl, oa, ob_, na, nb, dl)
        _scatter(K, oa, ob_, na, nb, np.full(na * nb, float(len(idx))))
    return T, Dl, K


def _r2r_bound(T, Dl, K, L: int, n_p: int, same_jp: bool) -> np.ndarray:
    """THE fp64-ATOMIC RUN-TO-RUN TOLERANCE (module docstring), per element."""
    g = np.vectorize(_gamma)
    gk = g(np.maximum(K - 1, 0).astype(int))
    if same_jp:
        return 2 * gk * (1 + _gamma(_n_e(L) + n_p + 2)) * T
    return (1 + _gamma(_n_e(L) + 1)) * (
        Dl + 2 * _gamma(n_p + 1) * T + 2 * gk * (1 + _gamma(n_p + 1)) * T)


def _jab_tol(T, K, L: int, n_p: int) -> np.ndarray:
    """`2 gamma_{N_JAB} T`, device vs host on the same `J_p`, per element."""
    n = (_n_e(L) + n_p + 2 + K).astype(int)
    return 2 * np.vectorize(_gamma)(n) * T


def _dq_host(basis, dc: DQClass) -> tuple[np.ndarray, np.ndarray]:
    """`(D_host, S)` in record order: `ao.j.dq_from_host` per shell pair on
    the class's own `Dcd` rows and weights, and the DQ bound's scale."""
    shells = basis.shells
    Dh, S = [], []
    for i, (c, d) in enumerate(dc.members):
        ep = epair(shells[c], shells[d])
        w = 1.0 if c == d else 2.0
        dcd = dc.dcd[i]
        Dh += dq_from_host(ep, [float(x) for x in dcd], w)
        for P, zeta, s in ep.recs:
            Ea = _e_abs(ep.la, ep.lb, zeta, [P[k] - ep.A[k] for k in range(3)],
                        [P[k] - ep.B[k] for k in range(3)])
            S.append(abs(w) * abs(s) * (np.abs(dcd) @ Ea))
    return np.array(Dh), np.array(S)


def _check_dq(basis, dc: DQClass, runs) -> float:
    """One DQ class on the card against the host (module docstring); the
    worst `|D_dev - D_host| / S` (the tolerance is `2 gamma_{N_DQ}`)."""
    (D0, M0), (D1, M1) = (r[dc.cls] for r in runs)
    assert np.all(np.isfinite(D0)) and np.all(np.isfinite(M0)), (
        f"DQ {dc.cls}: an unwritten (poisoned) or non-finite output")
    assert np.array_equal(D0, D1) and np.array_equal(M0, M1), (
        f"DQ {dc.cls}: two runs differ -- no atomic, so they may not")
    assert np.array_equal(M0, np.max(np.abs(D0), axis=1)), (
        f"DQ {dc.cls}: Dmax is not max_q |D_q| of the kernel's own D_q")
    Dh, S = _dq_host(basis, dc)
    n_cd, _ = dq_shape(*dc.cls)
    tol = 2 * _gamma(_n_e(sum(dc.cls)) + n_cd + 3) * S
    err = np.abs(D0 - Dh)
    assert np.all(err <= tol), (
        f"DQ {dc.cls}: worst |D - D_host| / S = "
        f"{float(np.max(err / np.where(S > 0, S, 1))):.3e} over "
        f"{2 * _gamma(_n_e(sum(dc.cls)) + n_cd + 3):.3e}")
    assert np.all(np.abs(M0 - np.max(np.abs(Dh), axis=1))
                  <= np.max(tol, axis=1)), f"DQ {dc.cls}: Dmax off the host's"
    return float(np.max(np.where(S > 0, err / np.where(S > 0, S, 1), 0.0)))


def _ratio(err: np.ndarray, scale: np.ndarray) -> float:
    return float(np.max(np.where(scale > 0, err / np.where(scale > 0, scale, 1),
                                 0.0)))


def _tier_units(backend: str, kernels) -> dict[str, Path] | None:
    """Under the kernel tier, the generate phase's own files for `kernels`,
    each asserted byte-identical to `emit(...)`; else None (the harness
    writes the text itself)."""
    out = os.environ.get("KERNEL_TIER_OUT")
    if not out or os.environ.get("KERNEL_TIER_BACKEND") != backend:
        return None
    emit, arch, ext = BACKEND[backend]
    got = {}
    for k in kernels:
        path = Path(out) / f"{k.name}.{ext}"
        assert path.read_text() == emit(k, arch), (
            f"{path} differs from what emit() writes now -- a stale generate "
            "phase, or a nondeterministic emitter")
        got[k.name] = path
    return got


# ---------------------------------------------------------------------------
# Part 1: each pair class's DQ and JAB unit alone, through (ff)
# ---------------------------------------------------------------------------


def _jab_inputs(cls: tuple[int, int]):
    """Every unique shell pair of `cls` in HF/cc-pVTZ, their records laid
    into one `J_p` buffer in REVERSED pair order (so `row[k] != k`), and a
    seeded `J_p`: `(members, data, ints, jp)`."""
    basis, _ = _case(PART1)
    pairs = _pairs_of(basis, cls)
    sizes = [len(epair(basis.shells[a], basis.shells[b]).recs) for a, b in pairs]
    starts, at = [0] * len(pairs), sum(sizes)
    for i, n in enumerate(sizes):
        at -= n
        starts[i] = at
    members = [(a, b, s) for (a, b), s in zip(pairs, starts, strict=True)]
    data, ints = jab_rows(basis, members)
    rng = np.random.default_rng(157 + 10 * cls[0] + cls[1])
    jp = rng.uniform(-1.0, 1.0, (sum(sizes), len(hermite_order(sum(cls)))))
    return members, data, ints, jp


def _jab_host(cls, members, jp) -> np.ndarray:
    basis, _ = _case(PART1)
    sh, off = basis.shells, basis.offsets
    J = [[0.0] * basis.n_ao for _ in range(basis.n_ao)]
    for a, b, start in members:
        ep = epair(sh[a], sh[b])
        jab_from_host(ep, jp[start:start + len(ep.recs)].tolist(),
                      (off[a], off[b]), J)
    return np.array(J)


def _pair_programs(backend: str, cls, workdir: Path):
    """Both part-1 programs of `cls`: the units are the tier's when it ran."""
    basis, _ = _case(PART1)
    units = _tier_units(backend, [build_dq_kernel(cls), build_jab_kernel(cls)])
    dc = _dq_classes(PART1)[cls]
    dqp = dq_program([dc], backend, workdir / "dq", units=units)
    members, data, ints, jp = _jab_inputs(cls)
    jabp = jab_program(cls, len(data), jp.shape[0], basis.n_ao, backend,
                       workdir / "jab", units=units)
    return dc, dqp, (members, data, ints, jp), jabp, units is not None


def _part1(backend: str, cls, request, run_device_binary, tmp_path: Path):
    tc = _toolchain(backend, request)
    arch = BACKEND[backend][1]
    basis, _ = _case(PART1)
    dc, dqp, (members, data, ints, jp), jabp, tier = _pair_programs(
        backend, cls, tmp_path)

    # DQ: every record of the class, one launch, two runs
    dq_bin = compile_checked(tc, dqp, tmp_path / "dq_bin")
    src, dst = tmp_path / "dq.in", tmp_path / "dq.out"
    dq_stage([dc]).tofile(src)
    out = run_device_binary(dq_bin, str(src), str(dst), vram=backend)
    assert f"fj-dq classes 1 launches {RUNS} runs {RUNS}" in out, out
    dq_worst = _check_dq(basis, dc, dq_read_back(
        [dc], np.fromfile(dst, dtype=np.float64)))

    # JAB: every record of the class, one launch per run, from a zeroed J
    jab_bin = compile_checked(tc, jabp, tmp_path / "jab_bin")
    src, dst = tmp_path / "jab.in", tmp_path / "jab.out"
    jab_stage(jp, data, ints).tofile(src)
    out = run_device_binary(jab_bin, str(src), str(dst), vram=backend)
    assert f"fj-jab launches {RUNS} runs {RUNS}" in out, out
    n = basis.n_ao
    J = np.fromfile(dst, dtype=np.float64).reshape(RUNS, n, n)
    host = _jab_host(cls, members, jp)
    L, n_p = sum(cls), jp.shape[1]
    T, Dl, K = _jab_scales(cls[0], cls[1], data, ints, jp, jp, n)
    owned = K > 0
    assert np.all(np.isfinite(J)) and np.all(J[:, ~owned] == 0.0), (
        f"JAB {cls}: a write outside the class's blocks")
    r2r = np.abs(J[0] - J[1])
    assert np.all(r2r <= _r2r_bound(T, Dl, K, L, n_p, same_jp=True)), (
        f"JAB {cls}: run to run {float(np.max(r2r)):.3e} over the "
        "fp64-atomic bound")
    err = np.abs(J[0] - host)
    assert np.all(err <= _jab_tol(T, K, L, n_p)), (
        f"JAB {cls}: worst |J - J_host| / T = {_ratio(err, T):.3e}")
    kmax = int(K.max())
    print(f"\n{backend} {arch.name} pair class {cls} ({PART1}, "
          f"{'tier units' if tier else 'emitted here'}): DQ {dc.n_rec} records "
          f"/ {len(dc.members)} shell pairs, worst |D-D_host|/S {dq_worst:.2e} "
          f"(tol {2 * _gamma(_n_e(L) + dq_shape(*cls)[0] + 3):.2e}), runs "
          f"bit-identical; JAB {len(data)} records / {len(members)} shell "
          f"pairs, worst |J-J_host|/T {_ratio(err, T):.2e} (tol <= "
          f"{2 * _gamma(_n_e(L) + n_p + 2 + kmax):.2e}), run to run "
          f"{_ratio(r2r, T):.2e}/T (bit-identical: "
          f"{bool(np.array_equal(J[0], J[1]))}), K <= {kmax}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", PAIR_PARAMS)
def test_cuda_pair_kernels_match_the_host(cls, request, run_device_binary,
                                          tmp_path) -> None:
    """Part 1 on the RTX 3080: `dq_{la}_{lb}` and `jab_{la}{lb}` alone."""
    _part1("cuda", cls, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", PAIR_PARAMS)
def test_hip_pair_kernels_match_the_host(cls, request, run_device_binary,
                                         tmp_path) -> None:
    """Part 1 on the RX 9060 XT: `dq_{la}_{lb}` and `jab_{la}{lb}` alone."""
    _part1("hip", cls, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Part 2: J_mn through the whole chain on milestone 6's driver, through d
# ---------------------------------------------------------------------------


def _plan_from(basis, dq_rows, n: int) -> LaunchPlan:
    """`ao.drive._chain_plan` over `dq_rows` (`(i, j, prim) -> (D_q,
    Dmax)`): kets keyed `G * Dmax` with THAT `Dmax`, sorted."""
    pairs = _unique_pairs(basis)
    gs = {(a, b): [bound_value(BOUND, zeta,
                               basis.shells[a].l + basis.shells[b].l)
                   for _, zeta, _ in epair(basis.shells[a],
                                           basis.shells[b]).recs]
          for a, b in pairs}
    plan = _chain_plan(basis, pairs, gs, dq_rows, n_batches=n, tau=0.0,
                       strict=False, resolve=None)
    assert max(cl.L for cl in plan) <= L_MAX_DRIVER, (
        "a [p|q] above L = 8 would enter the driver")
    return plan


def _rows_of(classes, per_class) -> dict:
    """`(i, j, prim) -> (D_q row, Dmax)` from per-class record-order outputs."""
    out = {}
    for dc in classes:
        D, M = per_class[dc.cls]
        k = 0
        for c, d in dc.members:
            n_rec = int(np.sum(dc.pair == dc.members.index((c, d))))
            for prim in range(n_rec):
                out[(c, d, prim)] = (D[k], float(M[k]))
                k += 1
        assert k == len(D)
    return out


def _ket_data(basis, dq_rows):
    """`ao.staged`'s `ket_data=` hook over `dq_rows`, gathered in the plan's
    sorted order -- `ao.drive.coulomb_matrix_chain`'s own gather."""
    def ket_data(recs, n_q: int) -> KetData:
        P = [epair(basis.shells[r.i], basis.shells[r.j]).recs[r.prim]
             for r in recs]
        got = [dq_rows[(r.i, r.j, r.prim)] for r in recs]
        return KetData(
            Q=np.array([p[0] for p in P], dtype=np.float64),
            eta=np.array([p[1] for p in P], dtype=np.float64),
            D=np.array([g[0] for g in got], dtype=np.float64).reshape(
                len(recs), n_q),
            Dmax=np.array([g[1] for g in got], dtype=np.float64))
    return ket_data


def _host_dq_rows(basis, classes) -> dict:
    """The host's `D_q` / `Dmax` per record (part 2's compile-only plan)."""
    per = {}
    for dc in classes:
        Dh, _ = _dq_host(basis, dc)
        per[dc.cls] = (Dh, np.max(np.abs(Dh), axis=1))
    return _rows_of(classes, per)


def _part2(backend: str, name: str, n: int, request, run_device_binary,
           tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    arch = BACKEND[backend][1]
    basis, D = _case(name)
    assert max(s.l for s in basis.shells) <= 2, "part 2 is through d"

    # step 2 on the card: D_q and Dmax per ket pair class; THE ROUND-TRIP
    classes = dq_classes(basis, D)
    dqp = dq_program(classes, backend, tmp_path / "dq")
    dq_bin = compile_checked(tc, dqp, tmp_path / "dq_bin")
    src, dst = tmp_path / "dq.in", tmp_path / "dq.out"
    dq_stage(classes).tofile(src)
    out = run_device_binary(dq_bin, str(src), str(dst), vram=backend)
    assert (f"fj-dq classes {len(classes)} launches {len(classes) * RUNS} "
            f"runs {RUNS}") in out, out
    dq_runs = dq_read_back(classes, np.fromfile(dst, dtype=np.float64))
    dq_worst = max(_check_dq(basis, dc, dq_runs) for dc in classes)
    dq_rows = _rows_of(classes, dq_runs[0])

    # the sort on the card's Dmax; steps 3 and 4 on milestone 6's driver
    plan = _plan_from(basis, dq_rows, n)
    ket_data = _ket_data(basis, dq_rows)
    drv = driver(plan, backend, tmp_path / "gen", chain=basis)
    binary = compile_checked(tc, drv, tmp_path / "driver")
    src, dst = tmp_path / "in.bin", tmp_path / "out.bin"
    stage(plan, basis, D, drv, ket_data=ket_data).tofile(src)
    out = run_device_binary(binary, str(src), str(dst), vram=backend)
    _check_counts(counts(out), drv, plan, warmup=0, reps=RUNS)
    nj = len(drv.jab)
    assert chain_counts(out) == dict(
        jab_classes=nj, uploads_jab=nj, jab_launches=nj * RUNS,
        j_memsets=RUNS, j_readbacks=RUNS), out
    raw = np.fromfile(dst, dtype=np.float64)
    jps, Js = read_back(drv, raw), read_back_j(drv, raw)
    for r in range(RUNS):
        assert np.all(np.isfinite(Js[r])), "non-finite J"
        for lp, jp in jps[r].items():
            assert np.all(np.isfinite(jp)), f"non-finite J_p[{lp}]"

    # J_p run to run: milestone 6's fp64-atomic bound, A_p over the staged
    # (DQ kernel's) D_q
    A = _abs_jp(plan, name, basis=basis, D=D, ket_data=ket_data)
    for lp, K in drv.writers.items():
        diff = np.abs(jps[0][lp] - jps[1][lp])
        assert np.all(diff <= 2.0 * _gamma(K - 1) * A[lp]), (
            f"J_p[{lp}] run to run: {float(np.max(diff)):.3e} over the "
            f"{K}-atomic bound")

    # J run to run (JAB's fp64 atomics, and J_p's run-to-run difference)
    n_ao = basis.n_ao
    acc = (np.zeros((n_ao, n_ao)), np.zeros((n_ao, n_ao)),
           np.zeros((n_ao, n_ao)))
    per_class = {}
    for x in drv.jab:
        one = _jab_scales(x.la, x.lb, x.data, x.ints, jps[0][x.lp],
                          jps[1][x.lp], n_ao)
        per_class[(x.la, x.lb)] = one
        for m, part in zip(acc, one, strict=True):
            m += part
    T, Dl, K = acc
    assert np.all(K > 0), "an element of J no bra pair owns"
    r2r_ok = np.ones((n_ao, n_ao), bool)
    host_tol = np.zeros((n_ao, n_ao))
    for x in drv.jab:
        t, dl, k = per_class[(x.la, x.lb)]
        mine = k > 0
        n_p = len(hermite_order(x.lp))
        r2r_ok[mine] = (np.abs(Js[0] - Js[1])[mine]
                        <= _r2r_bound(t, dl, k, x.lp, n_p, False)[mine])
        host_tol[mine] = _jab_tol(t, k, x.lp, n_p)[mine]
    assert np.all(r2r_ok), (
        f"J run to run {float(np.max(np.abs(Js[0] - Js[1]))):.3e} over the "
        "fp64-atomic bound")

    # step 4 localised: the host scatter of the card's own J_p
    J = Js[0]
    Jh = transform(basis, plan, jps[0])
    err_h = np.abs(J - Jh)
    assert np.all(err_h <= host_tol), (
        f"J vs the host scatter of the card's J_p: worst / T "
        f"{_ratio(err_h, T):.3e}")

    # the outside witnesses, inherited
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    ref = fx.arrays[manifest.J_ARRAY]
    scale = np.array(coulomb_matrix_and_scale(basis, D, pq=ref_pq)[1])
    oracle = walk(plan, basis, D, pq=ref_pq)
    Jo = np.array(oracle.J)
    assert oracle.n_skipped == 0
    jk = float(np.max(np.abs(J - ref) / scale))
    rel = float(np.max(np.abs(J - Jo)) / np.max(np.abs(Jo)))

    print(f"\n{backend} {arch.name} {name} n_batches={n}: DQ "
          f"{[dc.cls for dc in classes]} (worst |D-D_host|/S {dq_worst:.2e}, "
          f"runs bit-identical); [p|q] "
          f"{' '.join(f'({c.lp},{c.lq})b{c.n_batches}' for c in plan)} "
          f"(L <= {max(c.L for c in plan)}), refused {list(plan.refused)}; "
          f"JAB {[(x.la, x.lb) for x in drv.jab]}; {len(drv.launches)} [p|q] "
          f"+ {nj} JAB launches per run; max|J - get_jk|/scale_J = {jk:.2e} "
          f"(tol {JK_TOL:g}); max|J - oracle|/max|J| = {rel:.2e} (tol "
          f"{ORACLE_TOL:g}); |J - host scatter of J_p|/T = "
          f"{_ratio(err_h, T):.2e}; J_p runs bit-identical "
          f"{all(np.array_equal(jps[0][lp], jps[1][lp]) for lp in jps[0])}, "
          f"J runs bit-identical {bool(np.array_equal(Js[0], Js[1]))}")
    assert rel <= ORACLE_TOL, f"{name} on {arch.name}: J vs the launch oracle"
    assert np.all(np.abs(J - ref) <= JK_TOL * scale), (
        f"{name} on {arch.name}: J vs libcint, worst err/scale {jk:.2e}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "n"), CHAIN_PARAMS)
def test_cuda_j_through_the_chain(name, n, request, run_device_binary,
                                  tmp_path) -> None:
    """Part 2 on the RTX 3080: `J_mn`, every step generated, vs libcint."""
    _part2("cuda", name, n, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "n"), CHAIN_PARAMS)
def test_hip_j_through_the_chain(name, n, request, run_device_binary,
                                 tmp_path) -> None:
    """Part 2 on the RX 9060 XT: `J_mn`, every step generated, vs libcint."""
    _part2("hip", name, n, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions (`cuda` / `hip`, no `gpu`): no card needed
# ---------------------------------------------------------------------------


def _part1_compiles(backend: str, tc, cls, tmp_path: Path) -> None:
    _, dqp, _, jabp, _ = _pair_programs(backend, cls, tmp_path)
    compile_checked(tc, dqp, tmp_path / "dq_bin")
    compile_checked(tc, jabp, tmp_path / "jab_bin")


def _part2_compiles(backend: str, tc, name: str, n: int, tmp_path: Path):
    basis, D = _case(name)
    classes = dq_classes(basis, D)
    compile_checked(tc, dq_program(classes, backend, tmp_path / "dq"),
                    tmp_path / "dq_bin")
    plan = _plan_from(basis, _host_dq_rows(basis, classes), n)
    drv = driver(plan, backend, tmp_path / "gen", chain=basis)
    compile_checked(tc, drv, tmp_path / "driver")


@pytest.mark.cuda
@pytest.mark.parametrize("cls", PAIR_PARAMS)
def test_cuda_pair_programs_compile(cls, nvcc, tmp_path) -> None:
    _part1_compiles("cuda", nvcc, cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", PAIR_PARAMS)
def test_hip_pair_programs_compile(cls, hipcc, tmp_path) -> None:
    _part1_compiles("hip", hipcc, cls, tmp_path)


@pytest.mark.cuda
@pytest.mark.parametrize(("name", "n"), CHAIN_PARAMS)
def test_cuda_chain_compiles(name, n, nvcc, tmp_path) -> None:
    _part2_compiles("cuda", nvcc, name, n, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize(("name", "n"), CHAIN_PARAMS)
def test_hip_chain_compiles(name, n, hipcc, tmp_path) -> None:
    _part2_compiles("hip", hipcc, name, n, tmp_path)


def _planted_warning_fails(backend: str, tc, tmp_path: Path) -> None:
    """A warning planted in a pair unit fails `compile_checked`, naming the
    unit: #92's ratchet is live on these programs."""
    basis, D = _case("h2_sto3g_j")
    prog = dq_program(dq_classes(basis, D), backend, tmp_path / "dq")
    unit = prog.units[0]
    unit.write_text(unit.read_text() + "\n__device__ int fj_planted() "
                    "{ int fj_unused; return 0; }\n")
    with pytest.raises(AssertionError, match=r"warning\(s\) in generated text"
                       ) as e:
        compile_checked(tc, prog, tmp_path / "dq_bin")
    assert unit.name in str(e.value)


@pytest.mark.cuda
def test_cuda_a_warning_in_a_pair_unit_fails(nvcc, tmp_path) -> None:
    _planted_warning_fails("cuda", nvcc, tmp_path)


@pytest.mark.hip
def test_hip_a_warning_in_a_pair_unit_fails(hipcc, tmp_path) -> None:
    _planted_warning_fails("hip", hipcc, tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the programs' text and data, nothing compiled or launched
# ---------------------------------------------------------------------------


def _ladder_is_emits_text(tmp_path, lmax: int) -> dict[str, str]:
    """The pair ladder through `lmax`, written, is `emit`'s text per unit."""
    emit, arch, ext = BACKEND["cuda"]
    written = emit_cli.generate_pairs("cuda", arch, tmp_path, lmax)
    want = {f"{k.name}.{ext}": emit(k, arch)
            for c in emit_cli.pair_classes(lmax)
            for k in (build_dq_kernel(c), build_jab_kernel(c))}
    assert {p.name: p.read_text() for p in written} == want
    return want


def test_part1_reads_the_pair_ladder_through_d(tmp_path):
    """The ladder through (dd), every class at L <= FAST_L_MAX: `emit`'s
    text per unit. The default ladder, through (ff), is the `slow` twin."""
    want = _ladder_is_emits_text(tmp_path, FAST_L_MAX // 2)
    assert len(want) == 12


@pytest.mark.slow  # the ladder reaches (ff), L > FAST_L_MAX (tiers.py)
def test_part1_reads_the_pair_ladders_units(tmp_path):
    """The ladder KERNEL_GEN_CMD writes by default is exactly the 20 units
    part 1 looks for in KERNEL_TIER_OUT, with `emit`'s text."""
    want = _ladder_is_emits_text(tmp_path, emit_cli.DEFAULT_PAIR_L)
    assert len(want) == 20 and PAIR_CLASSES[-1] == (3, 3)


def test_hf_ccpvtz_reaches_every_ladder_class():
    basis, _ = _case(PART1)
    assert set(_dq_classes(PART1)) >= set(PAIR_CLASSES)
    for cls in PAIR_CLASSES:
        assert _pairs_of(basis, cls)


@pytest.mark.parametrize("name", ["h2o_sto3g_j"])
def test_the_programs_data_is_the_interpreted_kernels(name):
    """The arrays the card receives, run through the INTERPRETED kernels,
    give the host's `D_q` (bitwise: kernel_dq's docstring) and the host's
    scatter (bitwise: kernel_jab's) -- the staging, checked without a card."""
    basis, D = _case(name)
    for dc in dq_classes(basis, D):
        grid, block = dq_launch_shape(dc.n_rec, 32)
        d = dc.data
        got = Interpreter(build_dq_kernel(dc.cls),
                          launch=Launch(grid, block, 32)).run(
            n_rec=dc.n_rec, n_pair=len(dc.members), eta=d[:, 0],
            XPA=d[:, 1:4].copy(), XPB=d[:, 4:7].copy(), s=d[:, 7], w=d[:, 8],
            pair=dc.pair.astype(np.int32), Dcd=dc.dcd,
            D=np.zeros((dc.n_rec, dq_shape(*dc.cls)[1])), Dmax=np.zeros(dc.n_rec))
        Dh, _ = _dq_host(basis, dc)
        assert np.array_equal(got["D"], Dh), dc.cls
        assert np.array_equal(got["Dmax"], np.max(np.abs(Dh), axis=1))
    plan = _plan_from(basis, _host_dq_rows(basis, dq_classes(basis, D)), 1)
    rng = np.random.default_rng(3)
    jps = {cl.lp: rng.uniform(-1, 1, (len(cl.bras),
                                      len(hermite_order(cl.lp))))
           for cl in plan}
    J = np.zeros((basis.n_ao, basis.n_ao))
    for la, lb, lp, data, ints in jab_records(plan, basis):
        grid, block = jab_launch_shape(len(data), 32)
        J = Interpreter(build_jab_kernel((la, lb)),
                        launch=Launch(grid, block, 1)).run(
            n_rec=len(data), n_ao=basis.n_ao, n_rows=jps[lp].shape[0],
            zeta_ab=data[:, 0], XPA=data[:, 1:4].copy(),
            XPB=data[:, 4:7].copy(), s=data[:, 7],
            off_a=ints[:, 0].astype(np.int32), off_b=ints[:, 1].astype(np.int32),
            row=ints[:, 2].astype(np.int32), Jp=jps[lp], J=J)["J"]
    host = [[0.0] * basis.n_ao for _ in range(basis.n_ao)]
    sh, off = basis.shells, basis.offsets
    blocks = {}
    for cl in plan:
        blocks.setdefault(cl.lp, cl.bra_blocks)
    for lp, blks in blocks.items():
        for a, b, start, stop in blks:
            jab_from_host(epair(sh[a], sh[b]), jps[lp][start:stop].tolist(),
                          (off[a], off[b]), host)
    assert np.array_equal(J, np.array(host))


def test_the_chain_driver_is_the_plans_shape_plus_jab(tmp_path):
    """Without `chain=` the driver has no JAB text; with it, one JAB launch
    per bra pair class, all after the last `[p|q]` launch, each reading its
    bra class's `J_p` buffer, each unit `emit`'s text."""
    basis, D = _case("h2o_sto3g_j")
    plan = _plan_from(basis, _host_dq_rows(basis, dq_classes(basis, D)), 1)
    plain = driver(plan, "hip", tmp_path / "plain")
    assert "dJmn" not in plain.source and not plain.jab
    drv = driver(plan, "hip", tmp_path / "chain", chain=basis)
    emit, arch, _ = BACKEND["hip"]
    pairs = {(basis.shells[a].l, basis.shells[b].l)
             for a, b in _unique_pairs(basis)}
    assert [(x.la, x.lb) for x in drv.jab] == sorted(pairs)
    slot = {lp: i for i, (lp, _, _) in enumerate(drv.bra)}
    last_pq = max(drv.source.index(f"{x.ns}::{x.kernel.name}<<<")
                  for x in drv.launches)
    for x in drv.jab:
        at = drv.source.index(f"{x.ns}::{x.kernel.name}<<<")
        assert at > last_pq
        assert f"dJ[{slot[x.lp]}], dJmn)" in drv.source[at:at + 400]
    assert drv.units[-len(drv.jab):] and all(
        u.read_text() == emit(x.kernel, arch)
        for u, x in zip(drv.units[-len(drv.jab):], drv.jab, strict=True))
    n_rec = sum(len(epair(basis.shells[a], basis.shells[b]).recs)
                for a, b in _unique_pairs(basis))
    assert sum(x.n_rec for x in drv.jab) == n_rec
    data = stage(plan, basis, D, drv, ket_data=_ket_data(
        basis, _host_dq_rows(basis, dq_classes(basis, D))))
    assert data.size == drv.n_input


def _e_abs_bounds_e(fast: bool) -> None:
    """Over the classes on one side of the cut, each drawing the geometry
    it drew when one test walked them all."""
    rng = np.random.default_rng(5)
    for la, lb in PAIR_CLASSES:
        zeta = float(rng.uniform(0.2, 3.0))
        P, A, B = (tuple(rng.uniform(-2, 2, 3)) for _ in range(3))
        if is_fast(la + lb) != fast:
            continue
        xpa = [P[d] - A[d] for d in range(3)]
        xpb = [P[d] - B[d] for d in range(3)]
        Ea = _e_abs(la, lb, zeta, xpa, xpb)
        for r, (a, b) in enumerate(cart_pairs(la, lb)):
            e = np.array(e_block(a, b, zeta, P, A, B))
            assert np.all(np.abs(e) <= Ea[r] * (1 + 1e-12)), (la, lb, r)


def test_e_abs_bounds_e():
    _e_abs_bounds_e(fast=True)


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_e_abs_bounds_e_above_l4():
    _e_abs_bounds_e(fast=False)
