"""`ao.staged`: `J_mn` from the launch plan, class-grouped and staged (#143).

`coulomb_matrix_staged(launches(...), basis, D, boys=...)` builds each bra
class's buffers and `J_p` once, each ket class's buffers once, runs every
batch kernel of every `(lp, lq)` class in `interp.Interpreter` into its bra
class's one `J_p`, and hands the result to eq. 20 and the scatter. Interpreter
level: no compiler, no card, no PySCF (the device version is #144).

THE WITNESSES, at the tolerances `test_ao_j_interpreted.py` already uses and
does not loosen here:

  host sweep  `ao.j.coulomb_matrix_screened` at the SAME `tau` (the same
              screen on the same keys, as the sorted `break`),
              and `ao.j.coulomb_matrix_reference` (every ordered pair, weight
              1) at `tau = 0`: `|J - J_host| <= 1e-13 max|J_host|`;
  libcint     the tracked `get_jk` fixtures at `tau = 0`:
              `|J - J_pyscf| <= 1e-12 scale_J`.

The LAUNCH ORACLE (#142, the plan walked in Python) is not on main yet; the
staged-vs-oracle comparison is added when it lands. Until then the host
sweep is the same-screen witness: its skip decisions are the plan's,
bit for bit, when the plan's bound is `ao.j.pair_bound_pq` (#138 pins every
ket key equal to the host's `_Ket.bound`).

THE CASES, by cost:

  fast  a TINY s+p basis built here (two centres, one s and one p shell each,
        one or two primitives): every class `lp, lq` in 0..2, nine
        `ClassLaunch`es over three bra and three ket classes -- the smallest
        case where one `J_p` has several ket-class writers;
        H2/STO-3G against its libcint fixture (one class, `(ss|ss)`).
  slow  H2O/STO-3G against its libcint fixture and the host sweep, at one
        and three batches, with its buffer sizes reported.
"""

from __future__ import annotations

import dataclasses
from functools import cache, partial

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao import j as aoj
from felizjueves.ao.drive import KernelJP
from felizjueves.ao.shell import Basis, Shell
from felizjueves.ao.staged import BufferSize, coulomb_matrix_staged
from felizjueves.launch import LaunchPlan, launches
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq

BOUND = partial(aoj.pair_bound_pq, pq=pq)  # the host sweep's bound, exactly
HOST_TOL = 1e-13  # vs the host sweep, relative to its largest element
JK_TOL = 1e-12  # vs libcint, times scale_J: test_ao_j.py's, inherited


@cache
def _fixture(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return fx, ob.from_fixture(fx), np.array(fx.density)


@cache
def _tiny():
    """Two centres, an s (two primitives) and a p (one primitive) on each."""
    shells = []
    for c in ((0.0, 0.0, 0.0), (0.3, -0.2, 1.5)):
        shells.append(Shell(0, c, (3.0, 0.5), (0.4, 0.7)))
        shells.append(Shell(1, c, (0.8,), (1.0,)))
    basis = Basis(tuple(shells))
    n = basis.n_ao
    D = np.array([[1.0 / (1 + abs(a - b)) + (0.1 if a == b else 0.0)
                   for b in range(n)] for a in range(n)])
    return basis, D


def _plan(basis, D, *, tau=0.0, n=1, **kw) -> LaunchPlan:
    return launches(basis, D, bound=BOUND, n_batches=n, tau=tau,
                    strict=False, **kw)


def _staged(basis, D, *, tau=0.0, n=1, **kw):
    return coulomb_matrix_staged(_plan(basis, D, tau=tau, n=n), basis, D,
                                 boys=boys, **kw)


def _rel(J, H) -> float:
    J, H = np.asarray(J), np.asarray(H)
    return float(np.max(np.abs(J - H)) / np.max(np.abs(H)))


def _taus(basis, D) -> list[float]:
    """Three `tau` that screen something in the tiny case: quantiles of the
    plan's `G_bra * key_ket` products, so each rung skips a real share."""
    pl = _plan(basis, D)
    prods = sorted(b.key * k.key for cl in pl for b in cl.bras
                   for k in cl.kets)
    return [prods[len(prods) * f // 10] for f in (2, 5, 8)]


# --- the answer -------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 3])
def test_tiny_equals_host_sweep_and_reference_at_tau_zero(n):
    basis, D = _tiny()
    s = _staged(basis, D, n=n)
    H = aoj.coulomb_matrix_screened(basis, D, pq=pq, tau=0.0).J
    R = aoj.coulomb_matrix_reference(basis, D, pq=pq)
    assert _rel(s.J, H) <= HOST_TOL, _rel(s.J, H)
    assert _rel(s.J, R) <= HOST_TOL, _rel(s.J, R)
    # nine classes, each at the asked count where the plan admits it
    assert len(s.ran) == 9
    assert {(lp, lq) for lp, lq, _ in s.ran} == {
        (a, b) for a in range(3) for b in range(3)}


def test_tiny_equals_the_kernel_jp_seam():
    """The old path is still there and still agrees: two paths, not one
    replaced (#143's hazard). `KernelJP` is the unscreened seam, so tau = 0."""
    basis, D = _tiny()
    old = aoj.coulomb_matrix(basis, D, pq=pq, jp=KernelJP(boys=boys))
    assert _rel(_staged(basis, D).J, old) <= HOST_TOL


def test_lanes_do_not_change_the_answer():
    basis, D = _tiny()
    tau = _taus(basis, D)[1]
    one = _staged(basis, D, tau=tau)
    four = _staged(basis, D, tau=tau, lanes=4)
    assert _rel(four.J, one.J) <= HOST_TOL


def test_density_sort_off_still_equals_the_host_at_tau_zero():
    """`density_sort=False` ranks by one global Dmax; the staged driver
    ships that Dmax (its keys checked bit for bit), and at tau = 0 the
    answer cannot move."""
    basis, D = _tiny()
    pl = _plan(basis, D, density_sort=False)
    s = coulomb_matrix_staged(pl, basis, D, boys=boys)
    H = aoj.coulomb_matrix_screened(basis, D, pq=pq, tau=0.0).J
    assert _rel(s.J, H) <= HOST_TOL


@pytest.mark.parametrize("n", [1, 3])
def test_h2_matches_libcint_and_the_host(n):
    fx, basis, D = _fixture("h2_sto3g_j")
    s = _staged(basis, D, n=n)
    _, scale = aoj.coulomb_matrix_and_scale(basis, D, pq=pq)
    ref = fx.arrays[manifest.J_ARRAY]
    jk = float(np.max(np.abs(np.array(s.J) - ref) / np.array(scale)))
    H = aoj.coulomb_matrix_screened(basis, D, pq=pq, tau=0.0).J
    print(f"\nh2_sto3g_j n_batches={n}: ran {s.ran}; "
          f"max|J - get_jk|/scale = {jk:.2e}; vs host {_rel(s.J, H):.2e}")
    assert jk <= JK_TOL
    assert _rel(s.J, H) <= HOST_TOL


@pytest.mark.slow
@pytest.mark.parametrize("n", [1, 3])
def test_h2o_matches_libcint_and_the_host(n):
    fx, basis, D = _fixture("h2o_sto3g_j")
    s = _staged(basis, D, n=n)
    J, scale = aoj.coulomb_matrix_and_scale(basis, D, pq=pq)
    ref = fx.arrays[manifest.J_ARRAY]
    jk = float(np.max(np.abs(np.array(s.J) - ref) / np.array(scale)))
    print(f"\nh2o_sto3g_j n_batches={n}: ran {s.ran}; "
          f"max|J - get_jk|/scale = {jk:.2e}; vs host {_rel(s.J, J):.2e}")
    _report(f"h2o_sto3g_j n_batches={n}", s.sizes)
    assert jk <= JK_TOL
    assert _rel(s.J, J) <= HOST_TOL
    assert dict(s.builds) == {"bra": 3, "J_p": 3, "ket": 3}


# --- the buffers ------------------------------------------------------------


def _report(label: str, sizes) -> None:
    print(f"\n{label}: staged buffers")
    for b in sizes:
        print(f"  {b.side:>4} l={b.l}  records={b.n_records:>4}  "
              f"n_fn={b.n_fn:>2}  doubles={b.n_doubles:>6}  bytes={b.bytes}")
    print(f"  total bytes={sum(b.bytes for b in sizes)}")


@pytest.mark.parametrize("n", [1, 3])
def test_buffer_counts_drop_to_one_per_class(n):
    """One ket-buffer build per ket class and one J_p (and bra) buffer per
    bra class -- counted as built, not assumed -- against the adapter's shape,
    which builds a ket buffer for every (bra shell pair, ket class) call."""
    basis, D = _tiny()
    pl = _plan(basis, D, n=n)
    s = coulomb_matrix_staged(pl, basis, D, boys=boys)
    lps = {cl.lp for cl in pl}
    lqs = {cl.lq for cl in pl}
    assert dict(s.builds) == {"bra": len(lps), "J_p": len(lps),
                              "ket": len(lqs)} == {"bra": 3, "J_p": 3,
                                                   "ket": 3}
    assert len(pl) == 9  # nine launches share those six buffer sets

    calls = []
    k = KernelJP(boys=boys)

    def counting(bra_am, ket_am, bras, kets):
        calls.append((bra_am, ket_am))
        return k(bra_am, ket_am, bras, kets)

    aoj.coulomb_matrix(basis, D, pq=pq, jp=counting)
    # 10 bra shell pairs x 4 ket shell classes (ss, ps, sp, pp -- shells
    # s0 p0 s1 p1, so a >= b reaches both (p0, s0) and (s1, p0)): 40 uploads
    assert len(calls) == 40
    assert s.builds["ket"] < len(calls)


def test_buffer_sizes_are_reported_and_add_up():
    """The per-class sizes are a recorded fact: each is the formula of its
    contents, and the test prints them (`-s`)."""
    basis, D = _tiny()
    pl = _plan(basis, D)
    s = coulomb_matrix_staged(pl, basis, D, boys=boys)
    _report("tiny n_batches=1", s.sizes)
    n_fn = {0: 1, 1: 4, 2: 10}  # Hermite functions |t| <= L
    by_l = {}
    for cl in pl:
        by_l[("bra", cl.lp)] = len(cl.bras)
        by_l[("ket", cl.lq)] = len(cl.kets)
    want = set()
    for (side, lc), n in by_l.items():
        if side == "bra":
            want.add(BufferSize("bra", lc, n, 0, 5 * n))
            want.add(BufferSize("J_p", lc, n, n_fn[lc], n * n_fn[lc]))
        else:
            want.add(BufferSize("ket", lc, n, n_fn[lc], n * (6 + n_fn[lc])))
    assert set(s.sizes) == want and len(s.sizes) == len(want)
    for b in s.sizes:
        assert b.bytes == 8 * b.n_doubles


# --- the screen and the break -----------------------------------------------


def _expected_iterations(pl: LaunchPlan) -> tuple[dict, int]:
    """Per class, the ket-loop entries Alg. 1's break makes with lanes = 1
    (every ket over tau, plus the one that breaks), times its kernels; and
    the (bra, ket) pairs skipped over the whole plan."""
    its, skipped = {}, 0
    for cl in pl:
        n = 0
        for b in cl.bras:
            over = next((i for i, k in enumerate(cl.kets)
                         if b.key * k.key < pl.tau), len(cl.kets))
            n += over + (over < len(cl.kets))
            skipped += len(cl.kets) - over
        its[(cl.lp, cl.lq)] = n * len(cl.kernels)
    return its, skipped


@pytest.mark.parametrize("n", [1, 3])
def test_the_break_fires_and_still_matches_the_host(n):
    """With tau > 0 the screen is REAL (Ip, Iq, Dmax are the records', not
    the seam's ones): `Interpreter.iterations` falls below the full
    bra x ket count in exactly the shape a `break` makes, the skip count is
    the host sweep's, and J equals the host sweep at the same tau."""
    basis, D = _tiny()
    for tau in _taus(basis, D):
        pl = _plan(basis, D, tau=tau, n=n)
        s = coulomb_matrix_staged(pl, basis, D, boys=boys)
        host = aoj.coulomb_matrix_screened(basis, D, pq=pq, tau=tau)
        its, skipped = _expected_iterations(pl)
        full = {(cl.lp, cl.lq): len(cl.bras) * len(cl.kets) * len(cl.kernels)
                for cl in pl}
        assert s.iterations == its
        assert sum(s.iterations.values()) < sum(full.values())
        assert skipped == host.n_skipped > 0
        assert _rel(s.J, host.J) <= HOST_TOL, (tau, _rel(s.J, host.J))


def test_h2_break_fires():
    _, basis, D = _fixture("h2_sto3g_j")
    pl = _plan(basis, D, tau=0.1)
    s = coulomb_matrix_staged(pl, basis, D, boys=boys)
    host = aoj.coulomb_matrix_screened(basis, D, pq=pq, tau=0.1)
    its, skipped = _expected_iterations(pl)
    assert s.iterations == its == {(0, 0): 228}  # of 27 x 27 = 729
    assert skipped == host.n_skipped == 528
    assert _rel(s.J, host.J) <= HOST_TOL


def test_the_seam_placeholders_are_not_shipped():
    """Carrying `Ip = Iq = Dmax = 1, tau = 0` by accident would leave the
    screen untested: the staged buffers carry the plan's G and Dmax."""
    basis, D = _tiny()
    tau = _taus(basis, D)[1]
    pl = _plan(basis, D, tau=tau)
    s = coulomb_matrix_staged(pl, basis, D, boys=boys)
    assert pl.tau == tau > 0
    assert any(b.G != 1.0 for cl in pl for b in cl.bras)
    assert any(k.key != k.G for cl in pl for k in cl.kets)
    full = sum(len(cl.bras) * len(cl.kets) for cl in pl)
    assert sum(s.iterations.values()) < full


# --- the precondition -------------------------------------------------------


def _reversed_kets(pl: LaunchPlan) -> LaunchPlan:
    return dataclasses.replace(pl, classes=tuple(
        dataclasses.replace(cl, kets=cl.kets[::-1]) for cl in pl))


def test_unsorted_kets_are_refused():
    basis, D = _tiny()
    pl = _reversed_kets(_plan(basis, D, tau=_taus(basis, D)[1]))
    with pytest.raises(ValueError, match="not sorted"):
        coulomb_matrix_staged(pl, basis, D, boys=boys)


def test_injected_unsorted_kets_change_the_answer():
    """The plan establishes the sort and the kernel's break depends on it:
    the same kets ascending, with the refusal switched off, break early and
    give a DIFFERENT J -- a wrong answer, not a slower one."""
    basis, D = _tiny()
    tau = _taus(basis, D)[1]
    pl = _plan(basis, D, tau=tau)
    good = coulomb_matrix_staged(pl, basis, D, boys=boys)
    bad = coulomb_matrix_staged(_reversed_kets(pl), basis, D, boys=boys,
                                check_sorted=False)
    host = aoj.coulomb_matrix_screened(basis, D, pq=pq, tau=tau).J
    assert _rel(good.J, host) <= HOST_TOL
    assert _rel(bad.J, host) > 1e6 * HOST_TOL
    assert sum(bad.iterations.values()) < sum(good.iterations.values())


def test_a_ket_list_that_differs_between_launches_is_refused():
    """One upload per ket class is sound only if every launch of that class
    carries the same list."""
    basis, D = _tiny()
    pl = _plan(basis, D)
    cls = list(pl.classes)
    i = next(i for i, cl in enumerate(cls) if cl.lp == 1)
    cls[i] = dataclasses.replace(cls[i], kets=cls[i].kets[:-1])
    with pytest.raises(ValueError, match="different ket lists"):
        coulomb_matrix_staged(dataclasses.replace(pl, classes=tuple(cls)),
                              basis, D, boys=boys)
