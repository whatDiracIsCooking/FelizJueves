"""The launch oracle (#142): the plan walked in Python, against three witnesses.

`launch.walk` / `launch.coulomb_launched` walk `launches()` -- per class, per
bra record, down the class's sorted ket list, Alg. 1's `break` at the first
`G_bra * key < tau` -- and return `ao.j.Screened`, the one screening record
both paths fill. Three witnesses, three ancestries:

  host      `ao.j.coulomb_matrix_screened`: the same screen and the same
            break over ONE global sorted ket list, where the oracle walks
            per-class lists. Equality is TO 0, and is a statement about
            WHICH terms were summed: each `J_p` is an `fsum` on both paths, and
            each per-ket term is the host's own arithmetic. If the two
            disagree the partition or the key is wrong, never the tolerance.
            The counts (`n_pairs`, `n_skipped`) and the skipped bounds
            (`max_skipped`, the `fsum` `dropped`) are equal too.
  ordered   `ao.j.coulomb_matrix_reference` (every ordered shell pair, weight
            1) at `tau = 0`: the off-diagonal-weight check. A different sum
            of the same terms, so not bit-equal; held to test_ao_j.py's
            stated SWEEP_TOL (1e-14 of max|J|), the bound that comparison
            already carries -- inherited, not chosen.
  libcint   the tracked `get_jk` fixtures for H2 and H2O/STO-3G, to
            test_ao_j.py's inherited `1e-12 scale_J`. No PySCF is imported.

THE STAGED DRIVER (#143's deferred comparison). `ao.staged.coulomb_matrix_staged`
over the SAME plan, at several `tau` (0 and three that skip a real share):

  exact     the ket-loop count. The kernel screens `Ib * (Iq * Dmax)`, and
            the staged driver checks `Iq * Dmax == key` bit for bit, so its
            break decisions are the oracle's: per class,
            `Interpreter.iterations` (lanes = 1) equals the oracle's
            `n_examined` on that class alone, times the class's kernels.
  bounded   `J`. NOT to 0, and the summation structure is why: the staged
            `J_p` comes from the generated kernel's own recurrence and the
            injected Boys, plain-added in kernel order, the oracle's from
            `reference.pq` through `fsum`. The oracle equals the host sweep
            TO 0 (above), and test_launch_staged.py holds the staged build
            to `1e-13 max|J_host|` -- test_ao_j_interpreted.py's
            kernel-vs-`reference.pq` bound -- so that same bound, and no
            looser one, is the staged-vs-oracle bound.

THE BREAK IS A BREAK. `n_examined` counts the kets whose screen test ran.
The oracle breaks once per (bra record, class) row, so it examines exactly
`n_pairs - n_skipped` plus one per row that has a ket under `tau` -- counted
here independently from the plan's keys. A `continue` would examine every
pair; the host's sorted break examines one extra per BRA RECORD, its
unsorted continue all `n_pairs`.

Fast tier: H2 and the four-H2 chain of test_ao_screening.py (s only), and
H2O/STO-3G at the one rung where its screen fires hard; the rest of H2O, and
#88's tau ladder re-run through this path (`slow`, like the existing ladder
test). The ladder's skip fractions are printed beside the host sweep's and
asserted equal to data/ao/screening.json's -- recorded there by the host, and
NOT rewritten here: the oracle's are equal, so a second column would repeat
the first.
"""

from __future__ import annotations

import dataclasses
import json
from functools import cache, partial

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from test_ao_screening import DATA as SCREENING_JSON
from test_ao_screening import TAUS as LADDER_TAUS
from test_ao_screening import _case as _ladder_case
from test_ao_screening import _h2_chain, _hilbert
from test_launch_staged import HOST_TOL, _tiny
from test_launch_staged import _taus as _staged_taus

from felizjueves import measured_store
from felizjueves.ao.j import (
    Screened,
    coulomb_matrix_and_scale,
    coulomb_matrix_reference,
    coulomb_matrix_screened,
    pair_bound_pq,
)
from felizjueves.ao.staged import coulomb_matrix_staged
from felizjueves.launch import coulomb_launched, launches, walk
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq

SWEEP_TOL = 1e-14  # test_ao_j.py's, unique vs ordered sweep, of max|J|
JK_TOL = 1e-12  # test_ao_j.py's, against get_jk, of scale_J
BOUND = partial(pair_bound_pq, pq=pq)
TAUS = (0.0, 1e-10, 1e-6, 1e-3, 1e-1)


@cache
def _case(name: str):
    if name == "chain":
        basis, D = _ladder_case("extended")
        return basis, np.asarray(D)
    if name == measured_store.CHAIN_INPUT:  # #147's device chain: two H2
        basis = _h2_chain(2)
        return basis, _hilbert(basis.n_ao)
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return ob.from_fixture(fx), np.array(fx.density)


@cache
def _pair(name: str, tau: float) -> tuple[Screened, Screened]:
    """(oracle, host) at `tau`, each built once."""
    basis, D = _case(name)
    return (coulomb_launched(basis, D, pq=pq, tau=tau),
            coulomb_matrix_screened(basis, D, pq=pq, tau=tau))


# fast: s-only cases at every tau, and H2O where its screen fires hard
FAST = ([("h2_sto3g_j", t) for t in TAUS] + [("chain", t) for t in TAUS]
        + [("h2o_sto3g_j", 1e-3)])
SLOW = [("h2o_sto3g_j", t) for t in TAUS if t != 1e-3]
PARAMS = ([pytest.param(n, t, id=f"{n}-{t:g}") for n, t in FAST]
          + [pytest.param(n, t, id=f"{n}-{t:g}", marks=pytest.mark.slow)
             for n, t in SLOW])


# --- witness 1: the host sweep, to 0 -----------------------------------------


@pytest.mark.parametrize(("name", "tau"), PARAMS)
def test_equals_the_host_screened_sweep_to_zero(name, tau):
    got, want = _pair(name, tau)
    assert got.J == want.J, f"{name} tau={tau}: J differs from the host's"
    assert got.scale == want.scale
    assert (got.n_pairs, got.n_skipped) == (want.n_pairs, want.n_skipped)
    assert got.max_skipped == want.max_skipped
    assert got.dropped == want.dropped
    if tau == 0.0:
        assert got.n_skipped == 0 and got.dropped == 0.0
        assert got.n_examined == got.n_pairs


@pytest.mark.parametrize(("name", "tau"), PARAMS)
def test_the_break_is_a_break(name, tau):
    """Count the kets examined: one past the accepted ones per row that
    breaks, counted from the plan's keys independently of the walk."""
    basis, D = _case(name)
    got, host = _pair(name, tau)
    plan = launches(basis, D, bound=BOUND, n_batches=1, tau=tau)
    rows_that_break = sum(
        1 for c in plan for b in c.bras
        if any(b.G * k.key < tau for k in c.kets))
    assert got.n_examined == got.n_pairs - got.n_skipped + rows_that_break
    assert got.n_skipped >= rows_that_break
    # the host's sorted break: one extra per bra record with a skip
    bras_that_break = sum(
        1 for b in {(c.lp, i): b for c in plan
                    for i, b in enumerate(c.bras)}.values()
        if any(b.G * k.key < tau for c in plan for k in c.kets))
    assert host.n_examined == host.n_pairs - host.n_skipped + bras_that_break
    if got.n_skipped > rows_that_break:  # a continue would examine them all
        assert got.n_examined < got.n_pairs


def test_the_host_continue_examines_everything():
    basis, D = _case("chain")
    tau = 1e-8
    srt = coulomb_matrix_screened(basis, D, pq=pq, tau=tau)
    cont = coulomb_matrix_screened(basis, D, pq=pq, tau=tau, order="unsorted")
    assert cont.n_examined == cont.n_pairs > srt.n_examined
    assert cont.J == srt.J
    assert (cont.n_skipped, cont.max_skipped, cont.dropped) == (
        srt.n_skipped, srt.max_skipped, srt.dropped)
    assert srt.n_skipped > 0


def test_the_screen_fires_across_classes():
    """H2O has several (lp, lq) classes and skips in more than one, so the
    per-class partition is exercised, not just one global list."""
    basis, D = _case("h2o_sto3g_j")
    plan = launches(basis, D, bound=BOUND, n_batches=1, tau=1e-3)
    firing = {(c.lp, c.lq) for c in plan
              if any(b.G * k.key < plan.tau for b in c.bras for k in c.kets)}
    assert len(plan) > 1 and len(firing) > 1, firing
    got, _ = _pair("h2o_sto3g_j", 1e-3)
    assert got.n_skipped > 0


# --- witness 2: the ordered reference sweep, tau = 0 -------------------------


@pytest.mark.parametrize("name", ["h2_sto3g_j", "h2o_sto3g_j"])
def test_equals_the_ordered_reference_at_tau_zero(name):
    basis, D = _case(name)
    J = np.array(_pair(name, 0.0)[0].J) if name == "h2_sto3g_j" else np.array(
        coulomb_launched(basis, D, pq=pq, tau=0.0).J)
    R = np.array(coulomb_matrix_reference(basis, D, pq=pq))
    worst = float(np.max(np.abs(J - R)) / np.max(np.abs(R)))
    print(f"\n{name}: oracle vs ordered reference {worst:.2e} of max|J|")
    assert worst <= SWEEP_TOL, (
        f"{name}: {worst:.2e} -- an off-diagonal weight error shows here")


# --- witness 3: libcint, through the tracked fixtures ------------------------


@pytest.mark.parametrize("name", ["h2_sto3g_j", "h2o_sto3g_j"])
def test_equals_the_libcint_fixture(name):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    basis, D = _case(name)
    J = np.array(coulomb_launched(basis, D, pq=pq, tau=0.0).J)
    _, scale = (np.array(x) for x in coulomb_matrix_and_scale(basis, D, pq=pq))
    ref = fx.arrays[manifest.J_ARRAY]
    assert np.all(scale > 0)
    worst = float(np.max(np.abs(J - ref) / scale))
    print(f"\n{name}: max |J - get_jk| / scale_J = {worst:.2e}")
    assert worst <= JK_TOL


# --- the staged interpreted driver, over the same plan (#143's deferral) -----


def _staged_cases():
    basis, D = _tiny()
    return [0.0, *_staged_taus(basis, D)]


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("rung", range(4))
def test_the_staged_driver_equals_the_oracle(rung, n):
    """Same plan, same tau: the break decisions exactly, `J` to the staged
    driver's inherited bound (module docstring, "THE STAGED DRIVER")."""
    basis, D = _tiny()
    tau = _staged_cases()[rung]
    plan = launches(basis, D, bound=BOUND, n_batches=n, tau=tau,
                    strict=False)
    staged = coulomb_matrix_staged(plan, basis, D, boys=boys)
    got = walk(plan, basis, D, pq=pq)
    host = coulomb_matrix_screened(basis, D, pq=pq, tau=tau)
    assert got.J == host.J and got.n_skipped == host.n_skipped
    for cl in plan:
        alone = walk(dataclasses.replace(plan, classes=(cl,)), basis, D, pq=pq)
        assert staged.iterations[(cl.lp, cl.lq)] == (
            alone.n_examined * len(cl.kernels)), (cl.lp, cl.lq, tau)
    if n == 1:
        assert sum(staged.iterations.values()) == got.n_examined
    if tau > 0.0:
        assert got.n_skipped > 0
    Jo, Js = np.array(got.J), np.array(staged.J)
    worst = float(np.max(np.abs(Js - Jo)) / np.max(np.abs(Jo)))
    print(f"\ntau={tau:.3e} n={n}: staged vs oracle {worst:.2e} of max|J|, "
          f"{got.n_skipped} of {got.n_pairs} skipped")
    assert worst <= HOST_TOL, (tau, n, worst)


# --- the plan's options do not change what is summed -------------------------


def test_the_batch_count_does_not_change_j():
    basis, D = _case("chain")
    one = coulomb_launched(basis, D, pq=pq, tau=1e-8)
    three = coulomb_launched(basis, D, pq=pq, tau=1e-8, n_batches=3,
                             strict=False)
    assert three == one


@pytest.mark.parametrize("tau", [0.0, 1e-8, 1e-6])
def test_density_sort_off_is_a_looser_sound_screen(tau):
    """One global Dmax: the order changes and the screen loosens -- never a
    pair the host keeps is skipped, and the error claim still holds."""
    basis, D = _case("chain")
    host = _pair("chain", tau)[1]
    flat = walk(launches(basis, D, bound=BOUND, n_batches=1, tau=tau,
                         density_sort=False), basis, D, pq=pq)
    assert flat.n_pairs == host.n_pairs
    assert flat.n_skipped <= host.n_skipped
    J0 = np.array(_pair("chain", 0.0)[1].J)
    err = np.abs(np.array(flat.J) - J0)
    assert np.all(err <= tau * np.array(flat.scale))
    if flat.n_skipped == host.n_skipped == 0:
        assert flat.J == host.J


# --- #147: the two sort keys, on the oracle (no card) -------------------------
#
# Both keys at each of #147's (input, tau) cells -- and at #88's four-H2 chain,
# the input the card's Boys cannot reach (measured_store, THE SORT KEY) --
# walked by the oracle: the SAME J to the screen's claimed error bound (a
# different key may skip different terms, so the claim is the bound, not bit
# equality), and the skip counts, recorded here. They are the counts the
# device rows carry as raw 'build:n_skipped' (test/test_measured.py holds the
# two equal). (n_pairs, product skipped, global skipped).
SORT_KEY_SKIPS = {
    ("h2o_ccpvdz_j", 1e-3): (351649, 123896, 0),
    ("h2o_ccpvdz_j", 1e-10): (351649, 15684, 0),
    (measured_store.CHAIN_INPUT, 1e-3): (8100, 2790, 0),
    (measured_store.CHAIN_INPUT, 1e-10): (8100, 560, 0),
    ("chain", 1e-3): (104976, 66060, 0),
    ("chain", 1e-10): (104976, 36072, 0),
}
SORT_KEY_PARAMS = [
    pytest.param(n, t, id=f"{n}-{t:g}",
                 marks=[pytest.mark.slow] if n == "h2o_ccpvdz_j" else [])
    for n, t in SORT_KEY_SKIPS]


@cache
def _j0(name: str) -> Screened:
    basis, D = _case(name)
    return coulomb_launched(basis, D, pq=pq, tau=0.0)


def test_the_sort_key_cells_are_recorded_here():
    assert {k for k in SORT_KEY_SKIPS if k[0] != "chain"} == {
        (f, t) for f, t, _ in measured_store.SORT_KEY_CELLS}


@pytest.mark.parametrize(("name", "tau"), SORT_KEY_PARAMS)
def test_both_sort_keys_hold_the_screens_bound(name, tau):
    """#147's fast tier: product and global key, same input, same tau, one
    batch -- each J within `tau * scale` of J(0), and the skip counts as
    recorded. The global key skips none of these pairs: `I_pair` is
    pair-factor-free, so the distance decay of a ket lives in its `Dmax`
    alone, and a global `Dmax` erases it."""
    basis, D = _case(name)
    J0 = np.array(_j0(name).J)
    got = []
    for ds in (True, False):
        s = coulomb_launched(basis, D, pq=pq, tau=tau, density_sort=ds)
        err = np.abs(np.array(s.J) - J0)
        assert np.all(err <= tau * np.array(s.scale)), (name, tau, ds)
        got.append(s)
    prod, glob = got
    assert prod.n_pairs == glob.n_pairs
    assert glob.n_skipped <= prod.n_skipped
    assert (prod.n_pairs, prod.n_skipped, glob.n_skipped) == SORT_KEY_SKIPS[
        (name, tau)]
    print(f"\n{name} tau={tau:g}: product skips {prod.n_skipped}/"
          f"{prod.n_pairs} ({prod.skip_fraction:.4f}), global "
          f"{glob.n_skipped} ({glob.skip_fraction:.4f})")


def test_the_walk_refuses_a_foreign_basis():
    basis, D = _case("h2_sto3g_j")
    other, D2 = _case("chain")
    plan = launches(other, D2, bound=BOUND, n_batches=1, tau=0.0)
    with pytest.raises((ValueError, KeyError, IndexError)):
        walk(plan, basis, D, pq=pq)


# --- #88's tau ladder, through this path ------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("name", ["compact", "extended"])
def test_the_tau_ladder_through_the_oracle(name):
    """Same rungs, same error bound, and the skip fractions beside the host
    sweep's recorded ones: equal, rung by rung."""
    recorded = json.loads(SCREENING_JSON.read_text())["cases"][name]
    assert [r["tau"] for r in recorded["rungs"]] == list(LADDER_TAUS)
    basis, D = _ladder_case(name)
    zero = coulomb_launched(basis, D, pq=pq, tau=0.0)
    J0, S = np.array(zero.J), np.array(zero.scale)
    assert zero.n_pairs == recorded["n_pairs"]
    lines = [f"{name}:   tau     oracle skip   host skip (recorded)"]
    for tau, rec in zip(LADDER_TAUS, recorded["rungs"], strict=True):
        got = coulomb_launched(basis, D, pq=pq, tau=tau)
        host = coulomb_matrix_screened(basis, D, pq=pq, tau=tau)
        assert got.J == host.J, (name, tau)
        assert got.n_skipped == host.n_skipped == rec["n_skipped"], (name, tau)
        frac = got.n_skipped / got.n_pairs
        assert frac == rec["skip_fraction"]
        err = np.abs(np.array(got.J) - J0)
        assert np.all(err <= tau * S), (name, tau)
        assert float(err.max()) == rec["max_err"], (name, tau)
        lines.append(f"  {tau:9.0e}   {frac:.4f}        "
                     f"{rec['skip_fraction']:.4f}")
    print("\n" + "\n".join(lines))
