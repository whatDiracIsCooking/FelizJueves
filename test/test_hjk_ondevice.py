"""The split HGP `J` and `K` on the cards (#408): every `L <= 4` class in both
class modes against the host and the interpreted launches, and the whole
H2/STO-3G and H2O/STO-3G `J` and `K` against `walk_hjk`'s interpreted build,
`eri/j_coulomb/j.py` / `eri/k_exchange/k.py` and libcint's `vj` / `vk` -- on the RTX
3080
(`sm_86`) and the RX 9060 XT (`gfx1200`).

THE LAUNCHES ARE WALK_HJK'S (test/hjk_driver.py): `launch_hjk`'s ordered
pair stream (#403), recorded from `walk_hjk` (#406), replayed on the card
through `kernel_hj` / `kernel_hk` (#404, #405) emitted by `emit/` unchanged
(#407), one program per kernel set. Every program is compiled with
devtools/config.sh's flags and FAILS on a warning located in generated text
(#92, `plan_driver.compile_checked`). No HGP J / K unit is written by
KERNEL_GEN_CMD (the HGP path is `--hgp-*`, #407), so the driver emits every
unit itself, the same text `--hgp-class` writes.

PART 1, PER CLASS. All 70 ordered classes of total `L = la + lb + lc + ld
<= 4` (`hgp.orbits.ordered_classes(4, lmax_shell=None)`, test_kernel_hj.py's
ladder, every shell `l` up to 4), for each screen (`J`, `K`) in each mode:

  ordered    one case per ordered class: that class's own kernel
             (`hj_o_` / `hk_o_`), every launch of the class;
  canonical  one case per KERNEL: the orbit representative's kernel with
             every launch of every member class it serves, so every
             runtime `perm` the stream routes to it runs.

THE PER-CLASS BASIS. A real basis with a d shell reaches `(dd|dd)` (L = 8),
and nothing above L = 4 may be built, compiled or run here; so each class
runs on a small synthetic basis holding only its own shell types (`l` of
the class; two shells of each `l <= 2` -- two primitives and one -- and one
shell, two primitives, of each `l >= 3`, on three centres so one- two- and
three-centre quartets all occur), with a random symmetric density, planned
by `hjk_launches` at `tau = 0` and filtered to the case's kernel
(`hjk_driver.record_launches`). Its Schwarz bound is the UNIT bound: at
`tau = 0` the screen `G_ab (G_cd dmax) < 0` never fires, so no decision
reads it, and the real bound would be host ERIs at `2 (la + lb)` (L = 8 for
a `(dd)` pair). Part 2 runs the real bound. Each case asserts that its
launches meet EVERY ordered quartet of its classes on that basis, once.

The per-class witness is the HOST SWEEP RESTRICTED TO THOSE QUARTETS:
`eri.fock.fock.coulomb_exchange_reference`, the every-ordered-quartet weight-1
sweep that `eri/j_coulomb/j.py`'s `coulomb_matrix_reference` and `eri/k_exchange/k.py`'s
`exchange_matrix_reference` are (test_hjk_build.py's restricted sweeps use
it the same way; those two cannot be restricted, and unrestricted they
would integrate the basis's every class), fed `eri.eri`'s MMD blocks -- the
other engine -- for the met quartets and zeros elsewhere. Against it: the
card's matrix, the launches interpreted (`hjk_driver.interpret`), and the
card's two runs against each other.

PART 2, PER MOLECULE. H2/STO-3G and H2O/STO-3G (s and p shells: every class
`L <= 4`) at `tau = 0`, both screens and both modes, every launch of the
plan (the real Schwarz bound, test_hjk_build's `plan`) in order into one
matrix, against walk_hjk's interpreted build (`test_hjk_build.walk`),
`eri/j_coulomb/j.py` / `eri/k_exchange/k.py` (`test_hjk_build.reference`) and libcint's
`vj` /
`vk` (`manifest.fock_pair`, which asserts the `_j` and `_k` fixtures share
density, shells and provenance). H2O/STO-3G at `tau = 1e-2`, canonical,
both screens: the screen fires (asserted), against the walk and the sweep
restricted to the quartets the walk computed.

THE TOLERANCES, stated before any run and taken from the files that state
them, not tuned. `TOL = 1e-12` is test/test_hgp_eri.py's per-ERI
cross-engine bound (`gamma_n` for about 9000 roundings: HGP within
`1e-12 (scale_mmd + scale_hgp)` of MMD), carried through the linear `J` /
`K` sums by the triangle inequality exactly as test_hjk_build.py and
test_kernel_hj.py carry it:

    |X_card - X_mmd| <= TOL S,   S = SUM |D| (scale_mmd + scale_hgp)

`S` the same sweep on `|D|` with `eri.eri`'s absolute-value twin plus
`eri.hgp_drive`'s, the latter in the orientation the kernel RAN (read off
each recorded launch's `perm`); the card is one more operation order of
the HGP recurrence (FMA contraction and the fp64 atomics' order only
re-associate), inside the same `gamma_n`. Against the interpreted launches
`2 TOL S` (the triangle inequality through the sweep, both within `TOL S`
of it), and the card's two runs, which differ only in the atomics' order,
`2 TOL S` too (whether they repeat bit for bit is printed). Against
libcint `LIBCINT_TOL = 1e-12` of the same `S` (test_hjk_build.py's,
test_engine.py's). An element with `S = 0` no computed quartet touches and
must come back exactly 0.0. The screened case's skip decisions are the same
IEEE products on the card and in the interpreter (`G_ab (G_cd dmax)`, no
addition to contract), so both compute the same quartets.

COMPILE-ONLY COMPANIONS (`cuda` / `hip`, no `gpu`) build every program the
device tests run, so a vendor spelling problem, or a warning in generated
text, shows without the card. The device tests are `gpu` plus `cuda` or
`hip`, behind the card lock and the VRAM probe: on a box without the card
they skip, and kernel-tier.sh exits 2. No timing (#331).
"""

from __future__ import annotations

import itertools
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from hjk_driver import (
    HRec,
    hjk_program,
    hjk_stage,
    interpret,
    kernel_name,
    quartets,
    record,
    record_launches,
    run_on_card,
)
from plan_driver import compile_checked
from test_hjk_build import LIBCINT_TOL, TOL, molecule, plan, reference, sweep, walk
from test_kernel_device import BACKEND, _toolchain

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri import hgp_drive as HD
from felizjueves.eri.eri import eri_block_and_scale
from felizjueves.eri.fock import fock
from felizjueves.eri.fock import kernel_f as KF
from felizjueves.eri.hjk import walk_hjk as W
from felizjueves.eri.hjk.launch_hjk import hjk_launches
from felizjueves.hgp import orbits
from felizjueves.ir import Kernel
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq

MODES, SCREENS = W.MODES, W.SCREENS
CLASSES = orbits.ordered_classes(4, lmax_shell=None)  # the 70, L <= 4
SEED = 408
SIDE = 1.5  # bohr: the synthetic centres' cube


def _groups(screen: str, mode: str) -> dict[str, tuple]:
    """Kernel name -> the ordered classes it serves, over the 70."""
    out: dict[str, list] = {}
    for c in CLASSES:
        out.setdefault(kernel_name(screen, mode, c), []).append(c)
    return {k: tuple(v) for k, v in out.items()}


GROUPS = {(s, m): _groups(s, m) for s in SCREENS for m in MODES}
CLASS_CASES = [
    pytest.param(s, m, name, id=f"{s}-{m}-{name}")
    for s in SCREENS for m in MODES for name in GROUPS[s, m]
]
MOLECULES = [
    pytest.param(name, 0.0, mode, screen, id=f"{name}-tau0-{mode}-{screen}")
    for name in ("h2_sto3g", "h2o_sto3g") for mode in MODES
    for screen in SCREENS
] + [
    pytest.param("h2o_sto3g", 1e-2, "canonical", screen,
                 id=f"h2o_sto3g-tau0.01-canonical-{screen}")
    for screen in SCREENS
]


def _worst(got, want, scale) -> float:
    err = np.abs(got - want)
    assert np.all(err[scale == 0.0] == 0.0), "an untouched element moved"
    return float(np.max(np.divide(err, scale, out=np.zeros_like(err),
                                  where=scale > 0)))


def _build(backend: str, tc, kernels, tmp_path: Path, name: str) -> Path:
    name = f"prog_{name}"  # never a unit's own file name
    prog = hjk_program(kernels, backend, tmp_path, name=name)
    return compile_checked(tc, prog, tmp_path / f"{name}_bin")


def _kernels(recs) -> list[Kernel]:
    out: dict[str, Kernel] = {}
    for r in recs:
        out.setdefault(r.name, r.kernel)
    return list(out.values())


# ---------------------------------------------------------------------------
# Part 1's basis, launches and references
# ---------------------------------------------------------------------------


def unit_bound(sa: Shell, sb: Shell) -> tuple[float, ...]:
    """1 per primitive pair: read by no decision at `tau = 0` (docstring)."""
    return (1.0,) * (len(sa.exponents) * len(sb.exponents))


@cache
def class_basis(ls: tuple[int, ...]) -> tuple[Basis, np.ndarray]:
    """The synthetic basis of the shell types `ls` (sorted, distinct) and a
    random symmetric density on it (module docstring, THE PER-CLASS BASIS)."""
    rng = np.random.default_rng(SEED + sum(1 << (3 * x) for x in ls))
    centres = [tuple(float(x) for x in rng.uniform(0.0, SIDE, 3))
               for _ in range(3)]
    shells = []
    for l in ls:  # noqa: E741
        for i, k in enumerate((2, 1) if l <= 2 else (2,)):
            e = tuple(sorted((float(x) for x in rng.uniform(0.3, 3.0, k)),
                             reverse=True))
            c = tuple(float(x) for x in rng.uniform(0.3, 1.0, k))
            shells.append(Shell(l, centres[(l + i) % 3], e, c))
    basis = Basis(tuple(shells))
    r = rng.uniform(-1.0, 1.0, (basis.n_ao, basis.n_ao))
    return basis, (r + r.T) / 2.0


def _ls(basis: Basis, q) -> tuple[int, ...]:
    return tuple(basis.shells[s].l for s in q)


@cache
def group_recs(screen: str, mode: str, name: str
               ) -> tuple[tuple[HRec, ...], tuple, tuple[int, ...]]:
    """The case's launches on its basis, its classes and the basis key; the
    launches meet every ordered quartet of those classes exactly once."""
    members = GROUPS[screen, mode][name]
    ls = tuple(sorted(set(members[0])))
    assert all(tuple(sorted(set(c))) == ls for c in members)
    basis, D = class_basis(ls)
    p = hjk_launches(basis, D, bound=unit_bound, tau=0.0, mode=mode,
                     screen=screen)
    recs = tuple(record_launches(p, basis, D, keep=lambda n: n == name))
    assert recs and {r.name for r in recs} == {name}
    met = [q for r in recs for q in quartets(r)]
    want = {q for q in itertools.product(range(len(basis.shells)), repeat=4)
            if _ls(basis, q) in members}
    assert len(met) == len(set(met)) and set(met) == want, name
    return recs, members, ls


_MMD: dict = {}


def _mmd(ls, q) -> tuple[np.ndarray, np.ndarray]:
    """`eri.eri`'s block and twin of ordered quartet `q`, flat, once each."""
    if (ls, q) not in _MMD:
        sh = class_basis(ls)[0].shells
        blk, scale = eri_block_and_scale((sh[q[0]], sh[q[1]]),
                                         (sh[q[2]], sh[q[3]]), pq=pq)
        _MMD[ls, q] = (np.array(blk).ravel(), np.array(scale).ravel())
    return _MMD[ls, q]


def hgp_twins(basis: Basis, recs) -> dict:
    """Per ordered quartet, `eri.hgp_drive`'s twin in the orientation its
    launch RAN (built-order shells `x`, transposed back by `GROUP[perm]`),
    flat in the ordered quartet's `[a][b][c][d]` layout."""
    sh = basis.shells
    out = {}
    for r in recs:
        shells = orbits.GROUP[r.perm]
        axes = tuple(shells.index(i) for i in range(4))
        for q in quartets(r):
            x = tuple(q[shells[j]] for j in range(4))
            _, s = HD.eri_block_and_scale((sh[x[0]], sh[x[1]]),
                                          (sh[x[2]], sh[x[3]]), boys=boys)
            dims = [sh[i].n_cart for i in x]
            out[q] = np.array(s).reshape(dims).transpose(axes).ravel()
    return out


def restricted_sweep(basis: Basis, D, screen: str, blocks) -> np.ndarray:
    """`eri.fock.fock`'s every-ordered-quartet sweep (`J` or `K`) with the flat
    blocks `blocks[q]` for the quartets it holds and zeros elsewhere."""
    idx = {id(s): i for i, s in enumerate(basis.shells)}

    def eri(bra, ket):
        q = (idx[id(bra[0])], idx[id(bra[1])], idx[id(ket[0])],
             idx[id(ket[1])])
        n = (bra[0].n_cart * bra[1].n_cart, ket[0].n_cart * ket[1].n_cart)
        if q not in blocks:
            return np.zeros(n).tolist()
        return np.asarray(blocks[q]).reshape(n).tolist()

    J, K = fock.coulomb_exchange_reference(basis, np.asarray(D).tolist(),
                                           eri=eri)
    return np.array(J if screen == "j" else K)


@cache
def group_reference(screen: str, mode: str, name: str):
    """(interpreted X, restricted MMD sweep, S) of the case's launches."""
    recs, _, ls = group_recs(screen, mode, name)
    basis, D = class_basis(ls)
    met = {q for r in recs for q in quartets(r)}
    twin = hgp_twins(basis, recs)
    ref = restricted_sweep(basis, D, screen, {q: _mmd(ls, q)[0] for q in met})
    S = restricted_sweep(basis, np.abs(D), screen,
                         {q: _mmd(ls, q)[1] + twin[q] for q in met})
    assert np.max(np.abs(ref)) > 0  # the comparison is not of zeros
    return interpret(recs, basis.n_ao), ref, S


# ---------------------------------------------------------------------------
# Part 1: per class, against the restricted host sweep and the interpreter
# ---------------------------------------------------------------------------


def _part1(backend: str, screen: str, mode: str, name: str, request,
           run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    recs, members, ls = group_recs(screen, mode, name)
    kernels = [recs[0].kernel]
    binary = _build(backend, tc, kernels, tmp_path, name)
    basis, _ = class_basis(ls)
    Xs, _ = run_on_card(binary, hjk_stage(recs, kernels), basis.n_ao,
                        tmp_path, run_device_binary, backend=backend,
                        tag=name)
    X_int, ref, S = group_reference(screen, mode, name)
    w = {"host": _worst(Xs[0], ref, S), "interp": _worst(Xs[0], X_int, S),
         "runs": _worst(Xs[1], Xs[0], S)}
    print(f"\n{backend} {BACKEND[backend][1].name} {screen} {mode} {name} "
          f"({len(members)} classes {members}, {len(recs)} launches): worst "
          f"/ S host {w['host']:.2e} (tol {TOL:g}), interp "
          f"{w['interp']:.2e} (tol {2 * TOL:g}), runs {w['runs']:.2e}, "
          f"bit-identical {bool(np.array_equal(Xs[0], Xs[1]))}")
    assert w["host"] <= TOL, (backend, name, w)
    assert w["interp"] <= 2 * TOL, (backend, name, w)
    assert w["runs"] <= 2 * TOL, (backend, name, w)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("screen", "mode", "name"), CLASS_CASES)
def test_cuda_hjk_each_class_matches_the_host(screen, mode, name, request,
                                              run_device_binary,
                                              tmp_path) -> None:
    """Part 1 on the RTX 3080."""
    _part1("cuda", screen, mode, name, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("screen", "mode", "name"), CLASS_CASES)
def test_hip_hjk_each_class_matches_the_host(screen, mode, name, request,
                                             run_device_binary,
                                             tmp_path) -> None:
    """Part 1 on the RX 9060 XT."""
    _part1("hip", screen, mode, name, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Part 2: the whole molecule, against walk_hjk, eri/j_coulomb/j.py /
# eri/k_exchange/k.py, libcint
# ---------------------------------------------------------------------------


@cache
def molecule_recs(name: str, tau: float, mode: str, screen: str):
    basis, D, _, _ = molecule(name)
    recs, w = record(plan(name, tau, mode, screen), basis, D)
    return tuple(recs), w


def _part2(backend: str, name: str, tau: float, mode: str, screen: str,
           request, run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    recs, wrec = molecule_recs(name, tau, mode, screen)
    kernels = _kernels(recs)
    tag = f"hjk_{name}_{mode}_{screen}"
    binary = _build(backend, tc, kernels, tmp_path, tag)
    basis, _, vj, vk = molecule(name)
    Xs, _ = run_on_card(binary, hjk_stage(recs, kernels), basis.n_ao,
                        tmp_path, run_device_binary, backend=backend, tag=tag)
    w_int = walk(name, tau, mode, screen)  # test_hjk_build's interpreted walk
    assert w_int.kernel_screened == wrec.kernel_screened
    S = sweep(name, screen, block="scale", mode=mode)
    w = {"interp": _worst(Xs[0], np.array(w_int.X), S),
         "runs": _worst(Xs[1], Xs[0], S)}
    if tau == 0.0:
        assert not w_int.kernel_screened
        w["host"] = _worst(Xs[0], reference(name, screen), S)
        w["libcint"] = _worst(Xs[0], vj if screen == "j" else vk, S)
    else:
        assert w_int.kernel_screened and w_int.computed  # the screen fired
        kept = sweep(name, screen, block="eri", keep=w_int.computed)
        w["kept"] = _worst(Xs[0], kept, S)
    print(f"\n{backend} {BACKEND[backend][1].name} {name} tau={tau} {mode} "
          f"{screen}: {len(recs)} launches of {len(kernels)} kernels, "
          f"{len(w_int.kernel_screened)}/{len(w_int.visited)} skipped; worst "
          "/ S " + ", ".join(f"{k} {v:.2e}" for k, v in w.items())
          + f"; runs bit-identical {bool(np.array_equal(Xs[0], Xs[1]))}")
    assert w["interp"] <= 2 * TOL, (backend, name, w)
    assert w["runs"] <= 2 * TOL, (backend, name, w)
    if tau == 0.0:
        assert w["host"] <= TOL, (backend, name, w)
        assert w["libcint"] <= LIBCINT_TOL, (backend, name, w)
    else:
        assert w["kept"] <= TOL, (backend, name, w)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "mode", "screen"), MOLECULES)
def test_cuda_hjk_whole_molecule_matches_walk_host_and_libcint(
        name, tau, mode, screen, request, run_device_binary,
        tmp_path) -> None:
    """Part 2 on the RTX 3080."""
    _part2("cuda", name, tau, mode, screen, request, run_device_binary,
           tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "mode", "screen"), MOLECULES)
def test_hip_hjk_whole_molecule_matches_walk_host_and_libcint(
        name, tau, mode, screen, request, run_device_binary,
        tmp_path) -> None:
    """Part 2 on the RX 9060 XT."""
    _part2("hip", name, tau, mode, screen, request, run_device_binary,
           tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions (`cuda` / `hip`, no `gpu`): no card needed
# ---------------------------------------------------------------------------

COMPILE_CELLS = [
    *(pytest.param(("class", *p.values), id=f"class-{p.id}")
      for p in CLASS_CASES),
    *(pytest.param(("molecule", *p.values), id=f"molecule-{p.id}")
      for p in MOLECULES if p.values[1] == 0.0),  # tau changes no kernel
]


def _compiles(backend: str, tc, cell, tmp_path: Path) -> None:
    if cell[0] == "class":
        _, screen, mode, name = cell
        kernels = [group_recs(screen, mode, name)[0][0].kernel]
    else:
        _, name, tau, mode, screen = cell
        kernels = _kernels(molecule_recs(name, tau, mode, screen)[0])
        name = f"hjk_{name}_{mode}_{screen}"
    _build(backend, tc, kernels, tmp_path, name)


@pytest.mark.cuda
@pytest.mark.parametrize("cell", COMPILE_CELLS)
def test_cuda_hjk_programs_compile(cell, nvcc, tmp_path) -> None:
    _compiles("cuda", nvcc, cell, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cell", COMPILE_CELLS)
def test_hip_hjk_programs_compile(cell, hipcc, tmp_path) -> None:
    _compiles("hip", hipcc, cell, tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the cases, the restated launch loop, the program, the staging
# ---------------------------------------------------------------------------


def test_the_cases_are_every_l4_class_in_both_modes() -> None:
    assert len(CLASSES) == 70 and max(sum(c) for c in CLASSES) == 4
    for s in SCREENS:
        for m in MODES:
            served = [c for v in GROUPS[s, m].values() for c in v]
            assert sorted(served) == sorted(CLASSES)
        assert len(GROUPS[s, "ordered"]) == 70
        assert all(n.startswith(f"h{s}_o_") for n in GROUPS[s, "ordered"])
        assert len(GROUPS[s, "canonical"]) == len(
            {orbits.representative(c) for c in CLASSES})


def _same(a: HRec, b: HRec) -> bool:
    if (a.kernel, a.grid, a.threads) != (b.kernel, b.grid, b.threads):
        return False
    if set(a.args) != set(b.args):
        return False
    return all(np.array_equal(np.asarray(a.args[k], dtype=float),
                              np.asarray(b.args[k], dtype=float))
               for k in a.args)


@pytest.mark.parametrize(("name", "mode", "screen"), [
    pytest.param(n, m, s, id=f"{n}-{m}-{s}") for n in ("h2_sto3g", "h2o_sto3g")
    for m in MODES for s in SCREENS])
def test_the_restated_launch_loop_is_the_walks(name, mode, screen) -> None:
    """`record_launches` admitting every kernel is `record` (walk_hjk's own
    loop), launch for launch and argument for argument."""
    basis, D, _, _ = molecule(name)
    p = plan(name, 0.0, mode, screen)
    walked, _ = record(p, basis, D)
    again = record_launches(p, basis, D, keep=lambda _: True)
    assert len(walked) == len(again) == len(p.launches)
    assert all(_same(a, b) for a, b in zip(walked, again, strict=True))


def test_the_program_reads_the_kernels_parameters(tmp_path) -> None:
    """Every case passes one argument per kernel parameter, `perm` in the
    `diag` slot only for a canonical kernel."""
    for mode in MODES:
        recs, _ = molecule_recs("h2o_sto3g", 0.0, mode, "k")
        kernels = _kernels(recs)
        prog = hjk_program(kernels, "hip", tmp_path / mode)
        for k in kernels:
            at = prog.source.index(">>>(", prog.source.index(f"::{k.name}<<<"))
            call = prog.source[at:prog.source.index(");", at)]
            assert call.count(",") == len(k.params) - 1, k.name
            assert (" dg," in call) == (mode == "canonical"), k.name
            assert "dF" in call
    with pytest.raises(ValueError, match="not one matrix"):
        hjk_program([_kernels(molecule_recs("h2_sto3g", 0.0, "ordered",
                                            s)[0])[0] for s in SCREENS],
                    "cuda", tmp_path / "mixed")


def _unstage(data: np.ndarray, kernels, template) -> list[HRec]:
    """The input sliced as the program slices it, back into launches."""
    nl, n_sh, n_ao, tau = int(data[0]), int(data[1]), int(data[2]), data[3]
    heads = data[4:4 + 9 * nl].reshape(nl, 9)
    at = 4 + 9 * nl
    Dsh = data[at:at + n_sh * n_sh]
    at += n_sh * n_sh
    D = data[at:at + n_ao * n_ao]
    at += n_ao * n_ao
    out = []
    for h, t in zip(heads, template, strict=True):
        ki, nb, nk, kab, kcd, perm, th, nd, ni = (int(x) for x in h)
        k = kernels[ki]
        has = {p.name for p in k.params}
        rec = len(KF.REC)
        sizes = {"bra_prim": nb * kab * rec, "AB": 3 * nb, "G_ab": nb,
                 "ket_prim": nk * kcd * rec, "CD": 3 * nk, "G_cd": nk,
                 "bra_sh": 2 * nb, "bra_ao": 2 * nb, "ket_sh": 2 * nk,
                 "ket_ao": 2 * nk}
        args = {"n_bra": nb, "n_ket": nk, "k_ab": kab, "k_cd": kcd,
                "n_sh": n_sh, "n_ao": n_ao, "perm": perm, "tau": tau,
                "Dsh": Dsh, "D": D, t.out: None}
        start = at
        for n in ("bra_prim", "AB", "G_ab", "ket_prim", "CD", "G_cd",
                  "bra_sh", "bra_ao", "ket_sh", "ket_ao"):
            if n in has:
                args[n] = data[at:at + sizes[n]]
                at += sizes[n]
        assert at - start == nd + ni
        out.append(HRec(k, nb, th, {p.name: args[p.name] for p in k.params}))
    assert at == data.size
    return out


@pytest.mark.parametrize("mode", MODES)
def test_the_staged_data_through_the_interpreter_is_the_recording(mode
                                                                  ) -> None:
    """The input the card receives, sliced back as the program slices it and
    interpreted launch by launch, is the recorded launches interpreted, bit
    for bit: the staging, checked without a card."""
    recs, _ = molecule_recs("h2_sto3g", 0.0, mode, "k")
    kernels = _kernels(recs)
    basis, _, _, _ = molecule("h2_sto3g")
    again = _unstage(hjk_stage(recs, kernels), kernels, recs)
    assert np.array_equal(interpret(again, basis.n_ao),
                          interpret(recs, basis.n_ao))
