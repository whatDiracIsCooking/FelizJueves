"""`oracle.basis`: a PySCF `Mole` as our shells, libcint's three conventions
pinned, and the overlap as the cheapest witness that all three are right.

Two halves, by the oracle rule (a comparison AGAINST a tracked fixture is
fast-tier; PySCF is needed only to regenerate one):

- fast tier, numpy only: our shells are built from each fixture's `converted`
  record and their overlap -- computed here by Obara-Saika, which shares no
  code with libcint or with `mmd/hermite.py` -- is compared with the `int1e_ovlp`
  libcint wrote into the same fixture (both cases reach (ff), so that
  comparison is `slow` by test/tiers.py's cut); the recorded constants are compared
  with `common_factor`; AO order is read off PySCF's own `ao_labels`.
- `oracle`-marked: the same statements against a live PySCF, plus the two
  things only PySCF can show -- what a spherical `Mole` does, and that
  `bas_ctr_coeff` is the file's coefficients and gives the wrong overlap.

THE TOLERANCE. 1e-14 absolute on S, whose entries here are at most 4 pi/5
(libcint's `xx` self-overlap): a few ulps of the largest entry, with the
contraction sums (<= 8 x 8 primitive pairs, cc-pVTZ's F s) each rounding at
~1e-16. It is the issue's number, not one tuned to green; every observed
discrepancy is reported by the assertion if it is ever exceeded. A wrong
convention is a per-l constant (a factor 0.28 for s, 2.5 for xx), so the
margin between "right" and "any one convention wrong" is 13 orders.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import is_fast

from felizjueves.ao.shell import Basis, Shell, cartesians, primitive_norm

TOL = 1e-14
CASES = ("prim_spdf", "hf_ccpvtz")


def _fixture(name):
    return manifest.load(manifest.FIXTURES / f"{name}.json")


# --- an overlap of our shells, independent of libcint and of mmd/hermite ----


def _ovlp_1d(la, lb, a, b, xa, xb):
    """S[i][j] for i <= la, j <= lb over arrays of primitive pairs (a x b), by
    the Obara-Saika recurrence for one Cartesian axis."""
    zeta = a + b
    p = (a * xa + b * xb) / zeta
    pa, pb = p - xa, p - xb
    h = 1.0 / (2.0 * zeta)
    s = [[None] * (lb + 1) for _ in range(la + 1)]
    s[0][0] = np.sqrt(np.pi / zeta) * np.exp(-(a * b / zeta) * (xa - xb) ** 2)
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == j == 0:
                continue
            if i > 0:  # raise i from (i-1, j)
                v = pa * s[i - 1][j]
                if i > 1:
                    v = v + (i - 1) * h * s[i - 2][j]
                if j > 0:
                    v = v + j * h * s[i - 1][j - 1]
            else:  # raise j from (0, j-1)
                v = pb * s[0][j - 1]
                if j > 1:
                    v = v + (j - 1) * h * s[0][j - 2]
            s[i][j] = v
    return s


def overlap(b: Basis) -> np.ndarray:
    """S_mu,nu of our contracted Cartesian functions, AO order = Basis order."""
    out = np.zeros((b.n_ao, b.n_ao))
    for sa, oa in zip(b.shells, b.offsets, strict=True):
        ea = np.array(sa.exponents)[:, None]
        ca = np.array(sa.coefficients)[:, None]
        for sb, ob_ in zip(b.shells, b.offsets, strict=True):
            eb = np.array(sb.exponents)[None, :]
            cb = np.array(sb.coefficients)[None, :]
            axes = [_ovlp_1d(sa.l, sb.l, ea, eb, sa.centre[x], sb.centre[x])
                    for x in range(3)]
            for m, (i, j, k) in enumerate(cartesians(sa.l)):
                for n, (u, v, w) in enumerate(cartesians(sb.l)):
                    prim = axes[0][i][u] * axes[1][j][v] * axes[2][k][w]
                    out[oa + m, ob_ + n] = float(np.sum(ca * cb * prim))
    return out


def _unit_axis_self_overlap(fast: bool) -> None:
    for l in range(5):  # noqa: E741
        if is_fast(2 * l) != fast:  # the pair (l, l)
            continue
        a = 0.37 + l
        s = overlap(Basis((Shell(l, (0.1, -0.2, 0.3), (a,),
                                 (primitive_norm(l, a),)),)))
        assert s[0, 0] == pytest.approx(1.0, abs=TOL)


def test_the_independent_overlap_normalises_by_primitive_norm():
    """The reference itself, checked: N^2 <x^l|x^l> = 1, so a shell whose
    coefficient is primitive_norm has unit axis self-overlap at every l
    (through d here; (ff) and (gg) in the `slow` twin)."""
    _unit_axis_self_overlap(fast=True)


@pytest.mark.slow  # (ff) and (gg), L > FAST_L_MAX (tiers.py)
def test_the_independent_overlap_normalises_by_primitive_norm_above_l4():
    _unit_axis_self_overlap(fast=False)


# --- fast tier: the tracked fixtures, no PySCF -----------------------------


@pytest.mark.slow  # both cases reach (ff), L > FAST_L_MAX (tiers.py)
@pytest.mark.parametrize("name", CASES)
def test_overlap_of_our_shells_matches_libcint(name):
    """THE witness: all three conventions right, or a per-l constant appears."""
    fx = _fixture(name)
    s = overlap(ob.from_fixture(fx))
    ref = fx.arrays["ovlp"]
    err = np.max(np.abs(s - ref))
    assert err <= TOL, f"{name}: max |S - S_libcint| = {err:.3e}"


@pytest.mark.parametrize("name", CASES)
def test_ao_count_and_offsets_match_pyscf(name):
    fx = _fixture(name)
    b = ob.from_fixture(fx)
    assert fx.provenance.cart is True
    assert b.n_ao == fx.nao
    # libcint shell i (with nctr contractions) became nctr consecutive Shells;
    # its ao_loc is the offset of the first of them.
    first, k = [], 0
    for sh in fx.shells:
        first.append(k)
        k += len(sh.coefficients[0])
    assert tuple(ob.ao_offsets(b)[i] for i in first) == fx.ao_loc[:-1]
    assert ob.ao_offsets(b)[-1] == fx.ao_loc[-1]


def _component(label: str) -> tuple[int, int, int]:
    """'0  F 3dxy  ' -> (1, 1, 0): the xyz letters after the l letter."""
    tail = label.split()[-1].lstrip("0123456789")[1:]
    return (tail.count("x"), tail.count("y"), tail.count("z"))


@pytest.mark.parametrize("name", CASES)
def test_ao_order_is_cartesians(name):
    """Per shell, PySCF's own AO labels run in `cartesians(l)` order."""
    fx = _fixture(name)
    b = ob.from_fixture(fx)
    seen = set()
    for s, off in zip(b.shells, b.offsets, strict=True):
        got = tuple(_component(fx.ao_labels[off + m]) for m in range(s.n_cart))
        assert got == cartesians(s.l), (name, off, s.l)
        seen.add(s.l)
    assert seen == {0, 1, 2, 3}  # every case covers l = 0..3


@pytest.mark.parametrize("name", CASES)
def test_recorded_common_factor_is_libcints(name):
    """Convention 3, per l, as PySCF's data yielded it when the fixture was
    written (the PySCF version is in its provenance)."""
    fx = _fixture(name)
    rec = fx.converted.common_factor
    assert len(rec) == 4
    for l, v in enumerate(rec):  # noqa: E741
        assert v == pytest.approx(ob.common_factor(l), rel=4e-16), (
            f"l={l}: pyscf {fx.provenance.pyscf} yields {v!r}")


def test_common_factor_values():
    four_pi = 4.0 * math.pi
    assert ob.common_factor(0) == 1.0 / math.sqrt(four_pi)
    assert ob.common_factor(1) == math.sqrt(3.0 / four_pi)
    assert [ob.common_factor(lq) for lq in range(2, 8)] == [1.0] * 6
    with pytest.raises(ValueError):
        ob.common_factor(-1)


def _gto_norm(l, a):  # noqa: E741
    """libcint's radial normaliser, by our N: gto_norm = N sqrt(4pi/(2l+1))."""
    return primitive_norm(l, a) * math.sqrt(4.0 * math.pi / (2 * l + 1))


@pytest.mark.parametrize("name", CASES)
def test_recorded_coefficients_carry_gto_norm(name):
    """Convention 2, from the fixture alone: per contraction, our coefficient
    over the FILE's is gto_norm(l, a) * common_factor(l) * (one constant, the
    contraction normaliser). `bas_ctr_coeff` would leave gto_norm out, and the
    ratio would then vary with the exponent."""
    fx = _fixture(name)
    conv = iter(fx.converted.shells)
    multi = 0
    for raw in fx.shells:
        for c in range(len(raw.coefficients[0])):
            ours = next(conv)
            assert ours.l == raw.l and ours.exponents == raw.exponents
            kappa = [o / (r[c] * _gto_norm(raw.l, a) * ob.common_factor(raw.l))
                     for o, r, a in zip(ours.coefficients, raw.coefficients,
                                        raw.exponents, strict=True)
                     if r[c] != 0.0]
            assert kappa and max(kappa) == pytest.approx(min(kappa), rel=1e-13)
            multi += len(kappa) > 1
    assert multi or name == "prim_spdf"


@pytest.mark.parametrize("name", CASES)
def test_libcint_normalisation_against_ours(name):
    """How libcint's Cartesian functions line up with `primitive_norm`'s:
    axis self-overlap 1 at l <= 1 and 4pi/(2l+1) from d up; component ratios
    (2i-1)!!(2j-1)!!(2k-1)!!/(2l-1)!! -- the same as ours -- at every l."""

    def dfo(n):
        return math.prod(range(1, 2 * n, 2))

    fx = _fixture(name)
    b = ob.from_fixture(fx)
    d = np.diag(fx.arrays["ovlp"])
    for s, off in zip(b.shells, b.offsets, strict=True):
        axis = 1.0 if s.l <= 1 else 4.0 * math.pi / (2 * s.l + 1)
        for m, (i, j, k) in enumerate(cartesians(s.l)):
            want = axis * dfo(i) * dfo(j) * dfo(k) / dfo(s.l)
            assert d[off + m] == pytest.approx(want, rel=1e-13), (s.l, (i, j, k))


@pytest.mark.slow  # prim_spdf's overlap reaches (ff) (tiers.py)
def test_double_normalisation_is_caught_by_the_overlap():
    """The hazard: `_env` already carries gto_norm, so primitive_norm on top
    normalises twice -- and the overlap sees it."""
    fx = _fixture("prim_spdf")
    good = ob.from_fixture(fx)
    twice = Basis(tuple(
        Shell(s.l, s.centre, s.exponents,
              tuple(c * primitive_norm(s.l, a)
                    for c, a in zip(s.coefficients, s.exponents, strict=True)))
        for s in good.shells))
    assert np.max(np.abs(overlap(twice) - fx.arrays["ovlp"])) > 1e-3


def test_from_fixture_needs_a_converted_record():
    with pytest.raises(ValueError, match="records no converted basis"):
        ob.from_fixture(_fixture("h2_sto3g"))


def test_converted_record_round_trips_the_schema(tmp_path):
    """Write -> load -> `from_fixture`: the same doubles, exactly, no PySCF."""
    path = manifest.FIXTURES / "hf_ccpvtz.json"
    copy = tmp_path / path.name
    copy.write_text(manifest.dumps(json.loads(path.read_text())))
    assert ob.from_fixture(manifest.load(copy)) == ob.from_fixture(
        manifest.load(path))


@pytest.mark.parametrize(
    ("mutate", "says"),
    [
        (lambda r: r["converted"]["shells"].pop(), "contractions"),
        (lambda r: r["converted"]["shells"][0]["coefficients"].pop(),
         "entries, expected"),
        (lambda r: r["converted"].pop("common_factor"), "converted must be"),
        (lambda r: r["converted"]["shells"][0].update(atom=9), "atom index"),
    ],
    ids=["shell-dropped", "coef-short", "no-common", "bad-atom"],
)
def test_corrupted_converted_record_fails_naming_the_file(tmp_path, mutate,
                                                          says):
    raw = json.loads((manifest.FIXTURES / "prim_spdf.json").read_text())
    mutate(raw)
    path = tmp_path / "prim_spdf.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(manifest.FixtureError) as e:
        manifest.load(path)
    assert str(path) in str(e.value) and says in str(e.value)


def test_basis_module_does_not_import_pyscf_at_module_scope():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(ob.__file__).read_text())
    top = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    names = {(n.module or "") if isinstance(n, ast.ImportFrom) else a.name
             for n in top for a in n.names}
    assert not any(m.startswith("pyscf") for m in names)


# --- oracle: a live PySCF --------------------------------------------------


def _mol(name, **over):
    pytest.importorskip("pyscf", reason="oracle group not installed: "
                        "uv sync --group oracle")
    from oracle import generate
    from pyscf import gto

    case = generate.CASES[name]
    basis = case.basis
    if "\n" in basis:
        symbols = {s.split()[0] for s in case.atom.split(";") if s.strip()}
        basis = {s: gto.basis.parse(case.basis, symb=s) for s in symbols}
    kw = {"atom": case.atom, "unit": case.unit, "basis": basis,
          "cart": case.cart, "verbose": 0} | over
    return gto.M(**kw)


@pytest.mark.oracle
@pytest.mark.parametrize("name", CASES)
def test_live_overlap_and_ao_count(name):
    mol = _mol(name)
    b = ob.from_mole(mol)
    assert mol.nao == b.n_ao
    err = np.max(np.abs(overlap(b) - mol.intor("int1e_ovlp")))
    assert err <= TOL, f"{name}: max |S - S_libcint| = {err:.3e}"
    # The converted record in the fixture is what PySCF converts to today.
    assert b == ob.from_fixture(_fixture(name))


@pytest.mark.oracle
@pytest.mark.parametrize("l", range(4))
def test_live_common_factor(l):  # noqa: E741
    """Convention 3, per l, against what this PySCF's data yields."""
    pytest.importorskip("pyscf")
    assert ob.measured_common_factor(l) == pytest.approx(
        ob.common_factor(l), rel=4e-16)


@pytest.mark.oracle
@pytest.mark.parametrize("name", CASES)
def test_live_cart_true_is_required(name):
    """Convention 1: a spherical Mole has 2l+1 AOs per shell, not
    (l+1)(l+2)/2 -- the AO counts disagree before any value does -- and the
    converter refuses it."""
    sph = _mol(name, cart=False)
    cart = _mol(name)
    assert sph.nao == sum((2 * cart.bas_angular(i) + 1) * cart.bas_nctr(i)
                          for i in range(cart.nbas))
    assert sph.nao < cart.nao == ob.from_mole(cart).n_ao
    with pytest.raises(ValueError, match="cart=True"):
        ob.from_mole(sph)


@pytest.mark.oracle
@pytest.mark.parametrize("name", CASES)
def test_live_libcint_ctr_coeff_not_bas_ctr_coeff(name):
    """Convention 2: `bas_ctr_coeff` is `_libcint_ctr_coeff` with gto_norm
    divided out, and a basis built from it gets the overlap wrong."""
    from pyscf import gto

    mol = _mol(name)
    for i in range(mol.nbas):
        norm = gto.gto_norm(mol.bas_angular(i), mol.bas_exp(i))
        np.testing.assert_allclose(mol.bas_ctr_coeff(i) * norm[:, None],
                                   mol._libcint_ctr_coeff(i), rtol=1e-15)
        assert not np.allclose(norm, 1.0)
    wrong = overlap(ob.from_mole_file_coefficients(mol))
    assert np.max(np.abs(wrong - mol.intor("int1e_ovlp"))) > 1e-3
