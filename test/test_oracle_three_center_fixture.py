"""#442: the cc-pVTZ-jkfit three-center fixture -- the WHOLE `(mn|P)` of
H2O/STO-3G's 7 AOs against its 166 auxiliary AOs, reassembled from its
column pieces (one piece today: 7 x 7 x 166 values are ~190 KB, under the
512 KiB hook).

Fast tier, numpy only: these read tracked files
(`manifest.three_center_set`); the `oracle`-marked drift test in
test_oracle_manifest.py is what checks them against PySCF today.

THE m, n SYMMETRY IS TO 1e-14, NOT BITWISE -- a deliberate deviation from
#442's acceptance ("`(mn|P) == (nm|P)` bitwise"). The tracked tensor is
`df.incore.aux_e2(..., intor="int3c2e_cart", aosym="s1")`, as the issue
specifies, and libcint computes both triangles separately: 1,998 of the
8,134 entries differ from their transpose, by at most 4.4e-16 (6.2e-17 of
the largest element). The bound is the metric's (`metric_set`), 1e-14 of
max|(mn|P)|; the mismatch count and worst difference are pinned below so a
regeneration that changed them is seen.
"""

from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from oracle.manifest import FixtureError

SET = "h2o_sto3g_jkfit_3c"
PARENT = "h2o_sto3g_j"
NAO, NAUX = 7, 166
PIECE = manifest.FIXTURES / f"{SET}_cols0_{NAUX}.json"
ARRAY = manifest.THREE_CENTER_PREFIX + PARENT
CAP = 512 * 1024  # .pre-commit-config.yaml's check-added-large-files


@pytest.fixture(scope="module")
def ts():
    return manifest.three_center_set(SET)


def test_the_set_is_tracked_and_every_piece_is_under_the_cap(ts):
    assert manifest.three_center_sets() == [SET]
    assert [p.three_center.cols for p in ts.pieces] == [(0, NAUX)]
    for p in ts.pieces:
        assert p.path.stat().st_size < CAP, p.path.name


def test_the_pieces_reassemble_to_one_nao_nao_naux_tensor(ts):
    assert ts.nao == NAUX and ts.parent.name == PARENT
    assert ts.parent.nao == NAO
    assert ts.tensor.shape == (NAO, NAO, NAUX)
    assert not ts.tensor.flags.writeable
    assert np.array_equal(ts.tensor, ts.pieces[0].arrays[ARRAY])


def test_mn_symmetry_is_to_1e_14_not_bitwise(ts):
    """(mn|P) against (nm|P): see the module docstring for why not `==`."""
    a = ts.tensor
    d = np.abs(a - a.transpose(1, 0, 2))
    assert np.max(d) <= 1e-14 * np.max(np.abs(a))
    # what libcint's two triangles actually do, pinned
    assert np.count_nonzero(d) == 1998
    assert np.max(d) == 4.440892098500626e-16


def test_the_aux_basis_and_the_bra_side(ts):
    fx = ts.pieces[0]
    assert fx.provenance.basis == "cc-pvtz-jkfit" and fx.provenance.cart
    assert [a.symbol for a in fx.atoms] == ["O", "H", "H"]
    lmax = {}
    for s in ts.converted.shells:
        lmax[s.atom] = max(lmax.get(s.atom, 0), s.l)
    assert lmax == {0: 4, 1: 3, 2: 3}
    assert len(fx.shells) == len(ts.converted.shells) == 45
    # the bra side reaches p only: (ss|g), (sp|f), (pp|d) and below
    assert max(s.l for s in ts.parent.converted.shells) == 1
    assert ts.parent.atoms == fx.atoms
    assert ts.parent.provenance.mol == fx.provenance.mol
    basis = ob.from_fixture(ts)
    assert basis.n_ao == NAUX and ob.ao_offsets(basis) == fx.ao_loc


def _boys0(t: float) -> float:
    return 1.0 if t == 0.0 else 0.5 * math.sqrt(math.pi / t) * math.erf(
        math.sqrt(t))


def test_every_one_primitive_s_s_s_entry_is_the_closed_form(ts):
    """Gaussians a, b (bra, centres A, B) and c (aux, centre C), each one
    primitive in our convention: p = a + b, P the product centre,
    (ab|c) = c_a c_b c_c K_AB 2 pi^{5/2} / (p c sqrt(p + c))
    F_0(p c / (p + c) |P - C|^2). STO-3G's s shells are contractions, so the
    bra is summed over its primitives. An independent witness."""
    bra = ob.from_fixture(ts.parent)
    aux = ob.from_fixture(ts)
    worst, n = 0.0, 0
    for i, si in enumerate(bra.shells):
        for j, sj in enumerate(bra.shells):
            if si.l or sj.l:
                continue
            for k, sk in enumerate(aux.shells):
                if sk.l or len(sk.exponents) != 1:
                    continue
                c, cc = sk.exponents[0], sk.coefficients[0]
                want = 0.0
                for a, ca in zip(si.exponents, si.coefficients, strict=True):
                    for b, cb in zip(sj.exponents, sj.coefficients,
                                     strict=True):
                        p = a + b
                        P = [(a * x + b * y) / p for x, y in
                             zip(si.centre, sj.centre, strict=True)]
                        ab2 = sum((x - y) ** 2 for x, y in
                                  zip(si.centre, sj.centre, strict=True))
                        pc2 = sum((x - y) ** 2 for x, y in
                                  zip(P, sk.centre, strict=True))
                        want += (ca * cb * cc * math.exp(-a * b / p * ab2)
                                 * 2.0 * math.pi ** 2.5
                                 / (p * c * math.sqrt(p + c))
                                 * _boys0(p * c / (p + c) * pc2))
                got = ts.tensor[bra.offsets[i], bra.offsets[j],
                                aux.offsets[k]]
                worst = max(worst, abs(got - want) / abs(want))
                n += 1
    assert n > 0
    assert worst <= 1e-13, worst


# --- the reassembly's refusals ----------------------------------------------


def _copy(tmp_path, parent=True):
    shutil.copy(PIECE, tmp_path)
    if parent:
        shutil.copy(manifest.FIXTURES / f"{PARENT}.json", tmp_path)


def _slice(tmp_path, c0, c1):
    """A piece of columns [c0, c1), cut from the tracked one, in tmp_path."""
    raw = json.loads(PIECE.read_text())
    a = np.array(raw["arrays"][ARRAY]["values"]).reshape(
        raw["arrays"][ARRAY]["shape"])[:, :, c0:c1]
    raw["arrays"][ARRAY] = manifest.encode_array(a)
    raw["three_center"]["cols"] = [c0, c1]
    raw["name"] = f"{SET}_cols{c0}_{c1}"
    path = tmp_path / f"{raw['name']}.json"
    path.write_text(json.dumps(raw))
    return path


def test_sliced_pieces_reassemble_identically(tmp_path, ts):
    shutil.copy(manifest.FIXTURES / f"{PARENT}.json", tmp_path)
    for c0, c1 in ((0, 43), (43, 81), (81, NAUX)):
        _slice(tmp_path, c0, c1)
    again = manifest.three_center_set(SET, tmp_path)
    assert [p.three_center.cols for p in again.pieces] == [
        (0, 43), (43, 81), (81, NAUX)]
    assert np.array_equal(again.tensor, ts.tensor)


@pytest.mark.parametrize("cols", [((0, 43), (81, NAUX)), ((0, 43),),
                                  ((43, NAUX),)],
                         ids=["middle", "end", "start"])
def test_a_column_gap_is_refused(tmp_path, cols):
    shutil.copy(manifest.FIXTURES / f"{PARENT}.json", tmp_path)
    for c0, c1 in cols:
        _slice(tmp_path, c0, c1)
    with pytest.raises(FixtureError, match="one is missing"):
        manifest.three_center_set(SET, tmp_path)


def test_no_piece_at_all_is_refused(tmp_path):
    with pytest.raises(FixtureError, match="no three-center piece"):
        manifest.three_center_set(SET, tmp_path)


def test_an_overlapping_piece_is_refused(tmp_path):
    _copy(tmp_path)
    _slice(tmp_path, 0, 43)
    with pytest.raises(FixtureError, match="overlap"):
        manifest.three_center_set(SET, tmp_path)


def test_a_duplicate_piece_is_refused(tmp_path):
    """Two pieces of one range cannot share a directory (the stem names the
    range), so the duplicate is injected into the reassembly core."""
    _copy(tmp_path)
    fx = manifest.load(tmp_path / PIECE.name)
    with pytest.raises(FixtureError, match="duplicate piece"):
        manifest.assemble_three_center(SET, [fx, fx], tmp_path)


def test_a_piece_of_another_set_is_refused(tmp_path):
    _copy(tmp_path)
    fx = manifest.load(tmp_path / PIECE.name)
    with pytest.raises(FixtureError, match="is not a piece"):
        manifest.assemble_three_center("other", [fx], tmp_path)


def _rewrite(path, mutate):
    raw = json.loads(path.read_text())
    mutate(raw)
    path.write_text(json.dumps(raw))


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda r: r["atoms"][1]["coord"].__setitem__(0, 1.44), "atoms"),
        (lambda r: r["provenance"].update(libcint="0.0.0"),
         "provenance.libcint"),
        (lambda r: r["three_center"]["converted"]["shells"][0]
         ["coefficients"].__setitem__(0, 1.0), "three_center.converted"),
    ],
    ids=["atoms", "libcint", "converted"],
)
def test_a_piece_with_other_inputs_is_refused(tmp_path, mutate, field):
    shutil.copy(manifest.FIXTURES / f"{PARENT}.json", tmp_path)
    _slice(tmp_path, 0, 43)
    _rewrite(_slice(tmp_path, 43, NAUX), mutate)
    with pytest.raises(FixtureError) as e:
        manifest.three_center_set(SET, tmp_path)
    assert f"{field} differ" in str(e.value) and "cols43_166" in str(e.value)


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda r: r["atoms"][1]["coord"].__setitem__(0, 1.44), "atoms"),
        (lambda r: r["provenance"]["mol"].update(charge=1), "provenance.mol"),
    ],
    ids=["atoms", "mol"],
)
def test_a_parent_that_does_not_match_the_piece_is_refused(tmp_path, mutate,
                                                          field):
    _copy(tmp_path)
    _rewrite(tmp_path / f"{PARENT}.json", mutate)
    with pytest.raises(FixtureError) as e:
        manifest.three_center_set(SET, tmp_path)
    assert f"not its parent: {field} differ" in str(e.value)


def test_a_parent_of_another_nao_is_refused(tmp_path):
    """The piece's array is 7 AOs wide; a parent of other AOs on the same
    molecule (6-31G, 13 AOs) is not its bra side."""
    _copy(tmp_path, parent=False)
    raw = json.loads((manifest.FIXTURES / "h2o_631g_j.json").read_text())
    raw["name"] = PARENT
    (tmp_path / f"{PARENT}.json").write_text(json.dumps(raw))
    with pytest.raises(FixtureError, match="not its parent: nao differ"):
        manifest.three_center_set(SET, tmp_path)


def test_an_absent_parent_is_refused(tmp_path):
    _copy(tmp_path, parent=False)
    with pytest.raises(FixtureError, match="no fixture"):
        manifest.three_center_set(SET, tmp_path)


def test_a_tampered_tensor_is_refused_as_asymmetric(tmp_path):
    _copy(tmp_path)

    def bump(r):
        # the largest (m, n, P) with m != n, moved by 1e-9 of itself
        a = np.abs(np.array(r["arrays"][ARRAY]["values"]).reshape(
            NAO, NAO, NAUX))
        a[np.arange(NAO), np.arange(NAO), :] = 0.0
        r["arrays"][ARRAY]["values"][int(np.argmax(a))] *= 1.0 + 1e-9
    _rewrite(tmp_path / PIECE.name, bump)
    with pytest.raises(FixtureError, match="not symmetric"):
        manifest.three_center_set(SET, tmp_path)
