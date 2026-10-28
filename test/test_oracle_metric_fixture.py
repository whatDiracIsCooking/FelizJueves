"""#411: the cc-pVTZ-jkfit metric fixture -- the WHOLE (P|Q) and aux overlap
of H2O/cc-pVDZ's auxiliary basis, reassembled from its row-block pieces.

Fast tier, numpy only: these read tracked files (`manifest.metric_set`);
the `oracle`-marked drift test in test_oracle_manifest.py is what checks
them against PySCF today.
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

SET = "h2o_ccpvtz_jkfit_metric"
NAO = 166
CAP = 512 * 1024  # .pre-commit-config.yaml's check-added-large-files


@pytest.fixture(scope="module")
def ms():
    return manifest.metric_set(SET)


def test_the_set_is_tracked_and_every_piece_is_under_the_cap(ms):
    # #606's aux-i set beside it (test_kernel_df_lowstack.py reads it)
    assert manifest.metric_sets() == [SET, "prim_spd_f_auxi_metric"]
    assert [p.metric.rows for p in ms.pieces] == [
        (0, 43), (43, 81), (81, 121), (121, 166)]
    for p in ms.pieces:
        assert p.path.stat().st_size < CAP, p.path.name


def test_the_whole_matrices_are_166_square_and_symmetric(ms):
    assert ms.nao == NAO
    for a in (ms.metric, ms.ovlp):
        assert a.shape == (NAO, NAO)
        assert not a.flags.writeable
        # libcint computes both triangles: symmetric to rounding, not bitwise
        assert np.max(np.abs(a - a.T)) <= 1e-14 * np.max(np.abs(a))
    # a Coulomb metric and an overlap are positive definite
    assert np.all(np.diag(ms.metric) > 0) and np.all(np.diag(ms.ovlp) > 0)
    assert np.linalg.eigvalsh(ms.metric).min() > 0


def test_the_aux_basis_reaches_g_on_o_and_f_on_h(ms):
    fx = ms.pieces[0]
    assert fx.provenance.basis == "cc-pvtz-jkfit" and fx.provenance.cart
    assert [a.symbol for a in fx.atoms] == ["O", "H", "H"]
    lmax = {}
    for s in ms.converted.shells:
        lmax[s.atom] = max(lmax.get(s.atom, 0), s.l)
    assert lmax == {0: 4, 1: 3, 2: 3}
    assert len(fx.shells) == len(ms.converted.shells) == 45


def test_from_fixture_loads_the_aux_basis(ms):
    basis = ob.from_fixture(ms)
    assert len(basis.shells) == 45 and basis.n_ao == NAO
    assert ob.ao_offsets(basis) == ms.pieces[0].ao_loc
    centres = [a.coord for a in ms.pieces[0].atoms]
    for s, c in zip(basis.shells, ms.converted.shells, strict=True):
        assert s.centre == centres[c.atom] and s.l == c.l


def test_the_parent_is_the_tracked_h2o_ccpvdz_molecule(ms):
    parent = manifest.load(manifest.FIXTURES / f"{ms.parent}.json")
    assert ms.parent == "h2o_ccpvdz_j"
    assert parent.provenance.mol == ms.pieces[0].provenance.mol
    assert parent.atoms == ms.pieces[0].atoms
    assert parent.provenance.cart == ms.pieces[0].provenance.cart


def _boys0(t: float) -> float:
    return 1.0 if t == 0.0 else 0.5 * math.sqrt(math.pi / t) * math.erf(
        math.sqrt(t))


def test_every_s_s_entry_is_the_closed_form(ms):
    """Two one-primitive s functions c_a e^{-a r^2}, c_b e^{-b r^2}:
    (P|Q) = c_a c_b 2 pi^{5/2} / (a b sqrt(a + b)) F_0(a b / (a + b) R^2),
    in our convention (the converted coefficients carry libcint's whole
    normalisation). An independent witness of what the pieces recorded."""
    basis = ob.from_fixture(ms)
    off = basis.offsets
    s_shells = [i for i, s in enumerate(basis.shells) if s.l == 0]
    assert all(len(basis.shells[i].exponents) == 1 for i in s_shells)
    worst = 0.0
    for i in s_shells:
        for j in s_shells:
            si, sj = basis.shells[i], basis.shells[j]
            a, b = si.exponents[0], sj.exponents[0]
            r2 = sum((x - y) ** 2 for x, y in zip(si.centre, sj.centre,
                                                  strict=True))
            want = (si.coefficients[0] * sj.coefficients[0]
                    * 2.0 * math.pi ** 2.5 / (a * b * math.sqrt(a + b))
                    * _boys0(a * b / (a + b) * r2))
            got = ms.metric[off[i], off[j]]
            worst = max(worst, abs(got - want) / abs(want))
    assert worst <= 1e-13, worst


# --- the reassembly's refusals ----------------------------------------------


def _copy_set(tmp_path):
    for p in manifest.FIXTURES.glob(f"{SET}_rows*.json"):
        shutil.copy(p, tmp_path)
    return sorted(tmp_path.glob("*.json"))


def test_a_copied_set_reassembles_identically(tmp_path, ms):
    _copy_set(tmp_path)
    again = manifest.metric_set(SET, tmp_path)
    assert np.array_equal(again.metric, ms.metric)
    assert np.array_equal(again.ovlp, ms.ovlp)


@pytest.mark.parametrize("drop", ["rows0_43", "rows43_81", "rows121_166"])
def test_a_missing_piece_is_refused(tmp_path, drop):
    _copy_set(tmp_path)
    (tmp_path / f"{SET}_{drop}.json").unlink()
    with pytest.raises(FixtureError, match="one is missing"):
        manifest.metric_set(SET, tmp_path)


def test_no_piece_at_all_is_refused(tmp_path):
    with pytest.raises(FixtureError, match="no metric piece"):
        manifest.metric_set(SET, tmp_path)


def _rewrite(path, mutate):
    raw = json.loads(path.read_text())
    mutate(raw)
    path.write_text(json.dumps(raw))


def test_an_overlapping_piece_is_refused(tmp_path):
    """A second piece over rows [0, 81) beside [0, 43) and [43, 81)."""
    _copy_set(tmp_path)
    a = json.loads((tmp_path / f"{SET}_rows0_43.json").read_text())
    b = json.loads((tmp_path / f"{SET}_rows43_81.json").read_text())
    for k in manifest.METRIC_ARRAYS:
        a["arrays"][k] = {"shape": [81, NAO],
                          "values": a["arrays"][k]["values"]
                          + b["arrays"][k]["values"]}
    a["name"] = f"{SET}_rows0_81"
    a["metric"]["rows"] = [0, 81]
    (tmp_path / f"{SET}_rows0_81.json").write_text(json.dumps(a))
    with pytest.raises(FixtureError, match="overlap"):
        manifest.metric_set(SET, tmp_path)


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda r: r["atoms"][1]["coord"].__setitem__(0, 1.44), "atoms"),
        (lambda r: r["provenance"].update(libcint="0.0.0"),
         "provenance.libcint"),
        (lambda r: r["provenance"]["mol"].update(atom="O 0 0 0"),
         "provenance.mol"),
        (lambda r: r["metric"].update(parent="h2o_631g_j"), "metric.parent"),
        (lambda r: r["metric"]["converted"]["shells"][0]["coefficients"]
         .__setitem__(0, 1.0), "metric.converted"),
    ],
    ids=["atoms", "libcint", "mol", "parent", "converted"],
)
def test_a_piece_with_other_inputs_is_refused(tmp_path, mutate, field):
    _copy_set(tmp_path)
    _rewrite(tmp_path / f"{SET}_rows81_121.json", mutate)
    with pytest.raises(FixtureError) as e:
        manifest.metric_set(SET, tmp_path)
    assert f"{field} differ" in str(e.value)
    assert "rows81_121" in str(e.value)


def test_a_piece_whose_date_alone_differs_is_one_set(tmp_path):
    """When and where a piece was written is not an input."""
    _copy_set(tmp_path)
    _rewrite(tmp_path / f"{SET}_rows81_121.json",
             lambda r: r["provenance"].update(date="1999-01-01", host="x"))
    manifest.metric_set(SET, tmp_path)


def test_a_piece_of_a_tampered_matrix_is_refused_as_asymmetric(tmp_path):
    _copy_set(tmp_path)

    def bump(r):
        r["arrays"][manifest.METRIC_ARRAY]["values"][5] *= 1.0 + 1e-9
    _rewrite(tmp_path / f"{SET}_rows43_81.json", bump)
    with pytest.raises(FixtureError, match="not symmetric"):
        manifest.metric_set(SET, tmp_path)
