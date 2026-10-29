"""The Coulomb-gradient witness (#639): libcint's `int2e_ip1_cart` on
CO/cc-pVTZ (`co_ccpvtz_j`'s molecule, basis and recorded density),
contracted per `gjab` bra class and whole into `dE_J/dR`
(`manifest.j_gradient`, written by `python -m oracle.generate --regenerate
--j-gradient`). It stands in for `gradient.host.coulomb_gradient` on the
card's CO input, which is pure Python and dominated #619's on-card cell.

- THE FILE (fast): its inputs are the parent fixture's (the loader
  refuses a mismatch), it holds every `gjab` class of the basis, the
  classes sum to the whole, and the whole is translationally invariant.
- THE HOST TIED TO IT (`slow`, pure Python at minutes a class): every
  class, so together the whole `host.coulomb_gradient`, each within
  `TOL = 1e-14` of the whole `max|G|` -- an order under the card's
  `DEV_TOL = 1e-13`, so a cell held to the file at `DEV_TOL` is held to
  the host at no looser a bound. A class `(la, lb)` is the host's
  `class_pairs` of both orders: the file keys by angular momentum alone.
- THE DRIFT CHECK (`oracle`, ~2 s): the file is what PySCF computes today.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.gradient import host
from felizjueves.reference.pq import one_centre

TOL = 1e-14  # of the whole max|G|
CLASSES = [(a, b) for a in range(4) for b in range(a, 4)]


@cache
def _witness():
    return manifest.j_gradient()


@cache
def _parent():
    fx = _witness().parent
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms)
    basis = ob.from_fixture(fx)
    D = np.asarray(fx.density)
    return basis, nuclei, D, host.coulomb_kets(basis, D)


def test_the_witness_is_on_its_parents_inputs() -> None:
    w = _witness()
    assert w.parent.name == "co_ccpvtz_j" and w.parent.nao == 70
    assert w.provenance["density_sha256"] == manifest.density_sha256(
        w.parent.density)


def test_it_holds_every_gjab_class_of_the_basis() -> None:
    basis = ob.from_fixture(_witness().parent)
    assert sorted({s.l for s in basis.shells}) == [0, 1, 2, 3]
    assert sorted(_witness().classes) == CLASSES


def test_the_classes_sum_to_the_whole_and_it_is_invariant() -> None:
    w = _witness()
    scale = float(np.abs(w.whole).max())
    assert np.abs(sum(w.classes.values()) - w.whole).max() <= 1e-14 * scale
    assert np.abs(w.whole.sum(axis=0)).max() <= 1e-13 * scale


def test_a_changed_density_is_refused(tmp_path) -> None:
    import json
    raw = json.loads(_witness().path.read_text())
    raw["provenance"]["density_sha256"] = "0" * 64
    (tmp_path / f"{manifest.J_GRADIENT}.json").write_text(json.dumps(raw))
    with pytest.raises(manifest.FixtureError, match="density_sha256"):
        manifest.j_gradient(where=tmp_path)


def test_a_k_class_is_refused(tmp_path) -> None:
    import json
    raw = json.loads(_witness().path.read_text())
    raw["classes"][0]["am"] = [0, 0, 0, 0]
    (tmp_path / f"{manifest.J_GRADIENT}.json").write_text(json.dumps(raw))
    with pytest.raises(manifest.FixtureError, match="bad or duplicate"):
        manifest.j_gradient(where=tmp_path)


@pytest.mark.oracle
def test_the_witness_is_what_pyscf_computes_today() -> None:
    pytest.importorskip("pyscf")
    import json

    from oracle import generate
    fresh = generate.j_gradient_build()
    tracked = json.loads(_witness().path.read_text())
    assert generate.comparable(fresh) == generate.comparable(tracked)


@pytest.mark.slow
@pytest.mark.parametrize("am", CLASSES, ids=lambda a: "".join(map(str, a)))
def test_the_host_is_the_witness(am) -> None:
    basis, nuclei, D, kets = _parent()
    la, lb = am
    pairs = host.class_pairs(basis, la, lb)
    if la != lb:
        pairs += host.class_pairs(basis, lb, la)
    w = _witness()
    g = np.array(host.coulomb_gradient(basis, D, nuclei,
                                       one_centre=one_centre, pairs=pairs,
                                       kets=kets))
    scale = float(np.abs(w.whole).max())
    err = float(np.abs(g - w.classes[am]).max()) / scale
    assert err <= TOL, f"{am}: {err:.3e} of max|G|"
    print(f"\nhost {am} against libcint: {err:.2e} of max|G|")
