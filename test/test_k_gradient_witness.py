"""The exchange-gradient witness (#618): libcint's `int2e_ip1_cart` on
CO/cc-pVTZ (`co_ccpvtz_j`'s molecule, basis and recorded density),
contracted per K class and whole into `dE_K/dR` (`manifest.k_gradient`,
written by `python -m oracle.generate --regenerate --k-gradient`).

- THE FILE (fast): its inputs are the parent fixture's (the loader
  refuses a mismatch), it holds all 256 K classes of the basis, the
  classes sum to the whole, and the whole is translationally invariant.
- THE HOST TIED TO IT (`slow`, pure Python at minutes a class):
  `gradient.host.exchange_gradient` over four s, p, d classes, `(dd|dd)`,
  `(pd|pd)`, `(ds|ds)` and `(dp|sp)`, each within `TOL = 1e-12` of the
  whole `max|G|` (measured: <= 2.7e-16). The host is the interpreted
  kernels' witness (test_kernel_gk.py), so the fixture and the kernels
  are tied through it; on the card the kernels are held to the fixture
  directly (test_gk_device.py).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.gradient import host
from felizjueves.reference.pq import one_centre

TOL = 1e-12  # of the whole max|G|
HOST_CLASSES = [(2, 2, 2, 2), (1, 2, 1, 2), (2, 0, 2, 0), (2, 1, 0, 1)]


@cache
def _witness():
    return manifest.k_gradient()


@cache
def _parent():
    fx = _witness().parent
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms)
    return ob.from_fixture(fx), nuclei, np.asarray(fx.density)


def test_the_witness_is_on_its_parents_inputs() -> None:
    w = _witness()
    assert w.parent.name == "co_ccpvtz_j" and w.parent.nao == 70
    assert w.provenance["density_sha256"] == manifest.density_sha256(
        w.parent.density)


def test_it_holds_every_k_class_of_the_basis() -> None:
    basis, _, _ = _parent()
    ls = sorted({s.l for s in basis.shells})
    assert ls == [0, 1, 2, 3]
    want = {(a, c, b, d) for a in ls for c in ls for b in ls for d in ls}
    assert set(_witness().classes) == want


def test_the_classes_sum_to_the_whole_and_it_is_invariant() -> None:
    w = _witness()
    scale = float(np.abs(w.whole).max())
    assert np.abs(sum(w.classes.values()) - w.whole).max() <= 1e-14 * scale
    assert np.abs(w.whole.sum(axis=0)).max() <= 1e-13 * scale


def test_a_changed_density_is_refused(tmp_path) -> None:
    import json
    raw = json.loads(_witness().path.read_text())
    raw["provenance"]["density_sha256"] = "0" * 64
    (tmp_path / f"{manifest.K_GRADIENT}.json").write_text(json.dumps(raw))
    with pytest.raises(manifest.FixtureError, match="density_sha256"):
        manifest.k_gradient(where=tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("am", HOST_CLASSES,
                         ids=lambda a: "".join(map(str, a)))
def test_the_host_is_the_witness(am) -> None:
    basis, nuclei, D = _parent()
    w = _witness()
    g = np.array(host.exchange_gradient(
        basis, D, nuclei, one_centre=one_centre,
        quartets=host.class_quartets(basis, am)))
    scale = float(np.abs(w.whole).max())
    err = float(np.abs(g - w.classes[am]).max()) / scale
    assert err <= TOL, f"{am}: {err:.3e} of max|G|"
    print(f"\nhost {am} against libcint: {err:.2e} of max|G|")
