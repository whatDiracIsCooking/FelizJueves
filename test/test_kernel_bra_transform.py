"""The bra transform `sbra3_{la}{lb}` (#747), interpreted: `speri3_`'s packed
output (Cartesian bra, spherical aux) onto spherical bra slots, the other
leg of measure/measure_spherical_bra.py's clock against `ssperi3_`.

THE ORACLE is numpy applying the same linear map to the same input: the
interpreted `speri3_` block of each pair contracted by `aux_harmonics` on
both bra indices. That isolates the transform from the integrals, which
test_kernel_eri3_spherical.py already holds to the host. THE BOUND, per
element, `2 (nA + nB) eps` of `(|c^{la}| |c^{lb}|) . |X|`: the kernel's two
sums (`nA` terms into `Tmu`, `nB` into the element) and numpy's two, each a
sum of terms whose |.| sum is that scale.

SCOPE (minimal, by decision): one `la != lb` class and one diagonal one,
(pd|p) and (dd|p), on three AO shells; the output starts NaN, so an
element no pair stores is seen. Its card run is the clock's witness
(the clock holds `speri3_` + `sbra3_` to `ssperi3_` before timing).
"""

from __future__ import annotations

import numpy as np
import pytest

from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.eri.density_fit.eri3 import (
    bra_pairs,
    eri3_classes,
    packed_cells,
    packed_columns,
    spherical_offsets,
)
from felizjueves.eri.density_fit.eri3_drive import (
    launch_bra_transform,
    launch_packed,
)
from felizjueves.eri.density_fit.kernel_eri3 import (
    aux_harmonics,
    build_bra_transform_kernel,
)
from felizjueves.reference.boys import boys

EPS = 2.0 ** -52
AO = Basis((Shell(2, (0.0, 0.0, 0.0), (0.8,), (1.0,)),
            Shell(2, (0.3, -0.7, 1.1), (1.1, 0.35), (0.6, 0.5)),
            Shell(1, (-0.4, 0.2, 0.5), (0.9,), (1.0,))))
AUX = Basis((Shell(1, (-1.2, 0.4, 0.25), (0.7, 0.3), (0.8, 0.4)),))


@pytest.mark.parametrize("am", [(1, 2, 1), (2, 2, 1)],
                         ids=["pd|p", "dd|p"])
def test_the_transform_is_the_numpy_contraction_of_its_input(am):
    (cls,) = [c for c in eri3_classes(AO, AUX) if (c.la, c.lb, c.lP) == am]
    la, lb, _ = am
    offs, n_aux = spherical_offsets(AUX)
    cart, n_cart = packed_columns([cls])
    sph, n_sph = packed_columns([cls], spherical_bra=True)
    src = launch_packed(cls, packed_cells(cls, cart, n_aux, offs),
                        np.full((n_cart, n_aux), np.nan), boys=boys,
                        threads=5, spherical_aux=True)
    pairs = bra_pairs(cls, cart, sph)
    got = launch_bra_transform(la, lb, src, pairs, n_sph, threads=7)
    nA, nB = len(cartesians(la)), len(cartesians(lb))
    ca, cb = np.array(aux_harmonics(la)), np.array(aux_harmonics(lb))
    for s0, d0 in zip(pairs["src_col"], pairs["dst_col"], strict=True):
        X = src[s0:s0 + nA * nB].reshape(nA, nB, n_aux)
        want = np.einsum("ia,jb,abq->ijq", ca, cb, X)
        scale = np.einsum("ia,jb,abq->ijq", np.abs(ca), np.abs(cb),
                          np.abs(X))
        g = got[d0:d0 + want.shape[0] * want.shape[1]].reshape(want.shape)
        assert np.all(np.abs(g - want) <= 2 * (nA + nB) * EPS * scale)
    assert not np.isnan(got).any()  # every slot of every pair stored
    assert len(pairs["src_col"]) * (2 * la + 1) * (2 * lb + 1) == n_sph


def test_the_kernel_is_named_and_holds_one_block():
    k = build_bra_transform_kernel(3, 3)
    assert k.name == "sbra3_33"
    assert [p.name for p in k.params] == [
        "n_pair", "n_aux", "n_src_col", "n_dst_col", "src_col", "dst_col",
        "src", "eri3"]
