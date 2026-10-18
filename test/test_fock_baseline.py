"""The cross-tree ptxas baseline's join rule (#289): the predecessor's
`a >= b` keys and this repo's `la <= lb` ladder (hgp.vrr.CLASSES) are joined
by an explicit rule, a bijection on the eleven cells.

That rule is code, held here before any number is read. Everything that reads
data/measured/fock_baseline.json or kernels.json -- the rows' shape, the
boundary reading recomputed from them, each row's `source_sha256` against
today's emission, and the recompile under the predecessor's flags -- is a
check of a measurement artefact: measure/measure_fock_baseline.py
(`devtools/measure.sh check`, and `recheck --backend cuda` for the compile).
"""

from __future__ import annotations

from felizjueves.hgp import vrr

# The predecessor's eleven PTXAS_MEASURED keys with every shell l <= 2, in its
# own (a >= b, bra pair >= ket pair) canonicalisation and its table's order.
PREDECESSOR_CELLS = (
    (0, 0, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 1, 0, 0), (2, 0, 0, 0),
    (1, 1, 1, 0), (2, 0, 1, 0), (1, 1, 1, 1), (2, 0, 2, 0), (2, 1, 2, 1),
    (2, 2, 2, 2),
)


def canonical(cls: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """The same unordered quartet in this repo's ladder order: each pair
    ascending, then the two pairs ascending (hgp.vrr's `CLASSES` rule)."""
    bra, ket = tuple(sorted(cls[:2])), tuple(sorted(cls[2:]))
    return bra + ket if bra <= ket else ket + bra  # type: ignore[return-value]


# -- the mapping ------------------------------------------------------------------


def test_the_mapping_is_a_bijection_on_the_eleven_cells() -> None:
    image = [canonical(c) for c in PREDECESSOR_CELLS]
    assert len(set(PREDECESSOR_CELLS)) == len(set(image)) == 11
    assert set(image) <= set(vrr.CLASSES)
    # it is the same unordered quartet, by the dihedral symmetry of (ab|cd)
    for p, o in zip(PREDECESSOR_CELLS, image, strict=True):
        assert sorted(map(sorted, (p[:2], p[2:]))) == sorted(
            map(sorted, (o[:2], o[2:])))
        # the predecessor's own canonical form: a >= b, c >= d, bra >= ket
        assert p[0] >= p[1] and p[2] >= p[3] and p[:2] >= p[2:]
