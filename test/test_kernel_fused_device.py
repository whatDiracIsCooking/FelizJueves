"""The `fused` lever's [p|q] kernels (#344) on the card, against reference/.

The same harness and the same pin as test_kernel_device.py's lever arms
(`_run_batched`: the cell's whole launch sequence, every `CASES` input, the
oracle `reference.pq.one_centre` contracted with `D_q`), held to the SAME
`J_REL = 1e-12 * scale` -- not a looser one. The fused kernel forms the same
products with the same signs and accumulates them in a different order
(j_levers/fused.py), a reassociation of at most `n` terms per (bra, p) per
lane, bounded by `(n - 1) u * scale`: far inside J_REL, as test/
test_kernel_fused.py shows interpreted. FMA contraction on the device rounds
once where the interpreter rounds twice, which only tightens that bound.

Cells: one batch at L = 0..2 and the split cell L = 2 x 3 batches (atomic
write-back), within the milestone's per-PR cap; one batch at L = 3, 4, 7, 10
and the 3-batch cells at L = 4 and 7 are `slow` (the milestone's final tier,
#364, runs them).
"""

from __future__ import annotations

import pytest
from test_kernel_device import _run_batched

from felizjueves.j_levers.fused import FUSED
from felizjueves.kernel import pq_spec

CELLS = [
    *(pytest.param(pq_spec(L, (FUSED,)), id=f"L{L}-b1-fused")
      for L in range(3)),
    pytest.param(pq_spec(2, (FUSED,), 3), id="L2-b3-fused"),
    *(pytest.param(pq_spec(L, (FUSED,), n), id=f"L{L}-b{n}-fused",
                   marks=pytest.mark.slow)
      for L, n in ((3, 1), (4, 1), (4, 3), (7, 1), (7, 3), (10, 1))),
]


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", CELLS)
def test_cuda_fused_j_p_matches_the_reference(spec, request,
                                              compile_device_source,
                                              run_device_binary,
                                              tmp_path) -> None:
    """A fused-lever cell's launch sequence on the RTX 3080, against the oracle."""
    _run_batched("cuda", spec, request, compile_device_source,
                 run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", CELLS)
def test_hip_fused_j_p_matches_the_reference(spec, request,
                                             compile_device_source,
                                             run_device_binary,
                                             tmp_path) -> None:
    """A fused-lever cell's launch sequence on the RX 9060 XT, against the oracle."""
    _run_batched("hip", spec, request, compile_device_source,
                 run_device_binary, tmp_path)
