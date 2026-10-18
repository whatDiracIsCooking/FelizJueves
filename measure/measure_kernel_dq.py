"""#153's named assertion that the `[p|q]` text is untouched, against the
measured store: every `source_sha256` row of data/measured/kernels.json still
hashes the text the generator writes today.

Moved from test/test_kernel_dq.py, which keeps the goldens (the same claim
against tracked text, which is correctness); this one reads a measured store,
so it is `devtools/measure.sh check` (`--full`: `slow`, ~50 s, every
recorded cell re-emitted). measure_store.py runs the same check per chunk.
"""

from __future__ import annotations

import measure_store
import pytest


@pytest.mark.slow
def test_every_recorded_pq_source_sha256_is_untouched() -> None:
    for chunk in measure_store.CHUNKS:
        arch, target = chunk.values
        measure_store.test_the_recorded_source_is_what_the_generator_writes_today(
            arch, target)
