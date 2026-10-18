"""The tracked ceiling record behind each card's ridge (#141): the ridge
reads the measured values in data/measured/ceilings.json and no published
figure, so the vendors' block may change or vanish without moving a ceiling,
a ridge or a verdict.

Moved from test/test_device.py, which keeps `Arch` and `MeasuredKey` as code
(and the loader on files it writes itself): this one reads the tracked
record, so it is `devtools/measure.sh check`.
"""

from __future__ import annotations

import json
from pathlib import Path

from felizjueves.device import (
    ARCHS,
    CEILINGS_PATH,
    Arch,
    recorded_ceilings,
)


def test_the_ridge_reads_the_measured_values_and_no_published_figure(
        tmp_path: Path) -> None:
    # #141's acceptance: no verdict rests on an unstated assumption. The
    # published block (vendor figures, gfx1200's per-clock bound) may change
    # or vanish without moving a ceiling, a ridge or a verdict.
    data = json.loads(CEILINGS_PATH.read_text())
    data["published"] = {}
    stripped = tmp_path / "ceilings.json"
    stripped.write_text(json.dumps(data))
    for arch in ARCHS:
        got = recorded_ceilings(arch.name, stripped)
        assert got == recorded_ceilings(arch.name)
        assert Arch(arch.vendor, arch.name, arch.wavefront, **got).ridge == (
            arch.ridge)
