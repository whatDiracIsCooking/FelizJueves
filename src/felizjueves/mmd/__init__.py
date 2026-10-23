"""Engine #1: the McMurchie-Davidson recurrence of `[JEngine]` (#505).

The first ERI engine, beside `hgp/`. `mmd.kernel` is the `[p|q]` builder
every MMD consumer builds on, `mmd.recurrence` the Hermite recurrence as IR,
`mmd.e_fragment` the undressed `E` as device-code fragments, and
`mmd.hermite` the host-side `E` coefficients the AO layer dresses.

Each module keeps its own layering rule (test/test_layering.py); this package
marker imports nothing. `reference/pq.py` stays in `reference/`: it is the
generator-independent ground truth the engine is checked against.
"""
