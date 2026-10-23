"""The density-fitting integrals (#509): milestones 15 and 16.

`eri.density_fit.metric` is the DF Coulomb metric `(P|Q)`'s host reference
and staging, `metric_drive` its interpreted adapter and `kernel_metric` its
kernel builder; `eri3`, `eri3_drive` and `kernel_eri3` are the same three for
the three-center integral `(mn|P)`. Nothing consumes them yet: DF-J and DF-K
are out of scope.

Each module keeps its own layering rule (test/test_layering.py); this package
marker imports nothing.
"""
