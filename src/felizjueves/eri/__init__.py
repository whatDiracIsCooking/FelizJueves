"""Electron-repulsion integral builds, grouped by cell (#514).

`eri.hjk` is the 2x2's (HGP, split) cell: the HGP `J` and `K` kernels, their
ordered pair stream and its walker, kept together because `kernel_hk` imports
its loop nest and class modes from `kernel_hj`. `eri.density_fit` (#509) is
the density-fitting integrals: the metric `(P|Q)` and the three-center
`(mn|P)`, each with its host reference, its adapter and its builder.

Each module keeps its own layering rule (test/test_layering.py); this package
marker imports nothing.
"""
