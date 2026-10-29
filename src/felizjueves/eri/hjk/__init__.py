"""The split HGP `J` / `K` cell (milestone 14, moved here by #514).

`kernel_hj` and `kernel_hk` are the HGP Coulomb and exchange kernels (`hk`
reuses `hj`'s nest, staging and class modes), `kernel_rj` and `kernel_rk`
the same two on Rys quadrature (each through its HGP sibling's seam),
`launch_hjk` the ordered pair stream, and `walk_hjk` its oracle. They stay
in one package so that the exchange kernel's import of the Coulomb one
never crosses packages.

Each module keeps its own layering rule (test/test_layering.py); this package
marker imports nothing.
"""
