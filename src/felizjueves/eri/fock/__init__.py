"""The fused Fock build (milestone 12, moved here by #513).

`fock` is the host sweep `G = J - K/2` over every ordered quartet, `kernel_f`
the fused HGP kernel and `kernel_f_scatter` its scatter, `launch_f` the pair
stream, `walk_f` its oracle, and `levers/` the `F` levers. The coming Rys `F`
(milestone 17) builds on this location.

Each module keeps its own layering rule (test/test_layering.py); this package
marker imports nothing.
"""
