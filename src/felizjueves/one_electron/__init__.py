"""The one-electron family: `S`, `T`, `V` and the cross-basis overlap (#508).

The host references are `one_electron.one_electron` (the symmetric assembly
`S`, `T` and `V` share), `one_electron.overlap`, `one_electron.kinetic` and
`one_electron.potential`; the kernels are `one_electron.kernel_s`,
`kernel_sx`, `kernel_t` and `kernel_v`. Their launch plans
(`one_electron_launch`, `cross_overlap_launch`) stay in `launch`.

Each module keeps its own layering rule (test/test_layering.py); this package
marker imports nothing.
"""
