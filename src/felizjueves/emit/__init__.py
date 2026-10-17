"""Backends: a `Kernel` spelled as vendor source, `emit(kernel, arch) -> str`.

`cuda` renders for NVIDIA, `hip` for AMD; `_c` is the vendor-neutral C++
writer both subclass, so each vendor adds only what it spells differently.
`gpu` is the third subclass: one text both device compilers take, its vendor
differences pushed into FelizJueves's own `include/felizjueves/` headers
(no GPU library behind them), for a consumer that builds one source tree
against either vendor. The CLI does not offer it; it is the Python API. A backend
decides nothing: every choice is a `Kernel -> Kernel` pass the interpreter can
check, and the backend only spells the result. It reads the arch's facts off
its `device.Arch` argument and never spells an arch string or a wavefront
width itself (test/test_device.py).
"""

from __future__ import annotations
