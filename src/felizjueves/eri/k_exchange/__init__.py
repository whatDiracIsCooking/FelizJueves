"""The MMD exchange matrix `K` (milestone 10, moved here by #512).

`eri.k_exchange.k` is the host `K` (two sweeps, its Schwarz bound and screen,
`density_decay`), `kernel_k` the `[p|q]`-shaped exchange kernel and its split
class, `levers` its seams and kernel-body levers, `kernel_kblock` the block
exchange kernel on opt-UM's mapping, `launch_k` and `launch_kblock` their
launch plans and walks, `kernel_record_bound` the dressed record bound K's
screen is proved against, and `kdup` the duplicated-work count, J against K.

Each module keeps its own layering rule (test/test_layering.py); this package
marker imports nothing.
"""
