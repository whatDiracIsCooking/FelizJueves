"""FelizJueves: code generation of GPU kernels for high-L ERIs and J.

The generator emits one kernel family: the `[p|q]` kernel (`kernel`, spelled
by `emit/`), one batch or a split class. Around it, `plan/` counts, `measured`
records what the device toolchains and cards report, `literature/` scores
against the published tables, and `ao/` builds `J_mn` for a real molecule
host-side, with `J_p` from the generated kernel (docs/implementation/
j_engine.md, "The Coulomb matrix, witnessed from outside"). Only `measured`'s
`--remeasure` drives a device compiler; launching on a card is the tests'
business (test/conftest.py, devtools/kernel-tier.sh).
"""
