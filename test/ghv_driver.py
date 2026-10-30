"""The HGP nuclear-attraction gradient built ON A CARD by `ghv_` --
harness.

    ghv_steps(basis, nuclei, M)                         -> [Step]
    gradient_on_card(basis, nuclei, M, backend, tc, workdir) -> [run][n_atom][3]

`hgp_one_e_driver.py`'s shape for `gradient.kernel_ghv`: one launch per
pair class the basis reaches with `i <= j` (all four ordered classes of s
and p), one thread per CONTRACTED SHELL PAIR (`kernel_ghv.launch_shape`),
the arrays `gradient.host.pair_records` stages plus `M`, `Z`, `Cnuc`;
every argument named from the kernel's own parameter list. `G` is the one
OUT, carried from class to class from zero (the kernel's precondition).

ON THE SHARED RUNNER (#649): one runner per class's unit
(`harness_driver.build_steps`, compiled under #92's warning filter), the
chain run `RUNS` times, each run its own processes from a zeroed `G`. The
kernel adds an atom's pairs with fp64 atomics in an order the hardware
chooses, so the runs are compared by the test, not held bitwise (unlike
`hgp_one_e_driver`'s plain stores).

Harness, in `test/`, not a second emitter: the units are `emit/cuda.py` /
`emit/hip.py`'s text of `build_ghv_kernel`, unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from conftest import Toolchain
from harness_driver import Step, build_steps, carried, run_chain
from plan_driver import RUNS

from felizjueves.gradient import host
from felizjueves.gradient import kernel_ghv as kg
from felizjueves.ir import Intent

THREADS = 32  # lanes per block, one shell pair per thread; not numerical


def ghv_steps(basis, nuclei, M) -> list[Step]:
    """One `ghv_` launch per pair class of `basis` (`i <= j`), every argument
    but `G`."""
    classes = sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                      for b in basis.shells[k:]})
    steps = []
    for la, lb in classes:
        kern = kg.build_ghv_kernel((la, lb))
        rec = host.pair_records(basis, nuclei, la, lb)
        arrs = {k: np.asarray(v) for k, v in rec.items()}
        arrs |= {"M": np.asarray(M, dtype=np.float64),
                 "Z": np.array([z for z, _ in nuclei], dtype=np.float64),
                 "Cnuc": np.array([c for _, c in nuclei], dtype=np.float64)}
        want = {p.name for p in kern.params
                if p.array and p.intent is not Intent.OUT}
        if want != set(arrs):
            raise ValueError(f"{kern.name}: arrays {sorted(want)}, staged "
                             f"{sorted(arrs)}")
        n_pair = len(rec["rec0"])
        grid, block = kg.launch_shape(n_pair, THREADS)
        steps.append(Step(kern, grid, block, {
            "n_pair": n_pair, "n_rec": len(rec["s_ab"]), "n_ao": basis.n_ao,
            "n_atom": len(nuclei), **arrs}))
    return steps


def gradient_on_card(basis, nuclei, M, backend: str, tc: Toolchain,
                     workdir: Path) -> np.ndarray:
    """`G` built on the card, `[run][n_atom][3]` over `RUNS` runs. The
    caller holds the card (its device fixture)."""
    steps = ghv_steps(basis, nuclei, M)
    binaries = build_steps(tc, backend, steps, workdir)
    return np.array(run_chain(binaries, steps, carried(steps),
                              np.zeros((len(nuclei), 3)), workdir,
                              runs=RUNS))
