"""The split block exchange kernel on the card, at `(dd|dd)` (#548).

WHAT RUNS. A synthetic basis of two contracted d shells on two centres, so
every K class is `(dd|dd)` (L = 8) and each block has several bra and ket
records; `launch_kblock.kb_launches` plans it with the block kernel, and
test/kblock_driver.py's per-class binaries, built with `split=`, run it in
MODE 0 (`kernel_kblock`) and MODE 2 (`kernel_kblock_split`'s set at
`R = 1`, every kernel on MODE 0's upload), for the `column` and `element`
granularities.

THE ASSERTIONS, the tolerance inherited from test_k_ondevice.py and not
loosened: each mode's `K` within `ORACLE_TOL` (1e-13) of `max|K|` of
`walk_kb`'s and of the other mode's, and every mode bit-identical across
its two runs (no two atomics of a set meet). Interpreted the split set is
the block kernel bit for bit (test_kernel_kblock_split.py); on the card
the compiler may contract its multiply-adds differently, so whether the two
modes agree bitwise is printed, not asserted.

Only `(dd|dd)`, by the user's decision for #548, so every test here is
`slow` (L = 8 > `FAST_L_MAX`): the kernel tier's `--full`. The density
stack (`R > 1`) is witnessed interpreted only. Without a card the `gpu` tests SKIP:
green and empty. The compile-only companions (`cuda` / `hip`) fail on a
warning in generated text (#92).
"""

from __future__ import annotations

from functools import cache, partial
from pathlib import Path

import numpy as np
import pytest
from kblock_driver import class_binaries, kb_driver, run_classes
from plan_driver import compile_checked
from test_k_ondevice import ORACLE_TOL
from test_kernel_device import BACKEND, _toolchain

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.k_exchange.k import record_bounds
from felizjueves.eri.k_exchange.kernel_kblock import build_kblock_kernel
from felizjueves.eri.k_exchange.launch_kblock import KBLaunchPlan, kb_launches, walk_kb
from felizjueves.reference.pq import pq

GRANULARITIES = ("column", "element")


def dd_basis() -> tuple[Basis, np.ndarray]:
    """Two d shells, two primitives each, 1.4 bohr apart: one K class,
    `(dd|dd)`, four blocks, four records a side per block."""
    basis = Basis((Shell(2, (0.0, 0.0, 0.0), (1.3, 0.45), (0.6, 0.5)),
                   Shell(2, (0.3, 0.8, 1.1), (0.9, 0.35), (0.7, 0.4))))
    n = basis.n_ao
    return basis, np.array([[1.0 / (i + j + 1) for j in range(n)]
                            for i in range(n)])


@cache
def _plan() -> KBLaunchPlan:
    basis, D = dd_basis()
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq), tau=0.0,
                       build=build_kblock_kernel)
    assert [c.am for c in plan] == [(2, 2, 2, 2)]
    return plan


def _check(backend: str, granularity: str, request, run_device_binary,
           workdir: Path) -> None:
    tc = _toolchain(backend, request)
    basis, D = dd_basis()
    plan = _plan()
    parts = class_binaries(tc, plan, basis, D, backend, workdir,
                           split=granularity)
    Kw = np.array(walk_kb(plan, basis, D, pq=pq).K)
    big = float(np.max(np.abs(Kw)))
    assert big > 0.0
    got = {}
    for mode in ("kb", "kbs"):
        Ks, _, same = run_classes(parts, basis, mode,
                                  workdir / "runs")
        assert same, f"{mode}: the two runs differ, its atomics met"
        K = Ks[0]
        assert np.all(np.isfinite(K))
        rel = float(np.max(np.abs(K - Kw))) / big
        assert rel <= ORACLE_TOL, f"{mode}: K vs walk_kb {rel:.2e}"
        got[mode] = (K, rel)
    split_vs_block = float(np.max(np.abs(got["kbs"][0] - got["kb"][0]))) / big
    assert split_vs_block <= ORACLE_TOL
    bitwise = got["kbs"][0].tobytes() == got["kb"][0].tobytes()
    print(f"\n{backend} {BACKEND[backend][1].name} (dd|dd) by {granularity}: "
          f"{len(parts[0].drv.classes[0].kbs)} split kernels; kb vs walk "
          f"{got['kb'][1]:.2e}, kbs vs walk {got['kbs'][1]:.2e}, kbs vs kb "
          f"{split_vs_block:.2e} (bitwise {bitwise})")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_cuda_kblock_split_dddd(granularity, request, run_device_binary,
                                tmp_path) -> None:
    _check("cuda", granularity, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_hip_kblock_split_dddd(granularity, request, run_device_binary,
                               tmp_path) -> None:
    _check("hip", granularity, request, run_device_binary, tmp_path)


def _compile_only(backend: str, tc, tmp_path: Path) -> None:
    basis, _ = dd_basis()
    drv = kb_driver(_plan(), basis, backend, tmp_path / "gen",
                    split="column")
    compile_checked(tc, drv, tmp_path / "kb_driver")


@pytest.mark.cuda
@pytest.mark.slow
def test_cuda_kblock_split_driver_compiles(nvcc, tmp_path) -> None:
    _compile_only("cuda", nvcc, tmp_path)


@pytest.mark.hip
@pytest.mark.slow
def test_hip_kblock_split_driver_compiles(hipcc, tmp_path) -> None:
    _compile_only("hip", hipcc, tmp_path)
