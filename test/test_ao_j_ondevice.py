"""The device capstone (#89): `J_mn` for a molecule, `J_p` from generated
kernels COMPILED and LAUNCHED on the RTX 3080 and the RX 9060 XT, against the
same tracked libcint fixture #86 used.

WHAT RUNS WHERE. `ao.j.coulomb_matrix(basis, D, pq=pq, jp=...)` with the
`jp=` seam #87 opened (`ao.drive.KernelJP` is its interpreted twin): the ket
pre-contraction, the bra transform `E` and the scatter stay host-side numpy,
#86's and already witnessed against libcint, so a disagreement here
localises to the device kernels. Only `J_p = sum_k sum_q [p|q] D_q` runs on
the card, fp64 throughout (#22), Boys included (`fj_boys`, emitted).

THE KERNELS ARE THE CLASS SELECTOR'S. For each class `(la, lb | lc, ld)` the
case reaches, `python -m felizjueves.emit --class LA LB LC LD --n-batches N`
(the new CLI path; KERNEL_GEN_CMD's L = 0..16 ladder is untouched) writes the
cell's batch kernels and their launch `order`; the test asserts each file is
byte-identical to `emit(k, arch)` for `kernel.build_pq_kernels(class_spec)`
in that order, then compiles THOSE FILES.

THE LAUNCH MECHANISM IS test_kernel_device.py's, not a new one: its
`BATCHED_HARNESS` and `LAUNCH` (every batch kernel back to back on the
default stream into ONE zeroed `J`, each with the shared memory its own
`Jsh` asks for), its packed-binary input
`n_bra n_ket n_p n_q lanes tau | P zeta Ip | Q eta Iq Dmax | D`, and the
conftest card lock + VRAM probe (#91). What is new is the data: REAL pair
records (#83's `P`, `zeta`, `Q`, `eta`, dressed `D_q`) instead of
`rng.uniform`, and several classes. `Ip = Iq = Dmax = 1` and `tau = 0`, as in
`KernelJP`: the seam is unscreened (`ao.j` refuses `jp` with `tau > 0`), the
bound is never `< tau`, no ket is skipped, and the sorted-ket precondition --
which exists only to make the `break` correct -- is vacuous.

ONE LAUNCH PER CLASS. The seam is called once per (bra shell pair, ket
class); every call of one class has the same ket records (`ao.j` groups kets
by class) and its bra records differ only in which shell pair they came from,
which the kernel never sees (one block per bra record, `J_p` per record). So
`DeviceJ` first captures the sweep's calls through the seam (returning
zeros), concatenates each class's bra records into ONE launch, and then runs
the sweep again with a seam that replays the card's rows -- asserting each
replayed call's arguments are the captured ones. That is a harness economy
(one process per class instead of per shell pair), not a device-side
grouping: each launch is the kernel's ordinary contract.

THE BATCH AXIS, as test_ao_j_interpreted.py: for each `n_batches` in
`ao.drive.BATCH_COUNTS` one full `J` with every class at that count where the
plan admits it, at one batch where it does not; the test asserts exactly the
`(class, n_batches)` cells that ran and were refused.

THE TOLERANCES ARE INHERITED, NOT RE-DERIVED AND NOT LOOSENED, in the
device's precision (fp64):

  libcint    |J_dev - J_pyscf| <= 1e-12 scale_J, element by element
             (test_ao_j.py's derived tolerance; `scale_J` from
             `ao.j.coulomb_matrix_and_scale`).
  reference  |J_dev - J_pq| <= 1e-13 max|J_pq| (test_ao_j_interpreted.py's
             pin of the interpreted kernel). The card differs from the
             interpreter only by FMA contraction and libm `exp`/`sqrt`
             (test_kernel_device.py's docstring): a different rounding, far
             inside 1e-13; and a split cell's atomic adds of n_b <= 9
             partials round by at most 8u of the scale.

WHAT SKIPS: every device test is `gpu` plus `cuda` or `hip` and takes the
compiler and device fixtures, so on a box without the card it SKIPS and the
run is green and empty. The un-marked tests at the bottom are fast-tier: they
pin the class selector's contract and compile nothing.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from oracle import manifest
from test_ao_j_interpreted import _case, _classes, _expected, _pq_j
from test_kernel_device import BACKEND, BATCHED_HARNESS, LAUNCH, PORTABLE, _toolchain

from felizjueves.ao.drive import BATCH_COUNTS, admitted_batches
from felizjueves.ao.drive import class_spec as drive_class_spec
from felizjueves.ao.j import coulomb_matrix
from felizjueves.emit import __main__ as emit_cli
from felizjueves.mmd.kernel import (
    bra_functions,
    build_pq_kernels,
    ket_functions,
    launch_shape,
)
from felizjueves.performance import measured
from felizjueves.reference.pq import pq

CASES = ("h2_sto3g_j", "h2o_sto3g_j")
JK_TOL = 1e-12  # vs libcint, times scale_J: test_ao_j.py's, inherited
INTERP_TOL = 1e-13  # vs reference/pq, relative to max|J_pq|: #87's, inherited

Am = tuple[int, int, int, int]


def _shell_ls(name: str) -> list[int]:
    return [s.l for s in _case(name)[1].shells]


# ---------------------------------------------------------------------------
# The device-backed seam
# ---------------------------------------------------------------------------


class DeviceJ:
    """Compiles each (class, n_batches) cell from the class selector's output
    and launches it with real pair records (module docstring)."""

    def __init__(self, backend: str, n_batches: int, toolchain, compile_fn,
                 run_fn, workdir: Path) -> None:
        self.backend = backend
        self.n_batches = n_batches
        self.tc = toolchain
        self.compile_fn = compile_fn
        self.run_fn = run_fn
        self.workdir = workdir
        self.arch = BACKEND[backend][1]
        self.binaries: dict[tuple[Am, int], Path] = {}
        self.ran: set[tuple[Am, int]] = set()
        self.refused: set[tuple[Am, int]] = set()
        self.launches = 0

    def batches_for(self, am: Am) -> int:
        if self.n_batches in admitted_batches(am, (self.n_batches,)):
            return self.n_batches
        self.refused.add((am, self.n_batches))
        return 1

    def _binary(self, am: Am, n: int) -> Path:
        if (am, n) in self.binaries:
            return self.binaries[(am, n)]
        emit, arch, ext = BACKEND[self.backend]
        gen = self.workdir / "gen"
        argv = ["--backend", self.backend, "--arch", arch.name, "--out", str(gen),
                "--class", *map(str, am), "--n-batches", str(n)]
        assert emit_cli.main(argv) == 0, f"emit {argv} failed"
        cell = gen / emit_cli.cell_dir(am, n)
        chain = cell.joinpath("order").read_text().split()
        # #156: the order is the chain's, DQ -> [p|q] batches -> JAB; only
        # J_p runs on the card here, so only the batches are compiled.
        assert chain[0].startswith("dq_") and chain[-1].startswith("jab_"), chain
        order = chain[1:-1]
        kernels = build_pq_kernels(drive_class_spec(am, n))
        assert order == [f"{k.name}.{ext}" for k in kernels], (
            f"{cell}: launch order {order} is not the plan's")
        includes, launches = [], []
        for i, (fname, k) in enumerate(zip(order, kernels, strict=True)):
            path = cell / fname
            assert path.read_text() == emit(k, arch), (
                f"{path}: the CLI's text is not emit(build_pq_kernels(...))")
            includes.append(f'namespace fj_b{i} {{\n#include "{path}"\n}}')
            launches.append(LAUNCH.format(ns=f"fj_b{i}", name=k.name,
                                          slots=measured.jsh_slots(k), next=i + 1))
        prologue = "#include <hip/hip_runtime.h>" if self.backend == "hip" else ""
        src = BATCHED_HARNESS.format(
            prologue=prologue, portable=PORTABLE, includes="\n".join(includes),
            n_batches=len(kernels), launches="\n".join(launches),
            vram_kernels=", ".join(f"(const void*)fj_b{i}::{k.name}"
                                   for i, k in enumerate(kernels)))
        built = self.compile_fn(self.tc, src)
        dst = self.workdir / f"bin_{emit_cli.cell_dir(am, n)}"
        shutil.move(built, dst)  # the fixture reuses one output name
        self.binaries[(am, n)] = dst
        return dst

    def launch(self, am: Am, bras, kets) -> np.ndarray:
        """`J_p[n_bra, n_p]` for one class on the card."""
        n = self.batches_for(am)
        spec = drive_class_spec(am, n)
        n_p, n_q = len(bra_functions(spec)), len(ket_functions(spec))
        n_bra, n_ket = len(bras), len(kets)
        assert n_bra > 0 and n_ket > 0
        D = np.array([list(k[2]) for k in kets], dtype=np.float64)
        assert D.shape == (n_ket, n_q), f"{am}: D_q {D.shape}, want ({n_ket}, {n_q})"
        _, (_, lanes) = launch_shape(n_bra, self.arch.wavefront)
        head = np.array([n_bra, n_ket, n_p, n_q, lanes, 0.0], dtype=np.float64)
        body = [
            np.array([b[0] for b in bras], dtype=np.float64).ravel(),
            np.array([b[1] for b in bras], dtype=np.float64),
            np.ones(n_bra),
            np.array([k[0] for k in kets], dtype=np.float64).ravel(),
            np.array([k[1] for k in kets], dtype=np.float64),
            np.ones(n_ket), np.ones(n_ket),
            D.ravel(),
        ]
        binary = self._binary(am, n)
        tag = emit_cli.cell_dir(am, n)
        src, dst = self.workdir / f"{tag}.in", self.workdir / f"{tag}.out"
        np.concatenate([head, *body]).tofile(src)
        self.run_fn(binary, str(src), str(dst), "0", "1", vram=self.backend)
        got = np.fromfile(dst, dtype=np.float64).reshape(2, n_bra, n_p)[0]
        assert np.all(np.isfinite(got)), f"{am}: non-finite J_p on {self.arch.name}"
        self.ran.add((am, n))
        self.launches += 1
        return got

    def coulomb_matrix(self, basis, D) -> np.ndarray:
        """`ao.j.coulomb_matrix` with `J_p` from the card, one launch per class."""
        calls: list = []

        def capture(bra_am, ket_am, bras, kets):
            calls.append((bra_am, ket_am, list(bras), kets))
            return np.zeros((len(bras), len(bra_functions(
                drive_class_spec((*bra_am, *ket_am))))))

        coulomb_matrix(basis, D, pq=pq, jp=capture)
        by_class: dict[Am, list[int]] = defaultdict(list)
        for i, (bra_am, ket_am, _, _) in enumerate(calls):
            by_class[(*bra_am, *ket_am)].append(i)
        rows: list = [None] * len(calls)
        for am, idx in by_class.items():
            kets = calls[idx[0]][3]
            assert all(calls[i][3] is kets for i in idx), (
                f"{am}: the seam's ket records differ between calls")
            bras = [b for i in idx for b in calls[i][2]]
            got = self.launch(am, bras, kets)
            at = 0
            for i in idx:
                rows[i] = got[at:at + len(calls[i][2])]
                at += len(calls[i][2])
            assert at == len(got)
        replay = iter(range(len(calls)))

        def jp(bra_am, ket_am, bras, kets):
            i = next(replay)
            c = calls[i]
            assert (c[0], c[1]) == (bra_am, ket_am) and c[2] == list(bras) \
                and list(c[3]) == list(kets), f"replayed call {i} differs"
            return rows[i]

        J = np.array(coulomb_matrix(basis, D, pq=pq, jp=jp))
        assert next(replay, None) is None, "the replay sweep made fewer calls"
        return J


# ---------------------------------------------------------------------------
# J_mn on the card
# ---------------------------------------------------------------------------

# H2O/STO-3G at 6 and 9 batches are `slow`: ~15 s each on the RTX 3080 for
# the split the 1- and 3-batch cells already carry through the same driver.
DEVICE = [pytest.param(name, n, id=f"{name}-b{n}",
                       marks=(pytest.mark.slow,)
                       if name != "h2_sto3g_j" and n > 3 else ())
          for name in CASES for n in BATCH_COUNTS]


def _run(backend: str, name: str, n: int, request, compile_device_source,
         run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    fx, basis = _case(name)
    dev = DeviceJ(backend, n, tc, compile_device_source, run_device_binary,
                  tmp_path)
    J = dev.coulomb_matrix(basis, fx.density)

    ran, refused = _expected(basis, n)
    assert dev.ran == ran and dev.refused == refused
    assert dev.launches == len(_classes(basis))

    R, scale = _pq_j(name)
    ref = fx.arrays[manifest.J_ARRAY]
    jk = float(np.max(np.abs(J - ref) / scale))
    rel = float(np.max(np.abs(J - R)) / np.max(np.abs(R)))
    cells = " ".join(f"{''.join(map(str, am))}b{m}" for am, m in sorted(dev.ran))
    print(f"\n{backend} {dev.arch.name} {name} n_batches={n}: "
          f"{len(dev.ran)} cells ran [{cells}], {len(dev.refused)} refused; "
          f"max|J - get_jk|/scale_J = {jk:.2e}; "
          f"max|J - J_pq|/max|J_pq| = {rel:.2e}")
    assert np.all(np.abs(J - ref) <= JK_TOL * scale), (
        f"{name} b{n} on {dev.arch.name}: J vs libcint, worst err/scale {jk:.2e}")
    assert np.all(np.abs(J - R) <= INTERP_TOL * np.max(np.abs(R))), (
        f"{name} b{n} on {dev.arch.name}: J vs reference/pq J, worst {rel:.2e}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "n"), DEVICE)
def test_cuda_j_matches_libcint(name, n, request, compile_device_source,
                                run_device_binary, tmp_path) -> None:
    """`J_mn` with every class's `J_p` from the RTX 3080, against libcint."""
    _run("cuda", name, n, request, compile_device_source, run_device_binary,
         tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "n"), DEVICE)
def test_hip_j_matches_libcint(name, n, request, compile_device_source,
                               run_device_binary, tmp_path) -> None:
    """`J_mn` with every class's `J_p` from the RX 9060 XT, against libcint."""
    _run("hip", name, n, request, compile_device_source, run_device_binary,
         tmp_path)


# ---------------------------------------------------------------------------
# The class selector's contract (fast tier: nothing compiles)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
def test_the_selector_reaches_what_the_sweep_launches(name):
    """`--shell-ls` derives exactly the classes `ao.j`'s sweep calls the seam
    with -- the set the device test launches."""
    got = set(emit_cli.classes_reached(_shell_ls(name)))
    assert got == _classes(_case(name)[1])


def test_the_selector_key_is_the_drivers():
    for am in emit_cli.classes_reached(_shell_ls("h2o_sto3g_j")):
        for n in admitted_batches(am):
            assert emit_cli.class_spec(am, n) == drive_class_spec(am, n)


@pytest.mark.parametrize("backend", sorted(emit_cli.BACKENDS))
def test_the_selector_writes_the_admitted_cells(backend, tmp_path):
    """`--shell-ls ... --admitted-only`: every admitted cell written in plan
    order with `emit`'s text, every refused one reported and absent."""
    emit, arch, ext = BACKEND[backend]
    ls = _shell_ls("h2o_sto3g_j")
    argv = ["--backend", backend, "--arch", arch.name, "--out", str(tmp_path),
            "--shell-ls", *map(str, ls), "--n-batches", *map(str, BATCH_COUNTS),
            "--admitted-only"]
    assert emit_cli.main(argv) == 0
    for am in emit_cli.classes_reached(ls):
        for n in BATCH_COUNTS:
            cell = tmp_path / emit_cli.cell_dir(am, n)
            if n not in admitted_batches(am):
                assert not cell.exists()
                continue
            ks = build_pq_kernels(drive_class_spec(am, n))
            assert cell.joinpath("order").read_text().split()[1:-1] == [
                f"{k.name}.{ext}" for k in ks]  # between DQ and JAB (#156)
            for k in ks:
                assert (cell / f"{k.name}.{ext}").read_text() == emit(k, arch)


def test_a_refused_cell_is_an_error_without_admitted_only(tmp_path):
    argv = ["--backend", "cuda", "--out", str(tmp_path),
            "--class", "0", "0", "0", "0", "--n-batches", "3"]
    assert emit_cli.main(argv) == 2


def test_the_selector_and_the_ladder_are_separate_paths(tmp_path):
    out = str(tmp_path)
    assert emit_cli.main(["--backend", "cuda", "--out", out, "--L", "1",
                          "--class", "1", "0", "0", "0"]) == 2
    assert emit_cli.main(["--backend", "cuda", "--out", out, "--L", "1",
                          "--n-batches", "3"]) == 2
