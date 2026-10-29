"""`geri3` and `gmetric` interpreted with the DF-K weights (#620).

The kernels contract any `Gam` / `W` (test_kernel_geri3.py and
test_kernel_gmetric.py hold them to `gradient.host_df` on random ones); this
holds them to it on the EXCHANGE-shaped ones FreeBreakfast will stage:
`host_df.df_k_stage`'s `Gam^K` / `W^K` (symmetric, and built from occupied
coefficients through a metric solve), and the whole DF-RHF two-electron
weights `Gam^J - 1/2 Gam^K`, `W^J - 1/2 W^K`. One launch per class over
every staged cell / pair, summed, against the host's whole gradient.

ROUND-OFF, NOT BITWISE: `|G_kernel - G_host| <= TOL max(scale)`, `TOL = 16
eps`, `scale` the split-centre scale of test_kernel_geri3.py /
test_kernel_gmetric.py (the host's `|G|` cell by cell with every shell on a
row of its own, folded back onto the atoms).

SCOPE: the s and p shells of test_gradient_host_df.py's three-atom bases
(every class through (pp|p) and (p|p), so `L + 1 <= 4`: fast); the device
run is not here (the kernels already run on both cards with an arbitrary
`Gam`, test_grad_df_device.py).
"""

from __future__ import annotations

from functools import cache

import numpy as np
from df_grad_launch import run_eri3, run_metric

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.density_fit.eri3 import eri3_tensor
from felizjueves.eri.density_fit.metric import metric_matrix
from felizjueves.gradient import host_df
from felizjueves.gradient.host import shell_atoms
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq

EPS = 2.0 ** -52
TOL = 16 * EPS  # of max(scale)

_A, _B, _C = (0.1, -0.2, 0.3), (1.2, 0.4, -0.5), (-0.6, 0.9, 0.2)
NUCLEI = ((8.0, _A), (1.0, _B), (1.0, _C))
BASIS = Basis((Shell(0, _A, (1.3, 0.4), (0.9, 0.3)),
               Shell(1, _A, (0.8,), (1.1,)),
               Shell(1, _B, (0.6,), (0.7,)),
               Shell(0, _B, (1.7,), (0.5,))))
AUX = Basis((Shell(0, _C, (2.0, 0.4), (0.7, 0.5)),
             Shell(1, _A, (0.85,), (1.0,)),
             Shell(1, _B, (1.4,), (0.6,)),
             Shell(0, _A, (3.0,), (1.0,))))


@cache
def _stages() -> tuple[host_df.DfStage, host_df.DfStage]:
    C = np.random.default_rng(17).standard_normal((BASIS.n_ao, 3)) / 2
    T = eri3_tensor(BASIS, AUX, pq=pq)
    M = metric_matrix(AUX, pq=pq)
    D = (2.0 * C @ C.T).tolist()
    return (host_df.df_j_stage(T, M, D),
            host_df.df_k_stage(T, M, C.tolist()))


def _rhf() -> tuple[np.ndarray, np.ndarray]:
    j, k = _stages()
    return (np.array(j.Gam) - 0.5 * np.array(k.Gam),
            np.array(j.W) - 0.5 * np.array(k.W))


def _fold(g, split, real, sc) -> None:
    for i, row in enumerate(split):
        sc[real[i]] += g[row]


def _eri3_scale(Gam) -> float:
    nat, ns = len(NUCLEI), len(BASIS.shells)
    split = tuple(nat + i for i in range(ns))
    asplit = tuple(nat + ns + i for i in range(len(AUX.shells)))
    sc = np.zeros((nat, 3))
    for i in range(ns):
        for j in range(i, ns):
            for n in range(len(AUX.shells)):
                g = np.abs(np.array(host_df.eri3_gradient(
                    BASIS, AUX, Gam, NUCLEI, pq=pq, cells=[(i, j, n)],
                    atoms=split, aux_atoms=asplit)))
                _fold(g, split, shell_atoms(BASIS, NUCLEI), sc)
                _fold(g, asplit, shell_atoms(AUX, NUCLEI), sc)
    return float(sc.max())


def _metric_scale(W) -> float:
    nat, ns = len(NUCLEI), len(AUX.shells)
    split = tuple(nat + i for i in range(ns))
    sc = np.zeros((nat, 3))
    for i in range(ns):
        for j in range(i, ns):
            g = np.abs(np.array(host_df.metric_gradient(
                AUX, W, NUCLEI, pq=pq, pairs=[(i, j)], atoms=split)))
            _fold(g, split, shell_atoms(AUX, NUCLEI), sc)
    return float(sc.max())


def _close(got, want, scale: float, what: str) -> None:
    err = float(np.abs(got - want).max())
    print(f"\n{what}: {err / scale / EPS:.2f} eps of scale")
    assert err <= TOL * scale, f"{what}: {err / scale / EPS:.1f} eps of scale"


def test_the_bases_reach_pp_p_and_the_k_weights_are_symmetric() -> None:
    assert max(s.l for s in BASIS.shells) == max(s.l for s in AUX.shells) == 1
    _, k = _stages()
    g = np.array(k.Gam)
    assert np.abs(g - g.transpose(1, 0, 2)).max() <= 1e-13 * np.abs(g).max()
    assert np.array_equal(np.array(k.W), np.array(k.W).T)


def _eri3(Gam, what: str) -> None:
    G, ran = run_eri3(BASIS, AUX, NUCLEI, Gam, boys=boys)
    assert max(sum(c) for c in ran) == 3  # (pp|p)
    want = np.array(host_df.eri3_gradient(BASIS, AUX, Gam, NUCLEI, pq=pq))
    _close(G, want, _eri3_scale(Gam), what)


def _metric(W, what: str) -> None:
    G, ran = run_metric(AUX, NUCLEI, W, boys=boys)
    assert (1, 1) in ran
    want = np.array(host_df.metric_gradient(AUX, W, NUCLEI, pq=pq))
    _close(G, want, _metric_scale(W), what)


def test_geri3_with_the_df_k_gam_is_the_host() -> None:
    _eri3(np.array(_stages()[1].Gam), "geri3, Gam^K")


def test_gmetric_with_the_df_k_w_is_the_host() -> None:
    _metric(np.array(_stages()[1].W), "gmetric, W^K")


def test_the_whole_df_rhf_weights_interpreted_are_the_host() -> None:
    Gam, W = _rhf()
    _eri3(Gam, "geri3, Gam^J - 1/2 Gam^K")
    _metric(W, "gmetric, W^J - 1/2 W^K")
