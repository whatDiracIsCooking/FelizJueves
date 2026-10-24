"""The two-electron part of the Fock matrix, by EVERY ordered shell quartet.

    coulomb_exchange_reference(basis, D, *, eri) -> (J, K)   one sweep
    fock_two_electron_reference(basis, D, *, eri) -> G       G = J - K / 2
    fock_build_reference                                     = the above

This is the WITNESS, written before the kernel that needs it (#275): the
fused, symmetric Fock build of milestone 12 is checked against this module,
and this module against FJ's own `eri.j_coulomb.j` and `eri.k_exchange.k`
(test/test_ao_fock.py).
It has the discipline of `eri.j_coulomb.j.coulomb_matrix_reference` and
`eri.k_exchange.k.exchange_matrix_reference`, for the same reason: it is the only thing
a
symmetric path can be checked against without checking it against itself.

THE SWEEP. For every ORDERED shell quartet `(a, b, c, d)` -- all `n_sh^4` of
them, each visited exactly once -- the injected `eri` returns the block
`B = (ab|cd)` (rows `ab`, columns `cd`, each in `cartesians` order with the
first index outer: `eri.eri.eri_block`'s layout). The same block is digested
twice, both at WEIGHT 1:

    J[a_i][b_j] += sum_{kl} D[c_k][d_l] B[ij][kl]        (Coulomb)
    K[a_i][c_k] += sum_{jl} D[b_j][d_l] B[ij][kl]        (exchange)

Every term of the defining sums is reached exactly once: `J_mn`'s sum over
`(l, s)` is the quartets `(sh(m), sh(n), c, d)` over every `c, d`, and
`K_mn`'s is `(sh(m), b, sh(n), d)` over every `b, d`. NO SYMMETRY IS USED,
DELIBERATELY: no quartet is skipped because it is a permutation of another,
no weight other than 1 appears, and `G[n][m]` is computed, never copied from
`G[m][n]` -- so the symmetry of `G` is a CHECK on the sweep, not a property
it was built to have. A symmetric path's weights (the degeneracy factor,
the six blocks, the Coulomb blocks' 2) must be derived against this sum;
the predecessor checked its digestion against a re-typing of itself and
shipped the Coulomb blocks at weight 1 instead of 2 (milestone 12, pin 6).
Slow, and meant to be: do not "optimise" it.

THE ERI IS INJECTED (the tree's inversion of control): `eri(bra, ket)`
takes two shell pairs and returns `(ab|cd)`, so this module is
engine-neutral -- `functools.partial(eri.eri.eri_block, pq=...)` today, an
HGP evaluator later, with no edit here. It imports no `reference/`.

THE COMBINATION, AND THE DENSITY CONVENTION -- DERIVED, NOT TRANSCRIBED.
Closed-shell restricted Hartree-Fock, real orbitals and real AOs. Let the
occupied SPIN orbitals be `I = (i, sigma)`, spatial `phi_i = sum_m C_mi chi_m`.
In chemists' notation the two-electron energy is

    E2 = 1/2 sum_{I,J} [ (II|JJ) - (IJ|JI) ],

and the exchange integral carries a spin overlap `delta(sigma_I, sigma_J)`:
exchange couples only electrons of the same spin, Coulomb couples all.
Define the SPIN densities `P^s_mn = sum_{i occ in s} C_mi C_ni` and the TOTAL
(spin-summed) density `D = P^alpha + P^beta`. Expanding the MOs,

    sum_{I,J} (II|JJ) = sum_{mnls} D_mn D_ls (mn|ls)
    sum_{I,J} (IJ|JI) = sum_s sum_{mnls} P^s_ms P^s_nl (mn|ls)
                      = sum_s sum_{mnls} P^s_mn P^s_ls (ml|sn),

the second line from `sum_{I in s} C_mI C_sI = P^s_ms` (and the same for
`J`), the third by relabelling the dummies `(n, l, s) -> (l, s, n)`, so
that `P^s` carries `(m, n)` and `(l, s)`. The spin-`s` Fock
matrix is `dE2 / dP^s_mn`. Each quadratic form is symmetric in its two
density factors (the 8-fold symmetry of real ERIs), so differentiating
gives twice one factor, and the 2 cancels the leading 1/2. With
`dD / dP^s = 1`:

    G^s_mn = sum_ls D_ls (mn|ls) - sum_ls P^s_ls (ml|sn)
           = J[D]_mn - K[P^s]_mn,    (ml|sn) = (ml|ns) for real AOs,

where `J[X]_mn = sum_ls X_ls (mn|ls)` and `K[X]_mn = sum_ls X_ls (ml|ns)`
-- exactly `eri.j_coulomb.j`'s `J` and `eri.k_exchange.k`'s `K` (`K_ab = sum_cd D_cd
(ac|bd)`).
Closed shell: `P^alpha = P^beta = D / N_SPIN`, `N_SPIN = 2`, and `K` is
linear, so

    G = J[D] - (1 / N_SPIN) K[D] = J - K / 2,     D the TOTAL density.

The `1/2` is `1 / N_SPIN`: Coulomb sees both spins' density, exchange only
its own spin's. That is how `EXCHANGE_FRACTION` is spelled below.

RELATION TO PySCF. `get_jk(mol, dm)` returns `vj_mn = sum_ls (mn|ls) dm_ls`
and `vk_mn = sum_ls (ml|ns) dm_ls` (test/oracle/manifest.py records both
definitions), which are `J[dm]` and `K[dm]` above; and PySCF's RHF `dm` is
the total density. So `G = vj - vk / 2` follows from the derivation and
is CHECKED against the tracked `_j` / `_k` fixtures (which share one density,
asserted before use) -- it is not where the combination came from. The
fixtures' density is a Hilbert matrix, not an idempotent one; `G` is linear
in `D`, so the relation holds for any symmetric `D` read as a total density.

THE DENSITY MUST BE SYMMETRIC (`eri.j_coulomb.j`'s contract, shared): `(ml|sn) =
(ml|ns)` above and the identification with `K[D]` rest on it, and an
asymmetric `D` is rejected, never symmetrised. Pure Python floats; `D` is
anything indexable as `D[m][n]`; matrices are tuples of rows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from felizjueves.ao.density import checked_density
from felizjueves.ao.shell import Basis, Shell

ShellPair = tuple[Shell, Shell]
ERI = Callable[[ShellPair, ShellPair], Sequence[Sequence[float]]]
Matrix = tuple[tuple[float, ...], ...]

N_SPIN = 2  # closed shell: D = P^alpha + P^beta, each D / N_SPIN
COULOMB_FRACTION = 1.0  # J sees the whole density D (module docstring)
EXCHANGE_FRACTION = 1.0 / N_SPIN  # K sees one spin's density, D / N_SPIN


def _block(eri: ERI, s: tuple[Shell, Shell, Shell, Shell]) -> list[list[float]]:
    sa, sb, sc, sd = s
    got = eri((sa, sb), (sc, sd))
    nr, nc = sa.n_cart * sb.n_cart, sc.n_cart * sd.n_cart
    try:
        rows = [[float(v) for v in r] for r in got]
    except TypeError:
        rows = None
    if rows is None or len(rows) != nr or any(len(r) != nc for r in rows):
        raise ValueError(
            f"eri returned the wrong shape for quartet "
            f"{(sa.l, sb.l, sc.l, sd.l)}: expected {nr} x {nc}")
    return rows


def coulomb_exchange_reference(basis: Basis, D: object, *,
                               eri: ERI) -> tuple[Matrix, Matrix]:
    """`J[D]` and `K[D]` from ONE sweep over every ordered shell quartet,
    each block digested once into `J` and once into `K`, weight 1, no
    symmetry (module docstring)."""
    rows = checked_density(basis, D)
    shells, off = basis.shells, basis.offsets
    n_sh = len(shells)
    n = basis.n_ao
    J = [[0.0] * n for _ in range(n)]
    K = [[0.0] * n for _ in range(n)]
    for a in range(n_sh):
        for b in range(n_sh):
            for c in range(n_sh):
                for d in range(n_sh):
                    sa, sb, sc, sd = shells[a], shells[b], shells[c], shells[d]
                    B = _block(eri, (sa, sb, sc, sd))
                    na, nb, nc, nd = (sa.n_cart, sb.n_cart, sc.n_cart,
                                      sd.n_cart)
                    oa, ob, oc, od = off[a], off[b], off[c], off[d]
                    for i in range(na):
                        for j in range(nb):
                            row = B[i * nb + j]
                            acc = 0.0  # Coulomb: sum over the ket pair (c, d)
                            for k in range(nc):
                                drow = rows[oc + k]
                                for ll in range(nd):
                                    acc += drow[od + ll] * row[k * nd + ll]
                            J[oa + i][ob + j] += acc
                            # exchange: (a_i b_j|c_k d_l) D[b_j][d_l]
                            # to K[a_i][c_k]
                            dbj = rows[ob + j]
                            for k in range(nc):
                                acc = 0.0
                                for ll in range(nd):
                                    acc += dbj[od + ll] * row[k * nd + ll]
                                K[oa + i][oc + k] += acc
    return tuple(tuple(r) for r in J), tuple(tuple(r) for r in K)


def fock_two_electron_reference(basis: Basis, D: object, *,
                                eri: ERI) -> Matrix:
    """`G = J[D] - K[D] / 2` for the TOTAL closed-shell density `D`, by
    every ordered quartet at weight 1 (module docstring: the derivation).
    `F = h + G`; `h` is `one_electron.kinetic` + `one_electron.potential`, not formed
    here."""
    J, K = coulomb_exchange_reference(basis, D, eri=eri)
    return tuple(
        tuple(COULOMB_FRACTION * jv - EXCHANGE_FRACTION * kv
              for jv, kv in zip(jr, kr, strict=True))
        for jr, kr in zip(J, K, strict=True))


fock_build_reference = fock_two_electron_reference
