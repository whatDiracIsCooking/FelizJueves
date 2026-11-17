"""The packed AO ERI matrix, host-side: the staging, the reference, and `J`.

    pair_index(n_ao)               -> pidx[m][n], the packed index of (m, n)
    eri_classes(basis)             -> the `kernel_eri` launches, per class
    packed_eri(basis, *, pq)       -> ERI_pp, n_pp x n_pp, the reference
    packed_eri_and_scale(basis, *, pq) -> (ERI_pp, its |.| scale)
    pack_density(D) / unpack(Jp, n_ao)
    coulomb_from_packed(eri, D)    -> J[m][n] = SUM_ls (mn|ls) D_ls

THE PACKING. AO pairs `m >= n` are numbered `pidx(m, n) = m (m + 1) / 2 + n`,
and `pidx` is symmetric (`pidx[m][n] == pidx[n][m]`), so the eight-fold
permutational symmetry of `(mn|ls)` leaves one `n_pp x n_pp` matrix,
`n_pp = n_ao (n_ao + 1) / 2`, symmetric itself (`(mn|ls) = (ls|mn)`):

    ERI_pp[pidx(m, n), pidx(l, s)] = (mn|ls).

THE GEMV. `J_mn = SUM_ls (mn|ls) D_ls`, and for a symmetric `D` the two
orders of an off-diagonal pair carry one integral, so

    J_pp = ERI_pp Dp,   Dp[pidx(l, s)] = D_ls + D_sl  (l != s),  D_ll,

and `J_mn = J_pp[pidx(m, n)]`. For a non-symmetric `D` the same sum is `J` of
its symmetric part, which is what `(mn|ls)`'s symmetry makes `J` anyway.

THE STAGING (`eri_classes`) reuses `eri.schwarz.schwarz_classes`' pair
classes unchanged: every shell pair `i <= j` once, staged `l_A <= l_B`, its
primitive pairs one contiguous record range. A full-ERI class pairs a bra pair
class `X` with a ket pair class `Y` not before it (in `schwarz_classes`'
order), `am = (la, lb, lc, ld)`, and its TASKS are the shell-pair quartets it
owns: every (bra pair, ket pair) when `X != Y`, and `bra >= ket` when
`X == Y`. So every unordered pair of shell pairs is one task, once -- the
eight-fold symmetry, with the four-fold one inside a pair handled by the
scatter (a diagonal shell pair writes its lower triangle only, as
`kernel_schwarz`'s does) and the bra-ket one by the mirror store.

THE REFERENCE is `eri.eri.eri_block` per task, placed by the kernel's rule:
`eri.eri` is witnessed against libcint (test/test_ao_eri.py). `pq` is
INJECTED; nothing here imports `reference/`.

THE CAP. `N_AO_MAX = 150`: the packed matrix is ~1 GB of doubles there, and
its flat index `n_pp^2` stays well inside an `int32` (the kernel's index
type) up to `n_ao ~ 300`. `eri_classes` refuses a larger basis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from felizjueves.ao.shell import Basis
from felizjueves.eri.eri import PQ, eri_block_and_scale
from felizjueves.eri.schwarz import SchwarzClass, schwarz_classes

N_AO_MAX = 150
Matrix = list[list[float]]


def n_packed(n_ao: int) -> int:
    """`n_pp = n_ao (n_ao + 1) / 2`."""
    return n_ao * (n_ao + 1) // 2


def pair_index(n_ao: int) -> list[list[int]]:
    """`pidx[m][n] = M (M + 1) / 2 + N`, `M = max(m, n)`, `N = min(m, n)`."""
    return [[max(m, n) * (max(m, n) + 1) // 2 + min(m, n)
             for n in range(n_ao)] for m in range(n_ao)]


@dataclass(frozen=True)
class EriClass:
    """One `kernel_eri` launch: a bra and a ket pair class, and the tasks.

    `task_bra[t]` indexes `bra.pairs`, `task_ket[t]` `ket.pairs`."""

    bra: SchwarzClass
    ket: SchwarzClass
    task_bra: tuple[int, ...]
    task_ket: tuple[int, ...]

    @property
    def am(self) -> tuple[int, int, int, int]:
        return (self.bra.la, self.bra.lb, self.ket.la, self.ket.lb)

    @property
    def n_task(self) -> int:
        return len(self.task_bra)

    def arrays(self) -> dict[str, list]:
        """The kernel's array parameters except `pidx` and the output, in its
        parameter order (`kernel_eri.RECORD_ARRAYS` and friends), as lists."""
        b, k = self.bra.arrays(), self.ket.arrays()
        return {
            "zeta_ab": b["zeta_ab"], "P": b["P"], "XPA": b["XPA"],
            "XPB": b["XPB"], "s_ab": b["s_ab"],
            "zeta_cd": k["zeta_ab"], "Q": k["P"], "XQC": k["XPA"],
            "XQD": k["XPB"], "s_cd": k["s_ab"],
            "brec0": b["rec0"], "brec1": b["rec1"],
            "off_a": b["off_a"], "off_b": b["off_b"],
            "krec0": k["rec0"], "krec1": k["rec1"],
            "off_c": k["off_a"], "off_d": k["off_b"],
            "task_bra": list(self.task_bra), "task_ket": list(self.task_ket),
        }


def eri_classes(basis: Basis) -> tuple[EriClass, ...]:
    """Every full-ERI class of `basis` with its tasks (module docstring),
    sorted by `am`; refuses `n_ao > N_AO_MAX`."""
    if basis.n_ao > N_AO_MAX:
        raise ValueError(f"n_ao = {basis.n_ao} > N_AO_MAX = {N_AO_MAX}: the "
                         "packed ERI is a small-system path")
    pcs = schwarz_classes(basis)
    out = []
    for i, X in enumerate(pcs):
        for Y in pcs[i:]:
            if X is Y:
                tb = [b for b in range(X.n_pair) for k in range(b + 1)]
                tk = [k for b in range(X.n_pair) for k in range(b + 1)]
            else:
                tb = [b for b in range(X.n_pair) for _ in range(Y.n_pair)]
                tk = [k for _ in range(X.n_pair) for k in range(Y.n_pair)]
            out.append(EriClass(X, Y, tuple(tb), tuple(tk)))
    return tuple(out)


def _lower(n_cart: int, nb: int, diagonal: bool):
    """The rows `(a, b)` a pair writes: all, or `a >= b` when diagonal."""
    return [(a, b) for a in range(n_cart) for b in range(nb)
            if not (diagonal and b > a)]


def _place(basis: Basis, value, keep=None) -> Matrix:
    """`value(sa, sb, sc, sd)`'s block of every task, placed by the kernel's
    scatter, task by task in `eri_classes` order; only the classes whose
    `am` is in `keep` when it is given, 0.0 elsewhere."""
    sh, off = basis.shells, basis.offsets
    pidx = pair_index(basis.n_ao)
    n = n_packed(basis.n_ao)
    out = [[0.0] * n for _ in range(n)]
    for c in eri_classes(basis):
        if keep is not None and c.am not in keep:
            continue
        for tb, tk in zip(c.task_bra, c.task_ket, strict=True):
            a_, b_ = c.bra.pairs[tb]
            c_, d_ = c.ket.pairs[tk]
            blk = value(sh[a_], sh[b_], sh[c_], sh[d_])
            nb, nd = sh[b_].n_cart, sh[d_].n_cart
            for a, b in _lower(sh[a_].n_cart, nb, a_ == b_):
                row = pidx[off[a_] + a][off[b_] + b]
                for cc, d in _lower(sh[c_].n_cart, nd, c_ == d_):
                    col = pidx[off[c_] + cc][off[d_] + d]
                    v = blk[a * nb + b][cc * nd + d]
                    out[row][col] = v
                    out[col][row] = v
    return out


def packed_eri_and_scale(basis: Basis, *, pq: PQ, keep=None
                         ) -> tuple[Matrix, Matrix]:
    """`ERI_pp` and `eri.eri`'s absolute-value scale of each element; only
    the classes in `keep` (a set of `am`) when it is given."""
    cache: dict = {}

    def both(sa, sb, sc, sd):
        key = (id(sa), id(sb), id(sc), id(sd))
        if key not in cache:
            cache[key] = eri_block_and_scale((sa, sb), (sc, sd), pq=pq)
        return cache[key]

    return (_place(basis, lambda *s: both(*s)[0], keep),
            _place(basis, lambda *s: both(*s)[1], keep))


def owned(basis: Basis, keep=None) -> list[list[bool]]:
    """Which elements of `ERI_pp` the classes in `keep` (every class when
    None) write."""
    def ones(sa, sb, sc, sd):
        return [[1.0] * (sc.n_cart * sd.n_cart)] * (sa.n_cart * sb.n_cart)
    return [[x == 1.0 for x in row] for row in _place(basis, ones, keep)]


def packed_eri(basis: Basis, *, pq: PQ) -> Matrix:
    """`ERI_pp[pidx(m, n), pidx(l, s)] = (mn|ls)`, `n_pp x n_pp`."""
    return packed_eri_and_scale(basis, pq=pq)[0]


def pack_density(D: Sequence[Sequence[float]]) -> list[float]:
    """`Dp[pidx(l, s)] = D_ls + D_sl` for `l > s`, `D_ll` on the diagonal."""
    n = len(D)
    return [D[m][k] + D[k][m] if m != k else D[m][m]
            for m in range(n) for k in range(m + 1)]


def unpack(Jp: Sequence[float], n_ao: int) -> Matrix:
    """`J[m][n] = Jp[pidx(m, n)]`, symmetric."""
    pidx = pair_index(n_ao)
    return [[Jp[pidx[m][k]] for k in range(n_ao)] for m in range(n_ao)]


def coulomb_from_packed(eri: Sequence[Sequence[float]],
                        D: Sequence[Sequence[float]]) -> Matrix:
    """`J = unpack(ERI_pp @ pack_density(D))`, the GEMV on the host."""
    Dp = pack_density(D)
    Jp = [sum(e * d for e, d in zip(row, Dp, strict=True)) for row in eri]
    return unpack(Jp, len(D))
