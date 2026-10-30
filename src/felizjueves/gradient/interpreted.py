"""The d1 kernels as `gradient.rhf.Runners`, run in the interpreter (#565).

    interpreted_runners(boys=, lanes=)   -> Runners
    g1e_gradient(target, basis, nuclei, M, *, boys)      gs_ / gt_ / gv_
    gjab_gradient(basis, nuclei, D, *, boys)             [p|q]@L+1 -> gjab_
    gk_gradient(basis, nuclei, D, *, boys, lanes=)       gk_
    ghj_gradient(basis, nuclei, D, *, boys, threads=)    ghj_ (HGP J)
    ghj_class_gradient(basis, nuclei, D, cls, *, boys)   one ordered class
    gf_launches(basis, nuclei)                           the fused F's stream
    gf_pair(basis, atoms, a, b) / gf_run(am, diag, ...)  one record / launch
    gf_gradient(basis, nuclei, D, *, boys, threads=)     gf_, dE_J - 1/2 dE_K

Every term of the RHF gradient from the GENERATED kernels, with no compiler
and no card: one launch per class into one zeroed `G` (the kernels'
precondition, `kernel_g1e.ZEROED_G`), classes in sorted order, each launch's
arrays packed as `test/grad_driver.py` packs them for a card:

- `gs_` / `gt_` / `gv_{la}{lb}`: every record of the class's shell pairs
  `i <= j` (`gradient.host.records`); `gv_` gets the nuclei (`Z`, `Cnuc`)
  and computes the Hellmann-Feynman term itself. The default form
  (`kernel_g1e.g1e_lowstack`'s rule) is the one built.
- THE J CHAIN: per bra class `(la, lb)`, `J_p` at Hermite order `L + 1`
  from the interpreted d0 `[p|q]` kernel of `kernel_gjab.gjab_pq_spec`'s
  class (the bra raised by one, an ordinary `Target.J` class run by
  `eri.j_coulomb.drive.KernelJP`) over `gradient.host.coulomb_kets`, one
  ket class `lq` at a time; then `gjab_{la}{lb}` reading it. The host's
  `J_p` is not used.
- `gk_{la}{lc}_{lb}{ld}`: every K class the basis's `l`s reach, every
  ordered bra record against every ordered ket record
  (`gradient.host.exchange_records`), one warp of `lanes` per bra record.
- `ghj_{la}{lb}{lc}{ld}`, the Coulomb term on the HGP recurrence: every
  ordered class the basis's `l`s reach, its bra side every ordered shell
  pair of `(la, lb)` and its ket side every one of `(lc, ld)` --
  `eri.hjk.walk_hjk.pair_record`'s staged records, packed by
  `kernel_f.pack`, one launch per pair of contraction degrees (a packed
  side is degree-uniform) -- plus each bra pair's exponents and atoms.
  `interpreted_runners(coulomb="hgp")` runs it in place of the J chain.
- `gf_{la}{lb}{lc}{ld}`, THE FUSED TWO-ELECTRON TERM (`gradient.kernel_gf`):
  not one of `Runners`' terms but their sum, `dE_J - 1/2 dE_K`, in one
  kernel per F class. `gf_launches` is its stream, built here from the
  basis alone with nothing screened or pruned (the kernel has no screen):
  every unordered shell pair written canonically (`launch_f`'s rule,
  `la <= lb`, the lower index first on a tie), batched by `(la, lb)` and
  contraction degree, each batch pair `I <= J` one launch of class
  `(la_I, lb_I | la_J, lb_J)`, `diag = 1` exactly on `I == J` -- so every
  unique quartet is met once, as `kernel_f`'s canonical half requires.

BOYS IS INJECTED (`boys=`), as the interpreter requires: nothing here
imports `reference/`. The interpreted `G` is the sum the card's `atomicAdd`s
make, in a fixed order (the interpreter's), so it is reproducible here and
not on a card.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import cache
from itertools import product

import numpy as np

from felizjueves.ao.shell import Basis
from felizjueves.eri.fock.kernel_f import pack
from felizjueves.eri.hjk.walk_hjk import pair_record
from felizjueves.eri.j_coulomb.drive import KernelJP
from felizjueves.gradient import host
from felizjueves.gradient import kernel_gf as GF
from felizjueves.gradient.kernel_g1e import build_g1e_kernel, launch_shape
from felizjueves.gradient.kernel_ghj import build_ghj_kernel
from felizjueves.gradient.kernel_ghj import launch_shape as ghj_launch_shape
from felizjueves.gradient.kernel_gjab import build_gjab_kernel, n_p1
from felizjueves.gradient.kernel_gk import build_gk_kernel
from felizjueves.gradient.kernel_gk import launch_shape as gk_launch_shape
from felizjueves.gradient.rhf import Matrix, Runners
from felizjueves.ir import Kernel, Scalar
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.mmd.hermite import hermite_order
from felizjueves.one_electron.potential import Nucleus
from felizjueves.spec import Target

Boys = Callable[[int, float], float]
THREADS = 32  # threads per block for the per-record kernels; not numerical


@cache
def _g1e(target: Target, la: int, lb: int) -> Kernel:
    return build_g1e_kernel((target, la, lb))


@cache
def _gjab(la: int, lb: int) -> Kernel:
    return build_gjab_kernel((la, lb))


@cache
def _gk(am: tuple[int, int, int, int]) -> Kernel:
    return build_gk_kernel(am)


@cache
def _ghj(am: tuple[int, int, int, int]) -> Kernel:
    return build_ghj_kernel(am)


@cache
def _gf(am: tuple[int, int, int, int]) -> Kernel:
    return GF.build_gf_kernel(am)


def _pair_classes(basis: Basis) -> list[tuple[int, int]]:
    sh = basis.shells
    return sorted({(a.l, b.l) for k, a in enumerate(sh) for b in sh[k:]})


def _arrays(rec: dict[str, list], kern: Kernel) -> dict[str, np.ndarray]:
    """`rec`'s lists the kernel takes, typed by its parameters."""
    params = {p.name: p for p in kern.params if p.array}
    return {k: np.asarray(v, dtype=np.int32 if params[k].type is Scalar.I32
                          else np.float64)
            for k, v in rec.items() if k in params}


def _zero(nuclei: Sequence[Nucleus]) -> np.ndarray:
    return np.zeros((len(nuclei), 3))


def g1e_gradient(target: Target, basis: Basis, nuclei: Sequence[Nucleus],
                 M: Matrix, *, boys: Boys) -> np.ndarray:
    """`SUM M dX/dR`, `X` = `S`, `T` or `V` (Hellmann-Feynman included)."""
    G = _zero(nuclei)
    Mx = np.asarray(M, dtype=np.float64)
    for la, lb in _pair_classes(basis):
        kern = _g1e(target, la, lb)
        rec = host.records(basis, nuclei, la, lb)
        arrs = _arrays(rec, kern)
        if target is Target.V:
            arrs["Z"] = np.array([float(z) for z, _ in nuclei])
            arrs["Cnuc"] = np.array([[float(x) for x in c] for _, c in nuclei])
        n = len(rec["s_ab"])
        grid, block = launch_shape(n, THREADS)
        G = Interpreter(kern, launch=Launch(grid, block, 1), boys=boys).run(
            n_rec=n, n_ao=basis.n_ao, n_atom=len(nuclei), M=Mx, G=G,
            **arrs)["G"]
    return G


def _chain_jp(la: int, lb: int, rec: dict[str, list], kets,
              jp: KernelJP) -> np.ndarray:
    """`J_p` at `L + 1` per record, from the interpreted `[p|q]` kernel of
    the class with the bra raised by one, one ket class at a time."""
    bras = [((rec["p_x"][r], rec["p_y"][r], rec["p_z"][r]), rec["zeta_ab"][r])
            for r in range(len(rec["s_ab"]))]
    lq_of = {len(hermite_order(L)): L for L in range(16)}
    by_lq: dict[int, list] = {}
    for k in kets:
        by_lq.setdefault(lq_of[len(k.qs)], []).append((k.Q, k.eta, k.dq))
    out = np.zeros((len(bras), n_p1(la, lb)))
    for lq, ks in sorted(by_lq.items()):
        out += jp((la + 1, lb), (lq, 0), bras, ks)
    return out


def gjab_gradient(basis: Basis, nuclei: Sequence[Nucleus], D: Matrix, *,
                  boys: Boys) -> np.ndarray:
    """`dE_J/dR` through the interpreted chain `[p|q]@L+1 -> gjab_`."""
    G = _zero(nuclei)
    Dx = np.asarray(D, dtype=np.float64)
    kets = host.coulomb_kets(basis, D)
    jp = KernelJP(boys=boys)
    for la, lb in _pair_classes(basis):
        kern = _gjab(la, lb)
        rec = host.records(basis, nuclei, la, lb)
        n = len(rec["s_ab"])
        arrs = _arrays(rec, kern) | {
            "XPA": np.array([rec[f"pa_{a}"] for a in "xyz"]).T.copy(),
            "XPB": np.array([rec[f"pb_{a}"] for a in "xyz"]).T.copy(),
            "s": np.asarray(rec["s_ab"], dtype=np.float64),
            "row": np.arange(n, dtype=np.int32),
            "Jp": _chain_jp(la, lb, rec, kets, jp)}
        grid, block = launch_shape(n, THREADS)
        G = Interpreter(kern, launch=Launch(grid, block, 1)).run(
            n_rec=n, n_ao=basis.n_ao, n_atom=len(nuclei), n_rows=n, M=Dx,
            G=G, **arrs)["G"]
    return G


def gk_gradient(basis: Basis, nuclei: Sequence[Nucleus], D: Matrix, *,
                boys: Boys, lanes: int = 4) -> np.ndarray:
    """`dE_K/dR`, `E_K = 1/2 SUM D_ab D_cd (ac|bd)`, every K class."""
    G = _zero(nuclei)
    Dx = np.asarray(D, dtype=np.float64)
    ls = sorted({s.l for s in basis.shells})
    for am in [(a, c, b, d) for a in ls for c in ls for b in ls for d in ls]:
        kern = _gk(am)
        bra, ket = host.exchange_records(basis, nuclei, am)
        n_bra, n_ket = len(bra["s_ac"]), len(ket["s_bd"])
        if not (n_bra and n_ket):
            continue
        grid, block = gk_launch_shape(n_bra, lanes)
        G = Interpreter(kern, launch=Launch(grid, block, lanes),
                        boys=boys).run(
            n_bra=n_bra, n_ket=n_ket, n_ao=basis.n_ao, n_atom=len(nuclei),
            M=Dx, G=G, **_arrays({**bra, **ket}, kern))["G"]
    return G


def _ordered_pairs(basis: Basis, la: int, lb: int) -> dict[int, list]:
    """Every ordered shell pair of class `(la, lb)`, `a` outer, by its
    contraction degree `K_a K_b`."""
    sh = basis.shells
    out: dict[int, list] = {}
    for a, sa in enumerate(sh):
        for b, sb in enumerate(sh):
            if (sa.l, sb.l) == (la, lb):
                n = len(sa.exponents) * len(sb.exponents)
                out.setdefault(n, []).append((a, b))
    return out


def ghj_class_gradient(basis: Basis, nuclei: Sequence[Nucleus], D: Matrix,
                       cls: tuple[int, int, int, int], *, boys: Boys,
                       threads: int = 4, G: np.ndarray | None = None
                       ) -> np.ndarray:
    """One ordered class's `ghj_` launches over every ordered quartet of
    it, added into `G` (zeroed when not given)."""
    G = _zero(nuclei) if G is None else G
    Dx = np.asarray(D, dtype=np.float64)
    sh = basis.shells
    atom = host.shell_atoms(basis, nuclei)
    kern = _ghj(tuple(cls))
    la, lb, lc, ld = cls
    for bra in _ordered_pairs(basis, la, lb).values():
        for ket in _ordered_pairs(basis, lc, ld).values():
            pb = pack([pair_record(basis, a, b, G=0.0) for a, b in bra])
            pk = pack([pair_record(basis, c, d, G=0.0) for c, d in ket])
            arrs = {
                "bra_prim": np.array(pb.prim),
                "bra_exp": np.array([(x, y) for a, b in bra
                                     for x in sh[a].exponents
                                     for y in sh[b].exponents]),
                "AB": np.array(pb.sep), "ket_prim": np.array(pk.prim),
                "CD": np.array(pk.sep),
                "bra_ao": np.array(pb.ao, dtype=np.int32),
                "ket_ao": np.array(pk.ao, dtype=np.int32),
                "bra_atom": np.array([atom[s] for s in pb.sh],
                                     dtype=np.int32)}
            grid, block = ghj_launch_shape(pb.n, threads)
            G = Interpreter(kern, launch=Launch(grid, block, threads),
                            boys=boys).run(
                n_bra=pb.n, n_ket=pk.n, k_ab=pb.degree, k_cd=pk.degree,
                n_ao=basis.n_ao, n_atom=len(nuclei), D=Dx, G=G,
                **{p.name: arrs[p.name] for p in kern.params
                   if p.name in arrs})["G"]
    return G


def ghj_gradient(basis: Basis, nuclei: Sequence[Nucleus], D: Matrix, *,
                 boys: Boys, threads: int = 4) -> np.ndarray:
    """`dE_J/dR` on the HGP recurrence: every ordered class's `ghj_` unit
    over every ordered quartet of it (module docstring)."""
    G = _zero(nuclei)
    ls = sorted({s.l for s in basis.shells})
    for cls in product(ls, repeat=4):
        G = ghj_class_gradient(basis, nuclei, D, cls, boys=boys,
                               threads=threads, G=G)
    return G


def gf_pair(basis: Basis, atoms: Sequence[int], a: int, b: int
             ) -> GF.GPairRec:
    """`kernel_gf`'s record of the ordered shell pair `(a, b)`: every
    primitive pair, `i` outer (`walk_f.pair_record`'s rows, no pruning),
    with its two exponents."""
    sa, sb = basis.shells[a], basis.shells[b]
    A, B = sa.centre, sb.centre
    AB = tuple(u - v for u, v in zip(A, B, strict=True))
    ab2 = sum(x * x for x in AB)
    prims = []
    for x, ca in zip(sa.exponents, sa.coefficients, strict=True):
        for y, cb in zip(sb.exponents, sb.coefficients, strict=True):
            p = x + y
            P = tuple((x * u + y * v) / p for u, v in zip(A, B, strict=True))
            U = ca * cb * (math.pi / p) ** 1.5 * math.exp(-x * y / p * ab2)
            prims.append(GF.GPrimRec(
                P, tuple(u - v for u, v in zip(P, A, strict=True)), p, U, x,
                y))
    return GF.GPairRec(AB, tuple(prims), (a, b),
                       (basis.offsets[a], basis.offsets[b]),
                       (atoms[a], atoms[b]))


def gf_launches(basis: Basis, nuclei: Sequence[Nucleus]
                ) -> list[tuple[tuple[int, int, int, int], int, GF.Packed,
                                GF.Packed]]:
    """The fused F gradient's launches, `(class, diag, bra, ket)` (module
    docstring): every unique quartet of `basis` exactly once."""
    sh = basis.shells
    atoms = host.shell_atoms(basis, nuclei)
    batches: dict[tuple[int, int, int], list[GF.GPairRec]] = {}
    for i in range(len(sh)):
        for j in range(i, len(sh)):
            a, b = (i, j) if (sh[i].l, i) <= (sh[j].l, j) else (j, i)
            key = (sh[a].l, sh[b].l,
                   len(sh[a].exponents) * len(sh[b].exponents))
            batches.setdefault(key, []).append(gf_pair(basis, atoms, a, b))
    keys = sorted(batches)
    packed = [GF.pack(batches[k]) for k in keys]
    return [((*keys[i][:2], *keys[j][:2]), int(i == j), packed[i], packed[j])
            for i in range(len(keys)) for j in range(i, len(keys))]


def gf_run(am: tuple[int, int, int, int], diag: int, bra: GF.Packed,
           ket: GF.Packed, n_atom: int, D: np.ndarray, G: np.ndarray, *,
           boys: Boys, threads: int = 4,
           kernel: Kernel | None = None) -> np.ndarray:
    """One `gf_` launch of class `am` over packed sides `bra` and `ket`,
    `G` (`n_atom x 3`) carried in and returned; `kernel` replaces the
    class's built one (a test's mutation)."""
    kern = _gf(tuple(am)) if kernel is None else kernel
    arrs = {"n_bra": bra.n, "n_ket": ket.n, "k_ab": bra.degree,
            "k_cd": ket.degree, "n_atom": n_atom, "n_ao": D.shape[0],
            "diag": diag, "bra_prim": bra.prim, "AB": bra.sep,
            "bra_sh": bra.sh, "bra_ao": bra.ao, "bra_at": bra.at,
            "ket_prim": ket.prim, "CD": ket.sep, "ket_sh": ket.sh,
            "ket_ao": ket.ao, "ket_at": ket.at, "D": D, "G": G}
    args = {}
    for p in kern.params:
        v = arrs[p.name]
        if p.array and not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.int32 if p.type is Scalar.I32
                           else np.float64)
        args[p.name] = v
    grid, block = GF.launch_shape(bra.n, threads)
    return Interpreter(kern, launch=Launch(grid, block, 1),
                       boys=boys).run(**args)["G"]


def gf_gradient(basis: Basis, nuclei: Sequence[Nucleus], D: Matrix, *,
                boys: Boys, threads: int = 4) -> np.ndarray:
    """`dE_J/dR - 1/2 dE_K/dR` from the fused `gf_` kernels, every launch
    of `gf_launches` into one zeroed `G`."""
    G = _zero(nuclei)
    Dx = np.asarray(D, dtype=np.float64)
    if Dx.shape != (basis.n_ao, basis.n_ao) or not np.array_equal(Dx, Dx.T):
        raise ValueError(f"D is not a symmetric {basis.n_ao} x {basis.n_ao} "
                         "matrix (the derived weights assume D_ab = D_ba)")
    for am, diag, pb, pk in gf_launches(basis, nuclei):
        G = gf_run(am, diag, pb, pk, len(nuclei), Dx, G, boys=boys,
                   threads=threads)
    return G


def interpreted_runners(*, boys: Boys, lanes: int = 4,
                        coulomb: str = "mmd") -> Runners:
    """Every electronic term of `gradient.rhf` from the interpreted kernels
    (module docstring); `coulomb` picks the Coulomb term's engine, `"mmd"`
    (the J chain, `gjab_`) or `"hgp"` (`ghj_`)."""
    if coulomb not in ("mmd", "hgp"):
        raise ValueError(f"coulomb = {coulomb!r} is not 'mmd' or 'hgp'")
    jterm = gjab_gradient if coulomb == "mmd" else ghj_gradient
    return Runners(
        overlap=lambda b, n, M: g1e_gradient(Target.S, b, n, M, boys=boys),
        kinetic=lambda b, n, M: g1e_gradient(Target.T, b, n, M, boys=boys),
        potential=lambda b, n, M: g1e_gradient(Target.V, b, n, M, boys=boys),
        coulomb=lambda b, n, M: jterm(b, n, M, boys=boys),
        exchange=lambda b, n, M: gk_gradient(b, n, M, boys=boys,
                                             lanes=lanes))


__all__ = ["g1e_gradient", "gf_gradient", "gf_launches", "gf_pair", "gf_run",
           "ghj_class_gradient", "ghj_gradient",
           "gjab_gradient", "gk_gradient", "interpreted_runners"]
