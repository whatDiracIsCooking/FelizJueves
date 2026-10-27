# The Coulomb energy's analytic nuclear Hessian by McMurchie–Davidson

**Status: implemented host-side and as generated kernels, interpreted
only. Not compiled by nvcc or hipcc, and not run on either card.** This is
the derived note for the first `derivative_order = 2` key: the skeleton
Coulomb part of an SCF Hessian, `d2 E_J / dR dR` for a fixed density. It
says what is generated, what it is checked against, and what is not done.
The session that built it had no device compiler and no card; the kernel
texts passed clang 18's CUDA front end (device side, `-Wall -Wextra`, a
declaration stub for the runtime) with no diagnostic, for every class
through (dd) — a proxy, not the kernel tier.

**Nothing here is scored.** No published table covers it; the rule is
McMurchie & Davidson's (`docs/references.md` entry 46) applied twice, as
`experimental/mmd/hessian` (s and p, numpy, checked against libcint's
`int2e_ipip1` / `_ipvip1` / `_ip1ip2` and `pyscf.hessian.rhf`) applies it,
and the kernels are this repo's own design. No `claims.json` record, cell
or floor was added. The spike is not imported anywhere.

## The keys

`spec.DERIVATIVE_ORDERS` is `(0, 1, 2)`. Order 2 is legal on
`spec.HESSIAN_TARGETS` only — `JAB`, on MMD — and refused elsewhere with
`_HESSIAN_REFUSAL`. Order 1 gained `DQ` (`GRADIENT_TARGETS`), the ket
pre-contraction differentiated on each ket centre, which only the Hessian
reads. Every d0 and d1 key is byte-identical:

- `coulomb/d2/(la,lb)/b1/jab/literal` — `hjab_{la}{lb}`
  (`hessian/kernel_hjab.py`);
- `coulomb/d1/(lc,ld)/b1/dq/literal` — `hdq_{lc}{ld}`
  (`hessian/kernel_hdq.py`).

Unbatched, for each target's d0 reason.

## What is computed

    H[A, k, B, l] = d2 E_J / dR_{A,k} dR_{B,l},   E_J = 1/2 SUM D_ab D_cd (ab|cd)

for a symmetric `D` held fixed: no density response. The CPHF terms of an
SCF Hessian (`SUM D^y F^x`, the `J[D^y]` they need being the ordinary `J`
build on a response density) are the caller's.

With every ordered quartet summed, the ket–ket term equals the bra–bra
term and the ket–bra term equals the bra–ket term (rename `(ab) <-> (cd)`),
so the half of `E_J` cancels both twos:

    d2 E_J / dx dy = SUM D_ab D_cd [d_bra^x d_bra^y + d_bra^x d_ket^y] (ab|cd)

- **Bra–bra.** `SUM_ab D_ab SUM_t (dx dy E^{ab})_t J2_t`, `J2` the J
  chain's `J_p` of the bra record **two** Hermite orders up,
  `|t| <= la + lb + 2`. It is the d0 `[p|q]` kernel of
  `hjab_pq2_spec(la, lb, lc, ld) = (la + 2, lb | lc, ld)` over the J
  chain's own kets, so nothing in `[p|q]` or its plan is new. (`gjab` reads
  the same kernel one order up.)
- **Bra–ket.** `SUM_ab D_ab SUM_t (dx E^{ab})_t J1^{(C,l)}_t`, with
  `J1^{(C,l)}_t = SUM_q [t|q] D^{(C,l)}_q` over the **derivative kets** of
  atom `C`, axis `l`: a ket record whose first shell sits on `C` carries
  `D^C_{l,q} = s SUM_cd w_cd (dC_l E^{cd})_q`, and one whose second shell
  does carries the `dD_l` twin (`w_cd` the ket's weights, the factor 2 of an
  off-diagonal pair in them). It is the d0 `[p|q]` kernel of
  `hjab_pq1_spec(la, lb, lc, ld) = (la + 1, lb | lc + 1, ld)`, run once per
  atom and axis over the kets touching that atom. Each term is symmetric
  in `x <-> y` summed over every quartet; one bra class's share of the
  bra–ket term is not, so it is never mirrored.

The per-axis rule, twice:

    dA E^{ij}    = 2 alpha E^{i+1,j}    - i E^{i-1,j}
    dA dA E^{ij} = 2 alpha dA E^{i+1,j} - i dA E^{i-1,j}
    dA dB E^{ij} = 2 beta  dA E^{i,j+1} - j dA E^{i,j-1}
    dB dB E^{ij} = 2 beta  dB E^{i,j+1} - j dB E^{i,j-1}

on a twice-differentiated axis; on two different axes a product of two
first derivatives. No translational invariance is used: `H = H^T` and
`SUM_A H[A k, B l] = 0` are checks.

## The chain, as launches

1. `DQ` (d0) per ket class — the J chain's kets, unchanged.
2. `hdq_{lc}{ld}` per ket class: `D1[6, n_ket, n_q1]`, slab `l` the
   first shell's `dC_l` density and slab `3 + l` the second's `dD_l`,
   every element stored (no zeroing).
3. Host: gather, per atom `C` and axis `l`, the kets touching `C` with
   their slab of `C` (the J chain's sort is likewise the host's; a ket
   with both shells on `C` appears twice).
4. `[p|q]` (d0) at `hjab_pq2_spec` over step 1's kets → `Jp2[n_rows,
   n_p2]`, and at `hjab_pq1_spec` once per `(C, l)` over step 3's kets →
   `Jp1[n_atom, 3, n_rows, n_p1]` (zero where no ket touches `C`).
5. `hjab_{la}{lb}` per bra class, one thread per bra record: 36
   `AtomicAdd`s for the bra–bra term, then an atom loop (`gv`'s nucleus
   mapping) with 18 per atom for the bra–ket term, into `H[n_atom, 3,
   n_atom, 3]`, zeroed by the caller.

`hdq`'s parameters are the bra kernels' record head
(`zeta_ab XPA XPB s ea_bra eb_ket off_a off_b`, `M`, then `D1`); `hjab`'s
are `gjab`'s with `Jp2`, `Jp1` and `H` (`kernel_hjab`'s docstring). Both
reuse `gradient.kernel_g1e`'s record head, per-axis table readers, weights
and first-derivative Hermite densities; `hjab` adds `second_reader`, the
twice-applied rule as pooled locals, and the second-derivative densities
over `hermite_targets(L + 2)` (`AA`, `BB` for `k <= l`, `AB` for all nine).

## The host reference (`hessian/host.py`)

`coulomb_hessian(basis, D, nuclei, one_centre=, pairs=, atoms=, kets=,
dkets=, parts=, absolute=)` (`parts` "bb" / "bk" for one term;
`absolute=True` the same sums with every factor in `|.|`),
`coulomb_dkets` (the derivative kets per `(C, l)`),
`ket_derivative_records` (`hdq`'s output for one class) and
`nuclear_repulsion_hessian`. It reuses `gradient.host`'s pair weights,
kets, `J_p` (`coulomb_jp`, which reads any ket by duck type) and
`shell_atoms`.

## The generator

`python -m felizjueves.emit --hess-l [LMAX]` (1 bare, accepted 0..2, exit 2
above) adds `hdq_{la}{lb}` and `hjab_{la}{lb}` for every `la <= lb <=
LMAX`, beside the J ladders; never in the bare default nor in
`KERNEL_GEN_CMD`. The two `[p|q]` classes are written by the class
selector, not by this flag. Goldens: `test/golden/{cuda,hip}/{hdq,hjab}_`
for (ss), (sp), (pp), 12 files (`test_emit_hess.py`). The unrolled `hjab`
grows fast (cuda): 29 KB at (ss), 188 KB at (pp), 1.08 MB at (dd), 4.8 MB
at (ff); `hdq` is about a fifth of that.

## Witnesses, and what ran where

| What | Against | Where | Worst |
|---|---|---|---|
| host `H` (`test_hessian_host.py`) | central differences, `h = 1e-4`, of `gradient.host.coulomb_gradient` (itself held to FD of the libcint-witnessed `coulomb_matrix`) | the two-atom s, p basis; H₂O/STO-3G `slow` | 5.9e-9 of `max\|H\|` on the s, p basis (the `O(h²)` floor), gate 1e-7 |
| host symmetry, invariance, parts | `H = H^T`, `SUM_A H = 0`, `bb + bk = H`, each part symmetric alone | same | ~2e-15 of `max\|H\|`, gate 1e-13 |
| derivative kets | `SUM` of their `q = 0` entries is the overlap gradient at `M = D` | s, p basis | gate 1e-14 |
| `E_nuc` | FD of `nuclear_repulsion_gradient` | three nuclei | gate 1e-7 |
| `hdq` interpreted (`test_kernel_hessian.py`) | `host.ket_derivative_records` | every ordered class (ss)–(dd) on a three-atom s, p, d basis | **bitwise** |
| `hjab` interpreted | `host.coulomb_hessian(pairs=)`, host `J`s, the absolute twin as scale | the same classes and basis ((pd), (dp), (dd) `slow`) | worst 0.14 eps of the scale ((pp)), 0.08 at (dd); gate 16 eps |
| the chain: `hdq` → gather → `[p|q]` at both specs (interpreted) → `hjab` | `host.coulomb_hessian` | the s, p basis; H₂O/STO-3G `slow` | 0.14 eps of the scale on the s, p basis, gate 64 eps; symmetry and invariance 1e-13 |
| compile, `-Wall -Wextra` | clang 18 CUDA front end, device side, runtime stub | `hdq` and `hjab` for every class through (dd), (10), (21) | no diagnostic (a proxy) |
| compile, nvcc / hipcc (`test_emit_hess.py`) | #92's filter | — | **not run** (no compiler) |
| on a card | — | — | **not written, not run** |

## What is not done

- No compile by nvcc or hipcc, no card run, and no device driver or device
  test (unlike the gradient, whose device test is written).
- The skeleton Coulomb term only. Not built: the one-electron second
  derivatives (`S`, `T`, `V` with its nucleus terms), the exchange
  Hessian, the nuclear repulsion as a kernel (it is host-side), the
  derivative Fock matrices `F^x` and the CPHF response, so no whole RHF
  Hessian is assembled or checked against `pyscf.hessian.rhf`.
- The bra–ket term runs the `[p|q]` kernel `3 n_atom` times per class
  pair, each over the kets touching one atom (about `6 n_ket` kets in all),
  and `hjab` adds `18 n_atom` atomics per bra record whether or not the
  atom's `Jp1` row is zero. A per-atom ket mask or a block mapping is a
  follow-up.
- Unrolled text only; the ladder stops at (dd) (~1 MB). A low-stack or
  rolled form (#586's) is not built. Nothing is measured, nothing is
  screened, Cartesian only.
