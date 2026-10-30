# Analytic nuclear Hessians by McMurchie–Davidson: `E_J`, `E_K`, `S`, `T`, `V`, and the DF `(P|Q)` and `(mn|P)`

The Coulomb energy's Hessian is this note's first part. Three later
sections each have their own status line: the one-electron Hessians of
`S`, `T` and `V` ("The one-electron Hessians"), the DF metric's ("The DF
metric Hessian") and the three-center DF integral's ("The three-center DF
Hessian").

**Status: implemented host-side and as generated kernels, compiled by
nvcc and hipcc and run on both cards** (the RTX 3080 and the RX 9060 XT,
`hdq` and `hjab` per class through (dd) and the whole H₂O/STO-3G `H`,
against the host, with the host's `J`s; the on-card chain through the
`[p|q]` kernel is interpreted only). This is
the derived note for the first `derivative_order = 2` key: the skeleton
Coulomb part of an SCF Hessian, `d2 E_J / dR dR` for a fixed density. Its
exchange twin `d2 E_K / dR dR` (s and p) is interpreted only and has its
own status line ("The exchange Hessian"). It
says what is generated, what it is checked against, and what is not done.
The session that built it had no device compiler and no card (the kernel
texts passed clang 18's CUDA front end, `-Wall -Wextra`, through (dd), a
proxy); a later session (2026-10-27) wrote the device test and ran it.

**Nothing here is scored.** No published table covers it; the rule is
McMurchie & Davidson's (`docs/references.md` entry 46) applied twice, as
`experimental/mmd/hessian` (s and p, numpy, checked against libcint's
`int2e_ipip1` / `_ipvip1` / `_ip1ip2` and `pyscf.hessian.rhf`) applies it,
and the kernels are this repo's own design. No `claims.json` record, cell
or floor was added. The spike is not imported anywhere.

## The keys

`spec.DERIVATIVE_ORDERS` is `(0, 1, 2)`. Order 2 is legal on
`spec.HESSIAN_TARGETS` only — `JAB`, and since the one-electron Hessians
`S`, `T` and `V`, the exchange Hessian `K` and the DF `METRIC` and `ERI3`,
on MMD — and refused elsewhere with
`_HESSIAN_REFUSAL`. Order 1 gained `DQ` (`GRADIENT_TARGETS`), the ket
pre-contraction differentiated on each ket centre, which only the Hessian
reads. Every d0 and d1 key is byte-identical:

- `coulomb/d2/(la,lb)/b1/jab/literal` — `hjab_{la}{lb}`
  (`hessian/kernel_hjab.py`);
- `coulomb/d1/(lc,ld)/b1/dq/literal` — `hdq_{lc}{ld}`
  (`hessian/kernel_hdq.py`);
- `coulomb/d2/(la,lc|lb,ld)/b1/k/literal` — `hkx_{la}{lc}_{lb}{ld}`
  (`hessian/kernel_hkx.py`, "The exchange Hessian" below), K's Hermite
  pairing, bra first.

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
| compile, nvcc / hipcc (`test_emit_hess.py`) | #92's filter | `sm_86` and `gfx1200` | passed, no warning in generated text |
| on a card (`test_hess_j_device.py`, generator `test/hess_driver.py`, on the shared runner since #659, which ran on both cards on 2026-10-30) | `hdq`: `host.ket_derivative_records`, `1e-13` of `max\|D1\|`; `hjab`: `host.coulomb_hessian`, host `J`s, `1e-14` of the absolute twin | RTX 3080 and RX 9060 XT: `hdq` and `hjab` for every ordered class (ss)–(dd) on the three-atom s, p, d basis; the whole H₂O/STO-3G `H` (`slow`); 19 cases per card | `hdq` 4.98e-16 (`hdq_02`); `hjab` 7.16e-17 of the scale per class, 9.28e-17 whole H₂O; the same on both cards, per-class runs bit-identical |

## The exchange Hessian (`hkx`)

**Interpreted only, s and p only. Not compiled by nvcc or hipcc, not run
on either card.** The same skeleton for the exchange energy,

    H[A, k, B, l] = d2 E_K / dR_{A,k} dR_{B,l},   E_K = 1/2 SUM D_ab D_cd (ac|bd)

K's pairing (bra `(a, c)` on `A`, `C`; ket `(b, d)` on `B`, `D`), `D`
symmetric and fixed. The RHF exchange term is `-1/2 E_K`; the factor is
the caller's, as `gk`'s is. The renaming that halves `E_J` halves `E_K`
too, so over every ordered quartet

    d2 E_K / dx dy = SUM D_ab D_cd [d_bra^x d_bra^y + d_bra^x d_ket^y] (ac|bd)

**Why one kernel, not a chain.** `D_ab D_cd` straddles the bra and the
ket, so there is no ket derivative *density* to precompute as `hdq` does
for `J`: the bra–ket term needs the bra's and the ket's derivative inside
one (bra, ket) pair. `hkx_{la}{lc}_{lb}{ld}` (`Target.K` at d2) is
`kernel_gk`'s mapping — one warp per bra record, kets over the lanes, no
shared memory, no screen — running the plan of the d0 K class with the
bra raised by **two** (`hkx_pq_spec = k_class_spec(la + 2, lc, lb, ld)`;
every `[r]^(0)` either term reads has `|r| <= L + 2`, which is exactly
that plan's target set). Per (bra, ket) pair and `(b,d)` row:

- the ket's per-axis tables at `(lb + 1, ld + 1)` give `s_bd E^{bd}_q`
  and the six `s_bd (dY_l E^{bd})_q` (`Y = B, D`), from which
  `I[p] = SUM_q (-1)^|q| [p+q] s_bd E^{bd}_q` (`|p| <= lp + 2`) and
  `J{Y}{l}[p]` the same on the derivative (`|p| <= lp + 1`);
- the bra is `gk`'s raised-row scheme, composed: the rule `T_k(a) =
  2 alpha [a + 1_k] - a_k [a - 1_k]` applied twice is a short sum of
  SHIFTED rows (`dA_k dA_k` is `(2 alpha)^2 [a + 2_k] - (2 alpha)(2 a_k
  + 1) [a] + a_k (a_k - 1) [a - 2_k]`), generated by composition
  (`_derivative_terms`, pinned in the test). Each shifted row `(a', c')`
  is folded once, `V = SUM_p E^{a'c'}_p I[p]` (or `J{Y}{l}`), and
  digested into one accumulator per component and power of `2 alpha`,
  `2 gamma`, the powers multiplied in after the ket loop.

**The ket's atom varies across lanes**, so the bra–ket term cannot be
reduced once per bra record. The host groups each class's kets by their
atom pair (`hessian.host.exchange_hessian_records`: `seg_start`, `seg_b`,
`seg_d`, a stable regrouping of `gradient.host.exchange_records`' kets,
with the ket exponents added) and the kernel walks the segments serially
— a loop uniform across the warp — strip-mining each segment's kets, then
36 `WarpReduce`s and lane 0's 36 `AtomicAdd`s into `H[atom_X, k, seg_Y,
l]`. The bra–bra term is reduced once after every segment: 21
`WarpReduce`s and 36 `AtomicAdd`s (the `AC` block mirrored onto `[C l][A
k]`). The bra–ket term is never mirrored. One launch per class over every
ordered bra and ket record; the 16 s, p classes sum to the whole
`d2 E_K`. `H` is zeroed by the caller.

Host reference: `exchange_hessian(basis, D, nuclei, one_centre=,
quartets=, atoms=, parts=, absolute=)`, every ordered quartet or
`quartets=` (`gradient.host.class_quartets`), differentiating `E^{ac}`
and `E^{bd}` *before* the fold (`_Axes`), `reference.pq`'s `[r]`.

Generator: `python -m felizjueves.emit --hess-k-l [LMAX]` (1 bare,
accepted 0..1, exit 2 above) adds the 16 s, p units beside the J ladders;
never in the bare default nor in `KERNEL_GEN_CMD`. The text is large:
cuda 58 KB at `(ss|ss)`, 221 KB at `(ss|pp)`, 1.63 MB at `(pp|pp)`; the
ten units over 512 KiB (five per backend) are pinned by size and sha256
in `test/golden/hkx_digests.json`, the rest are goldens
(`test_emit_hess_k.py`).

| What | Against | Where | Worst |
|---|---|---|---|
| host `H` (`test_hessian_host.py`) | central differences, `h = 1e-4`, of `gradient.host.exchange_gradient` (itself held to FD of the libcint-witnessed `exchange_matrix`) | the two-atom s, p basis; H₂O/STO-3G `slow` | 5.7e-9 of `max\|H\|` on the s, p basis, gate 1e-7 |
| host symmetry, invariance, parts | `H = H^T`, `SUM_A H = 0`, `bb + bk = H` (each ~10x `max\|H\|`, cancelling), each part symmetric; class quartets sum to the whole; `atoms=` folds back | same | ~4e-15 of `max\|H\|`, gate 1e-13 |
| host, once, by hand (not a test) | `experimental/mmd/hessian`'s `two_electron2` (checked there against libcint's `int2e_ipip1` / `_ipvip1` / `_ip1ip2`) contracted with `D D / 2` | the two-atom s, p basis | 1.2e-14 of `max\|H\|` |
| `hkx` interpreted (`test_kernel_hkx.py`) | `host.exchange_hessian(quartets=)`, the absolute twin as scale | every s, p class on the two-atom basis (L + 2 > 4 `slow`) and on a three-atom basis with two primitives a shell (all but `(ss\|ss)` `slow`) | ~1 eps of the scale at `(pp\|pp)`, gate 16 eps |
| the whole `d2 E_K` from the 16 launches | the host sweep; symmetry, invariance | the two-atom basis (`slow`) | gate 16 eps; 1e-13 |
| compile, `-Wall -Wextra` | clang 18 CUDA front end, device side, runtime stub | all 16 cuda units | no diagnostic (a proxy) |
| compile, nvcc / hipcc (`test_emit_hess_k.py`) | #92's filter | — | **not run** (no compiler) |
| on a card | — | — | **not written, not run** |

Not done for `hkx`: d and above (the ladder refuses them; the unrolled
text would be far past (pp|pp)'s 1.6 MB, so a rolled form is the way
there), a device driver and test, any measurement, a screen, and the
cost of the segment loop (one serial pass per ket atom pair, `n_atom^2`
at most, with 36 reductions each) is unexamined.

## What is not done

- On a card, `hjab` takes the host's `Jp2` / `Jp1`: the chain `hdq` →
  gather → `[p|q]` → `hjab` is checked interpreted only. Nothing above
  (dd) has run on a card.
- The skeleton Coulomb and exchange terms and (below) the one-electron
  and DF `(P|Q)`, `(mn|P)` skeletons only. Not built: the exchange
  Hessian above p ("The exchange Hessian"), the nuclear repulsion as a
  kernel (it is host-side), the derivative Fock matrices `F^x` and the
  CPHF response, so no whole RHF Hessian is assembled or checked against `pyscf.hessian.rhf`.
- The bra–ket term runs the `[p|q]` kernel `3 n_atom` times per class
  pair, each over the kets touching one atom (about `6 n_ket` kets in all),
  and `hjab` adds `18 n_atom` atomics per bra record whether or not the
  atom's `Jp1` row is zero. A per-atom ket mask or a block mapping is a
  follow-up.
- Unrolled text only; the ladder stops at (dd) (~1 MB). A low-stack or
  rolled form (#586's) is not built. Nothing is measured, nothing is
  screened, Cartesian only.

## The one-electron Hessians (`S`, `T`, `V`)


**Status: implemented host-side and as generated kernels, compiled by
nvcc and hipcc and run on both cards** (H₂O/STO-3G and cc-pVDZ, classes
through (dd), against the host; the table below) -- in the unrolled
`d2v`; the low-stack `d2v`, the default from `la + lb = 4`, is interpreted
only, not compiled by a vendor compiler nor run on a card. This is
the derived note for the one-electron `derivative_order = 2` keys: the contracted
second nuclear derivatives of the overlap, the kinetic energy and the
nuclear attraction (its operator's centre–nucleus and nucleus–nucleus
terms included) -- the one-electron skeleton of an SCF hessian. It says
what is generated, what it is checked against, and what is not done. The
session that built it had no device compiler and no card (every ordered
class through (ff) passed clang 18's CUDA front end, `-fsyntax-only -Wall
-Wextra`, a proxy); a later session (2026-10-27) compiled and ran them.

**Nothing here is scored.** No published table covers these hessians; the
rule is McMurchie & Davidson's (`docs/references.md` entry 46) applied
twice, and the kernels are this repo's own design. No `claims.json`
record, cell or floor was added.

The spike this ports is `experimental/mmd/hessian/` (s and p, numpy),
which checked the same rule against libcint's `int1e_ipip*` / `_ip*ip`
and `pyscf.hessian.rhf` to ~1e-11 of max. That spike is not imported
anywhere; its two-electron second derivatives and its CPHF response are
not ported here (the skeleton Coulomb term is this note's first part).

### Their keys

`spec.DERIVATIVE_ORDERS` is `(0, 1, 2)`. Order 2 is legal on
`spec.HESSIAN_TARGETS` -- `S`, `T` and `V` beside `JAB`, `K`, `METRIC`
and `ERI3`, each on MMD -- and every
other target, and HGP, refuses it with `_HESSIAN_REFUSAL`. A hessian
class is its integral's class, so `overlap/d2/(2,1)/b1/s/literal`
differs from the d0 and d1 keys in the `d` segment alone, and every
existing key is byte-identical. Unbatched, for each target's d0 reason.

### What a d2 kernel writes

As at d1, not the derivative integrals (a `[n_atom, 3, n_atom, 3, n_ao,
n_ao]` tensor) but their contraction with a matrix:

    H[A, k, B, l] += SUM_mn M_mn d2X_mn / dR_{A,k} dR_{B,l}

`M = D` for `T` and `V`, `W` for `S`; the signs are the caller's. The
per-Cartesian-pair weights are `gradient.host.pair_weights`'
(`M[a, b] + M[b, a]` off the diagonal, `M[a, a']` on it), exact for any
`M`. `H` is zeroed by the caller; every kernel only adds.

### The rule

The gradient's rule (`gradient.md`) applied twice. On one axis factor
`f_ij` (`E^{ij}_t`, `S_ij`, or the kinetic `K_ij`):

    dA f_ij    = 2 alpha f_{i+1,j} - i f_{i-1,j}
    dA dA f_ij = 4 alpha^2 f_{i+2,j} - 2 alpha (2i + 1) f_ij + i (i - 1) f_{i-2,j}
    dA dB f_ij = dA (2 beta f_{i,j+1} - j f_{i,j-1})

and a component on two different axes is the product of the two first
derivatives. So the `E` tables are raised two above the class on each
side (`T`'s ket four, for the Laplacian), and `V`'s `[t]^(0)` two Hermite
orders above, from the same primitive pair. `V`'s operator moves with its
nucleus, `d/dC_k [t]^(0)(P - C) = -[t + 1_k]^(0)`:

    centre-centre    -Z SUM_t H^{XY,kl}_t [t]^(0)
    centre-nucleus   +Z SUM_t H^{X_k}_t   [t + 1_l]^(0)   at (X, k, C, l) and (C, l, X, k)
    nucleus-nucleus  -Z SUM_t H^0_t       [t + 1_k + 1_l]^(0)

No translational invariance is used: every ordered pair of centres and
nuclei is computed, so `SUM_A H[A, k, B, l] = 0` is a check.

### The host reference (`hessian/host_1e.py`)

`overlap_hessian`, `kinetic_hessian` and `potential_hessian(one_centre=)`
(`E_nuc`'s is `hessian.host`'s), pure Python on the AO layer, over
`gradient.host`'s sweep (shell pairs `i <= j`, `primitive_pairs` order,
`shell_atoms`, `pair_weights`). **By another route than the kernels**:
`experimental/mmd/hessian/ints2.py`'s, each 3-D Cartesian differentiated
as a list of `(coefficient, moved Cartesian)` (`d_centre`, `d2_centre`)
and every term an ordinary 3-D integral of moved Cartesians, `T` by the
3-D Laplacian on the ket. `absolute=True` takes every factor in `|.|`:
the round-off scale the kernel tests use.

### The kernels (`hessian/kernel_h1e.py`)

`build_h1e_kernel(spec | (target, la, lb))` → `d2s_`, `d2t_`,
`d2v_{la}{lb}` (not `hs_`: the HGP one-electron ladder has those names).
`kernel_g1e`'s record head, weights, table reader and launch are reused
unchanged, and so is its parameter block, with the one OUT
`H[n_atom, 3, n_atom, 3]`. One thread per primitive-pair record. Each
component `d/dX_k d/dY_l` is an operator per axis (`axis_ops`): `X` on
axis `k`, `Y` on axis `l`, both on one axis when `k == l`; every operated
factor is a pooled local declared where first read. 21 components per
record (`AA` and `BB` with `k <= l`, `AB` all nine), scattered with 36
`AtomicAdd`s (the symmetric blocks mirrored, `AB` written at `(A, k, B,
l)` and `(B, l, A, k)`).

- `d2s`: `PROD_d (op_d E)^{a_d b_d}_0`.
- `d2t`: the kinetic bracket `SUM_m PROD_d op_d (K on m, S elsewhere)`.
- `d2v`: the Hermite densities `H^{ops}_t = SUM_r w_r PROD_d (op_d E)_t`
  for every operator triple a component, a first derivative or nothing
  needs, once per record; then per nucleus, serially (`kernel_v`'s
  mapping), the plan's `[t]^(0)` at Hermite order `L + 2`
  (`kernel_v.v_plan((la, lb + 2))`, `recurrence_body` unchanged), the
  centre-centre accumulators and 45 nucleus atomics (36 centre-nucleus,
  9 nucleus-nucleus).

Text size (cuda, unrolled): `d2s` 16 KB at (ss) to 211 KB at (ff),
`d2t` 32 KB to 456 KB, `d2v` 35 KB, 190 KB at (pp), 440 KB at (pd),
1.08 MB at (dd) and 4.9 MB at (ff).

### The low-stack `d2v`

From `la + lb = 4` (`h1e_lowstack`: `V`'s Hermite order `la + lb + 2 >=
kernel_v.LOWSTACK_MIN_LSUM = 6`, the order at which `V`'s own low-stack form
takes over), `build_h1e_kernel` emits `d2v` in a looped form by default,
with the same name, parameters, launch and `H` contract;
`lowstack=True` / `False` forces either form of `V` at any class, and `S`
and `T` have one form (they refuse `True`). It forms no Hermite density.
Instead:

- the weights go into a local `W[n_rows]` by loops (the same rule);
- per nucleus, after the unchanged prelude and `recurrence_body`, the
  `[t]^(0)` go into a local `R` in `hermite_targets(L + 2)` order;
- per block row (`e_fragment.cartesian_loops`), each axis's operated
  factors `F{op}[d, t] = (op E)^{a_d b_d}_t` (`op` in `"" A B AA AB BB`)
  are filled by short loops, `operated`'s rule term for term (its
  `- n f` term behind an `If` on `0 < n` and its source's reach), so each
  entry is the unrolled pooled local's value bitwise;
- the 28 operator triples (21 components, 6 first derivatives, the plain
  density) are contracted as rolled `t` loops against `R[t + up]`, every
  shift a triple needs read in one pass, into 45 accumulators;
- then the same `hXY_kl -= Z_C q`, the same 45 nucleus atomics and the
  same 36-atomic scatter.

The sums are regrouped by row instead of by Hermite index, so the two
forms agree to rounding. Text (cuda): 113 KB at (pf), 115 KB at
(dd), 135 KB at (df), **161 KB at (ff)**, so every default hessian unit
through (ff), in both orientations and for both vendors, is under the
512 KiB file cap (`test_emit_hess_1e.py` asserts it). Below the rule the
unrolled text is unchanged byte for byte. The unrolled `d2v_22` is pinned
by its old digest, under the key `d2v_22.unrolled`.

The frame, read from clang 18's PTX (`-O3`, `__local_depot`; no ptxas
here, so not nvcc's figure): 7.4 KB at (dd) and 12.9 KB at (ff), of which
the `(la + 2, lb + 2)` axis tables, shared with `kernel_g1e`'s record
head, are 5.4 and 9.5 KB. The unrolled form shows no depot in PTX,
because its spills happen in ptxas. It keeps 1,760 densities (14 KB) live
across the nucleus loop at (dd) and 3,780 (30 KB) at (ff), and every
pooled factor besides. No `sm_86` or `gfx1200` frame, spill or timing is
measured for either form.

### The generator

`python -m felizjueves.emit --hess-one-e-l [LMAX]` (2 bare, 0..3, a 2 above)
adds `d2s_`, `d2t_`, `d2v_{la}{lb}` for every `la <= lb <= LMAX` beside
the J ladders; never in the no-flag output and not in `KERNEL_GEN_CMD`.
The goldens are the six classes (ss) through (dd) per backend
(`test/golden/{cuda,hip}/d2{s,t,v}_*`), plus `d2v_33` (the low-stack form
at (ff)); the unrolled `d2v_22`, over the 512 KiB file cap, is pinned by
byte count and sha256 in `test/golden/hessian_digests.json`.

### Witnesses, and what ran where

| What | Against | Where | Worst |
|---|---|---|---|
| `hessian.host_1e`, `S` `T` `V` | central FD (`h = 1e-4`) of `gradient.host`, itself FD-held to the libcint-witnessed integrals | H₂O/STO-3G, a two-atom s/p basis; H₂O/cc-pVDZ `slow` (`test_hessian_host_1e.py`) | ~3e-9 – 8e-9 of `max\|H\|` (`O(h²)`), gate 1e-7 |
| the same | `SUM_A H = 0`; `H = H^T` | the same | invariance 4.5e-12 of `max\|H\|` (`T`, the O 1s cancellation; gate 1e-10); symmetry ~5e-16 (gate 1e-13) |
| `kernel_h1e`, interpreted | `hessian.host_1e` per class, `M` non-symmetric | every ordered class (ss)–(ff) on HF/cc-pVTZ (above L + 2 = 4 `slow`); whole H₂O/STO-3G (`test_kernel_h1e.py`) | 0.35 eps of the absolute twin (`d2v_12`), gate 16 eps |
| the low-stack `d2v`, interpreted | the same, every class also in the form its default does not pick | every ordered class (ss)–(ff) on HF/cc-pVTZ; whole H₂O/STO-3G (`test_kernel_h1e.py`) | 0.27 eps of the absolute twin (`d2v_12` forced), gate 16 eps |
| the units | clang 18 CUDA front end, `-Wall -Wextra` | all 48 ordered classes through (ff), here; the low-stack `d2v_22` and `d2v_33` | no diagnostic (a proxy) |
| nvcc / hipcc compile | #92's warning filter | `test_emit_hess_1e.py`'s `cuda` / `hip` cases, `sm_86` and `gfx1200` | passed, no warning in generated text |
| on a card | `hessian.host_1e`, within 1e-14 of the absolute twin | `test_hess_device.py` (generator `test/hess_driver.py`, on the shared runner since #659, which ran on both cards on 2026-10-30), H₂O/STO-3G, cc-pVDZ `slow`; RTX 3080 and RX 9060 XT, 6 cases per card, the UNROLLED `d2v` (before the low-stack form) | 1.57e-16 (CUDA), 2.30e-16 (HIP) of the scale, both `d2v` on cc-pVDZ |
| the low-stack `d2v` on a card | the same | `test_hess_device.py` | **not run** (no card or vendor compiler in that session) |

### What is not done

- The two-electron second derivatives are this note's first part and
  "The exchange Hessian" (s and p only); no CPHF response, no assembled
  RHF hessian; `experimental/mmd/hessian` has the
  rest for s and p, in numpy.
- Nothing above (dd) has run on a card; nothing measured.
- The low-stack `d2v` is not measured: no frame, spill or clock on
  either card, and nothing says which form is faster. Its frame is mostly
  the shared `(la + 2, lb + 2)` axis tables, which keep slots no read
  reaches (`t > i + j`, the doubly raised corner); packing them is a
  follow-up. `S` and `T` stay unrolled (`d2t_33` is 456 KB).
- No screening, Cartesian only.

## The DF metric Hessian (`(P|Q)`)

**Status: implemented host-side and as a generated kernel, interpreted
only.** Not compiled by nvcc or hipcc (none in the session that built
it; all 9 units passed clang 18's CUDA front end, device side, `-Wall
-Wextra`, a proxy), no device test, not run on a card, nothing measured.
Nothing scored: no published board, no claim added.

### The key

`Target.METRIC` joins `spec.HESSIAN_TARGETS` (MMD only: `metric.hgp` at
d2 is refused). `am` is the aux pair `(lP, lQ)`, so
``coulomb/d2/(1,2)/b1/metric/literal`` differs from its d0 and d1 keys in
the `d` segment alone; every d0 and d1 key is byte-identical.

### What is computed

`H[A, k, B, l] += SUM_PQ W_PQ d2(P|Q) / dR_{A,k} dR_{B,l}` for any
`n_aux x n_aux` matrix `W`. A DF-J energy Hessian's metric term passes
`W = -1/2 d d^T` (the gradient's `W`, `gradient/host_df.py`), skeleton
only: `d`'s response and the assembled DF-J Hessian are not built (the
three-center `(mn|P)` Hessian is the next section).

### The rule: the Hermite shift twice

A single-center `E^a_t` does not depend on its centre, so moving `P`
moves only its Hermite Gaussians, and `[r]^(0)` reads `A - B`:

    d2(P|Q)/dA_k dA_l = SUM_t E^P_t SUM_tau (-1)^{|tau|} [t + tau + 1_k + 1_l]^(0) E^Q_tau =: h_kl
    d2(P|Q)/dA_k dB_l = -h_kl,      d2(P|Q)/dB_k dB_l = +h_kl

`kernel_gmetric`'s contraction with the tower read two indices up: no
raised `E`, and `h` symmetric, so six contractions per pair. It is exact,
not translational invariance assumed (`SUM_A H = 0` by construction).

### The host reference (`hessian/host_df.py`)

`metric_hessian(aux, W, nuclei, *, pq, pairs=, atoms=, absolute=)` takes
**another route**, `host_1e`'s: each aux function differentiated as a list
of moved Cartesians (`d_centre`, `d2_centre`), each term the ordinary
`(u|v)` on the single-center `E` two indices above the class, `[t|tau]`
injected. `absolute=True` is its round-off twin.

### The kernel (`hessian/kernel_d2metric.py`)

`build_d2metric_kernel(cls)`, `d2metric_{lP}{lQ}` (not `hmetric_`: that is
the HGP metric's, `eri/density_fit/kernel_hgp.py`). `kernel_gmetric`'s
mapping and parameter block, `H[n_atom, 3, n_atom, 3]` in place of `G`:
one thread per aux shell pair, both record ranges serially, the weighted
bra density once per bra record, the tower the plan of the `Target.J`
quartet `(lP + 2, 0 | lQ, 0)` at `L + 2` cut to the live
`t + tau + 1_k + 1_l` (`kernel_gmetric.live_body`), and once per pair 36
`AtomicAdd`s (`+h` on the two same-atom blocks, `-h` on the two mixed
ones). `H` is zeroed by the caller.

### The generator

`python -m felizjueves.emit --hess-metric-l [LMAX]` (4 bare, 0..4, a 2
above) adds `d2metric_{lP}{lQ}` for every `lP <= lQ`, `lP + lQ <= LMAX`
beside the J ladders; never in the no-flag output and not in
`KERNEL_GEN_CMD`. The 9 goldens per backend are
`test/golden/{cuda,hip}/d2metric_*` (largest ~36 KB).

### Witnesses, and what ran where

| What | Against | Where | Worst |
|---|---|---|---|
| `hessian.host_df` | central FD (`h = 1e-4`) of `gradient.host_df.metric_gradient`, itself FD-held to the libcint-witnessed `(P|Q)` | a synthetic s/p aux basis on three centres; s/p/d and f/g bases `slow` (`test_hessian_host_df.py`) | ~2.5e-9 of `max\|H\|` (`O(h²)`), gate 1e-7 |
| the same | `SUM_A H = 0`; `H = H^T`; split atoms folded back | the same | gates 1e-12 and 1e-13 of `max\|H\|` |
| `kernel_d2metric`, interpreted | `hessian.host_df` per class, `W` non-symmetric | every class `lP + lQ <= 4` on `test_kernel_gmetric.py`'s aux basis (above `lP + lQ + 2 = 4` `slow`); a whole s/p aux basis (`test_kernel_d2metric.py`) | 0.17 eps of the absolute twin (`d2metric_02`), gate 16 eps |
| the units | clang 18 CUDA front end, `-Wall -Wextra` | all 9 | no diagnostic (a proxy) |
| nvcc / hipcc compile | #92's warning filter | `test_emit_d2metric.py`'s `cuda` / `hip` cases | **not run** (no compiler in that session) |
| on a card | — | — | **not run** |

### What is not done

- No device test and no card run; no nvcc / hipcc compile.
- No `d` response, no assembled DF-J (or DF-K) Hessian (the three-center
  `(mn|P)` Hessian is the next section).
- Nothing above `lP + lQ = 4` against the host; nothing against libcint
  directly (the chain is FD → gradient host → libcint's `(P|Q)`).
- Nothing measured; Cartesian only.

## The three-center DF Hessian (`(mn|P)`)

**Status: implemented host-side and as generated kernels, interpreted
only.** Not compiled by nvcc or hipcc and not run on a card: the session
that built it had neither compiler nor card. Every unit through L = 4
passed clang 18's CUDA front end with `-fsyntax-only -Wall -Wextra`,
device side, with a runtime stub. That is a proxy, not the tier. This
section covers the first density-fitting `derivative_order = 2` key: the
contracted second nuclear derivatives of the three-center integral
`(mn|P)`, which is the three-center half of a DF-J skeleton Hessian.

**Nothing here is scored.** No published table covers it. The rule is
McMurchie & Davidson's (`docs/references.md` entry 46) applied twice, as
in the one-electron Hessians, and the kernel is this repo's own design.
No `claims.json` record, cell or floor was added.

### Its key

`Target.ERI3` joins `spec.HESSIAN_TARGETS` (MMD only; HGP refuses it with
`_HESSIAN_REFUSAL`). The
key is the d0 triple at `d2`, `coulomb/d2/(la,lb|lP)/b1/eri3/literal`, so
it differs from the d0 and d1 keys in the `d` segment alone. Every
existing key is byte-identical. Unbatched, for the d0 reason.

### What it writes

    H[X, k, Y, l] += SUM_mnP Gam_mnP d2(mn|P) / dR_{X,k} dR_{Y,l}

over the integral's three centres, `X, Y` in `{A, B, C}`. `Gam` is any
`[n_ao, n_ao, n_aux]` tensor (`eri3`'s layout), and it need not be
symmetric in `(m, n)`. The weights are `kernel_geri3`'s, applied per aux
function. A DF-J skeleton Hessian passes `Gam = D (x) d`, beside the
metric's `-1/2 d d^T` term (`kernel_d2metric`, the previous section). `H` is zeroed by the
caller, and the kernel only adds.

### The rule, on three centres

- **`A` and `B`**: the Cartesian rule, applied once or twice per axis,
  on the per-axis `E` two indices above the class. These are
  `kernel_h1e`'s `axis_ops` and `operated`, unchanged.
- **`C`**: the Hermite shift. The aux centre's single-center `E` does
  not depend on `C`, and `[r]^(0)` reads `P - C`, so
  `d/dC_l [r] = -[r + 1_l]` and `d2/dC_k dC_l [r] = +[r + 1_k + 1_l]`.

The 45 accumulators per cell (`AA`, `BB` and `CC` with `k <= l`; `AB`,
`AC` and `BC` all nine) are contractions of weighted Hermite densities
`H^{ops}_q,t`, one per operator triple and aux row, against the d0 inner
sum `u_t,q = SUM_tau [t + tau]^(0) Eket[q, tau]`:

- bra–bra: `u_t`;
- bra–aux: `-u_{t + 1_l}`;
- aux–aux: `u_{t + 1_k + 1_l}`.

Each carries the class sign `(-1)^{lP}`. The tower is the plan of the
`Target.J` quartet `(la + 2, lb | lP, 0)` at `L + 2`. Each cell ends with
81 `AtomicAdd`s into `H`, each value and its mirror.

### The host reference (`hessian/host_eri3.py`)

`eri3_hessian(basis, aux, Gam, nuclei, *, pq, cells=, atoms=, aux_atoms=,
absolute=)` takes `hessian.host_1e`'s route on every centre, the aux one
included. Each 3-D Cartesian is differentiated as a list of moved
Cartesians (`d_centre` / `d2_centre`), and each term is the ordinary
three-center integral `SUM E^{uv}_t [t|tau] E^x_tau` on `E` two up.

This is another route from the kernel's on every centre: moved 3-D
functions against per-axis operators, and the Cartesian rule on `C`
against the Hermite shift. `absolute=True` gives the round-off twin. The
sweep, the atoms and the injected `[t|tau]` are
`gradient.host_df.eri3_gradient`'s.

### The kernel (`hessian/kernel_d2eri3.py`)

`build_d2eri3_kernel(cls)` builds `d2eri3_{la}{lb}{lP}`. The name is not
`heri3_`, which is the HGP `(mn|P)`'s. It reuses `kernel_geri3`'s
mapping, which is one thread per cell, the bra records outer and the aux
records inner. It also reuses `kernel_geri3`'s parameter block (`H[n_atom,
3, n_atom, 3]` in place of `G`) and its weights, through `cell_weights`,
made public with its text byte-identical. The arrays are the d1 kernel's
(`test/df_grad_launch.run_d2eri3`).

The text is straight-line and grows fast with the class:

| Class (L) | cuda text |
|---|---|
| `(ss\|s)` (0) | 40 KB |
| `(pp\|s)` (2) | 206 KB |
| `(pp\|p)` (3) | 504 KB |
| `(pd\|p)` (4) | 1.25 MB |

No low-stack form is built.

### The generator

`python -m felizjueves.emit --hess-eri3-l [LMAX]` adds
`d2eri3_{la}{lb}{lP}` for every `la <= lb`, `la + lb + lP <= LMAX`,
beside the J ladders:

- LMAX is 3 when the flag is bare (13 units) and accepted from 0 to 4;
  above 4 the CLI exits 2.
- It is never in the no-flag output and not in `KERNEL_GEN_CMD`.

The goldens are the seven classes with L <= 2, kept as files per backend
(`test/golden/{cuda,hip}/d2eri3_*`). The other 15 classes through L = 4
are pinned by byte count and sha256 in `test/golden/d2eri3_digests.json`.

### Witnesses, and what ran where

| What | Against | Where | Worst |
|---|---|---|---|
| `hessian.host_eri3` | central FD (`h = 1e-4`) of `gradient.host_df.eri3_gradient`, itself FD-held to the libcint-witnessed `(mn\|P)` | a three-atom s, p AO / s, p, d aux basis; H₂O/STO-3G against the cc-pVTZ-JKFIT s and p shells `slow` (`test_hessian_host_eri3.py`) | 4.8e-9 of `max\|H\|` (`O(h²)`), gate 1e-7 |
| the same | `SUM_A H = 0`; `H = H^T` | the same | 4e-15 / 4e-16 of `max\|H\|` (gates 1e-12, 1e-13) |
| `kernel_d2eri3`, interpreted | `hessian.host_eri3` per class, `Gam` random and non-symmetric | every class `la <= lb`, `la + lb + lP <= 4` on `test_kernel_geri3.py`'s bases (L + 2 > 4 `slow`); whole H₂O/STO-3G against JKFIT s, p `slow` (`test_kernel_d2eri3.py`) | 0.40 eps of the absolute twin (`d2eri3_110`), gate 16 eps |
| the units | clang 18 CUDA front end, `-fsyntax-only -Wall -Wextra`, device side, runtime stub | all 22 classes through L = 4 | no diagnostic (a proxy) |
| nvcc / hipcc compile | #92's warning filter | `test_emit_d2eri3.py`'s `cuda` / `hip` cases | **not run** (no compiler in that session) |
| on a card | — | — | **not run**: no device test is written |

### What is not done

- No device test and no driver. The d1 `grad_driver` program is the
  pattern for one.
- Nothing has been compiled by a vendor compiler, run on a card or
  measured.
- No assembled DF-J Hessian. The DF-J Hessian is this term with
  `Gam = D (x) d`, plus the metric term (`d2metric_`) with
  `W = -1/2 d d^T`, plus the response of `d`. The last is not a skeleton
  term.
- No DF-K, no screen, nothing above L = 4. Cartesian only.
