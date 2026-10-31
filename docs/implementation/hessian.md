# Analytic nuclear Hessians by McMurchie–Davidson: `E_J`, `E_K`, `S`, `T`, `V`, and the DF `(P|Q)` and `(mn|P)`

The Coulomb energy's Hessian is this note's first part. Five later
sections each have their own status line: the one-electron Hessians of
`S`, `T` and `V` ("The one-electron Hessians"), the DF metric's ("The DF
metric Hessian"), the three-center DF integral's ("The three-center DF
Hessian"), and the CPHF right-hand side's exchange and Coulomb derivative
matrices ("The exchange derivative matrix", "The Coulomb derivative
matrix").

**Status: implemented host-side and as generated kernels through (ff),
compiled by nvcc and hipcc and run on both cards** (the RTX 3080 and the
RX 9060 XT): `hdq` and `hjab` per class through (ff), in both forms where
each compiles, against the host with the host's `J`s, and since #690 the
WHOLE CHAIN on the card -- `DQ` and `hdq`, the host's gather, the `[p|q]`
kernels at both raised classes to L = 14, `hjab` -- against libcint on
H₂O/STO-3G and cc-pVDZ and against the host on a two-atom s, f basis.
From `la + lb = 3` `hjab`, and from `lc + ld = 5` `hdq`, is a ROLLED form
("The rolled `hjab` and `hdq`", below). This is
the derived note for the first `derivative_order = 2` key: the skeleton
Coulomb part of an SCF Hessian, `d2 E_J / dR dR` for a fixed density. Its
exchange twin `d2 E_K / dR dR` runs on both cards through (ff|ff) (#691)
and has its own status line ("The exchange Hessian"). It
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

`python -m felizjueves.emit --hess-l [LMAX]` (1 bare, accepted 0..3 since
#690, exit 2 above) adds `hdq_{la}{lb}` and `hjab_{la}{lb}` for every `la
<= lb <= LMAX`, each in its default form, beside the J ladders; never in
the bare default nor in `KERNEL_GEN_CMD`. The two `[p|q]` classes are
written by the class selector, not by this flag. Goldens:
`test/golden/{cuda,hip}/{hdq,hjab}_` for (ss), (sp), (pp), 12 files, below
both floors and unchanged by #690; every other ladder unit through (ff),
and the unrolled `hjab_12` / `hjab_22` that ran on both cards before #690
(`<name>.unrolled`), are pinned by size and sha256 in
`test/golden/hjab_digests.json` (`test_emit_hess.py`). The unrolled `hjab`
grows fast (cuda): 29 KB at (ss), 188 KB at (pp), 1.08 MB at (dd), 4.8 MB
at (ff); `hdq` is about a fifth of that.

## The rolled `hjab` and `hdq` (#690)

The unrolled densities are straight-line text, every Hermite index times
every row: over #613's 14,000 B of `sm_86` stack from `hjab_12` (22,056 B;
`hjab_22` 50,368 B and 186 s of nvcc alone) and from `hdq_23` (20,096 B;
`hdq_33` 49,040 B and ~114 s). So each builder takes `lowstack=` and
defaults to a ROLLED form from its floor, `kernel_hjab.LOWSTACK_MIN_LSUM =
3` and `kernel_hdq.LOWSTACK_MIN_LSUM = 5`, the smallest `la + lb` with an
unrolled class over the budget in `data/measured/hjab.json` (the store's
check holds both to its rows). Same name, parameters, launch and output
contract; below the floor the text is byte-identical.

- `hdq`, rolled: a local `HD[6 n_q1]`, zeroed; per block row
  (`kernel_g1e.lowstack_rows`, the row's weight from `M`), the operated
  per-axis factors `F{op}[d, t]` for `op` in `""`, `A`, `B`
  (`kernel_h1e.operated_value`, bitwise the unrolled pooled locals), and
  the six first-derivative densities `HD[3 side + k, t] += wt ((F0 F1)
  F2)` as rolled loops over the reachable `t`; then `D1 = s HD`, every
  element stored. Rows in `block_rows` order, the product associated as
  the unrolled one: the SAME sums in the same order, so the rolled `hdq`
  is bitwise the unrolled one and the host.
- `hjab`, rolled: the same `HD` (the bra-ket densities, bitwise), 21
  accumulators for the bra-bra components, each `+= SUM_t wt PROD_d
  F{o_d}[d, t_d] Jp2[row, t]` rolled per row over the six operated factors
  (`"" A B AA AB BB`, the tables `(la + 2, lb + 2)`), `Jp2` read where it
  lies; the 36 bra-bra atomics; then the atom loop with `l` and `k` at run
  time, `H[X, k, C, l] += s SUM_n HD[k, n] Jp1[C, l, row, n]`, two atomics
  per iteration. The bra-bra sums are regrouped by row, so the forms agree
  to rounding.

Compiled (`data/measured/hjab.json`, `devtools/measure.sh hjab-compile`,
the rule declared before any row: within 120 s a unit on both arches;
the stack RECORDED, NOT GATED, as #691's `hkx`): every `hjab` / `hdq`
class through (ff) in its default form, its other form where buildable
(every `hdq`; `hjab` through `la + lb = 4`), and every `[p|q]` class the
ladder reads (62, to L = 14), each alone, nvcc 13.0 `sm_86` and hipcc
7.2.4 `gfx1200`. Default units: `hjab` at most 15.3 s of nvcc
(`hjab_32`) and 2.0 s of hipcc, `hdq` at most 4.5 s / 5.9 s; frames worst
`hjab_33` 20,528 B (`sm_86`) / 16,624 B (`gfx1200`), `hdq_33` 12,992 /
11,776 B; three `hjab` units over 14,000 B on `sm_86` (`hjab_23`, `_32`,
`_33`), one on `gfx1200`. The `[p|q]` towers: at most 28 s of nvcc
(`pq_L14_p7_q7`, 28,224 B) and 54 s of hipcc (`pq_L14_p8_q6`). Killed at
120 s, other form only: the unrolled `hjab_22` and `hdq_33` on `sm_86`.

## Witnesses, and what ran where

| What | Against | Where | Worst |
|---|---|---|---|
| host `H` (`test_hessian_host.py`) | central differences, `h = 1e-4`, of `gradient.host.coulomb_gradient` (itself held to FD of the libcint-witnessed `coulomb_matrix`) | the two-atom s, p basis; H₂O/STO-3G `slow` | 5.9e-9 of `max\|H\|` on the s, p basis (the `O(h²)` floor), gate 1e-7 |
| host symmetry, invariance, parts | `H = H^T`, `SUM_A H = 0`, `bb + bk = H`, each part symmetric alone | same | ~2e-15 of `max\|H\|`, gate 1e-13 |
| derivative kets | `SUM` of their `q = 0` entries is the overlap gradient at `M = D` | s, p basis | gate 1e-14 |
| `E_nuc` | FD of `nuclear_repulsion_gradient` | three nuclei | gate 1e-7 |
| `hdq` interpreted (`test_kernel_hessian.py`) | `host.ket_derivative_records` | every ordered class (ss)–(dd) on a three-atom s, p, d basis, and with an f shell on a two-atom s, p, d, f basis (`SPDF`, #690); both forms | **bitwise**, both forms |
| `hjab` interpreted | `host.coulomb_hessian(pairs=)`, host `J`s, the absolute twin as scale | the same classes and bases (L + 2 > 4 `slow`), the default form, and the other form through `la + lb = 4` | worst 0.53 eps of the scale (`hjab_33`, rolled), 0.27 through (dd); gate 16 eps |
| the chain: `hdq` → gather → `[p|q]` at both specs (interpreted) → `hjab` | `host.coulomb_hessian` | the s, p basis; H₂O/STO-3G `slow` | 0.14 eps of the scale on the s, p basis, gate 64 eps; symmetry and invariance 1e-13 |
| compile, `-Wall -Wextra` | clang 18 CUDA front end, device side, runtime stub | `hdq` and `hjab` for every class through (dd), (10), (21) | no diagnostic (a proxy) |
| compile, nvcc / hipcc (`test_emit_hess.py`, `data/measured/hjab.json`) | #92's filter; the store's declared 120 s | every ladder unit through (ff), both arches, each alone; the store also the other forms and the 62 `[p\|q]` classes read | no warning in generated text; default units ≤ 15.3 s nvcc, ≤ 5.9 s hipcc; `[p\|q]` ≤ 28 s / 54 s |
| on a card (`test_hess_j_device.py`, generator `test/hess_driver.py`, on the shared runner since #659, which ran on both cards on 2026-10-30) | `hdq`: `host.ket_derivative_records`, `1e-13` of `max\|D1\|`; `hjab`: `host.coulomb_hessian`, host `J`s, `1e-14` of the absolute twin | RTX 3080 and RX 9060 XT: `hdq` and `hjab` for every ordered class (ss)–(dd) on the three-atom s, p, d basis; the whole H₂O/STO-3G `H` (`slow`); 19 cases per card | `hdq` 4.98e-16 (`hdq_02`); `hjab` 7.16e-17 of the scale per class, 9.28e-17 whole H₂O; the same on both cards, per-class runs bit-identical |
| on a card through (ff) (#690, the same test, 2026-10-31) | the same | both cards: every ordered class (ss)–(ff) in its default form (f classes on `SPDF`), `hdq` in its other form but (ff), `hjab` in its other form through `la + lb = 3`, whole H₂O/STO-3G; 58 cases per card, at most two xdist workers | `hdq` 7.28e-16 (`hdq_32`, rolled); `hjab` 1.16e-16 of the twin (`hjab_33`, rolled); the same on both cards |
| THE CHAIN ON THE CARD (#690, `test_hjab_chain_ondevice.py`): `DQ` and `hdq` on the card, the host's gather, one process of the `Jp2` and `Jp1` `[p\|q]` launches and `hjab` | libcint (numeric goldens `hjab_h2o_*_bse`, `jab.d2`, scale their absolute twins `*_abs`) on H₂O/STO-3G and cc-pVDZ (BSE files); the host on H₂/STO-3G and `SF` (two atoms, s and f, `[p\|q]` to L = 14); `1e-14` of the twin | both cards (cc-pVDZ and `SF` `high_l`, run by hand); 4 cells per card | cc-pVDZ 2.9e-15 / 3.1e-15 of the twin (1.9e-12 / 2.0e-12 of `max\|H\|`), STO-3G 7.9e-16 / 7.4e-16, `SF` 5.2e-17 / 5.1e-17 (RTX 3080 / RX 9060 XT) |
| the host against libcint (`test_hjab_chain_ondevice.py`, host-only, `slow`) | `hjab_h2o_sto3g_bse`, its twin | H₂O/STO-3G (BSE) | 6.7e-16 of the twin (1.2e-12 of `max\|H\|`) |

## The exchange Hessian (`hkx`)

**Through (ff|ff) (#691): interpreted through (dd|dd) and at f, compiled
by nvcc and hipcc, every class run on both cards against the host, and
H₂O/STO-3G and cc-pVDZ against libcint.** The unrolled form below is the
s, p one; from `L = 3` and for every d or f class the kernel is a ROLLED
form ("The rolled form", below). The same skeleton for the exchange
energy,

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

**The rolled form (#691).** The unrolled text is large -- cuda 58 KB at
`(ss|ss)`, 221 KB at `(ss|pp)`, 1.63 MB at `(pp|pp)` -- and over #613's
14,000 B of `sm_86` stack from `L = 3` (`(pp|ps)` 16,936 B, `(pp|pp)`
33,952 B and 69 s of nvcc). So `kernel_hkx.hkx_lowstack` makes the
default rolled from `LOWSTACK_MIN_L = 3` and for every class with a d or
f shell (the unrolled form is built only with every `l <= 1`;
`lowstack=` forces either). It is `kernel_gk`'s #618 shape two orders
up: the bra's per-axis tables at `(la + 2, lc + 2)` and the ket's at
`(lb + 1, ld + 1)`, compact, the ket sign folded into the odd-`t`
entries; `[r]^(0)` stored into a local `R` as it forms; per `(b, d)` row
`I[p]` into a local, then each `J{Y}{l}` into the same local in turn; and
per `(a, c)` row the derivative terms as rolled sums over the bra tables.
The derivative AXES are loops too (`k`, `l` at run time, their unit
vectors I32 locals the shifted rows add, the accumulators local arrays
`BB[27]`, `BK[36]`): a bra-bra component is the rule applied twice, four
shifted rows with each lowered row guarded by the component it lowers;
a bra-ket one is the first-derivative rule against `J`. So the text is
the recurrence plus 25 rolled sums, whatever the class: 147 KB at
`(pp|ps)`, 361 KB at `(ff|ff)`. A first draft unrolled the components in
Python (~160 rolled sums, ~420 KB a unit) and took 75-90 s of nvcc a
unit alone; with the axes rolled every unit compiles in seconds. The
powers of `2 alpha`, `2 gamma` are multiplied in per term, so the forms
agree to round-off, not bitwise; the reductions, atomics, parameters,
launch and `H` contract are the unrolled form's.

Generator: `python -m felizjueves.emit --hess-k-l [LMAX]` (1 bare,
accepted 0..3, exit 2 above) adds the units beside the J ladders, each in
its default form; never in the bare default nor in `KERNEL_GEN_CMD`. The
11 unrolled s, p units below the floor and one rolled unit (`hkx_11_10`)
are goldens; every other default unit through (ff|ff) (244 a backend)
and the unrolled text of the five s, p classes from the floor
(`<name>.unrolled`, the pre-#691 default, its digests unchanged) are
pinned by size and sha256 in `test/golden/hkx_digests.json`
(`test_emit_hess_k.py`).

**Compiled, every unit** (`data/measured/hkx.json`, `devtools/measure.sh
hkx-compile`, the rule declared before any row: within 120 s a unit on
both arches; the stack RECORDED, NOT GATED, as #694's d2eri3, its gate
the card's): all 256 default units and the 16 s, p ones in their other
form, each its own compile, nvcc 13.0 `sm_86` and hipcc 7.2.4 `gfx1200`.
Every default unit at most 4.9 s of nvcc and 7.5 s of hipcc (medians 3.4
s, 4.4 s). Frames: worst `hkx_33_33`, 19,984 B of `sm_86` stack and
16,864 B of `gfx1200` scratch; 19 default units over 14,000 B on
`sm_86`, 3 on `gfx1200`, all `l = 3` classes. The floor is the rows':
the smallest `L` with an unrolled s, p unit over 14,000 B on `sm_86` is
3 (`hkx_11_01`, 17,088 B).

| What | Against | Where | Worst |
|---|---|---|---|
| host `H` (`test_hessian_host.py`) | central differences, `h = 1e-4`, of `gradient.host.exchange_gradient` (itself held to FD of the libcint-witnessed `exchange_matrix`) | the two-atom s, p basis; H₂O/STO-3G `slow` | 5.7e-9 of `max\|H\|` on the s, p basis, gate 1e-7 |
| host symmetry, invariance, parts | `H = H^T`, `SUM_A H = 0`, `bb + bk = H` (each ~10x `max\|H\|`, cancelling), each part symmetric; class quartets sum to the whole; `atoms=` folds back | same | ~4e-15 of `max\|H\|`, gate 1e-13 |
| host, once, by hand (not a test) | `experimental/mmd/hessian`'s `two_electron2` (checked there against libcint's `int2e_ipip1` / `_ipvip1` / `_ip1ip2`) contracted with `D D / 2` | the two-atom s, p basis | 1.2e-14 of `max\|H\|` |
| host against libcint (`test_hkx_device.py`, host-only, `slow`) | the numeric golden `hkx_h2o_sto3g_bse` (`oracle.numeric`, #691: libcint's `int2e_ipip1` / `_ipvip1` / `_ip1ip2` `_cart`, placed by the 8-fold symmetry and contracted with `D (x) D`) | H₂O over data/basis/'s STO-3G, the fixture's density | 4.5e-13 of `max\|H\|`, gate 1e-12 |
| `hkx` interpreted, unrolled (`test_kernel_hkx.py`) | `host.exchange_hessian(quartets=)`, the absolute twin as scale | every s, p class forced unrolled on the two-atom basis (L + 2 > 4 `slow`) and on a three-atom basis with two primitives a shell (all but `(ss\|ss)` `slow`) | ~1 eps of the scale at `(pp\|pp)`, gate 16 eps |
| `hkx` interpreted, rolled | the same | every s, p class forced rolled on the two-atom basis; every class with a d shell and none above (65) and `(ff\|ff)`, `(fs\|fs)`, `(fd\|pd)` on `TRI_SPDF` (three atoms, one bra pair against one off-centre ket pair; `slow`, ~2 min at `(dd\|dd)`) | ≤ ~0.4 eps of the scale, gate 16 eps |
| the whole `d2 E_K` from the 16 launches | the host sweep; symmetry, invariance | the two-atom basis (`slow`) | gate 16 eps; 1e-13 |
| compile, nvcc / hipcc (`test_emit_hess_k.py`, `data/measured/hkx.json`) | #92's filter; the store's declared 120 s | every default unit through (ff\|ff), both arches, each alone | no warning in generated text; ≤ 4.9 s nvcc, ≤ 7.5 s hipcc; frames ≤ 19,984 B (`sm_86`) / 16,864 B (`gfx1200`) |
| on a card, per class (`test_hkx_device.py`, `hess_driver.hkx_launches`) | the host over the launched quartets, `1e-14` of the absolute twin | RTX 3080 and RX 9060 XT: all 256 classes through (ff\|ff) on `TRI_SPDF` (one bra pair, two ket segments, three centres), each its own program in its default form | 1.03e-16 of the twin, both cards (2026-10-31) |
| on a card, rolled against unrolled | each other, `1e-14` of the twin | both cards: the five s, p classes from the floor | ≤ 6.5e-17 |
| on a card, whole molecule | the numeric goldens `hkx_h2o_{sto3g,ccpvdz}_bse` (libcint), the fp64 tier (1e-12 of `max\|H\|`) | both cards: H₂O over STO-3G and cc-pVDZ (every K class of the basis in one program) | STO-3G 1.8e-13 / 3.3e-13, cc-pVDZ 8.4e-13 / 9.3e-13 of `max\|H\|` (RTX 3080 / RX 9060 XT); run to run up to 3.9e-13 |

The whole-molecule margin is thin, and why: the two runs of one card
differ by up to 3.9e-13 of `max|H|` (STO-3G, the RX 9060 XT), so the
rounding scale of an assembled `d2 E_K` is not `max|H|`. The O s
exponents enter the bra-bra rule as `(2 alpha)^2` and the same-atom
blocks cancel (the twin is ~3e3 `max|H|` at f), and the host itself
meets libcint at 4.5e-13. The gate is the store's fp64 tier, stated
before the run and not widened; experimental/mmd/hessian's check uses
1e-11 for the same reason. The per-class witness is the tight one.

**On the card the frame is the launch's.** A launch reserves its stack
for every resident thread, ~2 GB of the RTX 3080's 10 GB at `(ff|ff)`:
under xdist every worker shares the card, and with six `hkx_33_32` and
`hkx_33_33` failed to launch ("cuda: launch: out of memory") where they
pass alone. The `high_l` cases are run with at most two workers.

Not done for `hkx`: a screen; any clock (no wall-clock row: the cost of
the segment loop, one serial pass per ket atom pair with 36 reductions
each, and of the run-time axis loops against the unrolled digests is
unexamined); the response term and an assembled RHF Hessian.

## What is not done

- The gather of the derivative kets is the host's (the design's step 3),
  so the on-card chain reads `D1` back once, as the J chain reads `Dmax`
  back for its sort; the derivative kets are unsorted and unscreened (the
  chain's plan runs at tau = 0). The chain is witnessed on four bases, not
  on a molecule with f (the f cell is a synthetic two-atom s, f basis,
  held to the host; a libcint golden over cc-pVTZ is not built).
- The skeleton Coulomb and exchange terms and (below) the one-electron
  and DF `(P|Q)`, `(mn|P)` skeletons only. Not built: the nuclear
  repulsion as a
  kernel (it is host-side), the derivative Fock matrices `F^x` (only
  their exchange part, "The exchange derivative matrix", and their
  one-electron part, "The one-electron derivative matrices", are built;
  `J^x` is #688) and the CPHF response, so no whole RHF Hessian is
  assembled or checked against `pyscf.hessian.rhf`.
- The bra–ket term runs the `[p|q]` kernel `3 n_atom` times per class
  pair, each over the kets touching one atom (about `6 n_ket` kets in all),
  and `hjab` adds `18 n_atom` atomics per bra record whether or not the
  atom's `Jp1` row is zero. A per-atom ket mask or a block mapping is a
  follow-up.
- The ladder stops at (ff). Only the compile is measured
  (`data/measured/hjab.json`); no clock, rolled against unrolled or
  otherwise. Nothing is screened, Cartesian only.

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
  "The exchange Hessian" (through (ff|ff), #691); no CPHF response, no assembled
  RHF hessian; `experimental/mmd/hessian` has the
  rest for s and p, in numpy.
- Nothing above (dd) has run on a card; nothing measured.
- The low-stack `d2v` is not measured: no frame, spill or clock on
  either card, and nothing says which form is faster. Its frame is mostly
  the shared `(la + 2, lb + 2)` axis tables, which keep slots no read
  reaches (`t > i + j`, the doubly raised corner); packing them is a
  follow-up. `S` and `T` stay unrolled (`d2t_33` is 456 KB).
- No screening, Cartesian only.

## The one-electron derivative matrices (`S^x`, `T^x`, `V^x`, #687)

**Status: implemented host-side and as generated kernels, interpreted
through (ff) and held to libcint, compiled by nvcc and hipcc through (dd);
not run on a card** (the session that built them had both compilers and
no card; the device test is written, its compile-only twins green).

An analytic RHF Hessian needs, per perturbation `x = (atom X, axis k)`,
the CPHF right-hand side `B^x_ai = F^x_ai - eps_i S^x_ai - ...`, with
`F^x = h^x + J^x[D] - 1/2 K^x[D]` the skeleton derivative Fock matrix, and
`B^x` has to exist explicitly in occ x virt for all `3 n_atom`
perturbations. A d1 kernel writes a contraction, `G[atom, k] += SUM_mn
M_mn dX_mn/dR`, which cannot produce it. These kernels are the
one-electron half: `S^x` and `h^x = T^x + V^x` as `[3, n_ao, n_ao]`
Cartesian matrices, row-major (spherical is the caller's, `U X U^T`).

### Their keys

The matrix form is a kernel-body LEVER (`levers.VOCABULARY`), as the DF
metric's uncontracted derivative is (#693, `gradient.kernel_ipmetric`):
#693's `uncontracted`, widened to `S`, `T` and `V`, is the ip1 form, and
`rinv`, legal on `V` alone, is the operator term. Both are legal at d1
only (`levers.order_refusal`) and on MMD only (the `.hgp` / `.rys` slots
refuse them), and `kernel_ip1e` takes exactly one of them:
`overlap/d1/(2,1)/b1/s/uncontracted`, `nuclear/d1/(2,1)/b1/v/rinv`. A key
without either is the contraction, unchanged. The case registry counts
a lever as a variant of its d1 case (`ips_` is `(mmd, s, 1)`).

### What they write

The **ip1 form** (`ips_`, `ipt_`, `ipv_{la}{lb}`): the derivative on the
centre of the matrix's FIRST index, every atom in one launch per pair
class (the `gs` / `gt` / `gv` launch, one thread per record),

    X1[k, m, n] += d X_mn / dA_k        A the centre of m

libcint's `int1e_ipovlp` / `int1e_ipkin` / `int1e_ipnuc`, negated. A
record of shell pair `i <= j` on `A`, `B` writes its `(m, n)` block (the
`dA` rule) and, off the diagonal, its `(n, m)` block (the `dB` rule); a
diagonal pair's records are every ordered primitive pair, so its `dA`
sum is already the block's whole first-index derivative and it writes
that alone. The caller gets atom `X`'s matrix as the rows of `m` on `X`
plus that slab's transpose. `kernel_g1e`'s record head, rule and per-axis
tables (`axis_tables_fragment(la + 1, lb + 1)`, `lb + 3` for `T`) are
reused; `ips` and `ipt` are `gs` / `gt` per row with no weight, unrolled.
`ipv` holds the basis function's derivative with every nucleus's operator
fixed: per record the nuclei's potential in Hermite space, `U_t = -SUM_C
Z_C [t]^(0)(P - C)` over `|t| <= L + 1` (`kernel_v.v_plan((la, lb + 1))`
per nucleus, into a local `R`, added into `U`), then per block row the
low-stack `gv`'s rolled sums (`kernel_g1e.lowstack_derivative`) against
`U` -- one form at every class, the frame the tables, `R` and `U`.

**`V`'s operator term** (`iprv_{la}{lb}`, the rinv form) is not in ip1:
`d/dC_k` of the operator at nucleus `C` moved with it,

    XC[k, m, n] += d V^C_mn / dC_k = s Z_C SUM_t E^{ab}_t [t + 1_k]^(0)(P - C)

which is `-Z_C (iprinv + iprinv^T)` with libcint's `rinv` origin at `C`.
One launch per nucleus, its index the kernel argument `nuc` (PySCF's
`with_rinv_at_nucleus` form), writing `[3, n_ao, n_ao]`; `gv`'s
Hellmann–Feynman branch has the same term reduced against `M`. The matrix
is symmetric, so a row writes one value to `(m, n)` and `(n, m)`.

So `dV/dR_X = (X1 rows on X) + (that)^T + XC(X)`, and translational
invariance is `X1 + X1^T + SUM_C XC(C) = 0`; `S` and `T` have no operator
term (`X1 + X1^T = 0`).

### The host reference (`gradient/host.py`)

`overlap_ip1`, `kinetic_ip1`, `potential_ip1` and `potential_rinv`, on the
gradient sweep's shell pairs and records (`_matrix_sweep`), each row by
its own arithmetic: `gradient.host`'s per-axis `E`, derivative rule and
kinetic bracket, `[t]^(0)` injected (`one_centre=`).

### The generator

`python -m felizjueves.emit --ip-one-e-l [LMAX]` (2 bare, 0..3, a 2 above) adds
`ips_`, `ipt_`, `ipv_` and `iprv_{la}{lb}` for every `la <= lb <= LMAX`
beside the J ladders; never in the no-flag output and not in
`KERNEL_GEN_CMD`. The goldens are the six classes (ss) through (dd) per
backend (`test/golden/{cuda,hip}/{ips,ipt,ipv,iprv}_*`, 24 each); every
unit through (ff), both orders, is under the 512 KiB cap (the largest,
`ipt_33`, ~210 KB).

### Witnesses, and what ran where

| What | Against | Where | Worst |
|---|---|---|---|
| the host, `S` `T` `V` ip1 and `V`'s rinv at each nucleus | libcint, `-int1e_ip*` and `-Z_C (iprinv + iprinv^T)`, element by element (#550's gradient fixtures) | H₂O/STO-3G; H₂O/cc-pVDZ `slow` (`test_kernel_ip1e.py`) | 2.8e-15 of `max\|libcint\|` (`V`, cc-pVDZ), gate 1e-12 |
| the host | `X1 + X1^T + SUM_C XC(C) = 0` (`V`); `X1 + X1^T = 0` (`S`, `T`) | the same | 3.2e-16 of `max\|X1\|`, gate 1e-13 |
| `kernel_ip1e`, interpreted | the host per class, every form, `iprv` at two nuclei | every ordered class (ss)–(ff) on HF/cc-pVTZ (above L + 1 = 4 `slow`) | `ips`, `ipt` bitwise; `ipv`, `iprv` <= 2 eps of the class's `max\|X\|`, gate 16 eps |
| the same, whole matrices | libcint, and the invariance | H₂O/STO-3G; cc-pVDZ `slow` | as the host |
| nvcc / hipcc compile | #92's warning filter | `test_emit_ip1e.py`'s 24 units per backend, `sm_86` and `gfx1200`; `test_ip1e_device.py`'s runners | passed, no warning in generated text |
| on a card | the host within 1e-14 of `max\|X\|`, libcint within 1e-12 | `test_ip1e_device.py` (generator `test/ip1e_driver.py`), H₂O/STO-3G, cc-pVDZ `slow` | **not run** (no card in that session) |

### What is not done

- Not run on a card; nothing measured. Above (dd) interpreted only (the
  ladder emits through (ff); the compile goldens stop at (dd)).
- `F^x`'s Coulomb part `J^x` (#688; its exchange part `K^x` is "The
  exchange derivative matrix", #689); the CPHF solve and the whole RHF
  Hessian.
- The ladder emits `la <= lb` only, as `--hess-one-e-l` does; a build
  whose shell order puts the higher `l` first needs the `(lb, la)` units,
  which build and are interpreted but are not on the ladder.
- Cartesian only, no screening; `X1` takes `3 n_ao^2` doubles per matrix
  and `iprv` one launch and one matrix per nucleus.

## The DF metric Hessian (`(P|Q)`)

**Status: implemented host-side and as a generated kernel through (i|i),
interpreted against the host, compiled by nvcc and hipcc, and run on both
cards against the host** (#694). Every class `lP <= lQ <= 6`
is interpreted against the host and compiled by nvcc 13.0 for `sm_86`,
and by hipcc (ROCm 7.2.4) for `gfx1200`, with each unit's compile time
and frame recorded on both arches (`data/measured/d2_df_lowstack.json`).
The device test (`test_hess_df_device.py`) ran on the RTX 3080 and the
RX 9060 XT on 2026-10-31: H₂O/cc-pVDZ against cc-pVDZ-JKFIT (every class
through (f|f)) and the aux-i system per aux `l` (every class through (i|i),
each in its default form), within 9.0e-18 of the host's twin on both, the
bound 1e-14. Nothing is
scored: there is no published board and no claim was added.

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

**The low-stack form (#694)** is the default from `lP + lQ >=
kernel_d2metric.LOWSTACK_MIN_L = 9` (`d2metric_lowstack`; `lowstack=`
forces either form, and the name, parameters, launch and `H` contract are
the same). It is `kernel_gmetric`'s low-stack form read one order up. No
`E` block, weight or density is held. Per record pair the tower is stored
into a local `R[n_t(L + 2)]` as it forms. Per Q row,
`U[t'] = SUM_tau R[t' + tau] E^b_tau` is formed for every `|t'| <= lP +
2` of `lP`'s parity. Per element, the weight is read from `M`, and the six
`h_kl +-= w wt SUM_t E^a_t U[t + 1_k + 1_l]` are rolled over the P
record's single-center tables. The floor is measured
(`devtools/measure.sh d2-df-lowstack-compile`, its rule declared before
any row): every unrolled class through `L = 8` is within #606's 14,000 B
of `sm_86` stack, and at `L = 9` the unrolled (d|i) is 19,216 B. The (g|h)
is 13,888 B there, but the floor is per level. The low-stack (i|i) is
9,200 B and 135 KB of text, and compiles in 1.9 s.

### The generator

`python -m felizjueves.emit --hess-metric-l [AUXMAX]` (2 bare, 0..6, a 2
above) adds `d2metric_{lP}{lQ}` for every `lP <= lQ <= AUXMAX` beside the
J ladders. AUXMAX is the aux `l` per shell, as `--grad-df-aux-l`'s
(#694); the flag was a total-`L` bound before. It is never in the
no-flag output and not in `KERNEL_GEN_CMD`. The 9 classes with `lP + lQ
<= 4` keep their goldens per backend, `test/golden/{cuda,hip}/d2metric_*`
(largest ~36 KB), unchanged by #694. The other 19 classes through (i|i)
are pinned by byte count and sha256 in
`test/golden/d2metric_digests.json`.

### Witnesses, and what ran where

| What | Against | Where | Worst |
|---|---|---|---|
| `hessian.host_df` | central FD (`h = 1e-4`) of `gradient.host_df.metric_gradient`, itself FD-held to the libcint-witnessed `(P|Q)` | a synthetic s/p aux basis on three centres; s/p/d and f/g bases `slow` (`test_hessian_host_df.py`) | ~2.5e-9 of `max\|H\|` (`O(h²)`), gate 1e-7 |
| the same | `SUM_A H = 0`; `H = H^T`; split atoms folded back | the same | gates 1e-12 and 1e-13 of `max\|H\|` |
| `kernel_d2metric`, interpreted | `hessian.host_df` per class, `W` non-symmetric | every class `lP + lQ <= 4` on `test_kernel_gmetric.py`'s aux basis (above `lP + lQ + 2 = 4` `slow`); a whole s/p aux basis (`test_kernel_d2metric.py`) | 0.17 eps of the absolute twin (`d2metric_02`), gate 16 eps |
| the same, the ladder (#694) | `hessian.host_df` per class | every class `lP <= lQ <= 6` in its default form, every class below the floor in the low-stack form forced, (d\|i) and (g\|h) unrolled forced, on a two-shell aux basis with a swapped and a diagonal pair (`test_kernel_d2_df_ladder.py`, 52 cases) | 0.34 eps of the twin (`d2metric_06`), gate 16 eps |
| nvcc 13.0, `sm_86` | #92's warning filter | all 28 classes (`test_emit_d2metric.py`), 2026-10-30 | clean |
| hipcc 7.2.4, `gfx1200` | #92's warning filter | all 28 classes, 2026-10-30 | clean |
| the same, frames | `d2-df-lowstack-compile` | every class low-stack on both arches; unrolled through `L = 9` (`sm_86`) and `L = 11` (`gfx1200`) | low-stack ≤ 9,200 B `sm_86`, ≤ 9,744 B `gfx1200` scratch; ≤ 10 s a unit |
| on a card | `test_hess_df_device.py`, `hess_driver.d2metric_launches`, `RUNS` = 2 runs into a zeroed `H` | RTX 3080 and RX 9060 XT, 2026-10-31: H₂O/cc-pVDZ against cc-pVDZ-JKFIT (10 classes through (f\|f)); the aux-i system, one program per `lQ` = 0..6 (all 28 classes through (i\|i), default forms) | 5.0e-18 (cuda) / 5.1e-18 (hip) of the twin on H₂O; ≤ 9.0e-18 on the ladder, bound 1e-14; ladder runs bit-identical, H₂O's not (atomics) |

### What is not done

- No `d` response, no assembled DF-J (or DF-K) Hessian (the three-center
  `(mn|P)` Hessian is the next section).
- Nothing against libcint directly (the chain is FD → gradient host →
  libcint's `(P|Q)`).
- No clock measured; Cartesian only.

## The three-center DF Hessian (`(mn|P)`)

**Status: implemented host-side and as generated kernels through (ff|i),
interpreted against the host, compiled by nvcc and hipcc, and run on both
cards against the host** (#694). Every class `la <= lb <= 3`,
`lP <= 6` is interpreted against the host and compiled by nvcc 13.0 for
`sm_86`, and by hipcc (ROCm 7.2.4) for `gfx1200`, with each unit's compile time
and frame recorded on both arches (`data/measured/d2_df_lowstack.json`).
The device test (`test_hess_df_device.py`) ran on the RTX 3080 and the
RX 9060 XT on 2026-10-31: H₂O/cc-pVDZ against cc-pVDZ-JKFIT (every class
through (dd|f)) and the aux-i system per aux `l` (all 70 classes through
(ff|i), each in its default form), within 1.2e-17 of the host's twin on
both, the bound 1e-14. This
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

The text of that form is straight-line and grows fast with the class:

| Class (L) | cuda text, unrolled | `sm_86` stack, unrolled | low-stack text | low-stack stack |
|---|---|---|---|---|
| `(ss\|s)` (0) | 40 KB | 376 B | — | — |
| `(pp\|s)` (2) | 206 KB | 7,680 B | — | — |
| `(pp\|p)` (3) | 504 KB | 20,280 B | 100 KB | 4,480 B |
| `(pd\|p)` (4) | 1.25 MB | not compiled | 115 KB | 6,752 B |
| `(ff\|s)` (6) | — | — | 178 KB | 19,904 B |
| `(ff\|i)` (12) | — | — | 293 KB | 24,544 B |

**The low-stack form (#694)** is the default from `la + lb + lP >=
kernel_d2eri3.LOWSTACK_MIN_L = 3` (`d2eri3_lowstack`; `lowstack=` forces
either form, and the name, parameters, launch and `H` contract are the
same). It is `kernel_geri3`'s low-stack form with the low-stack `d2v`'s
operated factors. No density and no weight is held. Per aux record, the
tower goes into a local `R[n_t(L + 2)]`. Per aux row,
`U[t] = SUM_tau R[t + tau] E^q_tau` is formed for every `|t| <= la + lb +
2`. Per element `(a, b, q)`, the weight is read from `M`, and each axis's
operated factors `F{op}[d, t]` (`op` in `"" A B AA AB BB`) are filled by
`kernel_h1e.operated_value`'s loops. Then the 28 operator triples are
contracted against `U[t + up]` by `kernel_h1e.rolled`, the low-stack
`d2v`'s contraction with `U` for `R`, into 45 accumulators. Per cell,
`hXY = s qXY`, `hXC = -s qX` and `hCC = s qC` (`s = (-1)^{lP}`) go out by
the same 81 atomics.

The floor is measured (`devtools/measure.sh d2-df-lowstack-compile`, its
rule declared before any row). Every unrolled class at `L <= 2` is within
#606's 14,000 B of `sm_86` stack. At `L = 3`, (sp|d), (sd|p), (pp|p) and
(pd|s) are not, at 14,768 to 20,280 B. **The low-stack frames are
recorded, not gated** (the rule says so, and why, before any row): the
bra tables two indices up are 9.5 KB alone at (ff|.), so 19 units exceed
14 KB on `sm_86` (9 on `gfx1200`, worst 20,640 B) ((pf|g) to (pf|i), (dd|g) to (dd|i), and every (df|.) from p and
every (ff|.)). The worst is (ff|i) at 24,544 B, 8 KB of it spill. Every unit compiles within 29 s ((df|p)), far inside the 300 s
budget. A form that brings (ff|.) under 14 KB is follow-up work. The
likely candidates are a triangular table layout (`i + j <= la + lb + 2` is
all the rule reads) or factors built from the tables one up on demand.

### The generator

`python -m felizjueves.emit --hess-eri3-l [LMAX] [--hess-eri3-aux-l
AUXMAX]` adds `d2eri3_{la}{lb}{lP}` for every `la <= lb <= LMAX`, `lP <=
AUXMAX`, beside the J ladders. The split is `--grad-df-l`'s (#694); the
flag was a total-`L` bound before:

- LMAX is 1 and AUXMAX 2 when the flags are bare or AUXMAX is omitted (9
  units). LMAX is accepted from 0 to 3 and AUXMAX from 0 to 6; above
  either, or `--hess-eri3-aux-l` without `--hess-eri3-l`, the CLI exits 2.
- It is never in the no-flag output and not in `KERNEL_GEN_CMD`.

The goldens are the seven classes with L <= 2, kept as files per backend
(`test/golden/{cuda,hip}/d2eri3_*`), unchanged by #694. Every other class
of the old ladder and the new one (71 in all, (sg|s) the old one's only)
is pinned in its default form by byte count and sha256 in
`test/golden/d2eri3_digests.json`. The 15 old classes at L = 3 and 4 also
keep their unrolled digests, unchanged, under `<name>.unrolled`.

### Witnesses, and what ran where

| What | Against | Where | Worst |
|---|---|---|---|
| `hessian.host_eri3` | central FD (`h = 1e-4`) of `gradient.host_df.eri3_gradient`, itself FD-held to the libcint-witnessed `(mn\|P)` | a three-atom s, p AO / s, p, d aux basis; H₂O/STO-3G against the cc-pVTZ-JKFIT s and p shells `slow` (`test_hessian_host_eri3.py`) | 4.8e-9 of `max\|H\|` (`O(h²)`), gate 1e-7 |
| the same | `SUM_A H = 0`; `H = H^T` | the same | 4e-15 / 4e-16 of `max\|H\|` (gates 1e-12, 1e-13) |
| `kernel_d2eri3`, interpreted | `hessian.host_eri3` per class, `Gam` random and non-symmetric | every class `la <= lb`, `la + lb + lP <= 4` on `test_kernel_geri3.py`'s bases (L + 2 > 4 `slow`); whole H₂O/STO-3G against JKFIT s, p `slow` (`test_kernel_d2eri3.py`) | 0.40 eps of the absolute twin (`d2eri3_110`), gate 16 eps |
| the same, the ladder (#694) | `hessian.host_eri3` per class | every class `la <= lb <= 3`, `lP <= 6` in its default form, every class below the floor in the low-stack form forced, (sp\|d) and (pp\|p) unrolled forced, on a two-shell AO basis with a swapped pair and, for `la == lb`, a diagonal one (`test_kernel_d2_df_ladder.py`, 79 cases, 2026-10-30) | every case within the gate; worst printed 0.49 eps of the twin (`d2eri3_010`, both forms; the 49 classes at L ≥ 5 ran under xdist, which kept no per-class figure), gate 16 eps |
| nvcc 13.0, `sm_86` | #92's warning filter | all 71 pinned classes (`test_emit_d2eri3.py`), 2026-10-30 | clean |
| hipcc 7.2.4, `gfx1200` | #92's warning filter | all 71 pinned classes, 2026-10-30 | clean |
| the same, frames | `d2-df-lowstack-compile` | every class low-stack on both arches; unrolled through `L = 3` | low-stack ≤ 24,544 B `sm_86`, ≤ 20,640 B `gfx1200` (recorded, not gated), ≤ 29 s a unit |
| on a card | `test_hess_df_device.py`, `hess_driver.d2eri3_launches`, `RUNS` = 2 | RTX 3080 and RX 9060 XT, 2026-10-31: H₂O/cc-pVDZ against cc-pVDZ-JKFIT (24 classes through (dd\|f)); the aux-i system, one program per `lP` = 0..6 (all 70 classes through (ff\|i), default forms, the over-14-KB low-stack frames included) | 4.5e-18 (cuda) / 3.2e-18 (hip) of the twin on H₂O; ≤ 1.2e-17 on the ladder (`lP` = 2), bound 1e-14; ladder runs bit-identical, H₂O's not (atomics) |

### What is not done

- The low-stack frames at f exceed #606's 14 KB (above). No clock is
  measured.
- No assembled DF-J Hessian. The DF-J Hessian is this term with
  `Gam = D (x) d`, plus the metric term (`d2metric_`) with
  `W = -1/2 d d^T`, plus the response of `d`. The last is not a skeleton
  term.
- No DF-K, no screen. Cartesian only.

## The exchange derivative matrix (`kx1`, `kxc`)

**Status: implemented host-side and as generated kernels, interpreted
through (dd|dd) and at three f classes, held to PySCF, compiled by nvcc
and hipcc at every class through (ff|ff) (the local-memory frame
recorded); not run on either card** (#689). The session that built it had
both compilers and no card: no device test or driver is written, so
nothing here has launched on the RTX 3080 or the RX 9060 XT.

**Nothing here is scored.** The rule is `kernel_gk`'s (`docs/references.md`
entry 46, McMurchie & Davidson); no `claims.json` record, cell or floor was
added.

### Why a matrix

The RHF Hessian's CPHF right-hand side needs, per perturbation `x =
(atom X, axis k)`, the skeleton derivative Fock matrix `F^x = h^x + J^x -
1/2 K^x` as an AO matrix, for all `3 N` perturbations. A d1 kernel writes a
contraction `G[atom, k]` (`spec.py`, "A GRADIENT CLASS"), which cannot give
it. This is the exchange part:

    K^X_ab = SUM_cd D_cd d(ac|bd) / dR_{X,k}        (K's pairing)

The one-electron half is #687 and `J^X` is #688; neither is built here.

### The two forms, and the assembly

`D_cd` straddles bra and ket, so there is no derivative density to reuse
(the reason `hkx` is one kernel), and both bra functions need a kernel:

- **ip1**, the derivative on the matrix index, every atom at once:
  `K1[k, a, b] = SUM_cd D_cd (dA_k a c|b d)`;
- **contracted**, the derivative on the contracted index, per atom:
  `KC_X[k, a, b] = SUM_{c on X} SUM_d D_cd (a dC_k c|b d)`.

For a symmetric `D`, the ket's two derivatives are the bra's transposed
(`(ac|b dd) = (b dd|a c)`), so

    K^X = P_X K1 + (P_X K1)^T + KC_X + KC_X^T

with `P_X` keeping the rows on `X`. That is PySCF's `hessian.rhf.make_h1`
exchange part times `-2`: its `vk1` is `KC_X^T`, and its `vk2` on the rows
of `X` is `P_X K1`. `K1` alone is `grad.rhf.get_jk`'s `vk`.

### Its keys

Two targets, `Target.KX1` (slot `kx1`) and `Target.KXC` (slot `kxc`). Each
is a K class in K's Hermite pairing `(la, lc | lb, ld)`, operator
`COULOMB`. They are legal at d1 **only** (`spec.D1_ONLY_TARGETS`: at d0 the
matrix is `Target.K`, and no d2 is asked for), and on MMD only:
`coulomb/d1/(la,lc|lb,ld)/b1/kx1/literal` and `.../kxc/literal`. One
builder makes both, as the dipole and quadrupole share one. No lever, one
batch.

### The kernel (`hessian/kernel_kx.py`)

`kx1_{la}{lc}_{lb}{ld}` and `kxc_{la}{lc}_{lb}{ld}` use `kernel_gk`'s
mapping: one warp per bra record, kets over the lanes, no shared memory,
no screen. They run `gk`'s plan (the d0 K class with the bra raised by
one). Both are built in the **rolled form only**, so #618's lesson applies
from the start: no `E` block and no straight-line fold. Each kernel holds:

- the bra's compact per-axis tables, one index up on the moved side only:
  `(la + 1, lc)` for `kx1`, `(la, lc + 1)` for `kxc`;
- `[r]^(0)` stored into a local `R`;
- the ket's compact tables, with the ket sign folded into their odd-`t`
  entries;
- one local scatter row `Y[3 n_a]`.

Then, per ket `b` row:

    Y = 0
    for every d row:  I[p] = s_bd SUM_q E^bd_q R[p + q]
        for every (a, c) row:  Y[k, a] += D_cd (2e SUM E^{+1_k} I - n_k SUM E^{-1_k} I)
    X[k, oa + a, ob + b] += s_ac Y[k, a]          AtomicAdd

There is no warp reduction: each lane's ket is its own, and lanes whose
kets share a shell pair meet in the atomics. `kxc` is launched once per
atom, with the bra records whose `c` sits on it
(`host_kx.kx_records(..., atom=X)`), so the atoms together cost one pass.
The output `X[3, n_ao, n_ao]` is zeroed by the caller. It is Cartesian;
spherical is the caller's.

### The host reference (`hessian/host_kx.py`)

`exchange_derivative_ip1`, `exchange_derivative_contracted(atom)` and
`exchange_derivative(atom)` (the assembly) sweep every ordered quartet as
`gradient.host.exchange_gradient` does: `E^{ac}` differentiated before the
fold, and `reference.pq`'s `[r]`. `absolute=True` is the round-off scale.
It is pure Python: H₂O/cc-pVDZ's `K1` takes ~5 min.

### Witnesses, and what ran where

| What | Against | Where | Result |
|---|---|---|---|
| `K1` (host) | the gradient fixtures' `vk1` (libcint `int2e_ip1`, `ijkl,jk->il`) | H₂O/STO-3G, + a d on O, cc-pVDZ (`slow`) | 8.9e-15 / 3.6e-14 / 8.3e-14 absolute |
| `K^X` per atom (host) | `test/oracle/k_deriv/*_kx.json`, PySCF's `make_h1` exchange part (`manifest.kx_witness`, `oracle.generate --kx`) | the same three (`slow`) | 2.3e-14 / 3.6e-14 / 1.5e-13 absolute |
| `K^X` (host) | central differences of `exchange_matrix` | a two-atom s basis (fast), the s, p basis (`slow`) | within 1e-7 of `max\|K^X\|` |
| `SUM_X K^X = 0` | — | host (s; s, p `slow`), the three witnesses, the kernels' assembly | 1e-13 of scale |
| `kx1`, `kxc` per class (interpreted) | the host, `quartets=` the class | every s, p class; every d class through (dd\|dd) (`slow` above L + 1 = 4); (ff\|ff), (fs\|fs), (fd\|pd) (`slow`) | within 16 eps of the absolute twin |
| nvcc / hipcc compile | #92's warning filter | `test_emit_hess_kx.py`: the 14 goldens and (ff\|ff) in both forms | 32 passed, no warning; slowest 6.0 s (`kx1_33_33`, hipcc) |
| frame | `data/measured/kx.json` (`devtools/measure.sh kx-compile`, rule declared before any row: stack ≤ 14,000 B, compile ≤ 30 s on both arches) | all 512 units, `sm_86` and `gfx1200` | every unit admissible on both: worst `kx1_33_33`, 9,824 B of `sm_86` stack (556 B spill stores at most) and 10,192 B of `gfx1200` scratch; slowest compile 5.6 s |
| on a card | — | — | **not run**: no device test or driver |

### The generator

`python -m felizjueves.emit --hess-kx-l [LMAX]` writes both forms of every
K class with each `l <= LMAX`. Bare is 1 (32 units), and it accepts
0..3. It is never in the bare default nor in `KERNEL_GEN_CMD`. There are
14 goldens (7 classes in both forms) plus digests of the other 498 units
per backend (`test/golden/kx_digests.json`).

### What is not done

- **No card.** Nothing has launched. The issue's "every class launches on
  the RTX 3080" and "run on both cards against the host" are open: what
  stands in for them is the compiled frame of every unit. The device test
  and its driver (`gk_driver.py`'s pattern, `X` zeroed, `kxc` once per
  atom) are the next step.
- `h^x`, `S^x` (#687) are not built (`J^X` is the next section), so
  `F^x` is not assembled, and neither is the CPHF right-hand side.
- Nothing is timed, and nothing is screened. A per-atom ket mask for
  `kxc`, or a block mapping, is a follow-up.

## The Coulomb derivative matrix (`J^X`)

**Status: host-side, held to PySCF; its bra kernel `jx_` generated,
interpreted through (ff) and compiled by nvcc and hipcc through (ff), not
run on a card** (#688). Its device test is written. The session that built
it had both compilers and no card.

The CPHF right-hand side of the RHF Hessian needs the derivative Fock
matrix `F^x` as an AO matrix, not contracted to `[atom, 3]`. Its
one-electron half is #687 and its exchange part `K^X` is #689 ("The
exchange derivative matrix", above). This is its Coulomb part: for atom `X` and axis `k`,

    J^X_mn = SUM_ls D_ls d(mn|ls) / dR_{X,k}

### The split

    J^X = BRA + KET
    BRA_mn = [m on X] J1[k, m, n] + [n on X] J1[k, n, m]
    J1[k, m, n] = SUM_ls (dA_k m n|ls) D_ls        A the centre of m
    KET_mn = SUM_ls (mn|d_{X,k}(ls)) D_ls

`J1` is the **ip1** form. It is PySCF's `grad.rhf.get_jk` `vj`, libcint's
`int2e_ip1` contracted `ijkl,lk->ij` and negated, since libcint's `nabla`
is minus the centre derivative. PySCF's `hessian.rhf.make_h1` builds the
same `J^X`: its `vj2` on X's rows plus the transpose is `BRA`, and its
`vj1 + vj1^T` is `KET`.

### Its key

`Target.JX`, a pair class `(la, lb)`, operator `COULOMB`, key
`coulomb/d1/(la,lb)/b1/jx/literal`. A target and not a lever, because it is
a different matrix from both `JAB`s: d0's `J`, and d1's contraction into
`G`. It exists at derivative order 1 **only** (`spec.D1_ONLY`); d0 and d2
are refused with the reason. It is unbatched for `JAB`'s structural reason,
and MMD only. Every earlier key is byte-identical.

### The bra kernel (`hessian/kernel_jx.py`)

`jx_{la}{lb}`, one per ordered bra class, all atoms in one launch. The
chain is `coulomb_gradient`'s (`gjab`) up to its last step. The ket phase
(`DQ`) is unchanged, and so is the raised `[p|q]` at `jx_pq_spec` =
`gjab_pq_spec`, `J_t` over `|t| <= la + lb + 1`. Then, per bra record (one
thread) and block row `(a, b)`:

    gA_k = s SUM_t (dA_k E^{ab})_t Jp[row, t]     J1[k, oa + i, ob + j] += gA_k
    gB_k = s SUM_t (dB_k E^{ab})_t Jp[row, t]     J1[k, ob + j, oa + i] += gB_k

These are `gjab`'s sums with no weight `M`. Where `gjab` contracts the rows
with `D` and adds into `G[atom]`, `jx` scatters each row the way `jab`
does. `J1` is not symmetric, so there is no mirror; the `B` term is the
transposed element.

A **diagonal shell pair** (`off_a == off_b`) writes the `A` term alone,
every `(i, j)`. Its records are every ordered primitive pair, so the `B`
term of `(i, j)` at `(p, q)` is the `A` term of `(j, i)` at `(q, p)`.

**One form, rolled.** The kernel takes `gjab`'s low-stack shape: the
per-axis tables one index up, and the six sums as loops
(`kernel_g1e.lowstack_derivative`, its weight the record scalar `s`)
reading the record's `Jp` row in global memory. Every row is its own output
here, so an unrolled form's text grows as rows × Hermite targets, with no
density to fold the rows first as `gjab`'s does. Whether an unrolled form
pays below some `L` is a measurement nobody has made, so it is not built.

The parameters are `jab`'s plus the exponents: `n_rec n_ao n_rows`,
`zeta_ab XPA XPB s ea_bra eb_ket off_a off_b row`, `Jp[n_rows, n_p1]`, and
the OUT `J1[3, n_ao, n_ao]`, zeroed by the caller. There is no `M` and no
atom; the caller's gather takes atom `X`'s rows.

### The ket half: no kernel

`KET` is the ordinary J build over the derivative kets of `(X, k)`.
`hdq_`'s `D1` slabs are gathered per atom and axis, as `hjab`'s bra–ket term
gathers them. Then the d0 `[p|q]` of `(la, lb | lc + ld + 1, 0)` runs over
them, and the d0 `jab_` scatters into `J`. Every piece is an existing
kernel. `test_kernel_jx.py` runs this chain interpreted.

### The host reference (`hessian/host_jx.py`)

`coulomb_ip1(basis, D, one_centre=)` is `J1`, by `gradient.host`'s sweep.
`J_t` comes from `coulomb_jp` at `L + 1`, and each derivative is formed
before the sum. It has an `absolute=True` twin, the kernel test's round-off
scale. `coulomb_derivative_matrix(basis, D, nuclei, one_centre=,
parts=)` is `J^X` (`parts` "both", "bra" or "ket"). Its ket half is a J
build over `hessian.host.coulomb_dkets`, the derivative kets whose records
`hdq_` writes.

### The generator

`python -m felizjueves.emit --jx-l [LMAX]` writes every ordered class with
each `l <= LMAX` (2 bare, 0..3). It is never in the bare default nor in
`KERNEL_GEN_CMD`. `test_emit_jx.py` holds 4 goldens per backend, (ss) (sp)
(ps) (pp).

### Witnesses, and what ran where

| What | Against | Where | Worst |
|---|---|---|---|
| `host_jx.coulomb_ip1` | PySCF `grad.rhf.get_jk` `vj` (golden `jx1_*`) | H₂O, `data/basis` STO-3G; cc-pVDZ `slow` (`test_kernel_jx.py`) | 1.37e-15 of `max\|J1\|` (STO-3G), gate 1e-12 |
| `host_jx.coulomb_derivative_matrix` | PySCF `hessian.rhf` `h1ao` Coulomb part (golden `jx_*`) | H₂O, STO-3G only | 4.32e-15 of `max\|J^X\|`, gate 1e-12 |
| the goldens, the host | `SUM_X J^X = 0`; each `J^X` symmetric | both goldens; an s, p pair | gate 1e-13 |
| `jx`, interpreted | `coulomb_ip1(pairs=)` per class, random symmetric `D` | every ordered class through (ff) on `test_kernel_gjab.py`'s s, p, d, f basis (L + 1 > 4 `slow`) | 0.73 eps of the absolute twin (`jx_00`; 0.10–0.62 elsewhere), gate 16 eps |
| the chain, interpreted | the host (`J1` and `J^X`); PySCF | an s, p pair; H₂O/STO-3G (`slow`, 59 s) | `J1` 0.39 eps of the absolute twin, `J^X` 1.8 eps of `max\|J^X\|` (gate 64 eps); against PySCF `J1` 1.07e-15, `J^X` 4.32e-15 |
| nvcc / hipcc compile | #92's warning filter | all 32 units through (ff) (`test_emit_jx.py`) | **pass** (nvcc 13.0 sm_86, hipcc gfx1200; 2026-10-30) |
| on a card | the host per class through (dd); PySCF on H₂O | `test_jx_device.py` | **not run**: no card in that session |

### What is not done

- Nothing has run on a card or been measured.
- The whole `J^X` over cc-pVDZ is not held to its golden by any test:
  the host's ket half there is nine J builds over ~10k derivative kets in
  pure Python, hours. `J1` over cc-pVDZ is (`slow`), and the `jx_` golden
  is tracked for a faster witness.
- No unrolled form, so no rule choosing between forms.
- No one-electron half (#687), so no assembled `F^x`; no CPHF solve.
- Cartesian only.
