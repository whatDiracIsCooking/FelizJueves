# Rys quadrature: the fused `F` (`Target.F`) and the split `J` and `K` (`Target.J`, `Target.K`) on `Engine.RYS`

**Status: implemented and interpreted; emitted for both vendors; NOT
compiled by nvcc or hipcc and NOT run on either card.** The session that
built it had no card and neither device compiler: every unit passed clang
18's CUDA front end (device side, `-fsyntax-only -Wall -Wextra`, a runtime
stub) with no diagnostic, which is a proxy and nothing more. The kernel
tier is where a number on a card is made, and none has been. **Nothing is
measured or scored.** Energy order, one density, Cartesian, every shell
`l <= 2` (the fused `F` ladder, through `(dd|dd)`).

This is the derived note for engine #3, the Rys spike of #468
(`experimental/rys/`) brought into `src/` as a GPU kernel emitter. It is a
sibling of `hgp_fock.md` and leans on it: the fused scheme, the digestion's
derived weights, the pair stream and the six-block screen are that note's.
The second Rys kernel, **the split Coulomb `J`** (`rj_` units, `kernel_hj`'s
nest with the same Rys VRR), has its own section, "The split `J` on Rys",
with the same status: interpreted and emitted, not compiled by a vendor
compiler, not run on a card, nothing measured. So has the third, **the
split exchange `K`** (`rk_` units, `kernel_hk`'s nest with the same Rys
VRR), "The split `K` on Rys", with the same status.

## What it is

`eri/fock/kernel_frys.py`'s `build_frys_kernel(cls)` builds `fr_{la}{lb}{lc}{ld}`:
`kernel_f`'s fused Fock kernel of `[Fock]` §3.2 (one block per bra shell
pair, one thread per contracted quartet, the `K_CD x K_AB` contraction
inside, the six-block screen that skips, the derived 8-fold weights, six
thread-private buffers, `F <- F + F^T` on the host) with **one thing
replaced: the vertical recurrence**. Where `f_` runs `[Fock]` eqs 25–26 off
one Boys evaluation per primitive quartet, `fr_` runs the `N = L // 2 + 1`
point Rys rule and, per node, the per-axis 2D recurrence. Their products
are the same contracted boundary `[e0|f0]^(0)` the HGP HRR (eq 28) reads,
so the HRR, tap, digestion, staging, screen, parameters and launch are
`kernel_f`'s own, through its seams (`eri/fock/levers/seams.py`, #368).
The hooks moved are `ket_prim`, `quartet_scalars`, `vrr` and `accumulate`;
every other hook is the default.

So an `fr_` unit is a **drop-in** for the `f_` unit of the same class: the
same signature, grid and output, the same `launch_f` plan, `pack`ed
records, `density_blocks` and `close`. `test_kernel_frys.py` holds the two
kernels' IR equal outside the `K_CD` loop.

| Piece | Where |
|---|---|
| the key | `Spec(..., Target.F, engine=Engine.RYS)`, `coulomb/d0/(la,lb\|lc,ld)/b1/f.rys/literal`; RYS on any other target is refused (`spec._NO_RYS`); the `f.rys` lever slot admits nothing |
| the rule | `rys/roots.py`: `roots_weights(n, T)` (host fp64) and `device_text()` (`fj_rys<N>`), one algorithm, one set of constants |
| its ground truth | `rys/exact.py`: moments in `decimal`, Gautschi's Chebyshev algorithm, Newton (#468's exact route) |
| the 2D VRR | `rys/recurrence.py`: `root_loop`, the per-root tables and the boundary `+=` |
| the IR statement | `ir.Rys(nodes, weights, n, t)`, fp64 only, two LOCAL F64 arrays of extent `n`; injected into the interpreter as `rys=`; spelled `fj_rys<n>(T, x, w);` by `emit/cuda.py` and `emit/hip.py`, which carry the routine; `emit/gpu.py` refuses it |
| the kernel | `eri/fock/kernel_frys.py` |
| the generator | `python -m felizjueves.emit --fock-l LMAX --rys` (or `--fock-class ... --rys`); never in the bare default nor in `KERNEL_GEN_CMD` |

## Sources

Citations are by `docs/references.md` key. **`[DRK]` (entry 49) and `[LRL]`
(entry 50) are closed access and were not read for this note**, so no
equation number is cited against either. Each equation below names the
openly readable source it was checked against, or says it is derived here
in the lines that follow it. The checked sources are three:

- `[GPU4PySCF]` (entry 26), arXiv v1's numbering: eq. 1, the ERI as
  `SUM_n w_n I_x(t_n) I_y(t_n) I_z(t_n)`, and eq. 3, `N = floor((a+b+c+d)/2)
  + 1`. It cites `[DRK]` and Rys–Dupuis–King 1983 (entry 42) for the
  recurrences, and prints none of them.
- `[Libcint]` (entry 51), its **source** at commit `4909749`: `src/g2e.c`'s
  `CINTg0_2e` forms the coefficients, and `CINTg0_2e_2d` runs the
  recurrences. This is the oracle that the spike (#468) and the
  interpreted checks below are held to, through PySCF. Code has no
  equation numbers, so the functions are named instead.
- `[JEngine, v2, eqs. 13, 18]` for the Boys function, as `j_engine.md`
  cites it.

`[DRK]` remains the origin of the recurrences, and `[LRL]` is their
re-derivation with a reduced operation count. Reading either one against
the forms below, and recording its equation numbers, is still open.

## The rule

The Boys function is `F_m(T) = ∫₀¹ t^{2m} exp(-T t²) dt` (`[JEngine, v2,
eqs. 13, 18]`). The change of variable `x = t²` (derived here: `dt = dx /
(2 sqrt(x))`) gives `F_m(T) = ∫₀¹ x^m (1/2) x^(-1/2) exp(-T x) dx`, so the
measure is `(1/2) x^(-1/2) exp(-T x) dx` on `[0, 1]` and the `N`-point rule
has `SUM w x^m = F_m(T)` for `m <= 2N - 1`. The node `x` is `t²`, the
square of the Rys root `t`. libcint carries `u = t²/(1 - t²)` instead
(`CINTg0_2e`'s comment `u(irys) = t2/(1-t2)`), so `x = u/(1 + u)`.
`N = L // 2 + 1` is `[GPU4PySCF, eq. 3]`, and `N_MAX = 5` reaches
`(dd|dd)` (L = 8).

- `T >= T_SWITCH = 80`: generalised Gauss–Laguerre with `alpha = -1/2`
  on constants, `x = s / T`, `w = v / (2 sqrt(T))`. This is derived here,
  with no source cited: extend the upper limit from `1` to `infinity` and
  substitute `s = T x`, which gives `F_m(T) ~ T^(-m - 1/2) ∫₀^∞ s^m (1/2)
  s^(-1/2) e^(-s) ds`. That is the `alpha = -1/2` Laguerre weight, with the
  rule's `x = s/T` and its weights scaled by `1/(2 sqrt(T))`. The dropped
  tail is
  `O(e^-T T^(2N - 3/2))`: at `N = 5` it is 5.6e-11 in the weights at
  `T = 50` (#468's switch, which held for `N <= 3` only) and below rounding
  at 80.
- `T < 80`: the measure in `t = sqrt(x)` on the positive half of an
  88-point Gauss–Legendre rule (constants computed once in `decimal`); the
  Stieltjes procedure for the monic three-term coefficients, re-running
  each `p_k(y_j)` from the coefficients so far so that `mu[44]` is the one
  array; the nodes by Sturm bisection on `[0, 1]` (64 halvings of the LDL^T
  pivot count); the weights as Christoffel numbers. No eigensolver, no
  library, every loop a fixed count. 72 points miss by 1e-12 at `N = 5`
  just under the switch and 80 is the edge, so 88 keeps margin. A Newton
  polish after the bisection was tried and bought nothing.

| Check (`test/test_rys_roots.py`) | Worst | Gate |
|---|---|---|
| nodes and weights against the 40-digit route, `N` 1..5, 30 values of `T` from 0 through `1e-300` to `1e6`, across the switch | 3.4e-15 relative | 1e-14 |
| moments `SUM w x^m` against `F_m(T)` in `decimal` | — | 1e-14 relative |
| `device_text()` compiled as host C++ (g++, `-ffp-contract=off`) on the same grid, against `roots_weights` | **bitwise** | `==` |

The last row is the routine's arithmetic on a CPU, not a card's: the
host's `exp` and `sqrt` are the CPU's. On a card those two are the vendor's,
so `fj_rys` is expected to differ from the host rule by rounding, which is
the kernel tier's to measure.

## The recurrence

The 2D VRR is #468's settled form. The spike flipped every sign in it
alone, and each flip was caught:

    B00 = x inv2pq,  B10 = inv2p (1 - x q/(p+q)),  B01 = inv2q (1 - x p/(p+q))
    c00_d = PA_d + x WP_d,   d00_d = QC_d + x WQ_d
    I(i+1, 0) = c00 I(i, 0) + i B10 I(i-1, 0)
    I(i, k+1) = d00 I(i, k) + k B01 I(i, k-1) + i B00 I(i-1, k)

The forms originate in `[DRK]`, unread here (see "Sources"). Each one is
checked against `[Libcint]`'s source, with `p`, `q` the exponent sums
(libcint's `aij`, `akl`) and `PQ = P - Q` (its `xij_kl`):

| Here (`rys/recurrence.py`) | libcint (`g2e.c`) | libcint's form, `u = x/(1 - x)`, `u2 = rho u` | Reduces to ours |
|---|---|---|---|
| `B00` | `b00` | `tmp5 = u2 tmp4`, `tmp4 = 1/(2 (u2 (p+q) + p q))` | `x/(2(p+q))` |
| `B10` | `b10` | `tmp5 + tmp4 akl` | `(1 - x q/(p+q))/(2p)` |
| `B01` | `b01` | `tmp5 + tmp4 aij` | `(1 - x p/(p+q))/(2q)` |
| `c00_d` | `c00x/y/z` | `rijrx - 2 tmp5 akl xij_kl` | `PA_d - x (q/(p+q)) PQ_d` |
| `d00_d` | `c0px/y/z` | `rklrx + 2 tmp5 aij xij_kl` | `QC_d + x (p/(p+q)) PQ_d` |
| bra step `I(i+1, 0)` | `gx(irys,0,n+1)` | `c00 gx(0,n) + n b10 gx(0,n-1)` | the same |
| ket step `I(i, k+1)`, `i <= 1` | `gx(irys,m+1,n)`, `n <= 1` | `c0p gx(m,n) + m b01 gx(m-1,n) + n b00 gx(m,n-1)` | the same |
| ket step `I(i, k+1)`, `i >= 2` | (not formed this way) | libcint steps the bra instead: `gx(m,n+1) = c00 gx(m,n) + n b10 gx(m,n-1) + m b00 gx(m-1,n)` | the mirror relation |

libcint's `gx(irys, m, n)` is ket index `m` and bra index `n`. The reduction
is algebra (`u2 (p+q) + pq = pq/(1 - x)`, so `tmp4 = (1 - x)/(2pq)` and
`tmp5 = x/(2(p+q))`). It was also evaluated at 20,000 random `(p, q, x, PA,
QC, PQ)`, and the two agree to 8.1e-16 relative. For `i >= 2`, libcint uses
the bra-side mirror of our ket step, the same three-term relation with the
roles exchanged. So our ket step there is held to libcint by the
interpreted checks below, not by this table.

There are three differences in arrangement and none in value. First,
libcint's ket name `c0p` is our `d00`; `rys/recurrence.py`'s docstring
spells it `D00`. Second, libcint builds on the higher-`l` centre of each
pair (`g2e.c`: `ibase = li_ceil > lj_ceil`, likewise `kbase`), whereas
here the build is always on `A` and `C`. That choice is the open item
under "What is not done" (#468's `P` near `B` rounding). Third, the order
of the table fill differs, as the last row shows.

**No misprint is recorded**, because no printed form of these recurrences
was read. The one sign they share with a printed paper is the `W - P` /
`W - Q` displacement. `hgp_fock.md` derives that sign and records `[Fock]`
eq 26's printed `W - Q` as wrong. Here it is held by the libcint rows above
and by #468's flip test.

`WP`, `WQ` are `kernel_f`'s derived signs (`W - P = -(q/(p+q)) PQ`). #468's
two transfer steps are **not** run per node: the HGP boundary `(e0|f0)`,
`la <= |e| <= la + lb`, `lc <= |f| <= lc + ld`, is exactly `I_x I_y I_z` of
the VRR, and the transfers run once per contracted quartet as eq 28. The
weight `S w_n` (`S = U_P U_Q (2/sqrt(pi)) theta`, the HGP fundamental's
scale) seeds the z axis, so each boundary value is two multiplies per root,
added straight into `kernel_f`'s accumulators: the root sum and the
contraction are one sum.

**The prefactor, reconciled.** `S = U_P U_Q (2/sqrt(pi)) theta` is the HGP
seed's scale, with `[00|00]^(m) = S F_m(T)`, `U_P = (pi/p)^(3/2) K_ab` and
`theta = sqrt(rho)`. `hgp/recurrence.py` derives it, and it equals `[Fock]`
§3.1's `R0_m` prefactor `2 pi^(5/2) K_ab K_cd / (p q sqrt(p+q))`. Because
`SUM_n w_n = F_0(T)` (the rule at `m = 0`, above), the `(ss|ss)` case is
`S SUM w_n = S F_0(T)`, which is HGP's own fundamental, so the two engines
share one scale by construction. libcint uses the same scale. `CINTg0_2e`
multiplies each weight by `fac1 = sqrt(a0 / a1^3) fac[0]`, with `a1 = pq`,
`a0 = rho`, and `fac[0] = common_factor K_ab K_cd` (`cint2e.c`'s `expijkl`
times contraction coefficients), where `common_factor = 2 pi^3 / sqrt(pi)`
times libcint's s/p normalisation. That makes `fac1 = (pi/p)^(3/2) (pi/q)^(3/2)
(2/sqrt(pi)) sqrt(rho) K_ab K_cd = S`, up to the contraction coefficients
and the normalisation, which this note's records carry separately. The
same 20,000-point evaluation agrees to rounding. libcint folds the weight
into the z seed too: its `CINTg0_2e_2d` seeds `gx = gy = 1` and leaves
`gz` holding the scaled weight. `[GPU4PySCF, eq. 1]` writes the same
product, `SUM_n w_n I_x I_y I_z`, with no prefactor shown.

## How it is checked (interpreted)

`test/test_kernel_frys.py`, the rule injected as `rys=roots_weights`:

| Check | Worst | Gate |
|---|---|---|
| the ERI tap, all 21 classes, one contracted quartet of 2-primitive shells on four centres, against `eri/eri.py` (MMD) | 6.3e-15 of the MMD absolute scale | `1e-13` (`test_kernel_f.TOL`) |
| the same against the HGP host path (#279's pinned VRR/HRR kernels) | 7.6e-15 | `1e-13` |
| the same with the ket 20 bohr away, every `T` past the switch (recorded) | — | `1e-13` |
| one orbit's closed deposit against `eri.fock.fock`'s ordered sweep, all 21 classes; a diagonal `(pp\|pp)` launch with every degeneracy | — | `1e-13` of `scale_G` |
| whole `G` of H₂/STO-3G and H₂O/STO-3G (`slow`) from `launch_f`'s plan, against the sweep | 3.1e-16, 5.7e-16 of scale | `1e-13` |
| the same against libcint's `vj - vk/2` | 2.2e-15, 1.5e-15 of scale | `1e-12` |
| the IR outside the `K_CD` loop `==` `kernel_f`'s, all 21 classes | `==` | — |

Emission: 21 goldens per backend (`test/golden/{cuda,hip}/fr_*`), the two
over 512 KiB, `(pd|dd)` and `(dd|dd)`, pinned in
`test/golden/frys_digests.json`. `fr_2222` is 1.61 MB of text against
`f_2222`'s 1.99 MB. The nvcc / hipcc compile cases
(`test_emit_frys.py`, `cuda` / `hip`) reuse `test_emit_fock.py`'s, and have
**not run**.

## What is not done

- **Nothing on a card, and no vendor compile.** The next step is the
  kernel tier: `test_emit_frys.py`'s compile cases on both compilers, then a
  device test on `test/fock_driver.py`'s harness (the signature is
  `f_`'s, so the driver should take the `fr_` unit unchanged), which would
  also be the first measurement of `fj_rys` against the host rule on a
  card.
- **No `gpu` backend spelling**: `emit/gpu.py` refuses an `ir.Rys`; the
  routine would need a header of its own, as `eval_boys.h` is.
- **Nothing measured or scored**, so nothing says Rys is faster or slower
  than HGP here. The Rys rule costs 44 `exp`s, a Stieltjes sweep and
  `64 N^2` Sturm steps per primitive quartet, against HGP's one Boys
  evaluation; it is written for correctness first.
- **Above `(dd|dd)`**: `rys.roots` stops at `N = 5`; `N` up to 9 for
  `(gg|gg)` needs `T_SWITCH` and `M_LEGENDRE` re-measured.
- **No lever** (`f.rys` admits none), no split `K` on Rys (the split `J`
  is below), no gradient.
- The bra-transfer rounding #468 measured (`P` near `B`) is now eq 28's,
  shared with `f_`; choosing the VRR centre by `l` is still open (libcint
  does choose it: "The recurrence").
- **`[DRK]` and `[LRL]` unread** (closed access): their equation numbers
  for the coefficients and the two recurrences, and any misprint in them,
  are not recorded ("Sources"). The forms are held to libcint's source and
  to the interpreted checks instead.

## The split `J` on Rys: `Target.J` on `Engine.RYS`

**Status: implemented and interpreted; emitted for both vendors; NOT
compiled by nvcc or hipcc and NOT run on either card.** As for `fr_`: the
session that built it had no card and neither device compiler, and every
unit through `(dd|dd)` passed clang 18's CUDA front end (device side,
`-fsyntax-only -Wall -Wextra`, a runtime stub) with no diagnostic, a proxy.
**Nothing is measured or scored.**

`eri/hjk/kernel_rj.py`'s `build_rj_kernel(spec)` is the HGP split `J` of
milestone 14 (`kernel_hj`, `hgp_fock.md`, "The split cell": every ordered
shell quartet once, `J[a][b] += D[c][d] (ab|cd)` by atomic add, one block
per bra shell pair, one thread per contracted quartet, the `K_AB x K_CD`
contraction inside, a one-block screen that skips) with the primitive
nest's engine part replaced by this note's rule and 2D VRR. It is to
`kernel_hj` what `kernel_frys` is to `kernel_f`, through a seam
`kernel_hj` now exposes:

| Piece | Where |
|---|---|
| the seam | `kernel_hj.Nest(need, body, note)` and `split_j_kernel(spec, nest_of, ...)`: the `K_CD` loop's body after the ket record's reads, the record scalars it reads (`PA_*`, `QC_*`) and the opening `Note`; `build_hj_kernel` is `split_j_kernel(spec, hgp_nest, ...)`, every `hj_` golden byte-identical |
| the key | `Spec(..., Target.J, engine=Engine.RYS)`, `coulomb/d0/(la,lb\|lc,ld)/b1/j.rys/{literal,ordered}`; `spec._NO_RYS` now refuses every target but `F` and `J` |
| the class modes | HGP J's own: canonical (`rj_{rep}`, one unit per 8-fold orbit, a runtime `perm`) and the `ordered` lever (`rj_o_{class}`), which is legal on the new slot `j.rys` and nowhere else new; the structure (built class, orientations, parameters) is read off the HGP twin (`kernel_rj.hgp_twin`) |
| the nest | bra outer, ket inner, as `kernel_hj`; per ket primitive pair `rys.recurrence.quartet_rule` (the quartet scalars, `T`, `S`, the rule; shared with `kernel_frys`, whose text is unchanged) and `root_loop` into the accumulators |
| the walk | `walk_hjk(..., build=)`: `launch_hjk`'s plan with each launch's kernel built by the caller from its HGP key, so one plan and one walker serve both engines |
| the generator | `python -m felizjueves.emit --hgp-j-l LMAX --rys` (LMAX 0..2) or `--hgp-class L0 L1 L2 L3 --rys` (`L <= 8`), `--ordered` for the ordered mode (the class path writes the `rk_` unit too since the split `K` on Rys, below); never in the bare default nor in `KERNEL_GEN_CMD` |
| the store | a `j.rys` row is refused, as `j.hgp`'s (`measured_store.HGP_JK_REFUSED`: any non-native engine) |

So an `rj_` unit is a **drop-in** for the `hj_` unit of the same class and
mode: the same parameters (signature text equal, `test_emit_rj.py`), grid
and output. `test_kernel_rj.py` holds the two kernels' IR equal outside
the `K_CD` loop, in both modes, for every class through `(dd|dd)`.

### How it is checked (interpreted)

`test/test_kernel_rj.py`, the rule injected as `rys=roots_weights`, on
test_kernel_hj.py's bases (two shells per repeated `l`, so `a = b` pairs
are met, two primitives each, distinct centres), records and witnesses.
Every ordered class through `L = 4` (all 70, shells up to `l = 4`) is fast;
the 31 classes with `5 <= L <= 8` and every shell `l <= 2` are `slow`.
`TOL = 1e-12` is test_hgp_eri.py's per-ERI cross-engine bound, none new.

| Check | Worst | Gate |
|---|---|---|
| the ERI tap against `eri/eri.py` (MMD), element by element, every class through `(dd\|dd)`, both modes | 5.6e-16 of `s_hgp + s_mmd` (`(pp\|pp)`) | `TOL (s_hgp + s_mmd)` |
| the same against `eri/hgp_drive.py`, per block (see below) | — | `TOL max_block(s_hgp + s_mmd)` |
| the ket 20 bohr away, every `T` past the switch, `(ss\|pp)` and `(pp\|dd)` | — | the same two |
| `J` against `eri/j_coulomb/j.py`'s sweep restricted to the class (MMD), both modes, every class through `(dd\|dd)`; every element off the class's blocks exactly 0 | 4.7e-16 of `s_hgp + s_mmd` | `TOL (s_hgp + s_mmd)` |
| canonical against ordered, every class through `L = 4` | — | `TOL (s_c + s_o)` |
| whole `J` of H₂/STO-3G (and H₂O/STO-3G, `slow`), `launch_hjk`'s plan walked with the Rys kernels, both modes, against the MMD sweep / libcint's `vj` | 5.6e-16 / 2.6e-15 of `S` (H₂O) | `1e-12 S` / `1e-12 S` (test_hjk_build.py's) |

The ERI tap is held to HGP per block, not per element, for a reason found
here: where the four centres coincide (`(pp|pp)` on one shell)
`hgp_drive` forms `W - P` as a difference, so its structural zeros come out
as rounding (4.6e-34, about half its own absolute-value twin); MMD and Rys
leave them exactly 0. No element-wise bound on either twin covers that, so
the element-wise check is against MMD.

Emission (`test_emit_rj.py`): 63 goldens per backend (13 canonical, 50
ordered, every class `L <= 4` with each shell `l <= 2`, `test/golden/
{cuda,hip}/rj_*`, the largest 60 KB) and the 8 canonical units above `L = 4`
through `(dd|dd)` pinned by digest in `test/golden/rj_digests.json`
(`slow`); `rj_2222` is 974752 B of text. The nvcc / hipcc compile cases
(`cuda` / `hip`) reuse `test_emit_hjk.py`'s and have **not run**.

### What is not done

- **Nothing on a card, and no vendor compile.** Next: `test_emit_rj.py`'s
  compile cases on both compilers, then a device test on
  `test/hjk_driver.py`'s harness (the signature is `hj_`'s, so it should
  take the `rj_` unit unchanged).
- **The split `K` on Rys** is built since: "The split `K` on Rys", below.
  No `gpu` backend spelling (`ir.Rys` is refused there).
- **Nothing measured**: no word on Rys against HGP here.
- **Above `(dd|dd)`**: the rule stops at `N = 5`, so `L <= 8`; `--hgp-j-l`
  with `--rys` refuses LMAX above 2.

## The split `K` on Rys: `Target.K` on `Engine.RYS`

**Status: implemented and interpreted; emitted for both vendors; NOT
compiled by nvcc or hipcc and NOT run on either card.** As for `rj_`: the
session that built it had no card and neither device compiler. All 102 cuda
units through `(dd|dd)` (21 canonical, 81 ordered) passed clang 18's CUDA
front end (device side, `-fsyntax-only -Wall -Wextra`, a runtime stub) with
no diagnostic, a proxy; the hip units differ from them in the prologue
alone and were not put through a HIP front end. **Nothing is measured or
scored.**

`eri/hjk/kernel_rk.py`'s `build_rk_kernel(spec)` is the HGP split `K` of
milestone 14 (`kernel_hk`: every ordered shell quartet once, `K[a][b] +=
D[c][d] (ac|bd)` by atomic add, `am` K's Hermite pairing `(la, lc | lb,
ld)`) with the primitive nest's engine part replaced by this note's rule
and 2D VRR. It is to `kernel_hk` what `kernel_rj` is to `kernel_hj`:

| Piece | Where |
|---|---|
| the seam | `kernel_hk.split_k_kernel(spec, nest_of, ...)`, taking `kernel_hj`'s `Nest`; `build_hk_kernel` is `split_k_kernel(spec, kernel_hk.hgp_nest, ...)`, `hgp_nest` being `kernel_hj.hgp_nest` under K's opening note, so every `hk_` golden is byte-identical |
| the nest | `kernel_rj.rys_nest` under K's opening note, statement for statement: the recurrence of a built class `(x0 x1 \| x2 x3)` does not know whether the deposit is `J`'s or `K`'s (`test_kernel_rk.py` holds the `rk_o_` nest `==` the `rj_o_` nest of the same built class) |
| the key | `Spec(..., Target.K, engine=Engine.RYS)`, `coulomb/d0/(la,lc\|lb,ld)/b1/k.rys/{literal,ordered}`; `spec._NO_RYS` now refuses every target but `F`, `J`, `K`, `S`, `T`, `V` |
| the class modes | HGP K's own: canonical (`rk_{rep}`, a runtime `perm`) and the `ordered` lever (`rk_o_{class}`), legal on the new slot `k.rys`; the structure read off the HGP twin (`kernel_rk.hgp_twin`) |
| the walk | `walk_hjk(..., build=)` with `launch_hjk`'s `"k"` plan, unchanged |
| the generator | `--rys` now takes `--hgp-k-l LMAX` (0..2) too, and `--hgp-class ... --rys` writes both the `rj_` and the `rk_` unit; a request past `(dd\|dd)` is a 2; never in the bare default nor in `KERNEL_GEN_CMD` |
| the store | a `k.rys` row is refused (`measured_store.HGP_JK_REFUSED`, any non-native engine) |

So an `rk_` unit is a **drop-in** for the `hk_` unit of the same class and
mode: the same parameters (signature text equal, `test_emit_rk.py`), grid
and output; `test_kernel_rk.py` holds the IR equal outside the `K_CD` loop,
both modes, every class through `(dd|dd)`.

### How it is checked (interpreted)

`test/test_kernel_rk.py`, the rule injected as `rys=roots_weights`, on
test_kernel_hk.py's launches, records and witnesses (its `launch` takes the
drop-in as `kernel=` / `rys=`, as test_kernel_hj.py's does). Every ordered
K class through `L = 4` (all 70) is fast; the 31 with `5 <= L <= 8` and
every shell `l <= 2` are `slow`. `TOL = 1e-12` is test_hgp_eri.py's, as
for `rj_`.

| Check | Worst | Gate |
|---|---|---|
| the ERI tap against `eri/eri.py` (MMD), element by element, every class through `(dd\|dd)`, both modes | 5.6e-16 of `s_hgp + s_mmd` (`(pp\|pp)`) | `TOL (s_hgp + s_mmd)` |
| the same against `eri/hgp_drive.py`, per block (`rj_`'s reason) | — | `TOL max_block(s_hgp + s_mmd)` |
| the built ket 20 bohr away, every `T` past the switch, `(ss\|pp)` and `(pp\|dd)` | — | the same two |
| `K` against `eri/k_exchange/k.py`'s `k_quartet` over the class's ordered quadruples (MMD), both modes, every class through `(dd\|dd)`; every element off the class's blocks exactly 0 | 3.2e-16 of `s_hgp + s_mmd` | `TOL (s_hgp + s_mmd)` |
| canonical against ordered, every class through `L = 4` | — | `TOL (s_c + s_o)` |
| a wrong density block (`(2, 3)`, `(0, 1)`) and the AO-order trap, both modes | fail, as `hk_`'s | excess `> 1e3` |
| whole `K` of H₂/STO-3G (and H₂O/STO-3G, `slow`), `launch_hjk`'s `"k"` plan walked with the Rys kernels, both modes, against the MMD sweep / libcint's `vk` | 3.8e-16 / 1.7e-15 of `S` | `1e-12 S` / `1e-12 S` |

Emission (`test_emit_rk.py`): 63 goldens per backend (13 canonical, 50
ordered, `test/golden/{cuda,hip}/rk_*`, the largest 91 KB) and the 8
canonical units above `L = 4` pinned by digest in
`test/golden/rk_digests.json` (`slow`; `rk_2222` is 974,664 B). The nvcc /
hipcc compile cases reuse `test_emit_hjk.py`'s and have **not run**.

### What is not done

- **Nothing on a card, and no vendor compile**: `test_emit_rk.py`'s compile
  cases, then a device test on `test/hjk_driver.py`'s harness (the
  signature is `hk_`'s).
- **Nothing measured**, no `gpu` backend spelling, nothing above
  `(dd|dd)`.
