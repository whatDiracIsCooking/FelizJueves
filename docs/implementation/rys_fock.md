# The fused Fock build on Rys quadrature: `Target.F` on `Engine.RYS`

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

## The rule

The measure is `(1/2) x^(-1/2) exp(-T x) dx` on `[0, 1]`, so the `N`-point
rule has `SUM w x^m = F_m(T)` for `m <= 2N - 1`. `N_MAX = 5` reaches
`(dd|dd)` (L = 8).

- `T >= T_SWITCH = 80`: generalised Gauss–Laguerre with `alpha = -1/2`
  on constants, `x = s / T`, `w = v / (2 sqrt(T))`. The dropped tail is
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

The 2D VRR is #468's settled form, every sign of which its spike flipped
alone and caught:

    B00 = x inv2pq,  B10 = inv2p (1 - x q/(p+q)),  B01 = inv2q (1 - x p/(p+q))
    c00_d = PA_d + x WP_d,   d00_d = QC_d + x WQ_d
    I(i+1, 0) = c00 I(i, 0) + i B10 I(i-1, 0)
    I(i, k+1) = d00 I(i, k) + k B01 I(i, k-1) + i B00 I(i-1, k)

`WP`, `WQ` are `kernel_f`'s derived signs (`W - P = -(q/(p+q)) PQ`). #468's
two transfer steps are **not** run per node: the HGP boundary `(e0|f0)`,
`la <= |e| <= la + lb`, `lc <= |f| <= lc + ld`, is exactly `I_x I_y I_z` of
the VRR, and the transfers run once per contracted quartet as eq 28. The
weight `S w_n` (`S = U_P U_Q (2/sqrt(pi)) theta`, the HGP fundamental's
scale) seeds the z axis, so each boundary value is two multiplies per root,
added straight into `kernel_f`'s accumulators: the root sum and the
contraction are one sum.

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
- **No lever** (`f.rys` admits none), no split `J` / `K` on Rys, no
  gradient.
- The bra-transfer rounding #468 measured (`P` near `B`) is now eq 28's,
  shared with `f_`; choosing the VRR centre by `l` is still open.
