# experimental/rys — the Rys 2D recurrence and its roots, settled on CPU

A standalone spike (#468). Nothing in `src/felizjueves/` imports it, nothing in
`test/` collects it (`testpaths = ["test"]`), and no tier runs it. The oracle is
libcint through PySCF, not this repo's engines.

```bash
uv sync --group oracle
uv run --group oracle python experimental/rys/check.py    # ~40 s, exit 0 = PASS
uv run --group oracle python experimental/rys/check_j.py  # ~45 s, the J build
```

| File | What it is |
|---|---|
| `roots.py` | `roots_weights(n, T)`, the fp64 Rys rule for `n` in 1..3, `T` in `[0, 1e6]`; and `roots_weights_exact(n, T, digits)`, an independent route in `decimal` used only to measure it. |
| `rys2d.py` | The three 2D tensors, `[ab\|cd]` for one primitive quartet (`prim_quartet`) and the contracted block (`contracted_block`), generic in `(la, lb, lc, ld)`; `prim_quartet_exact`, the same quadrature at 50 digits; `FORM`, the signs as multipliers (all 1); `shells_of(mol)`, a PySCF `Mole`'s shells with libcint's normalisation (s and p only, refusing above). |
| `check.py` | The four tables below; exits nonzero on a miss. |
| `jbuild.py` | The Coulomb matrix on these ERIs: `coulomb_matrix(shells, D, tau=0, eri=)` → `JBuild(J, n_quartets, n_skipped, bound)`, one sweep over the unique shell quartets with an optional Schwarz screen; `coulomb_energy`, `schwarz`, `ao_offsets`. |
| `check_j.py` | The three J tables in "The J build"; exits nonzero on a miss. |

## What was settled

**The form in the issue is correct as written: no sign or factor changed.**
With `x` the node, `AB = A − B`, `CD = C − D`, `W = (ζP + ηQ)/(ζ + η)`:

| Quantity | Settled form | Flipped alone, caught by |
|---|---|---|
| `B00` | `x / (2(ζ + η))` | 9 / 16 classes (any `p` on both sides) |
| `B10` | `(1 − x η/(ζ + η)) / (2ζ)` | 4 / 16 (`(pp\|··)`) |
| `B01` | `(1 − x ζ/(ζ + η)) / (2η)` | 4 / 16 (`(··\|pp)`) |
| `C00` | `(P − A) + x (W − P)` | 12 / 16 each term |
| `D00` | `(Q − C) + x (W − Q)` | 12 / 16 each term |
| bra VRR | `I(i+1,0,k,0) = C00 I(i,0,k,0) + i B10 I(i−1,0,k,0) + k B00 I(i,0,k−1,0)` | |
| ket VRR | `I(i,0,k+1,0) = D00 I(i,0,k,0) + k B01 I(i,0,k−1,0) + i B00 I(i−1,0,k,0)` | |
| bra transfer | `I(i,j+1,k,l) = I(i+1,j,k,l) + (A − B) I(i,j,k,l)` | 8 / 16 (`lb = 1` only) |
| ket transfer | `I(i,j,k,l+1) = I(i,j,k+1,l) + (C − D) I(i,j,k,l)` | 8 / 16 (`ld = 1` only) |
| seed | `I(0,0,0,0) = 1` | |
| scale on `W_n` | `(2/√π) √ρ · (π/ζ)^{3/2} K_AB · (π/η)^{3/2} K_CD`, contraction coefficients outside | |

"Flipped alone" is `check.py`'s table 4: each sign is negated in turn (`FORM`)
and the 16 classes are evaluated on one uncontracted quartet on three centres
with `A ≠ B` and `C ≠ D`. Every flip is caught, and only the settled form
agrees. The transfer signs are why all 16 **ordered** classes matter: `AB` is
caught only by `(sp|··)` and `(pp|··)`, so a set with `la ≥ lb` sees it only
through `(pp|··)`, and a quartet with `A = B` sees neither `AB` nor `P − A`
at all (the first version of table 4 picked such quartets and missed both).

Normalisation is libcint's by construction: exponents, centres and coefficients
come from the `Mole` (`_libcint_ctr_coeff`, which carries the radial
normalisation), times libcint's s/p common factor (`1/(2√π)` for s, `√(3/4π)`
for p, `CINTcommon_fac_sp`), compared Cartesian to Cartesian (`int2e_cart`).

## The roots

**Shipping route (fp64).** Below `T_SWITCH = 50`: `x = t²` turns the measure
into `exp(−T t²) dt` on `[0, 1]`; discretise with the positive half of an
80-point Gauss–Legendre rule, run the Stieltjes procedure for the recurrence
coefficients, and Golub–Welsch. At and above `T_SWITCH`: `x = s/T` and drop
`[T, ∞)`, leaving generalised Gauss–Laguerre with `α = −1/2`
(`scipy.special.roots_genlaguerre`), `x_n = s_n/T`, `W_n = w_n/(2√T)`.

**Large `T` does need its own branch at `N ≤ 3`.** A fixed Legendre
discretisation cannot resolve a Gaussian of width `1/√T` once `T` is large;
48 points already miss by 1e-10 at `T = 50`, 56 and above do not. The
Laguerre branch's dropped tail is `O(e^{−T} T^{2N−3/2})` relative, about 2e-16
of `F_5` at `T = 50`, so the switch is placed there; the moment error across
it (`nextafter(50, 0)`, 50, `nextafter(50, ∞)`) stays under 8.1e-16.

**What fixed the small-`T` branch.** The first version, on numpy's and then
scipy's Legendre tables, measured up to 4.8e-14 in the nodes and 1.5e-13 in
the weights, erratic in the point count (not monotone in M, and present at
`T = 0`, where the rule is exact in exact arithmetic). Those tables' individual
weights carry ~1e-14 relative errors. Computing the Legendre table once in
`decimal` (Newton at 40 digits, rounded once) removed all of it.

**Exact route.** The moments `F_m(T)` at 70 digits (the positive Taylor series
below `T = 200`; above it `F_0 = √(π/T)/2`, `erfc(√T) < e^{−200}` being below
any precision used, and upward recurrence), Gautschi's Chebyshev algorithm,
Newton on the orthogonal polynomial, Christoffel numbers. Run at 50 and at 70
digits it agrees with itself to 3.4e-67.

## The numbers (`check.py`, this box)

| Table | Worst | Gate |
|---|---|---|
| 1. moment identity `Σ W x^m = F_m(T)`, `m ≤ 2N−1`, `N` 1..3, 28 values of `T` from 0 through `1e-300` to `1e6`, against `scipy.special.gammainc` | 2.15e-15 relative | 4e-15 |
| 2. fp64 nodes and weights against the 50-digit route, same grid | 2.20e-15 relative (`N = 3`, weights, at `T_SWITCH`); 1.95e-15 nodes | 1e-14 |
| 3. contracted `(ab\|cd)` against libcint, all 16 ordered classes | 4.83e-14 of the block's max | 1e-13 |
| 3. uncontracted `(ab\|cd)` against the 50-digit quadrature | 9.25e-14 (`(sp\|pp)`); every other class ≤ 4.9e-14 | 1e-13 |
| 3. uncontracted `(ab\|cd)` against libcint | 1.72e-13 (`(sp\|sp)` and its three relatives) | see below |
| 4. each sign flipped alone | every flip caught | — |

Molecule: C, N, O at three non-collinear centres. Contracted: each atom's
3-primitive 6-31G valence s and p shells. Uncontracted: per atom two s and two
p single-primitive shells with distinct exponents from 6-31G. Per class, 1,296
uncontracted and 81 contracted blocks (22,032 in all); `T` ranged over `[0, 235]`, 8.4% of primitive quartets on
the large-`T` branch.

Worst relative error per class (table 3), of the block's largest entry:

| Class | `N` | uncontracted vs libcint | uncontracted vs 50 digits | libcint vs 50 digits | contracted vs libcint |
|---|---|---|---|---|---|
| (ss\|ss) | 1 | 6.98e-14 | 2.00e-15 | 6.96e-14 | 3.29e-14 |
| (ss\|sp) | 1 | 7.73e-14 | 8.45e-15 | 7.71e-14 | 4.80e-14 |
| (ss\|ps) | 1 | 7.73e-14 | 8.45e-15 | 7.71e-14 | 4.83e-14 |
| (ss\|pp) | 2 | 7.06e-14 | 6.58e-15 | 7.02e-14 | 2.70e-14 |
| (sp\|ss) | 1 | 7.71e-14 | 8.45e-15 | 7.71e-14 | 4.80e-14 |
| (sp\|sp) | 2 | 1.72e-13 | 1.82e-14 | 1.72e-13 | 4.61e-14 |
| (sp\|ps) | 2 | 1.72e-13 | 1.64e-14 | 1.72e-13 | 4.61e-14 |
| (sp\|pp) | 2 | 1.14e-13 | 9.25e-14 | 1.14e-13 | 3.85e-14 |
| (ps\|ss) | 1 | 7.71e-14 | 8.45e-15 | 7.71e-14 | 4.80e-14 |
| (ps\|sp) | 2 | 1.72e-13 | 1.64e-14 | 1.72e-13 | 4.60e-14 |
| (ps\|ps) | 2 | 1.72e-13 | 1.64e-14 | 1.72e-13 | 4.61e-14 |
| (ps\|pp) | 2 | 1.14e-13 | 5.95e-15 | 1.14e-13 | 3.85e-14 |
| (pp\|ss) | 2 | 7.07e-14 | 6.58e-15 | 7.04e-14 | 2.68e-14 |
| (pp\|sp) | 2 | 1.14e-13 | 2.68e-14 | 1.14e-13 | 3.85e-14 |
| (pp\|ps) | 2 | 1.14e-13 | 6.51e-15 | 1.14e-13 | 3.83e-14 |
| (pp\|pp) | 3 | 1.32e-13 | 4.85e-14 | 1.32e-13 | 4.04e-14 |

### Where the issue's acceptance is not met as written, and why

The issue asks for every class within `1e-13 · max|(ab|cd)|` of libcint,
uncontracted and contracted. **Contracted, it is met** (worst 4.83e-14).
**Uncontracted, 92 of 20,736 blocks miss libcint by up to 1.72e-13**, and in
every one of them **libcint is the one that is off**: the same quadrature
evaluated at 50 digits on the same float inputs (exact roots,
`prim_quartet_exact`) sits within 1e-13 of ours and farther from libcint's.
On `(ss|ss)`, where the quadrature is just `F_0` and the prefactor, ours is
2.0e-15 from the 50-digit value and libcint 7.0e-14. Single-primitive blocks
show libcint's per-primitive error undiluted; contraction averages it.

So `check.py` gates the uncontracted blocks on the 50-digit value instead, at
the same 1e-13, and lets a block miss libcint only if libcint is the farther of
the two from it ("referred", counted per class). The tolerance was not
loosened. The 50-digit route uses the same recurrence, so it referees
**rounding**, not the form; the form is settled by the contracted agreement
and table 4.

### What the numbers say about the recurrence's rounding

`(sp|pp)` at 9.25e-14 against the 50-digit value, against 5.95e-15 for its
mirror `(ps|pp)`, is the bra transfer: `I(0,1) = I(1,0) + (A − B) I(0,0)`
recovers `P − B` as `(P − A) + (A − B)`, losing `|AB| / |PB|` in relative
accuracy when `P` sits near `B`. Measured: the same 1,296 `(sp|pp)` blocks
built as `(ps|pp)` (VRR on the p centre, then transposed) are 5.95e-15 from
the 50-digit value. A kernel built on this recurrence should choose its VRR
centre (by `l` at least) rather than always building on `A` and `C`.

## The J build

`jbuild.coulomb_matrix` builds `J_ab = Σ_cd (ab|cd) D_cd` over a basis's
Cartesian AOs, host-side, on `rys2d.contracted_block`. One sweep over the
unique shell quartets (bra pair `i ≤ j`, ket pair `k ≤ l`, bra ≤ ket in pair
order), each block evaluated once and digested both ways, with weight 2 on
an off-diagonal pair and 1 on a diagonal one; `J_ji = J_ijᵀ`. `D` enters
through `(D + Dᵀ)/2` only, which loses nothing because `(ab|cd) = (ab|dc)`.

The screen is optional (`tau = 0`, the default, skips nothing). With
`Q_p = √max (ab|ab)` over a pair's diagonal block, a quartet is skipped when
`Q_p Q_q Σ|w D|` (over the other pair) is under `tau` **in both directions**,
so no skipped quartet moves any element of `J` by more than `tau`. Each
skip's bounds are summed into `JBuild.bound`, so `|J(τ) − J(0)| ≤ bound`
elementwise. It is a `continue`, not a sorted `break`. The ERI evaluator is
injected (`eri=`).

`check_j.py`, on this box (45 s):

| Table | What | Worst | Gate |
|---|---|---|---|
| 1. bookkeeping | the sweep against the dense `Σ_cd (ab\|cd) D_cd` over every ordered quartet of the same Rys blocks, H₂O/STO-3G, random **non-symmetric** `D` | 5.55e-16 of max\|J\| | 1e-14 |
| 2. libcint | `J` against PySCF's `get_jk(hermi=0)`: H₂O/STO-3G and H₂O/6-31G with their RHF densities, and `check.py`'s C, N, O molecule (contracted 6-31G valence, 12 AOs) with a random non-symmetric `D` | 7.16e-16, 1.99e-15, 5.90e-15 of max\|J\|; `E_J = ½Tr(DJ)` 4.5e-16, 3.0e-16, 2.8e-15 relative | 1e-12 |
| 3. screen | four H₂ along z, 3 Å apart, 6-31G, RHF `D`, `τ` 1e-12 … 1e-4: 3,550 to 6,942 of 9,316 quartets skipped, `max\|ΔJ\|` 2.5e-12 to 6.7e-4, the bound held at every element | `ΔJ/bound` ≤ 0.59 for `τ ≥ 1e-10` | bound, + 1e-14·max\|J\| for rounding |

At `τ = 1e-12` one element's difference, 4.4e-16 (two ulps of a 1.92 entry),
is over its 3.1e-16 bound: that is rounding, and it is what the slack is for.

It is about 1.5 s for H₂O/STO-3G and 7 s for H₂O/6-31G (1,035 unique shell
quartets), all in pure-Python primitive loops. It is a reference, not a
fast path.

## Not here

Anything above `(pp|pp)`; `d` and higher (`shells_of` refuses them; the
roots stop at N = 3, and `T_SWITCH = 50` will not hold for larger N, because
the Laguerre tail `O(e^{−T} T^{2N−3/2})` at N = 5 is ~5e-8 there);
derivatives; `K`; any `Spec` target, IR or emitted kernel; a place in
`src/`; the `Engine` protocol. Whether Rys becomes a third engine is decided
after this.
