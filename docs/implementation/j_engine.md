# The J engine: what the generator emits

**Status: specification, partly implemented.** This file is the derived spec
the generator must satisfy — our notation, our vocabulary, our batching config
— written so a test can be read against it rather than against a paper. Part of
it is now code (see CLAUDE.md, "The state of the tree"); the code is read
against this file, not the other way round.

**Implemented: pipeline steps 1–5, for one batch and for a split class.**

- *Step 1, the split.* Target enumeration is `src/felizjueves/plan/targets.py`;
  `batch_direction`, `in_batch` and the sub-batch partition are
  `src/felizjueves/plan/batching.py`. The partition's cut rule is **minimax** —
  the largest batch minimised, `[JEngine, v2, §5]` — with ties to the lowest
  boundaries and one set of cut points shared by `X`, `Y` and `Z`; that is the
  rule, wherever this file says "equalise".
- *Steps 2–3, MILD and the sieve*, and the per-batch counts, are host-side in
  `plan/` ([`mild.md`](mild.md) has the status): `dag.py`, `sieve.py`, and
  `cost.py`'s `PlanCost`, composed by `pipeline.plan(spec)`. They compute
  `[JEngine, v2, Tables 1–3]`, scored on the `literal` and `searched` boards.
- *Step 4, the schedule*, is `plan/schedule.py`; the recurrences below are
  built from it as IR by `src/felizjueves/recurrence.py`.
- *Step 5, the kernel*, is `build_pq_kernel` in `src/felizjueves/kernel.py` —
  the `[p|q]` contract below, `J_p = Σ_q [p|q] D_q` — spelled for each vendor
  by `emit/cuda.py` and `emit/hip.py`, for `n_batches = 1` with a plain
  (non-atomic) write-back; `build_pq_kernel` refuses a split class.
- *A split class* (`n_batches > 1`) is `build_pq_kernels(spec)`: one kernel
  per batch of the plan, each contracting only its batch's targets and writing
  `J_p` by atomic add (`passes.atomize`, `[JEngine, v2, §5]`), launched one
  after another into one zeroed `J`. Interpreted against `reference/` over
  every cell the plan admits for L = 0..16 × {1, 3, 6, 9}
  (`test/test_kernel_batched.py`); golden text for L = 4, `n_batches = 3`
  (`test/test_emit_batched.py`); run on both cards for every split cell of
  `measured.SUBGRID` (see "Measured" below). `python -m felizjueves.emit`
  (the kernel tier's `KERNEL_GEN_CMD`) still writes one-batch kernels only.

**Still specification:** the bra-side `E` transform
`J_ab = Σ_p E_p^{ab} J_p`, the `J_ab` scatter (pre- and post-contraction,
deferred to `[Fock]`), and **dispatch** — choosing `n_batches` per class at
generation time. `Spec.n_batches` defaults to 1; the measured recommendation
below is evidence for dispatch, not a policy the generator follows. A count
this file states is a claim about the code only where a test or a scored board
checks it.

The tree-search half (reduction directions, and the sieve that prunes them) is
[`mild.md`](mild.md). This file covers everything around it: the recurrences,
the Coulomb contraction, the kernel contract, the batching scheme, the pipeline
that ties them together, and the counts a correct pipeline has to reproduce.

Citations: `[JEngine, v2, …]` is entry 35 of [`../references.md`](../references.md),
indexed against the ChemRxiv v2 preprint; other works are cited by author and
DOI so they resolve regardless of that file's numbering.

## Notation

Fixed here, and used by both files in this directory.

| Symbol | Meaning |
|---|---|
| `a, b, c, d` | the four angular momenta of a shell quartet; `L = a + b + c + d` |
| `r = (rx, ry, rz)` | a Cartesian angular-momentum vector; `\|r\| = rx + ry + rz` |
| `[r]^(m)` | one-centre Hermite integral, auxiliary index `m` |
| `[p\|q]` | two-centre Hermite integral over bra index `p` and ket index `q` |
| **target** | a `[r]^(0)` with `\|r\| ≤ L` — what a kernel must produce |
| **intermediate** | an `[r]^(m)`, `m > 0` — what register pressure is made of |
| **fundamental** | `[0]^(m)`, the Boys-function values; never reduced |
| `1x, 1y, 1z` | the unit vectors `(1,0,0)`, `(0,1,0)`, `(0,0,1)` |
| `D_{a,b}` | a batch: direction `D ∈ {X,Y,Z}`, max-component range `a..b` |

Counts that fall out of the definitions, and that the generator should assert
rather than hardcode:

- Hermite functions in an `|ab]` shell pair: `(a+b+1)(a+b+2)(a+b+3)/6`.
- Targets in a class of total angular momentum `L`: `(L+1)(L+2)(L+3)/6`.

## The McMurchie–Davidson recurrences

McMurchie & Davidson (doi:10.1016/0021-9991(78)90092-X) expand the Cartesian
products of primitive Gaussians in Hermite polynomials, giving the primitive ERI
as a double sum over Hermite indices with transformation coefficients `E` on
each side, and the two-centre Hermite integral as

```
[p|q] = (-1)^|q| [p+q]^(0)
```

with component-wise vector addition. So **every integral in a class reduces to
the set of one-centre `[r]^(0)` with `|r| ≤ L`** — that set is what a kernel
computes, and `[JEngine, v2, eqs. 7–9]` is where this is set up.

The one-centre integrals come from three recurrences, one per direction
(`[JEngine, v2, eqs. 10–12]`), in raising form:

```
[r + 1d]^(m) = PQ_d · [r]^(m+1) + r_d · [r - 1d]^(m+1)        for d in x, y, z
```

and the fundamentals from the Boys function
`F_m(T) = ∫₀¹ t^{2m} exp(-t²T) dt` (`[JEngine, v2, eqs. 13, 18]`).

### The reduction form, and why unit components are cheap

The generator uses these backwards — it is building `[r]^(0)`, so substitute
`r → r - 1d`:

```
[r]^(m) = PQ_d · [r - 1d]^(m+1) + (r_d - 1) · [r - 2d]^(m+1)
```

This is the form worth internalising, because the cost of a node falls straight
out of it:

| `r_d` | parents | arithmetic |
|---|---|---|
| `1` | one (`[r - 1d]^(m+1)`) | one multiply — the second term vanishes |
| `≥ 2` | two (parent and grandparent) | one multiply-add |

That asymmetry is the entire reason the tree-search heuristic prefers small
components, and it is what Johnson's rules 2 and 3 are made of. See
[`mild.md`](mild.md).

### Two defects in the source, flagged

Both are in the ChemRxiv v2 preprint and neither changes the method, but the
generator must not transcribe them:

- **`1y` is misprinted** as `(1,1,0)` in `[JEngine, v2, eq. 17]`. It is
  `(0,1,0)`; `(1,1,0)` is not a unit vector and would break eq. 11.
- **`θ` is never defined, and eq. 13 is wrong in `m` under any definition.**
  The fundamental, as printed, is
  `[0]^(m) = (2/√π) θ (-2θ)^m F_m(|θ·PQ|²)` (`[JEngine, v2, eq. 13]`), and v2
  never says what `θ` is. (`[JEngine, v1]` did: `θ = √(ζη/(ζ+η))` in the text
  under its eq. 11. v2 dropped the line.) **Pinned, by our derivation** against
  McMurchie & Davidson — numbers are the report's, entry 46 of
  [`../references.md`](../references.md):
  - `θ² = ζη/(ζ+η)`, `ζ = α+β`, `η = γ+δ`. Forced by the Boys argument: MD's
    `T = α_P α_Q (α_P+α_Q)⁻¹ |PQ|²` (eq. 3.32) equals `|θ·PQ|²` only so.
  - **The `m`-dependence is `(-2θ²)^m`, not `(-2θ)^m`.** MD set
    `R_000j = (-2α)^j F_j(T)` with `T = α|PQ|²` (eqs. 4.2, 4.4), so that the
    unscaled recurrences (eqs. 4.6–4.8, ours above) hold; here `α = θ²`.
    With `(-2θ)^m` the recurrence no longer differentiates the fundamental.
    The sign is `(-1)^m`, on the fundamental, alongside `F_m`.
  - **`2/√π` lives on the fundamental, with one `θ`; `π³/(ζη)^{3/2}` does not.**
    MD's ERI prefactor is `λ = 2π^{5/2} ζ⁻¹ η⁻¹ (ζ+η)^{-1/2}` (eqs. 3.30–3.31),
    which factors exactly as `(2/√π) θ · (π/ζ)^{3/2} (π/η)^{3/2}`. Eq. 13 keeps
    the first factor. The pair factors `(π/ζ)^{3/2}` and `(π/η)^{3/2}` are ours
    to place: the generator folds them into the bra and ket pair data (with
    the `E` coefficients), never into `[0]^(m)`.

  So the generator emits `[0]^(m) = (2/√π) θ (-2θ²)^m F_m(θ²|PQ|²)`. The
  source states none of this in v2; it is our derivation, and a test should
  cite it here rather than the paper.

## Why this is a J engine and not an ERI engine

The Coulomb matrix is `J_µν = Σ_λσ (µν|λσ) D_λσ`. Because the density spans
only the ket, the ket-side contraction can be done **before** the bra-side
transformation (`[JEngine, v2, eqs. 20–22]`):

```
J_p  = Σ_q [p|q] D_q                   # the ONLY shell-quartet-level work
J_ab = Σ_p E_p^{ab} J_p                # bra transform — shell-pair level
```

The split is the point. `[p|q]` depends on both the bra pair and the ket pair,
so `J_p = Σ_q [p|q] D_q` is **the only contraction that happens at shell-quartet
level**; the `E` transform depends on the bra pair alone and so drops to
shell-pair level. That is the register saving the whole design rests on — the
kernel never materialises `[ab|cd]`.

Pre- and post-contraction (forming `D_q`, and scattering `J_ab` back) are out of
scope here; `[JEngine, v2]` defers them to `[Fock]`.

Screening: Cauchy–Schwarz bounds take the nominal `O(N⁴)` to `O(N²)`. Each
Hermite index carries a bound `I`, and the kernel tests `D_max · I_p · I_q`
against a threshold `τ`.

## The `E` transform

**Status: the undressed coefficients are implemented** —
`src/felizjueves/ao/hermite.py` (`e_axis`, `e_block`), checked by
`test/test_ao_hermite.py`. The dressing (`ao/pairs.py`) and the bra
back-transform `J_ab = Σ_p E_p^{ab} J_p` are not yet code.

`[JEngine, v2]` says only that the primitive ERI carries "transformation
coefficients `E` on each side"; it does not state their recurrence. That is
McMurchie & Davidson's, and the equation numbers below are the **report's**
(entry 46 of [`../references.md`](../references.md), UCRL-13791, as read from
its OSTI scan). The journal version (entry 39) was not read for this section,
so no equation number here is a claim about entry 39's numbering.

**The recurrence.** Per axis, for a primitive pair with exponents `α` on `A`
and `β` on `B`:

```
ζ = α + β        P = (αA + βB)/ζ        μ = αβ/ζ        (eqs. 2.8, 2.9)
X_PA = P_x − A_x  (eq. 2.14)     X_PB = P_x − B_x     X_AB = A_x − B_x

E^{00}_0    = exp(−μ X_AB²)
E^{i+1,j}_t = (1/2ζ) E^{ij}_{t−1} + X_PA E^{ij}_t + (t+1) E^{ij}_{t+1}   (eq. 2.20)
E^{i,j+1}_t = (1/2ζ) E^{ij}_{t−1} + X_PB E^{ij}_t + (t+1) E^{ij}_{t+1}   (eq. 2.21)
E^{ij}_t    = 0   for t < 0 or t > i+j
```

and the 3D coefficient over a Hermite index `t = (tx, ty, tz)` is the product
`E^{ab}_t = E_x E_y E_z`, enumerated in `hermite_targets(|a|+|b|)` order (the
plan layer's; `ao` enumerates the same order itself and a test asserts the two
agree, because `ao` may not import `plan`).

**How this differs from the report's `d`.** MD's coefficients are `d^{nn'}_N`
with `d^{00}_0 = 1` (eq. 2.22), and the Gaussian-product constant
`E_IJ = exp(−αβ(α+β)⁻¹|A−B|²)` (eq. 2.10) is kept outside them and applied once
per pair (eq. 2.27). Here each axis carries its own factor `exp(−μ X_AB²)` in
`E^{00}_0`, so the product over three axes reproduces `E_IJ · d d d` exactly.
In code, `μ X_AB²` is formed as `−ζ X_PA X_PB`, which equals it identically
(`X_PA = −(β/ζ) X_AB`, `X_PB = (α/ζ) X_AB`); so `e_axis` takes
`(i, j, ζ, X_PA, X_PB)` and no separate `X_AB` or `μ`.

**What it expands.** With `Λ_t(x) = ζ^{t/2} H_t(√ζ (x−P)) exp(−ζ(x−P)²)` —
MD's `Λ_t` (eqs. 2.15–2.16) times the Gaussian it multiplies — the coefficients
satisfy, pointwise,

```
x_A^i x_B^j exp(−α x_A²) exp(−β x_B²) = Σ_t E^{ij}_t Λ_t(x)     (eqs. 2.7, 2.17)
```

`test/test_ao_hermite.py` checks exactly this, with `H_t` from numpy's closed
form rather than from the recurrence under test, plus a Gauss–Hermite moment
check, `E^{00}_0`, the empty top of the table, parity at `A = B`, and that
`e_block` is the product of its three axes.

**Where the factors go — decided once.** MD's ERI prefactor is
`λ = 2π^{5/2} ζ⁻¹ η⁻¹ (ζ+η)^{−1/2}` (eqs. 3.30–3.31), and

```
λ = (2/√π) θ · (π/ζ)^{3/2} · (π/η)^{3/2},        θ² = ζη/(ζ+η)
```

- `(2/√π) θ` lives on the fundamental `[0]^(m)`, in the kernel (see "Two
  defects in the source" above).
- `(π/ζ)^{3/2}` goes on the **bra** pair and `(π/η)^{3/2}` on the **ket** pair,
  host-side, applied by one dressing function in `ao/pairs.py` together with
  the contraction coefficient. The primitive normalisation is **not** a
  dressing: a `Shell`'s coefficients multiply unnormalised primitives
  (`ao/shell.py`), so it is already in them, and applying it again in the
  dressing normalised twice — caught against libcint's `int2e` by #85.
- `ao/hermite.py` carries **none** of these. Any of them there breaks the
  pointwise identity above, which is the only independent check `E` has: if
  that identity needs a fudge factor to hold, the factor is in the wrong module.

The three placements multiply back to `λ` and nothing else, which is the check
that the placement is complete.

## The `[p|q]` kernel contract

What the generated kernel must look like from the outside
(`[JEngine, v2, Alg. 1]`), stated as a contract rather than as code, since the
code is the generator's output:

- **One warp per Hermite bra index.** The bra index is
  `threadIdx.x + blockIdx.x * blockDim.x`.
- **Kets are strip-mined over `threadIdx.y`**, stride `blockDim.y`.
- **Kets arrive pre-sorted** by `I_q · D_max`, descending. This is what lets the
  screening test **`break` rather than `continue`**: once the bound falls under
  `τ`, the whole remaining tail is under it too. A generator that emits
  `continue` here is correct but throws away the sort.
- **Accumulate `J_p` into shared memory**, then reduce across the warp.
- **The warp reduction is what makes the global write cheap** — one write per
  bra index per block instead of one per thread.
- **The global write needs atomics.** Not in the single-batch case, but see
  below: once a class is split into batches, several kernels contribute to the
  same `J_p`. `[JEngine, v2, Alg. 1]` presents line 10 as atomic-free and §5
  then corrects it; the reduction keeps the atomic traffic small enough that it
  stays a minor cost.

Nothing in this contract is specific to CUDA or HIP, which is the point — the
same emitted structure has to satisfy `nvcc` and `hipcc` both.

## Batching

A high-`L` class has too many intermediates live at once; they spill, and the
kernel goes memory-bound (`[JEngine, v2, §2]` measures up to 160 KB of spills at
`L = 16`). The fix is to **split the targets across several kernels**, each
computing a subset, so each one fits.

The split has to trade three quantities off (`[JEngine, v2, §5]`):

1. targets per batch — want fewer;
2. intermediates per batch — want fewer;
3. total FLOPs across all batches — grows with batch count, because
   intermediates shared between batches get recomputed.

### The scheme

Partition targets by the **direction of their maximum component**, then
subdivide each direction by that component's **magnitude**:

```python
AXES = ("x", "y", "z")

def batch_direction(r):
    """Direction of r's maximum component; ties by cyclic predecessor."""
    if r[0] == r[1] == r[2]:
        return "x"                             # three-way tie
    hi = max(r)
    for i in (0, 1, 2):
        if r[i] == hi and r[(i + 1) % 3] == hi:
            return AXES[i]
    return AXES[r.index(hi)]


def in_batch(r, direction, a, b):
    d = batch_direction(r)
    return d == direction and a <= r[AXES.index(d)] <= b
```

`[JEngine, v2, Alg. 4]`. Note the two deliberate inversions against
[`mild.md`](mild.md): this uses the **maximum** component where MILD uses the
minimum, and the **reverse** round-robin — ties go to the cyclic predecessor
(`{x,y}→x`, `{y,z}→y`, `{x,z}→z`), with all-equal landing in `X` rather than
MILD's `Z`. Zeros need no masking here, since a zero is never the maximum of a
non-zero vector.

A batch with `a = 0, b = L` is the whole direction, written `X`, `Y`, `Z`
unsubscripted.

Worked example, enumerated from `in_batch` above for `L = 5`, direction `Y`,
`a,b = 1,2`. For direction `Y` the predicate reduces to "`ry` is the maximum
and `rx` is not", which with `1 ≤ ry ≤ 2` and `|r| ≤ 5` admits eight targets —
in shell-letter notation:

```
Y_{1,2} = { p_y, d_yz, d_y², f_xy², f_y²z, g_xy²z, g_y²z², h_xy²z² }
```

This agrees with the set listed at `[JEngine, v2, §5]`, which makes it a real
check on the pseudocode rather than an illustration: **three of the eight
members are present only because of the tie-break.** `d_yz`, `g_y²z²` and
`h_xy²z²` all have `ry = rz`, so under the opposite convention they would sort
into `Z` instead. An implementation with the round-robin backwards fails on
exactly these three and on nothing else in this batch — which is why this
example belongs in the test suite.

### Why the maximum direction is the right split

Because it minimises what gets duplicated. Under MILD:

- between the `X`, `Y` and `Z` groups, the **only** shared intermediates are
  the fundamentals `[0]^(m)`;
- between sub-batches of one direction, the shared set is a subset of the
  **axial** integrals.

Two consequences, both real costs:

- **The Boys function is re-evaluated per batch** (the fundamentals are the
  duplicated set). `[JEngine, v2, §5]` measures ~2% of kernel time, shrinking
  as `L` grows.
- **`J_p` is now written by several kernels**, hence the atomics noted above.

### Config the generator exposes

- **Batch count is always a multiple of three.** Finer splits break the
  three-way symmetry; `[JEngine, v2, §5]` found them both unnecessary and worse.
- **Sub-batch ranges `a..b` are chosen to minimise the largest batch's target
  count** (minimax, `[JEngine, v2, §5]`), not fixed a priori: ties go to the
  lowest boundaries, and one set of cut points is shared by `X`, `Y` and `Z`.
  So the ranges are an output of the generator, not an input — the input is the
  batch count.
- `[JEngine, v2, §7]` reports that **at most 9 batches suffices for `L ≤ 16`**,
  and presents 1, 3, 6 and 9 — which is the range we should generate over.
  On our cards, under a definition declared before the data (`measured.py`,
  NINE AGAINST TWELVE), 9 does **not** suffice on either: 12 spills less on
  `sm_86` at L 13 and 16, and is faster on near-T on `gfx1200` at L 16
  (`jengine/nine-batches-suffice` in `data/literature/jengine/claims.json`,
  UNVERIFIABLE against their A100). So the measured subgrid runs to 12.
- **The default is one batch.** `Spec.n_batches` defaults to 1, and the
  measured recommendation below does not change that.

### Measured: a recommended batch count per `(L, arch)`

`[JEngine, v2, §7]` and the captions of Tables 1–3 bold the batch count that
minimised runtime on their A100 at each `L`. This is the **same kind of
statement about our cards** — and only that. It is not a reproduction of their
bolded choice: their kernels, compiler and card are not ours, and their bold
cells are deliberately not transcribed
(`data/literature/jengine/README.md`).

**Where it was measured.** One host, `ddbbc02dbb08 (container)` — a dev
container on this box carrying both device toolchains — with two cards:

- NVIDIA GeForce RTX 3080, `sm_86`, `nvcc 13.0`;
- AMD Radeon RX 9060 XT (`AMD Radeon Graphics (device 0x7590, gfx1200)`),
  `gfx1200`, `hipcc / ROCm 7.2.4`;

with `devtools/config.sh`'s `NVCC_FLAGS` / `HIPCC_FLAGS` (no fast-math), on
2026-10-11.

**What was measured.** `measured.SUBGRID`: `L ∈ {7, 10, 13, 16}` ×
`n_batches ∈ {1, 3, 6, 9, 12}`, the default split `pq_spec(L, n_batches)`,
every cell on both arches — **no cell dropped** for compile cost. A cell's
number is the host wall-clock of its whole launch sequence (every
`build_pq_kernels` batch into one zeroed `J`, then one synchronize), 5 warm-up
and **31 timed** sequences, `n_bra × n_ket = 2048 × 128`, no screening, run
alone on an uncontended box by `test/test_kernel_device.py`'s opt-in sweep —
and the timed binary is the one first checked against `reference/`. Two
sweeps: #53 timed `n_batches` 1, 3, 6, 9 (PR #75); #60 timed 12 (PR #77).
**Two inputs**, both with that protocol: *wide-T* (`T` up to ~430, a long
Boys series; the row's `runtime_ns`) and *near-T* (`T < ~2`; the row's raw
`wallclock:near:*`). Only the ranking over `n_batches` **within one arch** is a
claim; absolute times are not comparable across vendors (`gfx1200`
rate-limits fp64).

**The rule, from wall-clock only** — not spills, not registers, not
`KernelCost`: per `(arch, L, input)`, the lowest median wins. It is
*resolved* iff the runner-up's median exceeds it by more than the larger of
the two IQRs (#60's test of "faster"); *within spread* lists every count whose
median is within that distance of the best; *same sweep* says whether best and
runner-up came from one sweep — #60 measured cross-sweep drift of up to 1.7%
(10% on one noisy cell), IQR-scale, so a cross-sweep margin is weaker
evidence.

The data is `data/measured/batch_recommendation.json`, written by
`PYTHONPATH=src uv run python -m felizjueves.batchreport --write` from
`data/measured/kernels.json`; the tables below are its `--markdown`, and
`test/test_batchreport.py` recomputes both from the store and fails if either
has moved.

*wide-T input:*

| arch | L | best `n_batches` | median ms | IQR ms | repeats | runner-up (+ms) | resolved | within spread | same sweep |
|---|---|---|---|---|---|---|---|---|---|
| sm_86 | 7 | **1** | 24.121 | 0.500 | 31 | 3 (+40.370) | yes | 1 | yes |
| sm_86 | 10 | **1** | 37.015 | 0.659 | 31 | 3 (+55.785) | yes | 1 | yes |
| sm_86 | 13 | **1** | 66.983 | 0.381 | 31 | 3 (+81.773) | yes | 1 | yes |
| sm_86 | 16 | **1** | 157.271 | 1.703 | 31 | 3 (+133.313) | yes | 1 | yes |
| gfx1200 | 7 | **1** | 34.895 | 0.983 | 31 | 3 (+53.711) | yes | 1 | yes |
| gfx1200 | 10 | **1** | 52.338 | 1.222 | 31 | 3 (+76.070) | yes | 1 | yes |
| gfx1200 | 13 | **1** | 150.059 | 0.781 | 31 | 3 (+110.608) | yes | 1 | yes |
| gfx1200 | 16 | **1** | 531.570 | 1.551 | 31 | 3 (+50.446) | yes | 1 | yes |

*near-T input:*

| arch | L | best `n_batches` | median ms | IQR ms | repeats | runner-up (+ms) | resolved | within spread | same sweep |
|---|---|---|---|---|---|---|---|---|---|
| sm_86 | 7 | **1** | 3.682 | 0.017 | 31 | 3 (+2.530) | yes | 1 | yes |
| sm_86 | 10 | **1** | 8.486 | 0.096 | 31 | 3 (+3.850) | yes | 1 | yes |
| sm_86 | 13 | **3** | 29.317 | 0.157 | 31 | 1 (+2.924) | yes | 3 | yes |
| sm_86 | 16 | **9** | 99.387 | 0.127 | 31 | 12 (+2.734) | yes | 9 | no |
| gfx1200 | 7 | **1** | 4.485 | 0.069 | 31 | 3 (+4.192) | yes | 1 | yes |
| gfx1200 | 10 | **3** | 17.928 | 0.146 | 31 | 1 (+0.104) | no | 1, 3 | yes |
| gfx1200 | 13 | **3** | 80.160 | 0.378 | 31 | 6 (+0.045) | no | 3, 6 | yes |
| gfx1200 | 16 | **12** | 299.697 | 0.335 | 31 | 9 (+2.821) | yes | 12 | no |

What it says:

- **The answer depends on the input.** On wide-T, one batch wins at every
  `L` on both cards, by margins far beyond the spread: each batch recomputes
  its fundamentals ("Why the maximum direction is the right split"), and with
  a long Boys series that duplicated work outweighs anything splitting saves.
  On near-T, splitting wins from `L = 13`. "The batch count with the best
  median" is therefore not one number per `(L, arch)`, and a dispatch layer
  will need to know which regime of `T` it is serving.
- **The arches disagree.** On near-T at `L = 16`, `sm_86` is fastest at 9 and
  `gfx1200` at 12; at `L = 10`, `gfx1200` already prefers 3 (unresolved
  against 1) where `sm_86` keeps 1. That is the reason dispatch will have to be
  per target — not a disagreement to average away.
- **Two near-T cells are unresolved**: `gfx1200` at `L = 10` (3 vs 1, +0.104
  ms against a 0.146 ms IQR) and `L = 13` (3 vs 6, +0.045 ms against 0.378
  ms). Their recommendation is "either of the *within spread* counts".
- **Two near-T winners cross sweeps.** `gfx1200` `L = 16`: 12 (timed by #60)
  over 9 (timed by #53). #60 re-timed 9 beside 12 as a drift check, not stored:
  303.82 against 299.70 ms, IQR 0.67 ms — the same order. Within #53's sweep
  alone the best is 9, +5.46 ms ahead of 3. `sm_86` `L = 16`: 9 (#53) over 12
  (#60) by 2.73 ms; within #53's sweep, 9 leads 6 by 9.05 ms.
- **Outside the subgrid** (`L ∉ {7, 10, 13, 16}`) nothing is timed, so there is
  no recommendation there.
- **Literal kernels only.** The table reads the `levers = ()` rows. The one
  measured lever, `recycle` (#55), changes nothing it could rank: nvcc 13.0
  and hipcc / ROCm 7.2.4 emit identical machine code for both arms at every
  subgrid cell, and the A/B reads `sm_86` 20/20 neutral and `gfx1200` 16/20
  neutral (4 read as slower on that identical code, the protocol's noise
  floor). It stays opt-in.

**This does not change `Spec`'s default.** `n_batches` defaults to 1 and stays
there (`test/test_batchreport.py` holds it). Choosing a count per class at
generation time is **dispatch**, which has no home in this repo yet; this table
is evidence for that future layer, not a policy the generator silently
follows.

## The pipeline

This is the algorithm the repo *is* (`[JEngine, v2, §6]`):

1. **Split** the targets `{[r]^(0)}` into batches — `batch_direction` above.
2. **Build** the intermediate-dependency DAG per MILD — [`mild.md`](mild.md).
3. **Prune** it with the sieve — [`mild.md`](mild.md).
4. **Emit** the `[p|q]` evaluation as a **topological sort of the pruned DAG**.
5. **Emit** the kernel around it, per the contract above.

Step 4 is where the portability risk lives: the topological order is what
decides live ranges, and therefore what `nvcc` and `hipcc` each do with their
registers. Two compilers will not agree on the consequences of one order.

The generator should also **report FLOP and intermediate counts** per batch.
That is not a nicety: it is the only cost signal available without a card, and
it is what the anchors below are checked against.

## Validation targets

`[JEngine, v2, Tables 1–3]` report what their generator computed over
`L = 7…16` for 1, 3, 6 and 9 batches: total FLOPs, max intermediates per batch,
and max targets per batch. Those are *generator output*, not measurements, so a
correct implementation of steps 1–3 should **reproduce them** — and it is the
cheapest end-to-end check available on the tree search, needing neither a
compiler nor a card.

**The grid itself is deliberately not copied here.** The plan layer computes
these counts itself (`plan/cost.py`'s `PlanCost`); the paper's grid is
transcribed, with provenance, in `data/literature/jengine/table{1,2,3}.json`,
and the scoreboards in `data/scoreboards/jengine/` record how our counts
compare to it.

Four corner values, as anchors to bootstrap the first implementation:

| `L` | batches | targets/batch | intermediates/batch | FLOPs |
|---|---|---|---|---|
| 7 | 1 | 120 | 125 | 424 |
| 7 | 3 | 42 | 51 | 425 |
| 16 | 1 | 969 | 1647 | 5633 |
| 16 | 3 | 327 | 591 | 5712 |

Three checks that these cohere, each worth re-running against our own counts:

- **The 1-batch target count is just the closed form.** `L = 7` gives
  `8·9·10/6 = 120`; `L = 16` gives `17·18·19/6 = 969`. Both match, which
  confirms the quantity is "all targets" and not something screened.
- **The 3-batch target count is near `total/3`.** `120/3 = 40` vs. 42 at
  `L = 7`; `969/3 = 323` vs. 327 at `L = 16`. That is the equalise-target-count
  rule working, the small excess coming from the integer granularity of the
  `a..b` ranges.
- **The trade-off is the whole point.** Going 1 → 3 batches at `L = 16` cuts
  intermediates 2.8× and targets 3× for a 1.4% FLOP increase. The paper's
  9-batch column carries this to ~5× fewer intermediates for ~26% more FLOPs.

Two caveats before anyone reads a mismatch as our bug. **The counts exclude the
Boys function**, deliberately, so that batching schemes can be compared
(`[JEngine, v2, §7.1]`). And **the published tables bold the batch count that
was fastest for each `L`** — bolding a text extraction drops, so the fastest
column is not recorded here either; recover it from the paper before using these
to pick a default batch count.

`[JEngine, v2, Table 4]` gives average spills and register use. Not reproduced
either, and for a second reason on top of the first: those are A100
measurements of their kernels under their compiler, and neither of our targets
(`sm_86`, `gfx1200`) is that. Our equivalent numbers come from the kernel tier,
which is the only place this repo can make a numerical claim at all.
