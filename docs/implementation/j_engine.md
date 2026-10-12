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

**Implemented host-side, around the kernel: the Coulomb matrix** (milestone 4,
`src/felizjueves/ao/`). The `E` transform (`hermite.py`), pair data, the one
dressing function and the Schwarz bound (`pairs.py`), the ERI block
(`eri.py`), and `J_µν` — ket pre-contraction, `J_p`, bra back-transform
`J_ab = Σ_p E_p^{ab} J_p` and the scatter — by two sweeps (`j.py`), all pure
Python on the host. `J_p` can come from `reference.pq` or, through `ao.j`'s
`jp=` seam, from the **generated** kernel: interpreted (`drive.py`), or
compiled and launched on both cards (`test/test_ao_j_ondevice.py`, kernels
written by `python -m felizjueves.emit --class …`). Every one of those paths is
compared against PySCF/libcint — see "The Coulomb matrix, witnessed from
outside" below.

**Still specification:** **dispatch** — choosing `n_batches` per class at
generation time — and any device code for what is host-side above (the `E`
transform, the pre-contraction, the scatter, a device-resident `J`, sorted
kets with real bounds on the card). `Spec.n_batches` defaults to 1; the
measured recommendation below is evidence for dispatch, not a policy the
generator follows. A count this file states is a claim about the code only
where a test or a scored board checks it.

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
  `recurrence.reduction` transcribes eq. 17's reduction form, applying the
  shift `1d` as the direction's unit vector, and
  `test_eq17_witness.py::test_y_reduction_shifts_by_the_unit_vector_0_1_0`
  holds it there (`jengine/eq17-1y-misprint`, asserted, #107): on every node
  a real plan reduces along y with `r_y ≥ 1` and `r_x ≥ 1` (where `(1,1,0)`
  and `(0,1,0)` name different in-range parents), the `PQ_y` term must read
  `[r - (0,1,0)]^(m+1)` and the `(r_y - 1)` term `[r - (0,2,0)]^(m+1)`.
  Spelling `1y` as `(1,1,0)`, in the test's transcription or in the DAG and
  IR builder, fails it.
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

  **No longer only a derivation (#90): it has an outside witness.** `J_µν`
  built with the generated kernel's `J_p` matches PySCF/libcint, which
  shares no ancestry with this repo, and the claims registry
  (`data/literature/jengine/claims.json`) names the tests:
  `jengine/theta-undefined` is held by
  `test_ao_j_interpreted.py::test_interpreted_j_matches_reference_and_libcint`,
  and the factorisation of `λ` — `jengine/prefactor-factorisation` — by
  `test_ao_eri.py::test_block_matches_libcint`, per angular-momentum class.
  Both fail when the claim is false; "The Coulomb matrix, witnessed from
  outside" has the perturbations that show it. The `(-2θ²)^m` weight has its
  own record since #120, `jengine/m-weight-minus-two-theta-squared`, held by
  `test_ao_eri.py::test_the_m_weight_is_minus_two_theta_squared`: the same
  libcint block comparison, with the base perturbed (the printed `-2θ`, or
  `-2θ²·(1+1e-9)`) failing every fast-tier class above `(ss|ss)` and leaving
  `(ss|ss)`, which reaches only `m = 0`, unchanged. The `1y` misprint
  (`jengine/eq17-1y-misprint`) is asserted by the structural witness above
  (#107): `recurrence.py` transcribes eq. 17 with `1y = (0,1,0)`.

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

Pre- and post-contraction (forming `D_q`, and scattering `J_ab` back) are
deferred by `[JEngine, v2]` to `[Fock]`. This repo takes only as much of them as
a `J` build needs, host-side, in `src/felizjueves/ao/j.py` (milestone 4; "The
Coulomb matrix, witnessed from outside" below); they are not generated as
device code.

Screening: Cauchy–Schwarz bounds take the nominal `O(N⁴)` to `O(N²)`. Each
Hermite index carries a bound `I`, and the kernel tests `D_max · I_p · I_q`
against a threshold `τ`.

## The `E` transform

**Status: implemented, host-side.** The undressed coefficients are
`src/felizjueves/ao/hermite.py` (`e_axis`, `e_block`), checked by
`test/test_ao_hermite.py`; the dressing is `ao.pairs.dress`; the bra
back-transform `J_ab = Σ_p E_p^{ab} J_p` is in `ao/j.py`. None of it is
emitted as device code.

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

**Where it was measured.** One host, `1da02cac6d4e (container)` — a dev
container on this box carrying both device toolchains — with two cards:

- NVIDIA GeForce RTX 3080, `sm_86`, `nvcc 13.0`;
- AMD Radeon RX 9060 XT (`AMD Radeon Graphics (device 0x7590, gfx1200)`),
  `gfx1200`, `hipcc / ROCm 7.2.4`;

with `devtools/config.sh`'s `NVCC_FLAGS` / `HIPCC_FLAGS` (no fast-math; since
#92 they carry `-Xcompiler -Wall` / `-Wall -Wextra`, which change no register,
spill or remark), on 2026-10-11.

**What was measured.** `measured.SUBGRID`: `L ∈ {7, 10, 13, 16}` ×
`n_batches ∈ {1, 3, 6, 9, 12}`, the default split `pq_spec(L, n_batches)`,
every cell on both arches — **no cell dropped** for compile cost. A cell's
number is the host wall-clock of its whole launch sequence (every
`build_pq_kernels` batch into one zeroed `J`, then one synchronize), 5 warm-up
and **31 timed** sequences, `n_bra × n_ket = 2048 × 128`, no screening, run
alone on an uncontended box by `test/test_kernel_device.py`'s opt-in sweep —
and the timed binary is the one first checked against `reference/`. **One
session per card**: #93 re-timed every cell of an arch in one sweep, behind
the card lock and the CPU and card-utilisation gates, replacing the two
sweeps it was first taken in (#53: `n_batches` 1, 3, 6, 9, PR #75; #60: 12,
PR #77). Each row records the card's state as the gate let each input
through and just after it (raw `wallclock:gpu_util_before` / `_after`, and
`wallclock:near:...`): every `_before` reading was at most 20% (the gate
admits 30%), every `_after` at most 5%.
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
evidence. Since #93 every margin below is same-sweep.

The data is `data/measured/batch_recommendation.json`, written by
`PYTHONPATH=src uv run python -m felizjueves.batchreport --write` from
`data/measured/kernels.json`; the tables below are its `--markdown`, and
`test/test_batchreport.py` recomputes both from the store and fails if either
has moved.

*wide-T input:*

| arch | L | best `n_batches` | median ms | IQR ms | repeats | runner-up (+ms) | resolved | within spread | same sweep |
|---|---|---|---|---|---|---|---|---|---|
| sm_86 | 7 | **1** | 25.306 | 2.687 | 31 | 3 (+39.510) | yes | 1 | yes |
| sm_86 | 10 | **1** | 37.319 | 1.958 | 31 | 3 (+55.547) | yes | 1 | yes |
| sm_86 | 13 | **1** | 66.525 | 0.528 | 31 | 3 (+82.033) | yes | 1 | yes |
| sm_86 | 16 | **1** | 156.851 | 0.723 | 31 | 3 (+132.450) | yes | 1 | yes |
| gfx1200 | 7 | **1** | 34.779 | 1.404 | 31 | 3 (+54.360) | yes | 1 | yes |
| gfx1200 | 10 | **1** | 52.929 | 1.622 | 31 | 3 (+75.690) | yes | 1 | yes |
| gfx1200 | 13 | **1** | 150.011 | 0.876 | 31 | 3 (+110.581) | yes | 1 | yes |
| gfx1200 | 16 | **1** | 533.064 | 0.841 | 31 | 3 (+49.651) | yes | 1 | yes |

*near-T input:*

| arch | L | best `n_batches` | median ms | IQR ms | repeats | runner-up (+ms) | resolved | within spread | same sweep |
|---|---|---|---|---|---|---|---|---|---|
| sm_86 | 7 | **1** | 3.682 | 0.010 | 31 | 3 (+3.506) | yes | 1 | yes |
| sm_86 | 10 | **1** | 9.644 | 0.321 | 31 | 3 (+4.421) | yes | 1 | yes |
| sm_86 | 13 | **3** | 29.478 | 2.546 | 31 | 1 (+3.120) | yes | 3 | yes |
| sm_86 | 16 | **9** | 98.776 | 0.120 | 31 | 12 (+4.280) | yes | 9 | yes |
| gfx1200 | 7 | **1** | 4.507 | 0.058 | 31 | 3 (+4.283) | yes | 1 | yes |
| gfx1200 | 10 | **3** | 18.055 | 0.160 | 31 | 1 (+0.089) | no | 1, 3 | yes |
| gfx1200 | 13 | **6** | 80.227 | 0.686 | 31 | 3 (+0.050) | no | 3, 6 | yes |
| gfx1200 | 16 | **12** | 299.352 | 0.314 | 31 | 9 (+4.113) | yes | 12 | yes |

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
- **Two near-T cells are unresolved**: `gfx1200` at `L = 10` (3 vs 1, +0.089
  ms against a 0.160 ms IQR) and `L = 13` (6 vs 3, +0.050 ms against 0.686
  ms). Their recommendation is "either of the *within spread* counts".
- **Re-timed behind the gate (#93), and confirmed.** Every conclusion above
  is the one #53's and #60's sweeps gave: wide-T picks 1 everywhere, near-T
  splits from `L = 13`, the cards disagree at `L = 16` (9 on `sm_86`, 12 on
  `gfx1200`), and the same two cells are unresolved with the same *within
  spread* sets. The one recorded best that moved is `gfx1200` `L = 13`
  near-T, 3 → 6, inside that unresolved tie (3 led 6 by 0.045 ms before; 6
  leads 3 by 0.050 ms now). The two `L = 16` near-T winners, which used to
  cross sweeps, now hold within one: `gfx1200` 12 over 9 by 4.11 ms against
  a 0.80 ms IQR (#60's drift check had 303.82 against 299.70 ms), `sm_86` 9
  over 12 by 4.28 ms against 0.74 ms.
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

### Measured: the asymmetric split, measured and refused (#113)

`[JEngine, v2, §5]` calls splits that break the three-way symmetry
"unnecessary and worse". The generator still **refuses** such a count
(`spec.is_batch_count`, `partition`); `felizjueves.asym` is a study path that
builds the asymmetric batch list itself, checks it partitions the class, and
keys its rows `.../b<n>/j/asym@...` beside the literal ones, which it does not
touch. Its docstring is the rule, committed before any asymmetric row existed:
L ∈ {13, 16} × n ∈ {4, 5, 7, 8}; per-direction counts minimax by the paper's
own criterion (then the next-largest batch, ties to x); each cell timed in one
session against its nearest legal counts `3⌊n/3⌋` and `+3`, on both of #53's
inputs; an input is *slower* / *faster* against the better neighbour by more
than the larger IQR (#60's rule); a cell is WORSE if slower on one input and
faster on none. Medians in ms, asymmetric arm against its best neighbour:

| arch | L | n (kx, ky, kz) | wide-T | near-T |
|---|---|---|---|---|
| `sm_86` | 13 | 4 (2, 1, 1) | 183.4 vs b3 153.2: slower | 31.3 vs b3 30.1: slower |
| `sm_86` | 13 | 5 (2, 2, 1) | 208.1 vs b3 151.0: slower | 32.6 vs b3 29.7: slower |
| `sm_86` | 13 | 7 (3, 2, 2) | 275.0 vs b6 237.2: slower | 36.7 vs b6 34.3: slower |
| `sm_86` | 13 | 8 (3, 3, 2) | 308.1 vs b6 237.3: slower | 39.0 vs b6 34.3: slower |
| `sm_86` | 16 | 4 (2, 1, 1) | 350.0 vs b3 292.7: slower | 113.9 vs b6 108.9: slower |
| `sm_86` | 16 | 5 (2, 2, 1) | 409.1 vs b3 293.6: slower | 111.7 vs b6 109.5: slower |
| `sm_86` | 16 | 7 (3, 2, 2) | 501.2 vs b6 463.0: slower | 106.7 vs b9 100.2: slower |
| `sm_86` | 16 | 8 (3, 3, 2) | 539.2 vs b6 463.1: slower | 103.2 vs b9 100.2: slower |
| `gfx1200` | 13 | 4 (2, 1, 1) | 300.0 vs b3 260.4: slower | 80.2 vs b6 80.3: neutral |
| `gfx1200` | 13 | 5 (2, 2, 1) | 340.3 vs b3 260.1: slower | 80.7 vs b6 80.1: slower |
| `gfx1200` | 13 | 7 (3, 2, 2) | 432.6 vs b6 379.0: slower | 84.1 vs b6 85.3: neutral |
| `gfx1200` | 13 | 8 (3, 3, 2) | 486.1 vs b6 380.4: slower | 88.5 vs b6 80.3: slower |
| `gfx1200` | 16 | 4 (2, 1, 1) | 658.0 vs b3 582.2: slower | 307.3 vs b3 308.0: neutral |
| `gfx1200` | 16 | 5 (2, 2, 1) | 736.3 vs b3 582.3: slower | 310.1 vs b3 308.4: slower |
| `gfx1200` | 16 | 7 (3, 2, 2) | 867.2 vs b6 814.4: slower | 305.0 vs b9 303.8: slower |
| `gfx1200` | 16 | 8 (3, 3, 2) | 924.4 vs b6 813.9: slower | 304.9 vs b9 303.8: slower |

Every cell is WORSE on both cards — slower than the best legal count on wide-T
everywhere, never faster on either input — so
`jengine/symmetry-breaking-splits-are-worse` is **asserted**
(`test_measured.py::test_asymmetric_splits_are_worse_on_both_cards`). The
three near-T neutrals on `gfx1200` are where its near-T clock is flat in the
batch count (80–85 ms at L = 13, 304–310 ms at L = 16): there an asymmetric
split ties, and the wide-T loss decides. Spills and registers are reported
(`--asym-verdict`), never judged: on both cards every asymmetric cell's spill
sits strictly between its two neighbours'. Timed on 2026-10-12
behind the gate, one session per cell set; a phase whose after-reading showed
the card taken by something outside the container was discarded and retaken
(raw `wallclock:gate_discarded_phases`). Reproduce with
`python -m felizjueves.measured --remeasure --asym`, then the opt-in sweep with
`FJ_WALLCLOCK_ASYM=1`.

### Measured: each card's roofline ceilings (#109)

A roofline verdict (compute- or memory-bound) needs a **ridge**, the fp64 rate
over the memory bandwidth, and this repo takes both from **this box**, never
from a datasheet. `device.Arch` carries `fp64_flops_per_s` and
`mem_bandwidth_bytes_per_s`, `0.0` meaning *not characterised*; `Arch.ridge`
and `device.verdict` **refuse** (`Uncharacterised`) for a card without both.
The values are read from `data/measured/ceilings.json`, written by the probes
in `src/felizjueves/ceilings.py` — an fp64 FMA-saturating kernel (8 and 16
independent chains; the faster median wins) and STREAM's triad over three
256 MiB fp64 arrays, one portable translation unit — compiled with
`NVCC_FLAGS` / `HIPCC_FLAGS` and timed with device events, 5 warm-up and 31
timed launches, the median rate recorded.

**Behind the card gate.** `src/felizjueves/gate.py` is the gate the wall-clock
sweep used (lifted out of `test_kernel_device.py`, which now reads it): card
utilisation at most 30% and the CPU at most 10% busy before a phase, and the
settled reading after it. A contended probe does not add noise, it
under-reports — a lower ridge, every verdict pushed toward compute-bound — so a
phase the gate does not let through, or whose after-reading is non-idle or
unparseable, is **not recorded**, and that card stays uncharacterised. Each
record carries the probe, its source hash, the compile line, the toolchain,
host, date and the two card readings (`raw.gpu_util_before` / `_after`).
Reproduce with the record's `command`:
`env FJ_CEILINGS_STORE=data/measured/ceilings.json uv run pytest
test/test_ceilings_device.py -p no:xdist -s -rs -m 'gpu and <backend>' -k
record`, run alone.

**Achievable, not peak.** Each value is what a streaming / FMA-only kernel
reaches, recorded as `kind: "achievable"`, with its ratio to the vendor's
published figure (`published` in the data file, a sanity bound only: a rate
above it fails as a probe bug). `test/test_ceilings.py` checks each record's
provenance and bound, and nothing in `literature/` may read these fields.

Recorded on 2026-10-11, host `a37a1afdb77b (container)`, every phase let
through by the gate (card ≤ 22% before, ≤ 2% after):

| arch | fp64, flop/s | of published | bandwidth (triad), B/s | of published | ridge, flop/B |
|---|---|---|---|---|---|
| `sm_86` (RTX 3080, `nvcc 13.0`) | 4.39e11 | 94% of 0.465 T | 6.74e11 | 89% of 760 G | 0.65 |
| `gfx1200` (RX 9060 XT, `hipcc / ROCm 7.2.4`) | 4.46e11 | 56% of 0.801 T | 3.00e11 | 94% of 320 G | 1.49 |

`gfx1200`'s fp64 is slow by design; AMD publishes no fp64 figure for it, and
the 0.801 T denominator counts dual-issue fp32 (the data file's `note`). The
ceilings are observations of two cards on one day — **never compared against
the paper's A100**.

### Measured: the counter fields, and where a kernel's counters live (#110)

A roofline point needs the kernel's own **arithmetic intensity** — fp64 FLOPs
over memory bytes — read off the card's performance counters, never a model.
`Measured` carries `fp64_flops`, `dram_bytes_read`, `dram_bytes_written` and
`counter_ns` (the kernel's duration *under counter collection*, which
serialises and replays it — not a timing, never `runtime_ns`); intensity and
achieved FLOP/s are **derived** properties, never stored. `counter_tool`,
`counter_unit`, `command` and `raw` (every counter by its exact name) say
where each number came from. The driver (`felizjueves.measured.profile`) runs
`ncu` on NVIDIA (`fp64_flops = dadd + dmul + 2·dfma` thread instructions;
`dram__bytes_read/write.sum`) and `rocprofv3 --pmc` on AMD (GL2C↔fabric
requests × their size — L2 misses, an upper bound on DRAM bytes). **`gfx1200`
exposes no fp64 counter**, so an AMD row leaves `fp64_flops` unset: an AMD
intensity needs a FLOP count from elsewhere, labelled as such. A profiler
that is absent, refused the counters, or cannot load raises
`CouldNotMeasure` — no row, never a zero. (This container's `rocprofv3`
lacks `libdw.so.1` and `libhsa-amd-aqlprofile64.so.1`; `devtools/doctor.sh`
says so.)

**One row per batch kernel**, in their own store, `data/measured/roofline.json`,
keyed `MeasuredKey.key() + "#<batch>"` (0-based, plan order; a cell of
`n_batches` b has exactly `#0 … #b-1`). A cell's counters are **derived** —
summed over its kernels by `measured_store.cell_counters` — so a per-kernel
verdict (`[JEngine, v2, §7.2]`'s *some* kernels of a 9-batch `[7|7]` stay
memory-bound) and the per-cell one read the same numbers. `kernels.json`'s
literal rows are untouched: one row is still one cell there.

### Measured: roofline rows over the subgrid, `sm_86` only (#112)

`data/measured/roofline.json` holds **124 counter rows**: every batch kernel
of every `SUBGRID` cell on the RTX 3080 (`sm_86`, `nvcc 13.0`, `ncu
2025.3.1.0`). They were written by the opt-in sweep
`test_cuda_roofline_over_the_subgrid` (`env
FJ_ROOFLINE_STORE=data/measured/roofline.json uv run pytest
test/test_kernel_device.py -p no:xdist -s -rs -m "gpu and cuda" -k
roofline`, run alone). The sweep works as follows:

- It compiles each cell's batched harness, the binary the wall-clock sweep
  times. A cell whose emitted text differs from the text the `kernels.json`
  compile row hashed is refused.
- It checks every binary against the oracle at `J_REL`.
- It profiles one launch sequence per cell on the wide-T timing input
  (2048 × 128, τ = 0), so each batch kernel is launched once. The counter run
  is a **separate run** of the same binary: collection replays each kernel
  and may change occupancy, so `counter_ns` is never `runtime_ns`. No
  wall-clock cell was re-timed.
- Each profile runs behind `felizjueves.gate`. A phase whose after-reading is
  non-idle or unknown is discarded and retaken.

Every row carries the card's state (`raw` `roofline:gpu_util_before` /
`_after`, `roofline:cpu_busy_permille_before`,
`roofline:gate_discarded_phases`), the counters by their exact names and the
`ncu` invocation. The loader refuses a row without them, and it also refuses a
row for a card that has no measured ridge. On 2026-10-12, host
`a37a1afdb77b (container)`, the gate discarded **0** phases: every card
reading was ≤ 8% before and ≤ 7% after. The profiling took 140 s (the whole
sweep took 9 min 22 s, most of it serial compiles).

**The verdict is computed, never stored.** `measured_store.roofline_point`
divides `fp64_flops` by the DRAM bytes and asks `device.verdict` against the
measured ridge (#109: 0.65 FLOP/B on `sm_86`). `kernel_roofline` gives the
verdict per batch kernel and `cell_roofline` gives it per cell (counts summed).
`python -m felizjueves.measured --roofline` prints both.

| L | b = 1 | b = 3 | b = 6 | b = 9 | b = 12 |
|---|---|---|---|---|---|
| 7 | 300 C | 2542 C | 2893 C | 2327 C | 2391 C |
| 10 | 3.54 C | 40.2 C | 2255 C | 2086 C | 1935 C |
| 13 | 0.96 C | 4.95 C | 14.2 C | 47.2 C | 75.0 C |
| 16 | **0.40 M** | 1.52 C | 2.77 C | 4.80 C | 6.97 C |

Each cell shows its intensity in FLOP/B: C means compute-bound, M means
memory-bound. The achieved rate is 107–151 GFLOP/s in every cell, 24–35% of
the measured fp64 ceiling. **Per kernel**, all 124 kernels match their cell's
verdict: the one memory-bound kernel is L = 16's single unsplit kernel, and no
kernel of a split cell sits below the ridge. Kernel intensities within one
split cell spread widely. At L = 13 b = 12 they range from 23.6 to 1175
FLOP/B, because the batches whose working set stays in L2 hardly touch DRAM.

What these rows say (#114 scores them below, under its declared rule):

- On this card the ridge is low enough that only **L = 16, one batch** is
  memory-bound. Its spill traffic (70,628 bytes of spill stores per thread in `kernels.json`)
  puts it at 0.40 FLOP/B. Splitting into three batches already puts it past
  the ridge, at 1.52 FLOP/B.
- Below L = 16, every cell is compute-bound at one batch, so batching cannot
  move them across the ridge. They are already past it.

These are observations of one card, not compared to `[JEngine, v2, Fig. 7]`.

**`gfx1200`: could not measure. No rows exist, and that is not a zero.** The
reason is recorded in `measured_store.UNMEASURED` (the store header's
`could_not_measure`), and the loader refuses a `gfx1200` counter row. There
are two blockers:

- `rocprofv3` 1.1.0 cannot load `libdw.so.1` or
  `libhsa-amd-aqlprofile64.so.1` in this container.
  `docker/install-rocm.sh` now installs `libdw1` and `hsa-amd-aqlprofile`,
  but the image has not been rebuilt.
- More fundamentally, gfx1200 has no fp64 instruction counter in
  rocprofiler-sdk. A rebuilt image yields bytes but no measured FLOP count, so
  an AMD intensity needs a FLOP count from elsewhere, labelled as such.

**`L = 12` and `L = 14` stay out of `SUBGRID_LS`.** `L = 10` is already the
paper's `[5|5]`. The compile cost was measured, not guessed, by compiling the
ten cells `L ∈ {12, 14}` × b ∈ {1, 3, 6, 9, 12} into a scratch copy of the
store; `kernels.json` is untouched. Adding them costs 108 + 228 = **336
compile-seconds on `sm_86`** (+77% over the current subgrid's 439) and 184 +
352 = **537 on `gfx1200`** (+60% over 892). That sum is per pass over the
cells, and the kernel tier pays it more than once: the split cells and the
recycle cells are both drawn from `SUBGRID`. Adding the two values would also
oblige new literal wall-clock rows, recycle A/B rows and counter rows for 20
cells on each card. The roofline question does not need them: L = 13
(`(7,0|6,0)`) and L = 16 already bracket `[7|7]`, and the ridge crossing
appears only at L = 16.

### Measured: what batching buys on these cards — does it cross our ridge? (#114)

**The rule predates the rows; this section's code does not.** The rule is
issue #114's, created 2026-10-11T18:07:53Z and never edited. PR #134 wrote
the roofline rows and was merged 2026-10-12T04:27Z. `src/felizjueves/ridge.py`
was written after both, and after #134 had reported its summary. It copies
the issue's words into a function and adds only readings the repo already
defined (its docstring lists them). The issue's rule: for one arch and one L,
batching **crosses the ridge** iff the cell is memory-bound at
`n_batches = 1` and compute-bound at some higher `SUBGRID` count, with the
crossing count recorded. The **per-kernel variant** asks instead that every
kernel of that count's partition be compute-bound, and it is scored
separately. The two variants are reported side by side, and neither rescues
the other. Each card is scored on its own.

| L | `sm_86`, cell | `sm_86`, per kernel | `gfx1200` |
|---|---|---|---|
| 7 | compute-bound already at 1 batch (300 FLOP/B) | the same | could not measure |
| 10 | compute-bound already at 1 batch (3.54) | the same | could not measure |
| 13 | compute-bound already at 1 batch (0.96) | the same | could not measure |
| 16 | **crossed at 3** (0.40 → 1.52) | **crossed at 3** (every kernel ≥ 1.46) | could not measure |

The ridges are each card's own, measured on this box (#109): **0.65 FLOP/B on
`sm_86`** and **1.49 FLOP/B on `gfx1200`**. Fig. 7's A100 ridge is a
different number, and nothing here is compared with it.

**What batching buys on the RTX 3080, by the roofline:** a ridge crossing
only at L = 16, where the unsplit kernel's spill traffic holds it below the
ridge and three batches lift it past. Below L = 16, every class is past
the ridge at one batch, because consumer fp64 puts this card's ridge so low.
There is nothing for batching to cross there. Whatever batching buys at
those L (#56's near-T split from L = 13), it is not a move across the ridge.
That is a fact about this card. It says nothing against `[JEngine, v2, §7.2]`,
whose A100 measurement stands.

**On the RX 9060 XT the answer is absent.** `gfx1200` has no counter rows
(`measured_store.UNMEASURED`): `rocprofv3` cannot load in this container, and
gfx1200 has no fp64 counter. `ridge.crossing` returns *could not measure*,
never *never crossed*.

**Cross-read against #60** (9 batches do not suffice on either card). On
`sm_86`, every kernel of every 9-batch cell is compute-bound, so the roofline
**contradicts** the reading that 9 fails because kernels are still
memory-bound at 9. On this card 9 fails #60 on spills alone: 12 spills less at
L = 13 and 16, yet 12 is slower than 9 at every L on both inputs. The
roofline agrees with the clock: at 9 there is no ridge left to cross. On
`gfx1200`, where 12 wins on near-T at L = 16, the roofline can say nothing.

The record is the claim `jengine/batching-crosses-our-ridge` (asserted by
`test/test_ridge.py`, which recomputes every number above) and its entry in
`data/literature/jengine/limits.json`. That entry names what the comparison
cannot see: the ridge is the card's; the AMD answer is absent; the DRAM bytes
come from one input, with L2-resident batches; `[6|6]` is unmeasured. The
paper's own claim, `jengine/batching-crosses-the-ridge`, stays
`unverifiable`, and so does `jengine/nine-batches-suffice`.

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

## The Coulomb matrix, witnessed from outside

**Implemented, host-side, and compared against PySCF/libcint** (milestone 4,
#80–#89; #90 records it). `J_µν = Σ_λσ (µν|λσ) D_λσ` for a real molecule,
through the split above, with every piece but `J_p` in
`src/felizjueves/ao/`: shells and their Cartesian order (`shell.py`), the
undressed `E` (`hermite.py`), primitive-pair records, `dress` and the Schwarz
bound (`pairs.py`), and `J` by two sweeps that must agree (`j.py`). `J_p`
comes from one of three places, all behind the same `jp=` seam:
`reference.pq`; the generated kernel **interpreted** (`ao/drive.py`'s
`KernelJP`); or the generated kernel **compiled and launched** on the RTX 3080
and the RX 9060 XT (`test/test_ao_j_ondevice.py`, from the kernels
`python -m felizjueves.emit --class LA LB LC LD` writes). PySCF is never
imported by `src/`: the comparisons run in the fast tier against tracked
fixtures in `test/oracle/fixtures/`, and the `oracle` group only regenerates
them. Every comparison runs at `τ = 0`; the screen has its own section below.

| What | Against libcint | Test | Tier |
|---|---|---|---|
| shells, normalisation, `common_fac` | overlap `int1e_ovlp` | `test_oracle_basis.py` | fast |
| `(ab\|cd)`, one block per class, L ≤ 8 | `int2e`, `≤ 1e-12·scale` | `test_ao_eri.py` | s/p fast, d+ slow |
| `J_µν`, `reference.pq` | `get_jk`, `≤ 1e-12·scale_J` (derived from the ERI bound) | `test_ao_j.py` | H₂, H₂O STO-3G/6-31G fast; cc-pVDZ slow |
| `J_µν`, interpreted generated kernel | same | `test_ao_j_interpreted.py` | H₂ fast; H₂O ×3 slow |
| `J_µν`, generated kernel on each card | same | `test_ao_j_ondevice.py` | kernel tier (`gpu`); H₂, H₂O/STO-3G |

**What it proves, as records** (`data/literature/jengine/claims.json`):

- `jengine/theta-undefined` — `θ² = ζη/(ζ+η)` is the `θ` the kernel forms —
  is held by `test_ao_j_interpreted.py::test_interpreted_j_matches_reference_and_libcint`.
  Chosen over the on-device test because it runs without a card: the fast tier
  exercises it on H₂ (`θ` through `F_0` only) and the slow tier on three H₂O
  bases (`m > 0`); the on-device test corroborates on both cards but skips,
  green and empty, wherever there is none. **It fails when the claim is
  false:** with `θ` perturbed at every site that forms it — the kernel's
  prelude, `ao/j.py`, `ao/eri.py`, so the internal comparison against
  `reference.pq` still agrees to 0 — libcint alone catches it. A relative
  error of 1e-9 in `θ` gives `8.85e-10·scale_J` on H₂ and ~3.5e-9 on H₂O
  (STO-3G and 6-31G), against the `1e-12` tolerance, so the comparison
  resolves `θ` to about a part in 10¹²; the slip readings (`θ²` unrooted,
  `√(ζ+η)`, `√(ζη)`) give 0.35–0.50 `scale_J`.
- `jengine/prefactor-factorisation` — `λ = (2/√π)θ·(π/ζ)^{3/2}(π/η)^{3/2} =
  2π^{5/2}/(ζη√(ζ+η))`, placed as "Two defects" and "The `E` transform" say —
  is held by `test_ao_eri.py::test_block_matches_libcint`, per class: `dress`
  scaled by `1 + 1e-9`, or `(π/ζ)^1` for `(π/ζ)^{3/2}`, fails every fast-tier
  block.

**The search for a compensating error, and what it found.** libcint sees only
products, so the question is which wrong pair of factors multiplies out right.

- *A constant moved between the fundamental and `dress`* (`dress·2`,
  `2/√π ÷ 4`): every ERI block and `J` still match libcint, by construction.
  It is **not** a blind spot of the repo: each half has an in-repo absolute
  pin that fails — the fundamental against the closed form `erf(θR)/R`
  (`test_theta_convention.py`) and its defining integral
  (`test_reference_pq.py`), the dressing against an independent overlap
  (`test_ao_pairs.py`).
- *A `θ` error compensated elsewhere*: `θ` enters both the prefactor and the
  Boys argument `θ²|PQ|²`, and no constant cancels a change in the second
  across geometries and exponents; perturbing it fails as above.
- *A factor shared unequally between bra and ket*: one `dress` serves both
  sides, so only a factor with `f(ζ) f(η) = 1` for every pair could hide —
  `f = ±1`, and a sign on both sides is not an error in any observable.

So nothing is added to `limits.json`: the search found no compensating error
the comparisons leave open. What they do not **reach** is a coverage boundary,
not a structural limit, and it is stated here instead:

- **Classes with L = la+lb+lc+ld ≥ 9.** The ERI fixture stops at L = 8 (the
  nine HF/cc-pVTZ classes above it exceed the repo's 512 KB file hook), and
  the `J` fixtures stop at `d` (cc-pVDZ, so L ≤ 8). The kernel ladder runs to
  L = 16; above 8 it is checked only against `reference/`, which shares its
  ancestry. `θ` is formed once per pair by the same prelude at every L, so
  the record does not depend on L, but a recurrence defect that only appears
  above L = 8 would be invisible to libcint here. A `J` fixture with `f`
  functions is small (`n_ao²`), so this boundary can move.
- **The device's screen.** The capstone launches with `Ip = Iq = Dmax = 1`
  and `τ = 0`; sorted kets with real bounds on a card are the next milestone's
  driver. The host-side screen is #88's ladder, below.

## The screen as an error claim: a `τ` ladder (#88)

**Implemented, host-side.** `ao.j.coulomb_matrix_screened(basis, D, *, pq,
tau, order)` is the unique-pairs `J` sweep with the kernel's screen: a
(bra record, ket record) pair is skipped when `I_bra · (I_ket · Dmax) < τ`,
`I` the pair-factor-free bound of `ao/pairs.py` and `Dmax` over the weighted,
dressed ket vector. `order="sorted"` is the kernel's form — kets sorted
**descending by `I_ket · Dmax`, stable**, and a `break` at the first ket under
`τ`; `order="unsorted"` is the `continue` in the original order.
`coulomb_matrix` and `coulomb_matrix_reference` are unchanged (at `τ = 0` they
still evaluate no bound).

**Why the `break` is exact.** `I_bra ≥ 0`, and a rounded product with a fixed
non-negative factor is monotone in the other, so on kets sorted by exactly the
factor the screen multiplies, every ket after the first rejected one is
rejected too. The sort must be by that product and nothing else, and stable
(the H₂ chain's symmetry produces tied bounds, and the test sees them).

**The three claims**, each at every rung `τ ∈ {0, 1e-14, 1e-12, 1e-10, 1e-8,
1e-6}`, in `test/test_ao_screening.py`:

- **Sorted-`break` = unsorted-`continue`, to 0**, and the same skip count —
  `test_kernel.py`'s claim on synthetic data, here on real bounds. Each `J_p`
  is accumulated over kets with `math.fsum` (correctly rounded, so independent
  of order), which makes the equality a statement about *which* terms were
  summed rather than about summation order.
- **`|J(τ) − J(0)| ≤ τ · scale`**, element by element, with the scale stated
  and independent of `τ`: `scale_ab = n_ket · Σ_{bra records} Σ_p |E^{ab}_p|`.
  Every skipped pair has `|Σ_q [p|q] D_q| ≤ I_p I_ket Dmax ≤ I_bra I_ket Dmax <
  τ` for every `p`, moves `J_ab` by at most `τ Σ_p |E^{ab}_p|`, and a bra
  record skips at most `n_ket` kets. `J(0)` is the same path at `τ = 0`.
- **The error is non-decreasing in `τ`** (as is the skip count: the skipped
  sets are nested).

**The cases.** *Compact*: H₂O/STO-3G, the `h2o_sto3g_j` fixture's basis and
density. *Extended*: four H₂ (bond 1.4 bohr), centres 6 bohr apart, collinear,
STO-3G H from the `h2_sto3g_j` fixture and the fixtures' hilbert density
`D_ij = 1/(i+j+1)` — every `D_cd` non-zero, so a skip comes from the decay of
the ket's dressed `E`, not from a zero. Four molecules and not more: four reach
`T = θ²|PQ|²` ≈ 1289 and a fifth ≈ 2210, past the `T ≤ 2000` that
`reference.boys` has measured and refuses beyond; the test pins that the chain
is as long as the evaluator allows. (#88 ran two H₂ under the original
`T ≤ 500` cap; #122 gave `reference.boys` a measured large-`T` branch — the
asymptote `Γ(m+½)/(2T^(m+½))` from `T = 125`, where its truncation for every
`m ≤ 40` is under 1e-18, to `T = 2000`, its error under 8e-16 against the
Kummer series — and the ladder was re-run. The bound that bites is `T_MAX`,
not `M_MAX`: the chain is all s shells, so only `F_0` is ever asked for. The
device-side `fj_boys` is untouched by that widening.)

**Measured** (`data/ao/screening.json`; `err/scale` is the maximum over
elements of `|J(τ) − J(0)| / scale`, and the claim is that it is `≤ τ`):

| case | `τ` | max `\|J(τ) − J(0)\|` | err/scale | skip fraction |
|---|---|---|---|---|
| compact | 0 | 0 | 0 | 0.0000 |
| compact | 1e-14 | 0 | 0 | 0.0000 |
| compact | 1e-12 | 0 | 0 | 0.0000 |
| compact | 1e-10 | 0 | 0 | 0.0000 |
| compact | 1e-8 | 4.4e-9 | 3.0e-11 | 0.0023 |
| compact | 1e-6 | 3.1e-7 | 2.3e-9 | 0.0211 |
| extended | 0 | 0 | 0 | 0.0000 |
| extended | 1e-14 | 1.2e-14 | 4.0e-17 | 0.2651 |
| extended | 1e-12 | 1.4e-12 | 6.2e-15 | 0.2980 |
| extended | 1e-10 | 3.0e-10 | 1.3e-12 | 0.3436 |
| extended | 1e-8 | 1.7e-8 | 6.4e-11 | 0.4047 |
| extended | 1e-6 | 2.9e-6 | 9.1e-9 | 0.4664 |

**Read it as follows.** The compact molecule skips **nothing** at
`τ ≤ 1e-10`: there the error claim is vacuous (the error is exactly 0 because
nothing was dropped), and the test says so rather than passing quietly. The
extended case skips at every positive rung — 27% of record pairs at `1e-14`,
47% at `1e-6` (two H₂ skipped 5% and 15%: the longer chain has more distant
pairs to drop) — so the bound does reject whole inter-molecule blocks, and
nothing goes to `limits.json`. The stated scale is loose by design (it charges
every bra record for every ket); `err/scale` sits about two orders (1.9 to
2.6) under `τ`. Accuracy only: no timings and no `Measured` rows; the work saved is the
skip fraction. Regenerate with
`PYTHONPATH=src uv run python test/test_ao_screening.py --write`; the `slow`
test requires the file to equal a fresh run exactly.
