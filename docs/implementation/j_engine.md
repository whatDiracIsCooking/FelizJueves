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
  built from it as IR by `src/felizjueves/mmd/recurrence.py`.
- *Step 5, the kernel*, is `build_pq_kernel` in `src/felizjueves/mmd/kernel.py` —
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
  (the kernel tier's `KERNEL_GEN_CMD`) still writes one-batch `[p|q]`
  kernels only, beside the chain's pair kernels (#156, below).

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

**Implemented, around the kernel: a whole molecule's `J` from a launch plan**
(milestone 6, #138–#147). `src/felizjueves/launch.py` is the plan as a
value. It has one `ClassLaunch` per `(lp, lq)`: bra records in pair order,
ket records sorted descending by `I_q · max|D_q|` (`[JEngine, v2, §3.3]`'s
key), and the class's batch kernels. Three things walk it. The launch oracle
(`launch.walk`) is pure Python. The staged driver (`ao/staged.py`) runs the
kernels interpreted. A device `main()`, generated from the plan by
`test/plan_driver.py`, uploads each bra and ket class once, keeps one `J_p`
buffer per bra class and launches every batch kernel into it with fp64
atomics. That device build ran on **both cards** and was compared against the
oracle and the libcint fixtures. `src/felizjueves/performance/dispatch.py` chooses a
class's batch count from measured wall-clock and is injected into the plan.
See "The launch plan" below.

**Implemented since milestone 7: every arithmetic step of `J` as a generated
kernel** (#149–#157). The `E` transform is a device-code fragment
(`e_fragment.py`), inlined into two new kernel families: the ket
pre-contraction `Target.DQ` (`kernel_dq.py`) and the bra back-transform and
scatter `Target.JAB` (`kernel_jab.py`). `KERNEL_GEN_CMD` writes both beside
the `[p|q]` ladder. The chain runs interpreted against the host and libcint,
and compiled and launched on both cards: pair kernels through (ff), and a
whole molecule's `J` through d. The host-side `ao/` paths stay as the
references the kernels are held to. See "The J chain as device code" below,
the one place that says which steps are generated and where they ran.

**The one-electron matrices** (milestone 11) reuse this engine: `S`, `T`
and `V` on the same `E` fragment, `V` as the `[p|q]` recurrence with the
ket replaced by a unit point charge (`θ → √ζ`). They are not part of `J`
and nothing about them is scored against `[JEngine]`;
[`one_electron.md`](one_electron.md) is their note.

**Still host-side, or deliberately absent.** Host-side: the pair list, each
record's inputs (`ζ`, `X_PA`, `X_PB`, the dressing scale), the Schwarz bound,
the sort with its `Dmax` read-back, the staging, and the launch code (a
test-side driver). Nothing about the new kernels is measured. Absent: a `J`
kept resident on the card across SCF iterations, a `Launch` IR, and a
`LaunchCost` model, all declared out of scope (`limits.json`). Dispatch
exists as a library choice the plan takes, and the generator does not follow
it: `Spec.n_batches` still defaults to 1, and `KERNEL_GEN_CMD` writes
one-batch `[p|q]` kernels, by decision ("What the generator ships, and why"
below). A count this file states is a claim about the code only where a test
or a scored board checks it.

**What the generator ships** (milestone 8, #159–#174): the **literal** plan
(the minimax cut and the sieve's fixed seed order, `Spec.levers = ()`), one
batch, plus the pair ladder. The levers measured against it on both cards
are `recycle` (#55), `asym` (#113, a study path), `no-sieve` (#162) and
`tuned` (#163), and none moved the default: the declared verdicts are
neutral, worse, NOT SHOWN and NOT SHOWN. The plan-layer counts do not
predict the kernel on either card (#165), so they are not grounds either way.
"What the generator ships, and why (#166)" below has the decision, and
"Parity with the predecessor" the one difference from `jgen`.

**Measured, and recorded** (milestones 5 and 6): each card's roofline
ceilings, counter rows on `sm_86` (none on `gfx1200`, final since #140), and
the answer to whether batching crosses our ridge: "The roofline, in one place"
below. The whole-build wall-clock and the sort-key A/B on both cards: "The
launch plan" below. The status of every `[JEngine]` claim: "The `[JEngine]`
claims" below, rendered from `claims.json` and checked by
`test/test_claims.py`.

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
deferred by `[JEngine, v2]` to `[Fock]`, which this repo has not read. This
repo takes only as much of them as a `J` build needs. They were first written
host-side, in `src/felizjueves/ao/j.py` (milestone 4; "The Coulomb matrix,
witnessed from outside" below), and that host path is still the reference.
Since milestone 7 they are also **generated as device code**: the `DQ` and
`JAB` kernels ("The J chain as device code" below). Both are this repo's own
design, not a reconstruction of `[Fock]` or of anything else
(`limits.json`).

Screening: Cauchy–Schwarz bounds take the nominal `O(N⁴)` to `O(N²)`. Each
Hermite index carries a bound `I`, and the kernel tests `D_max · I_p · I_q`
against a threshold `τ`.

## The `E` transform

**Status: implemented host-side, and generated as device code (milestone
7).** The host reference is `src/felizjueves/mmd/hermite.py` (`e_axis`,
`e_block`), checked by `test/test_ao_hermite.py`; the dressing is
`ao.pairs.dress`, and it stays host-side; the host bra back-transform
`J_ab = Σ_p E_p^{ab} J_p` is in `ao/j.py`. The device form is
`src/felizjueves/mmd/e_fragment.py` (#152): an `ir.Fragment` of this recurrence,
undressed, bitwise `==` `e_block` for every class (ss) through (ff) when
interpreted (`test/test_e_fragment.py`). It is inlined into the `DQ` and `JAB`
kernels, which ran on both cards ("The J chain as device code" below).

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
  On our cards, under a definition declared before the data
  (`measured_store.py`'s docstring, NINE AGAINST TWELVE), 9 does **not** suffice on either: 12 spills less on
  `sm_86` at L 13 and 16, and is faster on near-T on `gfx1200` at L 16
  (`jengine/nine-batches-suffice` in `data/literature/jengine/claims.json`,
  UNVERIFIABLE against their A100). So the measured subgrid runs to 12.
- **The default is one batch, by decision (#167).** `Spec.n_batches`
  defaults to 1, and the measured recommendation below is the reason it
  stays there rather than a policy waiting to be applied ("The batch count
  it ships", under "What the generator ships, and why").

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
`PYTHONPATH=src uv run python -m felizjueves.performance.batchreport --write` from
`data/measured/kernels.json`; the tables below are its `--markdown`, and
`measure/measure_batchreport.py` (`devtools/measure.sh check`) recomputes both from the store and fails if either
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
  median" is therefore not one number per `(L, arch)`, and dispatch has to
  be told which regime of `T` it is serving (`dispatch.py` takes the input
  as a required argument, #145).
- **The arches disagree.** On near-T at `L = 16`, `sm_86` is fastest at 9 and
  `gfx1200` at 12; at `L = 10`, `gfx1200` already prefers 3 (unresolved
  against 1) where `sm_86` keeps 1. That is the reason dispatch is per
  target — not a disagreement to average away.
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
- **Literal kernels only.** The table reads the `levers = ()` rows, which
  are the kernels the generator ships (#166). `recycle` (#55) changes
  nothing it could rank: nvcc 13.0 and hipcc / ROCm 7.2.4 emit identical
  machine code for both arms at every subgrid cell, and the A/B reads
  `sm_86` 20/20 neutral and `gfx1200` 16/20 neutral (4 read as slower on
  that identical code, the protocol's noise floor). It stays opt-in. The
  plan levers `no-sieve` (#162) and `tuned` (#163) were timed against their
  same-session literal controls and are not read here either; neither
  moved the default ("What the generator ships, and why (#166)" above).

**This does not change `Spec`'s default, by decision.** `n_batches` defaults
to 1 and stays there (`measure/measure_batchreport.py` holds it, and its
`about` field gives these rows as the grounds; "The batch count it ships"
above has the override recipe). Choosing a count per class at
generation time is **dispatch**. Since #145 it is `src/felizjueves/performance/dispatch.py`:
this table's rule (best median, ties to fewer batches), recomputed from the
`Measured` rows per `(arch, class, input)`, with 1 for any class that has no
row. It is injected into a launch plan (`launch.launches(resolve=)`), and the
generator does not follow it: `KERNEL_GEN_CMD` still writes one-batch
`[p|q]` kernels.

### Measured: the asymmetric split, measured and refused (#113)

`[JEngine, v2, §5]` calls splits that break the three-way symmetry
"unnecessary and worse". The generator still **refuses** such a count
(`spec.is_batch_count`, `partition`); `felizjueves.performance.asym` is a study path that
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
(`measure/measure_store.py::test_asymmetric_splits_are_worse_on_both_cards`). The
three near-T neutrals on `gfx1200` are where its near-T clock is flat in the
batch count (80–85 ms at L = 13, 304–310 ms at L = 16): there an asymmetric
split ties, and the wide-T loss decides. Spills and registers are reported
(`--asym-verdict`), never judged: on both cards every asymmetric cell's spill
sits strictly between its two neighbours'. Timed on 2026-10-12
behind the gate, one session per cell set; a phase whose after-reading showed
the card taken by something outside the container was discarded and retaken
(raw `wallclock:gate_discarded_phases`). Reproduce with
`devtools/measure.sh remeasure --asym`, then the wall-clock sweep with
`FJ_WALLCLOCK_ASYM=1 devtools/measure.sh wallclock --backend <backend>`.

### Measured: each card's roofline ceilings (#109)

A roofline verdict (compute- or memory-bound) needs a **ridge**, the fp64 rate
over the memory bandwidth, and this repo takes both from **this box**, never
from a datasheet. `device.Arch` carries `fp64_flops_per_s` and
`mem_bandwidth_bytes_per_s`, `0.0` meaning *not characterised*; `Arch.ridge`
and `device.verdict` **refuse** (`Uncharacterised`) for a card without both.
The values are read from `data/measured/ceilings.json`, written by the probes
in `src/felizjueves/performance/ceilings.py` — an fp64 FMA-saturating kernel (8 and 16
independent chains; the faster median wins) and STREAM's triad over three
256 MiB fp64 arrays, one portable translation unit — compiled with
`NVCC_FLAGS` / `HIPCC_FLAGS` and timed with device events, 5 warm-up and 31
timed launches, the median rate recorded.

**Behind the card gate.** `src/felizjueves/performance/gate.py` is the gate the wall-clock
sweep used (lifted out of `test_kernel_device.py`, which now reads it): card
utilisation at most 30% and the CPU at most 10% busy before a phase, and the
settled reading after it. A contended probe does not add noise, it
under-reports — a lower ridge, every verdict pushed toward compute-bound — so a
phase the gate does not let through, or whose after-reading is non-idle or
unparseable, is **not recorded**, and that card stays uncharacterised. Each
record carries the probe, its source hash, the compile line, the toolchain,
host, date and the two card readings (`raw.gpu_util_before` / `_after`).
Reproduce with `devtools/measure.sh ceilings --backend <backend>`, run
alone (the records' `command` field still names the pre-`measure/` pytest
line it was taken with).

**Achievable, not peak.** Each value is what a streaming / FMA-only kernel
reaches, recorded as `kind: "achievable"`, with its ratio to the vendor's
published figure (`published` in the data file, a sanity bound only: a rate
above it fails as a probe bug — except `gfx1200`'s fp64, bounded per counted
GPU cycle instead, below). `test/test_ceilings.py` checks each record's
provenance and bound, and nothing in `literature/` may read these fields.

Recorded on 2026-10-11, host `a37a1afdb77b (container)`, every phase let
through by the gate (card ≤ 22% before, ≤ 2% after); `gfx1200`'s fp64
re-recorded on 2026-10-12 by #141 (card 4% before, 0% after, and 0% / 0%
around its cycle-counter phase):

| arch | fp64, flop/s | of published | bandwidth (triad), B/s | of published | ridge, flop/B |
|---|---|---|---|---|---|
| `sm_86` (RTX 3080, `nvcc 13.0`) | 4.39e11 | 94% of 0.465 T | 6.74e11 | 89% of 760 G | 0.65 |
| `gfx1200` (RX 9060 XT, `hipcc / ROCm 7.2.4`) | 4.46e11 | 111% of 0.4006 T (rated clock); 99.96% per clock | 3.00e11 | 94% of 320 G | 1.49 |

`gfx1200`'s fp64 is slow by design, and AMD publishes no fp64 figure for it.
**Settled by #141, the #116 assumption retired.** The old 0.801 T bound was
AMD's 25.6 TFLOPS fp32 at 1:32, counting RDNA4's fp32 dual-issue as if it
applied to fp64. It does not: the probe is plain `v_fma_f64` (VOPD dual-issue
has no f64 form), and a `GRBM_GUI_ACTIVE` cycle count (`rocprofv3`, behind the
gate) puts it at **127.95 flop per GPU cycle** for 16 chains and 127.93 for 8
— the single-issue 32 CUs × 2 fp64 FMA lanes × 2 = 128, at half what
dual-issue would allow. So `gfx1200`'s fp64 bound is now **per clock**
(`published.gfx1200.per_clock`, checked by `ceilings.check_per_clock` on every
counted launch), and its flop/s figure, 0.4006 T, is that rate at the 3130 MHz
rated boost. The recorded ceiling is **above** that figure because the card
ran the probe at a counted ~3.49 GHz (amd-smi's own `MAX_CLK` reads 2700 MHz):
the one row whose value legitimately exceeds its flop/s figure, reported and
not clamped. The ridge reads the recorded ceiling, never a published figure
or bound (`device.verdict`), so 1.49 FLOP/B is the card at that clock. The
ceilings are observations of two cards — **never compared against the paper's
A100**.

### Measured: the counter fields, and where a kernel's counters live (#110)

A roofline point needs the kernel's own **arithmetic intensity** — fp64 FLOPs
over memory bytes — read off the card's performance counters, never a model.
`Measured` carries `fp64_flops`, `dram_bytes_read`, `dram_bytes_written` and
`counter_ns` (the kernel's duration *under counter collection*, which
serialises and replays it — not a timing, never `runtime_ns`); intensity and
achieved FLOP/s are **derived** properties, never stored. `counter_tool`,
`counter_unit`, `command` and `raw` (every counter by its exact name) say
where each number came from. The driver (`felizjueves.performance.measured.profile`) runs
`ncu` on NVIDIA (`fp64_flops = dadd + dmul + 2·dfma` thread instructions;
`dram__bytes_read/write.sum`) and `rocprofv3 --pmc` on AMD (GL2C↔fabric
requests × their size — L2 misses, an upper bound on DRAM bytes). **`gfx1200`
exposes no fp64 counter**, so an AMD row leaves `fp64_flops` unset: an AMD
intensity needs a FLOP count from elsewhere, labelled as such. A profiler
that is absent, refused the counters, or cannot load raises
`CouldNotMeasure` — no row, never a zero. So does a reading whose byte
counters all read 0 (`measured.ROCPROF_ZERO_BYTES`, #140). The AMD byte
counters are read in two passes (`measured.ROCPROF_PASSES`, at most four
counters each), because asking gfx1200's GL2C for all five at once aborts
`rocprofv3` with error 38, and the profiled process then hangs; each pass has
a timeout. (`rocprofv3` now loads in this container; whether it can is
`devtools/doctor.sh`'s to say.)

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
`test_cuda_roofline_over_the_subgrid` (`devtools/measure.sh roofline
--backend cuda`, run alone; `measure/measure_kernel_device.py`). The sweep works as follows:

- It compiles each cell's batched harness, the binary the wall-clock sweep
  times. A cell whose emitted text differs from the text the `kernels.json`
  compile row hashed is refused.
- It checks every binary against the oracle at `J_REL`.
- It profiles one launch sequence per cell on the wide-T timing input
  (2048 × 128, τ = 0), so each batch kernel is launched once. The counter run
  is a **separate run** of the same binary: collection replays each kernel
  and may change occupancy, so `counter_ns` is never `runtime_ns`. No
  wall-clock cell was re-timed.
- Each profile runs behind `felizjueves.performance.gate`. A phase whose after-reading is
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
`python -m felizjueves.performance.measured --roofline` prints both.

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
`could_not_measure`), and the loader refuses a `gfx1200` counter row.
**Since #140 the refusal is final.** It used to be blamed on the container:
`rocprofv3` 1.1.0 could not load `libdw.so.1`. It loads now (ROCm 7.2.4,
rocprofiler-sdk 1.1.0), and loading is not measuring. Two reasons remain,
and neither is the image:

- **No fp64 counter.** In rocprofiler-sdk's counter definitions,
  `SQ_INSTS_VALU_{ADD,MUL,FMA}_F64` (and the derived `TOTAL_64_OPS` and
  `SQ_INSTS_VALU_FLOPS_FP64`) exist for CDNA only (gfx90a, gfx940–942,
  gfx950). The card's own `--list-avail` has none, and `SQ_INSTS_VALU` counts
  every VALU instruction regardless of type. Without a measured FLOP count
  there is no measured intensity. A FLOP count from the model or the
  interpreter would be a model, not a counter.
- **The byte counters read 0 on this box.** Every GL2C ↔ EA counter reads
  exactly 0 for a probe that streams 384 MiB, while `SQ_WAVES` and
  `GRBM_GUI_ACTIVE` count, so collection itself works. The driver refuses
  that reading rather than store it. The root cause was not found (amdgpu
  module 6.16.13, host Linux 6.8).

`test_toolchain.py` holds both. One test fails if the byte counters ever
count. The other reads the card's `--list-avail` and fails if an fp64
counter appears.

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
the roofline rows and was merged 2026-10-12T04:27Z. `src/felizjueves/performance/ridge.py`
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
(`measured_store.UNMEASURED`, final since #140): `rocprofv3` loads, but
gfx1200 has no fp64 counter and its byte counters read 0 on this box.
`ridge.crossing` returns *could not measure*,
never *never crossed*.

**Cross-read against #60** (9 batches do not suffice on either card). On
`sm_86`, every kernel of every 9-batch cell is compute-bound, so the roofline
**contradicts** the reading that 9 fails because kernels are still
memory-bound at 9. On this card 9 fails #60 on spills alone: 12 spills less at
L = 13 and 16, yet 12 is slower than 9 at every L on both inputs. The
roofline agrees with the clock: at 9 there is no ridge left to cross. On
`gfx1200`, where 12 wins on near-T at L = 16, the roofline can say nothing.

The record is the claim `jengine/batching-crosses-our-ridge` (asserted by
`measure/measure_ridge.py`, which recomputes every number above) and its entry in
`data/literature/jengine/limits.json`. That entry names what the comparison
cannot see: the ridge is the card's; the AMD answer is absent; the DRAM bytes
come from one input, with L2-resident batches; `[6|6]` is unmeasured. The
paper's own claim, `jengine/batching-crosses-the-ridge`, stays
`unverifiable`, and so does `jengine/nine-batches-suffice`.

### Measured: three instruments, one subgrid — the cross-read (#115)

The repo ranks a batch count three ways. Each has its own section above and
its own test:

- **the model**, `KernelCost.traffic` summed over a cell's batch kernels
  (#54). Its *pick* is the set of counts at its minimum, ties kept;
- **the clock**, the best-median count on each input (#56);
- **the counters**, through #114's crossing rule. The *first compute-bound
  count* is 1 when a class is compute-bound already at one batch, and the
  crossing count when it crosses.

`src/felizjueves/performance/crossread.py` reads them against each other per
`(L, arch)`. `data/measured/crossread.json` is the table, and
`measure/measure_kernelrank.py` recomputes it. The counters were taken on the
**wide-T** input only, so only the wide clock flag compares like with like.
The near flag sets a near-T clock against a wide-T verdict. `gfx1200` has no
counter rows (`measured_store.UNMEASURED`). Its crossing column says *could
not measure*, and every flag that needs the counters is `n/a` there: an
absence, never a disagreement and never a zero.

**The choice is the clock's, everywhere.** `choice` is the clock's best and
nothing else. The test swaps the model for its reverse and checks that no
choice moves. The model and the counters choose nothing. The counters can
only explain a choice or fail to. Four L on two cards gives an order, never a
prediction interval.

| arch | L | model pick (traffic, summed) | clock wide | clock near | first compute-bound (counters, wide-T) | model = clock (wide, near) | clock = crossing (wide, near) | model ∋ crossing |
|---|---|---|---|---|---|---|---|---|
| sm_86 | 7 | 3, 6, 9, 12 | **1** | **1** | already at 1 | **no**, **no** | yes, yes | **no** |
| sm_86 | 10 | 3, 6, 9, 12 | **1** | **1** | already at 1 | **no**, **no** | yes, yes | **no** |
| sm_86 | 13 | 6, 9, 12 | **1** | **3** | already at 1 | **no**, **no** | yes, **no** | **no** |
| sm_86 | 16 | 12 | **1** | **9** | crossed at 3 | **no**, **no** | **no**, **no** | **no** |
| gfx1200 | 7 | 3, 6, 9, 12 | **1** | **1** | could not measure | **no**, **no** | n/a, n/a | n/a |
| gfx1200 | 10 | 6, 9, 12 | **1** | **3** | could not measure | **no**, **no** | n/a, n/a | n/a |
| gfx1200 | 13 | 9, 12 | **1** | **6** | could not measure | **no**, **no** | n/a, n/a | n/a |
| gfx1200 | 16 | 12 | **1** | **12** | could not measure | **no**, yes | n/a, n/a | n/a |

(Flags are for #114's cell variant. The per-kernel variant gives the same
first compute-bound count at every L on `sm_86`, and its flags are in the data
file.)

**Does the clock's count coincide with the first compute-bound one?** On
`sm_86`, only where nothing crosses. At L = 7, 10 and 13 on wide-T, and at
L = 7 and 10 on near-T, both say 1. One batch is already past the ridge and
the clock does not split, so that agreement shows nothing about the
mechanism. Where the ridge *is* crossed (L = 16, at 3), the clock disagrees
on both inputs:

- **wide-T: crossing the ridge makes the kernel slower.** Three batches lift
  L = 16 from 0.40 to 1.52 FLOP/B. They get there by adding work as well as by
  removing traffic: counted fp64 FLOPs rise 1.55x while DRAM bytes fall 2.47x.
  Past the ridge the added FLOPs are paid in full, and the median rises 1.84x.
  So 1 stays fastest. Intensity is a ratio, and batching raises it partly by
  raising the numerator.
- **near-T: the clock improves past the crossing, and also without one.** At
  L = 16 the clock keeps improving up to 9, though every count from 3 is
  already past the ridge. At L = 13 it prefers 3 although one batch is already
  compute-bound. Both gains follow the spill falling, not a crossing. Both
  also set a near-T clock against a wide-T verdict: no near-T counter row
  exists.

So on this card the paper's implicit mechanism (the clock improves *because*
the ridge is crossed) is **not** what the measurements show. Where batching
pays (near-T, L ≥ 13), it pays through spills on a kernel that is already
compute-bound. Where the ridge is crossed, it costs. This says nothing about
`[JEngine, v2, §7.2]`'s A100, whose ridge is a different number.

**The model never names the clock's or the counters' count**, except once.
Its summed traffic only falls with `n_batches` and has no term for the
per-batch redundant work, so it never picks 1. Its one agreement with the
clock (gfx1200, L = 16, near-T, both 12) is the edge of the subgrid, not a
correct ranking. Against the counters it answers a different question. The
crossing count is a threshold, the first count that clears the ridge. The
model's pick is a minimum, and a register-traffic model carries no ridge.

Each kind of disagreement is a `StructuralLimit` in
`data/measured/limits.json` (`asserted_value.cross_read = true`), with its
mechanism and the numbers above. The test fails on any disagreement that no
record covers, and on any record that is no longer a disagreement.

### The roofline, in one place: what it sees, what it answered, what it cannot see

The sections from #109 to #115 above are the detail. This is the summary the
milestone closes on (#116).

**What the repo can see.** Arithmetic intensity, measured: fp64 FLOPs (`ncu`'s
`dadd + dmul + 2·dfma` thread instructions) over DRAM bytes
(`dram__bytes_read/write.sum`), per batch kernel, in
`data/measured/roofline.json` (124 rows, every kernel of every `SUBGRID` cell
on `sm_86`), against each card's **measured** ceilings
(`data/measured/ceilings.json`). A card without both ceilings gets no verdict
(`device.verdict` refuses), and the verdict is computed from the rows, never
stored.

**What it answered.** `[JEngine, v2, §7.2]`'s question — does batching move a
`[p|q]` kernel across the ridge? — asked of our kernels on our cards, under
the rule #114 declared before the rows existed:

- **`sm_86` (RTX 3080, ridge 0.65 FLOP/B):** a crossing only at L = 16, at 3
  batches (0.40 → 1.52 FLOP/B, both the cell and every kernel). L = 7, 10 and
  13 are compute-bound **already at one batch**, so there is nothing for
  batching to cross: that is the card's low fp64 ridge speaking, not batching.
  Where the ridge *is* crossed, the wide-T clock gets slower (the crossing
  adds FLOPs as well as removing bytes); where splitting pays (near-T,
  L ≥ 13) it pays through spills on a kernel already past the ridge (#115).
- **`gfx1200` (RX 9060 XT, ridge 1.49 FLOP/B): the answer is absent.** By
  user decision #112 measured CUDA only, and #140 made the refusal final.
  `rocprofv3` now loads in this container, but gfx1200 exposes no fp64
  counter, so it gives no measured FLOP count, and its GL2C byte counters
  read 0 on this box, so it gives no bytes either. No rows exist, the loader
  refuses one, and every verdict on this card is *could not measure*, never
  "never crossed" and never a zero (`measured_store.UNMEASURED`). A rebuilt
  image does not change that. A rocprofiler-sdk that defined an fp64 counter
  for gfx12 would.

This is the question Fig. 7 answers, asked again on different hardware. It is
**not** a reproduction of Fig. 7 or Fig. 8: those are an A100's numbers, whose
ridge is a different number, and nothing here is compared with them.

**What it structurally cannot see**, each recorded with its mechanism in
`data/literature/jengine/limits.json` (the `jengine/batching-crosses-our-ridge`
entry) or the data files:

- **One input.** The counters were taken on the wide-T input only; a near-T
  clock is set against a wide-T verdict wherever the cross-read uses one.
- **DRAM, not the hierarchy.** The bytes are DRAM traffic. A batch whose
  working set stays in L2 barely touches DRAM, so kernel intensities within a
  split cell spread over orders of magnitude, and an L2- or register-level
  bottleneck is not on this roofline at all.
- **Every fp64 instruction counts.** `ncu` counts what the kernel issues, the
  Boys evaluation included, while `[JEngine]` Table 1's FLOPs exclude the Boys
  function; an intensity here is not Table 1's FLOPs over anything.
- **A threshold, not a margin.** L = 13 at one batch is compute-bound at 0.96
  FLOP/B against 0.65; a different ridge would flip it.
- **The counter run is not the timed run.** Collection serialises and replays
  each kernel; `counter_ns` is never a timing.
- **Classes not measured.** `[6|6]` (L = 12) and L = 14 are outside `SUBGRID`
  (#112 measured the compile cost and declined them); L = 10 is Fig. 7's
  `[5|5]`, and L = 13 and 16 bracket its `[7|7]`.
- **Achievable ceilings, at the clock the card ran them.** The ridges come
  from achievable rates, not peaks. `gfx1200`'s fp64 ceiling is 128 flop per
  counted GPU cycle at ~3.49 GHz, above its rated boost (#141 retired the
  dual-issue assumption; "each card's roofline ceilings" above), so its ridge
  is the card at that clock.

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

### What the generator ships, and why (#166)

The generator ships the **literal** plan — the minimax cut and the
sieve's fixed seed order, `Spec.levers = ()` — and not the searched board's
tuned plan (#161's `tuned` lever), because under the rule #163 declared
before any tuned row, `measured_store.tuned_may_become_the_default` is False:
on `data/measured/kernels.json`'s 32 tuned rows against their same-session
literal controls, the search is NOT SHOWN on both cards. The predecessor
`jgen` ships its searched order; this repo ships literal **on measured
wall-clock**, and the `tuned` lever stays opt-in and measured.

The grounds, all read off rows and none off a model:

- **The rule** (`measured_store`'s docstring, THE TUNED ARM, THE DEFAULT):
  the default moves only if `tuned` WINS at every measurable cell on both
  cards. Measured, it WINS only at L = 16 (`sm_86` b1/b3/b6/b9, `gfx1200`
  b3/b6/b9) and at `gfx1200` L10 b1. It LOSES at L7 b9 on both cards, the
  one cell whose recorded cut is not the minimax cut: wide-T 137.2 → 154.5 ms
  on `sm_86`, 174.3 → 198.9 ms on `gfx1200`. `gfx1200` also LOSES at L10 b3
  and L13 b1 (near-T). Its LOSSes at L7 b3 and b6 are on text identical to
  literal, the protocol's noise floor, and the refusal stands without them.
- **The sieve stays too.** #162's no-sieve A/B is NOT SHOWN on both cards and
  `sieve_may_leave_the_default` is False.
- **`PlanCost` is not admitted as grounds.** On neither card do the
  plan-layer counts predict the kernel (#165:
  `data/measured/plan_vs_kernel.json`'s `summary.<card>.answer`, or
  `planrank.verdict(store, arch)`). A plan that counts cheaper is not, by
  that, the faster kernel.
- **Tuned cannot build every recommended count.** The four 12-batch cells
  have no recorded choice, and 12 is what `batch_recommendation.json`
  recommends for `gfx1200` at L = 16 on near-T.

**The consequence for #145.** `batch_recommendation.json`, and `dispatch.py`
which reads it, were measured on the literal kernels (`levers = ()` rows).
With literal shipped they remain observations of the kernel the generator
writes, so they need neither re-derivation nor a narrower scope. Likewise no
`source_sha256`, golden, `builds.json`, `roofline.json` or `crossread.json`
row moves. The refusal is the last record in `limits.json`.
`measure/measure_shipped_plan.py` recomputes that record from the stores and
asserts that the shipped path's text is the literal rows' text and not the
tuned rows'.

**The batch count it ships.** `KERNEL_GEN_CMD` writes the single-batch
`[p|q]` ladder plus the pair ladder (`test_emit_cli.py::test_the_default_writes_both_ladders_and_nothing_else`;
single-batch by construction, since `generate()` calls `build_pq_kernel` per
`L`). One batch is a decision, not an omission. Its grounds are measured
("Measured: a recommended batch count per `(L, arch)`" below): wide-T picks 1
everywhere, near-T splits from L = 13, and the two cards disagree at L = 16.
`dispatch` resolves per `(arch, input)`, so following it would make the
shipped unit **set** depend on the arch and the input, which it does not.
The rows stay evidence, and dispatch reads them for a launch plan
(`batch_recommendation.json`'s `about` says the same, written from
`batchreport.ABOUT`). To emit another count, use the class selector:
`python -m felizjueves.emit --class LA LB LC LD --n-batches 1 3 6` (or
`--shell-ls l...`). The legal counts are 1 or a positive multiple of 3
(`spec.is_batch_count`, `batching.partition`), and a pair target (`DQ`,
`JAB`) is always 1. A `(class, n_batches)` cell the plan refuses is an error
(exit 2) unless `--admitted-only` is given, in which case it is reported and
skipped. The sharp edge: `--n-batches` is **rejected** on the default ladder
path, with exit 2 (*"--n-batches / --admitted-only need --class or
--shell-ls"*).

**What this repo can score of Alg. 3, the sieve.** One claim is asserted:
`jengine/rule-4-needs-the-sieve`, that the sieve recovers the common parents
MILD cannot. Two stay `unverifiable`, at their final status.
`jengine/sieve-underspecified`: the three choices Alg. 3 leaves open are
named and pinned (#160: the seed order is `hermite_targets` order, the
re-expansion direction is the smallest `r_d'`, and the deletion sweep
reaches the child's other old parent). Its "~2%" is a range over the paper's
own seed orders, which the paper does not state, so no space here can score
it (#174). `jengine/sieve-non-monotone-in-spills`: the non-monotonicity
reproduces on both cards, and the published pattern fails on both (#164).
Neither can be promoted, because a test that failed on our rows would not
make the paper's claim false. On the boards, Tables 1–2's
`exact_given_inputs` stay refused on `literal`, because the seed order is
not recovered (#108). `searched` passes `at_or_under` on all three tables,
and that pass is a choice the board makes, not a recovered input.

**What still differs from `jgen`**: one item, Table 4, declined. The plan
that ships is a decision on measured grounds, not a gap ("Parity with the
predecessor: one difference, and one decision" below).

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

**Where the tables stand (#108, #116).** On the `literal` board every
Agreement threshold is set and passes; Table 3's `exact_given_inputs` passes
from its recovered cut (40/40); Tables 1–2's `exact_given_inputs` are
**refused** under the rules-only doctrine — the sieve's seed order is not
published, the best of 58 seed-order rules reproduces 33/40 and 38/40, and a
per-cell fit is not admitted — so they score UNSET with the decision recorded,
never PASS (`literal.json`, `limits.json`).

**The `searched` board (#168, #171, #172).** Since #172 the board PASSes
`at_or_under` on Tables 1, 2 and 3 in all 40 cells: 22 land on the printed
triple exactly and 18 come in under it. Each plan is a feasible cut plus one
explicit seed-order permutation per batch, climbed from the 58
`seed_orders(widened=True)` rules within a bounded neighbourhood.
`searched_cuts.json`'s `space` names that neighbourhood; the space is
budget-bound and is not a set of local optima. The plan is chosen from the
cell's Pareto frontier over (max intermediates, FLOPs) by #171's selection
rule, which was declared before the board was read and is unchanged. The pass
says the printed cells are reachable inside a declared space. It does not say
which orders the authors drew, and #108's refusal on the `literal` board
stands: a climbed order is a choice the board makes, not a recovered input.
The searched record in `limits.json` has an empty residual. Its
`previous_residual` keeps the earlier spaces' numbers as history, and so does
its evidence, which records how the residual fell space by space. The 22 and
18 are recomputed from `searched_cuts.json` by `test/test_claims.py`.

## The `[JEngine]` claims: what this repo discharges, and what it never will

`data/literature/jengine/claims.json` is the one file that answers it: every
non-numeric claim of `[JEngine, v2]`, one record each, `asserted` by a named
`claim`-marked test **that fails when the claim is false**, or `unverifiable`
with the mechanism that puts it out of reach. "Is that the whole list?" is
answered by `coverage.json` (#120): the paper read unit by unit — every
section, algorithm and figure caption — each unit listing exactly the records
that cite it, with a note on what else it says and why that is not a record;
`test/test_claims.py` holds the two files against each other. One asserted
record, `jengine/batching-crosses-our-ridge`, is this repo's own question put
to its own cards (the roofline summary above), not a claim of the paper's.
`jengine/ket-sort-key-is-the-product` (#139) is the structural half of the
ket sort key: the plan orders kets by `I_q · D_max`, and under any other key
the same `break` returns a different `J`. Whether the product is the better
key *in time*, against the global-`D_max` fallback, is the measured arm,
`jengine/ket-sort-key-beats-the-global-max` (#147). It is `unverifiable`
because it speaks of the paper's A100. On our cards the expectation, declared
before the rows, is **contradicted on both** ("The launch plan" below).

The table below is rendered from `claims.json` and checked verbatim:

<!-- claims-status:begin -->
**35 records: 18 asserted, 14 unverifiable, 3 pending** (recomputed from `claims.json` by `test/test_claims.py`).

| status | claim | kind | held by |
|---|---|---|---|
| asserted | `jengine/alg1-line10-atomic-free` | defect | `test/test_passes.py::test_atomize_writes_j_only_by_atomic_add` |
| asserted | `jengine/alg1-line10-factor-four` | defect | `test/test_ao_j_interpreted.py::test_j_p_is_written_unscaled_and_a_factor_four_is_caught` |
| asserted | `jengine/batch-count-multiple-of-3` | shape | `test/test_plan_batching.py::test_an_illegal_batch_count_raises` |
| asserted | `jengine/batch-overlap-is-fundamentals-and-axials` | shape | `test/test_plan_dag.py::test_batches_share_only_fundamentals_across_and_axials_within` |
| asserted | `jengine/batch-tie-break` | ordering | `test/test_plan_mild.py::test_mild_and_the_batching_pass_break_ties_oppositely` |
| asserted | `jengine/batching-crosses-our-ridge` | monotonicity | `measure/measure_ridge.py::test_batching_crosses_our_ridge_on_sm_86_and_gfx1200_could_not_measure` |
| asserted | `jengine/eq17-1y-misprint` | defect | `test/test_eq17_witness.py::test_y_reduction_shifts_by_the_unit_vector_0_1_0` |
| asserted | `jengine/fig4-l6-six-batches` | membership | `test/test_plan_batching.py::test_figure_4s_cut_at_l_6` |
| asserted | `jengine/ket-sort-key-is-the-product` | ordering | `test/test_launch.py::test_the_ket_sort_key_is_the_product_and_the_break_needs_it` |
| asserted | `jengine/m-weight-minus-two-theta-squared` | defect | `test/test_ao_eri.py::test_the_m_weight_is_minus_two_theta_squared` |
| asserted | `jengine/mild-keeps-rules-1-2-3-5` | shape | `test/test_plan_mild.py::test_johnsons_rule_holds_on_every_node` |
| asserted | `jengine/mild-suboptimal-from-l5` | shape | `test/test_mild_vs_optimum.py::test_mild_against_the_exact_optimum` |
| asserted | `jengine/mild-tie-break` | ordering | `test/test_plan_mild.py::test_the_whole_tie_break_on_every_node` |
| asserted | `jengine/prefactor-factorisation` | shape | `test/test_ao_eri.py::test_block_matches_libcint` |
| asserted | `jengine/rule-4-needs-the-sieve` | shape | `test/test_plan_sieve.py::test_the_sieve_recovers_common_parents_where_mild_cannot` |
| asserted | `jengine/symmetry-breaking-splits-are-worse` | monotonicity | `measure/measure_store.py::test_asymmetric_splits_are_worse_on_both_cards` |
| asserted | `jengine/theta-undefined` | defect | `test/test_ao_j_interpreted.py::test_interpreted_j_matches_reference_and_libcint` |
| asserted | `jengine/y12-example` | membership | `test/test_plan_batching.py::test_the_y_1_2_batch_is_the_papers_eight_targets` |
| unverifiable | `jengine/batch-count-is-hardware-dependent` | monotonicity | no test: the reason is in `claims.json` |
| unverifiable | `jengine/batching-crosses-the-ridge` | monotonicity | no test: the reason is in `claims.json` |
| unverifiable | `jengine/batching-reduces-spills` | monotonicity | no test: the reason is in `claims.json` |
| unverifiable | `jengine/batching-trades-flops-for-intermediates` | monotonicity | no test: the reason is in `claims.json` |
| unverifiable | `jengine/best-batch-count-rises-with-l` | monotonicity | no test: the reason is in `claims.json` |
| unverifiable | `jengine/conclusion-overstates-compute-bound` | defect | no test: the reason is in `claims.json` |
| unverifiable | `jengine/j-speedup-tracks-high-l-share` | monotonicity | no test: the reason is in `claims.json` |
| unverifiable | `jengine/ket-sort-key-beats-the-global-max` | monotonicity | no test: the reason is in `claims.json` |
| unverifiable | `jengine/memory-bound-grows-with-l` | monotonicity | no test: the reason is in `claims.json` |
| unverifiable | `jengine/mild-vs-exact-tradeoff` | monotonicity | no test: the reason is in `claims.json` |
| unverifiable | `jengine/nine-batches-suffice` | shape | no test: the reason is in `claims.json` |
| unverifiable | `jengine/optimal-batching-is-np-hard` | shape | no test: the reason is in `claims.json` |
| unverifiable | `jengine/sieve-non-monotone-in-spills` | monotonicity | no test: the reason is in `claims.json` |
| unverifiable | `jengine/sieve-underspecified` | defect | no test: the reason is in `claims.json` |
| pending | `jengine/k-contraction-at-quartet-level` | shape | no test: the reason is in `claims.json` |
| pending | `jengine/k-recompute-has-more-merit` | monotonicity | no test: the reason is in `claims.json` |
| pending | `jengine/k-splitting-redundant-hermite` | shape | no test: the reason is in `claims.json` |
<!-- claims-status:end -->

**Why the unverifiable ones stay so**, by mechanism (each record's `reason`
has the detail):

- *A measurement of the authors' kernels on their A100* — a roofline, a spill
  count, a timing: `batching-crosses-the-ridge`, `memory-bound-grows-with-l`,
  `batching-reduces-spills` (Table 4, deliberately not transcribed),
  `sieve-non-monotone-in-spills`, `best-batch-count-rises-with-l`,
  `batch-count-is-hardware-dependent`, `j-speedup-tracks-high-l-share`,
  `nine-batches-suffice`. Where this repo asked the same question of its own
  cards, the answer is a different record or a measured file (#56, #60,
  #114), never a discharge of the paper's. For `sieve-non-monotone-in-spills`
  #164 read the published pattern off #162's no-sieve rows
  (`measured_store.sieve_spills`): the effect is non-monotone on both cards
  and `[8|8]` spills less with the sieve on both, but the `[5|6]` stand-in,
  `(6,0|5,0)`, spills less on both and the per-class mean falls on gfx1200
  only, so the pattern fails on both cards. A test asserting it would fail on
  our rows without making the paper's claim false, so `limits.json` records
  the mechanism instead and `measure/measure_sieve_spill_claim.py` holds the
  reading.
- *A comparison with a search that is not here*: `mild-vs-exact-tradeoff`.
  #111 replaced it with this tree's own optimum, and the measurement
  disagrees with the printed percentages (`mild_vs_optimum.json`); the test
  that pins our gap holds `mild-suboptimal-from-l5` instead, because a test
  that would fail if the paper's numbers held cannot be their witness (#116).
- *A direction the source's own table does not keep*:
  `batching-trades-flops-for-intermediates` — post-sieve, so it rides on the
  unpublished seed order, and the printed Table 2 breaks its intermediates
  half at two steps.
- *Something the text does not carry*: `sieve-underspecified` (its "~2%"
  is a range over the paper's own seed orders, which the paper does not
  state; the 58-rule spread and #172's climbed orders are this repo's spaces,
  not that one),
  `conclusion-overstates-compute-bound` (two sentences of the source that
  cannot both hold), `optimal-batching-is-np-hard` (no finite computation
  establishes NP-hardness).

**What the milestone's result is, stated accurately.** The paper's algorithms
are reconstructed and witnessed (the plan layer against Tables 1–3, the kernel
and `J_µν` against libcint); its generator tables are matched where its
unpublished inputs allow (Table 3 exactly; Tables 1–2 up to the seed order);
and its central performance claim — batching moves a kernel across the ridge —
has been asked and answered on different hardware, with an answer about our
cards' ridges and nothing to say against its A100. Since milestone 6 a whole
molecule's `J` is built from a launch plan on both cards, the plan's order
witnessed by an independent Python driver and by libcint, and one of the
paper's §3.3 claims (the ket sort key) measured. Since milestone 7 every
arithmetic step of that build is a generated kernel, run on both cards: the
`E` transform, the pre-contraction and the scatter as well as `J_p`. The two
new kernels are this repo's own design, because the paper defers them to
`[Fock]`, so they discharge no claim of the paper's, and no record above
changed status for them.

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
| `(ab\|cd)`, one block per class: all 55 classes of HF/cc-pVTZ (L ≤ 12), H₂O/6-31G's s/p | `int2e`, `≤ 1e-12·scale` | `test_ao_eri.py` | s/p fast, d+ slow |
| `J_µν`, `reference.pq` | `get_jk`, `≤ 1e-12·scale_J` (derived from the ERI bound) | `test_ao_j.py` | H₂, H₂O STO-3G/6-31G fast; cc-pVDZ slow |
| `J_µν`, interpreted generated kernel | same | `test_ao_j_interpreted.py` | H₂ fast; H₂O ×3 slow |
| `J_µν`, generated kernel on each card | same | `test_ao_j_ondevice.py` | kernel tier (`gpu`); H₂, H₂O/STO-3G |
| `J_µν` from the launch plan on each card (#144) | same, plus `τ · scale` when `τ > 0` | `test_j_launched_ondevice.py` | kernel tier (`gpu`); H₂, H₂O/STO-3G, H₂O/cc-pVDZ |

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

- **Classes with L = la+lb+lc+ld ≥ 13.** Since #121 the ERI witness has no
  class boundary inside its basis: every one of the 55 canonical classes of
  HF/cc-pVTZ (s to f, L = 0..12) is compared block by block against
  `int2e` — L ≤ 8 in `hf_ccpvtz_eri.json`, L = 9..12 one file each
  (`hf_ccpvtz_eri_l9` … `_l12.json`, every file under the 512 KB hook), and
  `test_ao_eri.py` checks that their union is exactly the basis's 55 classes.
  Those blocks are built from `reference.pq`; the **generated kernel** meets
  libcint only through the `J` fixtures, which still stop at `d` (cc-pVDZ,
  so L ≤ 8). The kernel ladder runs to L = 16 and is checked against
  `reference/` at every L, and `reference/` is itself witnessed by libcint to
  L = 12 — so above L = 8 the kernel is held to libcint at one remove, and
  above L = 12 not at all. `θ` is formed once per pair by the same prelude at
  every L, so the record does not depend on L, but a recurrence defect that
  only appears above L = 12 would be invisible to libcint here (a basis with
  `g` functions would move it).
- **The device's screen.** This capstone launches with `Ip = Iq = Dmax = 1`
  and `τ = 0`. Sorted kets with real bounds run on the card in the launch-plan
  build (#144, "The launch plan" below). The host-side screen is #88's ladder,
  below.

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
`reference.boys` had measured (#122) when the ladder was run; the test pins
that the chain is as long as that extent allowed. #265 has since carried the
measurement to `T_MAX = 1e6` (for `V`'s point-charge kets, which reach
`T ≈ 1.25e5` on HF/cc-pVTZ), so the evaluator now reaches much further; the
chain is kept at #122's extent so the ladder is not rewritten. (#88 ran two H₂ under the original
`T ≤ 500` cap; #122 gave `reference.boys` a measured large-`T` branch — the
asymptote `Γ(m+½)/(2T^(m+½))` from `T = 125`, where its truncation for every
`m ≤ 40` is under 1e-18, to `T = 2000`, its error under 8e-16 against the
Kummer series — and the ladder was re-run. The bound that bites is `T_MAX`,
not `M_MAX`: the chain is all s shells, so only `F_0` is ever asked for. The
device-side `fj_boys` was untouched by that widening; #301 later gave it
`reference.boys`' whole domain.)

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

## The launch plan: a whole molecule's `J`, on both cards (milestone 6)

**Implemented, and run on the RTX 3080 and the RX 9060 XT** (#138–#147).
Everything above builds `J_µν` one `(bra shell pair, ket class)` launch at a
time through the `jp=` seam, with `Ip = Iq = Dmax = 1` and `τ = 0`. This
section builds a whole molecule's `J` from a **plan**, in the plan's shape,
with the real screen.

**What is generated, and what is not.** The generator emits the `[p|q]`
kernels (`J_p = Σ_q [p|q] D_q`) and nothing else. The plan is a Python value.
The device `main()` that runs it is written by a test harness
(`test/plan_driver.py`), not by `emit/`. The `E` transform, the ket
pre-contraction, eq. 20's back-transform `J_ab = Σ_p E_p^{ab} J_p` and the
`J_ab` scatter stay **host-side Python** (`ao/`), as they are in the
predecessor's `jgen`. That was milestone 6's state. Milestone 7 generated
them as device code and ran them on both cards on this same driver ("The J
chain as device code" below). The timed builds in this section
(`builds.json`) predate it, and still time `[p|q]` alone.

### The plan as a value (#138)

`launch.launches(basis, D, *, bound, n_batches, tau, density_sort=True,
strict=True, resolve=None) -> LaunchPlan` returns one `ClassLaunch(lp, lq,
n_batches, bras, kets, kernels, bra_blocks)` per `(lp, lq)` that has records
on both sides. A class with records on one side only is absent, never present
and empty. The kernels are `build_pq_kernels` of the class. One build serves
every shell class `(la, lb | lc, ld)` with those sums.

- **Records, not shell pairs.** A `PairRec(i, j, prim, L, G, key)` is one
  *primitive-pair* index record. The kernel maps one warp per bra record, and
  `ao/pairs.py`'s `I_pair` is proved per record. The predecessor's `jgen`
  maps one warp per **shell pair** and screens per shell pair. Both are
  readings of Alg. 1's per-index `Ip`/`Iq`; this one is finer, and it is the
  one with a proof here. That makes every `n_pairs` and `n_skipped` in this
  section a count of record pairs, not comparable one for one with a
  shell-pair screen's (`limits.json` records this as a reading, not a
  defect). The plan carries the record → shell pair map (`PairRec.shell_pair`,
  and `ClassLaunch.bra_blocks` on the bra side) because eq. 20 accumulates
  per shell pair.
- **The sweep** is `ao.j`'s unique one: shell pairs `a ≥ b` on both sides,
  with the ket off-diagonal weight 2 folded into `D_q` and so into `Dmax`.
  The ket pre-contraction is `ao.density.make_kets` itself, not a restatement.
- **The sort key, `[JEngine, v2, §3.3]`.** Bras stay in pair order, because
  the kernel maps warps to bras by index. Kets are sorted **descending by
  `key = I_q · max_q |D_q|`, stably** (`sorted(..., reverse=True)`,
  `ao.j._sorted_kets`' choice). With a fixed `I_bra ≥ 0`, `I_bra · key` is
  then non-increasing, so Alg. 1's `break` at the first ket under `τ` loses
  nothing. `density_sort=False` replaces every ket's `Dmax` with one global
  value, the largest per-record `Dmax`. The ket order is then by `I_q` alone:
  the fallback `jgen` keeps. That screen is sound but looser. The structural
  claim is `jengine/ket-sort-key-is-the-product`, asserted by
  `test/test_launch.py`: under the product the `break` equals the unsorted
  `continue` to 0, and under `I_q` alone or pair order the same `break` puts
  `J` outside `τ · scale`.
- **The batch count** is asked of the plan per class, with
  `ao.drive.KernelJP`'s `strict` contract (raise, or build at 1 and record
  the class in `refused`). `resolve=` injects a policy per class (#145).

### Three walkers, and what each one witnesses (#142, #143, #144)

| walker | where | what runs `[p\|q]` | held against |
|---|---|---|---|
| the launch oracle, `launch.walk` / `coulomb_launched` | pure Python (`src/`) | `reference.pq`, injected | `ao.j.coulomb_matrix_screened` **to 0** (same `J`, `n_pairs`, `n_skipped`, `max_skipped`, `dropped`) on H₂, H₂O/STO-3G and the four-H₂ chain at `τ` from 0 to 1e-1; the ordered reference within `1e-14·max\|J\|`; libcint within `1e-12·scale_J` |
| the staged driver, `ao.staged.coulomb_matrix_staged` | interpreted (`interp`) | the generated kernels | the host sweep within `1e-13·max\|J\|`, libcint within `1e-12·scale_J`, and the oracle's ket-loop count per class **exactly** |
| the device build, `test/plan_driver.py` | compiled with `NVCC_FLAGS` / `HIPCC_FLAGS`, run on each card | the generated kernels | the oracle on the same plan within `1e-13·max\|J\|`; libcint within `1e-12·scale_J` (plus `τ·scale` when `τ > 0`) |

The oracle rebuilds its per-pair data on every call and memoises nothing, so
the `τ` ladder measures the screen and not a cache. Its skip fractions equal
`data/ao/screening.json`'s on every rung.

### The residency shape: what "on the card" means here

Within **one build** of `J_p`:

- **Uploads:** one contiguous upload per bra class (`P ζ Ip`) and one per ket
  class (`Q η Iq Dmax D_q`, kets in the plan's sorted order). That is one
  upload per ket class, not one per (bra pair, ket class). A ket class shared
  by several `ClassLaunch`es must carry the identical ket list, or the staging
  refuses.
- **`J_p`:** one device buffer per bra class, zeroed once per build. Every
  `(lp, lq)` batch kernel is launched into its bra class's buffer, in plan
  order, and accumulates with **fp64 atomics** across ket classes. Split
  classes are atomised by the builder. One-batch kernels are atomised by the
  driver (`passes.atomize`), because there the kernel is no longer its
  buffer's only writer. The builder, its goldens and every `source_sha256` in
  `kernels.json` are unchanged.
- **The screen is real.** The kernels read the plan's `Ip`, `Iq`, `Dmax` and
  `τ`, and each ket's `Iq · Dmax` is checked bit for bit against its plan
  `key` before upload.
- **Read-back:** one copy per bra class after one synchronize. Eq. 20 and the
  scatter then run on the host.

**What it is not.** A `J` resident across SCF iterations, a `Launch` IR and a
`LaunchCost` model are out of scope by the milestone's own decision, and
`limits.json` records them as **deliberately absent**: the build is
measured, never predicted.

### On the cards (#144)

`test/test_j_launched_ondevice.py` (`gpu` + `cuda`/`hip`). Per-cell results,
identical on both cards to the digits shown (PR #182):

| cell | classes | kernels per build | vs libcint (`/scale_J`) | vs oracle (`/max\|J\|`) |
|---|---|---|---|---|
| H₂/STO-3G, `τ = 0`, 1 batch | (0,0) | 1 | 3.83e-15 | 1.57e-16 |
| H₂O/STO-3G, `τ = 0`, 1 batch | 9 | 9 | 5.09e-15 | 1.30e-16 |
| H₂O/STO-3G, `τ = 0`, 3 batches | 9, (0,0) refused → 1 | 25 | 5.09e-15 | 2.61e-16 |
| H₂O/STO-3G, `τ = 1e-3`, 3 batches | 9; the oracle skips 2024 of 18225 | 25 | 1.30e-3, inside `1e-12·scale_J + τ·scale` | 3.26e-17 (CUDA) / 6.52e-17 (HIP) |
| H₂O/cc-pVDZ, `τ = 0`, dispatched | 25, all at 1 batch | 25 | 1.24e-14 | 3.29e-16 |

The worst unscreened deviation from libcint is **1.24e-14·scale_J** against a
`1e-12` tolerance, and no tolerance was loosened. Two runs of the same
binary are held to an fp64-atomic run-to-run bound on `J_p`. In every cell
they were also bit-identical, which is recorded as an observation.

**What the cards reach.** STO-3G is s and p (classes up to L = 4), and
cc-pVDZ adds d (up to L = 8). The dispatched cell injects
`dispatch.Resolver(arch, "wide")`. On both cards the measured wide-T choice
is 1 at every L, so all 25 classes run at 1: 24 by fallback, and `(4, 3)` from
its measured row (`pq_spec(7)`'s split). No tracked fixture reaches a class
whose measured choice is above 1 (near-T: `(5, 5)` on `gfx1200`, `(7, 6)` on
both). When this was recorded the card's `fj_boys` was witnessed on
`T ∈ [0, 500]` and overflowed near `T ≈ 700`, so #88's four-H₂ chain (`T` up to
1289) had no on-card build. Since #301 the one `fj_boys<M>` template is
witnessed on `T ∈ [0, 1e6]` (on `sm_86`), but no on-card build of the four-H₂
chain has been added; its skip counts still come from the oracle. Both limits
are in `limits.json`.

### Measured: the whole build, behind the gate (#146)

`data/measured/builds.json` is the first store of a **build** rather than a
kernel (`measured_store.BuildRow`, its own file, because a build is not a
`Spec`). Each row records the card's gate readings, and the loader refuses a
null reading or one above the threshold. **The timed region** (the header's
`timed_region`): zero each bra class's `J_p` buffer; launch every batch
kernel in plan order; synchronize; copy each `J_p` back. That is 31 timed
builds after 5 warm-up ones, in one process. **Outside the clock:** the
uploads (once per process, recorded as a single `build:upload_ns`), every
allocation, host staging, and eq. 20 and the scatter. A row is the device
cost of one `J_p` for a molecule whose index data is already on the card. It
is not a cold start and not an SCF iteration.

| cell | card | median ± IQR, µs | device | read-back | upload (once) |
|---|---|---|---|---|---|
| H₂O/STO-3G, `τ = 0`, 1 batch | RTX 3080 | 930.1 ± 1.5 | 913.3 | 16.6 | 0.12 ms |
| H₂O/cc-pVDZ, `τ = 0`, dispatched | RTX 3080 | 11070.7 ± 1492.9 | 11041.8 | 30.1 | 0.15 ms |
| H₂O/cc-pVDZ, `τ = 1e-3`, dispatched | RTX 3080 | 10081.5 ± 797.1 | 10052.8 | 29.6 | 0.14 ms |
| H₂O/STO-3G, `τ = 0`, 1 batch | RX 9060 XT | 636.4 ± 52.8 | 564.2 | 72.4 | 21.9 ms |
| H₂O/cc-pVDZ, `τ = 0`, dispatched | RX 9060 XT | 9045.4 ± 276.6 | 8922.5 | 122.7 | 35.8 ms |
| H₂O/cc-pVDZ, `τ = 1e-3`, dispatched | RX 9060 XT | 7056.2 ± 185.1 | 6930.3 | 124.5 | 34.9 ms |

These are observations of two cards. They are not compared with each other
(gfx1200 rate-limits fp64) and not compared with the paper's A100. Three
things they do **not** say. **What the plan saved:** the per-pair
(`KernelJP`-shaped) launch path was not timed in the same session, so no row
measures the plan against it. **Why gfx1200's upload is slow:** its one-off
upload is 22–36 ms against about 0.1–0.16 ms on `sm_86`, a single observation
outside the timed region, and not explained. **Why sm_86's cc-pVDZ rows
are wide:** their IQRs are 0.8–1.5 ms on about 10–11 ms, and #147's
sessions found these builds bimodal (near 8.5 ms and near 10 ms).

### Measured: the sort key, product against a global `max|D|` (#147)

The question is whether `[JEngine, v2, §3.3]`'s product key builds `J_p`
faster than the global-`Dmax` fallback. #147 declared the expectation before
any row existed: *the global key is worse; if it is not, that is the result.*
The rule, transcribed into `measured_store` (THE SORT KEY) before the rows
existed, is #60's noise floor. Per cell, the global key is WORSE iff its
median is slower by more than the larger IQR, BETTER iff faster by that, and
otherwise UNRESOLVED. A card is WORSE iff every cell is, and CONTRADICTED iff
any cell is BETTER. Both arms of a cell are timed in one gated session per
card, and they differ only in `density_sort`.

| card | cell | skipped, product / global | median µs, product / global | global key |
|---|---|---|---|---|
| sm_86 | H₂O/cc-pVDZ `τ = 1e-3` | 123896 / 0 of 351649 | 10080.9 / 8510.4 | **better** |
| sm_86 | H₂O/cc-pVDZ `τ = 1e-10` | 15684 / 0 | 10451.9 / 9836.6 | unresolved |
| sm_86 | two-H₂ chain `τ = 1e-3` | 2790 / 0 of 8100 | 255.4 / 285.7 | worse |
| sm_86 | two-H₂ chain `τ = 1e-10` | 560 / 0 | 346.7 / 285.7 | **better** |
| gfx1200 | H₂O/cc-pVDZ `τ = 1e-3` | 123896 / 0 | 7066.6 / 7063.6 | unresolved |
| gfx1200 | H₂O/cc-pVDZ `τ = 1e-10` | 15684 / 0 | 8750.3 / 7059.9 | **better** |
| gfx1200 | two-H₂ chain `τ = 1e-3` | 2790 / 0 | 211.6 / 251.5 | worse |
| gfx1200 | two-H₂ chain `τ = 1e-10` | 560 / 0 | 273.8 / 251.0 | **better** |

**The expectation is contradicted on both cards.** The screen goes the
expected way everywhere: the global key skips **no** record pair on any
input. `I_pair` is pair-factor-free, so a ket's distance decay reaches the
screen only through its own `Dmax`, and a global value erases it. The clock
does not follow. Where the product key skips little, its ket **order** costs
more time than its skips save. Why is not measured; one candidate is lane
divergence in the device Boys loop, whose length grows with `T`. The claim
`jengine/ket-sort-key-beats-the-global-max` is `unverifiable` (it speaks of
the paper's A100) and carries this result. `limits.json` has the per-cell
verdicts and names the two unresolved cells as below what this box can
resolve. `test_measured.py` recomputes every verdict from the rows.

### Parity with the predecessor: one difference, and one decision

The predecessor's `jgen` reconstructs the same paper (`[JEngine]`) for the
energy-order `J`. Against it, this repo has **parity on `jgen`'s
energy-order `J` surface, with one difference: Table 4, declined**. It goes
beyond that surface in several ways: two vendors from one source, the kernel
as checked IR with an interpreter, the claims registry and its coverage
reading, gated measurement provenance, and an outside witness (libcint).

**The one difference: Table 4 is declined, by decision**
(`data/literature/jengine/README.md`). `jgen` scores a static
register-pressure proxy against Table 4's 40 cells. This repo uses a
different instrument on purpose: `lower.KernelCost`, plus real ptxas / hipcc
rows from two cards and `kernelrank`'s ranking against them. So "this repo
replicates `jgen`" is not true while Table 4 is declined.

**The one decision: which plan ships** (#166, resolving milestone 6's
qualification, #148). Until milestone 8 the searched plan did not reach the
emitter here, and in `jgen` it does; that was a second difference. It now
reaches it: #161's opt-in `tuned` lever (`pipeline.tuned_plan`) builds every
`[p|q]` kernel from the searched board's recorded cut and per-batch seed
orders (#172), and #163 measured it on both cards against its same-session
literal control (32 rows). The generator still ships the **literal** plan,
because the rule declared before those rows refuses the tuned default:
`tuned_may_become_the_default` is False, the search NOT SHOWN on both cards
("What the generator ships, and why (#166)" above). So "this repo is behind
`jgen`" is not true either: the wire is here and was measured, and literal
was chosen on that evidence. Nor does the choice say anything about `jgen`'s:
its searched order is its own, on its kernels and cards, and the two searches
are not the same search.

**Not a parity item: the batch count.** `jgen.tuning.tuned_plan(L,
n_batches=1, ...)` takes the count from its caller, as this repo's generator
does, and `dispatch.py`'s docstring says NOT PARITY. The shipped single batch
is this repo's own decision ("The batch count it ships", above).

This repo computes no gradients or Hessians, so none of this is parity with
anything that has them.

## The J chain as device code

**Status: implemented, and run on both cards (milestone 7, #149–#157).**
This section is the contract the chain's kernels were built against, and the
one place that answers *which steps of a `J` build does this repo generate,
and on which cards has that run?* Every **arithmetic** step of `J_µν` is a
generated kernel: the `E` fragment, the pre-contraction (`Target.DQ`), the
`[p|q]` kernel (`Target.J`) and the bra transform and scatter
(`Target.JAB`). `KERNEL_GEN_CMD` writes all three kernel families (#156),
and the whole chain ran on the RTX 3080 and the RX 9060 XT (#157). What is
**not** generated is everything that decides launches: the pair list and its
per-record inputs, the bound, the sort (with its `Dmax` read-back), the
staging, and the launch code itself, which is a test-side driver
(`test/plan_driver.py`) launched from Python. **Nothing about the new kernels
is measured**: no registers, spills, wall-clock or counters. There is no
device-resident `J` across SCF iterations, and dispatch does not reach the
generator.

`[JEngine, v2]` defers pre- and post-contraction to `[Fock]`, which this repo
has not read. So the `DQ` and `JAB` kernels below are **this repo's own
design**, built to equal `ao/j.py`'s host arithmetic, and not a
reconstruction of anything published (`limits.json`).

### The four steps

| step | computes, per primitive-pair record | keyed by | built by | interpreted | on the cards (#157) | host equivalent |
|---|---|---|---|---|---|---|
| 1. `E` | `E^{ab}_t`, undressed, over `hermite_order(la+lb)` | inlined, not launched | `e_fragment.block_fragment` (#152; an `ir.Fragment`, #149) | bitwise `==` `ao.hermite.e_block`, (ss)–(ff) (`test_e_fragment.py`) | inside steps 2 and 4 | `ao.hermite.e_block` |
| 2. pre-contraction | `D_q = w Σ_cd D_cd s E^{cd}_q` and `Dmax = max_q \|D_q\|` | ket pair class `(lc, ld)`, `Target.DQ` | `kernel_dq.build_dq_kernel` (#153), unit `dq_{lc}_{ld}` | bitwise `==` `ao.j.dq_from_host`, (ss)–(ff) (`test_kernel_dq.py`) | every class (ss)–(ff) alone; in `J` through (dd) | `ao.density.make_kets`; through the seam, `ao.j.dq_from_host` |
| 3. `[p\|q]` | `J_p = Σ_k Σ_q [p\|q] D_q` | class `(lp, lq)`, `Target.J` | `kernel.build_pq_kernel(s)` | against `reference/` | L = 0..16 against `reference/`; in `J` up to L = 8 | `ao.j.jp_from_pq` |
| 4. bra transform + scatter | `J_ab += Σ_p s E^{ab}_p J_p`, written into `J_µν` | bra pair class `(la, lb)`, `Target.JAB` | `kernel_jab.build_jab_kernel` (#154), unit `jab_{la}{lb}` | bitwise `==` `ao.j.jab_from_host`, (ss)–(ff) (`test_kernel_jab.py`) | every class (ss)–(ff) alone; in `J` through (dd) | `ao.j`'s transform and mirror; through the seam, `ao.j.jab_from_host` |

Steps 2 and 4 contain step 1: the `E` fragment is inlined into both. It is not
a kernel and has no buffer of its own. #149's rule is one level only (a
`Fragment` holds no `Inline`), so the block fragment is **composed at build
time** from the one axis builder, instantiated once per axis, and does not
nest the axis fragment. Every recurrence entry is one `Weighted`. The only
`Call`s are the three `EXP` seeds and the block's `MUL` products, and there
is no `Raw` (`limits.json`, the instruction form's reach).

**Composed in the interpreter** (#155). `ao.drive.coulomb_matrix_chain` builds
`J_µν` with every step a generated kernel run by `interp.py`. It runs `DQ`
per ket class, reads `Dmax` back, sorts, runs `[p|q]` through milestone 6's
staging (`ao.staged`, with its `ket_data=` and `transform=` hooks), then runs
`JAB` per bra class. A missing step raises. It is within `1e-12·scale_J` of
libcint on every tracked `J` fixture (H₂, H₂O/STO-3G, 6-31G, cc-pVDZ; worst
1.3e-14), and `==` the staged host path (`test/test_ao_j_chain.py`).

**Composed on the cards** (#157, `test/test_j_chain_ondevice.py`, `gpu` +
`cuda`/`hip`), in two parts:

- *Part 1, per pair class, (ss) through (ff).* Each `dq_{la}_{lb}` and
  `jab_{la}{lb}` unit that the kernel tier's generate phase wrote is compiled
  alone into a small harness and run over every record of its class in
  HF/cc-pVTZ, against `ao.j.dq_from_host` / `jab_from_host` on the same
  inputs. Worst: `DQ` 5.45e-16 `S`, `JAB` 4.04e-16 `T`, against bounds of
  2.66e-15 to 3.00e-14 by class (the error analysis is in the test module's
  docstring).
- *Part 2, per molecule, through d.* `J_µν` for H₂/STO-3G, H₂O/STO-3G at 1 and
  3 batches, and H₂O/cc-pVDZ. A `DQ` program runs every ket class and reads
  `D_q` and `Dmax` back. The host sorts on the kernel's `Dmax`. Milestone 6's
  driver then runs every `[p|q]` batch kernel, with its kets staged from the
  kernel's `D_q`, and in the same process one `JAB` launch per bra class adds
  into one zeroed `J_µν` on the card. Worst against libcint
  **1.24e-14·scale_J** (tolerance `1e-12`), against the launch oracle
  3.29e-16 `max|J|`, with the same digits on both cards. No `[p|q]` above
  L = 8 and no f shell enters a whole-molecule build, because the `J`
  fixtures stop at d.

**The last device run, per card** (PR #210, the full kernel tier, every unit
generated by `KERNEL_GEN_CMD`): CUDA, RTX 3080 (`sm_86`), **96 passed, 5
skipped**; HIP, RX 9060 XT (`gfx1200`), **97 passed, 5 skipped**. The five
skips on each card are the opt-in recordings (ceilings, `[p|q]` wall-clock,
roofline counters, build wall-clock, sort-key A/B). The largest single need
was part 1's `(23)` `DQ` program on the 3080: 7237 MiB of 8764 MiB free,
from a 72 KB per-thread stack over 104448 resident threads. The VRAM probe
passed, but an outside process holding more than about 1.5 GB of that card
would fail the case, with both figures (`limits.json`).

**The generator** (#156). With no ladder flag, `python -m felizjueves.emit`
writes 37 units: the `[p|q]` ladder L = 0..16 (`--L`, byte-identical to
before, its `source_sha256` pins unchanged) and the pair ladder,
`dq_{la}_{lb}` and `jab_{la}{lb}` for `la ≤ lb ≤ 3` (`--pair-l LMAX`, 0..8; 20
units by default). The class selector (`--class`, `--shell-ls`) writes each
cell's whole chain, with an `order` file in launch order: `dq_{lc}_{ld}`, the
`[p|q]` batch kernels, then `jab_{la}{lb}`.
Since #242 `--one-e-l LMAX` adds the one-electron ladder beside these, and
`KERNEL_GEN_CMD` asks for `--one-e-l 2` (37 + 18 units); the 37 above and
their pins are unchanged ([`one_electron.md`](one_electron.md), "The
generator").

### The layout every consumer indexes

One shell pair `(a, b)` of angular momenta `(la, lb)` and one primitive pair of
it give one **record**. Records are in `ao.pairs.primitive_pairs` order: the
`a`-side primitive outer. The `E` block of a record is a matrix:

- **rows**: one per Cartesian pair, `cartesians(la) × cartesians(lb)`, `a`
  outer. Row `r = i · ncart(lb) + j` is the pair `(cartesians(la)[i],
  cartesians(lb)[j])`. This is `ao.pairs.PrimitivePair.block`'s layout.
- **columns**: one per Hermite index `t`, in `hermite_order(la + lb)` order
  (`|t|` ascending, then `tx`, then `ty` descending), which equals
  `plan.targets.hermite_targets(la + lb)`. This is also the order of the
  `[p|q]` kernel's `D[k, q]` and `J[bra, p]`.

Per axis, `E^{ij}_t` exists for `t ≤ i + j` only. In the block, the entry at
`t` for the row `(a, b)` is zero exactly when `t_d > a_d + b_d` on some axis.
The block fragment omits those terms; it does not compute them as zeros.

The density block `D_cd` of a ket shell pair is indexed by the same row order:
element `r` is `D[off_c + i][off_d + j]`, with `i` and `j` as above.

### The `E` inputs, and the dressing pin

Each record gives the fragment `ζ` and, per axis `d`, `X_PA = P_d − A_d` and
`X_PB = P_d − B_d`. These are `P`, `A` and `B` as `primitive_pairs` and
`e_block` have them, with the subtraction in that order. The recurrence is the
one in "The `E` transform" above (MD report eqs. 2.20–2.21), with
`E^{00}_0 = exp(+ζ X_PA X_PB)`, which is `exp(−μ X_AB²)` because
`X_PA X_PB = −(αβ/ζ²) X_AB²`. (#152's issue text wrote the seed with a minus
sign. That was a slip: the fragment follows `ao/hermite.py`, and the Hermite
identity test holds the sign.) The fragment forms no `X_AB` and no `μ`.
It binds `zeta` as well as `half_inv_zeta` (`1/2ζ`) so that the seed costs no
extra division; a class with no raised entry, `(ss)`, binds no
`half_inv_zeta`.

**The fragment computes the undressed `E` of `ao/hermite.py` and nothing
else.** The dressing reaches the kernel as **one host-supplied scalar per
record**:

```
s = c_a c_b (π/ζ)^{3/2}         # ao.pairs.dress's scale; ao.density.EPair carries it
```

`ao.density.epair` takes `s` from `ao.pairs.dress` itself (`dress((1.0,), …)[0]`,
the scale times 1.0, which is exact). So the dressing stays one function's
decision. No kernel forms `(π/ζ)^{3/2}` or a contraction coefficient. A
dressing inside the fragment breaks the pointwise Hermite identity, which is
the only independent check `E` has (#85 is the failure this rules out).

### Step 2: the pre-contraction kernel, `Target.DQ`

**Status: implemented** (#153, `src/felizjueves/kernel_dq.py`; emitted as
`dq_{lc}_{ld}` by #156; run on both cards by #157). The built kernel has one
parameter the specification below did not spell, **`n_pair` (I32)**, after
`n_rec`. It is `Dcd`'s leading extent, because an array extent in the IR must
be a scalar `Param`. Row-major indexing never reads it, so the emitter casts
it to `(void)` (#156; hipcc's `-Wunused-parameter` otherwise fails #92's
filter). The kernel has no `ABS` or `MAX` function to call, so it writes
`|x|` and `max` with `LT` and `SUB` (`0.0 − x` is `−x` exactly).

One kernel per ket pair class `(lc, ld)`. With `n_q = |hermite_order(lc+ld)|`
and `n_cd = ncart(lc) · ncart(ld)`:

```
params  n_rec (I32)  n_pair (I32)
        eta[n_rec]  XPA[n_rec, 3]  XPB[n_rec, 3]  s[n_rec]  w[n_rec]   per record
        pair[n_rec] (I32)                    the record's row of Dcd
        Dcd[n_pair, n_cd]                    D_cd per ket shell pair, row order above
        D[n_rec, n_q]  (OUT)   Dmax[n_rec]  (OUT)
```

Per record `k`, with `E` the undressed block of record `k`:

```
acc_q   = Σ_{r in row order} Dcd[pair[k], r] · (s[k] · E[r, q])
D[k, q] = w[k] · acc_q
Dmax[k] = max_q |D[k, q]|
```

**Preconditions.**

- `D` is symmetric. The host checks this (`ao.density.checked_density`) and rejects an
  asymmetric `D`; the kernel never symmetrises.
- `w` is 2.0 for an off-diagonal ket shell pair (`c ≠ d`) and 1.0 for a
  diagonal one. **The host decides it**, and it is folded into `D_q` and so
  into `Dmax`. The kernel never decides `w`. A weight of 1 off the diagonal is
  the order-unity error this design is most likely to ship.
- Only unique ket shell pairs `c ≥ d` are records.
- `eta > 0`.

**What the output is.** `D` is the `D[n_ket, n_q]` buffer the `[p|q]` kernel
reads, and `Dmax` is the vector it screens on, over the weighted, dressed
`D_q`. The two are in **record order**, not the sorted order `[p|q]` requires
(see "The round-trip" below). `Dmax` is a reduction over the kernel's own
output. It is not the AO density's largest element and cannot be computed
from it. Lanes may map one thread per record; the mapping is not part of the
numerical contract.

**The q-major layout** (opt-in, `passes.transpose(kernel, ("D",))` on both
this kernel and `[p|q]`): `D[n_q, n_ket]` instead. `[p|q]`'s lanes are
consecutive kets, so row-per-ket puts one warp's read of `D[k, q]` `n_q`
doubles apart per lane, and q-major makes it contiguous, as `[Fock, §4.2]`
does ("Dq is transposed before kernel launches to ensure contiguous reads").
The two kernels must agree on the layout; the values and the bits are the
same (`test_passes.py`). Not run on a card here.

**The AO-density variant** (opt-in, `build_dq_kernel(cls, density="ao")`,
unit `dq_{lc}_{ld}_ao`). It reads `D_cd` where the density already is,
`Dao[off_c[k] + i, off_d[k] + j]` for row `r = (i, j)`, with the record's own
shell offsets. This is the addressing `JAB` uses to write `J`. `pair`, `Dcd`
and `n_pair` give way to `off_c`, `off_d`, `Dao[n_ao, n_ao]` and `n_ao`.
The values and their order are the gather's, so the two variants are `==`,
interpreted, for every class (ss)–(ff) on HF/cc-pVTZ (`test_kernel_dq.py`).
A consumer whose density is on the device needs no gather kernel and no `Dcd`
buffer. A pair stored swapped (`c` the shell of momentum `lc`, whatever its
index) reads its own orientation of the symmetric `D`, with no transposed
copy. `w` is still the host's. Not measured, and not run on a card.

**Equality.** In the operation order above, an interpreted kernel is `==` to
`ao.j.dq_from_host`. That reference also skips `D_cd = 0` terms, which changes
no bit, because adding `±0` to an accumulator that started at `+0.0` is the
identity. A kernel that factors `s · w` out of the sum is admitted. It agrees
with the host to fp64 round-off and not bitwise, and its test says which
order it uses.

### Step 3: the `[p|q]` kernel

**Status: implemented** (`kernel.py`, "The `[p|q]` kernel contract" above).
Its parameter block, `P zeta Ip | Q eta Iq Dmax | D | J`, is unchanged by the
chain. It takes `D` and `Dmax` from step 2, after the sort, and writes `J_p`
per bra record in `hermite_order(lp)`, one buffer per bra class (milestone 6),
by fp64 atomics when the class is split.

### Step 4: the bra-transform-and-scatter kernel, `Target.JAB`

**Status: implemented** (#154, `src/felizjueves/kernel_jab.py`; emitted as
`jab_{la}{lb}` by #156; run on both cards by #157). The built kernel departs
from the specification below in two places, both forced by the IR and
spelled in the block:

- The per-record exponent is **`zeta_ab`**, not `zeta`. The `E` fragment binds
  a *scalar* named `zeta`, and #149's hygiene refuses a host array of the
  same name. The kernel loads `zeta_ab[rec]` into a local `zeta` and binds
  that.
- **`n_rows` (I32)** is an extra parameter: `Jp`'s leading extent, which has to
  be a scalar `Param`. As with `DQ`'s `n_pair`, nothing reads it, and the
  emitter casts it to `(void)`.

The diagonal-pair branch is built only when `la == lb`. The launch shape is
one thread per record (`jab_launch_shape`).

One kernel per bra pair class `(la, lb)`. With `n_p = |hermite_order(la+lb)|`:

```
params  n_rec (I32)  n_ao (I32)  n_rows (I32)
        zeta_ab[n_rec]  XPA[n_rec, 3]  XPB[n_rec, 3]  s[n_rec]            per record
        off_a[n_rec]  off_b[n_rec]  (I32)    the record's shells' AO offsets
        row[n_rec] (I32)                     the record's row of Jp
        Jp[n_rows, n_p]                      the bra class's J_p buffer (step 3)
        J[n_ao, n_ao]  (IN/OUT, AtomicAdd)
```

Per record `k` and Cartesian pair `r = (i, j)`:

```
v = Σ_{p in hermite_order} (s[k] · E[r, p]) · Jp[row[k], p]
off-diagonal (off_a ≠ off_b):  J[off_a+i][off_b+j] += v;  J[off_b+j][off_a+i] += v
diagonal     (off_a = off_b):  only i ≥ j;  J[off_a+i][off_a+j] += v,
                               and for i ≠ j also J[off_a+j][off_a+i] += v
```

**The diagonal rule** reproduces the host bit for bit. It is not a choice made
for the kernel. `ao.j`'s host scatter assigns `J[a][b]` and then, transposed,
`J[b][a]`, row by row. On a diagonal block, element `(x, y)` is assigned twice
and keeps the later value, which is `J_ab[max(x, y), min(x, y)]`. So the
block's lower triangle, mirrored, is the host's result. Writing the whole
block and its mirror would double-count the diagonal pair. Writing the whole
block without a mirror would put `J_ab[x, y]` where the host has
`J_ab[y, x]`. The two are equal mathematically. On the tracked H₂O fixtures
(STO-3G, 6-31G, cc-pVDZ) they were also equal bit for bit when checked for
this section, but nothing makes the `E` recurrence round symmetrically in `a`
and `b`. The lower-triangle rule is the one that equals the host by
construction. Under it, every element of `J` is written by exactly one bra
shell pair, once per record.

**Preconditions.**

- `J` is **zeroed by the caller** before the first `JAB` launch of a build.
  This is the same convention `build_pq_kernels`' atomic write-back already
  requires.
- Only unique bra shell pairs `a ≥ b` are records.
- A record's `Jp` row is complete. Every ket class's `[p|q]` launch into it
  has finished.

**Why atomic.** Several records of one shell pair, and both mirror positions,
add into the same elements. The write is an `AtomicAdd`, with that reason as
its `Note`.

**Equality.** Interpreted with records in order, the kernel is `==` to
`ao.j.jab_from_host`. The `+=` chain starting from a zeroed `J` is the host's
`jab[r] += …` starting from `0.0`, and `+0.0 + v` is `v` (it differs only in
the sign of a zero, which `==` does not distinguish). On a card, fp64 atomics
make the summation order differ from run to run. The tolerance for that is
stated where the device test asserts, and not here.

### What stays host-side, and the one round-trip

**Host-side, by decision**, as in milestone 6:

- the **pair list**: unique shell pairs `a ≥ b`, the records of each, and the
  weight `w`;
- the **density check**;
- the **Schwarz bound** `I = Σ_p I_p` per record (`ao/pairs.py`,
  pair-factor-free, so it needs no `D` and no dressing). Since
  `Target.PAIR_BOUND` it is also generated, for a consumer whose records are
  on the device: `kernel_pair_bound.py`, one thread per record, the bound's
  closed form at `PQ = 0`, `sqrt(x) Σ_n D_n x^n` with `x = sqrt(ζ)` and
  `D_n` constants of `L = la + lb` folded at build time. It agrees with
  `ao.pairs.pair_bound` through `reference.pq`, a derivation that shares no
  code with it, within 16 eps, interpreted for every class `la ≤ lb ≤ 4` and
  on the RTX 3080 (`test_kernel_pair_bound.py`, `test_pair_bound_device.py`);
  the RX 9060 XT run is #431's. The launch code here still forms it host-side;
- the **sort** of each ket class, descending by `I_q · Dmax`, stable
  (`launch.launches`);
- the **staging**: class-grouped buffers, one upload per bra class and one per
  ket class (`ao/staged.py`, #143; `test/plan_driver.py`, #144);
- the **launches** and their order;
- each record's **inputs** to the kernels: `ζ` or `η`, `X_PA`, `X_PB`, the
  dressing scale `s`, and the AO offsets.

These decide launches, or feed the kernels, which is why they stay on the
host. Milestone 7 left all of them where milestone 6 put them.

**The round-trip.** The sort needs `Dmax`, and step 2 produces `Dmax` on the
device. So the chain reads `Dmax` back, sorts on the host, and then puts
`D_q` into the ket class's buffer in sorted order before step 3 launches.
The gather can be a host copy or a device gather. Either way it waits on the
read-back. This is the chain's **only** device-to-host transfer before `J`
itself, and it is named here, and in the chain adapter's docstring (#155),
as the one a launch program would remove. **Status: inherent to the
host-side sort;** removing it is out of this milestone's scope. On the cards
(#157) it is wider than the minimum: the `DQ` program and milestone 6's
driver are separate programs, so `D_q` is read back with `Dmax` and staged
into the driver's ket buffers by the host.

### The seams: `ao.j`'s `dq=` and `jab=`

**Status: implemented** (#151). `ao.j.coulomb_matrix(basis, D, *, pq,
tau=0.0, jp=None, dq=None, jab=None)` follows `jp=`'s discipline (#87). Each
seam replaces one step, so a disagreement in a `J` built through it is
localised to that one kernel. A single combined "device chain" hook would make
a disagreement impossible to localise. Both seams receive an `ao.density.EPair`
(`la, lb, A, B`, and per record `(P, ζ, s)` in `primitive_pairs` order), built
by `ao.density.epair`.

| seam | called | signature | the module keeps |
|---|---|---|---|
| `dq=` | once per unique ket shell pair `c ≥ d`, in sweep order | `dq(ket: EPair, dcd, w) -> rows`: one `D_q` row per record, over `hermite_order(lc+ld)` | the pair list, `w`, the extraction and symmetry check of `D`, `Q` and `η`, everything downstream |
| `jab=` | once per unique bra shell pair `a ≥ b`, in sweep order | `jab(bra: EPair, jps, (off_a, off_b), J) -> None`: adds into `J` by the rule of step 4 | the pair list, `J_p` (from `pq` or from `jp=`), and the zeroing of `J` |

- **Both seams are unscreened.** `tau > 0` with either one is refused, as with
  `jp=`.
- **The references.** `ao.j.dq_from_host` and `ao.j.jab_from_host` are each
  step's host arithmetic in the host's own order. With them, `coulomb_matrix`
  is `==` to itself without them, for each seam alone and for both, on the
  `pq` path and on the `jp=` path (`test/test_ao_j.py`).
- **With no hook supplied, every path runs exactly the operations it ran
  before the seams existed.** That includes `math.fsum`'s placement in the
  screened sweep, which the seams do not touch.
