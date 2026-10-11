# The J engine: what the generator emits

**Status: specification, not description.** The generator does not exist yet
(see CLAUDE.md, "The state of the tree"). This file is the derived spec it must
satisfy — our notation, our vocabulary, our batching config — written so a test
can be read against it rather than against a paper.

**Implemented so far: step 1 of the pipeline, the split.** Target enumeration
is `src/felizjueves/plan/targets.py`; `batch_direction`, `in_batch` and the
sub-batch partition are `src/felizjueves/plan/batching.py`. The partition's cut
rule is **minimax** — the largest batch minimised, `[JEngine, v2, §5]` — with
ties to the lowest boundaries and one set of cut points shared by `X`, `Y` and
`Z`; that is the rule, wherever this file says "equalise". It computes
`[JEngine, v2, Table 3]` (targets per batch), scored on the `literal` board.
Everything else here — the recurrences, MILD, the sieve, emission, Tables 1–2 —
is still specification.

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

**The grid itself is deliberately not copied here.** Once the generator exists
it computes these counts itself, so the artifact we want tracked is *our* output
as a golden file, with the paper cited as what it was cross-checked against.
Read the full grid from the paper when implementing; commit ours.

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
