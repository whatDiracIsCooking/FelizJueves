# Analytic nuclear hessians by McMurchie–Davidson: `S`, `T` and `V`

**Status: implemented host-side and as generated kernels, interpreted
only. Not compiled by nvcc or hipcc, and not run on either card.** This is
the derived note for the first `derivative_order = 2` keys: the contracted
second nuclear derivatives of the overlap, the kinetic energy and the
nuclear attraction (its operator's centre–nucleus and nucleus–nucleus
terms included) -- the one-electron skeleton of an SCF hessian. It says
what is generated, what it is checked against, and what is not done. The
session that built it had no device compiler and no card; every kernel
text (every ordered class through (ff)) passed clang 18's CUDA front end
(device side, `-fsyntax-only -Wall -Wextra`, a declaration stub for the
runtime) with no diagnostic, which is a proxy, not the kernel tier.

**Nothing here is scored.** No published table covers these hessians; the
rule is McMurchie & Davidson's (`docs/references.md` entry 46) applied
twice, and the kernels are this repo's own design. No `claims.json`
record, cell or floor was added.

The spike this ports is `experimental/mmd/hessian/` (s and p, numpy),
which checked the same rule against libcint's `int1e_ipip*` / `_ip*ip`
and `pyscf.hessian.rhf` to ~1e-11 of max. That spike is not imported
anywhere; its two-electron second derivatives and its CPHF response are
not ported (below).

## The keys

`spec.DERIVATIVE_ORDERS` is `(0, 1, 2)`. Order 2 is legal on
`spec.HESSIAN_TARGETS` only -- `S`, `T` and `V`, each on MMD -- and every
other target, and HGP, refuses it with `_HESSIAN_REFUSAL`. A hessian
class is its integral's class, so `overlap/d2/(2,1)/b1/s/literal`
differs from the d0 and d1 keys in the `d` segment alone, and every
existing key is byte-identical. Unbatched, for each target's d0 reason.

## What a d2 kernel writes

As at d1, not the derivative integrals (a `[n_atom, 3, n_atom, 3, n_ao,
n_ao]` tensor) but their contraction with a matrix:

    H[A, k, B, l] += SUM_mn M_mn d2X_mn / dR_{A,k} dR_{B,l}

`M = D` for `T` and `V`, `W` for `S`; the signs are the caller's. The
per-Cartesian-pair weights are `gradient.host.pair_weights`'
(`M[a, b] + M[b, a]` off the diagonal, `M[a, a']` on it), exact for any
`M`. `H` is zeroed by the caller; every kernel only adds.

## The rule

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

## The host reference (`hessian/host.py`)

`overlap_hessian`, `kinetic_hessian`, `potential_hessian(one_centre=)`
and `nuclear_repulsion_hessian`, pure Python on the AO layer, over
`gradient.host`'s sweep (shell pairs `i <= j`, `primitive_pairs` order,
`shell_atoms`, `pair_weights`). **By another route than the kernels**:
`experimental/mmd/hessian/ints2.py`'s, each 3-D Cartesian differentiated
as a list of `(coefficient, moved Cartesian)` (`d_centre`, `d2_centre`)
and every term an ordinary 3-D integral of moved Cartesians, `T` by the
3-D Laplacian on the ket. `absolute=True` takes every factor in `|.|`:
the round-off scale the kernel tests use.

## The kernels (`hessian/kernel_h1e.py`)

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
`d2t` 32 KB to 456 KB, `d2v` 35 KB, 190 KB at (pp), 1.08 MB at (dd) and
4.9 MB at (ff). No low-stack form is built.

## The generator

`python -m felizjueves.emit --hess-l [LMAX]` (2 bare, 0..3, a 2 above)
adds `d2s_`, `d2t_`, `d2v_{la}{lb}` for every `la <= lb <= LMAX` beside
the J ladders; never in the no-flag output and not in `KERNEL_GEN_CMD`.
The goldens are the six classes (ss) through (dd) per backend
(`test/golden/{cuda,hip}/d2{s,t,v}_*`); `d2v_22`, over the 512 KiB file
cap, is pinned by byte count and sha256 in
`test/golden/hessian_digests.json`.

## Witnesses, and what ran where

| What | Against | Where | Worst |
|---|---|---|---|
| `hessian.host`, `S` `T` `V` | central FD (`h = 1e-4`) of `gradient.host`, itself FD-held to the libcint-witnessed integrals | H₂O/STO-3G, a two-atom s/p basis; H₂O/cc-pVDZ `slow` (`test_hessian_host.py`) | ~3e-9 – 8e-9 of `max\|H\|` (`O(h²)`), gate 1e-7 |
| the same | `SUM_A H = 0`; `H = H^T` | the same | invariance 4.5e-12 of `max\|H\|` (`T`, the O 1s cancellation; gate 1e-10); symmetry ~5e-16 (gate 1e-13) |
| `nuclear_repulsion_hessian` | FD of `nuclear_repulsion_gradient` | H₂O | gate 1e-7 |
| `kernel_h1e`, interpreted | `hessian.host` per class, `M` non-symmetric | every ordered class (ss)–(ff) on HF/cc-pVTZ (above L + 2 = 4 `slow`); whole H₂O/STO-3G (`test_kernel_h1e.py`) | 0.35 eps of the absolute twin (`d2v_12`), gate 16 eps |
| the units | clang 18 CUDA front end, `-Wall -Wextra` | all 48 ordered classes through (ff), here | no diagnostic (a proxy) |
| nvcc / hipcc compile | #92's warning filter | `test_emit_hess.py`'s `cuda` / `hip` cases | **not run** (no compiler here) |
| on a card | `hessian.host`, within 1e-14 of the absolute twin | `test_hess_device.py` (generator `test/hess_driver.py`), H₂O/STO-3G, cc-pVDZ `slow` | **not run** (no card here) |

## What is not done

- No two-electron second derivatives (`d2 (ab|cd)`, `J` / `K`), no CPHF
  response, no assembled RHF hessian; `experimental/mmd/hessian` has all
  three for s and p, in numpy.
- Not compiled by a vendor compiler, not run on a card, nothing measured.
- Unrolled only: `d2v` is 1 MB of text at (dd) and 4.9 MB at (ff); a
  low-stack form (#586's per-axis loops) is the named follow-up.
- No screening, Cartesian only.
