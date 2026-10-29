# Rys quadrature `F` and `J`, and the device Boys

Moved verbatim from `.claude/CLAUDE.md`, "The state of the tree" (indexed by `../SKILL.md`).

**The fused `F` on Rys quadrature too, interpreted and emitted, NOT
compiled by nvcc / hipcc and NOT run on a card** (`docs/implementation/rys_fock.md`
is the one place with the detail; #468's spike, `experimental/rys/`, brought
into `src/`): `Target.F` on `Engine.RYS` (`spec.Engine.RYS`, key slot
`f.rys`, refused on every target but `F` and `J` (below), no lever), `eri/fock/kernel_frys.py`'s
`build_frys_kernel` (`fr_{la}{lb}{lc}{ld}`, the 21 classes through
`(dd|dd)`): `kernel_f`'s skeleton and seams with `ket_prim`,
`quartet_scalars`, `vrr` and `accumulate` replaced -- the `N = L // 2 + 1`
point Rys rule (`ir.Rys`, a new statement like `Boys`: fp64, injected into
the interpreter as `rys=`, spelled `fj_rys<N>` by `emit/cuda.py` /
`emit/hip.py`, refused by `emit/gpu.py`) and per node the 2D VRR
(`rys/recurrence.py`) into the HGP boundary, so the HRR, digestion,
staging, screen, parameters and launch are `f_`'s and an `fr_` unit is a
drop-in (its IR outside the `K_CD` loop `==` `kernel_f`'s, tested). The rule
(`rys/roots.py`: Laguerre constants from `T = 80`, below it Stieltjes on an
88-point Legendre half, Sturm bisection, Christoffel weights) is within
3.4e-15 of a 40-digit route (`rys/exact.py`) for `N <= 5`, `T` in
`[0, 1e6]`, and its device text run as host C++ is bitwise the host rule
(`test_rys_roots.py`). Interpreted (`test_kernel_frys.py`): the ERI tap
within 6.3e-15 of MMD's scale in all 21 classes, the deposit against
`eri.fock.fock`'s sweep, whole H₂ / H₂O STO-3G `G` within 2.2e-15 of scale
of libcint. `python -m felizjueves.emit --fock-l LMAX --rys`, never in the
bare default nor in `KERNEL_GEN_CMD`; 21 goldens per backend
(`test_emit_frys.py`, two in `frys_digests.json`). All 21 units passed clang
18's CUDA front end (a proxy). Nothing measured or scored.
**The split `J` on Rys quadrature too, interpreted and emitted, NOT
compiled by nvcc / hipcc and NOT run on a card** (`rys_fock.md`, "The split
`J` on Rys"): `Target.J` on `Engine.RYS` (slot `j.rys`, whose only legal
lever is HGP J's `ordered`), `eri/hjk/kernel_rj.py`'s `build_rj_kernel`
(`rj_{rep}` canonical / `rj_o_{class}` ordered, `L <= 8`): `kernel_hj`'s
split `J` through a seam it now exposes (`kernel_hj.Nest`,
`split_j_kernel`, `hgp_nest`; every `hj_` golden byte-identical) with the
Rys rule and 2D VRR as the nest (`rys.recurrence.quartet_rule`, shared with
`kernel_frys`, its text unchanged), so an `rj_` unit is a drop-in for the
`hj_` unit of the same class and mode (IR outside the `K_CD` loop `==`,
signature equal). `walk_hjk(..., build=)` walks `launch_hjk`'s plan with
the caller's kernels. Interpreted (`test_kernel_rj.py`): the ERI tap against
MMD element by element and HGP per block within `1e-12 (s_hgp + s_mmd)`,
the deposit against `eri/j_coulomb/j.py`'s restricted sweep, both modes, every class
through `(dd|dd)` (`L > 4` `slow`), whole H₂ (and H₂O, `slow`) STO-3G `J`
against the MMD sweep and libcint's `vj`. `python -m felizjueves.emit
--hgp-j-l LMAX --rys` (0..2) or `--hgp-class ... --rys`, never in
the bare default nor in `KERNEL_GEN_CMD`; 63 goldens per backend plus 8
digests (`test_emit_rj.py`, `rj_digests.json`). All 71 cuda units passed
clang 18's CUDA front end (a proxy). Nothing measured; a `j.rys` row is
refused by the store.
**The split `K` on Rys quadrature too, interpreted and emitted, NOT
compiled by nvcc / hipcc and NOT run on a card** (`rys_fock.md`, "The split
`K` on Rys"): `Target.K` on `Engine.RYS` (slot `k.rys`, `am` K's Hermite
pairing, only `ordered` legal), `eri/hjk/kernel_rk.py`'s `build_rk_kernel`
(`rk_{rep}` / `rk_o_{class}`, `L <= 8`): `kernel_hk`'s split `K` through a
seam it now exposes (`kernel_hk.split_k_kernel`, `hgp_nest`; every `hk_`
golden byte-identical) with `kernel_rj.rys_nest` as the nest, so an `rk_`
unit is a drop-in for the `hk_` unit (IR outside the `K_CD` loop `==`,
signature equal). Interpreted (`test_kernel_rk.py`, `L > 4` `slow`): the
tap against MMD and HGP within `1e-12 (s_hgp + s_mmd)`, the deposit against
`k_quartet` per class (worst 3.2e-16), both modes, every class through
`(dd|dd)`, the density mutation and AO-order trap failing, whole H₂ (and
H₂O, `slow`) STO-3G `K` against the MMD sweep and libcint's `vk` (worst
1.7e-15 of `S`). `--rys` takes `--hgp-k-l LMAX` (0..2), and `--hgp-class
... --rys` writes both `rj_` and `rk_`; 63 goldens per backend plus 8
digests (`test_emit_rk.py`, `rk_digests.json`). All 102 cuda units passed
clang 18's CUDA front end (a proxy). Nothing measured; a `k.rys` row is
refused by the store.
**The device Boys is one template, `fj_boys<M>`** (#301; `emit/_c.py`'s
`BOYS_DEVICE`, every unit with a fundamental): every order `F_0..F_M` of one
`T` from one `exp(−T)`, the Kummer series at `M` and the downward recurrence
below `T = M + 8`, `erf` and the upward recurrence from it. It is witnessed on
the RTX 3080 over `reference.boys`' whole domain, `T ∈ [0, 1e6]`, `M ≤ 16`
(worst 2.1e-15; `test_kernel_device.py`), and compiled only on `gfx1200`
(#431). Its text moved every J, K and F unit, so `kernels.json`'s
`source_sha256` pins are **stale** until those rows are re-measured:
`devtools/measure.sh check` fails until then.
