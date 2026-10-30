# The dipole, and one-electron on HGP and Rys

Moved verbatim from `.claude/CLAUDE.md`, "The state of the tree" (indexed by `../SKILL.md`).

**The dipole `MU = <a| q (r - C) |b>` is generated too, and ran on both
cards** (milestone 19, #517–#524; `one_electron.md`, "The dipole `MU`";
nothing consumes it yet): `Target.DIPOLE` (`spec.py`, #518; `am` a pair
`(la, lb)`, operator `DIPOLE`, unbatched for `S`'s structural reason, HGP
refused, no lever; the order `e` a builder argument and the origin `C` and
charge `q` runtime scalars, none of them key fields, so no stored key
moved) whose kernel (`one_electron/kernel_mu.py`, #521) is **one fused
kernel per pair class writing every component**: one thread per record,
`e_fragment.multipole_fragment(la, lb, 1)` (#519; the Hermite route
`SUM_t E_t M^e_t`, `M^e_t` by HJO §9.5, `docs/references.md` entry 48, its
equation number unchecked) inlined once, each value `AtomicAdd`-ed into
`MU[3, n_ao, n_ao]` as it forms through the fragment's `store=` hook, by
`kernel_jab`'s rule per component, `MU` zeroed by the caller;
`with_overlap=True` adds `S` as component 0. Host side,
`one_electron/multipole.py` (#520; not `ao/`, per milestone 18) is the
reference, general in `e`, `overlap_block` bitwise at `e = 0`, and
`walk_one_electron(..., Target.DIPOLE, origin=, q=)` rebuilds `MU` from a
`one_electron_launch` plan. The ladder is `python -m felizjueves.emit
--mu-l [LMAX]` (`mu_{la}{lb}`, `la <= lb`, 2 bare, **0..3 only** since
#627, exit 2 above), never in the bare default and not in `KERNEL_GEN_CMD`. The oracles
are libcint's `int1e_r_cart` at a non-zero origin and at 0 and PySCF's
`dip_moment`, through `h2o_{sto3g,631g,ccpvdz}_dipole` (#517, on their `_j`
siblings' inputs) and `hf_ccpvtz_spd_dipole` (#520, HF/cc-pVTZ without f,
40 AOs, no density, all nine ordered classes off-centre), plus the
origin-shift identity. **Correctness is tested at L <= 4 only** (every
shell `l <= 2`, (ss) through (dd)), by the user's decision: the host
(`test_ao_multipole.py`, worst 8.8e-16), the kernel interpreted, bitwise
against the host block (`test_kernel_mu.py`), the six goldens per backend
(`test_emit_mu.py`), and on **both** cards (`test_mu_device.py`, generator
`test/mu_driver.py` -- on the shared runner since #658, which ran on both cards on 2026-10-30, f cases included -- PR #546: 4 passed / 0 skipped on the RTX 3080 and on
the RX 9060 XT, the slow cc-pVDZ case 5 passed on each; worst 8.62e-16 of
`max|r|` against libcint and 3.16e-16 against `dip_moment`, the same on
both cards). #627 lifted the cap to (ff): `--mu-l` accepts 0..3 (2 bare), the driver refuses above (ff), and every class (ss) through (ff) is held to libcint's `int1e_r_cart` on the host and interpreted through `hf_ccpvtz_dipole` (HF/cc-pVTZ with its f shells, 50 AOs, no density; worst 2.25e-15 of `max|r|`), with ten goldens per backend, compiled warning-free by nvcc (sm_86) and hipcc (gfx1200); the f classes (`test_mu_device.py`'s `slow` f cases) **ran on both cards** for the first time with the #658 runner port (2026-10-30), passed on each. Nothing about it is measured or scored
(no published board; a `StructuralLimit`, #524; #627's compile reading, 0
stack / 0 spills on sm_86 through (ff) and 92 B a lane of scratch for
`mu_33` on gfx1200, is not a row, and no low-stack form was needed);
`e = 1` only, Cartesian only.
**The quadrupole (raw second moments) is generated too, interpreted only**
(`one_electron.md`, "The quadrupole"): `Target.QUADRUPOLE` (`spec.py`; `am`
a pair `(la, lb)`, operator `QUADRUPOLE`, unbatched as the dipole, HGP and
Rys refused, no lever) is the dipole's builder at `e = 2`
(`kernel_mu.build_quad_kernel`, `quad_{la}{lb}`; `kernel_mu.ORDER`): the six
components xx, xy, xz, yy, yz, zz of `<a| q (r - C)(r - C) |b>` into
`MU[6, n_ao, n_ao]`, the dipole's parameters, record and scatter, every
`mu_` text byte-identical. Not the traceless tensor. The plan carries it
(`OneEClass.arrays(Target.QUADRUPOLE)`, `walk_one_electron(...,
Target.QUADRUPOLE, origin=)`). The oracle is a separate witness,
`test/oracle/quadrupole/hf_ccpvtz_spd_quadrupole.json` (libcint's
`int1e_rr_cart` on `hf_ccpvtz_spd_dipole`'s inputs at its origin and at 0;
`python -m oracle.generate --regenerate --quadrupole`,
`manifest.quadrupole_integrals`, refused on any input not the parent's).
Interpreted, through (dd): the kernel bitwise the host block per class, the
plan and walk within 1.1e-15 of libcint (`test_kernel_quad.py`), the host
the same (`test_ao_multipole.py`); `--quad-l [LMAX]` (2 bare, 0..2 only),
never in the bare default nor `KERNEL_GEN_CMD`; 6 goldens per backend
(`test_emit_quad.py`). **Compiled by nvcc and hipcc, and run on both cards**:
`test_quad_device.py` (on `mu_driver`'s `target=`, on the shared runner
since #658) builds every class's runner under both compilers with no
warning (2 passed, 2026-10-30, a device-less cloud container), and its
device case (every class through (dd)) passed on the RTX 3080 and the
RX 9060 XT on 2026-10-30, its first card run. Nothing measured.
**`S`, `T`, `V` and the dipole are also built on the HGP engine, s and p
only, interpreted only** (`one_electron.md`, "The HGP route"):
`Target.S` / `T` / `V` / `DIPOLE` with `engine=Engine.HGP` (slots `s.hgp`,
`t.hgp`, `v.hgp`, `dipole.hgp`; `SX` stays refused), built by
`one_electron/kernel_hgp.py` (`hs_`, `ht_`, `hv_`, `hmu_{la}{lb}`, `l <= 1`,
refused above): one thread per contracted shell pair, `hgp.vrr`'s eq 25 at
`f = 0` per primitive pair (its point-charge limit for `V`), the contraction,
then `hgp.hrr`'s eq 28 on contracted values, plain stores, no atomics.
Staged by `OneEClass.hgp_arrays`; `python -m felizjueves.emit --hgp-one-e-l
[LMAX]` (1 bare, 0..1 only), never in the bare default nor `KERNEL_GEN_CMD`.
Witnessed interpreted (`test_kernel_hgp.py`) against the MMD host and
libcint's `int1e_ovlp` / `kin` / `nuc` / `r_cart` on H₂O/STO-3G and 6-31G;
goldens in `test_emit_hgp_one_e.py`. **Compiled and run on both cards** (`test_hgp_one_e_device.py`, generator `test/hgp_one_e_driver.py`
-- on the shared runner since #658, which ran on both cards on 2026-10-30 -- 16 passed on the RTX 3080 before the port: within 5.0e-16 of `max|M|` of the MMD host on the synthetic
basis and 1.0e-15 of libcint on H₂O/STO-3G and 6-31G, runs bit-identical);
since the port its 12 `gpu` cases passed on the RTX 3080 and, for the first time, the RX 9060 XT (2026-10-30), nothing measured.
**`S`, `T` and `V` are also built on the Rys engine, through (ff),
interpreted only** (`one_electron.md`, "The Rys route"): `Target.S` / `T` /
`V` with `engine=Engine.RYS` (slots `s.rys`, `t.rys`, `v.rys`; `DIPOLE` and
`SX` refused), built by `one_electron/kernel_rys.py` (`rs_`, `rt_`,
`rv_{la}{lb}`, `l <= 3`): per primitive pair a product of three 1D
integrals -- the nodeless overlap recurrence for `S` / `T`, and for `V` per
nucleus the `N = (la + lb)//2 + 1` point Rys rule (`ir.Rys`, `fj_rys<N>`)
with `rys.recurrence.root_loop`'s bra half at the point-charge limit --
into `kernel_hgp`'s contracted boundary, filling its pair skeleton
(`kernel_hgp.assemble`, extracted with every HGP text byte-identical), so a
Rys unit is a drop-in for the `hs_` / `ht_` / `hv_` unit (same parameters,
staging, launch). `python -m felizjueves.emit --rys-one-e-l [LMAX]` (3
bare, 0..3 only), never in the bare default nor `KERNEL_GEN_CMD`.
Interpreted (`test_kernel_rys_one_e.py`) within 2.8e-15 of `max|M|` of the
MMD host through (ff) and against libcint on H₂O/STO-3G, 6-31G, cc-pVDZ and
`prim_spdf` (worst 7.5e-14, `V` on `prim_spdf`, the host's own distance);
24 goldens per backend plus 6 digests (`test_emit_rys_one_e.py`). **Not
compiled by nvcc or hipcc and not run on a card** (clang 18's CUDA front
end passed all 30 units, a proxy); nothing measured.
