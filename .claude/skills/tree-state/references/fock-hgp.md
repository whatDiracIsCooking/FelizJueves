# The HGP engine, the fused `F` and the split HGP `J`/`K`

Moved verbatim from `.claude/CLAUDE.md`, "The state of the tree" (indexed by `../SKILL.md`).

**Since milestone 12 the repo has a second ERI engine and the first `F`**
(`docs/implementation/hgp_fock.md` is the one place with the detail): the
Head-Gordon–Pople recurrence of `[Fock]` §3.1 and the **fused** Fock kernel
of §3.2 on it, `Target.F` (`spec.py`, #272; `am` in AO order
`(la, lb | lc, ld)`, the opposite of `K`'s Hermite pairing; `n_batches = 1`
by its own refusal; the plan layer refuses it). The recurrence is `hgp/`:
the VRR (eqs 25–26, `vrr.py`, a DAG **not graded by `m`**), the HRR (eq 28,
`hrr.py`, on contracted values), the FLOP count (`cost.py`, calibrated
against nothing) and both as IR (`mmd/recurrence.py`, after #277's signed
`Term.c` and two derived `Factor`s). The kernel (`eri/fock/kernel_f.py`, #282, #283)
uses §3.2's mapping: one block per bra pair, one thread per contracted
quartet, the `K⁴` contraction **inside** the kernel because eq 28 needs
contracted values (961 boundary accumulators at `(dd\|dd)`, the bounded
`hgp.vrr.target_set`, not the issues' 1,225), a six-block screen that
**skips, never breaks**, and the 8-fold digestion into `G = J − K/2` with
its weights **derived** (Coulomb 2, exchange −½, `deg = |orbit|/8`) and
held to `eri/fock/fock.py`'s every-ordered-quartet sweep. Around it:
`eri/hgp_drive.py` (HGP ERIs interpreted, pinned to `eri/eri.py` in 21
classes, #281), `eri/fock/launch_f.py` (the pair stream: pruning, degree-uniform
batches, eq. 29's screen in its sound reading, #285), `eri/fock/walk_f.py` (its
oracle, #286), `ir/lower.py`'s loop nest (#284) and `engine.py` (#292, the
`Engine` protocol extracted last, its docstring listing what was pushed back
down into MMD). On both cards every one of the **21 classes** (every shell
`l ≤ 2`, through `(dd\|dd)`) matched the sweep (worst `2.2e-14·scale_G`
against `1e-13`), and the whole `F` of H₂O/STO-3G, 6-31G and cc-pVDZ matched
libcint's `vj − vk/2` (worst 7.35e-15 against `1e-12`), #288 and #290; all
42 F rows are in `kernels.json` with gated wall-clock (`(dd\|dd)`: 255
registers and 97,464 B of spill stores, 2.33 s per launch on `sm_86`; 192
VGPRs, 3.70 s on `gfx1200`). `[Fock]` §3's claims are scored:
10 [Fock] records: 9 asserted, 1 unverifiable, 0 pending
(`data/literature/fock/claims.json`, #291). **No published cell is scored**
(the paper has no register table for the §3.2 kernel, and its tables are
A100/V100 wall-clock); milestone 12 built only the **diagonal** cells of
`[Fock]`'s 2x2 (the HGP split is milestone 14's, below; no fused MMD `F`,
#326); no gradient, no lever, nothing persists across SCF iterations, `f` and above
are not built (#330), and the screening comparison the fused scheme most
wants is unmeasured on compact fixtures (#329, #331). `KERNEL_GEN_CMD`
writes no `F` unit: `--fock-l` / `--fock-class` is its own emit path.
**Since milestone 14 the 2x2's (HGP, split) cell exists too, and ran on the
RTX 3080 only** (#401–#408, PRs #469–#477; `hgp_fock.md`, "The split cell",
is the one place with the detail): `Target.J` and `Target.K` on
`Engine.HGP`. `Spec.engine` (#401) is the recurrence as a key field,
`None` resolving to the target's native engine (`F` HGP, the rest MMD), the
key spelling only an off-native one as a slot suffix (`.../j.hgp/...`), so
every stored key is byte-identical; HGP is legal on `J` and `K` only and MMD
`F` is refused (#326); `am` is the recurrence's pairing (HGP `K` is
`(la, lc | lb, ld)`, as MMD's). `hgp/orbits.py` (#402) is the 8-fold orbits
and each one's cheapest representative (50 ordered classes in 13 orbits at
L ≤ 4 with every shell `l ≤ 2`, 70 in 17 uncapped; the permutation checked
as an exact index map plus `eri.eri` within `1e-12` of scale, worst 2.6e-16,
not bitwise). Two class modes: **canonical** (the literal, one kernel per
orbit, `hj_{rep}` / `hk_{rep}`, a runtime `perm`) and **ordered** (the
`ordered` lever, legal on `j.hgp` / `k.hgp` only, `hj_o_*` / `hk_o_*`). The
kernels (`eri/hjk/kernel_hj.py`, `eri/hjk/kernel_hk.py`, #404, #405) take §3.2's mapping and
`kernel_f`'s record unchanged, deposit every ordered quartet at weight 1 by
`AtomicAdd` (no weights, no `close`), and are held to `eri/j_coulomb/j.py` / `eri/k_exchange/k.py`
within `1e-12·(s_hgp + s_mmd)` in all 70 L ≤ 4 classes, interpreted, the
ERI tap bitwise against `eri/hgp_drive`. The stream (`eri/hjk/launch_hjk.py`, #403)
prunes nothing and screens on one density block per build, a skip, never a
break; the walker (`eri/hjk/walk_hjk.py`, #406) builds whole `J` and `K`
interpreted (H₂, H₂O/STO-3G, H₂O/6-31G; worst 3.82e-15 against libcint),
and `Engine` gained `j` / `k`. On the RTX 3080, #408: 184 passed (every L ≤ 4
class in both modes, and H₂ and H₂O/STO-3G whole), worst 2.59e-15 against
libcint. **It has not run on the RX 9060 XT**: compile-only there (182
passed); the card run is not made, and no open issue tracks it; #431 closed on the card's detection fix, not the run. **Nothing is
timed** (`measured_store` refuses an HGP `J`/`K` row, `HGP_JK_REFUSED`),
nothing above L = 4 is tested, nothing persists, and
`fock/symmetry-vs-screening` stays unverifiable (#331).
**The 2x2's two diagonal cells are timed against each other, per class**
(`devtools/measure.sh scheme-ab`, `measure/measure_scheme_ab.py`,
`data/measured/scheme_ab.json`; `hgp_fock.md`, "The diagonal cells timed per
class"): fused HGP `F` (`kernel_f`, literal) against MMD `DQ` + `[p\|q]` +
`JAB` + `kernel_kblock`, (ss|ss) and (pp|pp) on gly5/10/15 in Cartesian
6-31G with an RHF density (`python -m oracle.polyglycine`, PySCF, an
untracked `.measure-cache/` pinned per row by sha256 and E_RHF), each class
as the sub-basis of its shells, the card's `F` held to its `J − K/2` first
(4.27e-15 / 1.83e-15 of `max|G|` at gly5, `τ = 0`), the verdict rule
declared in the store before any row. **All 12 cells `fused` on both cards**,
`mmd / f` 1.8–18 and growing with n. The scheme is still confounded with the
recurrence (the HGP split is untimed), so `fock/symmetry-vs-screening`
stays unverifiable.
