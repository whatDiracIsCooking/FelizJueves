"""`derivative_order = 1`: tree-state's `references/gradients.md`."""

from __future__ import annotations

from .rules import PROGRAM, Allow, ById, Classes, ladder, span

BOTH = ("rtx3080", "rx9060xt")

CASES = [
    # gs, gt, gv and the gjab chain through (ff) on both cards (#614, #619).
    *(case for q in ("s", "t", "v", "jab") for case in ladder(
        "mmd", q, 1, compile=range(5), gpu=range(7),
        cards=dict.fromkeys(BOTH, range(7)))),
    # hgs, S on HGP (kernel_hgs): compiled through (ff), not run on a card.
    *ladder("hgp", "s", 1, compile=range(7), gpu=range(7)),
    # gk: H2O/STO-3G and CO/cc-pVTZ's 256 classes on both cards (#618).
    *ladder("mmd", "k", 1, compile=range(5), gpu=range(13),
            cards=dict.fromkeys(BOTH, range(13))),
    # ghk, K at d1 on HGP: interpreted through (dd|dd), compiled at the s / p
    # classes and the d classes test_emit_grad_hk.py names; no card yet.
    *ladder("hgp", "k", 1, compile=(0, 1, 2, 3, 4, 6),
            interpreted=range(9)),
    # The DF-J terms through (ff|i) and (i|i) on both cards (#613).
    *ladder("mmd", "metric", 1, compile=range(5), gpu=range(13),
            cards=dict.fromkeys(BOTH, range(13))),
    *ladder("mmd", "eri3", 1, compile=range(5), gpu=range(13),
            cards=dict.fromkeys(BOTH, range(13))),
    # The HGP kinetic gradient, hgt_ (ss) through (ff): interpreted and
    # compiled by nvcc and hipcc, never run on a card.
    *ladder("hgp", "t", 1, compile=range(7)),
    # The HGP V gradient, ghv_ (ss) through (ff): interpreted and compiled by
    # nvcc and hipcc; its device tests (s and p) written, never run on a card.
    *ladder("hgp", "v", 1, compile=range(7), gpu=range(3)),
    # The HGP (mn|P) gradient: interpreted through L = 4, its s / p ladder
    # compiled; no card.
    *ladder("hgp", "eri3", 1, compile=range(4), interpreted=range(5)),
]

# #613's ladder: one program per aux l, every class of the aux-i system
# whose aux shell is l -- the metric's lP <= lQ = l, (mn|P)'s la <= lb <= 3
# with lP = l.
_LADDER = {
    **{f"metric-{aux}": span("mmd", "metric", 1, range(aux, 2 * aux + 1))
       for aux in range(7)},
    **{f"eri3-{aux}": span("mmd", "eri3", 1, range(aux, aux + 7))
       for aux in range(7)},
}

# A molecule's pair classes: every (la, lb) up to its basis's top shell, so
# L = la + lb from 0 to twice it -- H2 STO-3G s, H2O STO-3G p, H2O cc-pVDZ
# d, CO cc-pVTZ f.
_TOP_PAIR_L = {"h2_sto3g": 0, "h2o_sto3g": 2, "h2o_ccpvdz": 4,
               "co_ccpvtz_proton": 6, "co_ccpvtz": 6}
# A `which` -> (engine, quantity): `hs` is S on HGP (kernel_hgs).
_WHICH = {"s": ("mmd", "s"), "t": ("mmd", "t"), "v": ("mmd", "v"),
          "j": ("mmd", "jab"), "hs": ("hgp", "s")}
_GRAD = {
    f"{name}-{which}": span(*_WHICH[which], 1, range(_TOP_PAIR_L[name] + 1))
    for name, whiches in (("h2o_sto3g", ("s", "t", "v", "j", "hs")),
                          ("h2o_ccpvdz", ("s", "t", "v", "j", "hs")),
                          ("co_ccpvtz_proton", ("s", "t", "v", "hs")))
    for which in whiches
}
# The on-card gjab chain (#619): DQ -> the raised [p|q] -> gjab, per bra pair.
_GJAB = {f"{name}_j-{form}": span("mmd", "jab", 1, range(_TOP_PAIR_L[name] + 1))
         for name, forms in (("h2_sto3g", ("default", "lowstack")),
                             ("h2o_sto3g", ("default", "lowstack")),
                             ("co_ccpvtz", ("default",)))
         for form in forms}

# The paths whose change touches this family (sample.py's change weighting):
# its source, its drivers; its test files come from TESTS.
PATHS = (
    "src/felizjueves/gradient/",
    "test/grad_driver.py",
    "test/df_grad_launch.py",
    "test/gk_driver.py",
    "test/ghv_driver.py",
)

TESTS = {
    "test/test_emit_grad.py::test_X_compiles_the_gradient_unit": Classes("mmd", "s", 1),
    "test/test_emit_hgs.py::test_X_compiles_the_unit": Classes("hgp", "s", 1),
    "test/test_emit_grad_df.py::test_X_compiles_the_df_gradient_unit":
        Classes("mmd", "metric", 1),
    "test/test_emit_grad_k.py::test_X_compiles_the_exchange_gradient_unit":
        Classes("mmd", "k", 1),
    "test/test_emit_hgp_grad.py::test_X_compiles_the_unit": Classes("hgp", "t", 1),
    "test/test_emit_ghv.py::test_X_compiles_the_unit": Classes("hgp", "v", 1),
    # The HGP V gradient on the card: the synthetic basis and H2O reach
    # (ss) (sp) (pp).
    "test/test_ghv_device.py::test_X_ghv_matches_the_mmd_host":
        span("hgp", "v", 1, range(3)),
    "test/test_ghv_device.py::test_X_ghv_matches_pyscf":
        span("hgp", "v", 1, range(3)),
    "test/test_ghv_device.py::test_X_ghv_program_compiles":
        span("hgp", "v", 1, range(3)),
    "test/test_emit_grad_hk.py::test_X_compiles_the_hgp_exchange_gradient_unit":
        Classes("hgp", "k", 1),
    "test/test_emit_hgp_grad_df.py::test_X_compiles_the_hgp_gradient_unit":
        Classes("hgp", "eri3", 1),
    "test/test_grad_device.py::test_X_gradient_matches_the_host": ById(_GRAD),
    # #659's compile-only twins, the registry's compile tier (L <= 4)
    "test/test_grad_device.py::test_X_gradient_runners_compile": ById(
        {k: v for k, v in _GRAD.items() if k.startswith("h2o_")}),
    "test/test_grad_df_device.py::test_X_df_gradient_matches_the_host":
        Allow(PROGRAM),
    "test/test_grad_df_device.py::test_X_df_gradient_ladder_through_i": ById(_LADDER),
    "test/test_gjab_chain_ondevice.py::test_X_gjab_chain_matches_the_host":
        ById(_GJAB),
    # gk's classes on H2O/STO-3G (s and p, every quartet below L = 5) and
    # all 256 of CO/cc-pVTZ (#618).
    "test/test_gk_device.py::test_X_gk_h2o_sto3g_both_forms":
        span("mmd", "k", 1, range(5)),
    "test/test_gk_device.py::test_X_gk_h2o_sto3g_compiles":
        span("mmd", "k", 1, range(5)),
    "test/test_gk_device.py::test_X_gk_co_ccpvtz_against_libcint":
        span("mmd", "k", 1, range(13)),
}
