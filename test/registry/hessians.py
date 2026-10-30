"""`derivative_order = 2`: tree-state's `references/hessians.md`."""

from __future__ import annotations

from .rules import ById, Classes, ladder, span

BOTH = ("rtx3080", "rx9060xt")

CASES = [
    # hdq / hjab on both cards through (dd).
    *(case for q in ("dq", "jab") for case in ladder(
        "mmd", q, 2, compile=range(3), gpu=range(5),
        cards=dict.fromkeys(BOTH, range(5)))),
    # d2s / d2t / d2v, unrolled, on both cards through (dd); the low-stack
    # d2v at (ff) compiled only.
    *(case for q in "st" for case in ladder(
        "mmd", q, 2, compile=range(5), gpu=range(5),
        cards=dict.fromkeys(BOTH, range(5)))),
    *ladder("mmd", "v", 2, compile=(0, 1, 2, 3, 4, 6), gpu=range(5),
            cards=dict.fromkeys(BOTH, range(5))),
    # hkx, d2metric, d2eri3: no device test, not run on either card.
    *ladder("mmd", "k", 2, compile=range(5)),
    *ladder("mmd", "metric", 2, compile=range(5)),
    *ladder("mmd", "eri3", 2, compile=range(5)),
    # kx1 / kxc (#689), the exchange derivative matrix at d1: interpreted
    # through (dd|dd) and at three f classes, compiled at the goldens and
    # (ff|ff); no device test, not run on either card.
    *(case for q in ("kx1", "kxc") for case in ladder(
        "mmd", q, 1, compile=(0, 1, 2, 3, 4, 12),
        interpreted=(*range(9), 12))),
]

# H2O STO-3G's pair classes reach L = 2, cc-pVDZ's L = 4.
_HESS = {f"{name}-{which}": span("mmd", which, 2, range(top + 1))
         for name, top in (("h2o_sto3g", 2), ("h2o_ccpvdz", 4))
         for which in "stv"}

# The d1 derivative matrices (#687, kernel_ip1e): a variant of the d1 S, T,
# V cases (gradients.py's), as a lever is; the rinv form at each nucleus
# is V's.
_IP1E = {f"{name}-{which}": span("mmd", which[0] if which[0] in "st" else
                                 "v", 1, range(top + 1))
         for name, top in (("h2o_sto3g", 2), ("h2o_ccpvdz", 4))
         for which in ("s", "t", "v", "rinv0", "rinv1", "rinv2")}

# The paths whose change touches this family (sample.py's change weighting):
# its source, its drivers; its test files come from TESTS.
PATHS = (
    "src/felizjueves/hessian/",
    "test/hess_driver.py",
    "test/ip1e_driver.py",
)

TESTS = {
    "test/test_emit_hess.py::test_X_compiles_the_hessian_unit": Classes("mmd", "dq", 2),
    "test/test_emit_hess_1e.py::test_X_compiles_the_hessian_unit":
        Classes("mmd", "s", 2),
    "test/test_emit_hess_1e.py::test_X_compiles_the_unrolled_unit":
        Classes("mmd", "v", 2),
    "test/test_emit_hess_k.py::test_X_compiles_the_exchange_hessian_unit":
        Classes("mmd", "k", 2),
    "test/test_emit_hess_kx.py::test_X_compiles_the_derivative_matrix_unit":
        Classes("mmd", "kx1", 1),
    "test/test_emit_d2metric.py::test_X_compiles_the_metric_hessian_unit":
        Classes("mmd", "metric", 2),
    "test/test_emit_d2eri3.py::test_X_compiles_the_df_hessian_unit":
        Classes("mmd", "eri3", 2),
    "test/test_emit_ip1e.py::test_X_compiles_the_ip1_unit":
        Classes("mmd", "s", 1),
    "test/test_ip1e_device.py::test_X_ip1e_matches_the_host": ById(_IP1E),
    "test/test_ip1e_device.py::test_X_ip1e_runners_compile": ById(_IP1E),
    "test/test_hess_device.py::test_X_hessian_matches_the_host": ById(_HESS),
    "test/test_hess_device.py::test_X_hessian_runners_compile": ById(_HESS),
    "test/test_hess_j_device.py::test_X_hdq_matches_the_host": Classes("mmd", "dq", 2),
    "test/test_hess_j_device.py::test_X_hjab_class_matches_the_host":
        Classes("mmd", "jab", 2),
}
