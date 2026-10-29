"""Schwarz `Q`, the cross overlap `SX`, the DF metric `(P|Q)` and the
three-center `(mn|P)`, with their HGP twins: tree-state's
`references/schwarz-overlap-df.md`."""

from __future__ import annotations

from .rules import MOLECULE, PROGRAM, Allow, ById, Classes, ladder, span

BOTH = ("rtx3080", "rx9060xt")

CASES = [
    *ladder("mmd", "schwarz", 0, compile=(0, 2, 4), gpu=(0, 2, 4),
            cards=dict.fromkeys(BOTH, (0, 2, 4))),
    # The metric through (i|i) and (mn|P) through (ff|i): both cards since #606.
    *ladder("mmd", "metric", 0, compile=range(5), gpu=range(13),
            cards=dict.fromkeys(BOTH, range(13))),
    *ladder("mmd", "eri3", 0, compile=range(5), gpu=range(13),
            cards=dict.fromkeys(BOTH, range(13))),
    # The HGP twins, s and p: the RTX 3080 only.
    *ladder("hgp", "metric", 0, compile=range(3), gpu=range(3),
            cards={"rtx3080": range(3)}),
    *ladder("hgp", "eri3", 0, compile=range(4), gpu=range(4),
            cards={"rtx3080": range(4)}),
]

# Schwarz's L is 2 (la + lb) (tiers.py): through la + lb = 2.
_SCHWARZ = span("mmd", "schwarz", 0, (0, 2, 4))
# The HGP DF kernels are tested for s and p shells only (HGP_DF_L_MAX = 1).
_HGP_METRIC = span("hgp", "metric", 0, range(3))
_HGP_ERI3 = span("hgp", "eri3", 0, range(4))

TESTS = {
    # Schwarz.
    "test/test_schwarz_device.py::test_X_every_class_through_l4_matches_the_host":
        _SCHWARZ,
    "test/test_schwarz_device.py::test_X_schwarz_programs_compile": ById({
        "synthetic": _SCHWARZ, "h2o_sto3g_schwarz": Allow(MOLECULE),
        "h2o_631g_schwarz": Allow(MOLECULE)}),
    "test/test_schwarz_device.py::test_X_schwarz_matrix_matches_libcint":
        Allow(MOLECULE),
    # SX: every case is a pair of bases.
    "test/test_sx_device.py::test_X_cross_overlap_matches_the_host_and_libcint":
        Allow(MOLECULE),
    "test/test_sx_device.py::test_X_sx_programs_compile": Allow(MOLECULE),
    # The DF metric.
    "test/test_emit_metric.py::test_X_compiles_the_metric_class":
        Classes("mmd", "metric"),
    "test/test_metric_device.py::test_X_every_metric_class_through_l4_matches_the_host":
        span("mmd", "metric", 0, range(5)),
    "test/test_metric_device.py::test_X_metric_through_i_i_matches_libcint":
        span("mmd", "metric", 0, range(13)),
    "test/test_metric_device.py::test_X_metric_programs_compile": ById({
        "synthetic": span("mmd", "metric", 0, range(5)),
        "fixture": Allow(PROGRAM)}),
    # (mn|P).
    "test/test_emit_eri3.py::test_X_compiles_the_eri3_class": Classes("mmd", "eri3"),
    "test/test_eri3_device.py::test_X_every_eri3_class_through_l4_matches_the_host":
        span("mmd", "eri3", 0, range(5)),
    "test/test_eri3_device.py::test_X_eri3_through_ff_i_matches_libcint":
        span("mmd", "eri3", 0, range(13)),
    "test/test_eri3_device.py::test_X_eri3_programs_compile": ById({
        "synthetic": span("mmd", "eri3", 0, range(5)),
        "fixture": Allow(PROGRAM)}),
    # The HGP twins.
    "test/test_emit_hgp_df.py::test_X_compiles_the_hgp_df_unit":
        Classes("hgp", "metric"),
    "test/test_hgp_df_device.py::test_X_hgp_metric_matches_the_host_and_libcint":
        ById({"synthetic": _HGP_METRIC, "fixture": Allow(PROGRAM)}),
    "test/test_hgp_df_device.py::test_X_hgp_eri3_matches_the_host_and_libcint":
        ById({"synthetic": _HGP_ERI3, "fixture": Allow(PROGRAM)}),
    "test/test_hgp_df_device.py::test_X_hgp_df_programs_compile":
        ById({"synthetic": _HGP_METRIC + _HGP_ERI3, "fixture": Allow(PROGRAM)}),
}
