"""The `J` chain and the one-electron `S`, `T`, `V`: tree-state's
`references/j-chain.md`."""

from __future__ import annotations

from .rules import MACHINERY, MOLECULE, PROGRAM, Allow, ById, Classes, ladder, span

BOTH = ("rtx3080", "rx9060xt")

CASES = [
    # [p|q]: the last full kernel tier on both cards was #157's, every L.
    *ladder("mmd", "j", 0, compile=(0, 1, 4, 8), gpu=range(17),
            cards=dict.fromkeys(BOTH, range(17))),
    # DQ and JAB: every pair class through (ff) on both cards (#157, #586).
    *ladder("mmd", "dq", 0, compile=range(7), gpu=range(7),
            cards=dict.fromkeys(BOTH, range(7))),
    *ladder("mmd", "jab", 0, compile=range(7), gpu=range(7),
            cards=dict.fromkeys(BOTH, range(7))),
    # PAIR_BOUND on the RTX 3080 only; RECORD_BOUND on the RX 9060 XT only
    # (the RTX 3080 was down in that session).
    *ladder("mmd", "pair_bound", 0, gpu=range(9), cards={"rtx3080": range(9)}),
    *ladder("mmd", "record_bound", 0, gpu=range(7),
            cards={"rx9060xt": range(7)}),
    # S, T, V: every pair class through (ff) on both cards (milestone 11).
    *(case for q in "stv" for case in ladder(
        "mmd", q, 0, compile=range(5), gpu=range(7),
        cards=dict.fromkeys(BOTH, range(7)))),
]

_ONE_E_DD = {q.upper(): span("mmd", q, 0, range(5)) for q in "stv"}
_ONE_E_FF = {q.upper(): span("mmd", q, 0, range(7)) for q in "stv"}

TESTS = {
    # [p|q], one batch and split.
    "test/test_emit_X.py::test_it_compiles": Classes("mmd", "j"),
    "test/test_any_wavefront_device.py::test_X_any_wavefront_j_p_matches_the_reference":
        Classes("mmd", "j"),
    "test/test_kernel_device.py::test_X_j_p_matches_the_reference": Classes("mmd", "j"),
    "test/test_kernel_device.py::test_X_batched_j_p_matches_the_reference":
        Classes("mmd", "j"),
    "test/test_kernel_device.py::test_X_recycled_j_p_matches_the_reference":
        Classes("mmd", "j"),
    "test/test_kernel_device.py::test_X_unsieved_j_p_matches_the_reference":
        Classes("mmd", "j"),
    "test/test_kernel_device.py::test_X_tuned_j_p_matches_the_reference":
        Classes("mmd", "j"),
    "test/test_kernel_device.py::test_X_asym_j_p_matches_the_reference":
        Classes("mmd", "j"),
    "test/test_kernel_fused_device.py::test_X_fused_j_p_matches_the_reference":
        Classes("mmd", "j"),
    "test/test_kernel_mixed_device.py::test_X_mixed_j_p_meets_the_fp32_tier":
        Classes("mmd", "j"),
    # DQ and JAB: one pair program runs both.
    "test/test_emit_X.py::test_the_pair_goldens_compile": Classes("mmd", "dq"),
    "test/test_j_chain_ondevice.py::test_X_pair_kernels_match_the_host":
        Classes("mmd", "dq", also=("jab",)),
    "test/test_j_chain_ondevice.py::test_X_pair_programs_compile":
        Classes("mmd", "dq", also=("jab",)),
    # The bounds.
    "test/test_pair_bound_device.py::test_X_pair_bound_matches_the_host":
        Classes("mmd", "pair_bound"),
    "test/test_record_bound_device.py::test_X_record_bound_matches_the_host":
        Classes("mmd", "record_bound"),
    "test/test_record_bound_device.py::test_X_record_bound_on_a_molecule":
        Allow(MOLECULE),
    "test/test_record_bound_device.py::test_X_record_bound_on_an_f_molecule":
        Allow(MOLECULE),
    # S, T, V.
    "test/test_emit_X.py::test_the_one_e_units_compile": Classes("mmd", "s"),
    "test/test_one_electron_device.py::test_X_every_class_through_dd_matches_the_host":
        ById(_ONE_E_DD),
    "test/test_one_electron_device.py::test_X_every_class_through_ff_matches_the_host":
        ById(_ONE_E_FF),
    "test/test_one_electron_device.py::test_X_one_electron_programs_compile":
        Allow(PROGRAM),
    "test/test_one_electron_device.py::test_X_one_electron_matrices_match_libcint":
        Allow(MOLECULE),
    "test/test_hcore_device.py::test_X_hcore_and_its_trace_on_the_card":
        Allow(MOLECULE),
    # Whole molecules' J.
    "test/test_ao_j_ondevice.py::test_X_j_matches_libcint": Allow(MOLECULE),
    "test/test_j_chain_ondevice.py::test_X_j_through_the_chain": Allow(MOLECULE),
    "test/test_j_chain_ondevice.py::test_X_chain_compiles": Allow(MOLECULE),
    "test/test_j_launched_ondevice.py::test_X_j_from_the_launch_plan": Allow(MOLECULE),
    "test/test_j_launched_ondevice.py::test_X_driver_compiles": Allow(MOLECULE),
    "test/test_j_launched_ondevice.py::test_X_the_chain_under_both_keys":
        Allow(MOLECULE),
    # The gpu backend's link and wavefront checks (#549 and after).
    **{f"test/test_emit_gpu.py::{name}": Allow(MACHINERY) for name in (
        "test_one_text_compiles_under_both",
        "test_every_class_of_two_families_links_as_its_own_unit",
        "test_a_class_not_compiled_in_fails_to_link",
        "test_every_class_of_two_families_still_links_in_one_tu",
        "test_a_unit_whose_parameters_drifted_fails_to_compile",
        "test_one_any_wavefront_text_compiles_at_either_width",
        "test_every_pq_class_links_as_its_own_unit_under_a_fixed_block",
        "test_two_units_with_eval_boys_share_one_tu",
        "test_an_order_past_the_headers_is_a_compile_error",
    )},
}
