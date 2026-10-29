"""The HGP engine, the fused `F` and the split HGP `J` / `K`: tree-state's
`references/fock-hgp.md`."""

from __future__ import annotations

from .rules import MOLECULE, Allow, Classes, ladder

CASES = [
    # The fused F: all 21 classes through (dd|dd) on both cards.
    *ladder("hgp", "f", 0, compile=range(9), gpu=range(9),
            cards=dict.fromkeys(("rtx3080", "rx9060xt"), range(9))),
    # The split HGP J / K: every L <= 4 class on the RTX 3080 only (#408);
    # compile-only on the RX 9060 XT.
    *ladder("hgp", "j", 0, compile=range(5), gpu=range(5),
            cards={"rtx3080": range(5)}),
    *ladder("hgp", "k", 0, compile=range(5), gpu=range(5),
            cards={"rtx3080": range(5)}),
]

_F = Classes("hgp", "f", otherwise=Allow(MOLECULE))
_LEVERS = ("batched_vrr", "etr", "mixed", "sliced_hrr", "sliced_site_digest",
           "slim_fock", "struct_reads", "sub_slice", "subdag_vrr")

TESTS = {
    "test/test_emit_fock.py::test_X_compiles_the_fock_class": _F,
    "test/test_fock_ondevice.py::test_X_each_class_matches_the_host": _F,
    "test/test_fock_ondevice.py::test_X_fock_programs_compile": _F,
    "test/test_fock_ondevice.py::test_X_whole_molecule_matches_walk_and_libcint":
        Allow(MOLECULE),
    **{f"test/test_f_lever_{lever}_device.py::test_X_f_lever_{lever}_{what}": _F
       for lever in _LEVERS for what in (
           "compiles", "programs_compile", "each_class_matches_the_host")},
    **{f"test/test_f_lever_{lever}_device.py::test_X_f_lever_{lever}_whole_molecule_matches":
       Allow(MOLECULE) for lever in _LEVERS},
    "test/test_f_lever_block_fab_device.py::test_X_block_fab_kernels_compile": _F,
    "test/test_f_lever_block_fab_device.py::test_X_f_lever_block_fab_matches_the_host":
        _F,
    "test/test_f_lever_block_fab_device.py::test_X_f_lever_block_fab_whole_molecule_matches":
        Allow(MOLECULE),
    "test/test_f_lever_scatter_device.py::test_X_scatter_kernels_compile": _F,
    "test/test_f_lever_scatter_device.py::test_X_scatter_programs_compile": _F,
    "test/test_f_lever_scatter_device.py::test_X_f_lever_scatter_class_matches_the_host":
        _F,
    "test/test_f_lever_scatter_device.py::test_X_f_lever_scatter_molecule_matches_walk_and_libcint":
        Allow(MOLECULE),
    "test/test_f_lever_ket_break_device.py::test_X_ket_break_programs_compile": _F,
    "test/test_f_lever_ket_break_device.py::test_X_ket_break_matches_the_interpreted_walks":
        Allow(MOLECULE),
    # The split HGP J / K (milestone 14).
    "test/test_emit_hjk.py::test_X_compiles_the_hjk_unit": Classes("hgp", "j"),
    "test/test_hjk_ondevice.py::test_X_hjk_each_class_matches_the_host":
        Classes("hgp", "j"),
    "test/test_hjk_ondevice.py::test_X_hjk_programs_compile":
        Classes("hgp", "j", otherwise=Allow(MOLECULE)),
    "test/test_hjk_ondevice.py::test_X_hjk_whole_molecule_matches_walk_host_and_libcint":
        Allow(MOLECULE),
}
