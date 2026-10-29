"""The dipole `MU`, the quadrupole, and the one-electron matrices on the HGP
and Rys engines: tree-state's `references/one-electron-extra.md`."""

from __future__ import annotations

from .rules import MOLECULE, Allow, ById, Classes, ladder, span

CASES = [
    # The dipole on both cards through (dd) (PR #546); its (ff) cases (#627)
    # have not run on a card.
    *ladder("mmd", "dipole", 0, compile=range(7), gpu=range(7),
            cards=dict.fromkeys(("rtx3080", "rx9060xt"), range(5))),
    # The quadrupole: not compiled by nvcc or hipcc, not run on a card.
    *ladder("mmd", "quadrupole", 0, compile=range(5), gpu=range(5)),
    # HGP S, T, V and the dipole, s and p: the RTX 3080 only.
    *(case for q in ("s", "t", "v", "dipole") for case in ladder(
        "hgp", q, 0, compile=range(3), gpu=range(3),
        cards={"rtx3080": range(3)})),
    # Rys S, T, V: interpreted and emitted, never on a card.
    *(case for q in "stv" for case in ladder("rys", q, 0, compile=range(7))),
]

# HGP one-electron: s and p shells only (kernel_hgp.L_MAX = 1).
_HGP = {q: span("hgp", t, 0, range(3)) for q, t in (
    ("s", "s"), ("t", "t"), ("v", "v"), ("dipole", "dipole"))}

TESTS = {
    "test/test_emit_mu.py::test_X_compiles_the_mu_class": Classes("mmd", "dipole"),
    "test/test_mu_device.py::test_X_mu_every_class_through_dd_matches_host_and_libcint":
        span("mmd", "dipole", 0, range(5)),
    "test/test_mu_device.py::test_X_mu_every_class_through_ff_matches_host_and_libcint":
        span("mmd", "dipole", 0, range(7)),
    "test/test_mu_device.py::test_X_mu_programs_compile": ById({
        "hf_ccpvtz_spd_dipole": span("mmd", "dipole", 0, range(5)),
        "hf_ccpvtz_dipole": span("mmd", "dipole", 0, range(7)),
        **{name: Allow(MOLECULE) for name in (
            "h2o_sto3g", "h2o_631g", "h2o_ccpvdz")}}),
    "test/test_mu_device.py::test_X_mu_h2o_matches_libcint_and_dip_moment":
        Allow(MOLECULE),
    "test/test_mu_device.py::test_X_mu_origin_shift_against_the_cards_s":
        Allow(MOLECULE),
    "test/test_emit_quad.py::test_X_compiles_the_quad_class":
        Classes("mmd", "quadrupole"),
    "test/test_quad_device.py::test_X_quad_every_class_through_dd_matches_host_and_libcint":
        span("mmd", "quadrupole", 0, range(5)),
    "test/test_quad_device.py::test_X_quad_program_compiles":
        span("mmd", "quadrupole", 0, range(5)),
    "test/test_emit_hgp_one_e.py::test_X_compiles_the_unit": Classes("hgp", "s"),
    "test/test_hgp_one_e_device.py::test_X_hgp_one_e_matches_the_mmd_host": ById(_HGP),
    "test/test_hgp_one_e_device.py::test_X_hgp_one_e_program_compiles": ById(_HGP),
    "test/test_hgp_one_e_device.py::test_X_hgp_one_e_matches_libcint": Allow(MOLECULE),
    "test/test_emit_rys_one_e.py::test_X_compiles_the_unit": Classes("rys", "s"),
}
