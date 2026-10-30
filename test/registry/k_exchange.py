"""Exchange `K` and its block kernels: tree-state's
`references/k-exchange.md`."""

from __future__ import annotations

from .rules import MOLECULE, Allow, Classes, ladder, span

CASES = [
    # On both cards only for L <= 2 (#206); (dd|dd) at L = 8 on the RTX 3080
    # only (#259); every other K device case is written and not yet run (#231).
    *ladder("mmd", "k", 0, compile=range(1, 5),
            gpu=(0, 1, 2, 3, 4, 7, 8, 10, 13, 16),
            cards={"rtx3080": (0, 1, 2, 8), "rx9060xt": (0, 1, 2)}),
]

_LEVERS = ("coalesced", "interleave", "mixed", "shuffle", "split",
           "split_interleave", "stage_eac")
_COMPILES = {"coalesced": "coalesced_kernels", "interleave": "interleave_kernels",
             "mixed": "k_mixed_kernels", "shuffle": "shuffle_kernels",
             "split": "split_kernels",
             "split_interleave": "split_interleave_arms",
             "stage_eac": "stage_eac_kernels"}

# The paths whose change touches this family (sample.py's change weighting):
# its source, its drivers; its test files come from TESTS.
PATHS = (
    "src/felizjueves/eri/k_exchange/",
    "src/felizjueves/eri/launch_common.py",
    "test/kblock_driver.py",
)

TESTS = {
    "test/test_kernel_device.py::test_X_k_cell_matches_the_host": Classes("mmd", "k"),
    "test/test_kernel_device.py::test_X_k_check_cell_matches_the_host":
        Classes("mmd", "k"),
    **{f"test/test_k_lever_{lever}_device.py::test_X_k_lever_{lever}_matches_the_host":
       Classes("mmd", "k") for lever in _LEVERS},
    **{f"test/test_k_lever_{lever}_device.py::test_X_{name}_compile":
       Classes("mmd", "k") for lever, name in _COMPILES.items()},
    "test/test_k_ondevice.py::test_X_k_from_the_launch_plan": Allow(MOLECULE),
    "test/test_k_ondevice.py::test_X_k_driver_compiles": Allow(MOLECULE),
    "test/test_kblock_ondevice.py::test_X_kblock_from_the_launch_plan":
        Allow(MOLECULE),
    # The split `kbs` on (dd|dd), by granularity (#548).
    "test/test_kblock_split_ondevice.py::test_X_kblock_split_dddd":
        span("mmd", "k", 0, (8,)),
}
