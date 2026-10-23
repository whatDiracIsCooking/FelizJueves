"""`Target.PAIR_BOUND` on each card: the record bound against the host.

Each class's unit is the `gpu` backend's text (emit/gpu.py), compiled alone
into the harness below with test_kernel_device.py's portable layer, launched
one thread per record over test_kernel_pair_bound.py's exponents, and held to
`ao.pairs.pair_bound` through `reference.pq` at that file's `16 eps` of the
bound. The device `sqrt` is IEEE's correctly rounded one under both
compilers (no fast math, devtools/config.sh), so the card and the
interpreter should agree to the same few ulps. Classes with `la + lb` above
`tiers.FAST_L_MAX` are `slow`.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_kernel_device import PORTABLE, _toolchain
from test_kernel_pair_bound import CLASS_GRID, REL_TOL, ZETAS

from felizjueves.ao.pairs import pair_bound
from felizjueves.emit import gpu
from felizjueves.kernel_pair_bound import build_pair_bound_kernel
from felizjueves.performance.device import SM_86
from felizjueves.reference.pq import one_centre

# One flat float64 input file, `n_rec | zeta`, the bounds written back raw.
HARNESS = r"""
#include "{kernel}"
{portable}
int main(int argc, char** argv) {{
    if (argc != 3) {{ std::fprintf(stderr, "usage: in out\n"); return 2; }}
    std::vector<double> in;
    if (!slurp(argv[1], in) || in.size() < 2) {{
        std::fprintf(stderr, "bad input\n");
        return 2;
    }}
    const int n_rec = (int)in[0];
    if ((size_t)n_rec + 1 != in.size()) {{
        std::fprintf(stderr, "size mismatch\n");
        return 2;
    }}
    double* dz = upload(in.data() + 1, (size_t)n_rec);
    double* db = nullptr;
    if (!dz) {{ std::fprintf(stderr, "upload failed\n"); return 1; }}
    CHECK(devMalloc(&db, (size_t)n_rec * sizeof(double)));
    const int threads = 32;
    {kernel_name}<<<dim3((n_rec + threads - 1) / threads), dim3(threads)>>>(
        n_rec, dz, db);
    CHECK(devGetLastError());
    CHECK(devSynchronize());
    std::vector<double> bound((size_t)n_rec);
    CHECK(devMemcpy(bound.data(), db, bound.size() * sizeof(double), devMemcpyD2H));
    if (!spill(argv[2], bound)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    return 0;
}}
"""


def _run(backend: str, cls: tuple[int, int], request, compile_device_source,
         run_device_binary, tmp_path) -> None:
    tc = _toolchain(backend, request)
    k = build_pair_bound_kernel(cls)
    kpath = tmp_path / f"{k.name}.{tc.ext}"
    kpath.write_text(gpu.emit(k, SM_86))
    binary = compile_device_source(tc, HARNESS.format(
        kernel=kpath, portable=PORTABLE, kernel_name=k.name))
    src, dst = tmp_path / "in.bin", tmp_path / "out.bin"
    np.concatenate([[float(len(ZETAS))], ZETAS]).astype(np.float64).tofile(src)
    run_device_binary(binary, str(src), str(dst))
    got = np.fromfile(dst, dtype=np.float64)
    L = sum(cls)
    want = np.array([pair_bound(float(z), L, one_centre=one_centre)
                     for z in ZETAS])
    assert got.shape == want.shape
    rel = np.abs(got - want) / want
    assert float(rel.max()) <= REL_TOL, (cls, float(rel.max()))


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASS_GRID)
def test_cuda_pair_bound_matches_the_host(cls, request, compile_device_source,
                                          run_device_binary, tmp_path) -> None:
    _run("cuda", cls, request, compile_device_source, run_device_binary,
         tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASS_GRID)
def test_hip_pair_bound_matches_the_host(cls, request, compile_device_source,
                                         run_device_binary, tmp_path) -> None:
    _run("hip", cls, request, compile_device_source, run_device_binary,
         tmp_path)
