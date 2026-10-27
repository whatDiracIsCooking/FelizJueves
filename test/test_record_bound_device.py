"""`Target.RECORD_BOUND` on each card: K's dressed record bound against the
host (#499).

Each ordered class's unit is the `gpu` backend's text (emit/gpu.py),
compiled alone into the harness below with test_kernel_device.py's portable
layer, launched one thread per record over test_kernel_record_bound.py's
records -- every record of H2O/cc-pVDZ and of the H10 chain in cc-pVDZ, and
the synthetic contracted shells of every ordered class `la, lb <= 3` -- and
held to `eri.k_exchange.k.record_bounds` through `reference.pq` at that file's `16 eps`
of the bound, a host zero an exact zero. The device `sqrt` and `exp` are
compiled without fast math (devtools/config.sh), so the card and the
interpreter should agree to the same few ulps. Classes with `la + lb` above
`tiers.FAST_L_MAX` are `slow`, and so is every record of CO/cc-pVTZ, whose
f classes run the low-stack form (#586, stage 3), the default from f up.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_kernel_device import PORTABLE, _toolchain
from test_kernel_record_bound import (
    CLASS_GRID,
    EPS,
    F_MOLECULES,
    MOLECULES,
    REL_TOL,
    molecule,
    synthetic,
    worst,
)

from felizjueves.emit import gpu
from felizjueves.eri.k_exchange.kernel_record_bound import build_record_bound_kernel
from felizjueves.performance.device import SM_86

# One flat float64 input file, `n_rec | zeta_ab | XPA | XPB | s_ab` (the two
# vectors interleaved [n_rec, 3]), the bounds written back raw.
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
    const size_t n = (size_t)n_rec;
    if (8 * n + 1 != in.size()) {{
        std::fprintf(stderr, "size mismatch\n");
        return 2;
    }}
    const double* h = in.data() + 1;
    double* dz = upload(h, n);
    double* dpa = upload(h + n, 3 * n);
    double* dpb = upload(h + 4 * n, 3 * n);
    double* ds = upload(h + 7 * n, n);
    double* db = nullptr;
    if (!dz || !dpa || !dpb || !ds) {{
        std::fprintf(stderr, "upload failed\n");
        return 1;
    }}
    CHECK(devMalloc(&db, n * sizeof(double)));
    const int threads = 32;
    {kernel_name}<<<dim3((n_rec + threads - 1) / threads), dim3(threads)>>>(
        n_rec, dz, dpa, dpb, ds, db);
    CHECK(devGetLastError());
    CHECK(devSynchronize());
    std::vector<double> bound(n);
    CHECK(devMemcpy(bound.data(), db, n * sizeof(double), devMemcpyD2H));
    if (!spill(argv[2], bound)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    return 0;
}}
"""


def _binary(tc, cls, compile_device_source, tmp_path):
    k = build_record_bound_kernel(cls)
    kpath = tmp_path / f"{k.name}.{tc.ext}"
    kpath.write_text(gpu.emit(k, SM_86))
    return compile_device_source(tc, HARNESS.format(
        kernel=kpath, portable=PORTABLE, kernel_name=k.name))


def _launch(binary, args, run_device_binary, tmp_path) -> np.ndarray:
    n = len(args["zeta_ab"])
    src, dst = tmp_path / "in.bin", tmp_path / "out.bin"
    np.concatenate([[float(n)], args["zeta_ab"], args["XPA"].ravel(),
                    args["XPB"].ravel(), args["s_ab"]]
                   ).astype(np.float64).tofile(src)
    run_device_binary(binary, str(src), str(dst))
    got = np.fromfile(dst, dtype=np.float64)
    assert got.shape == (n,)
    return got


def _classes(backend, cls, request, compile_device_source, run_device_binary,
             tmp_path) -> None:
    tc = _toolchain(backend, request)
    ((got_cls, (args, want)),) = synthetic(cls).items()
    assert got_cls == cls
    got = _launch(_binary(tc, cls, compile_device_source, tmp_path), args,
                  run_device_binary, tmp_path)
    w = worst(got, want)
    assert w <= REL_TOL, (cls, w / EPS)


def _molecule(backend, name, request, compile_device_source,
              run_device_binary, tmp_path) -> None:
    tc = _toolchain(backend, request)
    recs = molecule(name)
    w = 0.0
    for cls, (args, want) in sorted(recs.items()):
        d = tmp_path / f"{cls[0]}{cls[1]}"
        d.mkdir()
        got = _launch(_binary(tc, cls, compile_device_source, d), args,
                      run_device_binary, d)
        w = max(w, worst(got, want))
    print(f"\n{backend} {name}: worst {w / EPS:.2f} eps over "
          f"{sum(len(v[1]) for v in recs.values())} records")
    assert w <= REL_TOL, (name, w / EPS)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASS_GRID)
def test_cuda_record_bound_matches_the_host(cls, request, compile_device_source,
                                            run_device_binary,
                                            tmp_path) -> None:
    _classes("cuda", cls, request, compile_device_source, run_device_binary,
             tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASS_GRID)
def test_hip_record_bound_matches_the_host(cls, request, compile_device_source,
                                           run_device_binary,
                                           tmp_path) -> None:
    _classes("hip", cls, request, compile_device_source, run_device_binary,
             tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", MOLECULES)
def test_cuda_record_bound_on_a_molecule(name, request, compile_device_source,
                                         run_device_binary, tmp_path) -> None:
    _molecule("cuda", name, request, compile_device_source, run_device_binary,
              tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", MOLECULES)
def test_hip_record_bound_on_a_molecule(name, request, compile_device_source,
                                        run_device_binary, tmp_path) -> None:
    _molecule("hip", name, request, compile_device_source, run_device_binary,
              tmp_path)


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", F_MOLECULES)
def test_cuda_record_bound_on_an_f_molecule(name, request,
                                            compile_device_source,
                                            run_device_binary,
                                            tmp_path) -> None:
    _molecule("cuda", name, request, compile_device_source, run_device_binary,
              tmp_path)


@pytest.mark.slow
@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", F_MOLECULES)
def test_hip_record_bound_on_an_f_molecule(name, request,
                                           compile_device_source,
                                           run_device_binary,
                                           tmp_path) -> None:
    _molecule("hip", name, request, compile_device_source, run_device_binary,
              tmp_path)
