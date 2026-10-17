"""test/devcache.py (#186): test-side device compiles through the compiler cache.

FAST (no marker): the launcher's resolution and its off switch (injected
`which`), the content-addressed key -- equal across `tmp_path`s, different
when ANY included unit changes, nested ones too -- and `compile_unit`'s
contract (always invoked, `-c` then a link, the binary copied out) with a
fake compiler.

`cuda` / `hip` (a compiler, no card): one plan's driver compiled from two
fresh `tmp_path`s through a PRIVATE sccache server (its own port and store,
so the counts are this test's alone and the first compile is a guaranteed
miss); the second compile is a hit, read from the server's stats. Then a
warning planted in an emitted unit fails #92's filter on the miss AND on
the hit, whose diagnostics sccache replays, still naming the unit.
"""

from __future__ import annotations

import os
import socket
import stat
import subprocess
from pathlib import Path

import devcache
import pytest

# --- the launcher -----------------------------------------------------------


def test_the_launcher_is_sccache_when_present() -> None:
    got = devcache.launcher(env={}, which=lambda t: f"/x/{t}", venv_paths="")
    assert got == ["/x/sccache"]


def test_no_sccache_is_a_cold_compile_not_an_error() -> None:
    assert devcache.launcher(env={}, which=lambda t: None, venv_paths="") == []


@pytest.mark.parametrize("value", ["1", "yes"])
def test_the_opt_out_wins_over_a_present_sccache(value: str) -> None:
    env = {devcache.NO_CACHE_VAR: value}
    assert devcache.launcher(env=env, which=lambda t: f"/x/{t}",
                             venv_paths="") == []


def test_an_empty_opt_out_is_not_an_opt_out() -> None:
    env = {devcache.NO_CACHE_VAR: ""}
    assert devcache.launcher(env=env, which=lambda t: f"/x/{t}",
                             venv_paths="") == ["/x/sccache"]


def test_a_venv_sccache_comes_before_path(tmp_path: Path) -> None:
    exe = tmp_path / "v" / "bin" / "sccache"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    got = devcache.launcher(env={}, which=lambda t: f"/x/{t}",
                            venv_paths=str(tmp_path / "v"))
    assert got == [str(exe)]


def test_the_switch_is_named_in_config_sh_and_passed_by_the_tier() -> None:
    root = devcache.REPO / "devtools"
    config = (root / "config.sh").read_text()
    assert f"\n{devcache.NO_CACHE_VAR}=${{{devcache.NO_CACHE_VAR}:-}}\n" in config
    tier = (root / "kernel-tier.sh").read_text()
    assert f"no_cache=${devcache.NO_CACHE_VAR}\n" in tier
    assert f"export {devcache.NO_CACHE_VAR}=1" in tier


# --- the content-addressed key ----------------------------------------------


def _tree(root: Path, b: str = "int b;\n") -> Path:
    """main.cu -> "a.h" (relative) -> "sub/b.h" (relative to a.h)."""
    (root / "sub").mkdir(parents=True)
    (root / "a.h").write_text('#include "sub/b.h"\nint a;\n')
    (root / "sub" / "b.h").write_text(b)
    return root


def _key(root: Path, main: str) -> str:
    text, units = devcache.localize(main, root)
    return devcache.unit_key("/usr/bin/nvcc", ["-O3"], "main.cu", text, units)


MAIN = '#include "a.h"\nint main() { return 0; }\n'


def test_the_key_ignores_where_the_caller_wrote(tmp_path: Path) -> None:
    one, two = _tree(tmp_path / "one"), _tree(tmp_path / "two")
    assert _key(one, MAIN) == _key(two, MAIN)
    # an ABSOLUTE include of each tree is rewritten away too
    abs_one = f'#include "{one / "a.h"}"\nint main() {{ return 0; }}\n'
    abs_two = f'#include "{two / "a.h"}"\nint main() {{ return 0; }}\n'
    assert _key(one, abs_one) == _key(two, abs_two) == _key(one, MAIN)
    text, units = devcache.localize(abs_one, None)
    assert str(tmp_path) not in text
    assert all(str(tmp_path).encode() not in u for u in units.values())
    assert set(units) == {"a.h", "b.h"}


def test_the_key_changes_with_any_included_unit(tmp_path: Path) -> None:
    base = _key(_tree(tmp_path / "base"), MAIN)
    assert _key(_tree(tmp_path / "nested", b="int b2;\n"), MAIN) != base
    top = _tree(tmp_path / "top")
    (top / "a.h").write_text('#include "sub/b.h"\nint a2;\n')
    assert _key(top, MAIN) != base
    assert _key(_tree(tmp_path / "main"), MAIN + "\n") != base


def test_the_key_changes_with_the_compiler_and_flags(tmp_path: Path) -> None:
    text, units = devcache.localize(MAIN, _tree(tmp_path))
    k = devcache.unit_key("nvcc", ["-O3"], "main.cu", text, units)
    assert devcache.unit_key("hipcc", ["-O3"], "main.cu", text, units) != k
    assert devcache.unit_key("nvcc", ["-O2"], "main.cu", text, units) != k


def test_two_units_with_one_basename_both_reach_the_key(tmp_path: Path) -> None:
    for d, v in (("x", "1"), ("y", "2")):
        (tmp_path / d).mkdir()
        (tmp_path / d / "k.h").write_text(f"int k{v};\n")
    text, units = devcache.localize('#include "x/k.h"\n#include "y/k.h"\n',
                                    tmp_path)
    assert sorted(units.values()) == [b"int k1;\n", b"int k2;\n"]
    assert len(set(units)) == 2 and all(f'"{n}"' in text for n in units)


def test_a_search_directory_resolves_what_base_does_not(tmp_path: Path) -> None:
    # The gpu backend's `#include "felizjueves/runtime.h"`: not beside the
    # unit, but under a search directory, as a compiler's -I finds it -- and
    # still carried, so the header's bytes reach the key.
    inc = tmp_path / "include"
    (inc / "lib").mkdir(parents=True)
    (inc / "lib" / "r.h").write_text('#include "s.h"\nint r;\n')
    (inc / "lib" / "s.h").write_text("int s;\n")
    main = '#include "lib/r.h"\n'
    with pytest.raises(ValueError, match="no such unit"):
        devcache.localize(main, tmp_path)
    text, units = devcache.localize(main, tmp_path, (inc,))
    assert text == '#include "r.h"\n'
    assert units == {"r.h": b'#include "s.h"\nint r;\n', "s.h": b"int s;\n"}
    # base wins over a search directory, as the compiler's quoted lookup does
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "r.h").write_text("int local;\n")
    _, units = devcache.localize(main, tmp_path, (inc,))
    assert units == {"r.h": b"int local;\n"}


def test_an_include_that_does_not_resolve_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no such unit"):
        devcache.localize('#include "missing.h"\n', tmp_path)


def test_the_plan_drivers_key_ignores_tmp_path(tmp_path: Path) -> None:
    from plan_driver import driver
    from test_j_launched_ondevice import _plan

    plan = _plan("cuda", "h2_sto3g_j", 0.0, 1)
    keys = []
    for d in ("a", "b"):
        drv = driver(plan, "cuda", tmp_path / d / "gen")
        text, units = devcache.localize(drv.path.read_text(), drv.path.parent)
        assert set(units) == {u.name for u in drv.units}
        keys.append(devcache.unit_key("nvcc", [], drv.path.name, text, units))
    assert keys[0] == keys[1]
    unit = drv.units[-1]
    unit.write_text(unit.read_text() + "\n// changed\n")
    text, units = devcache.localize(drv.path.read_text(), drv.path.parent)
    assert devcache.unit_key("nvcc", [], drv.path.name, text, units) != keys[0]


# --- compile_unit, with a fake compiler -------------------------------------

FAKE = """#!/bin/sh
# a fake compiler: log the argv, copy the input to -o, warn on stderr
echo "$*" >> "{log}"
out=; prev=; for a in "$@"; do [ "$prev" = -o ] && out=$a; prev=$a; done
echo "fake: warning: replayed" >&2
printf 'bin\\n' > "$out"
"""


def test_compile_unit_always_invokes_and_copies_out(tmp_path: Path,
                                                    monkeypatch) -> None:
    monkeypatch.setattr(devcache, "build_root", lambda: tmp_path / "cache")
    log = tmp_path / "log"
    cc = tmp_path / "fakecc"
    cc.write_text(FAKE.format(log=log))
    cc.chmod(cc.stat().st_mode | stat.S_IXUSR)
    env = {**os.environ, devcache.NO_CACHE_VAR: "1"}
    outs = []
    for i in range(2):
        out = tmp_path / f"t{i}" / "bin"
        got = devcache.compile_unit(str(cc), ["-O3", "-x", "hip"], "main.cu",
                                    "int main;\n", {"a.h": b"int a;\n"}, out,
                                    env=env)
        assert got.returncode == 0 and out.read_text() == "bin\n"
        assert got.diagnostics.count("fake: warning: replayed") == 2
        outs.append(got.workdir)
    assert outs[0] == outs[1]  # one content-addressed directory
    calls = log.read_text().splitlines()
    assert len(calls) == 4  # compiled AND linked, both times
    assert all(" -c main.cu " in c for c in calls[0::2])
    assert all("-x" not in c and ".o -o " in c for c in calls[1::2])
    assert sorted(p.name for p in outs[0].iterdir()) == ["a.h", "main.cu"]


# --- the cache, hit for real (a compiler, no card) ---------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cache_hits_and_replays(backend: str, tc, tmp_path: Path) -> None:
    from plan_driver import compile_checked, driver
    from test_j_launched_ondevice import _plan

    env = {k: v for k, v in os.environ.items()
           if k != devcache.NO_CACHE_VAR and not k.startswith("SCCACHE_")}
    env |= {"SCCACHE_DIR": str(tmp_path / "store"),
            "SCCACHE_SERVER_PORT": str(_free_port())}
    launch = devcache.launcher(env)
    if not launch:
        pytest.skip("no sccache: test-side compiles are cold here")
    subprocess.run([*launch, "--start-server"], env=env, check=True,
                   capture_output=True, timeout=60)
    try:
        plan = _plan(backend, "h2_sto3g_j", 0.0, 1)
        seen = [devcache.stats(env)]
        assert seen[0] == {"hits": 0, "misses": 0}, seen

        def step(name: str, planted: bool) -> AssertionError | None:
            drv = driver(plan, backend, tmp_path / name / "gen")
            if planted:
                unit = drv.units[0]
                unit.write_text(unit.read_text() + "\n__device__ int "
                                "fj_planted() { int fj_unused; return 0; }\n")
            err = None
            try:
                compile_checked(tc, drv, tmp_path / name / "driver", env=env)
            except AssertionError as e:
                err = e
            seen.append(devcache.stats(env))
            return err

        assert step("cold", False) is None
        assert seen[1]["misses"] > 0 and seen[1]["hits"] == 0, seen
        assert step("warm", False) is None
        assert seen[2]["hits"] > 0 and seen[2]["misses"] == seen[1]["misses"], (
            f"{backend}: the second compile of one driver did not hit: {seen}")
        for name, hit in (("planted-miss", False), ("planted-hit", True)):
            err = step(name, True)
            before, after = seen[-2], seen[-1]
            if hit:
                assert after["hits"] > before["hits"] and (
                    after["misses"] == before["misses"]), seen
            else:
                assert after["misses"] > before["misses"], seen
            assert err is not None, f"{name}: a planted warning passed #92"
            assert "warning(s) in generated text" in str(err), str(err)
            assert driver(plan, backend, tmp_path / "x" / "gen").units[0].name \
                in str(err)
    finally:
        subprocess.run([*launch, "--stop-server"], env=env,
                       capture_output=True, timeout=60)


@pytest.mark.cuda
def test_cuda_the_second_compile_is_a_hit_and_replays_warnings(
        nvcc, tmp_path: Path) -> None:
    _cache_hits_and_replays("cuda", nvcc, tmp_path)


@pytest.mark.hip
def test_hip_the_second_compile_is_a_hit_and_replays_warnings(
        hipcc, tmp_path: Path) -> None:
    _cache_hits_and_replays("hip", hipcc, tmp_path)
