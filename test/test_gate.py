"""The PR gate (#656, devtools/gate.sh): its selections hand the L cut to the
registry sampler through `l_cut` (test/tiers.py), it refuses to run rather
than pass without both device compilers, and its draw renders as the PR
body's section (`sample.markdown_lines`).

None of these runs the gate itself -- that would run the fast tier inside the
fast tier. The preflight's refusal is cheap (it exits before lint), and the
markers are read from a child pytest's collection of two files."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from registry import sample
from test_sample import BUDGET, CANDS

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "devtools" / "gate.sh"


def _bash(script: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "-c", script], cwd=ROOT, text=True,
                          capture_output=True, env={**os.environ, **env})


def _collected(markexpr: str, *files: str) -> list[str]:
    child = {k: v for k, v in os.environ.items()
             if not k.startswith(("PYTEST_", "FJ_"))}
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "-p", "no:xdist", "-o", "addopts=",
         "-m", markexpr, *files],
        cwd=ROOT, env=child, capture_output=True, text=True)
    assert done.returncode in (0, 5), done.stdout[-3000:] + done.stderr[-3000:]
    return [line for line in done.stdout.splitlines() if line.startswith("test/")]


def test_l_cut_marks_only_what_a_cut_made_slow() -> None:
    """`by_l` and `sample_compiles` mark `l_cut` beside `slow`; a function
    decorated `slow` keeps `slow` and loses `l_cut`, so the gate's
    `not slow or l_cut` leaves it out."""
    rj, kb = "test/test_emit_rj.py", "test/test_kblock_split_ondevice.py"
    cut = _collected("l_cut", rj, kb)
    slow = _collected("slow", rj, kb)
    assert cut and set(cut) <= set(slow)
    assert not [n for n in cut if n.startswith(kb)], "a decorated slow kept l_cut"
    assert [n for n in slow if n.startswith(kb)]
    # above the fast cut (by_l) and the compile sample's repeats, both cut
    assert any("[rj_2121-" in n for n in cut)
    nvcc = [n for n in _collected("cuda and not gpu", rj) if "nvcc" in n]
    assert set(nvcc) - set(cut), "the compile sample's first case was cut"
    assert set(nvcc) & set(cut)


def test_the_gate_selections_are_the_cuts_handed_to_the_sampler() -> None:
    done = _bash('. devtools/config.sh; printf "%s\\n" "$GATE_COMPILE_TEST_ARGS"'
                 ' "$GATE_GPU_TEST_ARGS"')
    compile_args, gpu_args = done.stdout.splitlines()
    for args in (compile_args, gpu_args):
        assert "(not slow or l_cut)" in args and "not high_l" in args
    assert "not gpu" in compile_args and "{backend}" in gpu_args
    text = GATE.read_text()
    assert text.count("--fj-sample") >= 2
    assert "KERNEL_TEST_ARGS=$GATE_GPU_TEST_ARGS" in text
    assert "card_present" in text and "card_absent" in text


def test_the_gate_refuses_without_both_compilers(tmp_path: Path) -> None:
    """Exit 2, before lint, naming the compiler: never a pass."""
    done = _bash(f"devtools/gate.sh --report {tmp_path}",
                 HIPCC="fj-no-such-hipcc")
    assert done.returncode == 2, done.stdout + done.stderr
    assert "fj-no-such-hipcc" in done.stderr
    assert "== gate: lint" not in done.stdout


def test_the_gate_help_and_a_bad_flag() -> None:
    assert "PR gate" in _bash("devtools/gate.sh --help").stdout
    assert _bash("devtools/gate.sh --frobnicate").returncode == 64


def test_a_draw_renders_as_the_pr_body_section() -> None:
    got = sample.draw(CANDS, seed="s", budget=BUDGET)
    pools = {p: d.to_json() for p, d in got.pools.items()}
    report = {"seed": "s", "merge_base": "a" * 40, "head": "b" * 40,
              "base_ref": "origin/main", "sampled": True,
              "budget_s": dict(BUDGET), "cap": {"med": None, "max": None},
              "default_cost_s": 10.0, "touch_weight": 4.0,
              "touched": ["j-chain"], "pools": pools, "deselected": 3}
    text = "\n".join(sample.markdown_lines(report, "sample: gpu cuda"))
    assert text.startswith("#### sample: gpu cuda")
    assert "seed `s`" in text and "`j-chain`" in text
    for pool, d in pools.items():
        assert f"| `{pool}` |" in text
        for band in ("med", "max"):
            for n in d[band]["drawn"]:
                assert f"`{n}`" in text
