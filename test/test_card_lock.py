"""One lock per CARD, not per backend.

The card lock (#91) used to be one file per backend, `/tmp/fj-gpu-cuda.lock`,
which is right on a box with one card of each vendor and wrong on one with
four of a kind: every run launched on device 0 and queued on that one file
while three cards sat idle. The lock is now the card's, `{index}` in
GPU_LOCK_PATH (devtools/config.sh) being the physical card the run launches
on, read from CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES (through
ROCR_VISIBLE_DEVICES), card 0 when none is set.

The rule is written twice -- devtools/lib.sh's gpu_card_index for the shell
tiers and doctor, test/conftest.py's card_index for the fixtures that take the
lock -- so the two are run over the same environments here and must agree.
Nothing in this file needs a card: the probes are stubbed or read from fake
trees, and the shell runs with a scrubbed environment.
"""

from __future__ import annotations

import fcntl
import fnmatch
import os
import subprocess
from pathlib import Path

import conftest
import pytest

REPO = Path(__file__).resolve().parent.parent
LIB_SH = REPO / "devtools" / "lib.sh"
CARD_VARS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


@pytest.fixture(autouse=True)
def _no_card_selection(monkeypatch) -> None:
    """This file's subject is the selection, so the caller's is removed: a
    run under `CUDA_VISIBLE_DEVICES=2` must not change what is asserted."""
    for v in (*CARD_VARS, "GPU_LOCK_PATH"):
        monkeypatch.delenv(v, raising=False)
    conftest.lock_path.cache_clear()
    yield
    conftest.lock_path.cache_clear()


def _sh(script: str, **env: str) -> subprocess.CompletedProcess[str]:
    """Run `script` after sourcing lib.sh, with only PATH, HOME and `env`."""
    return subprocess.run(
        ["bash", "-c", f". {LIB_SH}; {script}"], capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", **env})


# backend, environment, the card it names
CASES = [
    ("cuda", {}, "0"),                                   # today's default
    ("hip", {}, "0"),
    ("cuda", {"CUDA_VISIBLE_DEVICES": "2"}, "2"),
    ("cuda", {"CUDA_VISIBLE_DEVICES": "3,1"}, "3"),       # device 0 is the first
    ("cuda", {"CUDA_VISIBLE_DEVICES": " 1 , 0"}, "1"),    # whitespace dropped
    ("cuda", {"CUDA_VISIBLE_DEVICES": ""}, "0"),          # set and empty
    ("cuda", {"CUDA_VISIBLE_DEVICES": "GPU-ab/cd"}, "GPU-ab_cd"),  # a UUID, file-safe
    ("cuda", {"HIP_VISIBLE_DEVICES": "2"}, "0"),          # the other vendor's
    ("hip", {"CUDA_VISIBLE_DEVICES": "2"}, "0"),
    ("hip", {"HIP_VISIBLE_DEVICES": "1"}, "1"),
    ("hip", {"ROCR_VISIBLE_DEVICES": "2,3"}, "2"),        # HIP device 0 of ROCr's
    ("hip", {"HIP_VISIBLE_DEVICES": "1", "ROCR_VISIBLE_DEVICES": "2,3"}, "3"),
    ("hip", {"HIP_VISIBLE_DEVICES": "5", "ROCR_VISIBLE_DEVICES": "2,3"}, "5"),
    ("hip", {"HIP_VISIBLE_DEVICES": "08",
             "ROCR_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7,9"}, "9"),  # decimal, not octal
]


@pytest.mark.parametrize(("backend", "env", "card"), CASES)
def test_card_index_python_and_shell_agree(backend, env, card) -> None:
    assert conftest.card_index(backend, env) == card
    done = _sh(f"gpu_card_index {backend}", **env)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == card


def test_card_index_reads_the_process_environment(monkeypatch) -> None:
    """With no `env` argument it is this process's, which is what the fixtures
    and the xdist controller ask."""
    assert conftest.card_index("cuda") == "0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert conftest.card_index("cuda") == "2"


def test_an_unknown_backend_is_rc_2_not_a_card() -> None:
    bad = _sh("gpu_card_index gpu")
    assert bad.returncode == 2 and "unknown backend" in bad.stderr


# -- the path --------------------------------------------------------------


def test_config_sh_default_is_one_lock_per_card() -> None:
    done = _sh('printf %s "$GPU_LOCK_PATH"')
    assert done.stdout == "/tmp/fj-gpu-{backend}.{index}.lock"


@pytest.mark.parametrize(("backend", "env", "path"), [
    ("cuda", {}, "/tmp/fj-gpu-cuda.0.lock"),
    ("cuda", {"CUDA_VISIBLE_DEVICES": "2"}, "/tmp/fj-gpu-cuda.2.lock"),
    ("hip", {}, "/tmp/fj-gpu-hip.0.lock"),
    ("hip", {"HIP_VISIBLE_DEVICES": "1", "ROCR_VISIBLE_DEVICES": "0,4"},
     "/tmp/fj-gpu-hip.4.lock"),
])
def test_lock_path_python_and_shell_agree(monkeypatch, backend, env, path) -> None:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert str(conftest.lock_path(backend)) == path
    assert _sh(f"gpu_lock_path {backend}", **env).stdout.strip() == path


def test_an_explicit_lock_path_wins_as_is(monkeypatch) -> None:
    """A path with no placeholder is used verbatim, whatever card is chosen --
    which is how a run still on the old name keeps its lock."""
    env = {"GPU_LOCK_PATH": "/tmp/fj-gpu-cuda3.lock", "CUDA_VISIBLE_DEVICES": "3"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert str(conftest.lock_path("cuda")) == "/tmp/fj-gpu-cuda3.lock"
    assert _sh("gpu_lock_path cuda", **env).stdout.strip() == "/tmp/fj-gpu-cuda3.lock"


def test_an_explicit_lock_path_still_fills_its_placeholders(
        monkeypatch, tmp_path) -> None:
    env = {"GPU_LOCK_PATH": f"{tmp_path}/{{backend}}-{{index}}",
           "CUDA_VISIBLE_DEVICES": "1"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert conftest.lock_path("cuda") == tmp_path / "cuda-1"
    assert _sh("gpu_lock_path cuda", **env).stdout.strip() == f"{tmp_path}/cuda-1"


# -- the probes ask about the selected card ---------------------------------


def test_an_index_past_the_last_card_is_no_card(monkeypatch, tmp_path) -> None:
    """CUDA would see no device 7 on a four-card box, so neither may the
    presence probe -- the tests skip, as they do on a device-less runner."""
    gpus = tmp_path / "gpus"
    for i in range(4):
        (gpus / f"0000:0{i + 6}:00.0").mkdir(parents=True)
    monkeypatch.setattr(conftest, "NVIDIA_GPUS", gpus)
    assert conftest._nvidia_enumerated() is True
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert conftest._nvidia_enumerated() is True
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert conftest._nvidia_enumerated() is False
    # A UUID cannot be counted against, and is taken on trust.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-7e4d")
    assert conftest._nvidia_enumerated() is True


def test_the_hip_index_is_counted_against_kfd(monkeypatch, tmp_path) -> None:
    nodes = tmp_path / "nodes"
    for i, gfx in enumerate((0, 120000)):  # the CPU node, then one GPU
        (nodes / str(i)).mkdir(parents=True)
        (nodes / str(i) / "properties").write_text(f"gfx_target_version {gfx}\n")
    monkeypatch.setattr(conftest, "KFD_NODES", nodes)
    assert conftest._amd_enumerated() is True
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    assert conftest._amd_enumerated() is False


def test_the_shell_counts_the_same_way() -> None:
    """lib.sh's enumeration through gpu_card_count, stubbed to four cards."""
    four = "gpu_card_count() { echo 4; }; "
    assert _sh(four + "nvidia_enumerated").returncode == 0
    assert _sh(four + "nvidia_enumerated", CUDA_VISIBLE_DEVICES="3").returncode == 0
    assert _sh(four + "nvidia_enumerated", CUDA_VISIBLE_DEVICES="4").returncode == 1
    assert _sh("gpu_card_count() { echo 0; }; amd_enumerated").returncode == 1
    assert _sh("gpu_card_count() { echo 1; }; amd_enumerated").returncode == 0
    assert _sh("gpu_card_count() { echo 1; }; amd_enumerated",
               HIP_VISIBLE_DEVICES="1").returncode == 1


def test_the_card_node_is_the_selected_one(monkeypatch) -> None:
    """A launch on card 2 opens /dev/nvidia2; card 3's grant is not its
    problem. When the selected node is not here -- a container handed one card
    -- every card node is probed, as before."""
    here = ["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia0", "/dev/nvidia2"]
    monkeypatch.setattr(conftest.glob, "glob",
                        lambda pattern: fnmatch.filter(here, pattern))
    monkeypatch.setattr(conftest.Path, "is_char_device", lambda self: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert conftest._card_nodes("cuda") == [
        "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia2"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")  # no /dev/nvidia1 here
    assert conftest._card_nodes("cuda") == here


# -- choosing a card ---------------------------------------------------------


def test_select_card_exports_the_backends_variable() -> None:
    done = _sh('select_card cuda 2 && echo "$CUDA_VISIBLE_DEVICES $SELECTED_CARD"'
               ' && gpu_lock_path cuda')
    assert done.stdout.split() == ["2", "2", "/tmp/fj-gpu-cuda.2.lock"]
    done = _sh('select_card hip 1 && echo "$HIP_VISIBLE_DEVICES"')
    assert done.stdout.strip() == "1"
    bad = _sh("select_card cuda two")
    assert bad.returncode == 64 and "card index or 'auto'" in bad.stderr


def test_auto_picks_the_first_card_whose_lock_is_free(tmp_path) -> None:
    template = f"{tmp_path}/fj-{{backend}}.{{index}}.lock"
    three = "gpu_card_count() { echo 3; }; "
    held = tmp_path / "fj-cuda.0.lock"
    fd = os.open(held, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        done = _sh(three + "gpu_free_card cuda", GPU_LOCK_PATH=template)
        assert done.stdout.strip() == "1"  # 0 held; 1 has no file yet: free
        # Every lock held: card 0, where the run will wait and say for whom.
        for i in (1, 2):
            other = os.open(tmp_path / f"fj-cuda.{i}.lock",
                            os.O_RDWR | os.O_CREAT, 0o666)
            fcntl.flock(other, fcntl.LOCK_EX)
        done = _sh(three + "gpu_free_card cuda", GPU_LOCK_PATH=template)
        assert done.stdout.strip() == "0"
    finally:
        os.close(fd)


def test_kernel_tier_refuses_a_bad_device_before_any_work() -> None:
    done = subprocess.run(
        [str(REPO / "devtools" / "kernel-tier.sh"), "--backend", "cuda",
         "--device", "two"], capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"})
    assert done.returncode == 64, done.stderr
    assert "card index or 'auto'" in done.stderr


@pytest.mark.parametrize("script", ["kernel-tier.sh", "measure.sh", "gate.sh"])
def test_the_tiers_offer_device_selection(script) -> None:
    text = (REPO / "devtools" / script).read_text()
    assert "--device N|auto" in text  # in the usage header
    assert "--device)" in text        # and parsed
