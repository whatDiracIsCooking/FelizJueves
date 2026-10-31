"""Enumerated is not usable, and `present` means usable (#712).

#431 settled that a card is present when the kernel ENUMERATES it, not when its
device node exists. That rule is right and it is not enough: a container can
hold an enumerated card whose device-cgroup grant has since been revoked, and
then every enumeration probe still says yes while every open() returns EPERM.
`--gpus all` lands in HostConfig.DeviceRequests and is injected by an OCI hook
at container START, so a host driver reload, a `systemctl daemon-reload` or a
`docker update` regenerates the cgroup without it and the nodes stay visible
inside. That is what "the GPU dropped mid-session" is, and on 2026-10-31 it had
doctor.sh printing `verified on both cards` while gate.sh failed 311 cases on
`no CUDA-capable device is detected`.

So presence folds usability in, and a revoked card is ABSENT: the gpu sample is
skipped, the PR carries `unverified-on-card` and merges, and
verify-unverified.sh repays it -- while doctor.sh's `card usable` row and
devcontainer.sh FAIL loudly, so it is never SILENTLY absent. Two rules, kept
apart on purpose: `_present` / `card_present` decide whether a tier runs,
`card_revoked` decides whether a human is told their container is broken.

WHAT IS ASSERTED HERE, and what is not. Nothing in this file needs a card, a
compiler or a container: it is the fast tier, and its subject is the RULES --
which nodes each backend must open, what counts as an enumerated NVIDIA GPU,
and that denied collapses to absent -- read against fake sysfs and procfs trees
with the probe stubbed. Whether a real denied node is actually detected was
measured by hand on this box, in both directions and in both readings (a
container given `--gpus all` and then `docker update`: enumeration still named
the GPU, and all three nodes answered `Operation not permitted`; the same
container before the update opened every one), and cannot be asserted from a
test that must also pass on a device-less runner.

The drift guard at the bottom is the reason this file exists at all: the node
rule is written twice -- devtools/lib.sh's gpu_nodes for the shell tiers, and
conftest's GPU_NODE_GLOBS for the fixtures -- because a fixture cannot source a
shell file per test. Two copies of one rule is the thing that rots, so the
copies are compared.
"""

from __future__ import annotations

import re
from pathlib import Path

import conftest

REPO = Path(__file__).resolve().parent.parent
LIB_SH = REPO / "devtools" / "lib.sh"


# -- what counts as an enumerated NVIDIA GPU ---------------------------------


def test_nvidia_not_enumerated_when_no_gpu_is_bound(monkeypatch, tmp_path) -> None:
    """The loaded-module trap, which is #431 with the vendors swapped.

    /dev/nvidiactl is there whenever the nvidia module is loaded, card or not,
    so asking for it reports a card on a box that has none -- and a `gpu` test
    then FAILS where this repo's rule says it must skip. An empty
    /proc/driver/nvidia/gpus/ is the honest answer.
    """
    monkeypatch.setattr(conftest, "NVIDIA_GPUS", tmp_path / "gpus")
    monkeypatch.setattr(conftest.shutil, "which", lambda _tool: None)
    assert conftest._nvidia_enumerated() is False


def test_nvidia_enumerated_when_the_driver_bound_a_card(monkeypatch, tmp_path) -> None:
    """One directory per GPU the driver bound, named by PCI address."""
    gpus = tmp_path / "gpus"
    (gpus / "0000:0b:00.0").mkdir(parents=True)
    monkeypatch.setattr(conftest, "NVIDIA_GPUS", gpus)
    # No fallback needed, and none wanted: a procfs hit must not shell out.
    monkeypatch.setattr(conftest.shutil, "which",
                        lambda _tool: _fail_if_called())
    assert conftest._nvidia_enumerated() is True


def _fail_if_called():  # pragma: no cover - only runs on a regression
    raise AssertionError("nvidia-smi fallback used while procfs named a GPU")


# -- which render nodes are AMD's -------------------------------------------


def test_amd_render_nodes_exclude_the_other_vendor(monkeypatch, tmp_path) -> None:
    """A bare /dev/dri/renderD* glob is WRONG on a two-vendor box.

    This one has renderD128 on amdgpu and renderD129 on nvidia. ROCm never
    touches the second, so probing it would let an NVIDIA-side permissions
    quirk read as a revoked AMD card -- the kind of misattribution #431 was
    about. The filter is sysfs's vendor id, the field doctor.sh already picks
    the amdgpu card by for its VRAM reading.
    """
    for name, vendor in (("renderD128", "0x1002"), ("renderD129", "0x10de"),
                         ("card0", "0x1002")):
        dev = tmp_path / name / "device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text(vendor + "\n")
    monkeypatch.setattr(conftest, "DRM_CLASS", tmp_path)
    # card0 is display-side and not a render node, so it is not named either.
    assert conftest._amd_render_nodes() == ["/dev/dri/renderD128"]


def test_amd_render_nodes_empty_without_sysfs(monkeypatch, tmp_path) -> None:
    """No readable sysfs -> no render node, and /dev/kfd is probed alone.

    The right way to degrade: kfd is where the AMD-side permissions fault
    actually bites (a container user holding a `render` gid the bind-mounted
    node does not grant, which surfaces as hipErrorNoDevice), so the check
    still catches it and merely stops being able to name a second node.
    """
    monkeypatch.setattr(conftest, "DRM_CLASS", tmp_path / "absent")
    assert conftest._amd_render_nodes() == []


# -- denied collapses to absent (#712's acceptance) -------------------------


def test_a_denied_card_is_absent_not_present(monkeypatch, tmp_path) -> None:
    """The whole point: enumerated + denied must read as NO CARD.

    That is what makes a `gpu` test skip, `kernel-tier.sh` print
    `device: ABSENT`, and `gate.sh` label the PR `unverified-on-card` at exit
    0 -- instead of entering a device phase that fails every case on
    `no CUDA-capable device is detected`.
    """
    gpus = tmp_path / "gpus"
    (gpus / "0000:0b:00.0").mkdir(parents=True)
    monkeypatch.setattr(conftest, "NVIDIA_GPUS", gpus)
    monkeypatch.setattr(conftest, "_denied_nodes", lambda _b: ["/dev/nvidia0"])
    assert conftest._nvidia_enumerated() is True   # the card IS there
    assert conftest._present("cuda") is False      # and it is not usable
    assert conftest.PRESENT["cuda"]() is False


def test_an_openable_enumerated_card_is_present(monkeypatch, tmp_path) -> None:
    """The other side of the same rule, so it cannot degrade to always-absent."""
    gpus = tmp_path / "gpus"
    (gpus / "0000:0b:00.0").mkdir(parents=True)
    monkeypatch.setattr(conftest, "NVIDIA_GPUS", gpus)
    monkeypatch.setattr(conftest, "_denied_nodes", lambda _b: [])
    assert conftest._present("cuda") is True


def test_the_skip_reason_names_a_revocation_not_a_missing_card(
    monkeypatch, tmp_path
) -> None:
    """A revoked card and an absent one both skip, so the MESSAGE is the only
    thing that tells them apart -- and it is what `-rs` puts in front of you.

    It must name the denied nodes and the recreate, and must not claim the
    machine has no card: that is the sentence that sent three PRs looking at
    container passthrough.
    """
    gpus = tmp_path / "gpus"
    (gpus / "0000:0b:00.0").mkdir(parents=True)
    monkeypatch.setattr(conftest, "NVIDIA_GPUS", gpus)
    monkeypatch.setattr(conftest, "_denied_nodes", lambda _b: ["/dev/nvidiactl"])
    why = conftest._no_card_reason("cuda", "no NVIDIA device on this machine")
    assert "/dev/nvidiactl" in why
    assert "rebuild" in why and "cgroup" in why
    assert "no NVIDIA device on this machine" not in why

    # And with nothing denied, the plain absent message survives untouched.
    monkeypatch.setattr(conftest, "_denied_nodes", lambda _b: [])
    assert conftest._no_card_reason("cuda", "no NVIDIA device") == "no NVIDIA device"


# -- a device-less box reports absent, never denied -------------------------


def test_no_nodes_means_no_denial(monkeypatch) -> None:
    """A glob that matches nothing contributes nothing.

    This is what keeps a device-less runner -- no hosted one has a card --
    reporting ABSENT (a skip) rather than DENIED (a failure): _denied_nodes
    probes only nodes that exist, so an empty machine yields an empty list and
    the fixture never reaches the revocation branch.
    """
    monkeypatch.setitem(conftest.GPU_NODE_GLOBS, "cuda", ("/dev/fj-no-such-node*",))
    monkeypatch.setitem(conftest.GPU_NODE_GLOBS, "hip", ("/dev/fj-no-such-node*",))
    monkeypatch.setattr(conftest, "_amd_render_nodes", lambda: [])
    assert conftest._denied_nodes("cuda") == []
    assert conftest._denied_nodes("hip") == []


def test_a_regular_file_is_not_a_device(monkeypatch, tmp_path) -> None:
    """`is_char_device()` is load-bearing, not tidiness.

    The shell reading opens with bash's `<>`, which CREATES a missing file, so
    an unguarded probe would both report success and leave a regular file in
    /dev. Both readings therefore gate on the node being a character device --
    and a plain file matching the glob must be ignored, not probed.
    """
    (tmp_path / "nvidiactl").write_text("not a device")
    monkeypatch.setitem(conftest.GPU_NODE_GLOBS, "cuda", (str(tmp_path / "*"),))
    assert conftest._denied_nodes("cuda") == []


# -- the drift guard --------------------------------------------------------


def _lib_sh_nodes(backend: str) -> set[str]:
    """The device paths devtools/lib.sh's gpu_nodes names for one backend.

    Read out of the text rather than by running it: the function answers about
    THIS box, and what is being compared is the rule, not the hardware.
    """
    text = LIB_SH.read_text()
    body = re.search(rf"^    {backend}\)\n(.*?)^      ;;$", text,
                     re.MULTILINE | re.DOTALL)
    assert body, f"gpu_nodes has no {backend}) branch -- did lib.sh move?"
    # COMMENTS STRIPPED FIRST, and that is not incidental: the hip branch's
    # comment names `/dev/dri/renderD*` precisely to say why it is not used,
    # so a text scan that keeps comments finds the bug it is looking for in
    # the sentence explaining the fix.
    code = "\n".join(ln for ln in body.group(1).splitlines()
                     if not ln.lstrip().startswith("#"))
    return set(re.findall(r"/dev/[A-Za-z0-9*\[\]/_.-]+", code))


def test_cuda_node_rule_agrees_with_lib_sh() -> None:
    """The two readings of one rule must name the same CUDA nodes."""
    assert _lib_sh_nodes("cuda") == set(conftest.GPU_NODE_GLOBS["cuda"])


def test_hip_node_rule_agrees_with_lib_sh() -> None:
    """Same for HIP, where the render nodes are resolved rather than globbed.

    lib.sh walks /sys/class/drm and rebuilds `/dev/dri/<name>`; conftest's
    _amd_render_nodes returns the same paths. So the comparable part is the
    literal node list, and the shared requirement is that neither side carries
    a bare /dev/dri/renderD* glob -- which is the bug this pair guards.
    """
    lib = _lib_sh_nodes("hip")
    assert "/dev/kfd" in lib
    assert set(conftest.GPU_NODE_GLOBS["hip"]) == {"/dev/kfd"}
    assert "/dev/dri/renderD*" not in lib
