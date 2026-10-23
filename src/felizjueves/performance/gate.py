"""The card gate: wait for a quiet card (and box) before a measurement (#109).

ONE HELPER, LIFTED. It was test-local in `test/test_kernel_device.py`, wired
to the wall-clock sweep only (#55, #93). A ceiling probe (`ceilings.py`) is the
second measurement it must guard and #112's counter sweep the third, and a
driver cannot import a test module -- so it lives here, stdlib only, and the
wall-clock path reads the same code it read before (its section comment is
still the protocol).

WHAT IT GUARDS, AND WHY IT IS LOAD-BEARING. A measurement taken on a busy
card is not a noisy measurement: a timing is garbage (#55's bimodal rows,
while a process outside the container held the RTX 3080), and a ceiling
UNDER-REPORTS -- which lowers the ridge and pushes every roofline verdict
toward compute-bound. Correctness tests do not wait on it (test/conftest.py,
#91): a correct answer does not care how fast it arrives.

THE READINGS (#93's fields):

- `card_util(backend)`: the card's utilisation now, the highest of three SMI
  samples 0.2 s apart, or `None` -- UNKNOWN, never idle -- when no sample
  parsed (NVML's intermittent "Unknown Error" included).
- BEFORE: `wait_quiet` blocks until `card_util` is at most GPU_BUSY_MAX (for
  up to GPU_QUIET_WAIT_S) and then until the CPU is at most BUSY_MAX busy (for
  up to QUIET_WAIT_S), and returns both settled readings; past either deadline
  it raises `Contended` -- the phase is discarded, not averaged in.
- AFTER: `after_reading` -- one SMI read discarded, GPU_SETTLE_S of sleep,
  then `card_util`. nvidia-smi reports over a trailing window of up to 1 s
  (100% for the first second after a run, then 0%); rocm-smi's FIRST read
  after a pause reports the average SINCE THE PREVIOUS READ (96% at +0.08 s,
  75% at +1 s, 52% at +3 s, each followed at once by 0-1%) -- so a read that
  includes that window measures our own run. On both vendors the window it
  reads is the settle alone.

`gated(run, backend)` is the whole phase -- wait, run, read after -- and
REFUSES (raises `Contended`) when the after-reading is non-idle or unknown:
the ceiling probes use it, so a contended or unreadable card records nothing.
The wall-clock sweep records the after-reading instead of refusing on it, as
it did before this module existed (its rows carry both readings).
"""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass

GPU_BUSY_MAX = 30    # percent utilization of the card, from its own SMI
GPU_QUIET_WAIT_S = 900.0
GPU_SETTLE_S = 1.0
GPU_POLL_S = 10.0
BUSY_MAX = 0.10      # of all cores: ~2 of this box's 24, so no compile-bound job
QUIET_WAIT_S = 120.0

Reader = Callable[[str], "int | None"]


def _say(line: str) -> None:
    print(line, flush=True)


class Contended(RuntimeError):
    """The gate did not let the phase through: discard it, record nothing."""


def card_utilisation(backend: str) -> int | None:
    """The card's utilisation in percent, from its own SMI, or None -- UNKNOWN,
    never idle -- when the tool says nothing parseable.

    #55 found the RTX 3080 at 100% and 6.3 GiB held by a process outside this
    container (invisible to nvidia-smi's process list here, and to the CPU
    gate): the card's own utilisation is the only signal that sees that.
    """
    if backend == "cuda":
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu",
               "--format=csv,noheader,nounits"]
        pat = r"^\s*(\d+)"
    else:
        cmd = ["rocm-smi", "--showuse"]
        pat = r"GPU use \(%\):\s*(\d+)"
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(pat, out, re.MULTILINE)
    return int(m[1]) if m else None


def cpu_busy(seconds: float = 1.0) -> float:
    """Fraction of all cores busy over `seconds`, from /proc/stat."""
    def snap() -> tuple[int, int]:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        return sum(v), v[3] + v[4]
    t0, i0 = snap()
    time.sleep(seconds)
    t1, i1 = snap()
    return 1.0 - (i1 - i0) / max(1, t1 - t0)


def card_util(backend: str, *, read: Reader | None = None,
              sleep: Callable[[float], None] = time.sleep) -> int | None:
    """The highest of three SMI samples 0.2 s apart, or None if none parsed."""
    read = read or card_utilisation
    samples = []
    for _ in range(3):
        samples.append(read(backend))
        sleep(0.2)
    return max((s for s in samples if s is not None), default=None)


def is_idle(util: int | None) -> bool:
    """A reading the gate lets through: parsed, and at most GPU_BUSY_MAX."""
    return util is not None and util <= GPU_BUSY_MAX


def wait_card(backend: str, *, read: Reader | None = None,
              wait_s: float | None = None,
              sleep: Callable[[float], None] = time.sleep,
              say: Callable[[str], None] | None = None) -> int:
    """Block until the card reads idle; return that settled reading."""
    wait_s = GPU_QUIET_WAIT_S if wait_s is None else wait_s
    t0 = time.monotonic()
    deadline = t0 + wait_s
    while True:
        util = card_util(backend, read=read, sleep=sleep)
        if is_idle(util):
            return util  # type: ignore[return-value]
        if time.monotonic() >= deadline:
            raise Contended(f"card contended: {util}% busy > {GPU_BUSY_MAX}% "
                            f"for {wait_s:.0f} s" if util is not None else
                            f"card unreadable (SMI said nothing parseable) for "
                            f"{wait_s:.0f} s")
        (say or _say)(f"gate: card {util}% busy > {GPU_BUSY_MAX}%, waiting "
                      f"({time.monotonic() - t0:.0f} s)")
        sleep(GPU_POLL_S)


def wait_cpu(*, busy: Callable[[], float] | None = None,
             wait_s: float | None = None) -> float:
    """Block until the CPU is at most BUSY_MAX busy; return that fraction."""
    busy = busy or cpu_busy
    wait_s = QUIET_WAIT_S if wait_s is None else wait_s
    deadline = time.monotonic() + wait_s
    while True:
        b = busy()
        if b <= BUSY_MAX:
            return b
        if time.monotonic() >= deadline:
            raise Contended(f"box contended: CPU {b:.0%} busy > {BUSY_MAX:.0%} "
                            f"for {wait_s:.0f} s")


def wait_quiet(backend: str | None = None, *, read: Reader | None = None,
               wait_s: float | None = None,
               cpu_busy: Callable[[], float] | None = None,
               sleep: Callable[[float], None] = time.sleep,
               say: Callable[[str], None] | None = None,
               ) -> tuple[float, int | None]:
    """The BEFORE half: the card (given `backend`), then the CPU. Returns
    (CPU busy fraction, the card's settled reading or None without backend)."""
    util = None
    if backend is not None:
        util = wait_card(backend, read=read, wait_s=wait_s, sleep=sleep, say=say)
    cpu_wait = None if wait_s is None else min(wait_s, QUIET_WAIT_S)
    return wait_cpu(busy=cpu_busy, wait_s=cpu_wait), util


def after_reading(backend: str, *, read: Reader | None = None,
                  sleep: Callable[[float], None] = time.sleep) -> int | None:
    """The AFTER half: one read discarded, GPU_SETTLE_S, then `card_util`."""
    (read or card_utilisation)(backend)  # discarded: rocm-smi averages over
    sleep(GPU_SETTLE_S)                  # the pause since its last read
    return card_util(backend, read=read, sleep=sleep)


@dataclass(frozen=True)
class Phase:
    """The gate's readings around one measured phase."""

    cpu_before: float
    gpu_before: int
    gpu_after: int


def gated(run: Callable[[], object], backend: str, *, read: Reader | None = None,
          wait_s: float | None = None, cpu_busy: Callable[[], float] | None = None,
          sleep: Callable[[float], None] | None = None):
    """Wait quiet, `run()`, read after; return (run's result, Phase).

    Raises `Contended` -- the phase recorded nowhere -- if the card or CPU
    never reads idle before, or the after-reading is non-idle or unknown.
    """
    sleep = sleep or time.sleep
    cpu, before = wait_quiet(backend, read=read, wait_s=wait_s,
                             cpu_busy=cpu_busy, sleep=sleep)
    result = run()
    after = after_reading(backend, read=read, sleep=sleep)
    if not is_idle(after):
        raise Contended(
            f"card read {after}% after the phase (> {GPU_BUSY_MAX}%)"
            if after is not None else
            "card unreadable after the phase (UNKNOWN, never idle)")
    return result, Phase(cpu, before, after)  # type: ignore[arg-type]
