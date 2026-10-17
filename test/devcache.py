"""Test-side device compiles through the compiler cache (#186) -- harness.

    launcher(env=, which=, venv_paths=)        -> ["/path/sccache"] or []
    localize(text, base, search=)              -> text, {unit name: bytes}
    unit_key(compiler, flags, main, units)     -> hex digest of every input
    compile_unit(compiler, flags, main, text, units, out, ...) -> Compiled
    stats(env=)                                -> {"hits": n, "misses": n}

WHY. devtools/kernel-tier.sh puts sccache in front of nvcc / hipcc for the
units KERNEL_GEN_CMD writes; before #186 every compile a TEST issued
(`conftest.compile_device_source`, `plan_driver.compile_checked`) ran cold,
and a re-run of the device suite paid the whole compile again.

ONE LAUNCHER, ONE SWITCH. `launcher()` resolves sccache the way the tier's
`find_tool sccache` does -- the project venvs' `bin/` (VENV_PATHS), then
PATH -- and an absent sccache is an empty launcher: a cold compile, which is
still a correct one. The opt-out is the variable devtools/config.sh names,
`KERNEL_NO_CACHE`: non-empty means no launcher. `kernel-tier.sh --no-cache`
exports it for the device phase, and it works the same with pytest alone.

PATHS A CACHE CAN HIT ON. A source written into a fresh `tmp_path` reaches
the compiler's preprocessed output (line markers, `__FILE__`) through its
path, so a content-identical unit would miss every time. `compile_unit`
therefore compiles in a STABLE, CONTENT-ADDRESSED directory,
`$KERNEL_BUILD_DIR/test-cache/<key>/`, keyed by `unit_key`: the compiler,
its flags, the main source's name and text, and the name and bytes of EVERY
unit it includes (a unit left out of the key would let a stale binary pass
for a fresh one -- test_devcache.py pins that). The binary is then COPIED to
the caller's `out`, so a test may move or rename it.

ALWAYS INVOKED. The compiler is run on every call, through the launcher, even
when the directory already holds a binary: reusing one without compiling
would also skip #92's diagnostics, and sccache makes the invocation cheap and
REPLAYS the stored stderr on a hit (test_devcache.py's `cuda` / `hip` tests
plant a warning and see it fail the filter on a hit). sccache caches only
`-c`-style compiles, never a link, so an executable is built in two steps:
`<launcher> <compiler> <flags> -c main -o main.o`, then `<compiler> <flags>
main.o -o exe` uncached. Outputs inside the shared directory get a
per-invocation name, so two xdist workers compiling the same unit never write
one file; the sources are written atomically (same bytes by construction).

`kernel-tier.sh --fresh` removes KERNEL_BUILD_DIR, so it clears this
directory too; the sccache store (SCCACHE_DIR) is not touched, and the next
run recompiles through it.

OUT OF SCOPE, deliberately: `felizjueves.measured --remeasure` compiles stay
COLD. They parse `ptxas -v` and AMD resource remarks from the compiler's own
output, and a cache replaying a stored stderr there is a provenance question
that needs its own decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NO_CACHE_VAR = "KERNEL_NO_CACHE"  # devtools/config.sh; test_devcache.py pins it
DEFAULT_VENV_PATHS = ".venv /opt/venv"


def launcher(env: Mapping[str, str] | None = None,
             which: Callable[[str], str | None] = shutil.which,
             venv_paths: str | None = None) -> list[str]:
    """The compiler-cache launcher to prepend, or [] (cold, still correct).

    Resolved as kernel-tier.sh's `find_tool sccache`: each of VENV_PATHS'
    `bin/sccache` that is executable, then `which("sccache")`. An empty list
    when `KERNEL_NO_CACHE` is set non-empty in `env`, or nothing is found.
    """
    env = os.environ if env is None else env
    if env.get(NO_CACHE_VAR):
        return []
    paths = venv_paths if venv_paths is not None else env.get(
        "VENV_PATHS", DEFAULT_VENV_PATHS)
    for v in paths.split():
        p = Path(v) if Path(v).is_absolute() else REPO / v
        cand = p / "bin" / "sccache"
        if os.access(cand, os.X_OK):
            return [str(cand)]
    found = which("sccache")
    return [found] if found else []


def unit_key(compiler: str, flags: list[str] | tuple[str, ...], main: str,
             text: str | bytes, units: Mapping[str, bytes]) -> str:
    """sha256 of everything that decides the binary: the compiler, its flags,
    the main source's NAME and text, and each included unit's name and bytes
    (in name order). Nothing about where the caller's copy lives."""
    h = hashlib.sha256()

    def put(b: bytes) -> None:
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)

    put(Path(compiler).name.encode())
    put(compiler.encode())
    put(json.dumps(list(flags)).encode())
    put(main.encode())
    put(text.encode() if isinstance(text, str) else text)
    for name in sorted(units):
        put(name.encode())
        put(units[name])
    return h.hexdigest()


@cache
def build_root() -> Path:
    """`$KERNEL_BUILD_DIR/test-cache`, KERNEL_BUILD_DIR read from
    devtools/config.sh by sourcing it (so the environment still wins)."""
    done = subprocess.run(
        ["bash", "-c", '. devtools/config.sh && printf %s "$KERNEL_BUILD_DIR"'],
        cwd=REPO, capture_output=True, text=True, check=True)
    base = Path(done.stdout.strip() or ".kernel-tier")
    if not base.is_absolute():
        base = REPO / base
    return base / "test-cache"


def _put(path: Path, data: bytes) -> None:
    """Write `data` at `path` atomically, unless it is already there."""
    try:
        if path.read_bytes() == data:
            return
    except OSError:
        pass
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _link_flags(flags: list[str]) -> list[str]:
    """The compile flags minus a `-x <lang>`, which would make the driver
    read the object as source."""
    out, skip = [], False
    for f in flags:
        if skip:
            skip = False
        elif f == "-x":
            skip = True
        elif not f.startswith("-x"):
            out.append(f)
    return out


_QUOTED = re.compile(r'^(\s*#\s*include\s+)"([^"]+)"', re.MULTILINE)


def localize(text: str, base: Path | None,
             search: Sequence[Path] = ()) -> tuple[str, dict[str, bytes]]:
    """Every quoted `#include "X"` of `text`, carried: X (absolute, or
    relative to `base`, or failing that to each of `search` in turn, as a
    compiler's `-I` would) is read, its own quoted includes carried the same
    way, and the directive rewritten to a bare name in one flat directory.
    Returns the rewritten text and `{name: bytes}` of every carried unit --
    which is what `unit_key` must see, so a unit that does not resolve is an
    error rather than something the key silently misses.

    A unit keeps its basename unless two different units share one, when the
    later gets a numbered prefix; the rewritten directive is then the only
    place the original path survived, and it is gone, so the preprocessed
    output no longer depends on where the caller wrote anything.
    """
    units: dict[str, bytes] = {}
    named: dict[Path, str] = {}

    def carry(src: str, here: Path | None) -> str:
        def one(m: re.Match[str]) -> str:
            spelled = Path(m[2])
            if spelled.is_absolute():
                path = spelled
            else:
                dirs = ([here] if here is not None else []) + list(search)
                path = next((d / spelled for d in dirs
                             if (d / spelled).is_file()), None)
            if path is None or not path.is_file():
                raise ValueError(f'#include "{m[2]}": no such unit '
                                 f"(relative to {here}) to carry into the key")
            path = path.resolve()
            if path not in named:
                name = path.name
                n = 0
                while name in units:
                    n += 1
                    name = f"u{n}_{path.name}"
                named[path] = name
                units[name] = b""  # reserve the name before recursing
                units[name] = carry(path.read_text(), path.parent).encode()
            return f'{m[1]}"{named[path]}"'

        return _QUOTED.sub(one, src)

    return carry(text, base), units


@dataclass(frozen=True)
class Compiled:
    """What one `compile_unit` ran and said."""

    returncode: int
    diagnostics: str  # stdout + stderr of every step, in order
    commands: tuple[tuple[str, ...], ...]
    workdir: Path  # the content-addressed directory
    out: Path


def compile_unit(compiler: str, flags: list[str], main: str, text: str,
                 units: Mapping[str, bytes], out: Path, *,
                 mode: str = "link", env: Mapping[str, str] | None = None,
                 timeout: float = 1800) -> Compiled:
    """Compile `text` (written as `main`, with `units` beside it under their
    names) in the content-addressed directory, through the launcher, and copy
    the result to `out`.

    `mode`: "link" (an executable: a cached `-c`, then an uncached link),
    "-c" or "-cubin" (one cached-or-passed-through step, the object itself).
    Every step runs; a failure stops after it with its returncode.
    """
    env = dict(os.environ if env is None else env)
    key = unit_key(compiler, flags, main, text, units)
    work = build_root() / key[:24]
    work.mkdir(parents=True, exist_ok=True)
    _put(work / main, text.encode())
    for name, data in units.items():
        _put(work / name, data)
    tag = uuid.uuid4().hex[:12]
    stem = Path(main).stem
    launch = launcher(env)
    obj = work / f"{stem}.{tag}.o"
    steps: list[list[str]] = []
    if mode == "link":
        exe = work / f"{stem}.{tag}"
        steps.append([*launch, compiler, *flags, "-c", main, "-o", obj.name])
        steps.append([compiler, *_link_flags(flags), obj.name, "-o", exe.name])
        result = exe
    else:
        steps.append([*launch, compiler, *flags, mode, main, "-o", obj.name])
        result = obj
    diag, rc = [], 0
    try:
        for cmd in steps:
            done = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, cwd=work, env=env)
            diag.append(done.stdout + done.stderr)
            rc = done.returncode
            if rc != 0:
                break
        if rc == 0:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result, out)
    finally:
        for f in {obj, result}:
            f.unlink(missing_ok=True)
    return Compiled(rc, "".join(diag), tuple(tuple(s) for s in steps), work, out)


def stats(env: Mapping[str, str] | None = None) -> dict[str, int] | None:
    """The sccache server's total hit and miss counts (every language), or
    None when there is no launcher or the server cannot be asked."""
    launch = launcher(env)
    if not launch:
        return None
    try:
        done = subprocess.run([*launch, "--show-stats", "--stats-format=json"],
                              capture_output=True, text=True, timeout=60,
                              env=dict(os.environ if env is None else env))
        s = json.loads(done.stdout)["stats"]
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError):
        return None
    return {"hits": sum(s["cache_hits"]["counts"].values()),
            "misses": sum(s["cache_misses"]["counts"].values())}
