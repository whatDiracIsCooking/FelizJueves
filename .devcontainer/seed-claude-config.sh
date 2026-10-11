#!/usr/bin/env bash
# postCreateCommand for every variant under .devcontainer/: give this container
# a Claude Code config that is ALREADY LOGGED IN, and report what it got.
#
# Runs as `ubuntu` INSIDE the container, once per container creation (not per
# shell). It must stay idempotent: two of the paths it touches are named volumes
# that survive a `rebuild`, so nothing here may clobber a setting made in an
# earlier container.
#
# WHY A LOGIN WOULD OTHERWISE BE ASKED FOR EVERY TIME. Claude Code keeps the two
# halves of its auth in two places, and they are NOT in the same directory:
#
#   ~/.claude/.credentials.json   the OAuth token
#   ~/.claude.json                onboarding + oauthAccount, plus every
#                                 project's trust and allowedTools state
#
# ~/.claude.json sits BESIDE ~/.claude rather than inside it, so one mount
# cannot carry both. devcontainer.json binds the host's ~/.claude read-write --
# that is the token -- and the host's ~/.claude.json READ-ONLY at $SRC; this
# script takes a mutable per-container copy of the latter. The copy is the whole
# point: that file is one monolithic JSON rewritten whole on every change, so
# sharing it read-write between the host and several worktree containers is
# last-writer-wins.
#
# The second job, once logged in, is the TRUST DECISION for this checkout --
# keyed by absolute path, and this repo is mounted at /workspace rather than at
# its host path, so the host's entry does not apply to it. See the block that
# does it, below; it carries that decision and never makes one.
#
# THE WRITABILITY CHECK IS BELT-AND-BRACES, AND WARNS RATHER THAN FAILS.
# ~/.claude/plugins and ~/.claude/backups are named volumes, and a volume
# mounted over a directory inherits that directory's ownership from the IMAGE on
# first use -- which is why docker/Dockerfile.base pre-creates all three as uid
# 1000. `install -d -o 1000 a/b` applies the owner only to the LAST component,
# so listing just the two children there once left ~/.claude itself root-owned.
# Both causes are now fixed at the source: that install -d names ~/.claude
# explicitly, and devcontainer.json binds the host's over it anyway.
#
# It stays as a check because it caught that, and it stopped being FATAL for the
# same reason it was worth keeping: `exit 1` here aborts `devcontainer up` with
# "postCreateCommand failed" and a container id, naming neither the directory
# nor the mount. The container is still good for compiling and running kernels
# without an agent in it, so a config problem reports itself and gets out of the
# way.
set -uo pipefail

echo "seed-claude-config: user $(id -un) ($(id -u):$(id -g))"

degraded=0
for d in "$HOME/.claude" "$HOME/.claude/plugins" "$HOME/.claude/backups"; do
  mkdir -p "$d" 2>/dev/null
  [ -w "$d" ] && continue
  echo "  NOT WRITABLE: $d" >&2
  if [ "$d" = "$HOME/.claude" ]; then
    echo "    the host ~/.claude bind in .devcontainer/*/devcontainer.json is" >&2
    echo "    missing, or your host copy is not owned by uid 1000." >&2
  else
    echo "    check the volume's owner -- docker/Dockerfile.base's install -d." >&2
  fi
  degraded=1
done
[ "$degraded" = 0 ] && echo "  config dirs ok"

# The per-container copy of ~/.claude.json, from the host's read-only bind.
SRC="$HOME/.claude.json.host"
DST="$HOME/.claude.json"

# Where the workspace is mounted IN HERE -- /workspace, derived rather than
# spelled, so this keeps working if workspaceFolder ever changes. $FJ_HOST_WORKSPACE
# (containerEnv) is the same checkout's path ON THE HOST, and the two differ:
# that is what the trust carry-over below exists to bridge.
WS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [ -e "$DST" ]; then
  echo "  $DST already present, left as is"
elif [ ! -r "$SRC" ]; then
  echo "  no readable $SRC -- expect a login prompt; check the" >&2
  echo "  ~/.claude.json bind in .devcontainer/*/devcontainer.json" >&2
elif ! cp "$SRC" "$DST"; then
  echo "  could not copy $SRC to $DST -- expect a login prompt" >&2
else
  # Drop the HOST's installation bookkeeping, which does not describe this
  # container. installMethod records a native install at ~/.local/bin/claude, a
  # path that does not exist in here -- claude is installed BY THE IMAGE at
  # /usr/bin/claude, at the version docker/install-claude-code.sh pins -- and
  # the startup health check turns that into
  #   claude command at /home/ubuntu/.local/bin/claude missing or broken
  # Absent, installMethod derives to "unknown" and that check does not run.
  #
  # autoUpdates is pinned FALSE rather than dropped, because it defaults to
  # TRUE when absent and npm's global prefix in these images is root-owned /usr.
  # DISABLE_AUTOUPDATER in containerEnv says the same thing from the other side;
  # neither is load-bearing alone, and the version is a reviewable line in git
  # instead -- devtools/claude-version.sh --apply.
  #
  # oauthAccount and hasCompletedOnboarding -- the two keys that actually keep
  # you logged in -- are untouched.
  if python3 - "$DST" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as fh:
    cfg = json.load(fh)
for key in ("installMethod", "autoUpdatesProtectedForNative"):
    cfg.pop(key, None)
cfg["autoUpdates"] = False
with open(path, "w") as fh:
    json.dump(cfg, fh, indent=2)
PY
  then
    echo "  seeded $DST from the host (installMethod dropped, autoUpdates off)"
  else
    echo "  seeded $DST but could not rewrite it -- expect the health-check" >&2
    echo "  warning about ~/.local/bin/claude" >&2
  fi

  # Carry the TRUST DECISION for this checkout from its host path to the path it
  # is mounted at. Claude Code keys ~/.claude.json's `projects` map by absolute
  # path, so the host's entry for /home/you/.../main says nothing about
  # /workspace: without this, every new container asks the trust dialog again
  # and, until it is accepted, IGNORES the 40 permissions.allow entries in
  # .claude/settings.json -- it says so on startup.
  #
  # It carries a decision rather than making one: nothing is trusted here that
  # the host had not already trusted, and the keys copied are the DECISIONS
  # only. The host's entry also holds per-session telemetry (lastCost,
  # lastSessionId, lastFpsAverage...) describing a session that did not happen
  # in this container, and copying that would be a small lie in a file people
  # read.
  #
  # A FRESH .claude/worktrees/<name> IS NOT COVERED, by design: its host path
  # has no entry until you open claude there on the host, so its container asks
  # once. Widening this to "any path under the trusted primary checkout" would
  # be inventing the decision instead of carrying it.
  python3 - "$DST" "${FJ_HOST_WORKSPACE:-}" "$WS" <<'PY'
import json
import sys

path, host_ws, container_ws = sys.argv[1:4]

CARRY = (
    "hasTrustDialogAccepted",
    "allowedTools",
    "hasClaudeMdExternalIncludesApproved",
    "enabledMcpjsonServers",
    "disabledMcpjsonServers",
)

if not host_ws:
    print("  no FJ_HOST_WORKSPACE in env -- trust asked once in here (see containerEnv)")
    raise SystemExit(0)
if host_ws == container_ws:
    print(f"  workspace mounted at its host path ({host_ws}) -- trust already matches")
    raise SystemExit(0)

with open(path) as fh:
    cfg = json.load(fh)

src = cfg.get("projects", {}).get(host_ws, {})
if not src.get("hasTrustDialogAccepted"):
    print(f"  host has not trusted {host_ws} -- trust asked once in here")
    raise SystemExit(0)

dst = cfg.setdefault("projects", {}).setdefault(container_ws, {})
carried = [k for k in CARRY if k in src]
for key in carried:
    dst[key] = src[key]
with open(path, "w") as fh:
    json.dump(cfg, fh, indent=2)
print(f"  trust carried to {container_ws} from the host's {host_ws}")
print(f"    keys: {', '.join(carried)}")
PY
fi

# The pin, reported rather than enforced: docker/install-claude-code.sh already
# failed the BUILD if what landed was not what was asked for, so this is only
# so the container says out loud what it carries.
if command -v claude >/dev/null 2>&1; then
  echo "  claude $(claude --version 2>/dev/null | cut -d' ' -f1) (pinned ${CLAUDE_CODE_VERSION:-unset})"
else
  echo "  claude not on PATH -- expected in a GPU image; see docker/install-claude-code.sh" >&2
fi

# Auto-update is off through DISABLE_AUTOUPDATER in each devcontainer.json's
# containerEnv, not from here: npm's global prefix in these images is /usr and
# root-owned, so an in-container update cannot succeed.
if [ -n "${DISABLE_AUTOUPDATER:-}" ]; then
  echo "  autoupdater: disabled"
else
  echo "  autoupdater: ENABLED -- unexpected in a container; see containerEnv" >&2
fi

# Always 0: see the header. Nothing this script reports is worth failing
# `devcontainer up` over.
if [ "$degraded" = 0 ]; then
  echo "seed-claude-config: done"
else
  echo "seed-claude-config: done, with the warnings above" >&2
fi
exit 0
