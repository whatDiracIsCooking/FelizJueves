#!/usr/bin/env bash
# Report -- and bump -- the Antigravity CLI release the GPU images are pinned to.
#
#   devtools/agy-version.sh                  pinned vs upstream vs installed
#   devtools/agy-version.sh --apply          rewrite the pin to upstream's current release
#   devtools/agy-version.sh --apply 1.3.3-5524738307653632
#   devtools/agy-version.sh --apply --rebuild    ... and rebuild the images
#                                                that already exist locally
#   devtools/agy-version.sh --verify         download the pinned tarball and check its sha512
#   devtools/agy-version.sh --porcelain      one `key value` per line
#
# The sibling of devtools/claude-version.sh, and the same shape: the pin is
# `ARG AGY_RELEASE=` plus `ARG AGY_SHA512=` in each file named by AGY_PIN_FILES
# in devtools/config.sh -- the three GPU Dockerfiles, which each install
# docker/install-agy.sh as a leaf layer. That script's header has why the pin
# exists and why it is a release rather than a version; this one keeps the
# copies in step, because a bump that reaches two files out of three is exactly
# the drift it was meant to end.
#
# TWO LINES PER FILE, TREATED AS ONE. AGY_SHA512 belongs to the exact tarball
# AGY_RELEASE names, and install-agy.sh VERIFIES it rather than recording it, so
# a release rewritten without its hash is a build that fails its own checksum
# and reports a corrupt-or-compromised payload. Every rewrite here therefore
# sets both, from one source:
#
#   --apply with no release   both come from upstream's manifest, which serves
#                             the version, the URL and the sha512 as one
#                             document -- the hash is never paired with a
#                             release it does not belong to;
#   --apply <release>         the hash is not derivable (the manifest serves
#                             only the CURRENT release), so the tarball is
#                             downloaded and hashed. That is the only honest
#                             way to pin an older release, and it is reported.
#
# Exit status: 0 when every file agrees with every other (and, for --apply,
# when the rewrite landed; for --verify, when the hash matched), 1 otherwise.
# Being BEHIND upstream is not a failure -- a pin is a deliberate choice of
# release, and this script's job is to make the gap visible, not to decide it.
#:
# Reading upstream needs the network. It is one unauthenticated GET of the
# per-platform manifest with a short timeout; when it fails, the upstream
# column reads `-` and everything else still works, which is what keeps
# doctor.sh usable on a box with no outbound route. `--verify` is the one mode
# that pulls the ~62MB tarball, and nothing on the report path does.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

REL_ARG=${AGY_RELEASE_ARG:-AGY_RELEASE}
SHA_ARG=${AGY_SHA512_ARG:-AGY_SHA512}
MANIFEST_BASE=${AGY_MANIFEST_BASE:?set in devtools/config.sh}
MANIFEST_PLATFORM=${AGY_MANIFEST_PLATFORM:-linux_amd64}
TIMEOUT=${AGY_REGISTRY_TIMEOUT:-8}
DOWNLOAD_TIMEOUT=${AGY_DOWNLOAD_TIMEOUT:-900}

# The tarball URL is reconstructed from the release, not remembered: this is
# the same expression install-agy.sh builds, and the two must not drift.
AGY_PLATFORM=${AGY_PLATFORM:-linux-x64}
AGY_URL_BASE=${AGY_URL_BASE:-https://storage.googleapis.com/antigravity-public/antigravity-cli}
tarball_url() { printf '%s/%s/%s/cli_%s.tar.gz\n' "$AGY_URL_BASE" "$1" "$AGY_PLATFORM" "${AGY_PLATFORM//-/_}"; }

# --- the pin ---------------------------------------------------------------

# Every file that carries the ARGs. A path that does not exist is a hard error
# rather than a skip: a renamed Dockerfile that silently drops out of this list
# is the drift, not a tidying-up.
pin_files() {
  config_lines "${AGY_PIN_FILES:-}"
  PIN_FILES=("${CONFIG_LINES[@]}")
  local f
  for f in "${PIN_FILES[@]}"; do
    [ -f "$f" ] || { echo "agy-version: no such file: $f (AGY_PIN_FILES)" >&2; exit 1; }
  done
  [ "${#PIN_FILES[@]}" -gt 0 ] || {
    echo "agy-version: AGY_PIN_FILES is empty -- nothing to report on" >&2
    exit 1
  }
}

release_in() { sed -n "s/^ARG ${REL_ARG}=//p" "$1" | head -1; }
sha_in()     { sed -n "s/^ARG ${SHA_ARG}=//p"  "$1" | head -1; }

# The pinned release and hash, or nothing when the files disagree. DISAGREE is
# left holding a `file=release` list for EVERY file either way -- the whole
# list, not just up to the first clash, because a partial bump that reached one
# file of three is the case this exists to name.
#
# A file whose release agrees but whose sha512 does not is ALSO a disagreement,
# and the list says so with `release/sha-differs`: two images installing
# different bytes under the same release id is worse than two obvious releases.
pinned_release() {
  local f r s first="" first_sha=""
  DISAGREE=()
  PINNED=""
  PINNED_SHA=""
  for f in "${PIN_FILES[@]}"; do
    r=$(release_in "$f")
    s=$(sha_in "$f")
    if [ -n "$first" ] && [ "$r" = "$first" ] && [ "$s" != "$first_sha" ]; then
      DISAGREE+=("$f=${r:-NONE}/sha-differs")
    else
      DISAGREE+=("$f=${r:-NONE}")
    fi
    if [ -z "$first" ]; then first=$r; first_sha=$s; fi
    { [ "$r" = "$first" ] && [ "$s" = "$first_sha" ]; } || first=DISAGREE
  done
  [ "$first" = DISAGREE ] && return 1
  PINNED=$first
  PINNED_SHA=$first_sha
  [ -n "$PINNED" ] && [ -n "$PINNED_SHA" ]
}

# --- the other two numbers --------------------------------------------------

# Upstream's manifest: version, url and sha512 in one document. The RELEASE is
# the path segment of the url (version plus build id), which is the thing the
# pin addresses -- `version` alone is not fetchable.
UPSTREAM_RELEASE=""
UPSTREAM_SHA=""
upstream_release() {
  local url="${MANIFEST_BASE}/${MANIFEST_PLATFORM}.json" json
  json=$(curl -fsSL --max-time "$TIMEOUT" "$url" 2>/dev/null) || return 0
  read -r UPSTREAM_RELEASE UPSTREAM_SHA <<<"$(printf '%s' "$json" | python3 -c '
import json, sys
try:
    m = json.load(sys.stdin)
    # The release is the path segment before the platform directory.
    print(m["url"].split("/")[-3], m["sha512"])
except Exception:
    pass
' 2>/dev/null)" || true
}

# What `agy` on THIS side actually is -- and only half the pin: the binary
# reports 1.3.3 and never the build id, so this is comparable to the release up
# to its first dash and no further.
installed_version() {
  command -v agy >/dev/null 2>&1 || return 0
  agy --version 2>/dev/null | head -1
}

release_ok() { [[ $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]]; }
sha_ok()     { [[ $1 =~ ^[0-9a-f]{128}$ ]]; }

# --- download and hash ------------------------------------------------------

# The sha512 of a release's tarball, by fetching it. Used by --apply <release>,
# where the hash is not otherwise knowable, and by --verify.
FETCHED_SHA=""
fetch_sha() {
  local release=$1 url tmp
  url=$(tarball_url "$release")
  tmp=$(mktemp -d)
  # shellcheck disable=SC2064  # expand tmp now, not at trap time
  trap "rm -rf '$tmp'" RETURN
  echo "agy-version: fetching $url" >&2
  curl -fsSL --max-time "$DOWNLOAD_TIMEOUT" -o "$tmp/agy.tar.gz" "$url" || {
    echo "agy-version: could not download $url" >&2
    echo "  a release is addressed by version AND build id -- check the release string" >&2
    return 1
  }
  FETCHED_SHA=$(sha512sum "$tmp/agy.tar.gz" | cut -d' ' -f1)
}

# --- rewrite ----------------------------------------------------------------

# Both ARGs, in every file, or none: the loop writes release and hash together
# and re-reads each to confirm, so a sed that matched nothing is an error
# rather than a silently unchanged file.
apply_pin() {
  local want=$1 want_sha=$2 f before
  release_ok "$want" || { echo "agy-version: '$want' is not a release" >&2; exit 1; }
  sha_ok "$want_sha" || { echo "agy-version: '$want_sha' is not a sha512" >&2; exit 1; }
  for f in "${PIN_FILES[@]}"; do
    before=$(release_in "$f")
    [ -n "$before" ] || { echo "agy-version: $f has no ARG ${REL_ARG} line" >&2; exit 1; }
    [ -n "$(sha_in "$f")" ] || { echo "agy-version: $f has no ARG ${SHA_ARG} line" >&2; exit 1; }
    sed -i "s/^ARG ${REL_ARG}=.*/ARG ${REL_ARG}=${want}/" "$f"
    sed -i "s/^ARG ${SHA_ARG}=.*/ARG ${SHA_ARG}=${want_sha}/" "$f"
    if [ "$(release_in "$f")" != "$want" ] || [ "$(sha_in "$f")" != "$want_sha" ]; then
      echo "agy-version: rewrite of $f did not take" >&2; exit 1
    fi
    printf '  %s: %s -> %s\n' "$f" "$before" "$want"
  done
}

# Rebuild only what is already built -- nothing here invents a ~40GB
# `combined` build on a box that has never had one.
rebuild_built_images() {
  local f target
  for f in "${PIN_FILES[@]}"; do
    target=${f##*Dockerfile.}
    if docker image inspect "${PROJECT_NAME}:${target}" >/dev/null 2>&1; then
      echo "agy-version: + docker/build.sh $target"
      docker/build.sh "$target"
    else
      echo "agy-version: ${PROJECT_NAME}:${target} not built here -- skipping"
      echo "                build it with: docker/build.sh $target"
    fi
  done
}

# --- arguments --------------------------------------------------------------

mode=report
want=""
rebuild=
while [ $# -gt 0 ]; do
  case $1 in
    --apply)     mode=apply ;;
    --rebuild)   mode=apply; rebuild=1 ;;
    --verify)    mode=verify ;;
    --porcelain) mode=porcelain ;;
    -h|--help)   usage; exit 0 ;;
    -*)          echo "agy-version: unknown option $1" >&2; usage >&2; exit 2 ;;
    *)           want=$1 ;;
  esac
  shift
done

pin_files

case $mode in
  porcelain)
    pinned_release || true
    upstream_release
    printf 'pinned %s\n'      "${PINNED:--}"
    printf 'sha512 %s\n'      "${PINNED_SHA:--}"
    printf 'files %s\n'       "${DISAGREE[*]}"
    printf 'upstream %s\n'    "${UPSTREAM_RELEASE:--}"
    printf 'installed %s\n'   "$(installed_version || true)"
    [ -n "$PINNED" ]
    ;;

  report)
    upstream_release
    here=$(installed_version)
    if pinned_release; then
      printf 'pinned     %s   (%s)\n' "$PINNED" "${PIN_FILES[*]}"
      printf 'sha512     %s\n' "$PINNED_SHA"
    else
      printf 'pinned     DISAGREE: %s\n' "${DISAGREE[*]}"
      echo   '           fix with: devtools/agy-version.sh --apply <release>'
    fi
    printf 'upstream   %s   (%s/%s.json)\n' \
      "${UPSTREAM_RELEASE:-unreachable}" "$MANIFEST_BASE" "$MANIFEST_PLATFORM"
    printf 'installed  %s   (%s)\n' "${here:-none}" "$(command -v agy || echo 'not on PATH here')"
    echo
    if [ -n "$PINNED" ] && [ -n "$UPSTREAM_RELEASE" ] && [ "$PINNED" != "$UPSTREAM_RELEASE" ]; then
      echo "the pin is behind upstream: $PINNED -> $UPSTREAM_RELEASE"
      echo "  bump it:  devtools/agy-version.sh --apply"
      echo "  then:     docker/build.sh cuda   (and hip / combined as you use them)"
    elif [ -n "$PINNED" ] && [ "$PINNED" = "$UPSTREAM_RELEASE" ]; then
      echo "the pin is current"
    fi
    # The binary's version is only the half before the dash -- say which half
    # was compared, so `1.3.3` against `1.3.3-5524738307653632` does not read
    # as a mismatch.
    if [ -n "$here" ] && [ -n "$PINNED" ] && [ "$here" != "${PINNED%%-*}" ]; then
      echo "note: agy here reports $here, the pin's version half is ${PINNED%%-*}"
    fi
    [ -n "$PINNED" ]
    ;;

  verify)
    pinned_release || {
      echo "agy-version: the files disagree -- nothing single to verify: ${DISAGREE[*]}" >&2
      exit 1
    }
    fetch_sha "$PINNED" || exit 1
    if [ "$FETCHED_SHA" = "$PINNED_SHA" ]; then
      echo "agy-version: sha512 matches the pin for release $PINNED"
      echo "  $PINNED_SHA"
    else
      echo "agy-version: SHA512 MISMATCH for release $PINNED" >&2
      echo "  pinned $PINNED_SHA" >&2
      echo "  actual $FETCHED_SHA" >&2
      echo "  the build would refuse this tarball; do not rebuild until this is explained" >&2
      exit 1
    fi
    ;;

  apply)
    if [ -z "$want" ]; then
      upstream_release
      [ -n "$UPSTREAM_RELEASE" ] || {
        echo "agy-version: upstream unreachable and no release given" >&2; exit 1; }
      [ -n "$UPSTREAM_SHA" ] || {
        echo "agy-version: upstream manifest carried no sha512 -- refusing to pin a release without its hash" >&2
        exit 1; }
      echo "agy-version: upstream's current release is $UPSTREAM_RELEASE"
      apply_pin "$UPSTREAM_RELEASE" "$UPSTREAM_SHA"
    else
      # An explicit release: the manifest cannot answer for it, so the tarball
      # is the only source for its hash.
      echo "agy-version: $want is not upstream's current release, so its sha512"
      echo "             is not in the manifest -- hashing the tarball instead"
      fetch_sha "$want" || exit 1
      apply_pin "$want" "$FETCHED_SHA"
    fi
    if [ -n "$rebuild" ]; then
      rebuild_built_images
    else
      echo "agy-version: rebuild to pick it up -- docker/build.sh cuda (and hip / combined)"
      echo "             or re-run with --rebuild to do the ones already built here"
    fi
    ;;
esac
