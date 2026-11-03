#!/usr/bin/env bash
set -euo pipefail
QUIDRA="$1"
DEVICE="${2:--1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_PACKAGE_PATH="$(dirname "$ROOT")"
export QUIDRA_CACHE_DIR="$TMP/cache"
"$QUIDRA" "$ROOT/tests/nonfinite_outputs.qui" --device "$DEVICE" --internal > "$TMP/internal.out"
printf 'finite softmax output true
' > "$TMP/expected"
diff -u "$TMP/expected" "$TMP/internal.out"
# Infinity remains observable, unlike NaN. Exercise CPU and device paths.
"$QUIDRA" "$ROOT/tests/nonfinite_outputs.qui" --device "$DEVICE" > "$TMP/overflow.out" 2> "$TMP/overflow.err"
printf 'infinite relu output true\n' > "$TMP/expected-infinity"
diff -u "$TMP/expected-infinity" "$TMP/overflow.out"
if [[ -s "$TMP/overflow.err" ]]; then
    echo "unexpected NN infinity diagnostics:" >&2
    cat "$TMP/overflow.err" >&2
    exit 1
fi
