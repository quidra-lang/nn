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
set +e
"$QUIDRA" "$ROOT/tests/nonfinite_outputs.qui" --device "$DEVICE" > "$TMP/overflow.out" 2> "$TMP/overflow.err"
failure_status=$?
set -e
if [[ "$failure_status" -ne 101 ]]; then
    echo "expected NN nonfinite output failure with status 101, got $failure_status" >&2
    cat "$TMP/overflow.err" >&2; exit 1
fi
if [[ "$DEVICE" == -1 ]]; then
    grep -q 'NONFINITE_RESULT' "$TMP/overflow.err"
else
    grep -Fq 'Quidra runtime error[TENSOR]' "$TMP/overflow.err"
    grep -q 'produced a non-finite real32 value' "$TMP/overflow.err"
fi
