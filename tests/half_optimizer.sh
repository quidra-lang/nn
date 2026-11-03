#!/usr/bin/env bash
set -euo pipefail
QUIDRA="$1"
MODE="${2:-cpu}"
DEVICE="${3:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_PACKAGE_PATH="$(dirname "$ROOT")"
export QUIDRA_CACHE_DIR="$TMP/cache"
extra=()
case "$MODE" in
    cpu) ;;
    metal) extra=(--device "$DEVICE" --metal) ;;
    fake) extra=(--device "$DEVICE") ;;
    *) echo "expected cpu, metal or fake" >&2; exit 2 ;;
esac
"$QUIDRA" "$ROOT/tests/half_optimizer.qui" ${extra[@]+"${extra[@]}"} > "$TMP/output"
expected=9
if [[ "$MODE" == metal ]]; then expected=12; fi
if [[ "$(grep -c ' true$' "$TMP/output" || true)" != "$expected" ]] || grep -q ' false$' "$TMP/output"; then
    cat "$TMP/output" >&2; exit 1
fi
if [[ "$MODE" == fake ]]; then exit 0; fi
failure_extra=()
if [[ "$MODE" == metal ]]; then failure_extra=(--device "$DEVICE"); fi
for mode in conversion moment; do
    arg=()
    if [[ "$mode" == moment ]]; then arg=(--moment); fi
    set +e
    "$QUIDRA" "$ROOT/tests/half_optimizer_failure.qui" ${failure_extra[@]+"${failure_extra[@]}"} ${arg[@]+"${arg[@]}"} > "$TMP/$mode.out" 2> "$TMP/$mode.err"
    failure_status=$?
    set -e
    if [[ "$mode" == moment ]]; then
        # A +inf second moment alone is legal; no NaN or conversion error
        # occurs because the adjusted update remains exactly zero.
        if [[ "$failure_status" -ne 0 || "$(cat "$TMP/$mode.out")" != true || -s "$TMP/$mode.err" ]]; then
            echo "half optimizer valid infinity moment disagrees with the NaN-only contract (status=$failure_status):" >&2
            cat "$TMP/$mode.out" "$TMP/$mode.err" >&2
            exit 1
        fi
        continue
    fi
    if [[ "$failure_status" -ne 101 ]]; then
        echo "expected half optimizer $mode failure with status 101, got $failure_status" >&2
        cat "$TMP/$mode.err" >&2; exit 1
    fi
    if [[ "$mode" == conversion ]]; then
        grep -q 'NUMERIC_CONVERSION' "$TMP/$mode.err"
        grep -q 'tensor element' "$TMP/$mode.err"
    else
        if [[ "$MODE" == metal ]]; then
            grep -Fq 'Quidra runtime error[TENSOR]' "$TMP/$mode.err"
            grep -q 'nn.Adam.update produced a non-finite real32 value' "$TMP/$mode.err"
        else
            grep -q 'NONFINITE_RESULT' "$TMP/$mode.err"
        fi
    fi
    if grep -q 'NN_UPDATE_FAILED' "$TMP/$mode.err"; then cat "$TMP/$mode.err" >&2; exit 1; fi
done
