#!/usr/bin/env bash
set -euo pipefail
QUIDRA="$1"
DEVICE="${2:-0}"
OTHER="${3:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_PACKAGE_PATH="$(dirname "$ROOT")"
export QUIDRA_CACHE_DIR="$TMP/cache"
# Requires a working CUDA/cuDNN device and a second NCCL-capable device.
# A refusal or portable fallback does not stand in for the native check.
for mode in activation convolution collective direct-collective; do
    args=()
    operation=nn.activation
    if [[ "$mode" == convolution ]]; then args=(--convolution); operation=nn.Conv2D.forward; fi
    if [[ "$mode" == collective ]]; then args=(--collective); operation=nn.all_reduce_sum; fi
    if [[ "$mode" == direct-collective ]]; then args=(--collective --direct); operation=nn.all_reduce_sum; fi
    if "$QUIDRA" "$ROOT/tests/nonfinite_cuda.qui" --device "$DEVICE" --other "$OTHER" "${args[@]}" > "$TMP/$mode.out" 2> "$TMP/$mode.err"; then
        echo "expected native CUDA $mode failure" >&2; exit 1
    fi
    grep -q 'GPU_ASYNC' "$TMP/$mode.err"
    grep -Fq "package operation $operation produced a non-finite real32 value" "$TMP/$mode.err"
    if grep -q 'unexpected' "$TMP/$mode.out"; then cat "$TMP/$mode.out" >&2; exit 1; fi
done
