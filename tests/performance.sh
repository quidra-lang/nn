#!/usr/bin/env bash
# NN kernel timings (small = the first conv layer of an external training
# program, medium, large).
#
#   bash tests/performance.sh /path/to/quidra [sizes]
#
# sizes defaults to "small,medium"; add "large" (16x64x256x256) and "loss"
# (nn.cross_entropy forward+backward on 45 and 2880 rows) explicitly.
# The CPU always runs; GPU 0 runs when `quidra gpu` reports a real device.
# Set QUIDRA_NN_PERFORMANCE_CHECK=1 to fail when the small Metal/CUDA
# Conv2D forward+backward takes longer than 100 ms: native kernels stay far
# below that limit, while the portable gather/matmul fallback exceeds it.
# Set QUIDRA_NN_REQUIRE_NATIVE_LOSS_REDUCTIONS=1 (with "loss") to fail when
# the GPU loss costs 20 us or more per row, which means Math's last-axis
# reductions read rows back to the host one at a time instead of reducing
# on the device.
# Timings are reported, not compared, otherwise: they depend on the machine.
set -euo pipefail

QUIDRA="$1"
SIZES="${2:-small,medium}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

BINARY="$TMP/performance"
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" build "$ROOT/tests/performance.qui" -o "$BINARY" >/dev/null

devices=(-1)
set +e
gpu_info="$("$QUIDRA" gpu 2>&1)"
gpu_status=$?
set -e
if [[ $gpu_status -eq 0 ]] && grep -Fq "GPU 0" <<<"$gpu_info" &&
   ! grep -Fq "backend: TEST" <<<"$gpu_info"; then
    devices+=(0)
fi

for device in "${devices[@]}"; do
    label="cpu"
    [[ "$device" != "-1" ]] && label="gpu$device"
    output="$("$BINARY" --device "$device" --sizes "$SIZES")"
    while IFS= read -r line; do
        printf '[%s] %s\n' "$label" "$line"
    done <<<"$output"
    if [[ "${QUIDRA_NN_PERFORMANCE_CHECK:-0}" == "1" && "$device" != "-1" ]]; then
        small="$(grep '^small ' <<<"$output" || true)"
        if [[ -n "$small" ]]; then
            both="$(sed -E 's/.*conv2d forward\+backward ([0-9.]+) ms.*/\1/' <<<"$small")"
            if ! python3 -c "import sys; sys.exit(0 if float('$both') < 100.0 else 1)"; then
                echo "small Conv2D forward+backward on gpu$device took $both ms (limit 100 ms)" >&2
                exit 1
            fi
        fi
    fi
    if [[ "${QUIDRA_NN_REQUIRE_NATIVE_LOSS_REDUCTIONS:-0}" == "1" && "$device" != "-1" ]]; then
        loss="$(grep '^loss ' <<<"$output" || true)"
        if [[ -z "$loss" ]]; then
            echo "QUIDRA_NN_REQUIRE_NATIVE_LOSS_REDUCTIONS needs the loss size" >&2
            exit 1
        fi
        per_row="$(sed -E 's/.*per row (-?[0-9.]+) us.*/\1/' <<<"$loss")"
        if ! python3 -c "import sys; sys.exit(0 if float('$per_row') < 20.0 else 1)"; then
            echo "GPU cross_entropy costs $per_row us per row on gpu$device (limit 20 us): per-row host readback" >&2
            exit 1
        fi
    fi
done
echo "nn performance: ok"
