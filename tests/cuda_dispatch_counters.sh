#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORE_INCLUDE="${1:?usage: cuda_dispatch_counters.sh /path/to/core/include}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
link_flags=(-std=c++20)
if [[ "$(uname -s)" == Linux ]]; then link_flags+=(-ldl); fi
"${CXX:-c++}" -I "$CORE_INCLUDE" "$ROOT/tests/cuda_dispatch_counters.cpp" \
    "${link_flags[@]}" -o "$TMP/cuda-dispatch-counters"
"$TMP/cuda-dispatch-counters"
native_flags=(-std=c++20 -O1 -ffunction-sections -fdata-sections)
if [[ "$(uname -s)" == Darwin ]]; then
    native_flags+=(-Wl,-dead_strip)
else
    native_flags+=(-Wl,--gc-sections -ldl)
fi
"${CXX:-c++}" -I "$CORE_INCLUDE" "$ROOT/tests/nccl_dispatch_counters.cpp" \
    "${native_flags[@]}" -o "$TMP/nccl-dispatch-counters"
"$TMP/nccl-dispatch-counters"
