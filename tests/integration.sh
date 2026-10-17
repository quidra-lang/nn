#!/usr/bin/env bash
set -euo pipefail

report_failure() {
    if [[ $- == *e* ]]; then
        printf 'NN integration failed at line %s: %s\n' "$1" "$2" >&2
    fi
}
trap 'report_failure "$LINENO" "$BASH_COMMAND"' ERR

QUIDRA="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"

cat > "$TMP/use-nn.qui" <<'QUI'
import nn
import math

class Model()
    nn.FC dense

int | error run()
    nn.FC dense = try nn.FC(features_in = 2, features_out = 1)
    Model model = Model()
    model.dense = dense
    tensor<real32> samples = tensor.ones<real32>([1, 2])
    tensor<real32> targets = tensor.zeros<real32>([1, 1])
    tensor<real32> prediction = nn.relu(.model.dense.forward(samples.track()))
    tensor<real32> loss = nn.mse(prediction, targets)
    auto optimizer = try nn.SGD(&model, rate = 0.1)
    loss.backward(&model)
    .optimizer.update()
    print(prediction.shape()[0] == 1)
    print(NL)
    print(prediction.shape()[1] == 1)
    print(NL)

    nn.Conv2D conv = try nn.Conv2D(1, 1, 1)
    .conv.weight.replace(tensor.ones<real32>([1, 1, 1, 1]))
    .conv.bias.replace(tensor.zeros<real32>([1]))
    tensor<real32> image = tensor.ones<real32>([1, 1, 2, 2])
    tensor<real32> activated = nn.relu(.conv.forward(image))
    nat[] activated_shape = activated.shape()
    print(
        len(activated_shape) == 4
        and activated_shape[0] == 1
        and activated_shape[1] == 1
        and activated_shape[2] == 2
        and activated_shape[3] == 2
    )
    print(NL)

    nn.BatchNorm norm = try nn.BatchNorm(1)
    tensor<real32> normalized = .norm.infer(activated)
    nat[] normalized_shape = normalized.shape()
    print(
        len(normalized_shape) == 4
        and normalized_shape[0] == 1
        and normalized_shape[1] == 1
        and normalized_shape[2] == 2
        and normalized_shape[3] == 2
    )
    print(NL)

    nn.Dropout dropout = try nn.Dropout(0.5, seed = 1)
    tensor<real32> inference = dropout.infer(activated)
    print(inference[0, 0, 0, 0].item() == activated[0, 0, 0, 0].item())
    print(NL)
    return 0

auto | error result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
        print(NL)
QUI

if actual="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/use-nn.qui")"; then
    :
else
    status=$?
    if [[ "$(uname -s)" == Linux ]] && command -v gdb >/dev/null 2>&1; then
        QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" gdb --batch \
            -ex 'catch throw std::bad_alloc' -ex run -ex bt \
            --args "$QUIDRA" "$TMP/use-nn.qui" >&2 || true
    elif [[ "$(uname -s)" == Linux ]] && command -v c++ >/dev/null 2>&1; then
        cat > "$TMP/exception-stack.cpp" <<'CPP'
#include <cstdlib>
#include <dlfcn.h>
#include <execinfo.h>
#include <new>
#include <typeinfo>
extern "C" [[noreturn]] void __cxa_throw(void* exception, void* information,
                                        void (*destroy)(void*)) {
    const auto* type = static_cast<const std::type_info*>(information);
    if (type && *type == typeid(std::bad_alloc)) {
        void* frames[64];
        const int count = backtrace(frames, 64);
        backtrace_symbols_fd(frames, count, 2);
    }
    using Throw = void (*)(void*, void*, void (*)(void*));
    const auto original = reinterpret_cast<Throw>(dlsym(RTLD_NEXT, "__cxa_throw"));
    if (original) original(exception, information, destroy);
    std::abort();
}
CPP
        if c++ -std=c++20 -shared -fPIC -O0 "$TMP/exception-stack.cpp" \
            -o "$TMP/exception-stack.so" -ldl; then
            QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" LD_PRELOAD="$TMP/exception-stack.so" \
                "$QUIDRA" "$TMP/use-nn.qui" >/dev/null 2>"$TMP/exception-stack.err" || true
            cat "$TMP/exception-stack.err" >&2
            if command -v addr2line >/dev/null 2>&1; then
                sed -n 's|.*quidra(+\(0x[0-9a-fA-F]*\)).*|\1|p' \
                    "$TMP/exception-stack.err" | \
                    xargs -r addr2line -C -f -e "$QUIDRA" >&2 || true
            fi
        fi
    fi
    exit "$status"
fi
expected="$(printf 'true\n%.0s' {1..5})"
if [[ "$actual" != "$expected" ]]; then
    echo "unexpected NN integration output:" >&2
    printf '%s\n' "$actual" >&2
    exit 1
fi

# The same package-native program must survive AOT linking and REPL/JIT
# loading. Conv2D.forward exercises NN's native extension ABI rather than only
# package source composition, so these guard the two remaining execution paths.
AOT="$TMP/use-nn-aot"
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" build "$TMP/use-nn.qui" -o "$AOT"
aot_output="$("$AOT")"
if [[ "$aot_output" != "$expected" ]]; then
    echo "unexpected NN AOT output:" >&2
    printf '%s\n' "$aot_output" >&2
    exit 1
fi

repl_output="$(
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" repl < "$TMP/use-nn.qui"
)"
true_lines="$(grep -Fxc "true" <<< "$repl_output" || true)"
if [[ "$true_lines" -lt 5 ]]; then
    echo "NN package-native REPL/JIT load did not execute expected operations:" >&2
    printf '%s\n' "$repl_output" >&2
    exit 1
fi

# NN's fused CPU activation kernels reproduce the package source
# expressions exactly and carry first- and second-order autograd.
activation_output="$(
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" \
    "$QUIDRA" "$ROOT/tests/native_activations.qui"
)"
activation_expected_count="$(grep -c '^ *report("' "$ROOT/tests/native_activations.qui")"
activation_passed_count="$(grep -c ' true$' <<< "$activation_output" || true)"
if [[ "$activation_passed_count" -ne "$activation_expected_count" ]] ||
   grep -Fq ' false' <<< "$activation_output"; then
    echo "NN fused activation equivalence failed:" >&2
    printf '%s\n' "$activation_output" >&2
    exit 1
fi

# Signed infinity is valid; 0/0 and indeterminate activation results are not.
# Run each case independently so a failure cannot mask later edge cases.
for exceptional_case in {1..20}; do
    set +e
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$ROOT/tests/native_activations.qui" \
        --exceptional_case "$exceptional_case" > "$TMP/activation-invalid.out" 2> "$TMP/activation-invalid.err"
    activation_invalid_status=$?
    set -e
    if [[ "$exceptional_case" -le 2 ]]; then
        expected_inf="inf"
        [[ "$exceptional_case" != 2 ]] || expected_inf="-inf"
        if [[ "$activation_invalid_status" -ne 0 || "$(cat "$TMP/activation-invalid.out")" != "$expected_inf" || -s "$TMP/activation-invalid.err" ]]; then
            echo "NN infinity activation case $exceptional_case was not preserved:" >&2
            cat "$TMP/activation-invalid.out" "$TMP/activation-invalid.err" >&2
            exit 1
        fi
        continue
    fi
    if [[ "$activation_invalid_status" -ne 101 || -s "$TMP/activation-invalid.out" ]]; then
        echo "NN NaN activation case $exceptional_case did not fail with status 101 and empty output" >&2
        cat "$TMP/activation-invalid.err" >&2
        exit 1
    fi
    if [[ "$exceptional_case" -eq 3 ]]; then
        grep -q 'DIVISION_BY_ZERO' "$TMP/activation-invalid.err"
    else
        grep -q 'NONFINITE_RESULT' "$TMP/activation-invalid.err"
    fi
done

# nn.softmax/nn.cross_entropy semantics against closed forms.
loss_output="$(
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" \
    "$QUIDRA" "$ROOT/tests/losses.qui"
)"
loss_expected_count="$(grep -c '^ *report("' "$ROOT/tests/losses.qui")"
loss_passed_count="$(grep -c ' true$' <<< "$loss_output" || true)"
if [[ "$loss_passed_count" -ne "$loss_expected_count" ]] ||
   grep -Fq ' false' <<< "$loss_output"; then
    echo "NN loss semantics failed:" >&2
    printf '%s\n' "$loss_output" >&2
    exit 1
fi

# Optimizer updates consume accumulated gradients, are atomic, and update a
# keeps copied Parameters independent; nn.clear_grad discards gradients.
optimizer_output="$(
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" \
    "$QUIDRA" "$ROOT/tests/optimizer.qui" --scratch "$TMP"
)"
optimizer_expected_count="$(grep -c '^ *report("' "$ROOT/tests/optimizer.qui")"
optimizer_passed_count="$(grep -c ' true$' <<< "$optimizer_output" || true)"
if [[ "$optimizer_passed_count" -ne "$optimizer_expected_count" ]] ||
   grep -Fq ' false' <<< "$optimizer_output"; then
    echo "NN optimizer update semantics failed:" >&2
    printf '%s\n' "$optimizer_output" >&2
    exit 1
fi

extension_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$ROOT/tests/optimizer_extensions.qui")"
extension_passed_count="$(grep -c ' true$' <<< "$extension_output" || true)"
if [[ "$extension_passed_count" -ne 7 ]] || grep -Fq ' false' <<< "$extension_output"; then
    echo "NN extended optimizer semantics failed:" >&2
    printf '%s\n' "$extension_output" >&2
    exit 1
fi

# The independent persistence oracle includes all four storage dtypes,
# published legacy readers, exact format layout and transactional failures.
python3 "$ROOT/tests/checkpoint_format5.py" "$QUIDRA"
failure_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$ROOT/tests/optimizer_computation_failure.qui" "$TMP")"
failure_expected="$(printf 'computation error contract true true\nfresh optimizer resumes training true')"
if [[ "$failure_output" != "$failure_expected" ]]; then
    printf '%s\n' "$failure_output" >&2
    exit 1
fi

# NN's CPU Conv2D kernels agree with the portable graph and with finite
# differences across stride/padding/group configurations.
conv_output="$(
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" \
    "$QUIDRA" "$ROOT/tests/native_conv.qui"
)"
conv_expected_count="$(grep -c '^ *report("' "$ROOT/tests/native_conv.qui")"
conv_passed_count="$(grep -c ' true$' <<< "$conv_output" || true)"
if [[ "$conv_passed_count" -ne "$conv_expected_count" ]] ||
   grep -Fq ' false' <<< "$conv_output"; then
    echo "NN CPU Conv2D equivalence failed:" >&2
    printf '%s\n' "$conv_output" >&2
    exit 1
fi

# NN's native global average pooling matches the former compositional graph
# (values and three orders of autograd).
pooling_output="$(
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" \
    "$QUIDRA" "$ROOT/tests/native_pooling.qui"
)"
pooling_expected_count="$(grep -c '^ *report("' "$ROOT/tests/native_pooling.qui")"
pooling_passed_count="$(grep -c ' true$' <<< "$pooling_output" || true)"
if [[ "$pooling_passed_count" -ne "$pooling_expected_count" ]] ||
   grep -Fq ' false' <<< "$pooling_output"; then
    echo "NN native pooling equivalence failed:" >&2
    printf '%s\n' "$pooling_output" >&2
    exit 1
fi

# NN's CPU kernels split independent outputs across Core's CPU threads; the
# results must be bitwise the same for every thread count.
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" build "$ROOT/tests/cpu_threads.qui" \
    -o "$TMP/cpu-threads"
QUIDRA_CPU_THREADS=1 "$TMP/cpu-threads" "$TMP/cpu-threads-1.qnn"
QUIDRA_CPU_THREADS=3 "$TMP/cpu-threads" "$TMP/cpu-threads-3.qnn"
"$TMP/cpu-threads" "$TMP/cpu-threads-all.qnn"
if ! cmp -s "$TMP/cpu-threads-1.qnn" "$TMP/cpu-threads-3.qnn" ||
   ! cmp -s "$TMP/cpu-threads-1.qnn" "$TMP/cpu-threads-all.qnn"; then
    echo "NN CPU kernels depend on the thread count" >&2
    exit 1
fi

ir="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" ir "$TMP/use-nn.qui")"
if ! grep -Fq "compiler-extension nn.graph" <<< "$ir"; then
    echo "NN compiler extension was not registered" >&2
    exit 1
fi

echo "nn integration: ok"

# NN-owned finite validation and low precision optimizer state.
bash "$ROOT/tests/half_optimizer.sh" "$QUIDRA" cpu
bash "$ROOT/tests/nonfinite_outputs.sh" "$QUIDRA" -1
