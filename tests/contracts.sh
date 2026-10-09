#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"

cat > "$TMP/contracts.qui" <<'QUI'
import nn
import math

nn.mode.fast()
nn.mode.deterministic()
nn.mode.fast()

bool fc_rejected = false
nn.FC | error bad_fc = nn.FC(features_in = 2, features_out = 0)
match bad_fc
    nn.FC
        fc_rejected = false
    error
        fc_rejected = true
print(fc_rejected)
print(NL)

bool conv_rejected = false
nn.Conv2D | error bad_conv = nn.Conv2D(
    channels_in = 1,
    channels_out = 1,
    kernel = 3,
    stride = 0
)
match bad_conv
    nn.Conv2D
        conv_rejected = false
    error
        conv_rejected = true
print(conv_rejected)
print(NL)

bool norm_rejected = false
nn.BatchNorm | error bad_norm = nn.BatchNorm(features = 2, momentum = 1.5)
match bad_norm
    nn.BatchNorm
        norm_rejected = false
    error
        norm_rejected = true
print(norm_rejected)
print(NL)

bool dropout_rejected = false
nn.Dropout | error bad_dropout = nn.Dropout(rate = 1.0)
match bad_dropout
    nn.Dropout
        dropout_rejected = false
    error
        dropout_rejected = true
print(dropout_rejected)
print(NL)

bool sgd_rejected = false
nn.SGD | error bad_sgd = nn.SGD(rate = 0.0)
match bad_sgd
    nn.SGD
        sgd_rejected = false
    error
        sgd_rejected = true
print(sgd_rejected)
print(NL)

bool adam_rejected = false
nn.Adam | error bad_adam = nn.Adam(beta1 = 1.0)
match bad_adam
    nn.Adam
        adam_rejected = false
    error
        adam_rejected = true
print(adam_rejected)
print(NL)

real64 zero = 0.0
real64 nan_value = zero / zero
real64 infinity = 1.0 / zero

bool dropout_nan_rejected = false
nn.Dropout | error nan_dropout = nn.Dropout(rate = nan_value)
match nan_dropout
    nn.Dropout
        dropout_nan_rejected = false
    error
        dropout_nan_rejected = true
print(dropout_nan_rejected)
print(NL)

bool adam_nan_rejected = false
nn.Adam | error nan_adam = nn.Adam(beta1 = nan_value)
match nan_adam
    nn.Adam
        adam_nan_rejected = false
    error
        adam_nan_rejected = true
print(adam_nan_rejected)
print(NL)

bool initializer_bound_rejected = false
tensor<real32> | error bad_bound = nn.uniform_weights(
    count = 1, bound = infinity, seed = 1
)
match bad_bound
    tensor<real32>
        initializer_bound_rejected = false
    error
        initializer_bound_rejected = true
print(initializer_bound_rejected)
print(NL)

nn.FC dense = nn.normal_fc(
    features_in = 2,
    features_out = 1,
    standard_deviation = 0.0,
    bias_value = 1.0,
    seed = 5
)
print(dense.weight.raw()[0, 0].item() == real32(0))
print(NL)
print(dense.bias.raw()[0].item() == real32(1))
print(NL)

nn.Conv2D convolution = nn.normal_conv2d(
    channels_in = 1,
    channels_out = 1,
    kernel = 1,
    standard_deviation = 0.0,
    bias_value = 1.0,
    seed = 5
)
print(convolution.weight.raw()[0, 0, 0, 0].item() == real32(0))
print(NL)
print(convolution.bias.raw()[0].item() == real32(1))
print(NL)
QUI

actual="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/contracts.qui")"
expected="$(printf 'true\n%.0s' {1..13})"
if [[ "$actual" != "$expected" ]]; then
    echo "unexpected NN contract output:" >&2
    printf '%s\n' "$actual" >&2
    exit 1
fi

cat > "$TMP/const-sgd-step.qui" <<'QUI'
import nn

class Model
    nn.Parameter<real32> weight

Model model
model.weight = nn.Parameter<real32>(value = tensor.ones<real32>([1]))
const nn.SGD optimizer = nn.SGD()
optimizer.step(&model)
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/const-sgd-step.qui" --json >"$TMP/const-sgd-step.out" 2>&1
status=$?
set -e
if [[ "$status" -ne 1 ]] || ! grep -Fq 'WRITE_CAPABILITY' "$TMP/const-sgd-step.out"; then
    echo "const NN optimizer unexpectedly allowed mutable step()" >&2
    cat "$TMP/const-sgd-step.out" >&2
    exit 1
fi

# The library-wide clear operation discards both real32 and real64 gradient
# state, and remains safe to repeat when gradients are absent.
cat > "$TMP/clear-grad.qui" <<'QUI'
import nn
import math

class Model
    nn.Parameter<real32> weight
    nn.Parameter<real64> bias

Model model
model.weight = nn.Parameter<real32>(value = tensor.ones<real32>([1]))
model.bias = nn.Parameter<real64>(value = tensor.ones<real64>([1]))
math.sum(model.weight.track()).backward(&model)
math.sum(model.bias.track()).backward(&model)
print(model.weight.has_grad() and model.bias.has_grad())
print(NL)
nn.clear_grad(&model)
print(not model.weight.has_grad() and not model.bias.has_grad())
print(NL)
nn.clear_grad(&model)
print(not model.weight.has_grad() and not model.bias.has_grad())
print(NL)
QUI
clear_grad_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/clear-grad.qui")"
if [[ "$clear_grad_output" != "$(printf 'true\ntrue\ntrue')" ]]; then
    echo "unexpected nn.clear_grad output:" >&2
    printf '%s\n' "$clear_grad_output" >&2
    exit 1
fi

cat > "$TMP/private-bridge.qui" <<'QUI'
import nn
nn.nn_runtime_fast()
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/private-bridge.qui" --json >"$TMP/private-bridge.out" 2>&1
status=$?
set -e
if [[ "$status" -ne 1 ]] || ! grep -Eq 'UNKNOWN_MODULE_MEMBER|UNKNOWN_NAME' "$TMP/private-bridge.out"; then
    echo "NN runtime bridge leaked through the public API" >&2
    cat "$TMP/private-bridge.out" >&2
    exit 1
fi

echo "nn contracts: ok"
