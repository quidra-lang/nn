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

class Pair()
    nn.Parameter<real32> weight
    nn.Parameter<real32> bias

Pair pair = Pair()
pair.weight = nn.Parameter<real32>(value = tensor.ones<real32>([2, 2]))
pair.bias = nn.Parameter<real32>(value = tensor.zeros<real32>([2]))

.nn.mode.fast()
.nn.mode.deterministic()
.nn.mode.fast()

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
nn.SGD<Pair> | error bad_sgd = nn.SGD(&pair, rate = 0.0)
match bad_sgd
    nn.SGD<Pair>
        sgd_rejected = false
    error
        sgd_rejected = true
print(sgd_rejected)
print(NL)

bool adam_rejected = false
nn.Adam<Pair> | error bad_adam = nn.Adam(&pair, beta1 = 1.0)
match bad_adam
    nn.Adam<Pair>
        adam_rejected = false
    error
        adam_rejected = true
print(adam_rejected)
print(NL)

real64 invalid_rate = -0.5

bool dropout_rate_rejected = false
nn.Dropout | error invalid_dropout = nn.Dropout(rate = invalid_rate)
match invalid_dropout
    nn.Dropout
        dropout_rate_rejected = false
    error
        dropout_rate_rejected = true
print(dropout_rate_rejected)
print(NL)

bool adam_decay_rejected = false
nn.Adam<Pair> | error invalid_adam = nn.Adam(&pair, beta1 = invalid_rate)
match invalid_adam
    nn.Adam<Pair>
        adam_decay_rejected = false
    error
        adam_decay_rejected = true
print(adam_decay_rejected)
print(NL)

bool initializer_bound_rejected = false
tensor<real32> | error bad_bound = nn.uniform_weights(
    count = 1, bound = invalid_rate, seed = 1
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

// Optimizers are bound at construction: a bound Parameter whose shape or
// device changed makes update() fail, before anything is updated, until the
// a new optimizer is constructed for the changed structure.
auto bound_sgd = nn.SGD(&pair, rate = 0.5)
auto bound_adam = nn.Adam(&pair, 0.1)
.pair.weight.replace(tensor.ones<real32>([3, 3]))
bool sgd_reshape_rejected = false
match .bound_sgd.update()
    void
        sgd_reshape_rejected = false
    error
        sgd_reshape_rejected = true
print(sgd_reshape_rejected)
print(NL)
bool adam_reshape_rejected = false
match .bound_adam.update()
    void
        adam_reshape_rejected = false
    error
        adam_reshape_rejected = true
print(adam_reshape_rejected and pair.weight.raw().shape()[0] == 3)
print(NL)
auto rebound_sgd = nn.SGD(&pair, rate = 0.5)
auto rebound_adam = nn.Adam(&pair, 0.1)
tensor<real32> rebound_weight = .pair.weight.track()
math.sum(rebound_weight * rebound_weight).backward(&pair)
// update() consumes the gradients it used.
.rebound_adam.update()
print(pair.weight.raw()[0, 0].item() < real32(1) and not pair.weight.has_grad())
print(NL)
QUI

actual="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/contracts.qui")"
expected="$(printf 'true\n%.0s' {1..16})"
if [[ "$actual" != "$expected" ]]; then
    echo "unexpected NN contract output:" >&2
    printf '%s\n' "$actual" >&2
    exit 1
fi

# update() changes the optimizer's state (SGD's slots for the next values,
# Adam's moments), so a const optimizer cannot call it.
cat > "$TMP/const-sgd-update.qui" <<'QUI'
import nn

class Model()
    nn.Parameter<real32> weight

Model model = Model()
model.weight = nn.Parameter<real32>(value = tensor.ones<real32>([1]))
const auto optimizer = nn.SGD(&model)
.optimizer.update()
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/const-sgd-update.qui" --json >"$TMP/const-sgd-update.out" 2>&1
status=$?
set -e
if [[ "$status" -ne 1 ]] || ! grep -Fq 'WRITE_CAPABILITY' "$TMP/const-sgd-update.out"; then
    echo "const NN optimizer unexpectedly allowed mutable update()" >&2
    cat "$TMP/const-sgd-update.out" >&2
    exit 1
fi

cat > "$TMP/const-adam-update.qui" <<'QUI'
import nn

class Model()
    nn.Parameter<real32> weight

Model model = Model()
model.weight = nn.Parameter<real32>(value = tensor.ones<real32>([1]))
const auto optimizer = nn.Adam(&model)
.optimizer.update()
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/const-adam-update.qui" --json >"$TMP/const-adam-update.out" 2>&1
status=$?
set -e
if [[ "$status" -ne 1 ]] || ! grep -Fq 'WRITE_CAPABILITY' "$TMP/const-adam-update.out"; then
    echo "const NN optimizer unexpectedly allowed mutable update()" >&2
    cat "$TMP/const-adam-update.out" >&2
    exit 1
fi

# The update API is update() and nn.clear_grad(&model); zero_grad() and
# step() are gone.
cat > "$TMP/removed-optimizer-api.qui" <<'QUI'
import nn

class Model()
    nn.Parameter<real32> weight

Model model = Model()
model.weight = nn.Parameter<real32>(value = tensor.ones<real32>([1]))
auto adam = nn.Adam(&model)
auto sgd = nn.SGD(&model)
adam.zero_grad()
adam.step()
sgd.zero_grad()
sgd.step()
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/removed-optimizer-api.qui" --json >"$TMP/removed-optimizer-api.out" 2>&1
status=$?
set -e
if [[ "$status" -ne 1 ]] || [[ "$(grep -o 'UNKNOWN_MEMBER' "$TMP/removed-optimizer-api.out" | wc -l | tr -d ' ')" != 4 ]]; then
    echo "NN optimizers still accept zero_grad() or step()" >&2
    cat "$TMP/removed-optimizer-api.out" >&2
    exit 1
fi

# Machine-readable NN errors retain their specific human-oriented messages.
cat > "$TMP/error-codes.qui" <<'QUI'
import nn

class Model()
    nn.Parameter<real32> weight

Model model = Model()
model.weight = nn.Parameter<real32>(value = tensor.ones<real32>([1]))
nn.SGD<Model> | error invalid_rate = nn.SGD(&model, rate = real64(-1.0))
match invalid_rate
    nn.SGD<Model>
        print(false)
    error problem
        print(problem.code == "NN_ARGUMENT")
print(NL)
QUI
codes_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/error-codes.qui")"
if [[ "$codes_output" != "true" ]]; then
    printf 'NN error code mismatch: %s\n' "$codes_output" >&2
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


# Schema inspection borrows the model, but a selected leaf must be initialized.
# These are compile-only cases: load's file I/O is never executed here.
cat > "$TMP/state-leaves-initialized.qui" <<'QUI'
import nn

class Model()
    nn.Parameter<real32> weight = nn.Parameter<real32>(value = tensor.ones<real32>([1]))
    nn.State<real32> running = nn.State<real32>(value = tensor.zeros<real32>([1]))

Model model = Model()
nn.load(&model, "unused-state-file")
QUI
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/state-leaves-initialized.qui" --json >"$TMP/state-leaves-initialized.out" 2>&1
cat > "$TMP/state-leaf-uninitialized-parameter.qui" <<'QUI'
import nn

class Model()
    nn.Parameter<real32> weight

Model model = Model()
nn.load(&model, "unused-state-file")
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/state-leaf-uninitialized-parameter.qui" --json >"$TMP/state-leaf-uninitialized-parameter.out" 2>&1
status=$?
set -e
if [[ "$status" -ne 1 ]] || ! grep -Eq '"code"[[:space:]]*:[[:space:]]*"UNINITIALIZED"' "$TMP/state-leaf-uninitialized-parameter.out"; then
    echo "NN state schema accepted an uninitialized parameter leaf" >&2
    cat "$TMP/state-leaf-uninitialized-parameter.out" >&2
    exit 1
fi
cat > "$TMP/state-leaf-uninitialized-state.qui" <<'QUI'
import nn

class Model()
    nn.State<real32> running

Model model = Model()
nn.load(&model, "unused-state-file")
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/state-leaf-uninitialized-state.qui" --json >"$TMP/state-leaf-uninitialized-state.out" 2>&1
status=$?
set -e
if [[ "$status" -ne 1 ]] || ! grep -Eq '"code"[[:space:]]*:[[:space:]]*"UNINITIALIZED"' "$TMP/state-leaf-uninitialized-state.out"; then
    echo "NN state schema accepted an uninitialized state leaf" >&2
    cat "$TMP/state-leaf-uninitialized-state.out" >&2
    exit 1
fi

# Prepared generation retains supplied state and rejects incomplete configuration.
cat > "$TMP/prepared-layer-generation.qui" <<'QUI'
import nn

nn.Parameter<real32> weight_fc = nn.Parameter<real32>(value = tensor.ones<real32>([1, 2]) * real32(3.0))
nn.Parameter<real32> bias_fc = nn.Parameter<real32>(value = tensor.ones<real32>([1]) * real32(4.0))
nn.FC dense = nn.FC(2, 1, initial_weight = weight_fc, initial_bias = bias_fc)
print(dense.weight.raw()[0, 0].item() == real32(3.0) and dense.bias.raw()[0].item() == real32(4.0))
print(NL)

nn.Parameter<real32> weight_conv = nn.Parameter<real32>(value = tensor.ones<real32>([1, 1, 1, 1]) * real32(2.0))
nn.Parameter<real32> bias_conv = nn.Parameter<real32>(value = tensor.ones<real32>([1]) * real32(3.0))
nn.Conv2D convolution = nn.Conv2D(1, 1, 1, stride = 2, padding = 1, initial_weight = weight_conv, initial_bias = bias_conv)
print(convolution.weight.raw()[0, 0, 0, 0].item() == real32(2.0) and convolution.bias.raw()[0].item() == real32(3.0) and convolution.step == 2 and convolution.border == 1 and convolution.group_count == 1)
print(NL)

nn.Parameter<real32> scale_norm = nn.Parameter<real32>(value = tensor.ones<real32>([1]) * real32(5.0))
nn.Parameter<real32> bias_norm = nn.Parameter<real32>(value = tensor.ones<real32>([1]) * real32(6.0))
nn.State<real32> mean_norm = nn.State<real32>(value = tensor.ones<real32>([1]) * real32(7.0))
nn.State<real32> variance_norm = nn.State<real32>(value = tensor.ones<real32>([1]) * real32(8.0))
nn.BatchNorm normalization = nn.BatchNorm(1, initial_scale = scale_norm, initial_bias = bias_norm, initial_mean = mean_norm, initial_variance = variance_norm, initial_momentum = real32(0.25), initial_epsilon = real32(0.125))
print(normalization.scale.raw()[0].item() == real32(5.0) and normalization.bias.raw()[0].item() == real32(6.0) and normalization.running_mean.raw()[0].item() == real32(7.0) and normalization.running_variance.raw()[0].item() == real32(8.0) and normalization.running_momentum == real32(0.25) and normalization.variance_epsilon == real32(0.125))
print(NL)

bool rejected_fc = false
match nn.FC(2, 1, initial_weight = weight_fc)
    nn.FC
        rejected_fc = false
    error problem
        rejected_fc = problem.code == "NN_ARGUMENT"
print(rejected_fc)
print(NL)
bool rejected_conv = false
match nn.Conv2D(1, 1, 1, initial_bias = bias_conv)
    nn.Conv2D
        rejected_conv = false
    error problem
        rejected_conv = problem.code == "NN_ARGUMENT"
print(rejected_conv)
print(NL)
bool rejected_norm = false
match nn.BatchNorm(1, initial_scale = scale_norm)
    nn.BatchNorm
        rejected_norm = false
    error problem
        rejected_norm = problem.code == "NN_ARGUMENT"
print(rejected_norm)
print(NL)

bool rejects_settings(real32 configured_momentum, real32 configured_epsilon)
    nn.Parameter<real32> prepared_scale = nn.Parameter<real32>(value = tensor.ones<real32>([1]))
    nn.Parameter<real32> prepared_bias = nn.Parameter<real32>(value = tensor.zeros<real32>([1]))
    nn.State<real32> prepared_mean = nn.State<real32>(value = tensor.zeros<real32>([1]))
    nn.State<real32> prepared_variance = nn.State<real32>(value = tensor.ones<real32>([1]))
    match nn.BatchNorm(1, initial_scale = prepared_scale, initial_bias = prepared_bias, initial_mean = prepared_mean, initial_variance = prepared_variance, initial_momentum = configured_momentum, initial_epsilon = configured_epsilon)
        nn.BatchNorm
            return false
        error problem
            return problem.code == "NN_ARGUMENT"
print(rejects_settings(real32(1.0), real32(0.125)))
print(NL)
print(rejects_settings(real32(-0.25), real32(0.125)))
print(NL)
print(rejects_settings(real32(0.25), real32(0.0)))
print(NL)
QUI
prepared_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/prepared-layer-generation.qui")"
if [[ "$prepared_output" != $'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue' ]]; then
    printf 'NN prepared generation mismatch: %s\n' "$prepared_output" >&2
    exit 1
fi

echo "nn contracts: ok"
