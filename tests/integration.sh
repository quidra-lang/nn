#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/use-nn.qui" <<'QUI'
import nn
import math

class Model
    nn.FC dense

int | error run()
    nn.FC dense = try nn.FC(features_in = 2, features_out = 1)
    Model model
    model.dense = dense
    tensor<real32> samples = tensor.ones<real32>([1, 2])
    tensor<real32> targets = tensor.zeros<real32>([1, 1])
    tensor<real32> prediction = nn.relu(model.dense.forward(samples.track()))
    tensor<real32> loss = nn.mse(prediction, targets)
    nn.SGD optimizer = try nn.SGD(rate = 0.1)
    optimizer.zero_grad(&model)
    loss.backward(&model)
    optimizer.step(&model)
    print(prediction.shape()[0] == 1)
    print(NL)
    print(prediction.shape()[1] == 1)
    print(NL)

    nn.Conv2D conv = try nn.Conv2D(1, 1, 1)
    conv.weight.replace(tensor.ones<real32>([1, 1, 1, 1]))
    conv.bias.replace(tensor.zeros<real32>([1]))
    tensor<real32> image = tensor.ones<real32>([1, 1, 2, 2])
    tensor<real32> activated = nn.relu(conv.forward(image))
    int[] activated_shape = activated.shape()
    print(
        len(activated_shape) == 4
        and activated_shape[0] == 1
        and activated_shape[1] == 1
        and activated_shape[2] == 2
        and activated_shape[3] == 2
    )
    print(NL)

    nn.BatchNorm norm = try nn.BatchNorm(1)
    tensor<real32> normalized = norm.infer(activated)
    int[] normalized_shape = normalized.shape()
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

actual="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/use-nn.qui")"
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

ir="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" ir "$TMP/use-nn.qui")"
if ! grep -Fq "compiler-extension nn.graph" <<< "$ir"; then
    echo "NN compiler extension was not registered" >&2
    exit 1
fi

echo "nn integration: ok"
