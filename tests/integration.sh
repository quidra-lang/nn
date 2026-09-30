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
    tensor<float32> samples = tensor.ones<float32>([1, 2])
    tensor<float32> targets = tensor.zeros<float32>([1, 1])
    tensor<float32> prediction = nn.relu(model.dense.forward(samples.track()))
    tensor<float32> loss = nn.mse(prediction, targets)
    nn.SGD optimizer = try nn.SGD(rate = 0.1)
    optimizer.zero_grad(&model)
    loss.backward(&model)
    optimizer.step(&model)
    print(prediction.shape()[0] == 1)
    print(NL)
    print(prediction.shape()[1] == 1)
    print(NL)

    nn.Conv2D conv = try nn.Conv2D(1, 1, 1)
    conv.weight.replace(tensor.ones<float32>([1, 1, 1, 1]))
    conv.bias.replace(tensor.zeros<float32>([1]))
    tensor<float32> image = tensor.ones<float32>([1, 1, 2, 2])
    tensor<float32> activated = nn.relu(conv.forward(image))
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
    tensor<float32> normalized = norm.infer(activated)
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
    tensor<float32> inference = dropout.infer(activated)
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

ir="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" ir "$TMP/use-nn.qui")"
if ! grep -Fq "compiler-extension nn.graph" <<< "$ir"; then
    echo "NN compiler extension was not registered" >&2
    exit 1
fi

echo "nn integration: ok"
