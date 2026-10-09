#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"

# GPU stage probes are compile-only and use Core's domain-neutral test backend.
export QUIDRA_TEST_FAKE_GPU_COUNT=2

cat > "$TMP/fusion.qui" <<'QUI'
import nn

int | error run()
    nn.Conv2D layer = try nn.Conv2D(
        channels_in = 1,
        channels_out = 1,
        kernel = 1
    )
    layer.weight.replace(tensor.ones<real32>([1, 1, 1, 1]))
    tensor<real32> bias = tensor.zeros<real32>([1])
    bias[0] = real32(-2)
    layer.bias.replace(bias)
    tensor<real32> input = tensor.ones<real32>([1, 1, 1, 1])
    tensor<real32> output = nn.relu(layer.forward(input))
    print(output.shape()[1] == 1)
    print(NL)
    print(output[0, 0, 0, 0].item() == real32(0))
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

expected="$(printf 'true\ntrue')"
actual="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/fusion.qui")"
if [[ "$actual" != "$expected" ]]; then
    echo "NN fusion execution changed program behavior:" >&2
    printf '%s\n' "$actual" >&2
    exit 1
fi

fusion_ir="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" ir "$TMP/fusion.qui")"
fusion_body="$(awk '/^function run\(/,/^end$/' <<< "$fusion_ir")"
if ! grep -Fq "Conv2D.forward_relu" <<< "$fusion_body"; then
    echo "NN Conv2D/ReLU fusion did not replace executable IR" >&2
    printf '%s\n' "$fusion_body" >&2
    exit 1
fi

aot="$TMP/fusion-aot"
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" build "$TMP/fusion.qui" -o "$aot"
aot_output="$("$aot")"
if [[ "$aot_output" != "$expected" ]]; then
    echo "NN package-native AOT fusion changed program behavior:" >&2
    printf '%s\n' "$aot_output" >&2
    exit 1
fi

repl_output="$(
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" repl < "$TMP/fusion.qui"
)"
if ! grep -Fxq "true" <<< "$repl_output"; then
    echo "NN package-native REPL/JIT fusion did not execute" >&2
    printf '%s\n' "$repl_output" >&2
    exit 1
fi

cat > "$TMP/stages.qui" <<'QUI'
import nn

int | error conv_cpu_probe()
    nn.Conv2D layer = try nn.Conv2D(1, 1, 1)
    tensor<real32> output = layer.forward(
        tensor.ones<real32>([1, 1, 1, 1])
    )
    int ignored = output.shape()[0]
    return 0

int | error conv_gpu_probe()
    nn.Conv2D layer = try nn.Conv2D(1, 1, 1)
    tensor<real32> output = layer.forward(
        tensor.ones<real32>([1, 1, 1, 1], gpu = 0)
    )
    int ignored = output.shape()[0]
    return 0

int | error conv_gpu_fast_probe()
    nn.Conv2D layer = try nn.Conv2D(1, 1, 1)
    tensor<real32> input = tensor.ones<real32>([1, 1, 1, 1], gpu = 0)
    nn.mode.fast()
    tensor<real32> output = layer.forward(input)
    int ignored = output.shape()[0]
    return 0

int | error conv_gpu_deterministic_probe()
    nn.Conv2D layer = try nn.Conv2D(1, 1, 1)
    tensor<real32> input = tensor.ones<real32>([1, 1, 1, 1], gpu = 0)
    nn.mode.deterministic()
    tensor<real32> output = layer.forward(input)
    int ignored = output.shape()[0]
    return 0

int | error conv_relu_cpu_probe()
    nn.Conv2D layer = try nn.Conv2D(1, 1, 1)
    tensor<real32> output = nn.relu(
        layer.forward(tensor.ones<real32>([1, 1, 1, 1]))
    )
    int ignored = output.shape()[0]
    return 0

int | error conv_relu_gpu_probe()
    nn.Conv2D layer = try nn.Conv2D(1, 1, 1)
    tensor<real32> output = nn.relu(
        layer.forward(tensor.ones<real32>([1, 1, 1, 1], gpu = 0))
    )
    int ignored = output.shape()[0]
    return 0

int | error fc_relu_cpu_probe()
    nn.FC layer = try nn.FC(features_in = 2, features_out = 1)
    tensor<real32> output = nn.relu(
        layer.forward(tensor.ones<real32>([1, 2]))
    )
    int ignored = output.shape()[0]
    return 0

int | error fc_relu_gpu_probe()
    nn.FC layer = try nn.FC(features_in = 2, features_out = 1)
    tensor<real32> output = nn.relu(
        layer.forward(tensor.ones<real32>([1, 2], gpu = 0))
    )
    int ignored = output.shape()[0]
    return 0

int | error fc_gelu_cpu_probe()
    nn.FC layer = try nn.FC(features_in = 2, features_out = 1)
    tensor<real32> output = nn.gelu(
        layer.forward(tensor.ones<real32>([1, 2]))
    )
    int ignored = output.shape()[0]
    return 0

int | error fc_gelu_gpu_probe()
    nn.FC layer = try nn.FC(features_in = 2, features_out = 1)
    tensor<real32> output = nn.gelu(
        layer.forward(tensor.ones<real32>([1, 2], gpu = 0))
    )
    int ignored = output.shape()[0]
    return 0

int | error conv_gelu_cpu_probe()
    nn.Conv2D layer = try nn.Conv2D(1, 1, 1)
    tensor<real32> output = nn.gelu(
        layer.forward(tensor.ones<real32>([1, 1, 1, 1]))
    )
    int ignored = output.shape()[0]
    return 0

int | error conv_gelu_gpu_probe()
    nn.Conv2D layer = try nn.Conv2D(1, 1, 1)
    tensor<real32> output = nn.gelu(
        layer.forward(tensor.ones<real32>([1, 1, 1, 1], gpu = 0))
    )
    int ignored = output.shape()[0]
    return 0

int | error conv_batchnorm_probe()
    nn.Conv2D layer = try nn.Conv2D(1, 1, 1)
    nn.BatchNorm normalizer = try nn.BatchNorm(1)
    tensor<real32> output = normalizer.infer(
        layer.forward(tensor.ones<real32>([1, 1, 1, 1]))
    )
    int ignored = output.shape()[0]
    return 0

int | error conv_batchnorm_relu_probe()
    nn.Conv2D layer = try nn.Conv2D(1, 1, 1)
    nn.BatchNorm normalizer = try nn.BatchNorm(1)
    tensor<real32> output = nn.relu(
        normalizer.infer(
            layer.forward(tensor.ones<real32>([1, 1, 1, 1]))
        )
    )
    int ignored = output.shape()[0]
    return 0

int | error conv_batchnorm_gelu_probe()
    nn.Conv2D layer = try nn.Conv2D(1, 1, 1)
    nn.BatchNorm normalizer = try nn.BatchNorm(1)
    tensor<real32> output = nn.gelu(
        normalizer.infer(
            layer.forward(tensor.ones<real32>([1, 1, 1, 1]))
        )
    )
    int ignored = output.shape()[0]
    return 0

int | error tracked_fusion_probe()
    nn.Conv2D layer = try nn.Conv2D(1, 1, 1)
    tensor<real32> input = tensor.ones<real32>([1, 1, 1, 1]).track()
    tensor<real32> output = nn.relu(layer.forward(input))
    int ignored = output.shape()[0]
    return 0
QUI

stage_ir="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" ir "$TMP/stages.qui")"
assert_target() {
    local function_name="$1"
    local target="$2"
    local body
    body="$(awk "/^function $function_name\\(/,/^end$/" <<< "$stage_ir")"
    if ! grep -Fq "$target" <<< "$body"; then
        echo "NN compiler target missing for $function_name: $target" >&2
        printf '%s\n' "$body" >&2
        exit 1
    fi
}
assert_target "conv_cpu_probe" "Conv2D.forward_cpu_inference"
assert_target "conv_gpu_probe" "Conv2D.forward_cuda_inference"
assert_target "conv_gpu_fast_probe" "Conv2D.forward_cuda_fast_inference"
assert_target "conv_gpu_deterministic_probe" "Conv2D.forward_cuda_deterministic_inference"
assert_target "conv_relu_cpu_probe" "Conv2D.forward_relu_cpu_inference"
assert_target "conv_relu_gpu_probe" "Conv2D.forward_relu_cuda"
assert_target "fc_relu_cpu_probe" "FC.forward_relu_cpu"
assert_target "fc_relu_gpu_probe" "FC.forward_relu_cuda"
assert_target "fc_gelu_cpu_probe" "FC.forward_gelu_cpu"
assert_target "fc_gelu_gpu_probe" "FC.forward_gelu_cuda"
assert_target "conv_gelu_cpu_probe" "Conv2D.forward_gelu_cpu_inference"
assert_target "conv_gelu_gpu_probe" "Conv2D.forward_gelu_cuda"
assert_target "conv_batchnorm_probe" "BatchNorm.infer_conv"
assert_target "conv_batchnorm_relu_probe" "BatchNorm.infer_conv_relu"
assert_target "conv_batchnorm_gelu_probe" "BatchNorm.infer_conv_gelu"

tracked_body="$(awk '/^function tracked_fusion_probe\(/,/^end$/' <<< "$stage_ir")"
if ! grep -Fq "Conv2D.forward_relu" <<< "$tracked_body" ||
   grep -Eq "forward_relu_(cpu_inference|cuda)" <<< "$tracked_body"; then
    echo "tracked NN fusion was unsafely inference-specialized" >&2
    printf '%s\n' "$tracked_body" >&2
    exit 1
fi

cat > "$TMP/memory.qui" <<'QUI'
import nn

int unique_probe()
    tensor<real32> output = nn.relu(tensor.ones<real32>([4]))
    print(output[0].item() == real32(1))
    print(NL)
    return 0

int alias_probe()
    tensor<real32> values = tensor.ones<real32>([2])
    values[0] = real32(-1)
    tensor<real32> output = nn.relu(values)
    print(output[0].item() == real32(0))
    print(NL)
    print(values[0].item() == real32(-1))
    print(NL)
    return 0

int tracked_probe()
    tensor<real32> values = tensor.ones<real32>([2]).track()
    tensor<real32> output = nn.relu(values)
    int ignored = output.shape()[0]
    return 0

int unique_status = unique_probe()
int alias_status = alias_probe()
int tracked_status = tracked_probe()
QUI

memory_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/memory.qui")"
if [[ "$memory_output" != "$(printf 'true\ntrue\ntrue')" ]]; then
    echo "NN memory-planning execution changed value semantics:" >&2
    printf '%s\n' "$memory_output" >&2
    exit 1
fi
memory_ir="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" ir "$TMP/memory.qui")"
unique_body="$(awk '/^function unique_probe\(/,/^end$/' <<< "$memory_ir")"
alias_body="$(awk '/^function alias_probe\(/,/^end$/' <<< "$memory_ir")"
tracked_memory_body="$(awk '/^function tracked_probe\(/,/^end$/' <<< "$memory_ir")"
if ! grep -Fq "internal.relu_reuse" <<< "$unique_body"; then
    echo "NN memory planner did not select reuse for a unique last-use temporary" >&2
    printf '%s\n' "$unique_body" >&2
    exit 1
fi
if grep -Fq "internal.relu_reuse" <<< "$alias_body"; then
    echo "NN memory planner reused aliased tensor storage" >&2
    printf '%s\n' "$alias_body" >&2
    exit 1
fi
if grep -Fq "internal.relu_reuse" <<< "$tracked_memory_body"; then
    echo "NN memory planner reused tracked tensor storage" >&2
    printf '%s\n' "$tracked_memory_body" >&2
    exit 1
fi

cat > "$TMP/training.qui" <<'QUI'
import nn
import math

int run()
    tensor<real32> seed = tensor.zeros<real32>([1])
    tensor<real32> optimized = seed.track()
    tensor<real32> reference = seed.track()

    tensor<real32> optimized_output = nn.relu(optimized)
    tensor<real32> reference_output = (
        reference + math.abs(reference)
    ) / 2.0
    print(
        optimized_output.untrack()[0].item()
        == reference_output.untrack()[0].item()
    )
    print(NL)

    math.mean(optimized_output).backward(&optimized, track = true)
    math.mean(reference_output).backward(&reference, track = true)
    tensor<real32> optimized_first = optimized.grad
    tensor<real32> reference_first = reference.grad
    print(optimized_first.is_tracked())
    print(NL)
    print(reference_first.is_tracked())
    print(NL)
    print(
        optimized_first.untrack()[0].item()
        == reference_first.untrack()[0].item()
    )
    print(NL)

    optimized.clear_grad()
    reference.clear_grad()
    math.mean(optimized_first).backward(&optimized)
    math.mean(reference_first).backward(&reference)
    print(
        optimized.grad.untrack()[0].item()
        == reference.grad.untrack()[0].item()
    )
    print(NL)
    return 0

int ignored = run()
QUI

training_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/training.qui")"
if [[ "$training_output" != "$(printf 'true\ntrue\ntrue\ntrue\ntrue')" ]]; then
    echo "NN training-graph optimization changed autograd semantics:" >&2
    printf '%s\n' "$training_output" >&2
    exit 1
fi
training_ir="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" ir "$TMP/training.qui")"
training_body="$(awk '/^function run\(/,/^end$/' <<< "$training_ir")"
if ! grep -Fq "internal.relu_training" <<< "$training_body"; then
    echo "NN tracked ReLU did not select package custom-autograd target" >&2
    printf '%s\n' "$training_body" >&2
    exit 1
fi

echo "nn compiler optimization: ok"
