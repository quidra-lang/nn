#!/usr/bin/env bash
set -euo pipefail

QUIDRA="${1:-}"
if [[ -z "$QUIDRA" ]]; then
    echo "usage: $0 /path/to/quidra" >&2
    exit 2
fi

REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
GPU_INDEX="${QUIDRA_REAL_GPU_INDEX:-0}"
REQUIRE_REAL="${QUIDRA_REQUIRE_REAL_GPU:-0}"
REQUIRE_BACKEND="${QUIDRA_REQUIRE_GPU_BACKEND:-}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

set +e
gpu_info="$("$QUIDRA" gpu 2>&1)"
gpu_status=$?
set -e
skip_or_fail() {
    local reason="$1"
    if [[ "$REQUIRE_REAL" == "1" ]]; then
        echo "real NN GPU integration required but unavailable: $reason" >&2
        printf '%s\n' "$gpu_info" >&2
        exit 1
    fi
    echo "nn real GPU integration: skipped ($reason)"
    exit 0
}
if [[ $gpu_status -ne 0 ]]; then skip_or_fail "quidra gpu failed"; fi
if grep -Fq "backend: TEST" <<<"$gpu_info"; then skip_or_fail "fake GPU backend is active"; fi
if ! grep -Fq "GPU $GPU_INDEX" <<<"$gpu_info"; then skip_or_fail "gpu($GPU_INDEX) is not present"; fi
gpu_block="$(awk -v target="GPU $GPU_INDEX" '
    $0 == target { found = 1; print; next }
    found && /^GPU [0-9]+$/ { exit }
    found { print }
' <<<"$gpu_info")"
if [[ -n "$REQUIRE_BACKEND" ]] && ! grep -Fq "backend: $REQUIRE_BACKEND" <<<"$gpu_block"; then
    skip_or_fail "gpu($GPU_INDEX) is not backend $REQUIRE_BACKEND"
fi

cat > "$TMP/dnn-real-gpu.qui" <<QUI
import nn
import math

class FCModel
    nn.FC dense

class ConvModel
    nn.Conv2D convolution

class NormModel
    nn.BatchNorm normalization

bool near(float32 a, float32 b)
    float32 d = a - b
    return d > float32(-0.0005) and d < float32(0.0005)

int | error run()
    nn.FC cpu_linear
    cpu_linear.weight = nn.Parameter<float32>(value = tensor.ones<float32>([1, 2]))
    cpu_linear.bias = nn.Parameter<float32>(value = tensor.zeros<float32>([1]))
    nn.FC gpu_linear
    gpu_linear.weight = nn.Parameter<float32>(value = tensor.ones<float32>([1, 2], gpu = $GPU_INDEX))
    gpu_linear.bias = nn.Parameter<float32>(value = tensor.zeros<float32>([1], gpu = $GPU_INDEX))
    tensor<float32> cpu_samples = tensor.ones<float32>([1, 2])
    tensor<float32> gpu_samples = cpu_samples.gpu($GPU_INDEX)
    tensor<float32> cpu_linear_out = cpu_linear.forward(cpu_samples)
    tensor<float32> gpu_linear_out = gpu_linear.forward(gpu_samples).cpu()
    print(near(cpu_linear_out[0, 0].item(), gpu_linear_out[0, 0].item()))
    print(NL)

    tensor<float32> cpu_linear_relu = nn.relu(cpu_linear.forward(cpu_samples))
    tensor<float32> gpu_linear_relu = nn.relu(
        gpu_linear.forward(gpu_samples)
    ).cpu()
    print(near(cpu_linear_relu[0, 0].item(), gpu_linear_relu[0, 0].item()))
    print(NL)

    tensor<float32> cpu_linear_gelu = nn.gelu(cpu_linear.forward(cpu_samples))
    tensor<float32> gpu_linear_gelu = nn.gelu(
        gpu_linear.forward(gpu_samples)
    ).cpu()
    print(near(cpu_linear_gelu[0, 0].item(), gpu_linear_gelu[0, 0].item()))
    print(NL)

    nn.Conv2D cpu_conv
    cpu_conv.weight = nn.Parameter<float32>(value = tensor.ones<float32>([1, 1, 1, 1]))
    cpu_conv.bias = nn.Parameter<float32>(value = tensor.zeros<float32>([1]))
    cpu_conv.step = 1
    cpu_conv.border = 0
    nn.Conv2D gpu_conv
    gpu_conv.weight = nn.Parameter<float32>(value = tensor.ones<float32>([1, 1, 1, 1], gpu = $GPU_INDEX))
    gpu_conv.bias = nn.Parameter<float32>(value = tensor.zeros<float32>([1], gpu = $GPU_INDEX))
    gpu_conv.step = 1
    gpu_conv.border = 0
    tensor<float32> cpu_pixels = tensor.ones<float32>([1, 1, 2, 2])
    tensor<float32> gpu_pixels = cpu_pixels.gpu($GPU_INDEX)
    tensor<float32> cpu_conv_out = cpu_conv.forward(cpu_pixels)
    tensor<float32> gpu_conv_out = gpu_conv.forward(gpu_pixels).cpu()
    print(near(cpu_conv_out[0, 0, 1, 1].item(), gpu_conv_out[0, 0, 1, 1].item()))
    print(NL)

    tensor<float32> cpu_negative_pixels = tensor.ones<float32>([1, 1, 2, 2]) * float32(-1)
    tensor<float32> gpu_negative_pixels = cpu_negative_pixels.gpu($GPU_INDEX)
    tensor<float32> gpu_relu = nn.relu(
        gpu_conv.forward(gpu_negative_pixels)
    ).cpu()
    print(near(gpu_relu[0, 0, 1, 1].item(), float32(0)))
    print(NL)

    tensor<float32> cpu_conv_gelu = nn.gelu(
        cpu_conv.forward(cpu_negative_pixels)
    )
    tensor<float32> gpu_conv_gelu = nn.gelu(
        gpu_conv.forward(gpu_negative_pixels)
    ).cpu()
    print(near(cpu_conv_gelu[0, 0, 1, 1].item(), gpu_conv_gelu[0, 0, 1, 1].item()))
    print(NL)

    tensor<float32> gpu_reused_relu = nn.relu(
        tensor.ones<float32>([4], gpu = $GPU_INDEX)
    ).cpu()
    print(near(gpu_reused_relu[0].item(), float32(1)))
    print(NL)

    nn.Conv2D cpu_conv3
    cpu_conv3.weight = nn.Parameter<float32>(value = tensor.ones<float32>([2, 2, 3, 3]))
    cpu_conv3.bias = nn.Parameter<float32>(value = tensor.zeros<float32>([2]))
    cpu_conv3.step = 2
    cpu_conv3.border = 1
    nn.Conv2D gpu_conv3
    gpu_conv3.weight = nn.Parameter<float32>(value = tensor.ones<float32>([2, 2, 3, 3], gpu = $GPU_INDEX))
    gpu_conv3.bias = nn.Parameter<float32>(value = tensor.zeros<float32>([2], gpu = $GPU_INDEX))
    gpu_conv3.step = 2
    gpu_conv3.border = 1
    tensor<float32> cpu_conv3_input = tensor.ones<float32>([2, 2, 5, 7])
    tensor<float32> gpu_conv3_input = cpu_conv3_input.gpu($GPU_INDEX)
    tensor<float32> cpu_conv3_out = cpu_conv3.forward(cpu_conv3_input)
    tensor<float32> gpu_conv3_out = gpu_conv3.forward(gpu_conv3_input).cpu()
    print(
        cpu_conv3_out.shape()[2] == gpu_conv3_out.shape()[2] and
        cpu_conv3_out.shape()[3] == gpu_conv3_out.shape()[3] and
        near(cpu_conv3_out[0, 0, 0, 0].item(), gpu_conv3_out[0, 0, 0, 0].item()) and
        near(cpu_conv3_out[1, 1, 1, 1].item(), gpu_conv3_out[1, 1, 1, 1].item())
    )
    print(NL)
    nn.mode.deterministic()
    tensor<float32> deterministic_first = gpu_conv3.forward(gpu_conv3_input).cpu()
    tensor<float32> deterministic_second = gpu_conv3.forward(gpu_conv3_input).cpu()
    print(
        deterministic_first[0, 0, 0, 0].item() ==
        deterministic_second[0, 0, 0, 0].item() and
        deterministic_first[1, 1, 1, 1].item() ==
        deterministic_second[1, 1, 1, 1].item()
    )
    print(NL)
    nn.mode.fast()

    nn.BatchNorm cpu_norm
    cpu_norm.scale = nn.Parameter<float32>(value = tensor.ones<float32>([2]))
    cpu_norm.bias = nn.Parameter<float32>(value = tensor.zeros<float32>([2]))
    cpu_norm.running_mean = nn.State<float32>(value = tensor.zeros<float32>([2]))
    cpu_norm.running_variance = nn.State<float32>(value = tensor.ones<float32>([2]))
    cpu_norm.running_momentum = 0.1
    cpu_norm.variance_epsilon = 0.00001
    nn.BatchNorm gpu_norm
    gpu_norm.scale = nn.Parameter<float32>(value = tensor.ones<float32>([2], gpu = $GPU_INDEX))
    gpu_norm.bias = nn.Parameter<float32>(value = tensor.zeros<float32>([2], gpu = $GPU_INDEX))
    gpu_norm.running_mean = nn.State<float32>(value = tensor.zeros<float32>([2], gpu = $GPU_INDEX))
    gpu_norm.running_variance = nn.State<float32>(value = tensor.ones<float32>([2], gpu = $GPU_INDEX))
    gpu_norm.running_momentum = 0.1
    gpu_norm.variance_epsilon = 0.00001
    tensor<float32> cpu_norm_input = tensor.ones<float32>([2, 2])
    tensor<float32> gpu_norm_input = cpu_norm_input.gpu($GPU_INDEX)
    tensor<float32> cpu_norm_out = cpu_norm.infer(cpu_norm_input)
    tensor<float32> gpu_norm_out = gpu_norm.infer(gpu_norm_input).cpu()
    print(near(cpu_norm_out[0, 0].item(), gpu_norm_out[0, 0].item()))
    print(NL)

    nn.Dropout cpu_drop = nn.Dropout(rate = 0.5, seed = 17)
    nn.Dropout gpu_drop = nn.Dropout(rate = 0.5, seed = 17)
    tensor<float32> drop_input = tensor.ones<float32>([2, 2])
    tensor<float32> cpu_dropped = cpu_drop.forward(drop_input.track()).untrack()
    tensor<float32> gpu_dropped = gpu_drop.forward(drop_input.gpu($GPU_INDEX).track()).untrack().cpu()
    print(near(cpu_dropped[0, 0].item(), gpu_dropped[0, 0].item()))
    print(NL)
    print(near(cpu_dropped[1, 1].item(), gpu_dropped[1, 1].item()))
    print(NL)
    tensor<float32> cpu_dropped_next = cpu_drop.forward(drop_input.track()).untrack()
    tensor<float32> gpu_dropped_next = gpu_drop.forward(drop_input.gpu($GPU_INDEX).track()).untrack().cpu()
    print(near(cpu_dropped_next[0, 1].item(), gpu_dropped_next[0, 1].item()))
    print(NL)

    tensor<float32> cpu_activation = tensor.zeros<float32>([1, 2])
    cpu_activation[0, 0] = float32(-1)
    cpu_activation[0, 1] = float32(1)
    tensor<float32> gpu_activation = cpu_activation.gpu($GPU_INDEX)

    tensor<float32> cpu_relu = nn.relu(cpu_activation.track()).untrack()
    tensor<float32> gpu_activation_relu = nn.relu(gpu_activation.track()).untrack().cpu()
    print(near(cpu_relu[0, 0].item(), gpu_activation_relu[0, 0].item()) and near(cpu_relu[0, 1].item(), gpu_activation_relu[0, 1].item()))
    print(NL)

    tensor<float32> cpu_tanh = nn.tanh(cpu_activation.track()).untrack()
    tensor<float32> gpu_tanh = nn.tanh(gpu_activation.track()).untrack().cpu()
    print(near(cpu_tanh[0, 0].item(), gpu_tanh[0, 0].item()) and near(cpu_tanh[0, 1].item(), gpu_tanh[0, 1].item()))
    print(NL)

    tensor<float32> cpu_sigmoid = nn.sigmoid(cpu_activation.track()).untrack()
    tensor<float32> gpu_sigmoid = nn.sigmoid(gpu_activation.track()).untrack().cpu()
    print(near(cpu_sigmoid[0, 0].item(), gpu_sigmoid[0, 0].item()) and near(cpu_sigmoid[0, 1].item(), gpu_sigmoid[0, 1].item()))
    print(NL)

    tensor<float32> cpu_softmax = nn.softmax(cpu_activation.track()).untrack()
    tensor<float32> gpu_softmax = nn.softmax(gpu_activation.track()).untrack().cpu()
    print(near(cpu_softmax[0, 0].item(), gpu_softmax[0, 0].item()) and near(cpu_softmax[0, 1].item(), gpu_softmax[0, 1].item()))
    print(NL)

    tensor<float32> cpu_gelu = nn.gelu(cpu_activation.track()).untrack()
    tensor<float32> gpu_gelu = nn.gelu(gpu_activation.track()).untrack().cpu()
    print(near(cpu_gelu[0, 0].item(), gpu_gelu[0, 0].item()) and near(cpu_gelu[0, 1].item(), gpu_gelu[0, 1].item()))
    print(NL)

    tensor<float32> cpu_probability = tensor.zeros<float32>([1, 2])
    cpu_probability[0, 0] = float32(0.25)
    cpu_probability[0, 1] = float32(0.75)
    tensor<float32> gpu_probability = cpu_probability.gpu($GPU_INDEX)
    tensor<float32> cpu_binary_target = tensor.zeros<float32>([1, 2])
    cpu_binary_target[0, 1] = float32(1)
    tensor<float32> gpu_binary_target = cpu_binary_target.gpu($GPU_INDEX)

    float32 cpu_mse_loss = nn.mse(cpu_probability.track(), cpu_binary_target).item()
    float32 gpu_mse_loss = nn.mse(gpu_probability.track(), gpu_binary_target).item()
    print(near(cpu_mse_loss, gpu_mse_loss))
    print(NL)

    float32 cpu_bce = nn.binary_cross_entropy(cpu_probability.track(), cpu_binary_target).item()
    float32 gpu_bce = nn.binary_cross_entropy(gpu_probability.track(), gpu_binary_target).item()
    print(near(cpu_bce, gpu_bce))
    print(NL)

    float32 cpu_bce_logits = nn.binary_cross_entropy_with_logits(cpu_activation.track(), cpu_binary_target).item()
    float32 gpu_bce_logits = nn.binary_cross_entropy_with_logits(gpu_activation.track(), gpu_binary_target).item()
    print(near(cpu_bce_logits, gpu_bce_logits))
    print(NL)

    float32 cpu_cross_entropy = nn.cross_entropy(cpu_activation.track(), cpu_binary_target).item()
    float32 gpu_cross_entropy = nn.cross_entropy(gpu_activation.track(), gpu_binary_target).item()
    print(near(cpu_cross_entropy, gpu_cross_entropy))
    print(NL)

    FCModel cpu_model
    cpu_model.dense = cpu_linear
    FCModel gpu_model
    gpu_model.dense = gpu_linear
    tensor<float32> cpu_target = tensor.zeros<float32>([1, 1])
    tensor<float32> gpu_target = cpu_target.gpu($GPU_INDEX)
    tensor<float32> cpu_loss = nn.mse(
        cpu_model.dense.forward(cpu_samples.track()), cpu_target
    )
    tensor<float32> gpu_loss = nn.mse(
        gpu_model.dense.forward(gpu_samples.track()), gpu_target
    )
    nn.SGD cpu_sgd = try nn.SGD(rate = 0.1)
    nn.SGD gpu_sgd = try nn.SGD(rate = 0.1)
    cpu_sgd.zero_grad(&cpu_model)
    gpu_sgd.zero_grad(&gpu_model)
    cpu_loss.backward(&cpu_model)
    gpu_loss.backward(&gpu_model)
    cpu_sgd.step(&cpu_model)
    gpu_sgd.step(&gpu_model)
    print(near(
        cpu_model.dense.weight.raw()[0, 0].item(),
        gpu_model.dense.weight.raw()[0, 0].item()
    ))
    print(NL)

    ConvModel cpu_conv_model
    cpu_conv_model.convolution = cpu_conv
    ConvModel gpu_conv_model
    gpu_conv_model.convolution = gpu_conv
    tensor<float32> cpu_zero_image = tensor.zeros<float32>([1, 1, 2, 2])
    tensor<float32> gpu_zero_image = cpu_zero_image.gpu($GPU_INDEX)
    tensor<float32> cpu_conv_loss = nn.mse(
        cpu_conv_model.convolution.forward(cpu_pixels.track()), cpu_zero_image
    )
    tensor<float32> gpu_conv_loss = nn.mse(
        gpu_conv_model.convolution.forward(gpu_pixels.track()), gpu_zero_image
    )
    nn.SGD cpu_conv_sgd = try nn.SGD(rate = 0.1)
    nn.SGD gpu_conv_sgd = try nn.SGD(rate = 0.1)
    cpu_conv_sgd.zero_grad(&cpu_conv_model)
    gpu_conv_sgd.zero_grad(&gpu_conv_model)
    cpu_conv_loss.backward(&cpu_conv_model)
    gpu_conv_loss.backward(&gpu_conv_model)
    cpu_conv_sgd.step(&cpu_conv_model)
    gpu_conv_sgd.step(&gpu_conv_model)
    print(near(
        cpu_conv_model.convolution.weight.raw()[0, 0, 0, 0].item(),
        gpu_conv_model.convolution.weight.raw()[0, 0, 0, 0].item()
    ))
    print(NL)

    ConvModel cpu_conv3_model
    cpu_conv3_model.convolution = cpu_conv3
    ConvModel gpu_conv3_model
    gpu_conv3_model.convolution = gpu_conv3
    tensor<float32> cpu_conv3_target = tensor.zeros<float32>([2, 2, 3, 4])
    tensor<float32> gpu_conv3_target = cpu_conv3_target.gpu($GPU_INDEX)
    tensor<float32> cpu_conv3_loss = nn.mse(
        cpu_conv3_model.convolution.forward(cpu_conv3_input.track()),
        cpu_conv3_target
    )
    tensor<float32> gpu_conv3_loss = nn.mse(
        gpu_conv3_model.convolution.forward(gpu_conv3_input.track()),
        gpu_conv3_target
    )
    nn.SGD cpu_conv3_sgd = try nn.SGD(rate = 0.01)
    nn.SGD gpu_conv3_sgd = try nn.SGD(rate = 0.01)
    cpu_conv3_sgd.zero_grad(&cpu_conv3_model)
    gpu_conv3_sgd.zero_grad(&gpu_conv3_model)
    cpu_conv3_loss.backward(&cpu_conv3_model)
    gpu_conv3_loss.backward(&gpu_conv3_model)
    cpu_conv3_sgd.step(&cpu_conv3_model)
    gpu_conv3_sgd.step(&gpu_conv3_model)
    print(near(
        cpu_conv3_model.convolution.weight.raw()[0, 0, 1, 1].item(),
        gpu_conv3_model.convolution.weight.raw()[0, 0, 1, 1].item()
    ))
    print(NL)
    print(near(
        cpu_conv3_model.convolution.bias.raw()[1].item(),
        gpu_conv3_model.convolution.bias.raw()[1].item()
    ))
    print(NL)

    NormModel cpu_norm_model
    cpu_norm_model.normalization = cpu_norm
    NormModel gpu_norm_model
    gpu_norm_model.normalization = gpu_norm
    tensor<float32> cpu_norm_train = cpu_norm_model.normalization.forward(cpu_norm_input.track())
    tensor<float32> gpu_norm_train = gpu_norm_model.normalization.forward(gpu_norm_input.track())
    tensor<float32> cpu_norm_loss = math.mean(cpu_norm_train * cpu_norm_train)
    tensor<float32> gpu_norm_loss = math.mean(gpu_norm_train * gpu_norm_train)
    nn.SGD cpu_norm_sgd = try nn.SGD(rate = 0.1)
    nn.SGD gpu_norm_sgd = try nn.SGD(rate = 0.1)
    cpu_norm_sgd.zero_grad(&cpu_norm_model)
    gpu_norm_sgd.zero_grad(&gpu_norm_model)
    cpu_norm_loss.backward(&cpu_norm_model)
    gpu_norm_loss.backward(&gpu_norm_model)
    cpu_norm_sgd.step(&cpu_norm_model)
    gpu_norm_sgd.step(&gpu_norm_model)
    print(near(
        cpu_norm_model.normalization.scale.raw()[0].item(),
        gpu_norm_model.normalization.scale.raw()[0].item()
    ))
    print(NL)
    tensor<float32> cpu_norm_infer = cpu_norm_model.normalization.infer(cpu_norm_input)
    tensor<float32> gpu_norm_infer = gpu_norm_model.normalization.infer(gpu_norm_input).cpu()
    print(near(cpu_norm_infer[0, 0].item(), gpu_norm_infer[0, 0].item()))
    print(NL)

    nn.FC cpu_adam_layer
    cpu_adam_layer.weight = nn.Parameter<float32>(value = tensor.ones<float32>([1, 1]))
    cpu_adam_layer.bias = nn.Parameter<float32>(value = tensor.zeros<float32>([1]))
    nn.FC gpu_adam_layer
    gpu_adam_layer.weight = nn.Parameter<float32>(value = tensor.ones<float32>([1, 1], gpu = $GPU_INDEX))
    gpu_adam_layer.bias = nn.Parameter<float32>(value = tensor.zeros<float32>([1], gpu = $GPU_INDEX))
    FCModel cpu_adam_model
    cpu_adam_model.dense = cpu_adam_layer
    FCModel gpu_adam_model
    gpu_adam_model.dense = gpu_adam_layer
    tensor<float32> cpu_one = tensor.ones<float32>([1, 1])
    tensor<float32> gpu_one = cpu_one.gpu($GPU_INDEX)
    tensor<float32> cpu_zero = tensor.zeros<float32>([1, 1])
    tensor<float32> gpu_zero = cpu_zero.gpu($GPU_INDEX)
    tensor<float32> cpu_adam_loss = nn.mse(
        cpu_adam_model.dense.forward(cpu_one.track()), cpu_zero
    )
    tensor<float32> gpu_adam_loss = nn.mse(
        gpu_adam_model.dense.forward(gpu_one.track()), gpu_zero
    )
    nn.Adam cpu_adam = try nn.Adam(rate = 0.1)
    nn.Adam gpu_adam = try nn.Adam(rate = 0.1)
    cpu_adam.zero_grad(&cpu_adam_model)
    gpu_adam.zero_grad(&gpu_adam_model)
    cpu_adam_loss.backward(&cpu_adam_model)
    gpu_adam_loss.backward(&gpu_adam_model)
    cpu_adam.step(&cpu_adam_model)
    gpu_adam.step(&gpu_adam_model)
    print(near(
        cpu_adam_model.dense.weight.raw()[0, 0].item(),
        gpu_adam_model.dense.weight.raw()[0, 0].item()
    ))
    print(NL)
    print(near(
        cpu_adam_model.dense.bias.raw()[0].item(),
        gpu_adam_model.dense.bias.raw()[0].item()
    ))
    print(NL)

    tensor<float32> cpu_adam_loss2 = nn.mse(
        cpu_adam_model.dense.forward(cpu_one.track()), cpu_zero
    )
    tensor<float32> gpu_adam_loss2 = nn.mse(
        gpu_adam_model.dense.forward(gpu_one.track()), gpu_zero
    )
    cpu_adam.zero_grad(&cpu_adam_model)
    gpu_adam.zero_grad(&gpu_adam_model)
    cpu_adam_loss2.backward(&cpu_adam_model)
    gpu_adam_loss2.backward(&gpu_adam_model)
    cpu_adam.step(&cpu_adam_model)
    gpu_adam.step(&gpu_adam_model)
    print(near(
        cpu_adam_model.dense.weight.raw()[0, 0].item(),
        gpu_adam_model.dense.weight.raw()[0, 0].item()
    ))
    print(NL)
    print(near(
        cpu_adam_model.dense.bias.raw()[0].item(),
        gpu_adam_model.dense.bias.raw()[0].item()
    ))
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

output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/dnn-real-gpu.qui")"
expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$output" != "$expected" ]]; then
    echo "NN real GPU numerical equivalence failed on gpu($GPU_INDEX)" >&2
    printf '%s\n' "$output" >&2
    exit 1
fi

# Loss semantics on the GPU (closed-form value and gradient).
gpu_loss_output="$(
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" \
    "$QUIDRA" "$REPOSITORY_ROOT/tests/losses.qui" --device "$GPU_INDEX"
)"
gpu_loss_expected_count="$(grep -c '^ *report("' "$REPOSITORY_ROOT/tests/losses.qui")"
gpu_loss_passed_count="$(grep -c ' true$' <<<"$gpu_loss_output" || true)"
if [[ "$gpu_loss_passed_count" -ne "$gpu_loss_expected_count" ]] ||
   grep -Fq ' false' <<<"$gpu_loss_output"; then
    echo "NN loss semantics failed on gpu($GPU_INDEX)" >&2
    printf '%s\n' "$gpu_loss_output" >&2
    exit 1
fi

if grep -Fq "backend: Metal" <<<"$gpu_block"; then
    # NN-owned Metal kernels: CPU equivalence, gradients and determinism.
    metal_output="$(
        QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" \
        "$QUIDRA" "$REPOSITORY_ROOT/tests/metal_native.qui" --device "$GPU_INDEX"
    )"
    metal_expected_count="$(grep -c '^ *report("' "$REPOSITORY_ROOT/tests/metal_native.qui")"
    metal_passed_count="$(grep -c ' true$' <<<"$metal_output" || true)"
    if [[ "$metal_passed_count" -ne "$metal_expected_count" ]] ||
       grep -Fq ' false' <<<"$metal_output"; then
        echo "NN Metal native kernels failed on gpu($GPU_INDEX)" >&2
        printf '%s\n' "$metal_output" >&2
        exit 1
    fi

    # The Metal operations above must have run NN's native kernels: the
    # portable fallback gives the same values, so count dispatches instead.
    dispatch_output="$(
        QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" \
        "$QUIDRA" "$REPOSITORY_ROOT/tests/metal_dispatch.qui" --device "$GPU_INDEX"
    )"
    dispatch_expected_count="$(grep -c '^ *report("' "$REPOSITORY_ROOT/tests/metal_dispatch.qui")"
    dispatch_passed_count="$(grep -c ' true$' <<<"$dispatch_output" || true)"
    if [[ "$dispatch_passed_count" -ne "$dispatch_expected_count" ]] ||
       grep -Fq ' false' <<<"$dispatch_output"; then
        echo "NN Metal operations fell back to the portable path on gpu($GPU_INDEX)" >&2
        printf '%s\n' "$dispatch_output" >&2
        exit 1
    fi

    # Partially initialized, untracked GPU inputs must stop with
    # UNINITIALIZED, as they do on the portable path. NN's Metal kernels rely
    # on Core rejecting such views in qcore_tensor_device_handle(_const), so
    # all four scenarios must raise the error. With an older Core that does
    # not reject them, the kernels compute on unwritten memory; that is only
    # reported, and QUIDRA_NN_REQUIRE_METAL_UNINITIALIZED=1 makes it a
    # failure.
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" build \
        "$REPOSITORY_ROOT/tests/metal_uninitialized.qui" \
        -o "$TMP/metal-uninitialized" >/dev/null
    uninitialized_raised=0
    uninitialized_computed=0
    for scenario in 0 1 2 3; do
        set +e
        "$TMP/metal-uninitialized" --device "$GPU_INDEX" --scenario "$scenario" \
            >"$TMP/metal-uninitialized.out" 2>"$TMP/metal-uninitialized.err"
        uninitialized_status=$?
        set -e
        if [[ $uninitialized_status -ne 0 ]] &&
           grep -Fq "runtime error[UNINITIALIZED]" "$TMP/metal-uninitialized.err"; then
            uninitialized_raised=$((uninitialized_raised + 1))
        elif [[ $uninitialized_status -eq 0 ]] &&
             grep -q '^computed ' "$TMP/metal-uninitialized.out"; then
            uninitialized_computed=$((uninitialized_computed + 1))
        else
            echo "NN Metal uninitialized scenario $scenario failed unexpectedly on gpu($GPU_INDEX)" >&2
            cat "$TMP/metal-uninitialized.out" "$TMP/metal-uninitialized.err" >&2
            exit 1
        fi
    done
    if [[ $uninitialized_raised -eq 4 ]]; then
        :
    elif [[ $uninitialized_computed -eq 4 &&
            "${QUIDRA_NN_REQUIRE_METAL_UNINITIALIZED:-0}" != "1" ]]; then
        echo "nn Metal UNINITIALIZED checks: not enforced, this Core does not reject partially initialized views (NN's Metal kernels compute on partially initialized untracked inputs)"
    else
        echo "NN Metal kernels accepted partially initialized inputs on gpu($GPU_INDEX): $uninitialized_raised of 4 scenarios raised UNINITIALIZED" >&2
        exit 1
    fi

    # nn.Adam on Metal: parameter, m, v, step count and bias corrections after
    # every step against the element-wise Metal update (bitwise) and the CPU
    # (tolerances explained in the test); concurrent optimizers; tied
    # Parameters.
    adam_output="$(
        QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" QUIDRA_REAL_GPU_INDEX="$GPU_INDEX" \
        "$QUIDRA" "$REPOSITORY_ROOT/tests/metal_adam.qui" --device "$GPU_INDEX" --scratch "$TMP"
    )"
    adam_expected_count="$(grep -c '^ *report("' "$REPOSITORY_ROOT/tests/metal_adam.qui")"
    adam_passed_count="$(grep -c ' true$' <<<"$adam_output" || true)"
    if [[ "$adam_passed_count" -ne "$adam_expected_count" ]] ||
       grep -Fq ' false' <<<"$adam_output"; then
        echo "NN Metal Adam state comparison failed on gpu($GPU_INDEX)" >&2
        printf '%s\n' "$adam_output" >&2
        exit 1
    fi

    # Core does not yet run backward(track = true) on GPU graphs. Until it
    # does, the Metal Conv2D/GELU custom nodes must keep the portable graph's
    # diagnostic instead of silently dropping higher-order provenance; once
    # Core accepts the probe, their second-order callbacks must match the CPU.
    cat > "$TMP/metal-higher-order.qui" <<QUI
import nn
import math

nn.Conv2D convolution
convolution.weight = nn.Parameter<float32>(
    value = tensor.ones<float32>([1, 1, 2, 2], gpu = $GPU_INDEX)
)
convolution.bias = nn.Parameter<float32>(
    value = tensor.zeros<float32>([1], gpu = $GPU_INDEX)
)
convolution.step = 1
convolution.border = 0
tensor<float32> input = tensor.ones<float32>([1, 1, 3, 3], gpu = $GPU_INDEX).track()
math.mean(nn.gelu(convolution.forward(input))).backward(&convolution, &input, track = true)
print("unexpected success")
print(NL)
QUI
    set +e
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/metal-higher-order.qui" \
        >"$TMP/metal-higher-order.out" 2>"$TMP/metal-higher-order.err"
    higher_order_status=$?
    set -e
    if [[ $higher_order_status -eq 0 ]]; then
        second_output="$(
            QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" \
            "$QUIDRA" "$REPOSITORY_ROOT/tests/metal_second_order.qui" --device "$GPU_INDEX"
        )"
        second_expected_count="$(grep -c '^ *report("' "$REPOSITORY_ROOT/tests/metal_second_order.qui")"
        second_passed_count="$(grep -c ' true$' <<<"$second_output" || true)"
        if [[ "$second_passed_count" -ne "$second_expected_count" ]] ||
           grep -Fq ' false' <<<"$second_output"; then
            echo "NN Metal second-order autograd failed on gpu($GPU_INDEX)" >&2
            printf '%s\n' "$second_output" >&2
            exit 1
        fi
    elif ! grep -Fq "backward(track = true) currently requires CPU tensors" "$TMP/metal-higher-order.err"; then
        echo "NN Metal higher-order autograd changed its diagnostic on gpu($GPU_INDEX)" >&2
        cat "$TMP/metal-higher-order.out" "$TMP/metal-higher-order.err" >&2
        exit 1
    else
        echo "nn Metal second-order checks: waiting for Core GPU backward(track = true)"
    fi
fi

if grep -Fq "backend: NVIDIA" <<<"$gpu_block"; then
    cat > "$TMP/dnn-nvidia-higher-order.qui" <<QUI
import nn
import math

nn.Conv2D convolution
convolution.weight = nn.Parameter<float32>(
    value = tensor.ones<float32>([1, 1, 2, 2], gpu = $GPU_INDEX)
)
convolution.bias = nn.Parameter<float32>(
    value = tensor.zeros<float32>([1], gpu = $GPU_INDEX)
)
convolution.step = 1
convolution.border = 0

tensor<float32> input = tensor.ones<float32>(
    [1, 1, 3, 3], gpu = $GPU_INDEX
).track()
tensor<float32> output = convolution.forward(input)
math.mean(output).backward(&convolution, &input, track = true)
print(input.grad.is_tracked())
print(NL)
print(convolution.weight.gradient().is_tracked())
print(NL)
print(convolution.bias.gradient().is_tracked())
print(NL)

// Exercise each package-owned second-order cuDNN callback, not merely the
// tracked first-order attachment.
math.mean(input.grad).backward(&input)
math.mean(convolution.weight.gradient()).backward(&input)
math.mean(convolution.bias.gradient()).backward(&input)
print(input.grad.untrack().shape()[2] == 3)
print(NL)
QUI

    higher_order_output="$(
        QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" \
        "$QUIDRA" "$TMP/dnn-nvidia-higher-order.qui"
    )"
    higher_order_expected="$(printf 'true\ntrue\ntrue\ntrue')"
    if [[ "$higher_order_output" != "$higher_order_expected" ]]; then
        echo "NN NVIDIA higher-order Conv2D autograd failed on gpu($GPU_INDEX)" >&2
        printf '%s\n' "$higher_order_output" >&2
        exit 1
    fi
fi

echo "nn real GPU integration: ok on gpu($GPU_INDEX)"
