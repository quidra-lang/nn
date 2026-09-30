#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Effective only with the core's test-only fake GPU backend.
export QUIDRA_TEST_FAKE_GPU_COUNT=2

cat > "$TMP/device-check.qui" <<'QUI'
import nn
import math

int | error compile_device_surface()
    nn.FC layer = try nn.FC(features_in = 2, features_out = 1)
    tensor<float32> direct = tensor.zeros<float32>([1, 2], gpu = 0)
    tensor<float32> transferred = tensor.ones<float32>([1, 2]).gpu(0)
    tensor<float32> roundtrip = transferred.cpu()
    tensor<float32> output = layer.forward(direct)
    print(roundtrip.shape()[1])
    print(NL)
    print(output.shape()[1])
    print(NL)
    return 0
QUI

QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/device-check.qui" >/dev/null

cat > "$TMP/no-fallback-create.qui" <<'QUI'
tensor<float32> value = tensor.zeros<float32>([1], gpu = 2147483647)
print(value.shape()[0])
print(NL)
QUI

set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/no-fallback-create.qui" >"$TMP/create.out" 2>"$TMP/create.err"
status=$?
set -e
if [[ $status -eq 0 ]]; then
    echo "GPU creation unexpectedly fell back to CPU" >&2
    exit 1
fi
if ! grep -Fq "gpu(2147483647) is not available" "$TMP/create.err"; then
    echo "missing explicit unavailable-GPU diagnostic" >&2
    cat "$TMP/create.err" >&2
    exit 1
fi

cat > "$TMP/no-fallback-transfer.qui" <<'QUI'
tensor<float32> cpu = tensor.ones<float32>([1])
tensor<float32> value = cpu.gpu(2147483647)
print(value.shape()[0])
print(NL)
QUI

set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/no-fallback-transfer.qui" >"$TMP/transfer.out" 2>"$TMP/transfer.err"
status=$?
set -e
if [[ $status -eq 0 ]]; then
    echo "GPU transfer unexpectedly fell back to CPU" >&2
    exit 1
fi
if ! grep -Fq "gpu(2147483647) is not available" "$TMP/transfer.err"; then
    echo "missing explicit unavailable-GPU transfer diagnostic" >&2
    cat "$TMP/transfer.err" >&2
    exit 1
fi

expect_device_failure() {
    local source="$1"
    local expected="$2"
    set +e
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$source" >"$source.out" 2>"$source.err"
    local status=$?
    set -e
    if [[ $status -ne 101 ]]; then
        echo "expected runtime status 101 for $source, got $status" >&2
        cat "$source.out" >&2 || true
        cat "$source.err" >&2 || true
        exit 1
    fi
    if ! grep -Fq "$expected" "$source.err"; then
        echo "missing device diagnostic '$expected'" >&2
        cat "$source.err" >&2
        exit 1
    fi
    if [[ -s "$source.out" ]]; then
        echo "NN produced CPU output before GPU failure" >&2
        cat "$source.out" >&2
        exit 1
    fi
}

cat > "$TMP/parameter-device-mismatch.qui" <<'QUI'
import nn
import math

int | error run()
    nn.FC layer = try nn.FC(features_in = 2, features_out = 1)
    tensor<float32> samples_gpu = tensor.ones<float32>([1, 2], gpu = 0)
    tensor<float32> output = layer.forward(samples_gpu)
    print(output.shape()[1])
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
expect_device_failure "$TMP/parameter-device-mismatch.qui" "tensor operands are on different devices"

cat > "$TMP/same-device-compute.qui" <<'QUI'
import nn
import math

nn.FC layer
layer.weight = nn.Parameter<float32>( value = tensor.ones<float32>([1, 2], gpu = 0) )
layer.bias = nn.Parameter<float32>( value = tensor.zeros<float32>([1], gpu = 0) )
tensor<float32> samples = tensor.ones<float32>([1, 2], gpu = 0)
tensor<float32> output = layer.forward(samples)
print(output.shape()[0])
print(NL)
print(output.shape()[1])
print(NL)
print(output[0, 0].item())
print(NL)

tensor<float32> values = tensor.zeros<float32>([1, 2], gpu = 0)
values[0, 0] = float32(-1)
values[0, 1] = float32(1)
tensor<float32> activated = nn.relu(values)
tensor<float32> probabilities = nn.softmax(values)
print(activated[0, 0].item())
print(NL)
print(activated[0, 1].item())
print(NL)
float32 probability_total = probabilities[0, 0].item() + probabilities[0, 1].item()
print(probability_total > float32(0.9999) and probability_total < float32(1.0001))
print(NL)

nn.BatchNorm normalization
normalization.scale = nn.Parameter<float32>( value = tensor.ones<float32>([2], gpu = 0) )
normalization.bias = nn.Parameter<float32>( value = tensor.zeros<float32>([2], gpu = 0) )
normalization.running_mean = nn.State<float32>(value = tensor.zeros<float32>([2], gpu = 0))
normalization.running_variance = nn.State<float32>(value = tensor.ones<float32>([2], gpu = 0))
normalization.running_momentum = 0.1
normalization.variance_epsilon = 0.00001
tensor<float32> normalized = normalization.infer(
    tensor.ones<float32>([1, 2], gpu = 0)
)
print(normalized.shape()[1])
print(NL)
print(normalized[0, 0].item() > float32(0.9))
print(NL)
QUI

compute_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/same-device-compute.qui")"
compute_expected="$(printf '1\n1\n2.0\n0.0\n1.0\ntrue\n2\ntrue')"
if [[ "$compute_output" != "$compute_expected" ]]; then
    echo "unexpected same-GPU NN output:" >&2
    printf '%s\n' "$compute_output" >&2
    exit 1
fi

cat > "$TMP/convolution-gpu.qui" <<'QUI'
import nn
import math

tensor<float32> kernel = tensor.zeros<float32>([1, 1, 1, 1], gpu = 0)
kernel[0, 0, 0, 0] = float32(2)
nn.Conv2D convolution
convolution.weight = nn.Parameter<float32>(value = kernel)
convolution.bias = nn.Parameter<float32>( value = tensor.zeros<float32>([1], gpu = 0) )
convolution.step = 1
convolution.border = 0
tensor<float32> pixels = tensor.ones<float32>([1, 1, 2, 2], gpu = 0)
tensor<float32> filtered = convolution.forward(pixels)
print(filtered.shape()[2])
print(NL)
print(filtered.shape()[3])
print(NL)
print(filtered[0, 0, 1, 1].item())
print(NL)
QUI
conv_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/convolution-gpu.qui")"
conv_expected="$(printf '2\n2\n2.0')"
if [[ "$conv_output" != "$conv_expected" ]]; then
    echo "unexpected same-GPU convolution output:" >&2
    printf '%s\n' "$conv_output" >&2
    exit 1
fi

cat > "$TMP/tracked-gpu.qui" <<'QUI'
import nn
import math

class Model
    nn.FC dense

int | error run()
    nn.FC layer
    layer.weight = nn.Parameter<float32>( value = tensor.ones<float32>([1, 2], gpu = 0) )
    layer.bias = nn.Parameter<float32>( value = tensor.zeros<float32>([1], gpu = 0) )
    Model model
    model.dense = layer
    tensor<float32> samples = tensor.ones<float32>([1, 2], gpu = 0)
    tensor<float32> target = tensor.zeros<float32>([1, 1], gpu = 0)

    tensor<float32> tracked = samples.track()
    tensor<float32> prediction = model.dense.forward(tracked)
    tensor<float32> loss = nn.mse(prediction, target)
    print(prediction.untrack()[0, 0].item())
    print(NL)
    print(loss.item())
    print(NL)

    float32 before = model.dense.weight.raw()[0, 0].item()
    nn.SGD optimizer = try nn.SGD(rate = 0.1)
    optimizer.zero_grad(&model)
    loss.backward(&model)
    optimizer.step(&model)
    float32 after = model.dense.weight.raw()[0, 0].item()
    print(after < before)
    print(NL)
    print(after > float32(0.59) and after < float32(0.61))
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

tracked_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/tracked-gpu.qui")"
tracked_expected="$(printf '2.0\n4.0\ntrue\ntrue')"
if [[ "$tracked_output" != "$tracked_expected" ]]; then
    echo "unexpected tracked GPU NN output:" >&2
    printf '%s\n' "$tracked_output" >&2
    exit 1
fi



cat > "$TMP/training-gpu.qui" <<'QUI'
import nn
import math

class ConvModel
    nn.Conv2D convolution

class FCModel
    nn.FC dense

class ParameterModel
    nn.Parameter<float32> value

class Parameter64Model
    nn.Parameter<float> value

int | error run()
    tensor<float32> kernel = tensor.ones<float32>([1, 1, 1, 1], gpu = 0)
    nn.Conv2D convolution
    convolution.weight = nn.Parameter<float32>(value = kernel)
    convolution.bias = nn.Parameter<float32>( value = tensor.zeros<float32>([1], gpu = 0) )
    convolution.step = 1
    convolution.border = 0
    ConvModel conv_model
    conv_model.convolution = convolution
    tensor<float32> pixels = tensor.ones<float32>([1, 1, 2, 2], gpu = 0)
    tensor<float32> zero_image = tensor.zeros<float32>([1, 1, 2, 2], gpu = 0)
    tensor<float32> conv_prediction = conv_model.convolution.forward(pixels.track())
    tensor<float32> conv_loss = nn.mse(conv_prediction, zero_image)
    float32 conv_before = conv_model.convolution.weight.raw()[0, 0, 0, 0].item()
    nn.SGD sgd = nn.SGD(rate = 0.1)
    sgd.zero_grad(&conv_model)
    conv_loss.backward(&conv_model)
    sgd.step(&conv_model)
    print(conv_model.convolution.weight.raw()[0, 0, 0, 0].item() < conv_before)
    print(NL)

    nn.BatchNorm normalization
    normalization.scale = nn.Parameter<float32>( value = tensor.ones<float32>([2], gpu = 0) )
    normalization.bias = nn.Parameter<float32>( value = tensor.zeros<float32>([2], gpu = 0) )
    normalization.running_mean = nn.State<float32>(value = tensor.zeros<float32>([2], gpu = 0))
    normalization.running_variance = nn.State<float32>(value = tensor.ones<float32>([2], gpu = 0))
    normalization.running_momentum = 0.1
    normalization.variance_epsilon = 0.00001
    tensor<float32> norm_values = tensor.ones<float32>([2, 2], gpu = 0)
    tensor<float32> normalized = normalization.forward(norm_values.track())
    tensor<float32> norm_loss = math.mean((normalized * normalized))
    norm_loss.backward(&normalization)
    print(normalized.shape()[1])
    print(NL)
    print(normalization.infer(norm_values).shape()[1] == 2)
    print(NL)

    nn.Dropout dropout = try nn.Dropout(rate = 0.5, seed = 17)
    tensor<float32> dropped = dropout.forward(norm_values.track())
    tensor<float32> dropout_loss = math.mean((dropped * dropped))
    dropout_loss.backward(&norm_values)
    print(dropped.shape()[0])
    print(NL)
    print(dropout.infer(norm_values)[0, 0].item() == float32(1))
    print(NL)

    ParameterModel cpu_dropout_model
    cpu_dropout_model.value = nn.Parameter<float32>( value = tensor.ones<float32>([2, 2]) )
    ParameterModel gpu_dropout_model
    gpu_dropout_model.value = nn.Parameter<float32>( value = tensor.ones<float32>([2, 2], gpu = 0) )
    nn.Dropout cpu_dropout_backward = try nn.Dropout(rate = 0.5, seed = 29)
    nn.Dropout gpu_dropout_backward = try nn.Dropout(rate = 0.5, seed = 29)
    tensor<float32> cpu_dropout_output = cpu_dropout_backward.forward(
        cpu_dropout_model.value.track()
    )
    tensor<float32> gpu_dropout_output = gpu_dropout_backward.forward(
        gpu_dropout_model.value.track()
    )
    tensor<float32> cpu_dropout_loss = math.mean(cpu_dropout_output)
    tensor<float32> gpu_dropout_loss = math.mean(gpu_dropout_output)
    nn.SGD cpu_dropout_sgd = nn.SGD(rate = 0.1)
    nn.SGD gpu_dropout_sgd = nn.SGD(rate = 0.1)
    cpu_dropout_sgd.zero_grad(&cpu_dropout_model)
    gpu_dropout_sgd.zero_grad(&gpu_dropout_model)
    cpu_dropout_loss.backward(&cpu_dropout_model)
    gpu_dropout_loss.backward(&gpu_dropout_model)
    cpu_dropout_sgd.step(&cpu_dropout_model)
    gpu_dropout_sgd.step(&gpu_dropout_model)
    tensor<float32> gpu_dropout_weight = gpu_dropout_model.value.raw().cpu()
    print(
        cpu_dropout_model.value.raw()[0, 0].item() == gpu_dropout_weight[0, 0].item()
        and cpu_dropout_model.value.raw()[0, 1].item() == gpu_dropout_weight[0, 1].item()
        and cpu_dropout_model.value.raw()[1, 0].item() == gpu_dropout_weight[1, 0].item()
        and cpu_dropout_model.value.raw()[1, 1].item() == gpu_dropout_weight[1, 1].item()
    )
    print(NL)

    ParameterModel accumulation_model
    accumulation_model.value = nn.Parameter<float32>( value = tensor.ones<float32>([1], gpu = 0) )
    tensor<float32> accumulation_value = accumulation_model.value.track()
    tensor<float32> accumulation_loss = math.mean((accumulation_value + accumulation_value))
    nn.SGD accumulation_sgd = nn.SGD(rate = 0.1)
    accumulation_sgd.zero_grad(&accumulation_model)
    accumulation_loss.backward(&accumulation_model)
    accumulation_sgd.step(&accumulation_model)
    float32 accumulation_after = accumulation_model.value.raw()[0].item()
    print(
        accumulation_after > float32(0.7999)
        and accumulation_after < float32(0.8001)
    )
    print(NL)

    nn.FC dense
    dense.weight = nn.Parameter<float32>( value = tensor.ones<float32>([1, 1], gpu = 0) )
    dense.bias = nn.Parameter<float32>( value = tensor.zeros<float32>([1], gpu = 0) )
    FCModel model
    model.dense = dense
    nn.Adam adam = nn.Adam(
        rate = 0.1,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 0.00000001
    )
    tensor<float32> sample = tensor.ones<float32>([1, 1], gpu = 0)
    tensor<float32> target = tensor.zeros<float32>([1, 1], gpu = 0)
    float32 before = model.dense.weight.raw()[0, 0].item()
    tensor<float32> prediction = model.dense.forward(sample.track())
    tensor<float32> loss = nn.mse(prediction, target)
    adam.zero_grad(&model)
    loss.backward(&model)
    adam.step(&model)
    print(model.dense.weight.raw()[0, 0].item() != before)
    print(NL)
    float32 first_after = model.dense.weight.raw()[0, 0].item()
    print(first_after < before)
    print(NL)
    tensor<float32> second_prediction = model.dense.forward(sample.track())
    tensor<float32> second_loss = nn.mse(second_prediction, target)
    adam.zero_grad(&model)
    second_loss.backward(&model)
    adam.step(&model)
    print(model.dense.weight.raw()[0, 0].item() != first_after)
    print(NL)
    print(model.dense.weight.raw()[0, 0].item() < first_after)
    print(NL)

    ParameterModel zero_adam_model
    zero_adam_model.value = nn.Parameter<float32>(
        value = tensor.ones<float32>([1], gpu = 0) * float32(3)
    )
    nn.Adam zero_adam = nn.Adam(rate = 0.1)
    tensor<float32> zero_tracked = zero_adam_model.value.track()
    tensor<float32> zero_loss = math.mean((zero_tracked * float32(0)))
    zero_adam.zero_grad(&zero_adam_model)
    zero_loss.backward(&zero_adam_model)
    float32 zero_before = zero_adam_model.value.raw()[0].item()
    zero_adam.step(&zero_adam_model)
    print(zero_adam_model.value.raw()[0].item() == zero_before)
    print(NL)

    Parameter64Model zero_adam_model64
    zero_adam_model64.value = nn.Parameter<float>(
        value = tensor.ones<float>([1], gpu = 0) * 3.0
    )
    nn.Adam zero_adam64 = nn.Adam(rate = 0.1)
    tensor<float> zero_tracked64 = zero_adam_model64.value.track()
    tensor<float> zero_loss64 = math.mean((zero_tracked64 * 0.0))
    zero_adam64.zero_grad(&zero_adam_model64)
    zero_loss64.backward(&zero_adam_model64)
    float zero_before64 = zero_adam_model64.value.raw()[0].item()
    zero_adam64.step(&zero_adam_model64)
    print(zero_adam_model64.value.raw()[0].item() == zero_before64)
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

training_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/training-gpu.qui")"
training_expected="$(printf 'true\n2\ntrue\n2\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$training_output" != "$training_expected" ]]; then
    echo "unexpected GPU training output:" >&2
    printf '%s\n' "$training_output" >&2
    exit 1
fi

equivalence_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$REPOSITORY_ROOT/tests/fake_gpu_equivalence.qui")"
equivalence_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$equivalence_output" != "$equivalence_expected" ]]; then
    echo "fake GPU CPU-equivalence regression failed:" >&2
    printf '%s\n' "$equivalence_output" >&2
    exit 1
fi

cat > "$TMP/all-reduce.qui" <<'QUI'
import nn
import math

int | error run()
    tensor<float32> first = tensor.ones<float32>([2], gpu = 0)
    tensor<float32> second = (tensor.ones<float32>([2]) * float32(2)).gpu(1)
    tensor<float32>[] values = [first, second]
    try nn.all_reduce_sum(&values, [0, 1])
    tensor<float32> reduced0 = values[0].cpu()
    tensor<float32> reduced1 = values[1].cpu()
    print(reduced0[0].item() == float32(3))
    print(NL)
    print(reduced1[1].item() == float32(3))
    print(NL)

    tensor<float32>[] cpu_values = [
        tensor.ones<float32>([1]),
        tensor.ones<float32>([1]) * float32(4)
    ]
    try nn.all_reduce_sum(&cpu_values, [-1, -1])
    print(cpu_values[0][0].item() == float32(5))
    print(NL)
    print(cpu_values[1][0].item() == float32(5))
    print(NL)

    tensor<float32>[] tracked = [
        tensor.ones<float32>([1]).track(),
        tensor.ones<float32>([1])
    ]
    void | error tracked_result = nn.all_reduce_sum(&tracked, [-1, -1])
    bool tracked_rejected = false
    match tracked_result
        void
            tracked_rejected = false
        error
            tracked_rejected = true
    print(tracked_rejected)
    print(NL)

    tensor<float32>[] mismatched = [
        tensor.ones<float32>([1]),
        tensor.ones<float32>([2])
    ]
    void | error shape_result = nn.all_reduce_sum(&mismatched, [-1, -1])
    bool shape_rejected = false
    match shape_result
        void
            shape_rejected = false
        error
            shape_rejected = true
    print(shape_rejected)
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

all_reduce_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/all-reduce.qui")"
all_reduce_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$all_reduce_output" != "$all_reduce_expected" ]]; then
    echo "unexpected NN all-reduce output:" >&2
    printf '%s\n' "$all_reduce_output" >&2
    exit 1
fi

echo "nn device contracts: ok"
