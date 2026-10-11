#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"

cat > "$TMP/make-state.qui" <<QUI
import nn
import math

class Leaf()
    nn.Parameter<real32> weight
    nn.State<real32> running

class Model()
    Leaf[] layers
    nn.Parameter<real64> scale
    nn.State<real64> total

Model make_model()
    Leaf first = Leaf()
    first.weight = nn.Parameter<real32>(
        value = tensor.ones<real32>([1]) * real32(2)
    )
    first.running = nn.State<real32>(
        value = tensor.ones<real32>([1]) * real32(3)
    )
    Leaf second = Leaf()
    second.weight = nn.Parameter<real32>(
        value = tensor.ones<real32>([1]) * real32(4)
    )
    second.running = nn.State<real32>(
        value = tensor.ones<real32>([1]) * real32(5)
    )
    Model model = Model()
    model.layers = [first, second]
    model.scale = nn.Parameter<real64>(
        value = tensor.ones<real64>([1]) * 6.0
    )
    model.total = nn.State<real64>(
        value = tensor.ones<real64>([1]) * 7.0
    )
    return model

Model model = make_model()
nn.Parameter<real32> &first_identity = &model.layers[0].weight
nn.State<real32> &state_identity = &model.layers[0].running
nn.save(model, path = "$TMP/model-base.dnn")

.model.layers[0].weight.replace(tensor.zeros<real32>([1]))
.model.layers[0].running.replace(tensor.zeros<real32>([1]))
.model.layers[1].weight.replace(tensor.zeros<real32>([1]))
.model.layers[1].running.replace(tensor.zeros<real32>([1]))
.model.scale.replace(tensor.zeros<real64>([1]))
.model.total.replace(tensor.zeros<real64>([1]))
nn.load(&model, path = "$TMP/model-base.dnn")
print(model.layers[0].weight.raw()[0].item() == real32(2))
print(NL)
print(model.layers[0].running.raw()[0].item() == real32(3))
print(NL)
print(model.layers[1].weight.raw()[0].item() == real32(4))
print(NL)
print(model.layers[1].running.raw()[0].item() == real32(5))
print(NL)
print(model.scale.raw()[0].item() == 6.0)
print(NL)
print(model.total.raw()[0].item() == 7.0)
print(NL)
print(&first_identity == &model.layers[0].weight)
print(NL)
print(&state_identity == &model.layers[0].running)
print(NL)

auto optimizer = nn.Adam(&model, rate = 0.01)
tensor<real32> first_weight = .model.layers[0].weight.track()
tensor<real32> second_weight = .model.layers[1].weight.track()
math.sum(first_weight * second_weight).backward(&model)
tensor<real64> scale = .model.scale.track()
math.sum(scale * scale).backward(&model)
.optimizer.update()
nn.checkpoint(&model, optimizer, path = "$TMP/checkpoint-base.dnn")
QUI

normal_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/make-state.qui")"
normal_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$normal_output" != "$normal_expected" ]]; then
    echo "unexpected canonical NN persistence output:" >&2
    printf '%s\n' "$normal_output" >&2
    exit 1
fi

python3 - "$TMP" "$REPOSITORY_ROOT" <<'PY'
from pathlib import Path
import copy
import sys
sys.path.insert(0, str(Path(sys.argv[2]) / "tests"))
from checkpoint_format5 import decode, encode, seal
root = Path(sys.argv[1])

def save(name, text):
    (root / name).write_text(text, encoding="utf-8")

def changed(document, name, transform):
    item = copy.deepcopy(document)
    transform(item)
    save(name, encode(item))

model = decode(root / "model-base.dnn")
text = encode(model)
save("model-wrong-magic.dnn", text.replace("QUIDRA_DNN_STATE", "BROKEN_MAGIC", 1))
save("model-unsupported-version.dnn", text.replace("\n5\n", "\n999\n", 1))
save("model-prep-version.dnn", text.replace("\n5\n", "\n4\n", 1))
changed(model, "model-wrong-root.dnn", lambda d: d.update(root="OtherModel"))
changed(model, "model-wrong-kind.dnn", lambda d: d["entries"][0].update(kind="State"))
changed(model, "model-wrong-field-type.dnn", lambda d: d["entries"][0].update(kind="Unknown"))
changed(model, "model-wrong-dtype.dnn", lambda d: d["entries"][0].update(dtype="real64"))
changed(model, "model-wrong-path.dnn", lambda d: d["entries"][0].update(path="wrong.path"))
changed(model, "model-wrong-rank.dnn", lambda d: d["entries"][0].update(shape=[]))
changed(model, "model-wrong-shape.dnn", lambda d: d["entries"][0].update(shape=[2], values=d["entries"][0]["values"] * 2))
x = text.splitlines(); x[10] = "0"; save("model-wrong-count.dnn", "\n".join(seal(x[:-2])) + "\n")
changed(model, "model-oversized-shape.dnn", lambda d: d["entries"][0].update(shape=[9223372036854775807], values=[]))
save("model-truncated.dnn", "\n".join(text.splitlines()[:-1]) + "\n")
save("model-corrupt-checksum.dnn", "\n".join(text.splitlines()[:-1] + ["0:0"]) + "\n")
save("model-trailing-data.dnn", text + "TRAILING_DATA\n")
checkpoint = decode(root / "checkpoint-base.dnn")
text = encode(checkpoint)
save("checkpoint-wrong-optimizer.dnn", text.replace("\nAdam\n", "\nSGD\n", 1))
x = text.splitlines(); adam = x.index("Adam"); x[adam + 5] = str(int(x[adam + 5]) + 1); save("checkpoint-wrong-binding.dnn", "\n".join(seal(x[:-2])) + "\n")
save("checkpoint-prep-version.dnn", text.replace("\n5\n", "\n4\n", 1))
step = text.splitlines().index("AdamStep")
x = text.splitlines(); x[step] = "AdamSteps"; save("checkpoint-wrong-step-kind.dnn", "\n".join(seal(x[:-2])) + "\n")
changed(checkpoint, "checkpoint-wrong-step-path.dnn", lambda d: d["slots"][0].update(path="wrong.step.path"))
changed(checkpoint, "checkpoint-negative-step.dnn", lambda d: d["slots"][0]["step"].__setitem__(0, "-1"))
changed(checkpoint, "checkpoint-zero-step-correction.dnn", lambda d: d["slots"][0]["step"].__setitem__(0, "0"))
changed(checkpoint, "checkpoint-step-correction-range.dnn", lambda d: d["slots"][0]["step"].__setitem__(1, "1.5"))
x = text.splitlines(); del x[step:step + 5]; save("checkpoint-missing-step.dnn", "\n".join(seal(x[:-2])) + "\n")
for version in ["2", "3"]:
    save("checkpoint-legacy-v" + version + ".dnn", encode(checkpoint, version))
changed(checkpoint, "checkpoint-wrong-moment-dtype.dnn", lambda d: d["slots"][0]["first"].update(dtype="real64"))
changed(checkpoint, "checkpoint-wrong-binding-path.dnn", lambda d: d["slots"][0]["first"].update(path="wrong.optimizer.path"))
changed(checkpoint, "checkpoint-wrong-moment-shape.dnn", lambda d: d["slots"][0]["first"].update(shape=[2], values=d["slots"][0]["first"]["values"] * 2))
changed(checkpoint, "checkpoint-oversized-moment-shape.dnn", lambda d: d["slots"][0]["first"].update(shape=[9223372036854775807], values=[]))
save("checkpoint-truncated.dnn", "\n".join(text.splitlines()[:-1]) + "\n")
save("checkpoint-corrupt-checksum.dnn", "\n".join(text.splitlines()[:-1] + ["0:0"]) + "\n")
save("checkpoint-trailing-data.dnn", text + "TRAILING_DATA\n")
PY

write_model_failure_program() {
    local bad="$1"
    local before="$2"
    local after="$3"
    cat > "$TMP/model-failure.qui" <<QUI
import nn

class Leaf()
    nn.Parameter<real32> weight
    nn.State<real32> running

class Model()
    Leaf[] layers
    nn.Parameter<real64> scale
    nn.State<real64> total

Model make_model()
    Leaf first = Leaf()
    first.weight = nn.Parameter<real32>(
        value = tensor.ones<real32>([1]) * real32(2)
    )
    first.running = nn.State<real32>(
        value = tensor.ones<real32>([1]) * real32(3)
    )
    Leaf second = Leaf()
    second.weight = nn.Parameter<real32>(
        value = tensor.ones<real32>([1]) * real32(4)
    )
    second.running = nn.State<real32>(
        value = tensor.ones<real32>([1]) * real32(5)
    )
    Model model = Model()
    model.layers = [first, second]
    model.scale = nn.Parameter<real64>(
        value = tensor.ones<real64>([1]) * 6.0
    )
    model.total = nn.State<real64>(
        value = tensor.ones<real64>([1]) * 7.0
    )
    return model

Model model = make_model()
nn.save(model, path = "$before")
void | error loaded = nn.load(&model, path = "$bad")
match loaded
    void
        print(false)
        print(NL)
    error problem
        print(problem.code)
        print(NL)
nn.save(model, path = "$after")
QUI
}

# Each case is rejected with its code: the file's contents (NN_CHECKPOINT_FILE)
# or a file that describes another model (NN_CHECKPOINT).
for entry in \
    wrong-magic:NN_CHECKPOINT_FILE unsupported-version:NN_CHECKPOINT_FILE prep-version:NN_CHECKPOINT_FILE \
    wrong-root:NN_CHECKPOINT wrong-kind:NN_CHECKPOINT \
    wrong-field-type:NN_CHECKPOINT_FILE wrong-dtype:NN_CHECKPOINT \
    wrong-path:NN_CHECKPOINT wrong-rank:NN_CHECKPOINT \
    wrong-shape:NN_CHECKPOINT wrong-count:NN_CHECKPOINT_FILE \
    oversized-shape:NN_CHECKPOINT_FILE truncated:NN_CHECKPOINT_FILE \
    corrupt-checksum:NN_CHECKPOINT_FILE trailing-data:NN_CHECKPOINT_FILE
do
    case="${entry%%:*}"
    code="${entry#*:}"
    before="$TMP/model-$case-before.dnn"
    after="$TMP/model-$case-after.dnn"
    write_model_failure_program "$TMP/model-$case.dnn" "$before" "$after"
    output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/model-failure.qui")"
    if [[ "$output" == "false" ]]; then
        echo "NN model corruption case unexpectedly loaded: $case" >&2
        exit 1
    fi
    if [[ "$output" != "$code" ]]; then
        echo "NN model corruption case $case reported $output instead of $code" >&2
        exit 1
    fi
    if ! cmp -s "$before" "$after"; then
        echo "NN model changed after failed load: $case" >&2
        exit 1
    fi
done

write_checkpoint_failure_program() {
    local bad="$1"
    local before="$2"
    local after="$3"
    cat > "$TMP/checkpoint-failure.qui" <<QUI
import nn
import math

class Leaf()
    nn.Parameter<real32> weight
    nn.State<real32> running

class Model()
    Leaf[] layers
    nn.Parameter<real64> scale
    nn.State<real64> total

Model make_model()
    Leaf first = Leaf()
    first.weight = nn.Parameter<real32>(
        value = tensor.ones<real32>([1]) * real32(2)
    )
    first.running = nn.State<real32>(
        value = tensor.ones<real32>([1]) * real32(3)
    )
    Leaf second = Leaf()
    second.weight = nn.Parameter<real32>(
        value = tensor.ones<real32>([1]) * real32(4)
    )
    second.running = nn.State<real32>(
        value = tensor.ones<real32>([1]) * real32(5)
    )
    Model model = Model()
    model.layers = [first, second]
    model.scale = nn.Parameter<real64>(
        value = tensor.ones<real64>([1]) * 6.0
    )
    model.total = nn.State<real64>(
        value = tensor.ones<real64>([1]) * 7.0
    )
    return model

Model model = make_model()
auto optimizer = nn.Adam(&model, rate = 0.01)
for 0 <= nat round < 2
    tensor<real32> first_weight = .model.layers[0].weight.track()
    tensor<real32> second_weight = .model.layers[1].weight.track()
    math.sum(first_weight * second_weight).backward(&model)
    tensor<real64> scale = .model.scale.track()
    math.sum(scale * scale).backward(&model)
    .optimizer.update()
nn.checkpoint(&model, optimizer, path = "$before")
void | error loaded = nn.resume(&model, &optimizer, path = "$bad")
match loaded
    void
        print(false)
        print(NL)
    error problem
        print(problem.code)
        print(NL)
nn.checkpoint(&model, optimizer, path = "$after")
QUI
}

# Besides the file's contents, a checkpoint of another optimizer is
# NN_CHECKPOINT and one bound to other Parameters NN_BINDING.
for entry in \
    wrong-optimizer:NN_CHECKPOINT_FILE wrong-binding:NN_CHECKPOINT_FILE prep-version:NN_CHECKPOINT_FILE \
    wrong-binding-path:NN_CHECKPOINT_FILE wrong-moment-dtype:NN_CHECKPOINT \
    wrong-moment-shape:NN_CHECKPOINT oversized-moment-shape:NN_CHECKPOINT_FILE \
    truncated:NN_CHECKPOINT_FILE corrupt-checksum:NN_CHECKPOINT_FILE \
    trailing-data:NN_CHECKPOINT_FILE wrong-step-kind:NN_CHECKPOINT_FILE \
    wrong-step-path:NN_CHECKPOINT_FILE negative-step:NN_CHECKPOINT_FILE \
    zero-step-correction:NN_CHECKPOINT_FILE step-correction-range:NN_CHECKPOINT_FILE \
    missing-step:NN_CHECKPOINT_FILE
do
    case="${entry%%:*}"
    code="${entry#*:}"
    before="$TMP/checkpoint-$case-before.dnn"
    after="$TMP/checkpoint-$case-after.dnn"
    write_checkpoint_failure_program "$TMP/checkpoint-$case.dnn" "$before" "$after"
    output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/checkpoint-failure.qui")"
    if [[ "$output" == "false" ]]; then
        echo "NN checkpoint corruption case unexpectedly loaded: $case" >&2
        exit 1
    fi
    if [[ "$output" != "$code" ]]; then
        echo "NN checkpoint corruption case $case reported $output instead of $code" >&2
        exit 1
    fi
    if ! cmp -s "$before" "$after"; then
        echo "NN model/optimizer changed after failed checkpoint load: $case" >&2
        exit 1
    fi
done

for version in v3 v2
do
    cat > "$TMP/checkpoint-legacy.qui" <<QUI
import nn

class Leaf()
    nn.Parameter<real32> weight
    nn.State<real32> running

class Model()
    Leaf[] layers
    nn.Parameter<real64> scale
    nn.State<real64> total

Model make_model()
    Leaf first = Leaf()
    first.weight = nn.Parameter<real32>(value = tensor.zeros<real32>([1]))
    first.running = nn.State<real32>(value = tensor.zeros<real32>([1]))
    Leaf second = Leaf()
    second.weight = nn.Parameter<real32>(value = tensor.zeros<real32>([1]))
    second.running = nn.State<real32>(value = tensor.zeros<real32>([1]))
    Model model = Model()
    model.layers = [first, second]
    model.scale = nn.Parameter<real64>(value = tensor.zeros<real64>([1]))
    model.total = nn.State<real64>(value = tensor.zeros<real64>([1]))
    return model

Model model = make_model()
auto optimizer = nn.Adam(&model, rate = 0.5)
nn.resume(&model, &optimizer, path = "$TMP/checkpoint-legacy-$version.dnn")
nn.checkpoint(&model, optimizer, path = "$TMP/checkpoint-legacy-$version-resaved.dnn")
QUI
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/checkpoint-legacy.qui"
    if ! cmp -s "$TMP/checkpoint-base.dnn" "$TMP/checkpoint-legacy-$version-resaved.dnn"; then
        echo "NN $version checkpoint did not load as per-Parameter Adam state" >&2
        diff "$TMP/checkpoint-base.dnn" "$TMP/checkpoint-legacy-$version-resaved.dnn" >&2 || true
        exit 1
    fi
done

echo "nn persistence failures: ok"
