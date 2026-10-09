#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/make-state.qui" <<QUI
import nn

class Leaf
    nn.Parameter<real32> weight
    nn.State<real32> running

class Model
    Leaf[] layers
    nn.Parameter<real64> scale
    nn.State<real64> total

Model make_model()
    Leaf first
    first.weight = nn.Parameter<real32>(
        value = tensor.ones<real32>([1]) * real32(2)
    )
    first.running = nn.State<real32>(
        value = tensor.ones<real32>([1]) * real32(3)
    )
    Leaf second
    second.weight = nn.Parameter<real32>(
        value = tensor.ones<real32>([1]) * real32(4)
    )
    second.running = nn.State<real32>(
        value = tensor.ones<real32>([1]) * real32(5)
    )
    Model model
    model.layers = [first, second]
    model.scale = nn.Parameter<real64>(
        value = tensor.ones<real64>([1]) * 6.0
    )
    model.total = nn.State<real64>(
        value = tensor.ones<real64>([1]) * 7.0
    )
    return model

Model model = make_model()
nn.Parameter<real32> first_identity = model.layers[0].weight
nn.State<real32> state_identity = model.layers[0].running
nn.save(model, path = "$TMP/model-base.dnn")

model.layers[0].weight.replace(tensor.zeros<real32>([1]))
model.layers[0].running.replace(tensor.zeros<real32>([1]))
model.layers[1].weight.replace(tensor.zeros<real32>([1]))
model.layers[1].running.replace(tensor.zeros<real32>([1]))
model.scale.replace(tensor.zeros<real64>([1]))
model.total.replace(tensor.zeros<real64>([1]))
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
print(first_identity.same(model.layers[0].weight))
print(NL)
print(state_identity.same(model.layers[0].running))
print(NL)

nn.Adam optimizer = nn.Adam(rate = 0.01)
optimizer.step(&model)
nn.checkpoint(model, optimizer, path = "$TMP/checkpoint-base.dnn")
QUI

normal_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/make-state.qui")"
normal_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$normal_output" != "$normal_expected" ]]; then
    echo "unexpected canonical NN persistence output:" >&2
    printf '%s\n' "$normal_output" >&2
    exit 1
fi

python3 - "$TMP" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])

def load(name):
    return (root / name).read_text(encoding="utf-8").splitlines()

def save(name, lines):
    (root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

model = load("model-base.dnn")
assert model[0] == "QUIDRA_DNN_STATE"
assert model[1] == "2"
assert model[2] == "model"
assert model[8] == "Parameter"
assert model[9] == "Parameter<real32>"
assert model[10] == "real32"
assert model[12] == "1"
assert model[13] == "1"
assert model[14] == "1"
assert model[-2] == "ADLER65521-V1"

cases = {}

x = model.copy(); x[0] = "BROKEN_MAGIC"; cases["wrong-magic"] = x
x = model.copy(); x[1] = "999"; cases["unsupported-version"] = x
x = model.copy(); x[3] = "OtherModel"; cases["wrong-root"] = x
x = model.copy(); x[8] = "State"; cases["wrong-kind"] = x
x = model.copy(); x[9] = "State<real32>"; cases["wrong-field-type"] = x
x = model.copy(); x[10] = "real64"; cases["wrong-dtype"] = x
x = model.copy(); x[11] = "wrong.path"; cases["wrong-path"] = x

x = model.copy()
x[12] = "0"
del x[13]
cases["wrong-rank"] = x

x = model.copy()
x[13] = "2"
x[14] = "2"
x.insert(16, x[15])
cases["wrong-shape"] = x

x = model.copy(); x[14] = "0"; cases["wrong-count"] = x
x = model.copy()
x[13] = "9223372036854775807"
x[14] = "9223372036854775807"
cases["oversized-shape"] = x
x = model[:-1]; cases["truncated"] = x
x = model.copy(); x[-1] = "0:0"; cases["corrupt-checksum"] = x
x = model.copy(); x.append("TRAILING_DATA"); cases["trailing-data"] = x

for name, lines in cases.items():
    save(f"model-{name}.dnn", lines)

checkpoint = load("checkpoint-base.dnn")
assert checkpoint[0] == "QUIDRA_DNN_STATE"
assert checkpoint[1] == "3"
assert checkpoint[2] == "model+Adam"
adam = checkpoint.index("Adam")
assert int(checkpoint[adam + 8]) >= 1

x = checkpoint.copy()
x[adam] = "SGD"
save("checkpoint-wrong-optimizer.dnn", x)

x = checkpoint.copy()
x[adam + 8] = str(int(x[adam + 8]) + 1)
save("checkpoint-wrong-binding.dnn", x)

first = checkpoint.index("AdamFirst", adam + 1)
x = checkpoint.copy()
x[first + 2] = "real64"
save("checkpoint-wrong-moment-dtype.dnn", x)

x = checkpoint.copy()
x[first + 3] = "wrong.optimizer.path"
save("checkpoint-wrong-binding-path.dnn", x)

x = checkpoint.copy()
assert x[first + 4] == "1"
assert x[first + 5] == "1"
assert x[first + 6] == "1"
x[first + 5] = "2"
x[first + 6] = "2"
x.insert(first + 8, x[first + 7])
save("checkpoint-wrong-moment-shape.dnn", x)

x = checkpoint.copy()
x[first + 5] = "9223372036854775807"
x[first + 6] = "9223372036854775807"
save("checkpoint-oversized-moment-shape.dnn", x)

save("checkpoint-truncated.dnn", checkpoint[:-1])

x = checkpoint.copy()
x[-1] = "0:0"
save("checkpoint-corrupt-checksum.dnn", x)

x = checkpoint.copy()
x.append("TRAILING_DATA")
save("checkpoint-trailing-data.dnn", x)
PY

write_model_failure_program() {
    local bad="$1"
    local before="$2"
    local after="$3"
    cat > "$TMP/model-failure.qui" <<QUI
import nn

class Leaf
    nn.Parameter<real32> weight
    nn.State<real32> running

class Model
    Leaf[] layers
    nn.Parameter<real64> scale
    nn.State<real64> total

Model make_model()
    Leaf first
    first.weight = nn.Parameter<real32>(
        value = tensor.ones<real32>([1]) * real32(2)
    )
    first.running = nn.State<real32>(
        value = tensor.ones<real32>([1]) * real32(3)
    )
    Leaf second
    second.weight = nn.Parameter<real32>(
        value = tensor.ones<real32>([1]) * real32(4)
    )
    second.running = nn.State<real32>(
        value = tensor.ones<real32>([1]) * real32(5)
    )
    Model model
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
        print(true)
        print(NL)
nn.save(model, path = "$after")
QUI
}

for case in     wrong-magic unsupported-version wrong-root wrong-kind wrong-field-type     wrong-dtype wrong-path wrong-rank wrong-shape wrong-count oversized-shape truncated     corrupt-checksum trailing-data
do
    before="$TMP/model-$case-before.dnn"
    after="$TMP/model-$case-after.dnn"
    write_model_failure_program "$TMP/model-$case.dnn" "$before" "$after"
    output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/model-failure.qui")"
    if [[ "$output" != "true" ]]; then
        echo "NN model corruption case unexpectedly loaded: $case" >&2
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

class Leaf
    nn.Parameter<real32> weight
    nn.State<real32> running

class Model
    Leaf[] layers
    nn.Parameter<real64> scale
    nn.State<real64> total

Model make_model()
    Leaf first
    first.weight = nn.Parameter<real32>(
        value = tensor.ones<real32>([1]) * real32(2)
    )
    first.running = nn.State<real32>(
        value = tensor.ones<real32>([1]) * real32(3)
    )
    Leaf second
    second.weight = nn.Parameter<real32>(
        value = tensor.ones<real32>([1]) * real32(4)
    )
    second.running = nn.State<real32>(
        value = tensor.ones<real32>([1]) * real32(5)
    )
    Model model
    model.layers = [first, second]
    model.scale = nn.Parameter<real64>(
        value = tensor.ones<real64>([1]) * 6.0
    )
    model.total = nn.State<real64>(
        value = tensor.ones<real64>([1]) * 7.0
    )
    return model

Model model = make_model()
nn.Adam optimizer = nn.Adam(rate = 0.01)
optimizer.step(&model)
nn.checkpoint(model, optimizer, path = "$before")
void | error loaded = nn.resume(&model, &optimizer, path = "$bad")
match loaded
    void
        print(false)
        print(NL)
    error problem
        print(true)
        print(NL)
nn.checkpoint(model, optimizer, path = "$after")
QUI
}

for case in \
    wrong-optimizer wrong-binding wrong-binding-path wrong-moment-dtype wrong-moment-shape \
    oversized-moment-shape truncated corrupt-checksum trailing-data
do
    before="$TMP/checkpoint-$case-before.dnn"
    after="$TMP/checkpoint-$case-after.dnn"
    write_checkpoint_failure_program "$TMP/checkpoint-$case.dnn" "$before" "$after"
    output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/checkpoint-failure.qui")"
    if [[ "$output" != "true" ]]; then
        echo "NN checkpoint corruption case unexpectedly loaded: $case" >&2
        exit 1
    fi
    if ! cmp -s "$before" "$after"; then
        echo "NN model/optimizer changed after failed checkpoint load: $case" >&2
        exit 1
    fi
done

echo "nn persistence failures: ok"
