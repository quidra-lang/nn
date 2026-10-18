# Quidra NN

Quidra NN is the first-party generic neural-network package, imported as `nn`.

Dependency layers:

```text
Layer 1: Core
Layer 2: Math
Layer 3: NN / Vision / Video
Layer 4: DNN
```

Layer numbers constrain dependency direction; they do not imply that every
higher-layer package depends on every package below it. NN depends on Core and
Math. DNN currently depends on Core, Math, and NN; it does not require Vision
or Video.

Core owns language, tensor/autograd/device substrate, basic operators, and generic extension mechanisms. Math owns generic mathematical semantics. NN owns reusable neural-network semantics: Parameters/State, layers, activations, losses, optimizers, model-state persistence, distributed-training collectives, NN-native kernels, and NN compiler-extension policy. DNN sits above NN and owns deep model/architecture semantics such as AlexNet, VGG, and ResNet.

```quidra
import nn

nn.FC layer = nn.FC(128, 64)
tensor<real32> output = nn.relu(.layer.forward(input))
auto optimizer = nn.Adam(&model, 0.001)
loss.backward(&model)
.optimizer.update()
```

`nn.FC` and `nn.Conv2D` require their dimensions even when you supply
prepared `Parameter<real32>` values. Pass `initial_weight` and `initial_bias`
together to keep those values instead of generating uniform weights. Supplying
only one returns `NN_ARGUMENT`. Omitting both keeps ordinary initialization.

`nn.BatchNorm` accepts a complete prepared state with `initial_scale`,
`initial_bias`, `initial_mean`, `initial_variance`, `initial_momentum` and
`initial_epsilon`. The first two are Parameters, the next two are States, and
the final two are real32 configuration values. Supply all six or omit all six;
a partial set returns `NN_ARGUMENT`. Supplied momentum must be finite and in
[0, 1), and supplied epsilon must be finite and positive; invalid settings return
`NN_ARGUMENT` before the fields are stored. Feature and option validation still applies.
These prepared paths store the values in field order without generating tensors,
random values or floating-point arithmetic. Existing forward shape checks apply
to supplied tensors.

`nn.Adam<M>` and `nn.SGD<M>` hold a reference to their model. Keep the
model and optimizer in separate variables: storing the optimizer inside the
model would create an interior self-reference. Construction records each
Parameter's shape and device and allocates the optimizer's buffers. Updates
visit Parameters in place through `nn.parameters`, including private value
fields, array elements and tuple elements; reference fields are not followed.
A changed Parameter count, shape or device requires a new optimizer.

`Parameter<T>` and `State<T>` hold ordinary values. Copying a Parameter makes
an independent value and gradient destination. Share a weight by keeping one
owner and passing a reference to it where needed.

`backward(&model)` accumulates gradients. `.optimizer.update()` consumes the
gradients it uses and skips Parameters without gradients, keeping their
values, moments and step counts unchanged. `nn.clear_grad(&model)` discards
gradients without an update; an absent gradient differs from a zero gradient.

`update()` returns `void | error`. It checks the model's structure and every
pending gradient before any update. A failed structure or gradient check
changes nothing. Once those checks pass, each Parameter is updated in traversal
order and its gradient is consumed. Adam keeps one scratch buffer and two
in-place moment buffers per Parameter. A computation-time failure leaves model
and optimizer state unspecified; reload a checkpoint and create a new optimizer
before training again. Such returned failures carry `NN_UPDATE_FAILED`; numeric
traps keep Core's codes. Device failures detected later are reported at the next
synchronization. Bare marked calls fail fast; `try .optimizer.update()`
propagates a returned error.

For real16 and real16b Parameters, optimizer arithmetic uses exactly widened
real32 weights and gradients. The new weight is rounded once to its original
type at storage; Adam's moments stay real32. Model operations and gradients
retain their own element type. Real32 and real64 updates retain their existing
operation order.

## Explicit mixed precision

Ordinary training defaults to real32. For explicitly chosen low-precision
forward computation, keep `nn.Parameter<real32>` as the original weight and
cast its tracked tensor to real16 or real16b inside `forward`. Adam stays
bound to that real32 model and updates the original. Each forward casts the
current value again; the low-precision tensor is a temporary computation
value, not another Parameter.

```quidra
class Scale16()

    nn.Parameter<real32> weight
    weight = nn.Parameter<real32>(value = tensor.ones<real32>([1]))

    tensor<real16> .forward(tensor<real16> input)
        tensor<real16> low_weight = real16(.weight.track())
        return input * low_weight

class Scale16b()

    nn.Parameter<real32> weight
    weight = nn.Parameter<real32>(value = tensor.ones<real32>([1]))

    tensor<real16b> .forward(tensor<real16b> input)
        tensor<real16b> low_weight = real16b(.weight.track())
        return input * low_weight
```

Initialize the complete model first, then keep its optimizer in a separate
variable: `auto optimizer = nn.Adam(&model, rate = 0.001)`. After a loss uses
`.model.forward(input)`, call `loss.backward(&model)` and
`.optimizer.update()`. Do not copy the Parameter into a second model to use
it as a master, and do not store the optimizer inside its bound model.

The floating cast preserves tracking. Its backward converts the incoming
real16 or real16b gradient to real32, and contributions from several uses or
several backward calls accumulate in the original Parameter's real32 slot.
Low-precision forward and backward operations still round in their own
format; real32 accumulation does not recover information already rounded
away there. Calling `.untrack()` before the cast would cut this gradient
path and is not a substitute for a differentiable cast. These examples
require a Core version that supports tracked casts between floating tensor
types; older versions reject such casts. There is no automatic loss scaling
in this example.

Direct `nn.Parameter<real16>` or `nn.Parameter<real16b>` training remains
supported. Without an explicit master, the same layer stores its weight as
that low type and returns `input * .weight.track()`. Its gradient slot
also uses that type and rounds once per accumulated addition. Adam widens
that weight and gradient for real32 arithmetic, keeps real32 moments, and
rounds the updated weight once when storing it back. NN creates no hidden
real32 weight copy in either mode.

`nn.checkpoint(&model, optimizer, path)` saves the model's stored Parameters
and Adam state. For these explicit-master models that means the real32
originals and their real32 moments, not the temporary forward casts.
`nn.resume(&model, &optimizer, path)` restores a freshly initialized model
and optimizer bound to it. Keep the same field paths, shapes and element
types. Direct low-type models instead save their actual real16 or real16b
weights; widening optimizer moments does not change the weights' dtype.

[`examples/mixed_precision.qui`](examples/mixed_precision.qui) is independent
of the ordinary real32 training example. It selects real16 forward by
default and real16b forward with `--bfloat`; `--device N` places both master
weights and inputs on that device. `--checkpoint PATH` saves the real32
original after training. This explicit example choice does not change NN's
normal real32 default.

Checkpoints match entries by their fully qualified field paths, types and
shapes. Field order may change, but renaming a field or changing its type or
shape fails with a complete difference list before anything is loaded.
Private fields and tuple positions are part of these paths. `nn.checkpoint`
and `nn.resume` take `&model` and accept only the optimizer's bound model.

NN may use cuDNN/NCCL and package-owned native kernels behind its own boundary; on macOS, NN-owned Metal kernels serve Conv2D, ReLU/GELU, global average pooling and Adam. Generic GEMM/cuBLAS remains Math-owned. Core contains no NN-specific operation names, kernels, differentiation rules, or backend policy.

On Metal these kernels append to Core's command stream through the
encode-scope protocol of `quidra/native_extension.h`: an NN call never
commits or waits for a command buffer of its own, so consecutive NN, Math
and Core operations share Core's command buffer, and the host waits only
where it reads device data (`.item()`, `.cpu()`, `gpu.sync`, program exit).
A GPU failure is reported there like any other device error. Forward
outputs on a device are allocated without a zero fill, as write-only
outputs that the kernels store completely. First-order backward nodes use
Core's masked autograd ABI: a Conv2D computes only the gradients a backward
target needs (a tracked input that is not a target costs nothing), on the
CPU and on Metal.

NN's CPU kernels split their independent outputs (Conv2D output and input
planes and blocks of filters, pooling rows, activation and Adam elements)
across Core's CPU threads with `qcore_parallel_for`, never the terms of one
sum. `QUIDRA_CPU_THREADS` limits the threads, and results are bitwise the
same for every thread count.

Use `.qnn` (Quidra Neural Network) for saved models and checkpoints, such
as `model.qnn` and `checkpoint.qnn`. The Quidra NN State Format serializer
is owned here. The extension is a naming convention: save/load and
checkpoint/resume accept arbitrary paths, so existing `.dnn` files remain
usable without renaming. To preserve compatibility, the on-disk magic
remains `QUIDRA_DNN_STATE`. `nn.save` and `nn.checkpoint` write format 5, with one
entry per Parameter or State and an explicit `real16`, `real16b`, `real32`
or `real64` dtype name. Values use the scalar's shortest round-trip decimal
text, including negative zero and subnormals. No dtype conversion happens
on loading. Parameters are written before States; each group follows that
dtype order and the model traversal order.

The header holds the version, container kind (`model` or `model+Adam`), root
type name and a single model-entry count. Each entry contains its kind,
dtype name, complete field path, rank, dimensions, element count and one
line per value. A checkpoint then holds `Adam`, its four hyperparameters,
a single slot count, and an `AdamStep`/`AdamFirst`/`AdamSecond` triple per
Parameter, keyed by that Parameter's path. The step count and bias
corrections belong to that slot. Adam moments are real32 for real16,
real16b and real32 weights, and real64 for real64 weights. The unchanged
`ADLER65521-V1` trailer covers all preceding lines.

`nn.load` also reads published model-only format 2. `nn.resume` reads
published checkpoints 2 and 3; their global step count and corrections are
restored for each Parameter. Legacy `float32` and `float` tokens mean
real32 and real64. Format 4 was an unpublished preparation format and is
rejected, as are other versions. Older files omitted private fields: they
load only when their complete path set matches the current model. Field
order never substitutes for a path match.

Loading parses the complete file, checks the checksum and trailing data,
compares every path/kind/dtype/full shape, validates all Adam state, and
prepares device temporaries before replacing any Parameter, State or
optimizer field. A failure leaves weights, States, moments, step counts,
corrections, hyperparameters and pending gradients unchanged. An unknown
dtype name is retained in the difference list while its value lines are
skipped; it cannot silently load or convert. A malformed or out-of-range
known value reports its field path and element index with
`NN_CHECKPOINT_FILE`; its numeric parse error is preserved as the cause.
Models with Parameters or States under const fields, union alternatives,
map/set containers or recursive structures remain traversal errors rather
than silently skipped fields.

## Development

Development uses the permanent `develop` branch. See [`docs/development.md`](docs/development.md) for the canonical dependency-first release procedure and immutable-release checks.

## Errors

NN errors preserve their specific message text and carry a stable category
code. Errors propagated with `try` from Core or Math retain the original
provider's code.

| Code | Meaning | Typical message |
| --- | --- | --- |
| `NN_ARGUMENT` | Invalid layer, initializer or optimizer hyperparameter | `nn.Adam rate must be finite and positive` |
| `NN_SHAPE` | Rank, dimension, extent, or incompatible tensor geometry | `nn.FC input, weight, and bias dimensions do not match` |
| `NN_SIZE` | Tensor or workspace exceeds an implementation limit | `nn.Conv2D patch matrix is too large` |
| `NN_DEVICE` | Unsupported or invalid device placement | `nn.all_reduce_sum device must be -1 for CPU or a nonnegative GPU index` |
| `NN_TRACKED` | A tracked tensor where an untracked input is required | `nn.all_reduce_sum requires untracked tensors; reduce gradients after backward or call untrack() explicitly` |
| `NN_INTERNAL` | Broken internal invariant or compiler bridge failure | `nn.load reflection metadata does not match collected state` |
| `NN_COLLECTIVE` | Failed distributed collective | `nn.all_reduce_sum NCCL collective failed` |
| `NN_STRUCTURE` | A bound Parameter changed its shape or device | `nn.SGD.update: a bound Parameter<real32> changed its shape or device; create a new optimizer; nothing changed` |
| `NN_GRADIENT` | A gradient does not have its Parameter's shape or device | `nn.SGD.update: the gradient of a bound Parameter<real32> does not have its shape or device` |
| `NN_BINDING` | Optimizer bound to a different Parameter structure | `nn.Adam.load optimizer parameter binding does not match the target model` |
| `NN_CHECKPOINT` | Checkpoint kind, root model, or schema mismatch | `nn.load root model type does not match the target model` |
| `NN_UPDATE_FAILED` | Update failed after computation could have begun; state is unspecified | `nn.Adam.update failed during computation` |
| `NN_CHECKPOINT_FILE` | Malformed or unsupported checkpoint contents | `nn.load state checksum does not match payload` |


### Low precision optimizer storage

Direct `Parameter<real16>` and `Parameter<real16b>` training uses real32
first and second Adam moments. The fused CPU and Metal updates extend the
stored weight and gradient exactly, follow the elementwise real32 operation
order without fused multiply-adds, and round the new weight once on storage.
There is no hidden master weight. For explicitly chosen low precision forward
passes with real32 master weights, see `examples/mixed_precision.qui`.

The Metal half kernel keeps moments as binary32 bits. It guards every native
operand and intermediate against values below 2^-100, including products and
quotients that underflow to zero, and recomputes the affected element from its
unchanged inputs with integer binary32 operations. The test-only
`half_exact_probe` can prove that this path ran; production updates do not
synchronize to read a counter. A narrowed weight outside its storage range
reports `NUMERIC_CONVERSION` for a tensor element. NaN moments report
`NONFINITE_RESULT` on the CPU; a valid infinite second moment is retained,
including when the resulting update is zero. On Metal, conversion failures
keep `NUMERIC_CONVERSION`, while the deferred NaN moment check reports the
Core code of the synchronization that consumes it: `TENSOR` for `.cpu()`,
`GPU_SYNC` for `gpu.sync` or time synchronization, and `GPU_ASYNC` at exit.
The report retains the operation message and the statement that queued it.

The focused source fixtures are `tests/half_optimizer.sh` (CPU, fake backend
or Metal) and `tests/half_optimizer_failure.qui`. They compare every stored
weight and moment after each step, cover signed zeros, subnormals, ties and
carry boundaries, and check the separate failure codes.

NN owns finite-result validation for its CPU and Metal native bridges. CPU
checks report after worker completion; Metal checks reserve a status word before
encoding and hand it to Core after the submission closes. Every ordinary public
real32/real64 output is checked, while internal softmax masks may contain -inf.
The half optimizer uses separate conversion and moment status words. The
copy-only im2col bridge is finite-preserving. CUDA activation/convolution use a
package-owned bit scan on their existing default stream and reserve the check
before returning success. The public `all_reduce_sum` wrapper calls one native
collective entrypoint
with a borrowed tensor array. It preserves rank order and the existing group
start/reduce/end sequence, then reserves a check for every output on the same
default stream before returning success. The array view and element handles
stay within that extern call. No new group or normal-path wait is added.
These CUDA/NCCL sources await hardware validation.

The compatibility group bridge does not advertise self-validation: a direct
`nccl_all_reduce` only adds work to an open group. The self-validating flag
belongs only to the complete `nccl_collective` entrypoint. Callers must handle
its status: 1 refuses before computation and permits the portable path;
other nonzero statuses report collective failure and cannot retry via fallback.
