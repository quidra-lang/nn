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
tensor<float32> output = nn.relu(layer.forward(input))
nn.Adam optimizer = nn.Adam()
```

NN may use cuDNN/NCCL and package-owned native kernels behind its own boundary. Generic GEMM/cuBLAS remains Math-owned. Core contains no NN-specific operation names, kernels, differentiation rules, or backend policy.

The state serializer is owned here. For compatibility with the pre-split DNN package, the on-disk state magic remains `QUIDRA_DNN_STATE`; ownership of that format is now NN.

## Development

Development uses the permanent `develop` branch. See [`docs/development.md`](docs/development.md) for the canonical dependency-first release procedure and immutable-release checks.
