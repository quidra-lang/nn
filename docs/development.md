# Development and release workflow

This is the canonical development and release procedure for the `nn` Quidra package.

## Permanent branches

- `main` is the latest published stable release source once NN has published its first release. Before that first release, `main` may contain repository bootstrap history only and is not a release or installation identity.
- `develop` is the long-lived integration branch for the next release.

Routine work goes directly to `develop`. Never force-move either permanent branch,
and never publish an unreleased development branch as an installation identity.

## Package metadata

`project.toml` is the metadata source of truth. After changing the package
version, compatibility ranges, native sources, compiler extension metadata, or
release assets, regenerate and validate the compatibility manifest with the
released/current Core tooling:

```sh
quidra package sync .
quidra package validate .
```

The package depends on Core and Math. Its reusable neural-network semantics,
compiler policy, native kernels, cuDNN/NCCL integration, managed NVIDIA assets,
state persistence, and training behavior remain NN-owned.

NN uses the exact same `MAJOR.MINOR.PATCH` version as Core and Math. The
first-party package version is lockstep and must not be chosen independently.

## Native kernels and tests

`native/nn_native.cpp` holds the CPU kernels and the backend dispatch;
`native/nn_cuda.cpp` and the cuDNN/NCCL bridge serve NVIDIA GPUs; on macOS
`native/nn_metal.mm` provides NN-owned Metal kernels for Conv2D (forward,
backward and custom autograd), the fused ReLU/GELU activations, global
average pooling and the Adam update. Every kernel reproduces the package's
own source expressions, and every backend without a native kernel keeps the
portable Quidra graph.

The suites under `tests/` take the compiler path as their first argument.
When GPU 0 is a Metal device, `real_gpu_integration.sh` additionally runs
`tests/metal_native.qui` (values and gradients against the CPU),
`tests/metal_dispatch.qui` (the native Metal kernels actually ran: the
portable fallback gives the same values, so the test reads NN's per-kernel
dispatch counter, a test probe exported by `nn_native.cpp`),
`tests/metal_adam.qui` (Adam state after every step) and
`tests/metal_uninitialized.qui` (partially initialized, untracked GPU inputs
stop with `UNINITIALIZED` on the native kernels too, because Core rejects
such views in `qcore_tensor_device_handle(_const)`). With an older Core that
does not reject them, the kernels compute on unwritten memory; the suite
only reports that, and `QUIDRA_NN_REQUIRE_METAL_UNINITIALIZED=1` makes it
fail.

`tests/performance.sh` prints Conv2D/GELU timings for the small (the first
convolution of an external training program), medium and, on request, large
shapes on the CPU and GPU 0:

```sh
bash tests/performance.sh /path/to/quidra small,medium,large
QUIDRA_NN_PERFORMANCE_CHECK=1 bash tests/performance.sh /path/to/quidra small
```

The second form fails when a GPU runs the small Conv2D forward+backward
through the portable fallback instead of a native kernel.

## Dependency-first release order

NN uses the same `MAJOR.MINOR.PATCH` version as Core and Math. A release
requires immutable Core and Math tags with exactly that version, and the
declared dependency ranges must admit it. Release dependencies in this order
before NN:

```text
Core -> Math -> NN
```

Never validate a published NN release against dependency `develop` branches.

## Releasing

When instructed to release NN:

1. Fetch the latest remote `develop` and `main` HEADs.
2. Confirm all intended work is present in `develop`.
3. Update `project.toml` to the shared Core/Math/NN release version and exact
   tested Core/Math compatibility ranges, then run `quidra package sync .`.
4. Run the full NN CI surface on `develop`, including compiler optimization,
   persistence-failure atomicity, device contracts, managed NVIDIA bundle
   validation, examples, and any available real-GPU validation.
5. Merge the verified `develop` state into `main` while preserving valid
   history. Do not replace either branch wholesale.
6. The release workflow triggered by the `main` push validates that the
   same-version immutable Core and Math tags exist, checks out those exact tags,
   rebuilds Core, validates package metadata, reruns NN contracts, materializes
   the managed NVIDIA bundles, and refuses to reuse an existing NN tag.
7. Let that workflow create the immutable `vMAJOR.MINOR.PATCH` tag and GitHub
   Release on the tested `main` commit. Do not pre-create the tag manually.
8. Verify the tag, GitHub Release, release assets, and `quidra.package` version
   agree.
9. Bring any release-only changes back to `develop` if needed. Change NN's
   development version only when the shared Core first-party version advances;
   regenerate `quidra.package` and verify the resulting `develop` CI.

Published tags are immutable. Never force-move, delete/recreate, or reuse one.
