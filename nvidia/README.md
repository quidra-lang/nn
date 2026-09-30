# NVIDIA managed bundle contract

This directory is the package-owned location for an optional managed NVIDIA NN
runtime bundle. Quidra Core owns CUDA driver/device execution; the `nn`
package owns cuDNN/NCCL integration and pins the NVIDIA runtime libraries needed
to make those NN backends reproducible. Generic matmul/cuBLAS semantics and
backend policy belong to the separate Math package.

## Installed layout

A populated installed package uses:

```text
~/.quidra/packages/nn/
  nvidia/
    BUNDLE.json
    SHA256SUMS
    lib/
      <cuBLAS/cuBLASLt runtime dependencies when required by cuDNN>
      <cuDNN shared libraries>
      <NCCL shared libraries>
```

On Windows the equivalent root is
`%USERPROFILE%\.quidra\packages\nn\nvidia\lib`.

Core does not probe this package-owned directory and does not resolve cuDNN,
NCCL, or their transitive NVIDIA runtime dependencies for NN. NN-native
integrations are responsible for resolving this bundle, selecting NN
algorithms, and invoking vendor libraries through the package-owned native
backend without teaching Core the NN layout. The presence of cuBLAS/cuBLASLt
files here is dependency packaging for cuDNN, not a second generic matmul
backend; Math owns that integration.

## Metadata schema

`BUNDLE.json` uses schema version 1. The canonical shape is shown in
`BUNDLE.example.json`. It must contain the exact NN release version,
`platform`, `architecture`, a non-empty `cuda_compatibility` description,
and the components supported on that platform. The current bundle schema
requires `cublas`, `cudnn`, and `nccl` on Linux and `cublas` plus
`cudnn` on Windows for the currently selected cuDNN runtime set. The
`cublas` entry is a pinned transitive runtime dependency of that set, not a
NN-owned linear-algebra API. Each component records both an exact
`version` and an immutable `artifact` identity. The validator rejects partial metadata, extra/missing components,
unsafe checksum paths, symlinks, checksum omissions, extra checksum entries, and
checksum mismatches.

## Release contract

A managed bundle is release data, not an unversioned system dependency. Before a
NN release publishes a populated bundle:

1. `BUNDLE.json` must identify the NN release, supported platform/architecture,
   CUDA compatibility, and exact versions for every component required by the
   selected platform. Each component must also list its non-empty `files`
   inventory under `lib/`;
   inventories may not overlap and together must cover the bundle exactly.
2. Every file under `lib/` must be covered by `SHA256SUMS`.
3. The checksums must be computed from the exact artifacts shipped with the
   release; floating URLs or "latest" aliases are not acceptable.
4. The bundle must be validated on the same released Quidra Core baseline named
   by `requires.quidra`.
5. Replacing a published bundle requires a new NN release; published checksums
   are immutable.

The repository intentionally does not contain vendor binaries during ordinary
development until a release has selected redistributable artifacts and recorded
their exact versions and checksums. Normal CI therefore permits an unstaged
bundle contract, but the release workflow runs the validator with
`--require-populated` and refuses to create a NN release while the managed
bundle is empty. This prevents the managed-bundle path from silently becoming a
system-library search path or an unpinned download channel.


## Selected release artifacts

The immutable source selection for the managed bundle is recorded in
`SOURCES.json`. The selected CUDA family is 13. Linux x86_64 pins cuBLAS
13.8.0.4 as a cuDNN runtime dependency, cuDNN 9.26.0.51 (CUDA 13), and NCCL
2.31.2. Windows x86_64 pins the same cuBLAS runtime dependency and cuDNN
9.26.0.51 (CUDA 13). These bundle entries do not define generic GEMM ownership;
Math owns that. Every source artifact is pinned by exact identity and
SHA256; floating "latest" URLs are forbidden.

Vendor archives remain outside Git history. Bundle materialization must verify
the source SHA256 before extracting libraries into `nvidia/lib`, preserve the
vendor license files, then generate `BUNDLE.json` and `SHA256SUMS` for the
exact files that will be installed.

The package manifest declares the immutable Linux x86_64 and Windows
x86_64 GitHub release assets. A Quidra Core within the declared compatibility
range downloads the matching asset
during `quidra install nn`, validates the archive structure before extraction,
and atomically publishes the source package together with its managed NVIDIA
runtime. macOS installs no NVIDIA asset and continues to use the Metal backend.
