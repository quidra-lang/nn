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
