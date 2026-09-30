#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tests" / "validate_nvidia_bundle.py"
PROJECT = tomllib.loads((ROOT / "project.toml").read_text(encoding="utf-8"))
PACKAGE_VERSION = PROJECT["package"]["version"]
CORE_REQUIREMENT = PROJECT["requires"]["quidra"]


def run(
    root: Path, *, require_populated: bool = False
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(VALIDATOR)]
    if require_populated:
        command.append("--require-populated")
    command.append(str(root))
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )


def write_package(root: Path, version: str = PACKAGE_VERSION) -> None:
    (root / "quidra.package").write_text(
        (
            f"name = nn\nversion = {version}\n"
            f"requires.quidra = {CORE_REQUIREMENT}\n"
        ),
        encoding="utf-8",
    )


def populate(
    root: Path,
    *,
    version: str = PACKAGE_VERSION,
    payload: bytes = b"bundle",
    platform: str = "linux",
) -> None:
    write_package(root, version)
    nvidia = root / "nvidia"
    lib = nvidia / "lib"
    lib.mkdir(parents=True)

    if platform == "windows":
        binaries = {
            "lib/cublas64_13.dll": payload + b"-cublas",
            "lib/cudnn64_9.dll": payload + b"-cudnn",
        }
        components = {
            "cublas": {
                "version": "1.0",
                "artifact": "cublas-fixed",
                "files": ["lib/cublas64_13.dll"],
            },
            "cudnn": {
                "version": "9.0",
                "artifact": "cudnn-fixed",
                "files": ["lib/cudnn64_9.dll"],
            },
        }
    else:
        binaries = {
            "lib/libcublas.so.13": payload + b"-cublas",
            "lib/libcudnn.so.9": payload + b"-cudnn",
            "lib/libnccl.so.2": payload + b"-nccl",
        }
        components = {
            "cublas": {
                "version": "1.0",
                "artifact": "cublas-fixed",
                "files": ["lib/libcublas.so.13"],
            },
            "cudnn": {
                "version": "9.0",
                "artifact": "cudnn-fixed",
                "files": ["lib/libcudnn.so.9"],
            },
            "nccl": {
                "version": "2.0",
                "artifact": "nccl-fixed",
                "files": ["lib/libnccl.so.2"],
            },
        }

    for relative, content in binaries.items():
        path = nvidia / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    metadata = {
        "schema_version": 1,
        "nn_version": version,
        "platform": platform,
        "architecture": "x86_64",
        "cuda_compatibility": "CUDA 13",
        "components": components,
    }
    (nvidia / "BUNDLE.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    sums = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {relative}\n"
        for relative, content in sorted(binaries.items())
    )
    (nvidia / "SHA256SUMS").write_text(sums, encoding="utf-8")


def expect_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise SystemExit(
            f"{label}: expected success, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def expect_failure(
    result: subprocess.CompletedProcess[str], label: str, needle: str
) -> None:
    if result.returncode == 0:
        raise SystemExit(f"{label}: unexpectedly succeeded")
    combined = result.stdout + result.stderr
    if needle not in combined:
        raise SystemExit(
            f"{label}: expected {needle!r} in failure output\n{combined}"
        )


with tempfile.TemporaryDirectory(prefix="nn-bundle-validator-") as raw:
    root = Path(raw)
    write_package(root)
    expect_success(run(root), "empty optional bundle")

with tempfile.TemporaryDirectory(prefix="nn-bundle-validator-") as raw:
    root = Path(raw)
    write_package(root)
    expect_failure(
        run(root, require_populated=True),
        "empty release bundle",
        "release requires a populated managed NVIDIA bundle",
    )

with tempfile.TemporaryDirectory(prefix="nn-bundle-validator-") as raw:
    root = Path(raw)
    populate(root)
    expect_success(run(root), "valid populated bundle")

with tempfile.TemporaryDirectory(prefix="nn-bundle-validator-") as raw:
    root = Path(raw)
    populate(root, platform="windows")
    expect_success(run(root), "valid Windows bundle without NCCL")

with tempfile.TemporaryDirectory(prefix="nn-bundle-validator-") as raw:
    root = Path(raw)
    populate(root)
    (root / "nvidia" / "lib" / "libcudnn.so.9").write_bytes(b"tampered")
    expect_failure(run(root), "checksum mismatch", "checksum mismatch")

with tempfile.TemporaryDirectory(prefix="nn-bundle-validator-") as raw:
    root = Path(raw)
    populate(root)
    metadata_path = root / "nvidia" / "BUNDLE.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["nn_version"] = "9.9.9"
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    expect_failure(run(root), "version mismatch", "nn_version must match")

with tempfile.TemporaryDirectory(prefix="nn-bundle-validator-") as raw:
    root = Path(raw)
    populate(root)
    metadata_path = root / "nvidia" / "BUNDLE.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["components"]["nccl"]["files"] = ["lib/libcudnn.so.9"]
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    expect_failure(run(root), "component inventory overlap", "multiple components")

with tempfile.TemporaryDirectory(prefix="nn-bundle-validator-") as raw:
    root = Path(raw)
    write_package(root)
    (root / "nvidia" / "lib").mkdir(parents=True)
    (root / "nvidia" / "lib" / "libcudnn.so.9").write_bytes(b"x")
    expect_failure(run(root), "partial bundle", "BUNDLE.json")

print("managed NVIDIA bundle validator regressions passed")
