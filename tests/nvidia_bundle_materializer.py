#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_nvidia_bundle.py"
VALIDATOR = ROOT / "tests" / "validate_nvidia_bundle.py"
PROJECT = tomllib.loads((ROOT / "project.toml").read_text(encoding="utf-8"))
PACKAGE_VERSION = PROJECT["package"]["version"]
CORE_REQUIREMENT = PROJECT["requires"]["quidra"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tar_archive(path: Path, root: str, files: dict[str, bytes], links: dict[str, str] = {}) -> None:
    with tarfile.open(path, "w:xz") as archive:
        for relative, data in files.items():
            local = path.parent / ("payload-" + relative.replace("/", "_"))
            local.write_bytes(data)
            archive.add(local, arcname=f"{root}/{relative}")
        for relative, target in links.items():
            info = tarfile.TarInfo(f"{root}/{relative}")
            info.type = tarfile.SYMTYPE
            info.linkname = target
            archive.addfile(info)


def zip_archive(path: Path, root: str, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, data in files.items():
            archive.writestr(f"{root}/{relative}", data)


with tempfile.TemporaryDirectory(prefix="nn-materializer-") as raw:
    temp = Path(raw)
    cache = temp / "cache"
    cache.mkdir()
    package = temp / "package"
    package.mkdir()
    (package / "quidra.package").write_text(
        (
            f"name = nn\nversion = {PACKAGE_VERSION}\n"
            f"requires.quidra = {CORE_REQUIREMENT}\n"
        ),
        encoding="utf-8",
    )

    cublas_linux = cache / "cublas-linux.tar.xz"
    tar_archive(
        cublas_linux, "cublas",
        {
            "lib/libcublas.so.13.8.0.4": b"cublas",
            "lib/libcublasLt.so.13.8.0.4": b"cublas-lt",
            "LICENSE.txt": b"cublas license",
        },
        {
            "lib/libcublas.so.13": "libcublas.so.13.8.0.4",
            "lib/libcublasLt.so.13": "libcublasLt.so.13.8.0.4",
        },
    )
    cudnn_linux = cache / "cudnn-linux.tar.xz"
    tar_archive(
        cudnn_linux, "cudnn",
        {
            "lib/libcudnn.so.9.26.0": b"cudnn",
            "lib/libcudnn_ops.so.9.26.0": b"cudnn-ops",
            "LICENSE.txt": b"cudnn license",
        },
        {
            "lib/libcudnn.so.9": "libcudnn.so.9.26.0",
            "lib/libcudnn_ops.so.9": "libcudnn_ops.so.9.26.0",
        },
    )
    nccl = cache / "nccl.whl"
    zip_archive(
        nccl, "nvidia/nccl",
        {"lib/libnccl.so.2": b"nccl", "LICENSE.txt": b"nccl license"},
    )

    cublas_windows = cache / "cublas-windows.zip"
    zip_archive(
        cublas_windows, "cublas",
        {
            "bin/cublas64_13.dll": b"cublas-win",
            "bin/cublasLt64_13.dll": b"cublas-lt-win",
            "LICENSE.txt": b"cublas license",
        },
    )
    cudnn_windows = cache / "cudnn-windows.zip"
    zip_archive(
        cudnn_windows, "cudnn",
        {
            "bin/cudnn64_9.dll": b"cudnn-win",
            "bin/cudnn_ops64_9.dll": b"cudnn-ops-win",
            "LICENSE.txt": b"cudnn license",
        },
    )

    sources = {
        "schema_version": 1,
        "cuda_family": "13",
        "platforms": {
            "linux-x86_64": {
                "cublas": {"version": "1", "source": "nvidia-redist", "manifest": "https://developer.download.nvidia.com/test.json", "artifact": cublas_linux.name, "sha256": digest(cublas_linux)},
                "cudnn": {"version": "1", "source": "nvidia-redist", "manifest": "https://developer.download.nvidia.com/test.json", "artifact": cudnn_linux.name, "sha256": digest(cudnn_linux)},
                "nccl": {"version": "1", "source": "nvidia-pypi", "project": "nvidia-nccl-cu13", "artifact": nccl.name, "sha256": digest(nccl)},
            },
            "windows-x86_64": {
                "cublas": {"version": "1", "source": "nvidia-redist", "manifest": "https://developer.download.nvidia.com/test.json", "artifact": cublas_windows.name, "sha256": digest(cublas_windows)},
                "cudnn": {"version": "1", "source": "nvidia-redist", "manifest": "https://developer.download.nvidia.com/test.json", "artifact": cudnn_windows.name, "sha256": digest(cudnn_windows)},
            },
        },
    }
    source_path = temp / "SOURCES.json"
    source_path.write_text(json.dumps(sources), encoding="utf-8")

    for platform in ("linux-x86_64", "windows-x86_64"):
        output = temp / platform
        subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--platform", platform,
                "--sources", str(source_path),
                "--package-root", str(package),
                "--artifact-cache", str(cache),
                "--output", str(output),
            ],
            check=True,
        )
        subprocess.run(
            [sys.executable, str(VALIDATOR), "--require-populated", str(output)],
            check=True,
        )

print("NVIDIA bundle materializer regressions passed")
