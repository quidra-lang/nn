#!/usr/bin/env python3
"""Validate the immutable NVIDIA source-artifact lock used for NN bundles."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "nvidia" / "SOURCES.json"


def fail(message: str) -> None:
    raise SystemExit(f"NVIDIA source lock validation failed: {message}")


data = json.loads(LOCK.read_text(encoding="utf-8"))
if data.get("schema_version") != 1:
    fail("schema_version must be 1")
if data.get("cuda_family") != "13":
    fail("cuda_family must be the pinned major family '13'")

platforms = data.get("platforms")
if not isinstance(platforms, dict):
    fail("platforms must be an object")
if set(platforms) != {"linux-x86_64", "windows-x86_64"}:
    fail("platforms must contain exactly linux-x86_64 and windows-x86_64")

expected = {
    "linux-x86_64": {"cublas", "cudnn", "nccl"},
    "windows-x86_64": {"cublas", "cudnn"},
}
sha = re.compile(r"^[0-9a-f]{64}$")
for platform, components in platforms.items():
    if not isinstance(components, dict) or set(components) != expected[platform]:
        fail(f"{platform} has the wrong component set")
    for name, component in components.items():
        if not isinstance(component, dict):
            fail(f"{platform}.{name} must be an object")
        for field in ("version", "source", "artifact", "sha256"):
            value = component.get(field)
            if not isinstance(value, str) or not value:
                fail(f"{platform}.{name}.{field} must be non-empty")
        if not sha.fullmatch(component["sha256"]):
            fail(f"{platform}.{name}.sha256 must be lowercase SHA256")
        source = component["source"]
        if source == "nvidia-redist":
            manifest = component.get("manifest")
            if not isinstance(manifest, str) or not manifest.startswith(
                "https://developer.download.nvidia.com/"
            ):
                fail(f"{platform}.{name} must use an NVIDIA HTTPS redist manifest")
        elif source == "nvidia-pypi":
            if component.get("project") != "nvidia-nccl-cu13" or name != "nccl":
                fail(f"{platform}.{name} has an unsupported NVIDIA PyPI source")
        else:
            fail(f"{platform}.{name} has unsupported source {source!r}")

print("NVIDIA source artifact lock: valid")
