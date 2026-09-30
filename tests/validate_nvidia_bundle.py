#!/usr/bin/env python3
"""Validate an optional NN-owned managed NVIDIA runtime bundle."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

arguments = sys.argv[1:]
require_populated = False
if "--require-populated" in arguments:
    require_populated = True
    arguments.remove("--require-populated")
if len(arguments) > 1:
    raise SystemExit(
        "usage: validate_nvidia_bundle.py [--require-populated] [repository-root]"
    )

ROOT = (
    Path(arguments[0]).resolve()
    if arguments
    else Path(__file__).resolve().parents[1]
)
NVIDIA = ROOT / "nvidia"
BUNDLE = NVIDIA / "BUNDLE.json"
SUMS = NVIDIA / "SHA256SUMS"
LIB = NVIDIA / "lib"


def fail(message: str) -> None:
    raise SystemExit(f"managed NVIDIA bundle validation failed: {message}")


def package_version() -> str:
    for line in (ROOT / "quidra.package").read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            value = line.removeprefix("version = ").strip()
            if value:
                return value
    fail("quidra.package has no version")
    raise AssertionError


def require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"BUNDLE.json field {field!r} must be a non-empty string")
    return value.strip()


def load_metadata(expected_version: str) -> set[str]:
    try:
        data = json.loads(BUNDLE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"BUNDLE.json is invalid JSON: {exc}")
    if not isinstance(data, dict):
        fail("BUNDLE.json root must be an object")
    if data.get("schema_version") != 1:
        fail("BUNDLE.json schema_version must be 1")
    if require_text(data.get("nn_version"), "nn_version") != expected_version:
        fail(
            "BUNDLE.json nn_version must match quidra.package "
            f"({expected_version})"
        )
    platform = require_text(data.get("platform"), "platform").lower()
    if platform not in {"linux", "windows"}:
        fail("BUNDLE.json platform must be 'linux' or 'windows'")
    require_text(data.get("architecture"), "architecture")
    require_text(data.get("cuda_compatibility"), "cuda_compatibility")

    components = data.get("components")
    if not isinstance(components, dict):
        fail("BUNDLE.json components must be an object")
    expected = {"cublas", "cudnn"}
    if platform == "linux":
        expected.add("nccl")
    if set(components) != expected:
        required = ", ".join(sorted(expected))
        fail(
            f"BUNDLE.json components for {platform} must contain exactly "
            f"{required}"
        )
    inventory: set[str] = set()
    for name in sorted(expected):
        component = components[name]
        if not isinstance(component, dict):
            fail(f"BUNDLE.json component {name!r} must be an object")
        require_text(component.get("version"), f"components.{name}.version")
        require_text(component.get("artifact"), f"components.{name}.artifact")
        files = component.get("files")
        if not isinstance(files, list) or not files:
            fail(f"BUNDLE.json components.{name}.files must be a non-empty array")
        for index, value in enumerate(files):
            relative = require_text(value, f"components.{name}.files[{index}]")
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                fail(f"BUNDLE.json has unsafe component file path {relative!r}")
            normalized = path.as_posix()
            if path.parts[0] != "lib":
                fail(
                    f"BUNDLE.json component file {relative!r} must be under lib/"
                )
            if normalized in inventory:
                fail(f"BUNDLE.json assigns {normalized!r} to multiple components")
            inventory.add(normalized)
    return inventory


def load_sums() -> dict[str, str]:
    entries: dict[str, str] = {}
    pattern = re.compile(r"^([0-9a-fA-F]{64})  (.+)$")
    for number, raw in enumerate(SUMS.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        match = pattern.fullmatch(raw)
        if not match:
            fail(f"SHA256SUMS line {number} must be '<64 hex>  <relative path>'")
        digest, relative = match.groups()
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            fail(f"SHA256SUMS line {number} has unsafe path {relative!r}")
        if not path.parts or path.parts[0] != "lib":
            fail(f"SHA256SUMS line {number} must name a file under lib/")
        normalized = path.as_posix()
        if normalized in entries:
            fail(f"SHA256SUMS contains duplicate path {normalized!r}")
        entries[normalized] = digest.lower()
    if not entries:
        fail("SHA256SUMS contains no library files")
    return entries


def actual_library_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(LIB.rglob("*")):
        if path.is_symlink():
            fail(f"managed bundle may not contain symlink {path.relative_to(NVIDIA)}")
        if path.is_file():
            relative = path.relative_to(NVIDIA).as_posix()
            files[relative] = path
    if not files:
        fail("nvidia/lib contains no files")
    return files


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    populated = BUNDLE.exists() or SUMS.exists() or LIB.exists()
    if not populated:
        if require_populated:
            fail("release requires a populated managed NVIDIA bundle")
        print("managed NVIDIA bundle: no staged vendor artifacts")
        return 0

    if not BUNDLE.is_file():
        fail("populated bundle is missing nvidia/BUNDLE.json")
    if not SUMS.is_file():
        fail("populated bundle is missing nvidia/SHA256SUMS")
    if not LIB.is_dir():
        fail("populated bundle is missing nvidia/lib/")

    inventory = load_metadata(package_version())
    expected = load_sums()
    actual = actual_library_files()

    if inventory != set(actual):
        unowned = sorted(set(actual) - inventory)
        missing = sorted(inventory - set(actual))
        fail(
            "BUNDLE.json component file inventory mismatch: "
            f"unowned={unowned}, missing={missing}"
        )

    if set(expected) != set(actual):
        missing = sorted(set(actual) - set(expected))
        extra = sorted(set(expected) - set(actual))
        fail(f"SHA256SUMS/file set mismatch: missing={missing}, extra={extra}")

    for relative, path in actual.items():
        got = digest(path)
        if got != expected[relative]:
            fail(
                f"checksum mismatch for {relative}: "
                f"expected {expected[relative]}, got {got}"
            )

    print(f"managed NVIDIA bundle: validated {len(actual)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
