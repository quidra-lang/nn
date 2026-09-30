#!/usr/bin/env python3
"""Materialize a platform-specific managed NVIDIA NN bundle from pinned sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]

_PACKAGE = tomllib.loads(
    (ROOT / "project.toml").read_text(encoding="utf-8")
)["package"]
USER_AGENT = f"{_PACKAGE['display_name'].replace(' ', '-')}/{_PACKAGE['version']}"


def fail(message: str) -> None:
    raise SystemExit(f"NVIDIA bundle materialization failed: {message}")


def safe_relative(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        fail(f"unsafe archive path: {name!r}")
    return Path(*pure.parts)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_url(url: str, output: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response, output.open("wb") as target:
        shutil.copyfileobj(response, target, length=1024 * 1024)


def resolve_source_url(component: dict[str, object]) -> str:
    source = component["source"]
    artifact = str(component["artifact"])
    if source == "nvidia-redist":
        manifest = str(component["manifest"])
        base = manifest.rsplit("/", 1)[0] + "/"
        return base + artifact
    if source == "nvidia-pypi":
        project = str(component["project"])
        version = str(component["version"])
        api = f"https://pypi.org/pypi/{project}/{version}/json"
        request = urllib.request.Request(api, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            metadata = json.load(response)
        for candidate in metadata.get("urls", []):
            if candidate.get("filename") == artifact:
                remote_hash = candidate.get("digests", {}).get("sha256")
                if remote_hash != component["sha256"]:
                    fail(f"PyPI SHA256 metadata mismatch for {artifact}")
                return str(candidate["url"])
        fail(f"PyPI artifact not found: {artifact}")
    fail(f"unsupported source kind: {source}")
    raise AssertionError


def acquire(component: dict[str, object], cache: Path | None, target: Path) -> Path:
    artifact = str(component["artifact"])
    filename = Path(artifact).name
    cached = cache / filename if cache else None
    if cached and cached.is_file():
        shutil.copyfile(cached, target)
    else:
        download_url(resolve_source_url(component), target)
    expected = str(component["sha256"])
    actual = sha256(target)
    if actual != expected:
        fail(f"source SHA256 mismatch for {filename}: expected {expected}, got {actual}")
    return target


def extract_archive(archive: Path, root: Path) -> tuple[dict[str, Path], dict[str, str]]:
    files: dict[str, Path] = {}
    links: dict[str, str] = {}

    if archive.suffix == ".whl" or archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                if info.is_dir():
                    continue
                relative = safe_relative(info.filename)
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                files[relative.as_posix()] = destination
        return files, links

    if archive.name.endswith(".tar.xz") or archive.name.endswith(".tar.gz") or archive.name.endswith(".tgz"):
        with tarfile.open(archive, "r:*") as bundle:
            for member in bundle.getmembers():
                relative = safe_relative(member.name)
                normalized = relative.as_posix()
                if member.isfile():
                    source = bundle.extractfile(member)
                    if source is None:
                        fail(f"cannot read archive member: {member.name}")
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    files[normalized] = destination
                elif member.issym() or member.islnk():
                    links[normalized] = member.linkname
        return files, links

    fail(f"unsupported archive format: {archive.name}")
    raise AssertionError


def resolve_archive_file(name: str, files: dict[str, Path], links: dict[str, str]) -> Path:
    if name in files:
        return files[name]
    seen: set[str] = set()
    current = name
    while current in links:
        if current in seen:
            fail(f"archive link cycle at {name}")
        seen.add(current)
        target = PurePosixPath(links[current])
        if target.is_absolute():
            fail(f"absolute archive link target at {name}")
        current = (PurePosixPath(current).parent / target).as_posix()
        current = PurePosixPath(current).as_posix()
        if ".." in PurePosixPath(current).parts:
            fail(f"archive link escapes root at {name}")
        if current in files:
            return files[current]
    fail(f"archive member is missing backing file: {name}")
    raise AssertionError


def aliases(component: str, platform: str, files: dict[str, Path], links: dict[str, str]) -> list[tuple[str, str]]:
    names = set(files) | set(links)
    by_base: dict[str, list[str]] = {}
    for name in names:
        by_base.setdefault(PurePosixPath(name).name, []).append(name)

    def exact(base: str) -> tuple[str, str]:
        choices = by_base.get(base, [])
        if not choices:
            fail(f"{component} archive does not contain {base}")
        return base, sorted(choices)[0]

    if platform == "linux-x86_64":
        if component == "cublas":
            return [exact("libcublas.so.13"), exact("libcublasLt.so.13")]
        if component == "nccl":
            return [exact("libnccl.so.2")]
        if component == "cudnn":
            found = []
            for base in sorted(by_base):
                if base.startswith("libcudnn") and base.endswith(".so.9"):
                    found.append(exact(base))
            if not found:
                fail("cuDNN archive contains no lib*.so.9 runtime libraries")
            return found
    elif platform == "windows-x86_64":
        if component == "cublas":
            return [exact("cublas64_13.dll"), exact("cublasLt64_13.dll")]
        if component == "cudnn":
            found = []
            for base in sorted(by_base):
                lowered = base.lower()
                if lowered.startswith("cudnn") and lowered.endswith(".dll"):
                    found.append(exact(base))
            if not found:
                fail("cuDNN archive contains no runtime DLLs")
            return found
    fail(f"unsupported component/platform pair: {platform}/{component}")
    raise AssertionError


def copy_license(component: str, files: dict[str, Path], destination: Path) -> None:
    candidates = [
        (name, path) for name, path in files.items()
        if "license" in PurePosixPath(name).name.lower()
    ]
    if not candidates:
        return
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sorted(candidates)[0][1], destination / f"{component}.txt")


def materialize(args: argparse.Namespace) -> None:
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    components = sources["platforms"].get(args.platform)
    if not isinstance(components, dict):
        fail(f"platform is not locked in SOURCES.json: {args.platform}")

    package_version = None
    for line in (args.package_root / "quidra.package").read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            package_version = line.removeprefix("version = ").strip()
    if not package_version:
        fail("quidra.package has no version")

    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    lib = output / "nvidia" / "lib"
    licenses = output / "nvidia" / "licenses"
    lib.mkdir(parents=True)

    inventories: dict[str, list[str]] = {}
    artifact_identities: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix="quidra-nn-nvidia-") as temporary:
        temporary_root = Path(temporary)
        for component_name, component in components.items():
            archive = temporary_root / Path(str(component["artifact"])).name
            acquire(component, args.artifact_cache, archive)
            extracted = temporary_root / f"extract-{component_name}"
            extracted.mkdir()
            files, links = extract_archive(archive, extracted)

            inventory: list[str] = []
            for output_name, source_name in aliases(
                component_name, args.platform, files, links
            ):
                source = resolve_archive_file(source_name, files, links)
                target = lib / output_name
                shutil.copyfile(source, target)
                inventory.append(f"lib/{output_name}")
            copy_license(component_name, files, licenses)
            inventories[component_name] = sorted(inventory)
            artifact_identities[component_name] = (
                f"{component['source']}:{component['artifact']}#{component['sha256']}"
            )

    platform_name, architecture = args.platform.split("-", 1)
    metadata = {
        "schema_version": 1,
        "nn_version": package_version,
        "platform": platform_name,
        "architecture": architecture,
        "cuda_compatibility": f"CUDA {sources['cuda_family']}",
        "components": {
            name: {
                "version": str(components[name]["version"]),
                "artifact": artifact_identities[name],
                "files": inventories[name],
            }
            for name in sorted(components)
        },
    }
    (output / "nvidia" / "BUNDLE.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    sum_lines = []
    for path in sorted(lib.iterdir()):
        if path.is_file():
            sum_lines.append(f"{sha256(path)}  lib/{path.name}\n")
    (output / "nvidia" / "SHA256SUMS").write_text(
        "".join(sum_lines), encoding="utf-8"
    )
    shutil.copyfile(args.package_root / "quidra.package", output / "quidra.package")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, choices=("linux-x86_64", "windows-x86_64"))
    parser.add_argument("--sources", type=Path, default=ROOT / "nvidia" / "SOURCES.json")
    parser.add_argument("--package-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-cache", type=Path)
    args = parser.parse_args()
    materialize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
