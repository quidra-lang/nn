#!/usr/bin/env python3
"""Synchronize only NN-owned NVIDIA example metadata from project.toml."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project.toml"
BUNDLE = ROOT / "nvidia" / "BUNDLE.example.json"


def render() -> str:
    project = tomllib.loads(PROJECT.read_text(encoding="utf-8"))
    data = json.loads(BUNDLE.read_text(encoding="utf-8"))
    data["nn_version"] = project["package"]["version"]
    return json.dumps(data, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    expected = render()
    actual = BUNDLE.read_text(encoding="utf-8")
    if actual == expected:
        return 0
    if args.check:
        print(
            "nvidia/BUNDLE.example.json disagrees with project.toml; "
            "run python3 scripts/sync_bundle_example.py",
            file=sys.stderr,
        )
        return 1
    BUNDLE.write_text(expected, encoding="utf-8")
    print("updated nvidia/BUNDLE.example.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
