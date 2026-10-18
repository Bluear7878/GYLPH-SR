#!/usr/bin/env python3
"""Verify bundled evidence hashes and, optionally, downloaded checkpoint files."""

import argparse
import hashlib
import json
from pathlib import Path


def check_file(path, expected):
    if not path.is_file():
        return f"missing: {path}"
    if path.stat().st_size != expected["bytes"]:
        return f"size mismatch: {path}"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected["sha256"]:
        return f"SHA-256 mismatch: {path}"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "evaluation/tmlr2026/manifest.json").read_text())
    items = [(root / item["path"], item) for item in manifest["files"]]
    if args.checkpoint_dir is not None:
        models = json.loads((root / "docs/checkpoints.json").read_text())
        items += [(args.checkpoint_dir / item["path"], item) for item in models["assets"]]
    errors = [error for path, item in items if (error := check_file(path, item))]
    for error in errors:
        print(error)
    print(f"Verified {len(items) - len(errors)}/{len(items)} files")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
