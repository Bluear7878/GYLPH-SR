#!/usr/bin/env python3
"""Verify bundled evidence hashes and, optionally, downloaded checkpoint files."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def resolve_evidence_root(root: Path | None = None) -> Path:
    """Locate evidence in a checkout, a wheel prefix, or a pip --target directory."""
    if root is not None:
        candidates = [Path(root).expanduser().resolve()]
    else:
        source_root = Path(__file__).resolve().parents[2]
        candidates = [
            source_root,
            Path(sys.prefix) / "share" / "glyph_sr",
            source_root / "share" / "glyph_sr",
        ]
    for candidate in dict.fromkeys(candidates):
        if (candidate / "evaluation" / "tmlr2026" / "manifest.json").is_file():
            return candidate
    locations = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Evaluation evidence not found in {locations}. "
        "Use --root to select a checkout or installed share/glyph_sr directory."
    )


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        help="evidence directory containing evaluation/ and docs/ (auto-detected by default)",
    )
    args = parser.parse_args(argv)
    root = resolve_evidence_root(args.root)
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
