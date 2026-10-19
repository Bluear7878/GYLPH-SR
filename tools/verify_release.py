#!/usr/bin/env python3
"""Compatibility entry point; prefer ``glyph-sr verify``."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glyph_sr.evaluation.verify import check_file, main  # noqa: E402,F401

if __name__ == "__main__":
    raise SystemExit(main())
