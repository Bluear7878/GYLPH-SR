#!/usr/bin/env python3
"""Compatibility entry point; prefer ``glyph-sr score``."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glyph_sr.evaluation.ocr import (  # noqa: E402,F401
    aggregate_counts,
    corrected_word_counts,
    legacy_counts,
    main,
    normalize,
    normalize_legacy,
    parse_args,
    safe_ratio,
)

if __name__ == "__main__":
    raise SystemExit(main())
