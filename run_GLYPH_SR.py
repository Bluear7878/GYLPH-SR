#!/usr/bin/env python3
"""Compatibility entry point; prefer ``glyph-sr infer``."""

from glyph_sr.config import InferenceConfig, legacy_config_values
from glyph_sr.inference import main, parse_args, validate_inputs


class Config(InferenceConfig):
    """Accept historical script-level config keyword names."""

    def __init__(self, **values):
        super().__init__(**legacy_config_values(values))


__all__ = ["Config", "main", "parse_args", "validate_inputs"]
if __name__ == "__main__":
    raise SystemExit(main())
