#!/usr/bin/env python3
"""Compatibility entry point; prefer ``glyph-sr train``."""

from glyph_sr.config import TrainingConfig, legacy_config_values
from glyph_sr.training import main, parse_args


class TrainConfig(TrainingConfig):
    """Accept historical script-level config keyword names."""

    def __init__(self, **values):
        super().__init__(**legacy_config_values(values))


__all__ = ["TrainConfig", "main", "parse_args"]
if __name__ == "__main__":
    raise SystemExit(main())
