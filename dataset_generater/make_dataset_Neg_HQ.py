"""Compatibility entrypoint; prefer ``glyph-sr data negatives``."""

from glyph_sr.data_generation.negatives import main, parse_args, run

__all__ = ["main", "parse_args", "run"]


if __name__ == "__main__":
    main()
