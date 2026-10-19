"""Compatibility entrypoint; prefer ``glyph-sr data prompts``."""

from glyph_sr.data_generation.prompts import main, parse_args, run

__all__ = ["main", "parse_args", "run"]


if __name__ == "__main__":
    main()
