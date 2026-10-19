"""Compatibility imports; new code uses glyph_sr.models.diffusion."""

from glyph_sr.models.diffusion import (
    DiffusionRestorer,
    SR_backbone,
)

__all__ = ["DiffusionRestorer", "SR_backbone"]
