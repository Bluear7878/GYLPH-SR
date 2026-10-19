"""Compatibility imports; use :mod:`glyph_sr.data.training` for new code."""

from glyph_sr.data.training import (
    HQ_PROMPT,
    LQ_PROMPT,
    NEGATIVE_PROMPT,
    POSITIVE_PROMPT,
    HQLQDataset,
    collate_training_samples,
    make_hqlq_dataloader,
)

__all__ = [
    "HQ_PROMPT",
    "LQ_PROMPT",
    "POSITIVE_PROMPT",
    "NEGATIVE_PROMPT",
    "HQLQDataset",
    "collate_training_samples",
    "make_hqlq_dataloader",
]
