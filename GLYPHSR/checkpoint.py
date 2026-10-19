"""Compatibility imports; new code uses glyph_sr.checkpoints."""

from glyph_sr.checkpoints import (
    get_state_dict,
    load_state_dict,
    load_ts_checkpoint,
    load_TS_pretrained_model,
)

__all__ = [
    "get_state_dict",
    "load_state_dict",
    "load_TS_pretrained_model",
    "load_ts_checkpoint",
]
