"""Compatibility imports for the historical misspelled package.

New applications should use glyph_sr.data and glyph_sr.data_generation.
"""

from glyph_sr.data.io import ImageFolderDataset, load_metadata, make_image_loader
from glyph_sr.data_generation.common import random_masking_prompt, sanitize_prompt
from glyph_sr.data_generation.negatives import (
    build_neg_lq,
    create_restoration_model,
    process_record,
)
from glyph_sr.data_generation.prompts import generate_prompt_meta_jsonl, get_synthesis_prompt

__all__ = [
    "ImageFolderDataset",
    "load_metadata",
    "make_image_loader",
    "random_masking_prompt",
    "sanitize_prompt",
    "build_neg_lq",
    "create_restoration_model",
    "process_record",
    "generate_prompt_meta_jsonl",
    "get_synthesis_prompt",
    "create_SUPIR_model",
    "degrade_image",
]


def create_SUPIR_model(config_path, SUPIR_sign=None, load_default_setting=False):
    return create_restoration_model(config_path, SUPIR_sign, load_default_setting)


def degrade_image(*args, **kwargs):
    from glyph_sr.images import degrade_image as implementation

    return implementation(*args, **kwargs)
