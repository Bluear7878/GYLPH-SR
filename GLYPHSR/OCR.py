"""Compatibility imports; new code uses glyph_sr.guidance."""

from glyph_sr.guidance import (
    describe_image,
    describe_text_locations,
    generate_ocr_text,
    get_img_describe,
    get_text_position,
)

__all__ = [
    "describe_image",
    "describe_text_locations",
    "generate_ocr_text",
    "get_img_describe",
    "get_text_position",
]
