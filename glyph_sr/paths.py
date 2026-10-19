"""Portable model and package-resource locations."""

import os
from pathlib import Path

LLAVA_CLIP_PATH = os.getenv("GLYPH_LLAVA_CLIP_PATH", "openai/clip-vit-large-patch14-336")
LLAVA_MODEL_PATH = os.getenv("GLYPH_LLAVA_MODEL_PATH", "lmms-lab/llama3-llava-next-8b")
FINE_TUNED_LLAVA_MODEL_PATH = os.getenv("GLYPH_LLAVA_ADAPTER_PATH", "./checkpoints/Llava-next_FT")
SDXL_CLIP1_PATH = os.getenv("GLYPH_CLIP1_PATH", "openai/clip-vit-large-patch14")
SDXL_CLIP2_CKPT_PTH = os.getenv("GLYPH_CLIP2_PATH", "laion2b_s39b_b160k")


def resource_path(name):
    """Resolve bundled configuration independently of the current directory."""
    root = Path(__file__).resolve().parent / "configs"
    path = (root / str(name)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"Unknown packaged configuration: {name}")
    return path
