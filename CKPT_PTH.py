"""Optional local model paths; set environment variables instead of editing source."""
import os

LLAVA_CLIP_PATH = os.environ.get("GLYPH_LLAVA_CLIP_PATH", "openai/clip-vit-large-patch14-336")
LLAVA_MODEL_PATH = os.environ.get("GLYPH_LLAVA_MODEL_PATH", "lmms-lab/llama3-llava-next-8b")
FINE_TUNED_LLAVA_MODEL_PATH = os.environ.get("GLYPH_LLAVA_ADAPTER_PATH", "./checkpoints/Llava-next_FT")
SDXL_CLIP1_PATH = os.environ.get("GLYPH_CLIP1_PATH", "openai/clip-vit-large-patch14")
SDXL_CLIP2_CKPT_PTH = os.environ.get("GLYPH_CLIP2_PATH", "laion2b_s39b_b160k")
