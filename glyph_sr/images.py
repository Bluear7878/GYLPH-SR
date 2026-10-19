"""Image conversion and degradation used by the released restoration pipeline."""

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate


def degrade_image(img_pil: Image.Image, down_factor: float = 1.5) -> Image.Image:
    """Downsample and upsample with the released bicubic degradation."""
    width, height = img_pil.size
    if down_factor <= 0 or min(width / down_factor, height / down_factor) < 1:
        raise ValueError("down_factor must produce positive image dimensions")
    small = img_pil.resize(
        (int(width / down_factor), int(height / down_factor)), Image.Resampling.BICUBIC
    )
    return small.resize((width, height), Image.Resampling.BICUBIC).convert("RGB")


def pil_to_tensor(img, upscale=1, min_size=1024, fix_resize=None):
    """Return RGB ``[C,H,W]`` in ``[-1,1]`` plus the requested output height/width.

    The diffusion input is rounded to multiples of 64. The returned output size
    follows ``upscale`` (or ``fix_resize``) before that rounding, preserving the
    released preprocessing convention.
    """
    width, height = img.size
    width *= upscale
    height *= upscale
    output_width, output_height = round(width), round(height)
    if min(width, height) <= 0:
        raise ValueError("upscale must produce positive image dimensions")
    if min(width, height) < min_size:
        scale = min_size / min(width, height)
        width *= scale
        height *= scale
    if fix_resize is not None:
        scale = fix_resize / min(width, height)
        width *= scale
        height *= scale
        output_width, output_height = round(width), round(height)
    width = int(np.round(width / 64.0)) * 64
    height = int(np.round(height / 64.0)) * 64
    if min(width, height) <= 0:
        raise ValueError("diffusion input must be at least 64 pixels per dimension")
    image = img.resize((width, height), Image.Resampling.BICUBIC)
    array = np.array(image).round().clip(0, 255).astype(np.uint8)
    array = array / 255 * 2 - 1
    tensor = torch.tensor(array, dtype=torch.float32).permute(2, 0, 1)
    return tensor, output_height, output_width


def tensor_to_pil(x, h0, w0):
    """Resize an RGB ``[C,H,W]`` tensor in ``[-1,1]`` to a PIL image."""
    x = interpolate(x.unsqueeze(0), size=(h0, w0), mode="bicubic")
    array = (
        (x.squeeze(0).permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    )
    return Image.fromarray(array)


def convert_dtype(dtype_str):
    """Translate the precision names used by the model configuration."""
    types = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if dtype_str not in types:
        raise NotImplementedError(f"Unsupported precision: {dtype_str}")
    return types[dtype_str]


PIL2Tensor = pil_to_tensor
Tensor2PIL = tensor_to_pil
