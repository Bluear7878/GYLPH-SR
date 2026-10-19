"""Small helpers shared by the generation pipelines."""

import random
import re
import string


def sanitize_prompt(raw, extra_style=None):
    match = re.search(r"[\"“”‘’`](.+?)[\"“”‘’`]", raw)
    if match:
        raw = raw.replace(match.group(0), match.group(1).strip(), 1)
    raw = raw.rstrip().rstrip(".")
    return f"{raw}, {extra_style}" if extra_style else raw


def random_masking_prompt(caption, mask_prob=0.5):
    """Preserve the original first-quoted-span character corruption rule."""
    if not 0 <= mask_prob <= 1:
        raise ValueError("mask_prob must be between zero and one")
    pool = string.ascii_letters + string.digits

    def replace(match):
        return "".join(
            char if char.isspace() or random.random() > mask_prob else random.choice(pool)
            for char in match.group(2)
        )

    caption = re.sub(r"([\"“”‘’`])([^\"“”‘’`]+?)\1", replace, caption, count=1)
    return re.sub(r"[\"“”‘’`]", "", caption).strip().rstrip(".")


def move_image_tensors(images, device, dtype):
    """LLaVA processors may return either a stacked tensor or a patch list."""
    if isinstance(images, list):
        return [image.to(device=device, dtype=dtype) for image in images]
    if isinstance(images, tuple):
        return tuple(image.to(device=device, dtype=dtype) for image in images)
    return images.to(device=device, dtype=dtype)


def load_prompt_templates(path):
    import yaml

    from llava.constants import DEFAULT_IMAGE_TOKEN

    with open(path, encoding="utf-8") as stream:
        prompts = yaml.safe_load(stream)
    return {
        name: text.format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN)
        for name, text in prompts.items()
        if name in {"ocr_prompt", "img_prompt"}
    }


def release_memory():
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
