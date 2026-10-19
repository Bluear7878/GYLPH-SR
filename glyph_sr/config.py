"""Shared CLI configuration with explicit, correctly directed boolean options."""

import argparse
import os
from dataclasses import dataclass, fields

from .paths import FINE_TUNED_LLAVA_MODEL_PATH, LLAVA_MODEL_PATH, resource_path

_FIXED_TRAINING_OPTIONS = {
    "upscale",
    "min_size",
    "text_steps",
    "positive_text_prompt",
    "negative_text_prompt",
}


@dataclass
class RuntimeConfig:
    model_config: str = str(resource_path("sdxl.yaml"))
    variant: str = "Q"
    upscale: int = 2
    seed: int = -1
    min_size: int = 1024
    steps: int = 50
    text_steps: int = None
    num_samples: int = 1
    restoration_scale: float = -1
    s_churn: float = 5
    s_noise: float = 1.003
    cfg_scale: float = 7.5
    control_scale: float = 1.0
    color_fix_type: str = "Wavelet"
    linear_cfg: bool = True
    linear_control_scale: bool = False
    cfg_scale_start: float = 4.0
    control_scale_start: float = 0.0
    ae_dtype: str = "bf16"
    diff_dtype: str = "fp16"
    tile_vae: bool = False
    encoder_tile_size: int = 512
    decoder_tile_size: int = 64
    positive_image_prompt: str = "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."
    positive_text_prompt: str = "literal text restoration, No other characters except text.Focus on restoring text details. Do not degrade overall image quality. ultra-sharp glyphs, pixel-level text clarity, perfect readability, preserve exact font shapes and spacing, high contrast, legible"
    negative_image_prompt: str = "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"
    negative_text_prompt: str = "No non-textual shapes, No object-like artifacts, No abstract forms, No objects obscuring the textNo random shapes, No blur, No halos, No ringing, No distortion, No character misplacement, No illegible segments, low resolution."


@dataclass
class InferenceConfig(RuntimeConfig):
    input_path: str = None
    output_dir: str = "./results"
    metadata_filename: str = "metadata.jsonl"
    prompt_config: str = str(resource_path("prompts.yaml"))
    ts_checkpoint: str = os.getenv("GLYPH_TS_CKPT", "./checkpoints/TS-ControlNet_cp.ckpt")
    ocr_adapter: str = FINE_TUNED_LLAVA_MODEL_PATH
    caption_model: str = LLAVA_MODEL_PATH
    sr_device: str = "cuda:0"
    ocr_device: str = "cuda:1"
    caption_device: str = "cuda:2"
    down_factor: int = 1


@dataclass
class TrainingConfig(RuntimeConfig):
    # These are the effective, fixed values used by the released validation path.
    min_size: int = 256
    hq_metadata: str = None
    negative_metadata: str = None
    validation_hq_metadata: str = None
    validation_negative_metadata: str = None
    ts_checkpoint: str = ""
    batch_size: int = 12
    epochs: int = 200
    max_steps: int = -1
    train_image_size: int = 512
    validation_image_size: int = 256
    devices: int = 1
    precision: str = "32-true"
    learning_rate: float = 1e-6
    gradient_clip_val: float = 1.0
    log_dir: str = "lightning_logs"
    checkpoint_dir: str = "checkpoints"
    checkpoint_filename: str = "epoch-{epoch:03d}-loss-{loss:.4f}"
    monitor_metric: str = "loss"
    monitor_mode: str = "min"

    def __post_init__(self):
        self._validate_supported_options()

    def _validate_supported_options(self):
        for name in sorted(_FIXED_TRAINING_OPTIONS):
            default = TrainingConfig.__dataclass_fields__[name].default
            if getattr(self, name) != default:
                raise ValueError(
                    f"{name} is not configurable for training; omit it to preserve "
                    "the released validation recipe (upscale=2, min_size=256, image guidance only)."
                )


_SPECIAL_FLAGS = {
    "input_path": "--input",
    "model_config": "--model-config",
    "prompt_config": "--prompt-config",
    "positive_image_prompt": "--image-prompt",
    "positive_text_prompt": "--text-prompt",
    "negative_image_prompt": "--negative-image-prompt",
    "negative_text_prompt": "--negative-text-prompt",
}


def parse_config(config_type, argv=None):
    """Parse canonical hyphenated flags and documented legacy spellings."""
    default = config_type()
    required = (
        {"input_path"}
        if config_type is InferenceConfig
        else {
            "hq_metadata",
            "negative_metadata",
            "validation_hq_metadata",
            "validation_negative_metadata",
        }
    )
    parser = argparse.ArgumentParser(
        description="Restore scene text."
        if config_type is InferenceConfig
        else "Train the text-control branch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for field in fields(default):
        name = field.name
        value = getattr(default, name)
        if config_type is TrainingConfig and name in _FIXED_TRAINING_OPTIONS:
            parser.set_defaults(**{name: value})
            continue
        options = [_SPECIAL_FLAGS.get(name, "--" + name.replace("_", "-"))]
        kwargs = {"dest": name, "default": value, "help": name.replace("_", " ")}
        if isinstance(value, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = (
                int if name == "text_steps" else (type(value) if value is not None else str)
            )
            kwargs["required"] = name in required
        parser.add_argument(*dict.fromkeys(options), **kwargs)
    return parser.parse_args(argv)


def validate_runtime(config):
    if isinstance(config, TrainingConfig):
        config._validate_supported_options()
    for name in (
        "upscale",
        "steps",
        "num_samples",
        "min_size",
        "encoder_tile_size",
        "decoder_tile_size",
    ):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if config.text_steps is not None and config.text_steps < 1:
        raise ValueError("text_steps must be positive when provided")
    if config.variant != "Q":
        raise ValueError("The released checkpoint uses variant Q.")
    if config.ae_dtype not in {"bf16", "fp32"}:
        raise ValueError("The VAE supports bf16 or fp32; fp16 can produce NaNs.")
    if config.diff_dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError("diff_dtype must be bf16, fp16, or fp32")
    if config.color_fix_type not in {"Wavelet", "None"}:
        raise ValueError("color_fix_type must be Wavelet or None")
