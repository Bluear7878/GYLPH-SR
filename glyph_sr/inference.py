"""The public restoration pipeline and image inference command."""

import json
from dataclasses import asdict
from pathlib import Path

from .config import InferenceConfig, parse_config, validate_runtime


def parse_args(argv=None):
    return parse_config(InferenceConfig, argv)


def validate_inputs(config):
    validate_runtime(config)
    for name in ("input_path", "model_config", "prompt_config", "ts_checkpoint"):
        value = getattr(config, name)
        if not value or not Path(value).is_file():
            raise FileNotFoundError(f"{name} must name an existing file: {value}")
    adapter = Path(config.ocr_adapter)
    if not (adapter / "adapter_config.json").is_file() or not any(
        (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError(f"OCR adapter config/weights not found in {adapter}")
    if config.down_factor < 1:
        raise ValueError("down_factor must be positive")


class RestorationPipeline:
    """Reuse loaded models across images without changing checkpoint tensor names."""

    def __init__(self, config):
        import copy

        import torch
        import yaml
        from peft import PeftModel

        from glyph_sr._vendor.llava.constants import DEFAULT_IMAGE_TOKEN
        from glyph_sr._vendor.llava.model.builder import load_pretrained_model

        from .models.controlnet import load_controlnet

        self.config = config
        self.model, _ = load_controlnet(
            config.model_config,
            config,
            device=config.sr_device,
            sign=config.variant,
            pretrained=config.ts_checkpoint,
        )
        self.model.eval()
        self.tokenizer, self.caption_model, self.image_processor, _ = load_pretrained_model(
            config.caption_model, None, "llava_llama_3", device_map="cpu", attn_implementation=None
        )
        self.ocr_model = (
            PeftModel.from_pretrained(
                copy.deepcopy(self.caption_model), config.ocr_adapter, device_map=config.ocr_device
            )
            .eval()
            .to(config.ocr_device)
        )
        self.caption_model.eval().to(config.caption_device)
        with open(config.prompt_config, encoding="utf-8") as stream:
            self.prompts = yaml.safe_load(stream)
        self.prompts["img_prompt"] = self.prompts["img_prompt"].format(
            DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN
        )
        self.prompts["ocr_prompt"] = self.prompts["ocr_prompt"].format(
            DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN
        )
        self.dtype = torch.float16

    def restore(self, image):
        """Return every generated image and the automatic LR-derived cue metadata."""
        from glyph_sr._vendor.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from glyph_sr._vendor.llava.conversation import conv_templates
        from glyph_sr._vendor.llava.mm_utils import process_images

        from .guidance import describe_image, describe_text_locations, generate_ocr_text
        from .images import degrade_image, pil_to_tensor, tensor_to_pil

        cfg = self.config
        image = image.convert("RGB")
        degraded = (
            degrade_image(image, down_factor=cfg.down_factor) if cfg.down_factor > 1 else image
        )
        processed = process_images([degraded], self.image_processor, self.caption_model.config)
        caption_images = [
            value.to(dtype=self.dtype, device=cfg.caption_device) for value in processed
        ]
        caption = describe_image(
            image_tensor=caption_images,
            image=degraded,
            model=self.caption_model,
            tokenizer=self.tokenizer,
            prompt=self.prompts["img_prompt"],
            conv_templates=conv_templates,
            image_token_index=IMAGE_TOKEN_INDEX,
            device=cfg.caption_device,
        )
        ocr_images = [value.to(dtype=self.dtype, device=cfg.ocr_device) for value in caption_images]
        text, _ = generate_ocr_text(
            self.prompts["ocr_prompt"],
            ocr_images,
            [degraded.size],
            self.ocr_model,
            self.tokenizer,
            conv_templates,
            "llava_llama_3",
            IMAGE_TOKEN_INDEX,
            cfg.ocr_device,
        )
        locations = describe_text_locations(
            ocr_images,
            text,
            degraded,
            self.ocr_model,
            self.tokenizer,
            self.prompts.get("location_prompt", ""),
            conv_templates,
            default_image_token=DEFAULT_IMAGE_TOKEN,
            max_new_tokens=128,
            image_token_index=IMAGE_TOKEN_INDEX,
            device=cfg.ocr_device,
        )
        low_quality, height, width = pil_to_tensor(
            degraded, upscale=cfg.upscale, min_size=cfg.min_size
        )
        low_quality = low_quality.unsqueeze(0).to(cfg.sr_device)[:, :3]
        samples = self.model.ping_pong_sampling(
            low_quality,
            caption,
            [locations + " " + caption[0]],
            image_steps=cfg.steps // 2,
            text_steps=(
                cfg.steps - cfg.steps // 2 if cfg.text_steps is None else cfg.text_steps // 2
            ),
            restoration_scale=cfg.restoration_scale,
            s_churn=cfg.s_churn,
            s_noise=cfg.s_noise,
            cfg_scale=cfg.cfg_scale,
            control_scale=cfg.control_scale,
            seed=cfg.seed,
            num_samples=cfg.num_samples,
            p_p=cfg.positive_image_prompt,
            a_text_prompt=cfg.positive_text_prompt + cfg.positive_image_prompt,
            n_p=cfg.negative_image_prompt,
            n_text_prompt=cfg.negative_text_prompt + cfg.negative_image_prompt,
            color_fix_type=cfg.color_fix_type,
            use_linear_CFG=cfg.linear_cfg,
            use_linear_control_scale=cfg.linear_control_scale,
            cfg_scale_start=cfg.cfg_scale_start,
            control_scale_start=cfg.control_scale_start,
        )
        images = [tensor_to_pil(sample, height, width) for sample in samples]
        return images, {
            "caption": caption,
            "pred_text": text,
            "text_loc": locations,
            "ts_checkpoint_load_report": self.model._ts_checkpoint_load_report,
        }


def run(config):
    """Restore one input and persist all samples with reproducible settings."""
    validate_inputs(config)
    from PIL import Image

    pipeline = RestorationPipeline(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(config.input_path) as image:
        images, cues = pipeline.restore(image)
    paths = []
    with (output_dir / config.metadata_filename).open("a", encoding="utf-8") as stream:
        index = 1
        for image in images:
            while (
                path := output_dir / f"{Path(config.input_path).stem}_sr_{index:04d}.png"
            ).exists():
                index += 1
            image.save(path)
            record = {
                "filename": path.name,
                "input": str(config.input_path),
                **cues,
                "settings": asdict(config),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            paths.append(path)
            index += 1
    return paths


def main(argv=None):
    config = InferenceConfig(**vars(parse_args(argv)))
    for path in run(config):
        print(f"Saved {path}")
    return 0
