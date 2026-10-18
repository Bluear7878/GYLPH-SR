#!/usr/bin/env python3
"""
Entry point for GLYPH-SR processing pipeline.
Parses command-line arguments, loads models, and executes super-resolution
and OCR-guided text restoration on input images.
"""
import argparse
import copy
import json
import os
from dataclasses import asdict, dataclass, fields

from pathlib import Path

from CKPT_PTH import FINE_TUNED_LLAVA_MODEL_PATH, LLAVA_MODEL_PATH


@dataclass
class Config:
    # Paths and I/O
    img_path: str = None  # Required input image
    sr_device: str = "cuda:0"
    ocr_device: str = "cuda:1"
    vlm_device: str = "cuda:2"
    llava_model_path: str = LLAVA_MODEL_PATH
    save_dir: str = "./results"  # Output directory
    meta_filename: str = "metadata.jsonl"
    hq_meta_filename: str = "metadata_HQ.jsonl"
    neg_meta_filename: str = "metadata_with_neg.jsonl"
    yaml_file: str = "./model_configs/juggernautXL.yaml"
    prompt_yaml_file: str = "./prompts/prompt_config.yaml"

    upscale: int = 2
    sign: str = "Q"
    seed: int = -1
    min_size: int = 1024
    edm_steps: int = 50
    text_steps: int = 50
    s_stage1: int = -1
    s_churn: int = 5
    s_noise: float = 1.003
    s_cfg: float = 7.5
    s_stage2: float = 1.0
    num_samples: int = 1
    down_factor: int = 1
    GYLPH_pretrained_ckpt: str = os.environ.get("GLYPH_TS_CKPT", "./checkpoints/TS-ControlNet_cp.ckpt")
    VLM_pretrained_ckpt: str = FINE_TUNED_LLAVA_MODEL_PATH

    # Prompts
    a_prompt: str = (
        "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
        "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, "
        "extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."
    )
    a_text_prompt: str = (
        "literal text restoration, "
        "No other characters except text."
        "Focus on restoring text details. Do not degrade overall image quality. "
        "ultra-sharp glyphs, pixel-level text clarity, perfect readability, "
        "preserve exact font shapes and spacing, high contrast, legible"
    )
    n_prompt: str = (
        "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, "
        "CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, "
        "frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"
    )
    n_text_prompt: str = (
        "No non-textual shapes, No object-like artifacts, No abstract forms, "
        "No objects obscuring the text"
        "No random shapes, No blur, No halos, No ringing, No distortion, "
        "No character misplacement, No illegible segments, low resolution."
    )

    # Color & config
    color_fix_type: str = "Wavelet"
    linear_CFG: bool = True
    linear_s_stage2: bool = False
    spt_linear_CFG: float = 4.0
    spt_linear_s_stage2: float = 0.0
    ae_dtype: str = "bf16"
    diff_dtype: str = "fp16"

    # Flags
    no_llava: bool = False
    loading_half_params: bool = False
    use_tile_vae: bool = False
    encoder_tile_size: int = 512
    decoder_tile_size: int = 64
    load_8bit_llava: bool = False
    log_history: bool = False

    # Sampling & distributed
    top_k: int = 3
    beams_num: int = 10
    world_size: int = 1
    local_rank: int = 0
    mask_prob: float = 0.3


def parse_args(argv=None):
    """
    Parse command-line arguments into a Config instance.
    """
    parser = argparse.ArgumentParser(
        description="GLYPH-SR: OCR-guided super-resolution pipeline."
    )
    # Dynamically add arguments for each field in Config
    for field in fields(Config):
        name = f"--{field.name}"
        default = field.default
        if isinstance(default, bool):
            # Boolean flags
            action = "store_true" if not default else "store_false"
            parser.add_argument(name, action=action, help=f"(flag) default={default}")
        else:
            arg_type = type(default) if default is not None else str
            required = default is None
            parser.add_argument(
                name,
                type=arg_type,
                default=default,
                required=required,
                help=f"type={arg_type.__name__}, default={default}",
            )
    return parser.parse_args(argv)


def validate_inputs(cfg):
    for field in ("img_path", "yaml_file", "prompt_yaml_file", "GYLPH_pretrained_ckpt"):
        value = getattr(cfg, field)
        if not value or not Path(value).is_file():
            raise FileNotFoundError(f"--{field} must name an existing file: {value}")
    adapter = Path(cfg.VLM_pretrained_ckpt)
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"OCR adapter_config.json not found in {adapter}")
    if not any((adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise FileNotFoundError(f"OCR adapter weights not found in {adapter}")
    if cfg.sign != "Q":
        raise ValueError("The released paper configuration uses --sign Q.")
    if cfg.upscale < 1 or cfg.down_factor < 1:
        raise ValueError("upscale and down_factor must be positive integers")


def main():
    # Load command-line configuration
    args = parse_args()
    # Convert Namespace to Config object
    cfg = Config(**vars(args))

    validate_inputs(cfg)

    # Import the model stack only after parsing/validation so --help works on CPU.
    import torch
    import yaml
    from peft import PeftModel
    from PIL import Image
    from GLYPHSR.ControlNet import load_TS_ControlNet
    from GLYPHSR.OCR import generate_ocr_text, get_img_describe, get_text_position
    from GLYPHSR.util import PIL2Tensor, Tensor2PIL, degrade_image
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images
    from llava.model.builder import load_pretrained_model

    # Ensure save directory exists
    os.makedirs(cfg.save_dir, exist_ok=True)
    meta_path = os.path.join(cfg.save_dir, cfg.meta_filename)
    if not os.path.exists(meta_path):
        open(meta_path, "w", encoding="utf-8").close()

    # Load prompt configuration
    if not os.path.isfile(cfg.prompt_yaml_file):
        raise FileNotFoundError(f"Prompt YAML file not found: {cfg.prompt_yaml_file}")
    with open(cfg.prompt_yaml_file, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    # Prepare prompts
    img_prompt = prompts["img_prompt"].format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN)
    ocr_prompt = prompts["ocr_prompt"].format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN)
    verify_prompt = prompts.get("verify_prompt", "")
    location_prompt = prompts.get("location_prompt", "")

    # Load TS-ControlNet for GLYPH-SR
    cfg_path = cfg.yaml_file
    GYLPH_pretrained_ckpt = cfg.GYLPH_pretrained_ckpt

    GLYPH_SR_model, _ = load_TS_ControlNet(
        cfg_path=cfg_path,
        args=cfg,
        device=cfg.sr_device,
        sign=cfg.sign,
        pretrained=GYLPH_pretrained_ckpt,
    )
    GLYPH_SR_model.eval()

    # Load language-vision model
    MODEL_PATH = cfg.llava_model_path
    conv_template = "llava_llama_3"

    tokenizer, VLM_model, image_processor, max_length = load_pretrained_model(
        MODEL_PATH, None, conv_template, device_map="cpu", attn_implementation=None
    )
    # Load OCR model
    OCR_model = (
        PeftModel.from_pretrained(
            copy.deepcopy(VLM_model), cfg.VLM_pretrained_ckpt, device_map=cfg.ocr_device
        )
        .eval()
        .to(cfg.ocr_device)
    )
    VLM_model.eval().to(cfg.vlm_device)

    image = Image.open(cfg.img_path).convert("RGB")

    # Apply degradation as needed
    if cfg.down_factor > 1:
        degraded = degrade_image(image, down_factor=cfg.down_factor)
    else:
        degraded = image

    # Prepare image tensor for VLM
    img_proc = process_images([degraded], image_processor, VLM_model.config)
    img_proc = [t.to(dtype=torch.float16, device=cfg.vlm_device) for t in img_proc]

    # Generate image caption
    caption = get_img_describe(
        image_tensor=img_proc,
        image=degraded,
        model=VLM_model,
        tokenizer=tokenizer,
        prompt=img_prompt,
        conv_templates=conv_templates,
        image_token_index=IMAGE_TOKEN_INDEX,
        device=cfg.vlm_device,
    )

    ocr_img_proc = [t.to(dtype=torch.float16, device=cfg.ocr_device) for t in img_proc]

    # OCR text prediction
    pred_text, _ = generate_ocr_text(
        ocr_prompt,
        ocr_img_proc,
        [degraded.size],
        OCR_model,
        tokenizer,
        conv_templates,
        conv_template,
        IMAGE_TOKEN_INDEX,
        cfg.ocr_device,
    )

    # Get text locations
    text_loc = get_text_position(
        ocr_img_proc,
        pred_text,
        degraded,
        OCR_model,
        tokenizer,
        location_prompt,
        conv_templates,
        default_image_token=DEFAULT_IMAGE_TOKEN,
        max_new_tokens=128,
        image_token_index=IMAGE_TOKEN_INDEX,
        device=cfg.ocr_device,
    )

    # Prepare low-quality tensor for SR
    LQ_img, h0, w0 = PIL2Tensor(degraded, upscale=cfg.upscale, min_size=cfg.min_size)
    LQ_img = LQ_img.unsqueeze(0).to(cfg.sr_device)[:, :3, :, :]

    # Combine prompts
    a_text = cfg.a_text_prompt + cfg.a_prompt
    n_text = cfg.n_text_prompt + cfg.n_prompt

    texture_prompt_with_loc = [text_loc + " " + caption[0]]

    # Sampling
    sample_fn = getattr(GLYPH_SR_model, "ping_pong_sampling")
    samples = sample_fn(
        LQ_img,
        caption,
        texture_prompt_with_loc,
        image_steps=cfg.edm_steps // 2,
        text_steps=cfg.text_steps // 2,
        restoration_scale=cfg.s_stage1,
        s_churn=cfg.s_churn,
        s_noise=cfg.s_noise,
        cfg_scale=cfg.s_cfg,
        control_scale=cfg.s_stage2,
        seed=cfg.seed,
        num_samples=cfg.num_samples,
        p_p=cfg.a_prompt,
        a_text_prompt=a_text,
        n_p=cfg.n_prompt,
        n_text_prompt=n_text,
        color_fix_type=cfg.color_fix_type,
        use_linear_CFG=cfg.linear_CFG,
        use_linear_control_scale=cfg.linear_s_stage2,
        cfg_scale_start=cfg.spt_linear_CFG,
        control_scale_start=cfg.spt_linear_s_stage2,
    )

    # Convert tensor back to PIL and save
    output_img = Tensor2PIL(samples[0], h0, w0)

    try:
        with open(meta_path, "r", encoding="utf-8") as mf:
            existing = sum(1 for _ in mf)
    except FileNotFoundError:
        existing = 0
    idx = existing + 1
    # Save with .png extension
    out_filename = f"GYLPH_SR{idx}.png"
    out_path = os.path.join(cfg.save_dir, out_filename)
    output_img.save(out_path)
    print(f"Saved super-resolved image to {out_path}")

    # Append metadata to JSONL
    metadata = {
        "filename": out_filename,
        "caption": caption,
        "pred_text": pred_text,
        "text_loc": text_loc,
        "settings": asdict(cfg),
        "ts_checkpoint_load_report": GLYPH_SR_model._ts_checkpoint_load_report,
    }
    with open(meta_path, "a", encoding="utf-8") as mf:
        mf.write(json.dumps(metadata, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
