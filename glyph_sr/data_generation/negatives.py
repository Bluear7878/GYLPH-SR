"""Restore degraded HQ images and collect text-corrupted negative examples."""

import argparse
import os
from pathlib import Path

from glyph_sr.data.io import load_metadata, write_record

from .common import move_image_tensors, random_masking_prompt, release_memory


def process_record(
    image_processor,
    ocr_model,
    sr_model,
    ocr_prompt,
    img_prompt,
    tokenizer,
    conv_templates,
    conv_template,
    mask_prob,
    rec,
    root_dir,
    neg_dir,
    down_factors,
    args,
):
    import torch
    from PIL import Image

    from glyph_sr.guidance import describe_image, generate_ocr_text
    from glyph_sr.images import degrade_image, pil_to_tensor, tensor_to_pil
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images

    hq_path = Path(root_dir) / rec["hq"]
    if not hq_path.is_file():
        raise FileNotFoundError(f"HQ image not found: {hq_path}")
    with Image.open(hq_path) as source:
        image = source.convert("RGB")
    images = move_image_tensors(
        process_images([image], image_processor, ocr_model.config),
        ocr_model.device,
        torch.float16,
    )
    original_text, _ = generate_ocr_text(
        ocr_prompt,
        images,
        [image.size],
        ocr_model,
        tokenizer,
        conv_templates,
        conv_template,
        IMAGE_TOKEN_INDEX,
        ocr_model.device,
        max_new_tokens=50,
    )
    caption = describe_image(
        image_tensor=images,
        image=image,
        model=ocr_model,
        tokenizer=tokenizer,
        prompt=img_prompt,
        conv_templates=conv_templates,
        image_token_index=IMAGE_TOKEN_INDEX,
        conv_template=conv_template,
        device=ocr_model.device,
    )
    caption[0] = random_masking_prompt(caption[0], mask_prob)
    for factor in down_factors:
        degraded = degrade_image(image, down_factor=factor)
        tensor, height, width = pil_to_tensor(
            degraded,
            upscale=args.upscale,
            min_size=args.min_size,
        )
        tensor = tensor.unsqueeze(0).to(sr_model.device)[:, :3]
        restored = sr_model.just_sampling(
            tensor,
            caption,
            num_steps=args.edm_steps,
            restoration_scale=args.s_stage1,
            s_churn=args.s_churn,
            s_noise=args.s_noise,
            cfg_scale=args.s_cfg,
            control_scale=args.s_stage2,
            seed=args.seed,
            num_samples=1,
            p_p=args.a_prompt,
            n_p=args.n_prompt,
            color_fix_type=args.color_fix_type,
            use_linear_CFG=args.linear_CFG,
            use_linear_control_scale=args.linear_s_stage2,
        )
        result = tensor_to_pil(restored[0], height, width)
        # Preserve the original single-factor branch: no OCR rejection step.
        if len(down_factors) == 1:
            path = Path(neg_dir) / f"{rec['id']}_neg.jpg"
            result.save(path)
            return True, {**rec, "neg": str(path), "neg_prompt": caption, "OCR": original_text}
        images = move_image_tensors(
            process_images([result], image_processor, ocr_model.config),
            ocr_model.device,
            torch.float16,
        )
        restored_text, _ = generate_ocr_text(
            ocr_prompt,
            images,
            [image.size],
            ocr_model,
            tokenizer,
            conv_templates,
            conv_template,
            IMAGE_TOKEN_INDEX,
            ocr_model.device,
            max_new_tokens=50,
        )
        if original_text != restored_text:
            path = Path(neg_dir) / f"{rec['id']}_neg.jpg"
            result.save(path)
            return True, {
                **rec,
                "neg": str(path),
                "neg_prompt": caption,
                "OCR": original_text,
                "SR_OCR": restored_text,
            }
    return False, rec


def ranked_filename(filename, world_size, local_rank):
    path = Path(filename)
    if path.name != filename:
        raise ValueError("Metadata filenames must not contain directories")
    if world_size == 1:
        return filename
    return f"{path.stem}.rank{local_rank:05d}-of-{world_size:05d}{path.suffix}"


def build_neg_lq(
    root_dir,
    image_processor,
    ocr_model,
    sr_model,
    ocr_prompt,
    img_prompt,
    tokenizer,
    conv_templates,
    conv_template,
    down_factors,
    args,
    hq_meta_filename="metadata_HQ.jsonl",
    neg_meta_filename="metadata_with_neg.jsonl",
    mask_prob=0.3,
    world_size=1,
    local_rank=0,
    output_dir=None,
    process_fn=None,
):
    """Write independent metadata shards; keep image paths relative to the HQ JSONL."""
    if world_size < 1 or not 0 <= local_rank < world_size:
        raise ValueError("Require world_size >= 1 and 0 <= local_rank < world_size")
    if not down_factors or any(factor <= 0 for factor in down_factors):
        raise ValueError("down_factors must contain positive values")
    if not 0 <= mask_prob <= 1:
        raise ValueError("mask_prob must be between zero and one")
    root = Path(root_dir)
    source_path = root / "metadata.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(f"Synthesis metadata not found: {source_path}")
    output = Path(output_dir) if output_dir else root
    hq_path = output / ranked_filename(hq_meta_filename, world_size, local_rank)
    neg_path = output / ranked_filename(neg_meta_filename, world_size, local_rank)
    if hq_path.resolve() == neg_path.resolve() or source_path.resolve() in {
        hq_path.resolve(),
        neg_path.resolve(),
    }:
        raise ValueError("Input, HQ and negative metadata must use distinct paths")
    negative_dir = output / "neg_lq"
    negative_dir.mkdir(parents=True, exist_ok=True)
    process_fn = process_fn or process_record
    total = negative_count = 0
    with (
        hq_path.open("w", encoding="utf-8") as hq_stream,
        neg_path.open("w", encoding="utf-8") as neg_stream,
    ):
        for index, record in enumerate(load_metadata(source_path)):
            if index % world_size != local_rank:
                continue
            stem = str(record["id"])
            if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
                raise ValueError(f"Metadata id must be a filename stem: {stem!r}")
            image_path = root / record["hq"]
            if not image_path.is_file():
                raise FileNotFoundError(f"HQ image not found: {image_path}")
            total += 1
            write_record(
                hq_stream,
                {
                    "id": record["id"],
                    "prompt": record.get("prompt", ""),
                    "HQ_path": os.path.relpath(image_path, output),
                    "OCR": record.get("OCR", ""),
                },
            )
            success, updated = process_fn(
                image_processor,
                ocr_model,
                sr_model,
                ocr_prompt,
                img_prompt,
                tokenizer,
                conv_templates,
                conv_template,
                mask_prob,
                record,
                str(root),
                str(negative_dir),
                down_factors,
                args,
            )
            if success:
                write_record(
                    neg_stream,
                    {
                        "id": updated["id"],
                        "prompt": updated.get("prompt", ""),
                        "neg_path": os.path.relpath(updated["neg"], output),
                        "OCR": updated.get("OCR", ""),
                        "SR_OCR": updated.get("SR_OCR", ""),
                    },
                )
                negative_count += 1
    return {
        "hq_metadata": str(hq_path),
        "negative_metadata": str(neg_path),
        "processed": total,
        "negatives": negative_count,
    }


def create_restoration_model(config_path, sign="Q", load_default_setting=False):
    from omegaconf import OmegaConf

    from glyph_sr.checkpoints import load_state_dict
    from sgm.util import instantiate_from_config

    if sign not in {None, "F", "Q"}:
        raise ValueError("SUPIR sign must be F or Q")
    config = OmegaConf.load(config_path)
    paths = [config.get("SDXL_CKPT"), config.get("SUPIR_CKPT")]
    if sign:
        paths.append(config[f"SUPIR_CKPT_{sign}"])
    for path in paths:
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"Restoration checkpoint not found: {path}")
    model = instantiate_from_config(config.model).cpu()
    for path in paths:
        if path is not None:
            model.load_state_dict(load_state_dict(path), strict=False)
    return (model, config.default_setting) if load_default_setting else model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-dir", "--img_dir", required=True)
    parser.add_argument("--save-dir", "--save_dir")
    parser.add_argument("--yaml-file", "--yaml_file")
    parser.add_argument("--prompt-yaml-file", "--prompt_yaml_file")
    parser.add_argument("--hq-meta-filename", "--hq_meta_filename", default="metadata_HQ.jsonl")
    parser.add_argument(
        "--neg-meta-filename", "--neg_meta_filename", default="metadata_with_neg.jsonl"
    )
    parser.add_argument("--down-factors", "--down_factors", type=int, nargs="+", default=[10, 15])
    parser.add_argument(
        "--supir-sign", "--SUPIR_sign", dest="SUPIR_sign", choices=["F", "Q"], default="Q"
    )
    parser.add_argument("--ocr-model", default="lmms-lab/llama3-llava-next-8b")
    parser.add_argument("--ocr-device", default="cuda:1")
    parser.add_argument("--sr-device", default="cuda:0")
    parser.add_argument("--conv-template", default="llava_llama_3")
    for name, default, kind in (
        ("upscale", 1, int),
        ("seed", -1, int),
        ("min_size", 1024, int),
        ("edm_steps", 50, int),
        ("s_stage1", -1, int),
        ("s_churn", 5, int),
        ("s_noise", 1.003, float),
        ("s_cfg", 7.5, float),
        ("s_stage2", 1.0, float),
        ("world_size", 1, int),
        ("local_rank", 0, int),
        ("mask_prob", 0.3, float),
        ("encoder_tile_size", 512, int),
        ("decoder_tile_size", 64, int),
    ):
        flags = list(dict.fromkeys((f"--{name.replace('_', '-')}", f"--{name}")))
        parser.add_argument(*flags, dest=name, type=kind, default=default)
    parser.add_argument(
        "--a-prompt",
        "--a_prompt",
        default=(
            "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
            "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, "
            "extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."
        ),
    )
    parser.add_argument(
        "--n-prompt",
        "--n_prompt",
        default=(
            "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, "
            "CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, "
            "frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"
        ),
    )
    parser.add_argument(
        "--color-fix-type",
        "--color_fix_type",
        choices=["None", "AdaIn", "Wavelet"],
        default="Wavelet",
    )
    parser.add_argument("--ae-dtype", "--ae_dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument(
        "--diff-dtype", "--diff_dtype", choices=["fp32", "fp16", "bf16"], default="fp16"
    )
    parser.add_argument(
        "--linear-CFG",
        "--linear_CFG",
        dest="linear_CFG",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--linear-s-stage2", "--linear_s_stage2", dest="linear_s_stage2", action="store_true"
    )
    parser.add_argument("--loading-half-params", "--loading_half_params", action="store_true")
    parser.add_argument("--use-tile-vae", "--use_tile_vae", action="store_true")
    return parser.parse_args(argv)


def run(args):
    source = Path(args.img_dir) / "metadata.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"Synthesis metadata not found: {source}")
    if args.world_size < 1 or not 0 <= args.local_rank < args.world_size:
        raise ValueError("Require world_size >= 1 and 0 <= local_rank < world_size")
    if not args.down_factors or any(factor <= 0 for factor in args.down_factors):
        raise ValueError("down_factors must contain positive values")
    if not 0 <= args.mask_prob <= 1:
        raise ValueError("mask_prob must be between zero and one")
    if args.upscale < 1 or args.min_size < 1 or args.edm_steps < 1:
        raise ValueError("upscale, min_size and edm_steps must be positive")
    import torch

    from glyph_sr.images import convert_dtype
    from glyph_sr.paths import resource_path
    from llava.conversation import conv_templates
    from llava.model.builder import load_pretrained_model

    from .common import load_prompt_templates

    model_yaml = args.yaml_file or resource_path("models/SUPIR.yaml")
    prompt_yaml = args.prompt_yaml_file or resource_path("prompts/prompt_config.yaml")
    for path in (model_yaml, prompt_yaml):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Generation configuration not found: {path}")
    torch.manual_seed(args.seed)
    prompts = load_prompt_templates(prompt_yaml)
    tokenizer, ocr, processor, _ = load_pretrained_model(
        args.ocr_model,
        None,
        args.conv_template,
        device_map="cpu",
        attn_implementation=None,
    )
    ocr.eval().to(args.ocr_device)
    sr = create_restoration_model(model_yaml, args.SUPIR_sign)
    if args.loading_half_params:
        sr = sr.half()
    if args.use_tile_vae:
        sr.init_tile_vae(
            encoder_tile_size=args.encoder_tile_size, decoder_tile_size=args.decoder_tile_size
        )
    sr.ae_dtype = convert_dtype(args.ae_dtype)
    sr.model.dtype = convert_dtype(args.diff_dtype)
    sr.eval().to(args.sr_device)

    def generate(*positional):
        try:
            with torch.inference_mode():
                return process_record(*positional)
        finally:
            release_memory()

    result = build_neg_lq(
        args.img_dir,
        processor,
        ocr,
        sr,
        prompts["ocr_prompt"],
        prompts["img_prompt"],
        tokenizer,
        conv_templates,
        args.conv_template,
        args.down_factors,
        args,
        args.hq_meta_filename,
        args.neg_meta_filename,
        args.mask_prob,
        args.world_size,
        args.local_rank,
        args.save_dir,
        generate,
    )
    print(f"Processed {result['processed']} images; saved {result['negatives']} negatives")
    print(f"HQ metadata: {result['hq_metadata']}; negative metadata: {result['negative_metadata']}")
    return result


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
