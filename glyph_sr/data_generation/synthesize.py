"""Generate the HQ corpus using FLUX.1-dev, Nunchaku and the realism LoRA."""

import argparse
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

from glyph_sr.data import load_metadata, write_record

# The style list is the original augmentation recipe, in its original order.
AUGMENTATION_STYLES = (
    "overcast lighting",
    "Fujifilm Pro 400H color palette",
    "cinematic bokeh",
    "vintage grain",
    "soft natural lighting",
    "dramatic shadows",
    "warm color tones",
    "shallow depth of field",
    "sunset glow",
    "golden hour warmth",
    "high-key lighting",
    "low-key moody lighting",
    "lens flare accents",
    "silhouette composition",
    "wide-angle perspective",
    "telephoto compression",
    "tilt-shift miniaturization",
    "macro close-up detail",
    "reflections in water",
    "symmetry and leading lines",
    "rule-of-thirds framing",
    "cinematic aspect ratio",
    "soft vignette",
    "rainy window droplets",
    "misty atmosphere",
    "backlit rim lighting",
    "analog film scratches",
    "HDR-like dynamic range",
    "pastel color grading",
    "neon color accents",
    "cold blue tones",
    "warm amber highlights",
    "soft focus glow",
    "textured overlay grain",
    "dramatic sky replacement",
    "urban night lights",
    "forest dappled light",
    "architectural leading lines",
)


def chunk(records, size):
    if size < 1:
        raise ValueError("batch_size must be positive")
    records = iter(records)
    while group := list(islice(records, size)):
        yield group


def expand_prompts(records, styles=()):
    for record in records:
        stem = str(record["id"])
        if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
            raise ValueError(f"Metadata id must be a filename stem: {stem!r}")
        for style in styles or (None,):
            prompt = record["prompt"]
            variant = stem
            if style is not None:
                variant = f"{stem}_{style.replace(' ', '_').replace('/', '')}"
                prompt = f"{style} , {prompt}"
            yield {**record, "id": variant, "prompt": prompt}


def synthesize_records(
    records, pipeline, output_dir, batch_size=1, steps=50, styles=(), generator=None
):
    """Run an injected Flux pipeline; record each image's actual prompt."""
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    output = Path(output_dir)
    hq_dir = output / "hq"
    hq_dir.mkdir(parents=True, exist_ok=True)
    metadata = output / "metadata.jsonl"
    count = 0
    seen = set()
    with metadata.open("w", encoding="utf-8") as stream:
        for group in chunk(expand_prompts(records, styles), batch_size):
            for record in group:
                if record["id"] in seen:
                    raise ValueError(f"Duplicate output id: {record['id']}")
                seen.add(record["id"])
            options = {"num_inference_steps": steps}
            if generator is not None:
                options["generator"] = generator
            images = pipeline([record["prompt"] for record in group], **options).images
            if len(images) != len(group):
                raise RuntimeError("Flux returned a different number of images than prompts")
            for image, record in zip(images, group):
                relative = Path("hq") / f"{record['id']}.png"
                image.save(output / relative)
                result = {"id": record["id"], "prompt": record["prompt"], "hq": relative.as_posix()}
                if "OCR" in record:
                    result["OCR"] = record["OCR"]
                write_record(stream, result)
                count += 1
    return str(metadata), count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-meta", required=True, help="OCR/prompt JSONL produced by the prompts command"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--augment-styles",
        action="store_true",
        help="Expand each prompt using the original 38 photographic styles",
    )
    parser.add_argument("--model-path", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument(
        "--transformer-path",
        help="Override the precision-dependent mit-han-lab/svdq-<precision>-flux.1-dev model",
    )
    parser.add_argument("--lora-path", default="XLabs-AI/flux-RealismLora/lora.safetensors")
    parser.add_argument("--lora-strength", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seed", type=int, help="Optional seed; omitted preserves the original unseeded generation"
    )
    return parser.parse_args(argv)


def run(args):
    if not Path(args.prompt_meta).is_file():
        raise FileNotFoundError(f"Prompt metadata not found: {args.prompt_meta}")
    if args.batch_size < 1 or args.steps < 1:
        raise ValueError("batch_size and steps must be positive")
    import torch
    from diffusers import FluxPipeline
    from nunchaku import NunchakuFluxTransformer2dModel
    from nunchaku.utils import get_precision

    transformer_path = args.transformer_path or f"mit-han-lab/svdq-{get_precision()}-flux.1-dev"
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(transformer_path)
    pipeline = FluxPipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    pipeline.transformer.update_lora_params(args.lora_path)
    pipeline.transformer.set_lora_strength(args.lora_strength)
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda")
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        result = synthesize_records(
            load_metadata(args.prompt_meta),
            pipeline,
            args.output_dir,
            args.batch_size,
            args.steps,
            AUGMENTATION_STYLES if args.augment_styles else (),
            generator,
        )
    print(f"Saved {result[1]} HQ images and metadata to {result[0]}")
    return result


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
