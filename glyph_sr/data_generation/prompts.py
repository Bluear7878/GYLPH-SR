"""Extract OCR-conditioned synthesis prompts from an image folder."""

import argparse
import copy
import hashlib
from pathlib import Path

from glyph_sr.data.io import make_image_loader, write_record

from .common import move_image_tensors, release_memory, sanitize_prompt


def get_synthesis_prompt(
    image_tensor,
    image,
    model,
    tokenizer,
    ranked_candidates,
    conv_templates,
    image_token_index,
    conv_template="llava_llama_3",
    num_beams=1,
    temperature=0.7,
    do_sample=True,
    max_new_tokens=64,
    device="cuda",
):
    import torch

    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import tokenizer_image_token

    if "[TEXT]" in ranked_candidates and len(ranked_candidates.strip()) <= len("[TEXT]") + 1:
        return None
    ranked_candidates = ranked_candidates.splitlines()[0].strip()
    question = f"""{DEFAULT_IMAGE_TOKEN}
    You will write exactly one fluent English sentence to serve as a text-to-image prompt.
    The image contains the text {ranked_candidates}. Describe naturally how and where that text appears—
    specify the object or surface (e.g. plaque, sign, poster), the font style and color,
    the lighting and perspective, and the surrounding environment—while preserving the text
    exactly as shown within the asterisks."""
    conversation = copy.deepcopy(conv_templates[conv_template])
    conversation.tokenizer = tokenizer
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], None)
    input_ids = (
        tokenizer_image_token(
            conversation.get_prompt(), tokenizer, image_token_index, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(device)
    )
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=do_sample,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=False,
        )
    raw = tokenizer.decode(outputs.sequences[0].cpu().tolist(), skip_special_tokens=True).strip()
    prompt = "".join(char for char in raw if 32 <= ord(char) <= 126).strip()
    return prompt or None


def generate_prompt_meta_jsonl(
    yaml_file,
    loader,
    image_processor,
    base_model,
    tokenizer,
    conv_templates,
    conv_template,
    IMAGE_TOKEN_INDEX,
    prompt_bank="./prompt_bank",
    guide_prompt="realistic photograph, 35 mm film style, soft natural lighting.",
    max_samples=10000,
    device="cuda",
):
    import torch

    from glyph_sr.guidance import generate_ocr_text
    from llava.mm_utils import process_images

    from .common import load_prompt_templates

    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    ocr_prompt = load_prompt_templates(yaml_file)["ocr_prompt"]
    output = Path(prompt_bank)
    output.mkdir(parents=True, exist_ok=True)
    meta_path = output / "prompts_meta.jsonl"
    count = 0
    with meta_path.open("w", encoding="utf-8") as stream:
        for batch in loader:
            # Legacy callers yielded one dict; the canonical loader retains full batches.
            for sample in [batch] if isinstance(batch, dict) else batch:
                image = sample["image"]
                images = move_image_tensors(
                    process_images([image], image_processor, base_model.config),
                    device,
                    torch.float16,
                )
                try:
                    text, _ = generate_ocr_text(
                        ocr_prompt,
                        images,
                        [image.size],
                        base_model,
                        tokenizer,
                        conv_templates,
                        conv_template,
                        IMAGE_TOKEN_INDEX,
                        device,
                        max_new_tokens=77,
                    )
                    if not text:
                        continue
                    raw = get_synthesis_prompt(
                        images,
                        image,
                        base_model,
                        tokenizer,
                        text,
                        conv_templates,
                        IMAGE_TOKEN_INDEX,
                        conv_template=conv_template,
                        device=device,
                    )
                    if raw is None:
                        continue
                    clean = f"{sanitize_prompt(raw)}, {guide_prompt}"
                    if text not in clean:
                        continue
                    wrapped = clean.replace(text, f"**{text}**")
                    stem = hashlib.md5(wrapped.encode("utf-8")).hexdigest()[:16]
                    write_record(stream, {"id": stem, "OCR": text, "prompt": wrapped})
                    count += 1
                    if count >= max_samples:
                        return str(meta_path), count
                finally:
                    del images
                    release_memory()
    return str(meta_path), count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", "--image_dir", "-i", required=True)
    parser.add_argument("--prompt-bank", "--prompt_bank", "-p", default="prompt_bank")
    parser.add_argument("--prompt-yaml-file", "--prompt_yaml_file")
    parser.add_argument("--model-path", "--model_path", default="lmms-lab/llama3-llava-next-8b")
    parser.add_argument("--conv-template", "--conv_template", default="llava_llama_3")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4)
    parser.add_argument("--max-samples", "--max_samples", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def run(args):
    if not Path(args.image_dir).is_dir():
        raise NotADirectoryError(f"Image directory not found: {args.image_dir}")
    if args.batch_size < 1 or args.num_workers < 0 or args.max_samples < 1:
        raise ValueError("batch_size/max_samples must be positive and num_workers nonnegative")
    from glyph_sr.paths import resource_path
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.model.builder import load_pretrained_model

    prompt_yaml = args.prompt_yaml_file or resource_path("prompts/prompt_config.yaml")
    if not Path(prompt_yaml).is_file():
        raise FileNotFoundError(f"Prompt configuration not found: {prompt_yaml}")
    loader = make_image_loader(args.image_dir, args.batch_size, num_workers=args.num_workers)
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path,
        None,
        args.conv_template,
        device_map="cpu",
        attn_implementation=None,
    )
    model.eval().to(args.device)
    result = generate_prompt_meta_jsonl(
        prompt_yaml,
        loader,
        processor,
        model,
        tokenizer,
        conv_templates,
        args.conv_template,
        IMAGE_TOKEN_INDEX,
        args.prompt_bank,
        max_samples=args.max_samples,
        device=args.device,
    )
    print(f"Saved {result[1]} prompts to {result[0]}")
    return result


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
