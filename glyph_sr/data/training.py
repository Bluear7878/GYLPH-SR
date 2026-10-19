"""The released four-partition dataset, with unchanged prompts and degradation."""

import random
import re
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .io import load_metadata

# Conditioning strings are part of the released four-partition training recipe.
HQ_PROMPT = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
    "hyper detailed photo-realistic maximum detail, 32k, Color Grading, ultra HD, "
    "extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations"
)
LQ_PROMPT = (
    "painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, "
    "3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, "
    "watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"
)
POSITIVE_PROMPT = (
    "The scene text in the image is rendered in crystal-clear clarity with razor-sharp edges, "
    "every letter faithfully preserved in high-resolution detail, seamless spacing, "
    "and no distortion or noise"
)

NEGATIVE_PROMPT = (
    "The scene text appears blurred, smeared or partially broken;"
    "misaligned text, shifted text, incorrect position, distorted glyphs, random letters."
)


class HQLQDataset(Dataset):
    def __init__(
        self,
        hq_meta_path: str,
        neg_meta_path: str,
        img_size: int = 512,
        include_types=("P_HQ", "N_HQ", "P_LQ", "N_LQ"),
    ):
        super().__init__()
        if img_size < 9:
            raise ValueError("img_size must be at least 9 for the released degradation factors")
        self.img_size = img_size
        self.include_types = set(include_types)
        unknown = self.include_types - {"P_HQ", "P_LQ", "N_HQ", "N_LQ"}
        if unknown:
            raise ValueError(f"Unknown training partitions: {sorted(unknown)}")

        self.to_tensor = transforms.Compose(
            [
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.degrade_factors = [4, 5, 6, 7, 8, 9]
        self.blur_radius_range = (0.5, 2.5)

        self.hq_base = Path(hq_meta_path).parent
        self.rows = []

        unwanted = [
            "realistic photograph",
            "35 mm film style",
            "soft natural lighting",
            "overcast lighting",
        ]

        def clean_meta(prompt: str) -> str:
            parts = [p.strip() for p in prompt.rstrip(".").split(",")]
            parts = [p for p in parts if not any(u in p for u in unwanted)]
            return ", ".join(parts)

        # Positive partitions share each HQ image.
        for rec in load_metadata(hq_meta_path):
            meta = clean_meta(rec["prompt"])
            img_path = self.hq_base / rec["HQ_path"]

            if "P_HQ" in self.include_types:
                self.rows.append(
                    {
                        "type": "P_HQ",
                        "path": str(img_path),
                        "txt": f"{meta}.{POSITIVE_PROMPT}. {HQ_PROMPT}.",
                        "SR_OCR": "",
                        "OCR": "",
                    }
                )
            if "P_LQ" in self.include_types:
                self.rows.append(
                    {
                        "type": "P_LQ",
                        "path": str(img_path),
                        "txt": f"{meta}. {POSITIVE_PROMPT}. {LQ_PROMPT}.",
                        "SR_OCR": "",
                        "OCR": "",
                    }
                )

        # Keep the historical case-sensitive OCR overlap filter.
        for rec in load_metadata(neg_meta_path):
            meta = clean_meta(rec["prompt"])

            # 1) if OCR is “[TEXT]” ---> skip
            ocr_text = rec.get("OCR", "").strip()
            if ocr_text == "[TEXT]":
                continue

            # 2) Skip if there is even one OCR word that matches exactly in SR_OCR
            sr_ocr_text = rec.get("SR_OCR", "")
            ocr_words = re.findall(r"\w+", ocr_text)
            sr_words = re.findall(r"\w+", sr_ocr_text)
            if any(word in sr_words for word in ocr_words):
                continue

            img_path = self.hq_base / rec["neg_path"]

            if "N_HQ" in self.include_types:
                self.rows.append(
                    {
                        "type": "N_HQ",
                        "path": str(img_path),
                        "txt": f"{meta}.{NEGATIVE_PROMPT}. {HQ_PROMPT}.",
                        "SR_OCR": rec["SR_OCR"],
                        "OCR": rec["OCR"],
                    }
                )
            if "N_LQ" in self.include_types:
                self.rows.append(
                    {
                        "type": "N_LQ",
                        "path": str(img_path),
                        "txt": f"{meta}. {NEGATIVE_PROMPT}. {LQ_PROMPT}.",
                        "SR_OCR": rec["SR_OCR"],
                        "OCR": rec["OCR"],
                    }
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        with Image.open(rec["path"]) as source:
            img = source.convert("RGB")

        if rec["type"].endswith("_LQ"):
            factor = random.choice(self.degrade_factors)
            small = img.resize((self.img_size // factor, self.img_size // factor), Image.BICUBIC)
            img = small.resize((self.img_size, self.img_size), Image.BICUBIC)

            radius = random.uniform(*self.blur_radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        tensor = self.to_tensor(img)
        return {
            "input_tensor": tensor,
            "txt": rec["txt"],
            "type": rec["type"],
            "SR_OCR": rec["SR_OCR"],
            "OCR": rec["OCR"],
        }


def make_hqlq_dataloader(
    hq_meta_path: str,
    neg_meta_path: str,
    batch_size: int = 8,
    img_size: int = 512,
    include_types=("P_HQ", "P_LQ", "N_HQ", "N_LQ"),
    shuffle: bool = True,
    num_workers: int = 4,
):
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    ds = HQLQDataset(
        hq_meta_path=hq_meta_path,
        neg_meta_path=neg_meta_path,
        img_size=img_size,
        include_types=include_types,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_training_samples,
    )


def collate_training_samples(batch):
    """Stack equal-size inputs and retain the conditioning metadata unchanged."""
    shapes = {tuple(sample["input_tensor"].shape) for sample in batch}
    if len(shapes) != 1:
        raise ValueError(
            f"Training images produced unequal tensor shapes {sorted(shapes)}. "
            "Prepare images with a consistent aspect ratio (square for mixed HQ/LQ batches) "
            "or use batch_size=1; the loader does not crop images."
        )
    return {
        "input_tensor": torch.stack([sample["input_tensor"] for sample in batch]),
        **{key: [sample[key] for sample in batch] for key in ("txt", "type", "OCR", "SR_OCR")},
    }
