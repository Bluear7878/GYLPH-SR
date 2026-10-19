"""CPU checks for the metadata and pixel contracts used by projection training."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image, ImageFilter
from torchvision import transforms

from glyph_sr.data import HQLQDataset, collate_training_samples, make_hqlq_dataloader


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.hq = self.root / "metadata_HQ.jsonl"
        self.neg = self.root / "metadata_with_neg.jsonl"
        Image.new("RGB", (36, 36), (0, 127, 255)).save(self.root / "positive.png")
        Image.new("RGB", (36, 36), (255, 127, 0)).save(self.root / "negative.png")
        self.hq.write_text(
            json.dumps(
                {
                    "prompt": "sign, realistic photograph, soft natural lighting.",
                    "HQ_path": "positive.png",
                }
            )
            + "\n"
        )
        self.write_negative([{"OCR": "STOP", "SR_OCR": "ST0P"}])

    def write_negative(self, records):
        self.neg.write_text(
            "".join(
                json.dumps(
                    {
                        "prompt": "sign",
                        "neg_path": "negative.png",
                        **record,
                    }
                )
                + "\n"
                for record in records
            )
        )

    def dataset(self, **options):
        return HQLQDataset(self.hq, self.neg, img_size=36, **options)

    def test_four_partitions_keep_metadata_and_prompt_contract(self):
        dataset = self.dataset()
        self.assertEqual([row["type"] for row in dataset.rows], ["P_HQ", "P_LQ", "N_HQ", "N_LQ"])
        for row in dataset.rows:
            self.assertNotIn("soft natural lighting", row["txt"])
            self.assertNotIn("realistic photograph", row["txt"])
        self.assertIn("every letter faithfully preserved", dataset.rows[0]["txt"])
        self.assertIn("distorted glyphs, random letters.", dataset.rows[2]["txt"])
        self.assertEqual((dataset.rows[0]["OCR"], dataset.rows[2]["OCR"]), ("", "STOP"))
        self.assertEqual(dataset.degrade_factors, [4, 5, 6, 7, 8, 9])
        self.assertEqual(dataset.blur_radius_range, (0.5, 2.5))

    def test_negative_filter_is_case_sensitive_and_rejects_any_word_overlap(self):
        self.write_negative(
            [
                {"OCR": "[TEXT]", "SR_OCR": "noise"},
                {"OCR": "STOP HERE", "SR_OCR": "STOP THERE"},
                {"OCR": "STOP", "SR_OCR": "stop"},
                {"OCR": "STOP", "SR_OCR": "ST0P"},
            ]
        )
        dataset = self.dataset(include_types=("N_HQ",))
        self.assertEqual([row["SR_OCR"] for row in dataset.rows], ["stop", "ST0P"])

    def test_negative_paths_remain_relative_to_hq_metadata_directory(self):
        elsewhere = self.root / "metadata"
        elsewhere.mkdir()
        moved = elsewhere / self.neg.name
        self.neg.rename(moved)
        dataset = HQLQDataset(self.hq, moved, img_size=36)
        self.assertEqual(Path(dataset.rows[2]["path"]), self.root / "negative.png")
        self.assertEqual(tuple(dataset[2]["input_tensor"].shape), (3, 36, 36))

    def test_rgb_normalization_and_loader_batch(self):
        loader = make_hqlq_dataloader(
            self.hq, self.neg, batch_size=4, img_size=36, num_workers=0, shuffle=False
        )
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["input_tensor"].shape), (4, 3, 36, 36))
        expected = torch.tensor([-1.0, 127 / 127.5 - 1, 1.0])
        torch.testing.assert_close(batch["input_tensor"][0, :, 0, 0], expected)
        self.assertEqual(batch["type"], ["P_HQ", "P_LQ", "N_HQ", "N_LQ"])

    def test_degradation_pixels_match_original_resize_and_blur_order(self):
        image = Image.new("RGB", (36, 36))
        for x in range(36):
            for y in range(36):
                image.putpixel((x, y), ((x * 19) % 256, (y * 23) % 256, (x * y) % 256))
        image.save(self.root / "positive.png")
        dataset = self.dataset(include_types=("P_LQ",))
        with (
            patch("glyph_sr.data.random.choice", return_value=6),
            patch("glyph_sr.data.random.uniform", return_value=1.5),
        ):
            actual = dataset[0]["input_tensor"]
        degraded = image.resize((6, 6), Image.Resampling.BICUBIC).resize(
            (36, 36), Image.Resampling.BICUBIC
        )
        degraded = degraded.filter(ImageFilter.GaussianBlur(radius=1.5))
        expected = transforms.functional.to_tensor(degraded) * 2 - 1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_unequal_shapes_fail_without_implicit_crop(self):
        Image.new("RGB", (72, 36)).save(self.root / "positive.png")
        dataset = self.dataset(include_types=("P_HQ", "P_LQ"))
        self.assertEqual(tuple(dataset[0]["input_tensor"].shape), (3, 36, 72))
        with self.assertRaisesRegex(ValueError, "does not crop"):
            collate_training_samples([dataset[0], dataset[1]])

    def test_invalid_partition_is_reported(self):
        with self.assertRaisesRegex(ValueError, "Unknown training partitions"):
            self.dataset(include_types=("typo",))


if __name__ == "__main__":
    unittest.main()
