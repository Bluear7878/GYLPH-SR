"""Exercise generation orchestration with CPU inputs and injected model outputs."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from glyph_sr.data import load_metadata, make_image_loader, move_image_tensors
from glyph_sr.data_generation.negatives import build_neg_lq, process_record
from glyph_sr.data_generation.prompts import generate_prompt_meta_jsonl
from glyph_sr.data_generation.synthesize import AUGMENTATION_STYLES, synthesize_records


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_every_command_help_works_without_site_packages(self):
        for command in ("prompts", "synthesize", "negatives"):
            result = subprocess.run(
                [sys.executable, "-S", "-m", f"glyph_sr.data_generation.{command}", "--help"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)

    def test_image_loader_does_not_drop_larger_batches(self):
        for number in range(3):
            Image.new("RGB", (16, 16)).save(self.root / f"{number}.PNG")
        loader = make_image_loader(self.root, batch_size=2, num_workers=0)
        batches = list(loader)
        self.assertEqual([len(batch) for batch in batches], [2, 1])
        self.assertEqual(len({sample["path"] for batch in batches for sample in batch}), 3)

    def test_prompt_generation_forwards_template_device_and_processes_every_image(self):
        from glyph_sr.data_generation import prompts

        images = [{"image": Image.new("RGB", (16, 16))} for _ in range(3)]
        model = SimpleNamespace(config=SimpleNamespace())
        with (
            patch(
                "glyph_sr.data.load_prompt_templates",
                return_value={"ocr_prompt": "read"},
            ),
            patch(
                "glyph_sr.guidance.generate_ocr_text",
                side_effect=[("ONE", None), ("TWO", None), ("THREE", None)],
            ),
            patch(
                "glyph_sr._vendor.llava.mm_utils.process_images",
                return_value=torch.zeros(1, 3, 16, 16),
            ),
            patch.object(
                prompts, "get_synthesis_prompt", side_effect=["sign ONE", "sign TWO", "sign THREE"]
            ) as synthesis,
            patch.object(prompts, "release_memory"),
        ):
            path, count = generate_prompt_meta_jsonl(
                "unused.yaml",
                [images[:2], images[2:]],
                None,
                model,
                None,
                {"custom": None},
                "custom",
                -200,
                str(self.root),
                device="cpu",
            )
        self.assertEqual(count, 3)
        self.assertEqual([row["OCR"] for row in load_metadata(path)], ["ONE", "TWO", "THREE"])
        self.assertTrue(
            all(
                call.kwargs["conv_template"] == "custom" and call.kwargs["device"] == "cpu"
                for call in synthesis.call_args_list
            )
        )

    def test_tensor_and_list_processor_outputs_are_supported(self):
        tensor = torch.ones(1, 3, 2, 2)
        for value in (tensor, [tensor], (tensor,)):
            result = move_image_tensors(value, "cpu", torch.float16)
            items = result if isinstance(result, (list, tuple)) else [result]
            self.assertTrue(all(item.dtype == torch.float16 for item in items))

    def test_augmented_synthesis_records_actual_prompt_for_duplicate_base_id(self):
        calls = []

        def pipeline(prompts, **options):
            calls.append((prompts, options))
            return SimpleNamespace(images=[Image.new("RGB", (16, 16)) for _ in prompts])

        source = [{"id": "one", "prompt": "STOP sign", "OCR": "STOP"}]
        path, count = synthesize_records(
            source, pipeline, self.root, batch_size=2, styles=("warm light", "cool light")
        )
        rows = list(load_metadata(path))
        self.assertEqual(count, 2)
        self.assertEqual(
            [row["prompt"] for row in rows], ["warm light , STOP sign", "cool light , STOP sign"]
        )
        self.assertEqual(len(AUGMENTATION_STYLES), 38)
        self.assertEqual(calls[0][1]["num_inference_steps"], 50)
        self.assertTrue(all((self.root / row["hq"]).is_file() for row in rows))
        self.assertEqual(rows[0]["OCR"], "STOP")

    def test_rank_shards_are_disjoint_and_paths_follow_output_metadata(self):
        source = self.root / "source"
        source.mkdir()
        rows = []
        for number in range(4):
            Image.new("RGB", (16, 16)).save(source / f"{number}.png")
            rows.append({"id": str(number), "prompt": "sign", "hq": f"{number}.png"})
        (source / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        output = self.root / "out"

        def process(*values):
            record, directory = values[9], Path(values[11])
            path = directory / f"{record['id']}_neg.jpg"
            Image.new("RGB", (16, 16)).save(path)
            return True, {**record, "neg": str(path), "OCR": "STOP", "SR_OCR": "ST0P"}

        results = [
            build_neg_lq(
                source,
                None,
                None,
                None,
                "",
                "",
                None,
                {},
                "custom",
                [10, 15],
                None,
                world_size=2,
                local_rank=rank,
                output_dir=output,
                process_fn=process,
            )
            for rank in range(2)
        ]
        self.assertNotEqual(results[0]["hq_metadata"], results[1]["hq_metadata"])
        ids = [set(row["id"] for row in load_metadata(result["hq_metadata"])) for result in results]
        self.assertEqual(ids, [{"0", "2"}, {"1", "3"}])
        for result in results:
            self.assertEqual((result["processed"], result["negatives"]), (2, 2))
            for field, key in (("hq_metadata", "HQ_path"), ("negative_metadata", "neg_path")):
                self.assertTrue(
                    all((output / row[key]).is_file() for row in load_metadata(result[field]))
                )

    def test_single_and_multiple_degradation_factor_selection_contract(self):
        from glyph_sr.data_generation import negatives

        Image.new("RGB", (36, 36)).save(self.root / "source.png")
        args = negatives.parse_args(["--img-dir", str(self.root)])
        tensor = torch.zeros(3, 36, 36)
        ocr = SimpleNamespace(config=SimpleNamespace(), device="cpu")
        sr = SimpleNamespace(device="cpu", just_sampling=lambda *a, **k: tensor.unsqueeze(0))
        for factors, outputs, expected_calls in (
            ([10], ["STOP"], 1),
            ([10, 15], ["STOP", "STOP", "ST0P"], 3),
        ):
            with (
                self.subTest(factors=factors),
                patch(
                    "glyph_sr.guidance.generate_ocr_text",
                    side_effect=[(text, None) for text in outputs],
                ) as recognize,
                patch("glyph_sr.guidance.describe_image", return_value=["a sign"]),
                patch("glyph_sr._vendor.llava.mm_utils.process_images", return_value=[tensor]),
                patch("glyph_sr.images.degrade_image", side_effect=lambda image, **kwargs: image),
                patch("glyph_sr.images.pil_to_tensor", return_value=(tensor, 36, 36)),
                patch("glyph_sr.images.tensor_to_pil", return_value=Image.new("RGB", (36, 36))),
            ):
                success, record = process_record(
                    None,
                    ocr,
                    sr,
                    "read",
                    "describe",
                    None,
                    {},
                    "custom",
                    0.3,
                    {"id": "one", "hq": "source.png"},
                    str(self.root),
                    str(self.root),
                    factors,
                    args,
                )
            self.assertTrue(success)
            self.assertEqual(recognize.call_count, expected_calls)
            self.assertEqual(record.get("SR_OCR", ""), "" if len(factors) == 1 else "ST0P")


if __name__ == "__main__":
    unittest.main()
