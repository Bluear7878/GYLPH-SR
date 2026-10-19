"""Exercise public inference orchestration without loading model weights."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from glyph_sr.config import InferenceConfig
from glyph_sr.inference import RestorationPipeline, run


def stub_pipeline(**settings):
    config = InferenceConfig(
        sr_device="cpu", ocr_device="cpu", caption_device="cpu", min_size=64, **settings
    )
    pipeline = RestorationPipeline.__new__(RestorationPipeline)
    pipeline.config = config
    samples = torch.stack(
        [torch.full((3, 64, 128), value) for value in torch.linspace(-1, 1, config.num_samples)]
    )
    pipeline.model = SimpleNamespace(
        ping_pong_sampling=Mock(return_value=samples),
        _ts_checkpoint_load_report={"load_target": "full_model", "matched_tensor_count": 2},
    )
    pipeline.caption_model = SimpleNamespace(config=SimpleNamespace())
    pipeline.ocr_model = object()
    pipeline.image_processor = object()
    pipeline.tokenizer = object()
    pipeline.dtype = torch.float16
    pipeline.prompts = {"img_prompt": "caption", "ocr_prompt": "OCR", "location_prompt": "location"}
    return pipeline


def restore_with_stub_guidance(pipeline, processed):
    with (
        patch("glyph_sr.guidance.describe_image", return_value=["scene"]),
        patch("glyph_sr.guidance.generate_ocr_text", return_value=("STOP", None)),
        patch("glyph_sr.guidance.describe_text_locations", return_value="location"),
        patch("llava.mm_utils.process_images", return_value=processed),
    ):
        return pipeline.restore(Image.new("L", (80, 40)))


class PipelineTests(unittest.TestCase):
    def test_odd_step_count_and_every_sample_keep_the_requested_output_size(self):
        for processed in (torch.zeros(1, 3, 40, 80), [torch.zeros(3, 40, 80)]):
            with self.subTest(processor_output=type(processed).__name__):
                pipeline = stub_pipeline(steps=7, num_samples=3)
                images, cues = restore_with_stub_guidance(pipeline, processed)
                self.assertEqual(len(images), 3)
                self.assertEqual([image.size for image in images], [(160, 80)] * 3)
                self.assertEqual(
                    [image.getpixel((0, 0)) for image in images],
                    [(0, 0, 0), (127, 127, 127), (255, 255, 255)],
                )
                args, options = pipeline.model.ping_pong_sampling.call_args
                self.assertEqual(args[0].shape, (1, 3, 64, 128))
                self.assertEqual(args[0].device.type, "cpu")
                self.assertEqual(args[1:], (["scene"], ["location scene"]))
                self.assertEqual((options["image_steps"], options["text_steps"]), (3, 4))
                self.assertEqual(options["num_samples"], 3)
                self.assertEqual(cues["pred_text"], "STOP")
                self.assertEqual(
                    cues["ts_checkpoint_load_report"], pipeline.model._ts_checkpoint_load_report
                )

    def test_legacy_text_steps_override_preserves_the_two_half_step_counts(self):
        for steps, text_steps, expected in ((7, 12, (3, 6)), (50, 50, (25, 25))):
            with self.subTest(steps=steps, text_steps=text_steps):
                pipeline = stub_pipeline(steps=steps, text_steps=text_steps)
                restore_with_stub_guidance(pipeline, torch.zeros(1, 3, 40, 80))
                options = pipeline.model.ping_pong_sampling.call_args.kwargs
                self.assertEqual((options["image_steps"], options["text_steps"]), expected)

    def test_run_saves_every_sample_and_appends_metadata_across_repeated_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "input.png"
            Image.new("RGB", (80, 40)).save(source)
            output = folder / "outputs"
            output.mkdir()
            checkpoint = folder / "model.ckpt"
            checkpoint.touch()
            adapter = folder / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapter_model.safetensors").touch()
            metadata = output / "records.jsonl"
            history = '{"filename":"historical.png","note":"preserve this row"}\n'
            metadata.write_text(history)
            existing = {}
            for index in (1, 3):
                path = output / f"input_sr_{index:04d}.png"
                Image.new("RGB", (3, 3), (10, 20, index)).save(path)
                existing[path] = path.read_bytes()

            config = InferenceConfig(
                input_path=str(source),
                output_dir=str(output),
                num_samples=3,
                ts_checkpoint=str(checkpoint),
                ocr_adapter=str(adapter),
                metadata_filename=metadata.name,
            )
            images = [
                Image.new("RGB", (160, 80), color) for color in ((1, 2, 3), (4, 5, 6), (7, 8, 9))
            ]
            cues = {"caption": ["scene"], "pred_text": "STOP", "text_loc": "location"}
            pipeline = SimpleNamespace(restore=Mock(return_value=(images, cues)))
            with patch("glyph_sr.inference.RestorationPipeline", return_value=pipeline):
                first = run(config)
                second = run(config)

            self.assertEqual(
                [path.name for path in first],
                ["input_sr_0002.png", "input_sr_0004.png", "input_sr_0005.png"],
            )
            self.assertEqual(
                [path.name for path in second],
                ["input_sr_0006.png", "input_sr_0007.png", "input_sr_0008.png"],
            )
            for path, original in existing.items():
                self.assertEqual(path.read_bytes(), original)
            text = metadata.read_text()
            self.assertTrue(text.startswith(history))
            rows = [json.loads(line) for line in text.splitlines()]
            self.assertEqual(len(rows), 7)
            self.assertEqual(
                [row["filename"] for row in rows[1:]], [path.name for path in first + second]
            )
            for index, path in enumerate(first + second):
                with Image.open(path) as saved:
                    self.assertEqual(saved.size, (160, 80))
                    self.assertEqual(saved.getpixel((0, 0)), images[index % 3].getpixel((0, 0)))
                self.assertEqual(rows[index + 1]["settings"]["num_samples"], 3)
                self.assertEqual(rows[index + 1]["pred_text"], "STOP")


if __name__ == "__main__":
    unittest.main()
