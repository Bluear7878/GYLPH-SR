import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from GLYPHSR.checkpoint import load_TS_pretrained_model
from tools.score_ocr_predictions import aggregate_counts, corrected_word_counts, legacy_counts
import run_GLYPH_SR as inference
import train_GLYPH_SR as training


def tiny_model():
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.diffusion_model = torch.nn.Module()
    model.model.diffusion_model.project_modules = torch.nn.Linear(2, 2)
    model.model.diffusion_model.base = torch.nn.Linear(2, 2)
    return model


class ReleaseTests(unittest.TestCase):
    def test_entrypoint_help_without_runtime_imports(self):
        for script in ("run_GLYPH_SR.py", "train_GLYPH_SR.py"):
            # -S removes site packages: help must not import torch/transformers.
            result = subprocess.run([sys.executable, "-S", script, "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--yaml_file", result.stdout)

    def test_required_inputs_and_device_options(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                inference.parse_args([])
            with self.assertRaises(SystemExit):
                training.parse_args([])
        args = inference.parse_args(["--img_path", "input.png", "--ocr_device", "cuda:0"])
        self.assertEqual(args.ocr_device, "cuda:0")

    def test_missing_input_fails_before_model_loading(self):
        with self.assertRaisesRegex(FileNotFoundError, "img_path"):
            inference.validate_inputs(inference.Config(img_path="missing-input.png"))

    def test_checkpoint_environment_paths(self):
        from omegaconf import OmegaConf
        with patch.dict(os.environ, {"GLYPH_SDXL_CKPT": "/custom/model.safetensors"}):
            config = OmegaConf.load("model_configs/juggernautXL.yaml")
            self.assertEqual(config.SDXL_CKPT, "/custom/model.safetensors")

    def test_lightning_and_bare_checkpoint_weights_are_loaded(self):
        for kind in ("lightning", "full", "inner"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                source, target = tiny_model(), tiny_model()
                for value in source.parameters():
                    torch.nn.init.constant_(value, 3.0)
                state = source.model.state_dict() if kind == "inner" else source.state_dict()
                payload = {"state_dict": state, "epoch": 1} if kind == "lightning" else state
                path = Path(tmp) / "model.ckpt"
                torch.save(payload, path)
                load_TS_pretrained_model(target, path, "cpu")
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(source.parameters(), target.parameters())))
                self.assertEqual(target._ts_checkpoint_load_report["matched_projection_tensor_count"], 2)

    def test_unrelated_and_incomplete_checkpoints_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.ckpt"
            target = tiny_model()
            torch.save({"unrelated": torch.ones(2)}, path)
            with self.assertRaisesRegex(RuntimeError, "matches no"):
                load_TS_pretrained_model(target, path, "cpu")
            weights = target.state_dict()
            weights.pop("model.diffusion_model.project_modules.bias")
            torch.save(weights, path)
            with self.assertRaisesRegex(RuntimeError, "every text projection"):
                load_TS_pretrained_model(target, path, "cpu")

    def test_extra_predictions_penalized_only_by_corrected_metric(self):
        self.assertEqual(corrected_word_counts("STOP", "STOP EXTRA"), (1, 1, 0))
        self.assertEqual(legacy_counts("STOP", "STOP EXTRA"), (1, 0, 0))

    def test_micro_counts_handle_duplicates_and_empty_prediction(self):
        self.assertEqual(corrected_word_counts("a a", "a"), (1, 0, 1))
        rows = [{"ground_truth": "a a", "predicted": "a"}, {"ground_truth": "b", "predicted": ""}]
        result = aggregate_counts(rows, corrected_word_counts)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 2))
        self.assertEqual(result["f1"], 50.0)

    def test_image_resize_with_supported_pillow(self):
        from PIL import Image
        from llava.mm_utils import auto_pad_images, resize_and_center_crop
        image = Image.new("RGB", (40, 20), (10, 20, 30))
        crop = resize_and_center_crop(image, 12)
        padded = auto_pad_images(image, [16, 32])
        self.assertEqual(crop.size, (12, 12))
        self.assertEqual(crop.getpixel((0, 0)), (10, 20, 30))
        self.assertIn(padded.size, [(16, 16), (16, 32), (32, 16), (32, 32)])

    def test_scorer_cli(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "predictions.jsonl", Path(tmp) / "scores.json"
            source.write_text(json.dumps({"filename": "synthetic", "ground_truth": "STOP", "predicted": "STOP EXTRA"}) + "\n")
            result = subprocess.run([sys.executable, "tools/score_ocr_predictions.py", "--input", str(source), "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            scores = json.loads(output.read_text())
            self.assertAlmostEqual(scores["corrected_word"]["f1"], 200 / 3)
            self.assertEqual(scores["legacy_submitted_word"]["f1"], 100)


if __name__ == "__main__":
    unittest.main()
