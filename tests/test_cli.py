"""Check canonical and historical CLI options without constructing models."""

import contextlib
import io
import subprocess
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

from glyph_sr.config import (
    InferenceConfig,
    RuntimeConfig,
    TrainingConfig,
    parse_config,
    validate_runtime,
)
from run_GLYPH_SR import Config
from train_GLYPH_SR import TrainConfig

ROOT = Path(__file__).resolve().parents[1]


def inference_options(*arguments):
    return parse_config(InferenceConfig, ["--input", "input.png", *arguments])


def training_options(*arguments):
    return parse_config(
        TrainingConfig,
        [
            "--hq-metadata",
            "hq.jsonl",
            "--negative-metadata",
            "negative.jsonl",
            "--validation-hq-metadata",
            "validation.jsonl",
            "--validation-negative-metadata",
            "validation-negative.jsonl",
            *arguments,
        ],
    )


class CLITests(unittest.TestCase):
    def test_canonical_boolean_flags_follow_their_names(self):
        defaults = inference_options()
        self.assertTrue(defaults.linear_cfg)
        self.assertFalse(defaults.linear_control_scale)
        self.assertFalse(defaults.tile_vae)
        self.assertFalse(hasattr(defaults, "loading_half_params"))
        for arguments, name, expected in (
            (("--linear-cfg",), "linear_cfg", True),
            (("--no-linear-cfg",), "linear_cfg", False),
            (("--linear-control-scale",), "linear_control_scale", True),
            (
                ("--linear-control-scale", "--no-linear-control-scale"),
                "linear_control_scale",
                False,
            ),
            (("--tile-vae",), "tile_vae", True),
            (("--tile-vae", "--no-tile-vae"), "tile_vae", False),
        ):
            with self.subTest(arguments=arguments):
                self.assertIs(getattr(inference_options(*arguments), name), expected)

    def test_legacy_boolean_direction_is_retained(self):
        for option, name, expected in (
            ("--linear_CFG", "linear_cfg", False),
            ("--linear_s_stage2", "linear_control_scale", True),
            ("--use_tile_vae", "tile_vae", True),
        ):
            with self.subTest(option=option):
                self.assertIs(getattr(inference_options(option), name), expected)

    def test_legacy_and_canonical_values_produce_the_same_configuration(self):
        canonical = parse_config(
            InferenceConfig,
            [
                "--input",
                "image.png",
                "--steps",
                "7",
                "--text-steps",
                "12",
                "--cfg-scale",
                "6",
                "--no-linear-cfg",
                "--output-dir",
                "results",
            ],
        )
        legacy = parse_config(
            InferenceConfig,
            [
                "--img_path",
                "image.png",
                "--edm_steps",
                "7",
                "--text_steps",
                "12",
                "--s_cfg",
                "6",
                "--linear_CFG",
                "--save_dir",
                "results",
            ],
        )
        self.assertEqual(vars(canonical), vars(legacy))
        self.assertIsInstance(legacy.text_steps, int)
        self.assertIsNone(inference_options().text_steps)

    def test_legacy_python_config_constructors_translate_keyword_names(self):
        inference = Config(img_path="image.png", edm_steps=7, text_steps=12, linear_CFG=False)
        self.assertEqual(inference.input_path, "image.png")
        self.assertEqual(inference.steps, 7)
        self.assertEqual(inference.text_steps, 12)
        self.assertFalse(inference.linear_cfg)
        self.assertEqual(inference.edm_steps, inference.steps)
        training = TrainConfig(
            hq_meta_path="hq.jsonl", neg_meta_path="negative.jsonl", num_epochs=3
        )
        self.assertEqual(training.hq_metadata, "hq.jsonl")
        self.assertEqual(training.negative_metadata, "negative.jsonl")
        self.assertEqual(training.epochs, 3)

    def test_training_boolean_options_share_the_contract(self):
        canonical = training_options("--no-linear-cfg", "--tile-vae")
        legacy = training_options("--linear_CFG", "--use_tile_vae")
        self.assertEqual(vars(canonical), vars(legacy))
        self.assertIsNone(legacy.text_steps)
        self.assertEqual((legacy.upscale, legacy.min_size), (2, 256))
        self.assertTrue(legacy.tile_vae)
        self.assertFalse(legacy.linear_cfg)

    def test_removed_half_loading_options_are_rejected_by_inference_and_training(self):
        for parse in (inference_options, training_options):
            for option in (
                "--loading-half-params",
                "--loading_half_params",
                "--no-loading-half-params",
            ):
                with (
                    self.subTest(parser=parse.__name__, option=option),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit) as error:
                        parse(option)
                    self.assertEqual(error.exception.code, 2)

    def test_legacy_python_half_loading_is_rejected_instead_of_silently_ignored(self):
        for config_type in (RuntimeConfig, InferenceConfig, TrainingConfig, Config, TrainConfig):
            with self.subTest(config=config_type.__name__):
                with self.assertRaisesRegex(ValueError, "loading_half_params=True is unsupported"):
                    config_type(loading_half_params=True)
                self.assertNotIn("loading_half_params", asdict(config_type()))
        mutated = InferenceConfig()
        mutated.loading_half_params = True
        with self.assertRaisesRegex(ValueError, "loading_half_params=True is unsupported"):
            validate_runtime(mutated)

    def test_training_rejects_flags_that_do_not_affect_its_validation_recipe(self):
        for arguments in (
            ("--upscale", "3"),
            ("--min-size", "1024"),
            ("--min_size", "1024"),
            ("--text-steps", "12"),
            ("--text_steps", "12"),
            ("--text-prompt", "custom"),
            ("--a_text_prompt", "custom"),
            ("--negative-text-prompt", "custom"),
            ("--n_text_prompt", "custom"),
        ):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    training_options(*arguments)
                self.assertEqual(error.exception.code, 2)

    def test_python_training_config_preserves_fixed_validation_defaults(self):
        default = TrainingConfig()
        self.assertEqual((default.upscale, default.min_size, default.text_steps), (2, 256, None))
        validate_runtime(default)
        for config_type, options in (
            (TrainingConfig, {"upscale": 3}),
            (TrainingConfig, {"min_size": 1024}),
            (TrainingConfig, {"text_steps": 12}),
            (TrainingConfig, {"positive_text_prompt": "custom"}),
            (TrainingConfig, {"negative_text_prompt": "custom"}),
            (TrainConfig, {"a_text_prompt": "custom"}),
            (TrainConfig, {"n_text_prompt": "custom"}),
        ):
            with self.subTest(config=config_type.__name__, options=options):
                with self.assertRaisesRegex(ValueError, "is not configurable for training"):
                    config_type(**options)
        default.min_size = 1024
        with self.assertRaisesRegex(ValueError, "min_size is not configurable for training"):
            validate_runtime(default)

    def test_invalid_legacy_text_step_override_fails_runtime_validation(self):
        with self.assertRaisesRegex(ValueError, "text_steps must be positive"):
            validate_runtime(InferenceConfig(text_steps=0))

    def test_help_avoids_runtime_imports_with_site_packages_disabled(self):
        script = """
import sys
from glyph_sr.cli import main
try:
    result = main(sys.argv[1:])
except SystemExit as error:
    result = error.code
assert result == 0, result
runtime = {'torch', 'numpy', 'PIL', 'transformers', 'diffusers', 'lpips', 'pytorch_lightning', 'llava', 'sgm'}
assert not runtime.intersection(sys.modules), runtime.intersection(sys.modules)
"""
        for arguments in (("--help",), ("infer", "--help"), ("train", "--help")):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [sys.executable, "-S", "-c", script, *arguments],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
                self.assertNotIn("--loading-half-params", result.stdout)
                self.assertNotIn("--loading_half_params", result.stdout)
                if arguments[0] == "train":
                    for option in (
                        "--upscale",
                        "--min-size",
                        "--text-steps",
                        "--text-prompt",
                        "--negative-text-prompt",
                    ):
                        self.assertNotIn(option, result.stdout)


if __name__ == "__main__":
    unittest.main()
