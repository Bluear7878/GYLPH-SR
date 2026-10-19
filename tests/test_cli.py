"""Validate command options and startup without constructing models."""

import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path

from glyph_sr.config import InferenceConfig, TrainingConfig, parse_config, validate_runtime
from glyph_sr.inference import validate_inputs

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
    def test_boolean_flags_follow_their_names(self):
        defaults = inference_options()
        self.assertTrue(defaults.linear_cfg)
        self.assertFalse(defaults.linear_control_scale)
        self.assertFalse(defaults.tile_vae)
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
        training = training_options("--no-linear-cfg", "--tile-vae")
        self.assertTrue(training.tile_vae)
        self.assertFalse(training.linear_cfg)

    def test_sampling_values_have_the_expected_types(self):
        options = inference_options(
            "--steps", "7", "--text-steps", "12", "--cfg-scale", "6", "--output-dir", "results"
        )
        self.assertEqual((options.steps, options.text_steps, options.cfg_scale), (7, 12, 6.0))
        self.assertEqual(options.output_dir, "results")
        self.assertIsNone(inference_options().text_steps)
        with self.assertRaisesRegex(ValueError, "text_steps must be positive"):
            validate_runtime(InferenceConfig(text_steps=0))

    def test_required_inputs_fail_before_loading_model_dependencies(self):
        for config_type in (InferenceConfig, TrainingConfig):
            with (
                self.subTest(config=config_type.__name__),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as error:
                    parse_config(config_type, [])
                self.assertEqual(error.exception.code, 2)
        with self.assertRaisesRegex(FileNotFoundError, "input_path"):
            validate_inputs(InferenceConfig(input_path="missing-input-image.png"))

    def test_training_rejects_options_that_do_not_affect_its_validation_recipe(self):
        for arguments in (
            ("--upscale", "3"),
            ("--min-size", "1024"),
            ("--text-steps", "12"),
            ("--text-prompt", "custom"),
            ("--negative-text-prompt", "custom"),
            ("--loading-half-params",),
        ):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    training_options(*arguments)
                self.assertEqual(error.exception.code, 2)

    def test_python_training_config_preserves_fixed_validation_defaults(self):
        config = TrainingConfig()
        self.assertEqual((config.upscale, config.min_size, config.text_steps), (2, 256, None))
        validate_runtime(config)
        for options in (
            {"upscale": 3},
            {"min_size": 1024},
            {"text_steps": 12},
            {"positive_text_prompt": "custom"},
            {"negative_text_prompt": "custom"},
        ):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "is not configurable for training"):
                    TrainingConfig(**options)
        config.min_size = 1024
        with self.assertRaisesRegex(ValueError, "min_size is not configurable for training"):
            validate_runtime(config)

    def test_all_command_help_avoids_runtime_imports(self):
        script = """
import sys
from glyph_sr.cli import main
try:
    result = main(sys.argv[1:])
except SystemExit as error:
    result = error.code
assert result == 0, result
runtime = {'torch', 'numpy', 'PIL', 'transformers', 'diffusers', 'lpips', 'pytorch_lightning'}
assert not runtime.intersection(sys.modules), runtime.intersection(sys.modules)
assert not any(name.startswith('glyph_sr._vendor') for name in sys.modules)
"""
        commands = [(), ("infer",), ("train",), ("score",), ("verify",), ("data",)]
        commands += [("data", name) for name in ("prompts", "synthesize", "negatives")]
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, "-S", "-c", script, *command, "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
