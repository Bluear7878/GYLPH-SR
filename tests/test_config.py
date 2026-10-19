"""Protect portable resources and the two distinct SDXL checkpoint defaults."""

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from glyph_sr.paths import resource_path


class PackagedConfigTests(unittest.TestCase):
    def test_restoration_and_negative_generation_keep_distinct_sdxl_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            restoration = OmegaConf.load(resource_path("sdxl.yaml"))
            negatives = OmegaConf.load(resource_path("supir.yaml"))
            self.assertEqual(
                restoration.SDXL_CKPT,
                "./checkpoints/juggernautXL_v8Rundiffusion.safetensors",
            )
            self.assertEqual(negatives.SDXL_CKPT, "./checkpoints/sd_xl_base_1.0_0.9vae.safetensors")
            for config in (restoration, negatives):
                self.assertEqual(config.SUPIR_CKPT_F, "./checkpoints/SUPIR-v0F.ckpt")
                self.assertEqual(config.SUPIR_CKPT_Q, "./checkpoints/SUPIR-v0Q.ckpt")

    def test_checkpoint_environment_overrides_are_independent(self):
        overrides = {
            "GLYPH_SDXL_CKPT": "/models/restoration.safetensors",
            "GLYPH_SUPIR_SDXL_CKPT": "/models/negative-generation.safetensors",
            "GLYPH_SUPIR_F_CKPT": "/models/f.ckpt",
            "GLYPH_SUPIR_Q_CKPT": "/models/q.ckpt",
        }
        with patch.dict(os.environ, overrides, clear=True):
            restoration = OmegaConf.load(resource_path("sdxl.yaml"))
            negatives = OmegaConf.load(resource_path("supir.yaml"))
            self.assertEqual(restoration.SDXL_CKPT, overrides["GLYPH_SDXL_CKPT"])
            self.assertEqual(negatives.SDXL_CKPT, overrides["GLYPH_SUPIR_SDXL_CKPT"])
            for config in (restoration, negatives):
                self.assertEqual(config.SUPIR_CKPT_F, overrides["GLYPH_SUPIR_F_CKPT"])
                self.assertEqual(config.SUPIR_CKPT_Q, overrides["GLYPH_SUPIR_Q_CKPT"])

    def test_packaged_model_targets_are_importable(self):
        def check(value):
            if isinstance(value, dict):
                if "target" in value:
                    module, name = value["target"].rsplit(".", 1)
                    self.assertTrue(callable(getattr(importlib.import_module(module), name)))
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)

        for filename in ("sdxl.yaml", "supir.yaml"):
            with self.subTest(config=filename):
                check(OmegaConf.to_container(OmegaConf.load(resource_path(filename))))

    def test_bundled_resources_do_not_depend_on_working_directory(self):
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                self.assertEqual(resource_path("sdxl.yaml").parent.name, "configs")
                self.assertTrue(resource_path("prompts.yaml").is_file())
                with self.assertRaises(FileNotFoundError):
                    resource_path("../../pyproject.toml")
        finally:
            os.chdir(previous)
