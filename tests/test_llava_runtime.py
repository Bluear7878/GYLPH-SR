"""Check lazy LLaVA imports and explicit model placement without downloading weights."""

import subprocess
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch


class LlavaRuntimeTests(unittest.TestCase):
    def test_package_imports_need_no_optional_dependencies_or_architectures(self):
        script = (
            "import sys; import llava; import llava.model; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "assert not any(name.startswith('llava.model.language_model.') for name in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_selected_llama_class_keeps_its_identity_and_skips_other_architectures(self):
        script = (
            "import sys; from llava.model import LlavaLlamaForCausalLM; "
            "assert LlavaLlamaForCausalLM.__module__ == 'llava.model.language_model.llava_llama'; "
            "assert 'llava.model.language_model.llava_qwen' not in sys.modules; "
            "assert 'llava.model.language_model.llava_mistral' not in sys.modules; "
            "assert 'llava.model.language_model.llava_mixtral' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Failed to import", result.stdout)

    def test_builder_keeps_requested_vision_device_and_automatic_dispatch(self):
        import llava.model as registry
        from llava.model import builder

        for device in ("cpu", "cuda:7", "auto"):
            with self.subTest(device=device):
                config = SimpleNamespace(
                    mm_use_im_start_end=False,
                    mm_use_im_patch_token=False,
                    max_position_embeddings=4096,
                )
                tokenizer = Mock()
                tokenizer.__len__ = Mock(return_value=8)
                tower = SimpleNamespace(
                    is_loaded=False,
                    load_model=Mock(),
                    to=Mock(),
                    image_processor=object(),
                )
                model = SimpleNamespace(
                    config=config, resize_token_embeddings=Mock(), get_vision_tower=lambda: tower
                )
                architecture = SimpleNamespace(from_pretrained=Mock(return_value=model))
                module = ModuleType("llava.model.language_model.llava_llama")
                module.LlavaConfig = SimpleNamespace(from_pretrained=Mock(return_value=config))
                with (
                    patch.dict(registry.__dict__, {"LlavaLlamaForCausalLM": architecture}),
                    patch.dict(sys.modules, {module.__name__: module}),
                    patch.object(
                        builder,
                        "AutoTokenizer",
                        SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
                    ),
                    patch.object(builder, "rank0_print"),
                ):
                    output = builder.load_pretrained_model(
                        "local/llava",
                        None,
                        "llava_llama_3",
                        device_map=device,
                        multimodal=False,
                        attn_implementation=None,
                    )
                self.assertEqual(output[3], 4096)
                self.assertIs(output[2], tower.image_processor)
                tower.load_model.assert_called_once_with(device_map=device)
                if device == "auto":
                    tower.to.assert_not_called()
                else:
                    tower.to.assert_called_once_with(device=device, dtype=torch.float16)
                kwargs = architecture.from_pretrained.call_args.kwargs
                self.assertEqual(kwargs["device_map"], device)
                self.assertNotIn("multimodal", kwargs)

    def test_invalid_dtype_raises_before_model_loading(self):
        from llava.model import builder

        with patch.object(builder, "AutoTokenizer") as tokenizer:
            with self.assertRaisesRegex(ValueError, "Unsupported torch_dtype"):
                builder.load_pretrained_model(
                    "unused", None, "llava_llama_3", torch_dtype="invalid"
                )
            tokenizer.from_pretrained.assert_not_called()

    def test_split_device_maps_are_left_to_dispatch(self):
        from llava.model.builder import _vision_tower_device

        self.assertEqual(_vision_tower_device({"": "cpu"}), "cpu")
        self.assertEqual(_vision_tower_device(3), "cuda:3")
        self.assertIsNone(_vision_tower_device({"model.layers.0": 0, "model.vision_tower": 1}))


if __name__ == "__main__":
    unittest.main()
