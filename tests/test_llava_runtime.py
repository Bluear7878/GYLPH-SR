"""Check lazy LLaVA imports and explicit model placement without downloading weights."""

import subprocess
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch


def supported_config(**changes):
    values = dict(
        model_type="llava",
        architectures=["LlavaLlamaForCausalLM"],
        mm_vision_tower="openai/clip-vit-large-patch14-336",
        mm_vision_select_layer=-2,
        mm_vision_select_feature="patch",
        mm_projector_type="mlp2x_gelu",
        mm_resampler_type=None,
        image_aspect_ratio="anyres",
        mm_patch_merge_type="spatial_unpad",
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        max_position_embeddings=8192,
    )
    return SimpleNamespace(**{**values, **changes})


class LlavaRuntimeTests(unittest.TestCase):
    def test_package_imports_need_no_optional_dependencies_or_architectures(self):
        script = (
            "import sys; import glyph_sr._vendor.llava; import glyph_sr._vendor.llava.model; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "assert not any(name.startswith('glyph_sr._vendor.llava.model.language_model.') for name in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_selected_llama_class_keeps_its_identity_and_skips_other_architectures(self):
        script = (
            "import sys; from glyph_sr._vendor.llava.model import LlavaLlamaForCausalLM; "
            "assert LlavaLlamaForCausalLM.__module__ == 'glyph_sr._vendor.llava.model.llama'; "
            "assert 'glyph_sr._vendor.llava.model.language_model.llava_qwen' not in sys.modules; "
            "assert 'glyph_sr._vendor.llava.model.language_model.llava_mistral' not in sys.modules; "
            "assert 'glyph_sr._vendor.llava.model.language_model.llava_mixtral' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Failed to import", result.stdout)

    def test_builder_keeps_requested_vision_device_and_automatic_dispatch(self):
        import glyph_sr._vendor.llava.model as registry
        from glyph_sr._vendor.llava.model import builder

        for device in ("cpu", "cuda:7", "auto"):
            with self.subTest(device=device):
                config = supported_config(max_position_embeddings=4096)
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
                module = ModuleType("glyph_sr._vendor.llava.model.llama")
                module.LlavaConfig = SimpleNamespace(from_pretrained=Mock(return_value=config))
                with (
                    patch.dict(
                        registry.__dict__,
                        {
                            "LlavaLlamaForCausalLM": architecture,
                            "LlavaConfig": SimpleNamespace(
                                from_pretrained=Mock(return_value=config)
                            ),
                        },
                    ),
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
        from glyph_sr._vendor.llava.model import builder

        with patch.object(builder, "AutoTokenizer") as tokenizer:
            with self.assertRaisesRegex(ValueError, "Unsupported torch_dtype"):
                builder.load_pretrained_model(
                    "unused", None, "llava_llama_3", torch_dtype="invalid"
                )
            tokenizer.from_pretrained.assert_not_called()

    def test_split_device_maps_are_left_to_dispatch(self):
        from glyph_sr._vendor.llava.model.builder import _vision_tower_device

        self.assertEqual(_vision_tower_device({"": "cpu"}), "cpu")
        self.assertEqual(_vision_tower_device(3), "cuda:3")
        self.assertIsNone(_vision_tower_device({"model.layers.0": 0, "model.vision_tower": 1}))

    def test_released_conversation_keeps_the_exact_chat_template_inputs(self):
        import copy

        from glyph_sr._vendor.llava.conversation import conv_templates

        self.assertEqual(set(conv_templates), {"llava_llama_3"})
        conversation = copy.deepcopy(conv_templates["llava_llama_3"])
        conversation.tokenizer = Mock()
        conversation.tokenizer.apply_chat_template.return_value = "formatted"
        conversation.append_message(conversation.roles[0], "<image>\nRead the sign.")
        conversation.append_message(conversation.roles[1], None)
        self.assertEqual(conversation.get_prompt(), "formatted")
        conversation.tokenizer.apply_chat_template.assert_called_once_with(
            [
                {
                    "role": "system",
                    "content": "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
                },
                {"role": "user", "content": "<image>\nRead the sign."},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertEqual(conversation.stop_token_ids, [128009])

    def test_unsupported_configurations_fail_before_loading_weights_or_tokenizer(self):
        from glyph_sr._vendor.llava.model import builder

        cases = (
            {"architectures": ["LlavaQwenForCausalLM"]},
            {"model_type": "qwen2"},
            {"mm_vision_tower": "google/siglip-so400m-patch14-384"},
            {"mm_projector_type": "pooler"},
            {"mm_resampler_type": "qformer"},
            {"s2": True},
            {"image_aspect_ratio": "pad"},
        )
        for changes in cases:
            with (
                self.subTest(changes=changes),
                patch.object(builder, "AutoTokenizer") as tokenizer,
                patch.object(builder, "_selected_model_class") as architecture,
            ):
                with self.assertRaises(ValueError):
                    builder.load_pretrained_model(
                        "unused",
                        None,
                        "llava_llama_3",
                        customized_config=supported_config(**changes),
                    )
                tokenizer.from_pretrained.assert_not_called()
                architecture.return_value.from_pretrained.assert_not_called()
        with self.assertRaisesRegex(ValueError, "architecture"):
            builder.load_pretrained_model("unused", None, "llava_qwen")
        with self.assertRaisesRegex(ValueError, "complete"):
            builder.load_pretrained_model("unused", "base", "llava_llama_3")

    def test_released_vision_projection_keys_and_pixel_feature_math_are_preserved(self):
        from glyph_sr._vendor.llava.model.vision import (
            CLIPVisionTower,
            build_vision_projector,
            build_vision_resampler,
        )

        config = supported_config(mm_hidden_size=3, hidden_size=4)
        projector = build_vision_projector(config)
        self.assertEqual(list(projector.state_dict()), ["0.weight", "0.bias", "2.weight", "2.bias"])
        state = {
            key: torch.arange(value.numel(), dtype=value.dtype).reshape(value.shape) / 10
            for key, value in projector.state_dict().items()
        }
        projector.load_state_dict(state, strict=True)
        values = torch.tensor([[0.5, -0.2, 0.7]])
        expected = torch.nn.functional.linear(
            torch.nn.functional.gelu(
                torch.nn.functional.linear(values, state["0.weight"], state["0.bias"])
            ),
            state["2.weight"],
            state["2.bias"],
        )
        torch.testing.assert_close(projector(values), expected, rtol=0, atol=0)
        identity = build_vision_resampler(config)
        self.assertEqual(dict(identity.state_dict()), {})
        self.assertIs(identity(values), values)
        tower = CLIPVisionTower.__new__(CLIPVisionTower)
        torch.nn.Module.__init__(tower)
        tower.select_layer, tower.select_feature = -2, "patch"
        states = [torch.arange(20).reshape(1, 5, 4) + index * 100 for index in range(3)]
        self.assertTrue(
            torch.equal(
                tower.feature_select(SimpleNamespace(hidden_states=states)), states[-2][:, 1:]
            )
        )


if __name__ == "__main__":
    unittest.main()
