import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytorch_lightning as pl
import torch
from torch import nn

from glyph_sr.config import RuntimeConfig
from glyph_sr.models.controlnet import TextControlNet, _residual_mean, _tree_to, load_controlnet
from glyph_sr.models.diffusion import DiffusionRestorer
from glyph_sr.models.layers import ZeroSFT
from utils.tilevae import GroupNormParam, VAEHook


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_blocks = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.out = nn.Linear(2, 2, bias=False)
        self.project_modules = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.model_channels = 2
        self.num_classes = None
        self.original_projections = None
        self.fail_text = False
        for layer in (self.input_blocks[0], self.out, self.project_modules[0]):
            nn.init.eye_(layer.weight)

    def forward(self, x):
        if self.fail_text and self.project_modules is not self.original_projections:
            raise RuntimeError("text forward failed")
        return self.out(self.project_modules[0](self.input_blocks[0](x)))


@dataclass
class ResidualOutput:
    down_block_residuals: list
    mid_block_residual: torch.Tensor


class RecordingSampler:
    def __init__(self):
        self.steps = []

    def init_loop(self, noise, condition, uc, num_steps):
        self.noise = noise.clone()
        return (
            noise,
            torch.ones(len(noise)),
            torch.ones(num_steps + 1),
            num_steps + 1,
            condition,
            uc,
        )

    def step(self, latent, index, s_in, sigmas, denoiser, condition, unconditional, **kwargs):
        self.steps.append((index, condition, unconditional, kwargs, latent.clone()))
        return latent + 1


def sampling_model():
    # Construct only the method host; no VGG/SDXL weights or downloads are needed.
    model = DiffusionRestorer.__new__(DiffusionRestorer)
    nn.Module.__init__(model)
    model.p_p, model.n_p = "positive", "negative"
    model.sampler_config = SimpleNamespace(
        params=SimpleNamespace(guider_config=SimpleNamespace(params=SimpleNamespace()))
    )
    model.encode_first_stage_with_denoise = Mock(side_effect=lambda value, **kwargs: value + 10)
    model.decode_first_stage = Mock(side_effect=lambda value: value * 2)
    model.encode_first_stage = Mock(side_effect=lambda value: value + 1)
    model.prepare_condition = Mock(
        side_effect=lambda latent, prompts, positive, negative, count: (
            prompts,
            [negative] * count,
        )
    )
    return model


class ModelTests(unittest.TestCase):
    def test_loader_applies_precision_and_optional_runtime_settings(self):
        model = nn.Module()
        model.model = nn.Module()
        model.model.diffusion_model = TinyUNet()
        model.init_tile_vae = Mock()
        captured = {}

        def instantiate(config):
            if config["target"] == "tests.TinyRestorer":
                captured.update(config.params)
                return model
            return nn.Identity()

        args = RuntimeConfig(
            ae_dtype="fp32",
            diff_dtype="bf16",
            tile_vae=True,
            encoder_tile_size=128,
            decoder_tile_size=32,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.yaml"
            path.write_text(
                "model:\n  target: tests.TinyRestorer\n  params: {}\n"
                "SDXL_CKPT: null\nSUPIR_CKPT: null\n"
            )
            with patch("sgm.util.instantiate_from_config", side_effect=instantiate):
                result, trainable = load_controlnet(path, args, device="cpu")
        self.assertIs(result, model)
        self.assertEqual(captured, {"ae_dtype": "fp32", "diffusion_dtype": "bf16"})
        self.assertTrue(all(value.dtype == torch.float32 for value in model.parameters()))
        model.init_tile_vae.assert_called_once_with(encoder_tile_size=128, decoder_tile_size=32)
        self.assertEqual(
            list(trainable), list(model.model.diffusion_model.project_modules.parameters())
        )

    def test_loader_rejects_legacy_half_loading_before_reading_configuration(self):
        with patch("omegaconf.OmegaConf.load") as read_config:
            with self.assertRaisesRegex(ValueError, "loading_half_params=True is unsupported"):
                load_controlnet(
                    "missing.yaml", SimpleNamespace(loading_half_params=True), device="cpu"
                )
        read_config.assert_not_called()

    def test_tiled_vae_uses_the_input_device_and_restores_the_module_device(self):
        original = nn.Parameter(torch.zeros(1))
        net = SimpleNamespace(
            parameters=lambda: iter([original]),
            to=Mock(),
            original_forward=Mock(return_value="output"),
        )
        hook = VAEHook(net, 64, False, False, False, False, to_gpu=True)
        inputs = SimpleNamespace(shape=(1, 3, 4, 4), device=torch.device("cuda:2"))
        self.assertEqual(hook(inputs), "output")
        self.assertEqual(
            net.to.call_args_list, [call(torch.device("cuda:2")), call(torch.device("cpu"))]
        )

    def test_tiled_group_statistics_stay_on_their_tensor_device(self):
        statistics = GroupNormParam()
        layer = nn.GroupNorm(32, 32)
        statistics.add_tile(torch.randn(1, 32, 3, 3), layer)
        statistics.add_tile(torch.randn(1, 32, 3, 2), layer)
        normalized = statistics.summary()(torch.randn(1, 32, 3, 3))
        self.assertEqual(normalized.device.type, "cpu")
        self.assertTrue(torch.isfinite(normalized).all())

    def test_constructor_retains_encoder_copy_and_metric_checkpoint_keys(self):
        def initialize_engine(model, *args, **kwargs):
            pl.LightningModule.__init__(model)
            model.model = nn.Module()
            model.model.load_control_model = lambda control: setattr(
                model.model, "control_model", control
            )
            model.first_stage_model = nn.Module()
            model.first_stage_model.encoder = nn.Linear(2, 2)

        with (
            patch("glyph_sr.models.diffusion.DiffusionEngine.__init__", initialize_engine),
            patch(
                "glyph_sr.models.diffusion.instantiate_from_config",
                side_effect=lambda config: nn.Linear(2, 2),
            ),
            patch(
                "glyph_sr.models.diffusion.lpips.LPIPS",
                side_effect=lambda **kwargs: nn.Linear(2, 2),
            ),
        ):
            model = DiffusionRestorer(
                {}, sampler_config={}, ae_dtype="bf16", diffusion_dtype="fp16"
            )
        self.assertEqual(
            set(model.state_dict()),
            {
                "model.control_model.weight",
                "model.control_model.bias",
                "first_stage_model.encoder.weight",
                "first_stage_model.encoder.bias",
                "first_stage_model.denoise_encoder.weight",
                "first_stage_model.denoise_encoder.bias",
                "_lpips.weight",
                "_lpips.bias",
            },
        )
        self.assertIsNot(model.first_stage_model.denoise_encoder, model.first_stage_model.encoder)
        torch.testing.assert_close(
            model.first_stage_model.denoise_encoder.weight,
            model.first_stage_model.encoder.weight,
        )
        self.assertEqual(model.ae_dtype, torch.bfloat16)
        self.assertEqual(model.model.dtype, torch.float16)

    def test_wrapper_preserves_checkpoint_keys_freezing_and_gradient_flow(self):
        model = TextControlNet(TinyUNet())
        self.assertEqual(
            set(model.state_dict()),
            {
                "base.input_blocks.0.weight",
                "base.out.weight",
                "base.project_modules.0.weight",
                "input_blocks.0.weight",
                "out.weight",
                "project_modules.0.weight",
            },
        )
        self.assertIs(model.input_blocks, model.base.input_blocks)
        self.assertIs(model.out, model.base.out)
        self.assertEqual(model.model_channels, 2)
        self.assertTrue(all(not value.requires_grad for value in model.base.parameters()))
        self.assertTrue(all(value.requires_grad for value in model.project_modules.parameters()))
        with torch.no_grad():
            model.project_modules[0].weight.mul_(3)
        inputs = torch.tensor([[2.0, 4.0]])
        output = model(inputs)
        torch.testing.assert_close(output, inputs * 2)
        output.sum().backward()
        torch.testing.assert_close(model.project_modules[0].weight.grad, inputs.repeat(2, 1) * 0.5)
        self.assertTrue(all(value.grad is None for value in model.base.parameters()))
        self.assertIs(model.train(), model)
        self.assertFalse(model.base.training)
        self.assertTrue(model.project_modules[0].training)
        self.assertIs(model.eval(), model)
        self.assertFalse(model.project_modules[0].training)

    def test_text_forward_failure_restores_the_base_projections(self):
        model = TextControlNet(TinyUNet())
        original = model.base.project_modules
        # Avoid registering the saved module as an additional checkpoint child.
        object.__setattr__(model.base, "original_projections", original)
        model.base.fail_text = True
        with self.assertRaisesRegex(RuntimeError, "text forward failed"):
            model(torch.ones(1, 2))
        self.assertIs(model.base.project_modules, original)

    def test_nested_residual_weighting_applies_beta_at_every_level(self):
        base = ResidualOutput([torch.tensor([2.0]), (torch.tensor([4.0]),)], torch.tensor([6.0]))
        text = ResidualOutput([torch.tensor([10.0]), (torch.tensor([12.0]),)], torch.tensor([14.0]))
        output = _residual_mean(base, text, beta=0.25)
        torch.testing.assert_close(output.down_block_residuals[0], torch.tensor([4.0]))
        torch.testing.assert_close(output.down_block_residuals[1][0], torch.tensor([6.0]))
        torch.testing.assert_close(output.mid_block_residual, torch.tensor([8.0]))
        moved = _tree_to({"residual": output, "unchanged": "text"}, device="cpu")
        self.assertIsInstance(moved["residual"], ResidualOutput)
        self.assertEqual(moved["unchanged"], "text")
        with self.assertRaisesRegex(ValueError, "different lengths"):
            _residual_mean([torch.ones(1)], [])

    def test_default_average_retains_released_tensor_operation_order(self):
        base = torch.tensor([1.0000001, 100000.0])
        text = torch.tensor([1.0000002, -99999.0])
        self.assertTrue(torch.equal(_residual_mean(base, text), (base + text) * 0.5))

    def test_projection_scale_zero_returns_the_uncontrolled_features(self):
        projection = ZeroSFT(32, 32)
        features, control = torch.randn(1, 32, 3, 3), torch.randn(1, 32, 3, 3)
        self.assertTrue(torch.equal(projection(control, features, control_scale=0), features))

    def test_sampling_alternates_guidance_and_preserves_latent_centers(self):
        model, sampler = sampling_model(), RecordingSampler()
        inputs = torch.zeros(1, 3, 2, 2)
        with (
            patch(
                "glyph_sr.models.diffusion.instantiate_from_config",
                return_value=sampler,
            ),
            patch("glyph_sr.models.diffusion.seed_everything") as seed,
        ):
            output = model.ping_pong_sampling(
                inputs,
                ["scene"],
                ["text"],
                image_steps=3,
                text_steps=2,
                seed=123,
                control_scale=0.7,
                use_linear_CFG=True,
                cfg_scale=4,
                cfg_scale_start=1,
            )
        seed.assert_called_once_with(123)
        self.assertEqual(
            [step[1] for step in sampler.steps],
            [["scene"], ["text"], ["scene"], ["text"], ["scene"]],
        )
        torch.testing.assert_close(sampler.steps[0][3]["x_center"], torch.full_like(inputs, 21))
        for step in sampler.steps[1:]:
            torch.testing.assert_close(step[3]["x_center"], step[4])
            self.assertEqual(step[3]["control_scale"], 0.7)
        torch.testing.assert_close(output, (sampler.noise + 5) * 2)
        self.assertEqual(model.sampler_config.params.guider_config.params.scale_min, 4)
        self.assertEqual(model.sampler_config.params.guider_config.params.scale, 1)

    def test_multiple_samples_repeat_both_guidance_streams(self):
        model, sampler = sampling_model(), RecordingSampler()
        with (
            patch(
                "glyph_sr.models.diffusion.instantiate_from_config",
                return_value=sampler,
            ),
            patch("glyph_sr.models.diffusion.seed_everything"),
        ):
            output = model.ping_pong_sampling(
                torch.zeros(1, 3, 2, 2),
                ["scene"],
                ["text"],
                image_steps=1,
                text_steps=1,
                num_samples=3,
                seed=123,
            )
        self.assertEqual(output.shape[0], 3)
        self.assertEqual(sampler.steps[0][1], ["scene"] * 3)
        self.assertEqual(sampler.steps[1][1], ["text"] * 3)


if __name__ == "__main__":
    unittest.main()
