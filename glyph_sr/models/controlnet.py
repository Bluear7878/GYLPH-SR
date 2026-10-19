"""Frozen restoration branch and trainable text-control projections."""

import os
from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import nn

from glyph_sr.checkpoints import get_state_dict, load_state_dict, load_ts_checkpoint


def _is_residual_output(obj):
    return hasattr(obj, "down_block_residuals") and hasattr(obj, "mid_block_residual")


def _tree_to(obj, *, device=None):
    """Move tensors in a nested residual structure to one device."""
    if torch.is_tensor(obj):
        return obj.to(device=device)
    if _is_residual_output(obj):
        return type(obj)(
            down_block_residuals=[_tree_to(t, device=device) for t in obj.down_block_residuals],
            mid_block_residual=_tree_to(obj.mid_block_residual, device=device),
        )
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tree_to(item, device=device) for item in obj)
    if isinstance(obj, dict):
        return {key: _tree_to(value, device=device) for key, value in obj.items()}
    return obj


@contextmanager
def _swap(obj, attr, temporary):
    """Restore the original module even when the temporary forward fails."""
    original = getattr(obj, attr)
    setattr(obj, attr, temporary)
    try:
        yield
    finally:
        setattr(obj, attr, original)


def _residual_mean(base, text, beta=0.5):
    """Interpolate matching residual structures with the released equal average."""
    if torch.is_tensor(base):
        # Preserve the exact operation order of the released beta=0.5 model.
        if beta == 0.5:
            return (base + text) * beta
        return (1 - beta) * base + beta * text
    if _is_residual_output(base):
        return type(base)(
            down_block_residuals=_residual_mean(
                list(base.down_block_residuals), list(text.down_block_residuals), beta
            ),
            mid_block_residual=(1 - beta) * base.mid_block_residual
            + beta * text.mid_block_residual,
        )
    if isinstance(base, (list, tuple)):
        if len(base) != len(text):
            raise ValueError("residual structures have different lengths")
        return type(base)(_residual_mean(a, b, beta) for a, b in zip(base, text))
    raise TypeError(type(base))


class TextControlNet(nn.Module):
    """Share a frozen UNet while learning a separate copy of its projections.

    Attribute names and their duplicate module registrations are retained because
    released Lightning checkpoints include those state-dictionary paths.
    """

    def __init__(self, base_unet: nn.Module, beta=0.5):
        super().__init__()
        self.base = base_unet.eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.beta = beta

        for name in (
            "input_blocks",
            "middle_block",
            "output_blocks",
            "time_embed",
            "label_emb",
            "out",
            "model_channels",
            "num_classes",
        ):
            if hasattr(self.base, name):
                setattr(self, name, getattr(self.base, name))

        self.project_modules = nn.ModuleList(deepcopy(self.base.project_modules))
        for module in self.project_modules:
            module.requires_grad_(True)
            module.eval()

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            base_output = self.base(*args, **kwargs)
        device = next(self.base.parameters()).device
        text_args = _tree_to(args, device=device)
        text_kwargs = _tree_to(kwargs, device=device)
        with _swap(self.base, "project_modules", self.project_modules):
            text_output = self.base(*text_args, **text_kwargs)
        return _residual_mean(base_output, text_output, self.beta)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        for module in self.project_modules:
            module.train(mode)
        return self


def load_controlnet(
    cfg_path,
    args,
    loss_fn_config=None,
    device="cuda",
    sign=None,
    warm_up_steps=5000,
    total_steps=70000,
    beta=0.5,
    pretrained=None,
):
    """Build the released frozen restorer and trainable text projections."""
    from glyph_sr.config import _reject_legacy_half_loading

    _reject_legacy_half_loading(getattr(args, "loading_half_params", False))

    from omegaconf import OmegaConf

    from sgm.util import instantiate_from_config

    cfg = OmegaConf.load(cfg_path)
    # Resolve and check local paths before allocating the diffusion/VLM stack.
    checkpoint_keys = ["SDXL_CKPT", "SUPIR_CKPT"]
    if sign is not None:
        if sign != "Q":
            raise ValueError("The released model supports only the Q SUPIR checkpoint")
        checkpoint_keys.append("SUPIR_CKPT_Q")
    for key in checkpoint_keys:
        path = cfg.get(key)
        if path is not None and not os.path.isfile(path):
            raise FileNotFoundError(f"{key} checkpoint not found: {path}; see docs/checkpoints.md")

    if pretrained is not None and not os.path.isfile(pretrained):
        raise FileNotFoundError(f"TS checkpoint not found: {pretrained}")

    for option, key in (("ae_dtype", "ae_dtype"), ("diff_dtype", "diffusion_dtype")):
        precision = getattr(args, option, None)
        if precision is not None:
            cfg.model.params[key] = precision

    model = instantiate_from_config(cfg.model).cpu()

    if cfg.SDXL_CKPT is not None:
        model.load_state_dict(load_state_dict(cfg.SDXL_CKPT), strict=False)
    if cfg.SUPIR_CKPT is not None:
        model.load_state_dict(load_state_dict(cfg.SUPIR_CKPT), strict=False)
    if sign is not None:
        model.load_state_dict(load_state_dict(cfg.SUPIR_CKPT_Q), strict=False)

    model.eval().to(device)

    base_unet = model.model.diffusion_model
    model.model.diffusion_model = TextControlNet(base_unet, beta=beta).to(device)

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.model.diffusion_model.project_modules.parameters():
        p.requires_grad_(True)

    trainable = model.model.diffusion_model.project_modules.parameters()

    if loss_fn_config is None:
        loss_fn_config = {
            "target": "sgm.modules.diffusionmodules.loss.StandardDiffusionLoss",
            "params": {
                "sigma_sampler_config": {
                    "target": "sgm.modules.diffusionmodules.sigma_sampling.DiscreteSampling",
                    "params": {
                        "discretization_config": {
                            "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
                        },
                        "num_idx": 1000,
                        "do_append_zero": False,
                        "flip": True,
                    },
                },
                "type": "l2",
                "batch2model_keys": ["control_scale"],
            },
        }
    optimizer_config = {
        "target": "torch.optim.AdamW",
        "params": {"weight_decay": 1e-4},
    }
    scheduler_config = {
        "target": "sgm.lr_scheduler.LambdaWarmUpCosineScheduler",
        "params": {
            "warm_up_steps": warm_up_steps,
            "lr_min": 0.1,
            "lr_max": 1.0,
            "lr_start": 0.01,
            "max_decay_steps": total_steps,
        },
    }
    model.learning_rate = 1e-6
    model.input_key = "input_tensor"

    if loss_fn_config is not None:
        model.loss_fn = instantiate_from_config(loss_fn_config)
    if optimizer_config is not None:
        model.optimizer_config = optimizer_config
    if scheduler_config is not None:
        model.scheduler_config = scheduler_config

    model.upscale = 2
    model.min_size = 256
    model.val_sample_kwargs = {
        "num_steps": args.edm_steps,
        "restoration_scale": args.s_stage1,
        "s_churn": args.s_churn,
        "s_noise": args.s_noise,
        "cfg_scale": args.s_cfg,
        "control_scale": args.s_stage2,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "p_p": args.a_prompt,
        "n_p": args.n_prompt,
        "color_fix_type": args.color_fix_type,
        "use_linear_CFG": args.linear_CFG,
        "use_linear_control_scale": args.linear_s_stage2,
        "cfg_scale_start": args.spt_linear_CFG,
        "control_scale_start": args.spt_linear_s_stage2,
    }

    if pretrained is not None:
        model = load_ts_checkpoint(model, pretrained, device=device, strict=False)
        print(f"Loaded {pretrained} to {model.__class__.__name__}")

    if getattr(args, "use_tile_vae", False):
        model.init_tile_vae(
            encoder_tile_size=getattr(args, "encoder_tile_size", 512),
            decoder_tile_size=getattr(args, "decoder_tile_size", 64),
        )

    return model, trainable


ProjectTSControlNet = TextControlNet
load_TS_ControlNet = load_controlnet
load_TS_pretrained_model = load_ts_checkpoint

__all__ = [
    "TextControlNet",
    "ProjectTSControlNet",
    "load_controlnet",
    "load_TS_ControlNet",
    "get_state_dict",
    "load_state_dict",
    "load_TS_pretrained_model",
]
