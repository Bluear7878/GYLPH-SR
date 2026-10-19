# Modified for GLYPH-SR: retain the released CLIP/MLP2/identity vision path only.
"""Vision components required by the released Llama3-LLaVA-NeXT checkpoint."""

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from glyph_sr._vendor.llava.utils import rank0_print

CLIP_MODEL = "openai/clip-vit-large-patch14-336"


def validate_model_config(config):
    """Reject other checkpoint families before allocating their language weights."""
    expected = {
        "architectures": ["LlavaLlamaForCausalLM"],
        "mm_projector_type": "mlp2x_gelu",
        "mm_resampler_type": None,
        "mm_vision_select_layer": -2,
        "mm_vision_select_feature": "patch",
        "image_aspect_ratio": "anyres",
        "mm_patch_merge_type": "spatial_unpad",
    }
    for name, value in expected.items():
        actual = getattr(config, name, None)
        if actual != value:
            raise ValueError(
                f"The released Llama3-LLaVA-NeXT checkpoint requires {name}={value!r}; got {actual!r}"
            )
    if getattr(config, "model_type", None) not in {"llava", "llava_llama"}:
        raise ValueError("Only the released Llama3-LLaVA-NeXT model type is supported")
    _validate_vision_config(config)


def _validate_vision_config(config):
    identifier = getattr(config, "mm_vision_tower", None)
    if (
        not isinstance(identifier, str)
        or not identifier
        or (identifier != CLIP_MODEL and not Path(identifier).is_dir())
    ):
        raise ValueError(f"Only {CLIP_MODEL} or a local copy of that CLIP model is supported")
    if identifier != CLIP_MODEL:
        with (Path(identifier) / "config.json").open(encoding="utf-8") as stream:
            local = json.load(stream)
        local = local.get("vision_config", local)
        if any(
            local.get(name) != value
            for name, value in {
                "hidden_size": 1024,
                "image_size": 336,
                "patch_size": 14,
            }.items()
        ):
            raise ValueError(
                "The local vision checkpoint must use the released CLIP ViT-L/14-336 configuration"
            )
    if getattr(config, "s2", False):
        raise ValueError("The released checkpoint does not use an S2 vision tower")
    if (
        getattr(config, "mm_vision_select_layer", None) != -2
        or getattr(config, "mm_vision_select_feature", "patch") != "patch"
    ):
        raise ValueError("The released CLIP tower requires layer -2 patch features")


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(
                f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True."
            )
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(
                f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`."
            )
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print(
                "{} is already loaded, `load_model` called again, skipping.".format(
                    self.vision_tower_name
                )
            )
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name, device_map=device_map
        )
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        select_feature_type = self.select_feature

        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
            select_every_k_layer = len(image_forward_outs.hidden_states) // 4
            image_features = torch.cat(
                [
                    image_forward_outs.hidden_states[i]
                    for i in range(
                        select_every_k_layer + self.select_layer,
                        len(image_forward_outs.hidden_states),
                        select_every_k_layer,
                    )
                ],
                dim=-1,
            )
            select_feature_type = select_feature_type.replace("slicefour_", "")
        elif self.select_feature in [
            "slice_m25811_f6_patch",
            "slice_m25811_f6_cls_patch",
        ]:
            select_layers = [-2, -5, -8, -11, 6]
            image_features = torch.cat(
                [image_forward_outs.hidden_states[i] for i in select_layers], dim=-1
            )
            select_feature_type = select_feature_type.replace("slice_m25811_f6_", "")
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]

        if select_feature_type == "patch":
            image_features = image_features[:, 1:]
        elif select_feature_type == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {select_feature_type}")
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        _hidden_size = self.config.hidden_size
        if "slicefour" in self.select_feature:
            _hidden_size *= 4
        if "slice_m25811_f6" in self.select_feature:
            _hidden_size *= 5
        return _hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        _num_patches = (self.config.image_size // self.config.patch_size) ** 2
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches

    @property
    def image_size(self):
        return self.config.image_size


class IdentityMap(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}


def build_vision_tower(config, **kwargs):
    _validate_vision_config(config)
    return CLIPVisionTower(config.mm_vision_tower, args=config, **kwargs)


def build_vision_projector(config, delay_load=False, **kwargs):
    if getattr(config, "mm_projector_type", None) != "mlp2x_gelu":
        raise ValueError("The released checkpoint requires the mlp2x_gelu projector")
    # Keep the exact Sequential indices used by model.mm_projector.{0,2} weights.
    return nn.Sequential(
        nn.Linear(config.mm_hidden_size, config.hidden_size),
        nn.GELU(),
        nn.Linear(config.hidden_size, config.hidden_size),
    )


def build_vision_resampler(config, delay_load=False, **kwargs):
    if getattr(config, "mm_resampler_type", None) is not None:
        raise ValueError("The released checkpoint uses an identity vision resampler")
    return IdentityMap()
