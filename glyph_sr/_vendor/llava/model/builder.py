#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Modified for GLYPH-SR: load selected architectures and respect explicit device placement.


import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from glyph_sr._vendor.llava import model as model_registry
from glyph_sr._vendor.llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from glyph_sr._vendor.llava.utils import rank0_print

from .vision import validate_model_config


def _selected_model_class(model_name):
    if model_name not in {"llava_llama_3", "llama3-llava-next-8b"}:
        raise ValueError("Only the released llama3-llava-next-8b architecture is supported")
    return model_registry.LlavaLlamaForCausalLM


def _vision_tower_device(device_map):
    """Respect explicit placement; leave automatic or split dispatch to Accelerate."""
    if device_map == "auto":
        return None
    if isinstance(device_map, dict):
        placement = device_map.get("")
        if placement is None or placement == "disk":
            return None
        return f"cuda:{placement}" if isinstance(placement, int) else placement
    return f"cuda:{device_map}" if isinstance(device_map, int) else device_map


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    torch_dtype="float16",
    attn_implementation="flash_attention_2",
    customized_config=None,
    overwrite_config=None,
    **kwargs,
):
    """Load the released complete Llama3-LLaVA-NeXT base; OCR adapters load separately."""
    if model_base is not None:
        raise ValueError(
            "Only complete Llama3-LLaVA-NeXT checkpoints are supported; load OCR adapters separately"
        )
    if torch_dtype not in {"float16", "bfloat16"}:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype!r}; use float16 or bfloat16")
    if load_8bit and load_4bit:
        raise ValueError("Choose at most one quantization mode")
    model_class = _selected_model_class(model_name)
    config = customized_config or model_registry.LlavaConfig.from_pretrained(model_path)
    for name, value in (overwrite_config or {}).items():
        setattr(config, name, value)
    validate_model_config(config)
    kwargs.pop("multimodal", None)
    kwargs["device_map"] = device_map
    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16 if torch_dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = model_class.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        config=config,
        attn_implementation=attn_implementation,
        **kwargs,
    )
    if getattr(model.config, "mm_use_im_patch_token", True):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if getattr(model.config, "mm_use_im_start_end", False):
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    tower = model.get_vision_tower()
    if not tower.is_loaded:
        tower.load_model(device_map=device_map)
    vision_device = _vision_tower_device(device_map)
    if vision_device is not None:
        tower.to(device=vision_device, dtype=torch.float16)
    context_len = next(
        (
            getattr(model.config, name)
            for name in (
                "max_sequence_length",
                "max_position_embeddings",
                "tokenizer_model_max_length",
            )
            if hasattr(model.config, name)
        ),
        2048,
    )
    rank0_print(f"Loaded LLaVA model: {model_path}")
    return tokenizer, model, tower.image_processor, context_len
