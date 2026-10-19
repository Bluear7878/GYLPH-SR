# Modified for GLYPH-SR: resolve architecture exports lazily.
"""Lazy architecture exports, preserving checkpoint class identities."""

from importlib import import_module

AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",
    "llava_gemma": "LlavaGemmaForCausalLM, LlavaGemmaConfig",
}

_EXPORTS = {
    name.strip(): module
    for module, names in AVAILABLE_MODELS.items()
    for name in names.split(",")
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(import_module(f".language_model.{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
