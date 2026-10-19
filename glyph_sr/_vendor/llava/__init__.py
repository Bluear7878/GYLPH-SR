# Modified for GLYPH-SR: load model exports lazily.
"""The retained LLaVA runtime; model code loads only when requested."""

__all__ = ["LlavaLlamaForCausalLM"]


def __getattr__(name):
    if name in __all__:
        from . import model

        value = getattr(model, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
