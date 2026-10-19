"""Checkpoint loading with explicit format and complete model coverage checks."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class _LoadTarget:
    name: str
    model: torch.nn.Module
    weights: Mapping
    expected: Mapping
    matching: set

    @property
    def complete(self):
        return bool(self.expected) and self.matching == set(self.expected)

    @property
    def matched_elements(self):
        return sum(self.expected[key].numel() for key in self.matching)


def get_state_dict(checkpoint):
    """Unwrap a Lightning checkpoint or validate a bare state dictionary."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a state dictionary")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
        raise TypeError("checkpoint state_dict must map string keys to tensors")
    return state


def load_state_dict(ckpt_path, location="cpu"):
    """Read trusted PyTorch/Lightning or safetensors weights on ``location``."""
    if Path(ckpt_path).suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(str(ckpt_path), device=str(location))
    else:
        checkpoint = torch.load(ckpt_path, map_location=location, weights_only=False)
    return get_state_dict(checkpoint)


def _assign_ts_checkpoint(model, ckpt_path, strict):
    """Validate and assign weights, then release all temporary tensor snapshots."""
    state = load_state_dict(ckpt_path)
    full_state = model.state_dict()
    has_outer_weights = any(key in full_state and not key.startswith("model.") for key in state)
    candidates = [("full_model", model, state)]
    if hasattr(model, "model") and not has_outer_weights:
        candidates.append(("inner_model", model.model, state))
        if any(key.startswith("model.") for key in state):
            candidates.append(
                (
                    "inner_model_stripped_model_prefix",
                    model.model,
                    {
                        key.removeprefix("model."): value
                        for key, value in state.items()
                        if key.startswith("model.")
                    },
                )
            )

    ranked = []
    for label, target, weights in candidates:
        target_state = target.state_dict()
        matching = {
            key
            for key, value in weights.items()
            if key in target_state
            and torch.is_tensor(value)
            and target_state[key].shape == value.shape
        }
        ranked.append(_LoadTarget(label, target, weights, target_state, matching))
    selected = max(
        ranked,
        key=lambda item: (item.complete, item.matched_elements, len(item.matching)),
    )
    if not selected.matching:
        raise RuntimeError(f"TS checkpoint matches no model tensors: {ckpt_path}")

    projections = {key for key in selected.expected if "diffusion_model.project_modules." in key}
    if not projections or not projections.issubset(selected.matching):
        raise RuntimeError("TS checkpoint does not contain every text projection tensor")
    missing = set(selected.expected) - selected.matching
    if missing:
        examples = ", ".join(sorted(missing)[:3])
        raise RuntimeError(
            f"TS checkpoint does not contain the full model: {len(missing)} "
            f"missing or incompatible tensors ({examples})"
        )

    incompatible = selected.model.load_state_dict(selected.weights, strict=strict, assign=True)
    return {
        "load_policy": "full_model",
        "load_target": selected.name,
        "expected_tensor_count": len(selected.expected),
        "matched_tensor_count": len(selected.matching),
        "matched_projection_tensor_count": len(projections),
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(incompatible.unexpected_keys),
    }


def load_ts_checkpoint(model, ckpt_path, device, strict=False):
    """Load complete TS weights in full-model or wrapped inner-model format.

    Text projections alone are insufficient: the released configuration also
    restores the frozen branch and the other tensors in the selected model.
    ``strict=False`` permits extra source keys, but never missing model tensors.
    """
    # End the assignment scope before transferring weights: its state dictionaries
    # hold the old tensors and would otherwise retain their GPU storage during .to().
    model._ts_checkpoint_load_report = _assign_ts_checkpoint(model, ckpt_path, strict)
    model.to(device)
    return model
