"""Load released TS weights with explicit key/shape coverage checks."""

import torch


def load_TS_pretrained_model(model, ckpt_path, device, strict=False):
    # Released files include Lightning checkpoints, not only bare state dicts.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    candidates = [("full_model", model, state), ("inner_model", model.model, state)]
    if any(key.startswith("model.") for key in state):
        candidates.append((
            "inner_model_stripped_model_prefix", model.model,
            {key.removeprefix("model."): value for key, value in state.items()
             if key.startswith("model.")},
        ))
    ranked = []
    for label, target, weights in candidates:
        target_state = target.state_dict()
        matching = {
            key for key, value in weights.items()
            if key in target_state and torch.is_tensor(value)
            and target_state[key].shape == value.shape
        }
        ranked.append((sum(target_state[key].numel() for key in matching),
                       len(matching), label, target, weights, target_state, matching))
    _, count, label, target, weights, target_state, matching = max(
        ranked, key=lambda item: (item[0], item[1])
    )
    if not count:
        raise RuntimeError(f"TS checkpoint matches no model tensors: {ckpt_path}")
    projections = {key for key in target_state if "diffusion_model.project_modules." in key}
    if not projections or not projections.issubset(matching):
        raise RuntimeError("TS checkpoint does not contain every text projection tensor")
    incompatible = target.load_state_dict(weights, strict=strict, assign=True)
    model._ts_checkpoint_load_report = {
        "load_policy": "full_model",
        "load_target": label,
        "matched_tensor_count": count,
        "matched_projection_tensor_count": len(projections),
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(incompatible.unexpected_keys),
    }
    del checkpoint, state, candidates, ranked, weights, target_state
    model.to(device)
    return model
