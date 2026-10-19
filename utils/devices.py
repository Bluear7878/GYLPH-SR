"""Device helpers retained for the bundled tiled VAE implementation."""

import torch


def has_mps():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_cuda_device_string():
    return "cuda"


def torch_gc(target_device=None):
    target = torch.device(target_device) if target_device is not None else None
    if torch.cuda.is_available() and (target is None or target.type == "cuda"):
        with torch.cuda.device(target or get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    if has_mps() and (target is None or target.type == "mps"):
        torch.mps.empty_cache()


def enable_tf32():
    """Retain the numerical backend settings of the released CUDA pipeline."""
    if torch.cuda.is_available():
        if any(
            torch.cuda.get_device_capability(index) == (7, 5)
            for index in range(torch.cuda.device_count())
        ):
            torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


enable_tf32()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    if torch.all(torch.isnan(x)).item():
        raise NansException(f"A tensor with all NaNs was produced in {where}.")
