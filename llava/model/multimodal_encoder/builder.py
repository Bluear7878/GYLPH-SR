# Modified for GLYPH-SR: load only the selected vision encoder.
import os


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    if not isinstance(vision_tower, str) or not vision_tower:
        raise ValueError("A nonempty mm_vision_tower or vision_tower identifier is required")
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, "s2", False)
    if (
        is_absolute_path_exists
        or vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
        or "ShareGPT4V" in vision_tower
    ):
        from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2

        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "siglip" in vision_tower:
        from .siglip_encoder import SigLipVisionTower

        return SigLipVisionTower(
            vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs
        )
    elif vision_tower.startswith("hf:"):
        from .hf_vision import HFVisionTower

        return HFVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower in ["imagebind_huge"]:
        from .imagebind import ImageBindWrapper

        return ImageBindWrapper(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("open_clip_hub"):
        from .open_clip_encoder import OpenCLIPVisionTower

        return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "mlcd-vit-bigG-patch14" in vision_tower:
        from .mlcd_encoder import MLCDVisionTower, MLCDVisionTowerS2

        if use_s2:
            return MLCDVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return MLCDVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
