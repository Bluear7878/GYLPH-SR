from .encoders.modules import (GeneralConditioner,
                               GeneralConditionerWithControl,
                               PreparedConditioner)

UNCONDITIONAL_CONFIG = {
    "target": "glyph_sr._vendor.sgm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
