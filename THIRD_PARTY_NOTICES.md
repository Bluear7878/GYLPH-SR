# Third-party provenance

GLYPH-SR retains the upstream module names needed by its released model configs
and checkpoints. The refactor does not assign a new license to all repository content.

| Component | Retained code | Upstream and terms |
| --- | --- | --- |
| LLaVA-NeXT | `llava/` runtime model, tokenizer and vision integration | [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT); Apache 2.0 headers retained, full text in `licenses/Apache-2.0.txt` |
| Generative models / SDXL | `sgm/` diffusion runtime | [generative-models](https://github.com/Stability-AI/generative-models); upstream code terms in `licenses/StabilityAI-LICENSE-CODE.txt` |
| SUPIR | restoration layers, image utilities and tiled VAE | [SUPIR](https://github.com/Fanghua-Yu/SUPIR); upstream terms and model restrictions remain applicable |
| Nunchaku / SVDQuant | optional external backend for synthetic data generation | [Nunchaku](https://github.com/nunchaku-tech/nunchaku); Apache 2.0 |
| LI YI's tiled VAE | `utils/tilevae.py` | Original file attribution identifies MIT; verified historical text in `licenses/LiYi-TiledVAE-MIT.txt`, with provenance limits below |
| StableSR color correction | `utils/colorfix.py` | [LI YI's sd-webui-stablesr source](https://github.com/pkuliyi2015/sd-webui-stablesr/blob/6f616446bae4ed8051ed9661e8122c9dea2f27ab/srmodule/colorfix.py); repository's S-Lab License 1.0 retained in `licenses/StableSR-SLab-LICENSE.txt` |

The tiled-VAE file originally attributes LI YI at Nanyang Technological
University, dated 2023-03-02, and states MIT License. Primary upstream
[source](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/300e42fd52c8e4b63992941a9b874fa49847a85b/scripts/vae_optimize.py)
and [license](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/300e42fd52c8e4b63992941a9b874fa49847a85b/LICENSE)
at commit `300e42fd52c8e4b63992941a9b874fa49847a85b` confirm that historical
notice. The `get_var_mean`, `custom_group_norm`, and `inplace_nonlinearity`
function bodies match that source after excluding docstrings. The exact
revision copied into GLYPH-SR was not recorded; full-file provenance beyond
those matched routines remains unresolved. The historical license text does
not establish the terms of all later revisions or the entire retained file.

The color-correction source has a separate provenance: its primary repository's
[license at commit `6f616446bae4ed8051ed9661e8122c9dea2f27ab`](https://github.com/pkuliyi2015/sd-webui-stablesr/blob/6f616446bae4ed8051ed9661e8122c9dea2f27ab/LICENSE)
is S-Lab License 1.0, copyright 2022 S-Lab. Its exact text is retained separately
from the tiled-VAE historical MIT notice. Both license files are included in
source and wheel distributions.

The original Nunchaku snapshot identifies itself as `0.3.0dev0`; an exact upstream
commit was not recorded. To preserve the actual source used by the generator,
[training instructions](docs/training.md) pin the archived backend to GLYPH-SR
commit `e5e45de5960ab236ca61f9139fe0101a8de960ff` and its original subdirectory.
That immutable snapshot includes the native build sources. The current tree keeps
only GLYPH-SR's data-generation entry points, not duplicated CUDA libraries,
benchmarks, web applications, or upstream test suites.

Weights, datasets and pretrained model releases have their own terms. Their
references and hashes are documented in [checkpoints](docs/checkpoints.md).
Historical source, experiment scripts and the environment freeze remain available
in the repository history at the same pre-refactor commit.
