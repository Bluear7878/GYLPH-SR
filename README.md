# GLYPH-SR

Official implementation of **A Bi-Objective Framework for Scene Text Super-Resolution: Balancing Legibility and Fidelity**, accepted at **Transactions on Machine Learning Research (TMLR), 2026**.

Mingyu Sung, Seungjae Ham, Kangwoo Kim, Il-Min Kim, Sangseok Yun, and Jae-Mo Kang.

[Paper and reviews](https://openreview.net/forum?id=9CnYnByhvB) ·
[Checkpoints](docs/checkpoints.md) · [Training](docs/training.md) ·
[Evaluation](docs/evaluation.md) · [Recorded results](evaluation/tmlr2026)

GLYPH-SR restores scene text while pursuing text legibility and perceptual image
quality as separate objectives. A frozen scene-restoration model is combined with
adapted text-control modules. During sampling, a ping-pong schedule selects
scene-caption or OCR text/location conditioning; continuous schedules interpolate
those conditioning tensors. The shared U-Net is evaluated with pretrained and
adapted fusion modules, and its predictions are averaged with fixed equal weights.
The OCR guider is LLaVA-NeXT 8B with a separately trained LoRA adapter.

```mermaid
flowchart LR
    LR["LR image"] --> SC["Scene-caption conditioning"]
    LR --> TX["OCR text/location conditioning"]
    SC --> Q["Select or interpolate<br/>by schedule lambda_i"]
    TX --> Q
    LR --> H["Shared LR representation h"]
    Q --> C["Condition-dependent<br/>control features"]
    H --> C
    Z["Noisy latent and noise level"] --> C
    C --> U["Shared U-Net evaluated twice<br/>pretrained and adapted fusion"]
    Q --> U
    Z --> U
    U --> M["Fixed prediction average<br/>0.5 base + 0.5 adapted"]
    M --> E["CFG and EDM update"]
```

Each prompt condition uses the same LR representation; CFG combines positive
and negative prompt evaluations before the EDM update.

## Installation

Restoration uses Linux, Python **3.10**, and NVIDIA GPUs. Install the matching
PyTorch and xformers wheels first, then install the package from this checkout:

```bash
git clone https://github.com/Bluear7878/GYLPH-SR.git
cd GYLPH-SR
conda create -n glyph-sr python=3.10 -y
conda activate glyph-sr
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install xformers==0.0.32.post2
python -m pip install -e '.[runtime]'
```

The runtime dependency pins match the recorded review environment. The default
inference placement uses `cuda:0` for SR, `cuda:1` for OCR, and `cuda:2` for scene
captioning. Device options can place components together when memory permits.
Download the components listed in [checkpoints.md](docs/checkpoints.md):

```text
checkpoints/
├── juggernautXL_v8Rundiffusion.safetensors
├── SUPIR-v0Q.ckpt
├── TS-ControlNet_cp.ckpt
└── Llava-next_FT/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

The LLaVA base and text encoders use their model caches or configured local paths.

## Restore an image

```bash
glyph-sr infer \
  --input /path/to/low_resolution.png \
  --output-dir ./results/demo \
  --upscale 4 --down-factor 1 --seed 1234 \
  --sr-device cuda:0 --ocr-device cuda:1 --caption-device cuda:2
```

`--down-factor 1` uses the supplied LR input. The command writes a restored PNG and
`metadata.jsonl` with the caption, recognized text, and verbalized locations.
The default configuration uses 50 restoration steps and nominal CFG increasing
from 4.0 toward 7.5 as the noise level decreases. Dataset-specific evaluation
settings are documented with the [recorded results](docs/evaluation.md).

`python -m glyph_sr` provides the same commands. Use `glyph-sr infer --help`,
`glyph-sr train --help`, and `glyph-sr data --help` for all options. The earlier
`run_GLYPH_SR.py` and `train_GLYPH_SR.py` entry points remain compatibility wrappers.
See [training.md](docs/training.md) for the two adaptation stages and data format.

## Recorded results

The following **measured** results use three diffusion seeds (1234, 2345, 3456), a
minimum internal side of 1024 pixels, and Wavelet color correction. Values are
mean ± sample standard deviation across seeds. WM is the paper's nonstandard
word-matching score; exact-word F1 counts every unmatched predicted/reference
word. Both OCR columns are percentages, and MANIQA uses its native 0–1 scale.

| Dataset | Scale | OpenOCR WM F1 | OpenOCR exact-word F1 | MANIQA |
| --- | --- | ---: | ---: | ---: |
| SVT | ×4 | 65.67 ± 0.97 | 25.50 ± 0.66 | 0.4422 ± 0.0005 |
| SVT | ×8 | 36.38 ± 0.39 | 13.75 ± 0.23 | 0.4159 ± 0.0010 |
| SCUT-CTW1500 | ×4 | 41.62 ± 0.20 | 29.54 ± 0.35 | 0.6606 ± 0.0037 |
| SCUT-CTW1500 | ×8 | 8.65 ± 0.02 | 6.48 ± 0.01 | 0.4368 ± 0.0043 |
| CUTE80 | ×4 | 76.74 ± 0.30 | 48.54 ± 0.58 | 0.4813 ± 0.0009 |
| CUTE80 | ×8 | 66.49 ± 0.91 | 38.77 ± 0.71 | 0.4533 ± 0.0010 |

These conditions show dataset-dependent trade-offs. Full GLYPH-SR/PASD/ResShift-v3
comparisons, CER, exact-transcription accuracy, common-guider studies, ablations,
and layout subsets are in [evaluation.md](docs/evaluation.md). The public evidence
contains 780 aggregate metric entries, all 732 paired tests, and the camera-ready
appendix's 252 displayed statistical comparisons, with recorded source hashes.

## Score and verify

Saved OCR predictions can be scored without loading restoration models:

```bash
python -m pip install -e '.[evaluation]'
glyph-sr score --input predictions.jsonl --output scores.json
glyph-sr verify
```

The scorer returns word-matching and exact-word metrics separately. Input format,
normalization, statistical definitions, and table links are in
[evaluation.md](docs/evaluation.md). The evidence package contains recorded
aggregate measurements; raw images and per-image OCR archives are separate from
this repository. See [validation](docs/release-validation.md) for software checks.

## Citation

```bibtex
@article{sung2026glyphsr,
  title   = {A Bi-Objective Framework for Scene Text Super-Resolution: Balancing Legibility and Fidelity},
  author  = {Sung, Mingyu and Ham, Seungjae and Kim, Kangwoo and Kim, Il-Min and Yun, Sangseok and Kang, Jae-Mo},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  note    = {Accepted},
  url     = {https://openreview.net/forum?id=9CnYnByhvB}
}
```

Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

## Acknowledgements and terms

This work was supported by the National Research Foundation of Korea(NRF) grant
funded by the Korea government(MSIT) (RS-2026-25493331).

The implementation builds on [SUPIR](https://github.com/Fanghua-Yu/SUPIR),
[LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT),
[generative-models](https://github.com/Stability-AI/generative-models), and
[Nunchaku](https://github.com/nunchaku-tech/nunchaku). Their code and model terms
remain applicable. See [third-party provenance](THIRD_PARTY_NOTICES.md) and the
retained license notices for code and model terms.
