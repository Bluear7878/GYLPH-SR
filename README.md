# GLYPH-SR

Official implementation of **A Bi-Objective Framework for Scene Text Super-Resolution: Balancing Legibility and Fidelity**.

Mingyu Sung, Seungjae Ham, Kangwoo Kim, Il-Min Kim, Sangseok Yun, and Jae-Mo Kang.

**Accepted at TMLR; camera-ready preparation in progress (2026).**
[Paper and review](https://openreview.net/forum?id=9CnYnByhvB) ·
[Checkpoints](docs/checkpoints.md) · [Training](docs/training.md) ·
[Evaluation](docs/evaluation.md) · [Recorded results](evaluation/tmlr2026)

## Overview

GLYPH-SR restores text in full natural scenes using vision-language guidance and
an SDXL-based diffusion restorer. It combines a frozen scene-restoration path,
a trainable text-control path, and a ping-pong schedule that alternates the two
guidance streams. OCR text, verbalized locations and a scene caption provide
conditioning signals.

The paper treats legibility and perceptual fidelity as separate objectives.
Results on SVT, SCUT-CTW1500 and CUTE80 depend on the dataset and degradation;
improvements on one axis need not imply improvements on the other. See the
paper for complete comparisons and limitations.

![GLYPH-SR architecture: scene and text guidance enter separate control paths and alternate during diffusion sampling](https://github.com/user-attachments/assets/9b7f2f98-caec-4c75-a2a1-c368b38f435e)

## October 2026 release update

- Updated paper identity, citation and model documentation, including the actual
  **LLaVA-NeXT 8B** base and separately trained OCR LoRA.
- Replaced machine-specific defaults with relative paths/environment variables
  and exposed SR, OCR and caption-model device selection.
- Added checkpoint key-coverage checks and support for Lightning checkpoint
  containers, so incompatible TS weights fail explicitly.
- Added a portable OCR scorer separating corrected exact-word metrics from
  historical matching scores.
- Included recorded matched-review aggregates, statistics, analysis plan and
  selected LaTeX tables with a hash verifier.

The old README's ×6 table and broad superiority statements are superseded by
the paper and the scoped evaluation records. The old environment freeze remains
in `requirements/legacy-environment.txt` for provenance.

## Installation

Use Linux, Python **3.10**, and NVIDIA GPUs for restoration. The dependency pins
follow the recorded review environment; upgrading Transformers independently can
break the bundled LLaVA code. The following CUDA 12.8/PyTorch 2.8 wheel pair follows
[PyTorch's installation matrix](https://pytorch.org/get-started/previous-versions/).
Use a driver compatible with that wheel build.

```bash
git clone https://github.com/Bluear7878/GYLPH-SR.git
cd GYLPH-SR
conda create -n glyph-sr python=3.10 -y
conda activate glyph-sr
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install xformers==0.0.32.post2
python -m pip install -r requirements.txt
```

The diffusion model, caption VLM and OCR-adapted VLM are large. The inference
entry point defaults to three logical devices (`cuda:0`, `cuda:1`, `cuda:2`).
They may share a device if enough memory is available; the release update does
not establish a new minimum-VRAM or single-GPU performance claim.

Download and verify the files in [docs/checkpoints.md](docs/checkpoints.md).
The default directory layout is:

```text
checkpoints/
├── juggernautXL_v8Rundiffusion.safetensors
├── SUPIR-v0Q.ckpt
├── TS-ControlNet_cp.ckpt
└── Llava-next_FT/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

The LLaVA base and CLIP encoders use the Hugging Face/OpenCLIP caches unless local
paths are configured. Model weights are not stored in this Git repository.

## Restore an image

Run from the repository root after checkpoint setup:

```bash
python run_GLYPH_SR.py \
  --img_path /path/to/low_resolution.png \
  --save_dir ./results/demo \
  --upscale 4 --down_factor 1 --seed 1234 \
  --yaml_file model_configs/juggernautXL.yaml \
  --GYLPH_pretrained_ckpt ./checkpoints/TS-ControlNet_cp.ckpt \
  --VLM_pretrained_ckpt ./checkpoints/Llava-next_FT \
  --sr_device cuda:0 --ocr_device cuda:1 --vlm_device cuda:2
```

`--down_factor 1` keeps the supplied LR input; larger values deliberately degrade
it first. The script writes a PNG and `metadata.jsonl` containing its caption,
OCR text and verbalized locations. This single-image example is not a complete
reproduction of the paper's matched experiment protocol.

`python run_GLYPH_SR.py --help` and `python train_GLYPH_SR.py --help` work without
loading the model stack. See [training.md](docs/training.md) for the actual
metadata arguments and [evaluation.md](docs/evaluation.md) for scoring saved OCR
predictions and interpreting the included results.

## Validation

The release checks use CPU toy models and synthetic predictions, not new SR
experiments:

```bash
python -m pip install -r requirements/evaluation.txt
python -m unittest discover -s tests -v
python tools/verify_release.py
```

The public bundle contains aggregate review evidence, not the complete raw
image/OCR archive or a turnkey reproduction of every table. Historical metric
protocol differences are documented in [evaluation.md](docs/evaluation.md).

## Citation

```bibtex
@article{sung2026glyphsr,
  title   = {A Bi-Objective Framework for Scene Text Super-Resolution: Balancing Legibility and Fidelity},
  author  = {Sung, Mingyu and Ham, Seungjae and Kim, Kangwoo and Kim, Il-Min and Yun, Sangseok and Kang, Jae-Mo},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  note    = {Accepted; camera-ready version in preparation},
  url     = {https://openreview.net/forum?id=9CnYnByhvB}
}
```

Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

## Acknowledgements

This work was supported by the National Research Foundation of Korea(NRF) grant
funded by the Korea government(MSIT) (RS-2026-25493331).

The implementation builds on [SUPIR](https://github.com/Fanghua-Yu/SUPIR),
[LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT),
[generative-models](https://github.com/Stability-AI/generative-models), and
[Nunchaku](https://github.com/nunchaku-tech/nunchaku). Their code and model terms
remain applicable; this update does not assign a new repository-wide license.
