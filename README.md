# GLYPH-SR

**A Bi-Objective Framework for Scene Text Super-Resolution: Balancing Legibility and Fidelity**

Mingyu Sung, Seungjae Ham, Kangwoo Kim, Il-Min Kim, Sangseok Yun, and Jae-Mo Kang

*Accepted at TMLR 2026*

[Paper](https://openreview.net/forum?id=9CnYnByhvB) · [Setup](#setup) · [Inference](#inference) · [Training and evaluation](docs/guide.md)

GLYPH-SR combines a diffusion restorer with text-aware control and alternating scene/OCR guidance for scene text super-resolution.

## Method

![GLYPH-SR framework from the paper](https://github.com/user-attachments/assets/9b7f2f98-caec-4c75-a2a1-c368b38f435e)

## Results

![Qualitative comparisons from the paper](https://github.com/user-attachments/assets/4023afef-5311-4769-b7da-fcbea46bfefe)

GLYPH-SR results from **Table 1** of the paper. The three OCR columns report **word-matching F1**. All scores use the paper's **0–100 display scale**, including MANIQA and CLIP-IQA.

| Dataset | Scale | OpenOCR | GOT-OCR | LLaVA-NeXT | MANIQA | CLIP-IQA | MUSIQ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SVT | ×4 | 67.54 | 71.72 | 73.22 | 47.75 | 59.40 | 70.99 |
| SVT | ×8 | 48.79 | 56.16 | 58.54 | 47.40 | 56.78 | 69.93 |
| SCUT-CTW1500 | ×4 | 38.26 | 36.96 | 42.90 | 70.33 | 57.88 | 70.31 |
| SCUT-CTW1500 | ×8 | 11.09 | 14.71 | 14.67 | 61.94 | 48.21 | 63.43 |
| CUTE80 | ×4 | 73.09 | 55.62 | 85.01 | 49.77 | 65.93 | 69.96 |
| CUTE80 | ×8 | 63.66 | 45.65 | 73.71 | 47.75 | 65.85 | 68.85 |

See the [paper](https://openreview.net/forum?id=9CnYnByhvB) for the full comparisons.

## Setup

Linux, Python 3.10, and NVIDIA GPUs are required for restoration.

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

### Pretrained models

Download the following files and place them under `checkpoints/`:

| Model | Download | Local path |
| --- | --- | --- |
| Juggernaut XL v8 | [Hugging Face](https://huggingface.co/RunDiffusion/Juggernaut-XL-v8/resolve/5c38059bf38387bed17f70e5c1e5c00544caf41e/juggernautXL_v8Rundiffusion.safetensors) | `checkpoints/juggernautXL_v8Rundiffusion.safetensors` |
| SUPIR-v0Q | [Weights](https://weights.replicate.delivery/default/SUPIR-v0Q.ckpt) | `checkpoints/SUPIR-v0Q.ckpt` |
| GLYPH-SR | [Google Drive](https://drive.google.com/file/d/121NK-KAOzrKPgQ8XGIf57TQ3HqLFM3yb/view) | `checkpoints/TS-ControlNet_cp.ckpt` |
| OCR LoRA config | [Google Drive](https://drive.google.com/file/d/1AvgzyFtdx0Q0K3O21qYaFLl1fb1m08JG/view) | `checkpoints/Llava-next_FT/adapter_config.json` |
| OCR LoRA weights | [Google Drive](https://drive.google.com/file/d/1m7PIIE-_uScY085a48xKLy-10AdzsvEj/view) | `checkpoints/Llava-next_FT/adapter_model.safetensors` |

The [LLaVA-NeXT 8B base](https://huggingface.co/lmms-lab/llama3-llava-next-8b) and CLIP encoders are downloaded to their model caches on first use. For local model directories and pinned revisions, see [model setup](docs/guide.md#checkpoints).

## Inference

Run on a low-resolution input image:

```bash
glyph-sr infer \
  --input /path/to/input.png \
  --output-dir ./results/demo \
  --upscale 4 \
  --down-factor 1 \
  --steps 50 \
  --seed 42 \
  --sr-device cuda:0 \
  --ocr-device cuda:1 \
  --caption-device cuda:2
```

The command saves `input_sr_0001.png` and `metadata.jsonl` in `results/demo/`.

- `--upscale` sets the output scale; `--down-factor 1` uses the input without additional synthetic degradation.
- `--sr-device`, `--ocr-device`, and `--caption-device` set the three model placements. The defaults use three CUDA devices; they can share a device when memory permits.
- `--ts-checkpoint`, `--ocr-adapter`, and `--caption-model` override the model paths above.
- `--steps` and `--seed` control sampling. Use `glyph-sr infer --help` for all options.

`python -m glyph_sr infer` is equivalent. [Training](docs/guide.md#training), [data preparation](docs/guide.md#data-preparation), and [OCR scoring](docs/guide.md#ocr-scoring) are covered in the usage guide.

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

## Acknowledgements

This work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (RS-2026-25493331).

Built on [SUPIR](https://github.com/Fanghua-Yu/SUPIR), [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), [generative-models](https://github.com/Stability-AI/generative-models), and [Nunchaku](https://github.com/nunchaku-tech/nunchaku). See [third-party notices](THIRD_PARTY_NOTICES.md) for code and model terms.
