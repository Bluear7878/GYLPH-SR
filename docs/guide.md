# GLYPH-SR guide

## Checkpoints

Place the recorded model components under `checkpoints/` or configure their paths.
The TS checkpoint uses the `full_model` loading policy; loading only its text
projections changes the model configuration.

| Component | Source | Path under `checkpoints/` |
| --- | --- | --- |
| Juggernaut XL v8 | [Pinned file](https://huggingface.co/RunDiffusion/Juggernaut-XL-v8/resolve/5c38059bf38387bed17f70e5c1e5c00544caf41e/juggernautXL_v8Rundiffusion.safetensors) | `juggernautXL_v8Rundiffusion.safetensors` |
| SUPIR-v0Q | [Recorded download](https://weights.replicate.delivery/default/SUPIR-v0Q.ckpt), [upstream](https://github.com/Fanghua-Yu/SUPIR) | `SUPIR-v0Q.ckpt` |
| TS-ControlNet | [Google Drive](https://drive.google.com/file/d/121NK-KAOzrKPgQ8XGIf57TQ3HqLFM3yb/view) | `TS-ControlNet_cp.ckpt` |
| OCR LoRA configuration | [Google Drive](https://drive.google.com/file/d/1AvgzyFtdx0Q0K3O21qYaFLl1fb1m08JG/view) | `Llava-next_FT/adapter_config.json` |
| OCR LoRA weights | [Google Drive](https://drive.google.com/file/d/1m7PIIE-_uScY085a48xKLy-10AdzsvEj/view) | `Llava-next_FT/adapter_model.safetensors` |
| LLaVA-NeXT base | [lmms-lab/llama3-llava-next-8b](https://huggingface.co/lmms-lab/llama3-llava-next-8b) | Model cache or local snapshot |

[checkpoints.json](checkpoints.json) records recovered file sizes and SHA-256
identities. The large payloads were not downloaded again during repository cleanup.
Verify downloaded copies with:

```bash
glyph-sr verify --checkpoint-dir ./checkpoints
```

The recorded LLaVA base revision is `6d6dcb8d7948364bebef9c11aba6b0e6c8835391`.
For a pinned local snapshot:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "lmms-lab/llama3-llava-next-8b",
    revision="6d6dcb8d7948364bebef9c11aba6b0e6c8835391",
    local_dir="checkpoints/llama3-llava-next-8b",
)
```

Set `GLYPH_LLAVA_MODEL_PATH` to that local directory to use it. The vision tower
and text encoders otherwise follow their model configurations and caches.
Checkpoint defaults are relative to the current working directory; packaged YAML
configurations resolve automatically. A custom `--model-config` path is read as supplied.

| Variable | Component |
| --- | --- |
| `GLYPH_SDXL_CKPT` | Main restorer's Juggernaut XL v8 |
| `GLYPH_SUPIR_Q_CKPT` | SUPIR-v0Q |
| `GLYPH_TS_CKPT` | TS-ControlNet; also `--ts-checkpoint` |
| `GLYPH_LLAVA_MODEL_PATH` | LLaVA base; also `--caption-model` |
| `GLYPH_LLAVA_ADAPTER_PATH` | OCR adapter; also `--ocr-adapter` |
| `GLYPH_CLIP1_PATH` | SDXL CLIP-L |
| `GLYPH_CLIP2_PATH` | OpenCLIP weight file or pretrained tag |
| `GLYPH_SUPIR_SDXL_CKPT` | Original SDXL base for negative-data generation |

Model weights retain their upstream terms. Load trusted checkpoints: the released
Lightning format uses PyTorch pickle deserialization.

## Training

The paper uses two adaptation stages. OCR adaptation updates a LoRA adapter on
the frozen LLaVA-NeXT 8B base (rank 8, alpha 32, dropout 0.05). SR fine-tuning then
updates the text-control fusion modules while holding the diffusion backbone,
pretrained scene-SR control path, conditioning encoders, VAE, and OCR guider fixed.
`glyph-sr train` runs the SR stage; the released OCR adapter supplies guidance
during restoration.

SR training samples positive/HQ, positive/LQ, negative/HQ, and negative/LQ images.
For each sampled image `I`, the frozen denoising encoder produces the control
representation `h`; decoding and re-encoding `h` produces the target latent `z0`.
The conditional diffusion loss predicts that image's `z0` from its noisy latent.
A sampled negative or LQ image therefore uses its own denoised/re-encoded target.
Training prompts combine the scene description with the partition's image-quality
and glyph-condition prompts. Explicit OCR text/location cues are used at inference.
Pretrained and adapted fusion modules evaluate the shared U-Net, with predictions
averaged using fixed 0.5 weights.

After runtime installation and checkpoint setup, supply disjoint training and
validation metadata. This is a starting template with a conservative batch size:

```bash
glyph-sr train \
  --hq-metadata /path/to/train/metadata_HQ.jsonl \
  --negative-metadata /path/to/train/metadata_with_neg.jsonl \
  --validation-hq-metadata /path/to/val/metadata_HQ.jsonl \
  --validation-negative-metadata /path/to/val/metadata_with_neg.jsonl \
  --batch-size 1 --epochs 200
```

The paper's matched ablations use 5,000 updates, batch size 12, image size 512,
learning rate `1e-6`, and 5,000 warm-up updates; the full checkpoint has 22,695
updates. Use `glyph-sr train --help` for training-budget, precision, device-count,
checkpoint, and logging controls.

HQ metadata contains a scene prompt and positive image reference:

```json
{"prompt": "a street sign", "HQ_path": "images/positive.png"}
```

Negative metadata also records source and candidate OCR strings:

```json
{"prompt": "a street sign", "neg_path": "images/negative.png", "OCR": "STOP", "SR_OCR": "ST0P"}
```

These examples are synthetic. Relative image paths resolve from the HQ metadata
file's directory. LQ variants use bicubic down/up-sampling and Gaussian blur.
Negatives whose source OCR is `[TEXT]` or shares any case-sensitive exact word
with the candidate OCR are excluded. Training uses all four partitions; validation uses positive/HQ.
Batches larger than one require compatible image dimensions; the loader preserves
image aspect ratios and rejects mixed tensor shapes.

## Data preparation

The maintained commands operate on data supplied by the user:

```bash
glyph-sr data prompts --image-dir /path/to/source --prompt-bank /path/to/prompts
glyph-sr data synthesize --prompt-meta /path/to/prompts/prompts_meta.jsonl --output-dir /path/to/positive
glyph-sr data negatives --img-dir /path/to/positive --save-dir /path/to/training
```

Prompt extraction writes `prompts_meta.jsonl`. Synthesis writes `hq/` images and
`metadata.jsonl`. Negative generation consumes that metadata and writes `neg_lq/`,
`metadata_HQ.jsonl`, and `metadata_with_neg.jsonl`, with image references relative
to the output metadata directory. Multiple ranks write separate metadata shards.
The default multi-factor negative generation screens for OCR mismatches; a single
down-factor preserves the historical behavior of bypassing that screening.

Negative restoration uses SUPIR's original SDXL base,
`sd_xl_base_1.0_0.9vae.safetensors`. Set `GLYPH_SUPIR_SDXL_CKPT` to its path or place
it under `./checkpoints/`. The main GLYPH-SR restorer uses Juggernaut XL v8.
Generation models and weights need their own environment and setup. The optional
native Nunchaku backend is pinned to the recorded source snapshot:

```bash
python -m pip install --no-build-isolation \
  'nunchaku @ git+https://github.com/Bluear7878/GYLPH-SR.git@e5e45de5960ab236ca61f9139fe0101a8de960ff#subdirectory=third_party/make_dataset_with_nunchaku'
```

Install it in a separate generation environment with compatible PyTorch/CUDA build
dependencies. The snapshot reports `0.3.0dev0`; an equivalent upstream tag was not
recorded. See [third-party provenance](../THIRD_PARTY_NOTICES.md) for its origin.
The synthetic corpus and original generation orchestration are not bundled.

## OCR scoring

Scoring saved, aligned OCR predictions needs only the evaluation extra:

```bash
python -m pip install -e '.[evaluation]'
glyph-sr score --input predictions.jsonl --output scores.json
```

Each line uses this format; the example is synthetic:

```json
{"filename": "example.png", "ground_truth": "STOP", "predicted": "STOP EXTRA"}
```

`--prediction-field pred_text` selects another prediction field. Rows with
`ground_truth: null` are excluded and counted.

| Output field | Meaning |
| --- | --- |
| `corrected_word` | Dataset-level exact-word precision, recall, and F1 from multiset counts; all unmatched predicted/reference words count as FP/FN |
| `legacy_submitted_word` | The paper's nonstandard word-matching precision, recall, and F1 |
| `character_error_rate` | Total character edits / reference characters after exact-score normalization and removal of spaces; percent, possibly above 100 |
| `exact_image_accuracy` | Percentage of complete normalized transcriptions matching exactly |
| `per_image` | User-supplied transcriptions, exact-word counts, and edit distances for inspection |

Word matching counts one-to-one exact matches as TP. Each remaining reference word
counts as FP if its most similar unused prediction has `SequenceMatcher` similarity
at least 0.5; otherwise it counts as FN. Additional unused predictions are ignored,
and ties retain the first candidate. It removes non-word/non-whitespace characters,
lowercases, and strips boundary whitespace. Letters, digits, and underscores remain.

Exact-word scoring uses Unicode NFKC, case folding, non-word-symbol removal, and
collapsed whitespace. It computes multiset intersections per image and sums counts
across the dataset before calculating precision, recall, and F1. Word order is
ignored, and these rules do not measure spatial localization. Undefined zero-denominator
ratios become zero. CER and exact-transcription accuracy use exact-score normalization;
the paper's historical word accuracy and 1-NED are separate measures.

## Recorded results

[results/aggregates.json](results/aggregates.json) preserves 780 measured aggregate
metric entries. [results/statistics.json](results/statistics.json) preserves all
732 paired tests in eight declared families. Both files retain their recorded
source bytes; the [manifest](results/manifest.json) records their hashes and provenance.

| Analysis family | Tests |
| --- | ---: |
| Main benchmark | 216 |
| Common guider | 102 |
| SVT cues | 54 |
| SVT scheduler | 18 |
| SVT CFG | 72 |
| SVT control strength | 54 |
| SVT corpus | 72 |
| Curved/dense layouts | 144 |

English `*_f1` fields use the paper's word-matching rule; `*_strict_f1` fields use
exact-word scoring. Non-Latin `unicode_f1` uses grapheme matching. MANIQA and
CLIP-IQA use native 0–1 scales, while MUSIQ uses its native scale. The matched runs
used `pyiqa==0.1.16`; scaling older results does not establish protocol equivalence.

The benchmark comparison uses seeds 1234, 2345, and 3456, matched LR inputs, and
matched target canvases. GLYPH-SR uses a minimum internal side of 1024 and Wavelet
correction. SVT has
350 images (349 OCR annotations), SCUT-CTW1500 has 500, and CUTE80 has 80. ResShift-v3
at ×8 uses 2× bicubic pre-upsampling before its released ×4 model.

SVT cue/strength/corpus ablations use 1024/Wavelet, 50 steps, and seed 1234; the
corpus checkpoints use 5,000 updates, and the control sweep uses an FP32 denoiser
and BF16 VAE. The SVT scheduler comparison uses three seeds. Common-guider studies
hold weights/settings fixed across Chinese/Hindi crops and an English control;
cue ablations share scene captions with text identities removed. Curved/dense
SCUT subsets are selected from annotations without consulting restoration outcomes:
curved requires a 14-vertex polygon with normalized centerline bending ≥0.05, and
dense requires at least 10 non-ignored instances. The subsets can overlap.

Paired 95% intervals use 10,000 image-cluster bootstrap resamples, retaining all
seeds of each image together. Holm adjustment covers every test in its declared
family. Differences are comparison minus reference; seed standard deviations
are different from these intervals.

The [analysis plan](results/analysis_plan.json) retains settings, family definitions,
and seed-level cohort counts. Per-image identifier lists are omitted, and its
original source hash is retained in the manifest. Dataset images, annotation bodies,
and individual OCR transcriptions are not included in these recorded results.

## Verification

```bash
glyph-sr verify
```

Verification checks all three result files against their recorded sizes and hashes.
It locates source-checkout or installed-wheel evidence automatically, including
`pip --target` installations. `--root /path/to/checkout` selects a particular copy.
GPU training, full restoration inference, and new measurements of paper results
were not performed by this repository cleanup.
