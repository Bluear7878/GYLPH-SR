# Training and data preparation

## Two adaptation stages

The paper uses two separate stages. First, the frozen LLaVA-NeXT 8B base is
adapted for OCR using a LoRA adapter (rank 8, alpha 32, dropout 0.05). Second,
SR fine-tuning updates the text-control fusion modules while keeping the
pretrained diffusion backbone, scene-SR control path, conditioning encoders,
VAE, and OCR guider fixed. The released OCR adapter can be downloaded through
[checkpoints.md](checkpoints.md). The `train` command runs the SR stage.

SR training draws from four partitions: positive/HQ, positive/LQ, negative/HQ,
and negative/LQ. For each sampled partition image `I`, both the control
representation and the target come from that image:

```text
sampled partition image I
        │
        ▼
frozen denoising encoder ──► h (control representation)
        │
        ▼
decode, then re-encode ────► z0 (target latent)
        │
        ▼
add diffusion noise ───────► z_sigma
```

The conditional diffusion loss predicts `z0`. In particular, a sampled LQ or
negative image uses its own denoised/re-encoded target. Training conditioning
combines the scene description with the partition's quality and glyph-condition
prompts. OCR metadata selects negatives; explicit OCR text/location guidance is
used during restoration inference. The shared U-Net is evaluated with the
pretrained and adapted fusion modules, and their predictions are averaged with
fixed 0.5 weights.

## Run SR fine-tuning

Install the runtime and prepare the backbone checkpoints first. Provide separate
training and validation metadata:

```bash
glyph-sr train \
  --hq-metadata /path/to/train/metadata_HQ.jsonl \
  --negative-metadata /path/to/train/metadata_with_neg.jsonl \
  --validation-hq-metadata /path/to/val/metadata_HQ.jsonl \
  --validation-negative-metadata /path/to/val/metadata_with_neg.jsonl \
  --batch-size 1 --epochs 200
```

This command is a starting template with a conservative batch size. The paper's
matched training ablations used 5,000 updates, batch size 12, image size 512,
learning rate `1e-6`, and 5,000 warm-up updates; those ablations differ from the
full 22,695-update checkpoint. Use `glyph-sr train --help` for checkpoint,
precision, device-count, logging, and other settings. The default packaged model
configuration is `glyph_sr/configs/sdxl.yaml`; `--model-config` accepts a custom
configuration. Earlier underscore options remain compatibility aliases.

## Metadata

HQ JSONL rows contain a scene prompt and positive image reference:

```json
{"prompt": "a street sign", "HQ_path": "images/positive.png"}
```

Negative rows contain the restored negative candidate and its source/candidate
OCR strings:

```json
{"prompt": "a street sign", "neg_path": "images/negative.png", "OCR": "STOP", "SR_OCR": "ST0P"}
```

Relative `HQ_path` and `neg_path` references resolve from the HQ metadata file's
directory. The loader constructs HQ records directly and creates LQ variants
using bicubic down/up-sampling and Gaussian blur. It excludes negative records
whose source OCR equals `[TEXT]` or shares any exact word with the candidate's
OCR. Training includes all four partitions; validation uses positive/HQ records.
For batches larger than one, prepare images with compatible tensor dimensions;
the loader does not silently crop mixed image shapes. Keep training and validation
source images disjoint and record data, checkpoint, and seed identities.

## Data generation

The maintained preparation commands are:

```bash
glyph-sr data prompts --image-dir /path/to/source --prompt-bank /path/to/prompts
glyph-sr data synthesize --prompt-meta /path/to/prompts/prompts_meta.jsonl --output-dir /path/to/positive
glyph-sr data negatives --img-dir /path/to/positive --save-dir /path/to/training
```

Prompt extraction writes `prompts_meta.jsonl`. Positive synthesis writes `hq/`
images and `metadata.jsonl`. Negative generation writes `neg_lq/` candidates,
`metadata_HQ.jsonl`, and `metadata_with_neg.jsonl`; image references are relative
to that metadata directory. Multiple ranks write distinct metadata shard names,
which must be combined explicitly. Use each command's `--help` for model paths,
devices, seeds, degradation factors, and batch settings. The default multi-factor
negative generation screens for OCR mismatches; a single down-factor retains the
historical behavior of bypassing that screening.

Negative candidate restoration uses SUPIR's separate original SDXL base checkpoint,
`sd_xl_base_1.0_0.9vae.safetensors`. Set `GLYPH_SUPIR_SDXL_CKPT` to its path, or place
it under `./checkpoints/`. The main GLYPH-SR restorer uses Juggernaut XL v8.

Generation models and their weights have separate dependencies and access
requirements. The complete synthetic corpus and the original generation
orchestration are not bundled.

The optional historical native Nunchaku backend is retained by an exact source
reference, in a separate generation environment after compatible PyTorch/CUDA
build dependencies are installed:

```bash
python -m pip install --no-build-isolation \
  'nunchaku @ git+https://github.com/Bluear7878/GYLPH-SR.git@e5e45de5960ab236ca61f9139fe0101a8de960ff#subdirectory=third_party/make_dataset_with_nunchaku'
```

That copied source reports `0.3.0dev0`; no equivalence to an upstream release tag
is assumed. Follow its build requirements for the selected GPU environment.
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) records the dependency's origin
and retained license terms. Generation does not require keeping its native
implementation vendored in the restoration package.
