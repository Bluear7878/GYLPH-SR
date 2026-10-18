# Training and data preparation

`train_GLYPH_SR.py` trains the text-control projection branch while keeping the
pretrained scene-restoration backbone frozen. Prepare the backbone checkpoints
as described in [checkpoints.md](checkpoints.md) first. The OCR LoRA is a separate
pretrained component; this command does not train it.

Run from the repository root with separate training and validation metadata:

```bash
python train_GLYPH_SR.py \
  --hq_meta_path /path/to/train/metadata_HQ.jsonl \
  --neg_meta_path /path/to/train/metadata_with_neg.jsonl \
  --val_hq_meta_path /path/to/val/metadata_HQ.jsonl \
  --val_neg_meta_path /path/to/val/metadata_with_neg.jsonl \
  --yaml_file model_configs/juggernautXL.yaml \
  --batch_size 1 --gpus 1 \
  --log_dir lightning_logs --ckpt_dir checkpoints/training
```

This is a command template with a conservative batch size, not the complete
original training recipe or a promise to reproduce the published scores.
Use `--pretrained_ckpt` only when deliberately initializing from a trained TS
checkpoint. The checkpoint loader accepts both bare state dictionaries and
Lightning `state_dict` containers and checks that all text projections load.

## Metadata format

HQ JSONL rows contain `prompt` and `HQ_path`:

```json
{"prompt": "a street sign", "HQ_path": "images/positive.png"}
```

Negative JSONL rows contain `prompt`, `neg_path`, `OCR` and `SR_OCR`:

```json
{"prompt": "a street sign", "neg_path": "images/negative.png", "OCR": "STOP", "SR_OCR": "ST0P"}
```

Both `HQ_path` and `neg_path` are resolved relative to the directory containing
the HQ metadata file. The loader uses the `P_HQ`, `P_LQ`, `N_HQ`, `N_LQ`
partitions and creates degraded low-quality variants during loading. Inspect
`GLYPHSR/dataloader.py` for the exact filtering and degradation rules.

The historical data-generation code is under `dataset_generater/` (the spelling
is intentional in the existing repository) and
`third_party/make_dataset_with_nunchaku/`. These scripts have their own model,
prompt and environment requirements. The full synthetic corpus and an automated
recreation of its original generation run are not bundled with this update.
Keep train/validation images disjoint and record model, data and seed identities
when reproducing an experiment.
