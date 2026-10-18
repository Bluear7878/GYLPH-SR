# Evaluation and recorded results

GLYPH-SR evaluates two separate axes: text legibility and perceptual image quality.
Use the [paper](https://openreview.net/forum?id=9CnYnByhvB) for the full comparison
and limitations. Scores depend on the dataset, degradation, checkpoint, guidance,
seed, preprocessing and evaluator; they do not establish universal superiority.

## OCR from saved predictions

The portable scorer requires Python 3.10+ and RapidFuzz; it does not load models.

```bash
python -m pip install -r requirements/evaluation.txt
python tools/score_ocr_predictions.py \
  --input predictions.jsonl \
  --output scores.json
```

Each input line contains an image identifier, annotation and OCR prediction:

```json
{"filename": "example.png", "ground_truth": "STOP", "predicted": "STOP EXTRA"}
```

This example is synthetic. Use `--prediction-field pred_text` when your input
uses that field name. Predictions must already be aligned with the correct
image annotations. Rows with `ground_truth: null` are excluded and counted.

| Output | Definition |
| --- | --- |
| `corrected_word` | Micro precision/recall/F1 from exact word multiset counts; extra predicted words are false positives and missing words are false negatives |
| `legacy_submitted_word` | Historical partial-word matching rule; leftover predicted words are not fully penalized |
| `character_error_rate` | Total character edits divided by total GT characters, after normalization and removal of spaces; may exceed 100% |
| `exact_image_accuracy` | Fraction of images with an exactly matching normalized transcription |
| `per_image` | Counts, edit distances and predictions for inspection |

All four score groups use a 0–100 scale. Normalization applies Unicode NFKC,
case folding, punctuation removal and whitespace normalization. Digits are
retained. Exact-word matching ignores word order and is not a spatial text
detection metric. The legacy compatibility calculation is not a guarantee of
bit-for-bit reproduction of every historical evaluator's preprocessing.

The older `metric/delta_metric.py` remains available for provenance. Its word
matcher is nonstandard: a partially matching GT word can count as FP rather than
FN, and unmatched extra predictions are ignored. Do not label those legacy
scores as conventional exact-word F1 or mix them with corrected scores.

## Perceptual scores

The matched review runs used `pyiqa==0.1.16`. Record evaluator weight hashes,
input bytes, resize/crop rules and random seeds as well as the package version.
MANIQA and CLIP-IQA in the matched artifacts use their native 0–1 scale; MUSIQ
uses its native scale. Older paper tables displayed some scores on 0–100 scales.
The historical IQA checkpoint/crop protocol was not fully recovered, so scaling
alone does not make the two sets directly comparable.

## Existing matched review artifacts

[evaluation/tmlr2026](../evaluation/tmlr2026) contains the recorded September 2026
aggregates, statistics, analysis plan and selected paper tables. These files are
copied without changing their numerical values. The analysis plan’s original
checkout prefix was removed from three provenance paths; its settings are
unchanged and both source and packaged hashes are recorded. Verify their recorded hashes:

```bash
python tools/verify_release.py
```

There are 780 aggregate metric entries and 732 paired tests across eight
families. The main comparison uses GLYPH-SR, PASD and ResShift-v3, three seeds
(1234, 2345, 3456), SVT/SCUT-CTW1500/CUTE80 and ×4/×8. The artifact fields preserve
sample counts and conditions, including null and adverse results.

- `*_f1` fields are **legacy** scores; `*_strict_f1` fields are corrected scores.
- `seed_sample_std` describes variation over seeds.
- Paired 95% confidence intervals use 10,000 image-cluster bootstrap resamples,
  retaining all seeds of an image together. They are not seed standard deviations.
- Holm adjustment applies within the declared analysis families, including all
  216 main-comparison tests. See the included analysis plan for definitions.

These artifacts do not replace the original single-run tables. Raw per-image
OCR logs, generated images and the full orchestration environment are not bundled
in this code update, so this package alone is not a complete rerun of every table.
No hypothetical or inferred missing metrics are included in the public results.
