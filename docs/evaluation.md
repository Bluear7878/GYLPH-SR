# Evaluation and recorded results

GLYPH-SR reports text legibility and perceptual image quality separately.
The [paper](https://openreview.net/forum?id=9CnYnByhvB) defines each experiment's
image pool, degradation, checkpoint, guidance, and evaluator. Comparisons below
refer to the recorded matched runs.

## Score saved OCR predictions

The portable scorer needs Python 3.10+ and the evaluation extra; it loads no
restoration or OCR models:

```bash
python -m pip install -e '.[evaluation]'
glyph-sr score --input predictions.jsonl --output scores.json
```

Each input line aligns an OCR prediction with its image annotation:

```json
{"filename": "example.png", "ground_truth": "STOP", "predicted": "STOP EXTRA"}
```

Use `--prediction-field pred_text` for a different prediction field. Rows with
`ground_truth: null` are excluded and counted. The output retains these fields:

| Field | Definition |
| --- | --- |
| `corrected_word` | Dataset-level exact-word multiset precision, recall, and F1; all unmatched predicted/reference words count as FP/FN |
| `legacy_submitted_word` | Paper word-matching precision, recall, and F1 under the historical matching rule |
| `character_error_rate` | Total character edits / total reference characters after exact-score normalization and removal of spaces; percent, possibly greater than 100 |
| `exact_image_accuracy` | Percentage of images whose complete normalized transcription matches |
| `per_image` | Exact-word counts, character edits, and transcriptions for inspection |

The paper word-matching rule first counts one-to-one exact words as TP. For each
remaining reference word, its most similar unused prediction counts as FP when
`SequenceMatcher` similarity is at least 0.5; otherwise the reference word counts
as FN. Additional unused predictions do not enter these counts. Tied candidates
retain the first prediction, as in the original evaluator. Word-matching
normalization removes non-word/non-whitespace characters, lowercases, and strips
boundary whitespace. It preserves letters, digits, and underscores.

Exact-word scoring uses Unicode NFKC, case folding, removal of non-word symbols,
and collapsed whitespace. Within each image it counts word multiset intersections
as TP and all unmatched predictions/references as FP/FN. Counts are summed across
the dataset before precision, recall, and F1 are calculated. Word order is ignored;
these scores do not measure text localization. Undefined zero-denominator ratios
are assigned zero. CER and exact-transcription accuracy use the exact-score
normalization. Historical word accuracy and 1-NED in the paper are distinct from
exact-transcription accuracy and CER.

The old `tools/score_ocr_predictions.py` command and its JSON field names remain
available. The canonical implementation is `glyph_sr.evaluation.ocr`.

## Sampling controls

By default, `--steps 50` runs 50 alternating scene/text conditioning steps.
The optional compatibility setting `--text-steps` (also `--text_steps`) preserves
the historical split-budget convention. For example, `--steps 50 --text-steps 20`
runs 25 + 10 = 35 total steps, with scene/text conditioning alternating throughout.
Omit that override for the default total-step behavior. CFG defaults to a nominal
4.0-to-7.5 noise-dependent schedule; `--no-linear-cfg` selects a constant scale.

## Matched benchmark comparison

The table gives **measured** mean ± sample standard deviation over diffusion
seeds 1234, 2345, and 3456. All methods use matched LR inputs and target canvases.
GLYPH-SR uses a minimum internal side of 1024 and Wavelet color correction. SVT has
350 images, including 349 with OCR annotations; SCUT-CTW1500 and CUTE80 have 500
and 80 images. OCR values are percentages; MANIQA uses its native 0–1 scale.
ResShift-v3 ×8 uses 2× bicubic pre-upsampling before its released ×4 model.

| Dataset | Scale | Method | OpenOCR WM F1 | OpenOCR exact-word F1 | MANIQA |
| --- | --- | --- | ---: | ---: | ---: |
| SVT | ×4 | GLYPH-SR | 65.67 ± 0.97 | 25.50 ± 0.66 | 0.4422 ± 0.0005 |
| SVT | ×4 | PASD | 13.22 ± 0.89 | 10.10 ± 0.79 | 0.3178 ± 0.0007 |
| SVT | ×4 | ResShift-v3 | 55.85 ± 1.08 | 28.77 ± 0.74 | 0.3632 ± 0.0001 |
| SVT | ×8 | GLYPH-SR | 36.38 ± 0.39 | 13.75 ± 0.23 | 0.4159 ± 0.0010 |
| SVT | ×8 | PASD | 6.90 ± 1.01 | 5.62 ± 0.94 | 0.3007 ± 0.0053 |
| SVT | ×8 | ResShift-v3 | 14.94 ± 0.57 | 8.41 ± 0.40 | 0.3416 ± 0.0005 |
| SCUT-CTW1500 | ×4 | GLYPH-SR | 41.62 ± 0.20 | 29.54 ± 0.35 | 0.6606 ± 0.0037 |
| SCUT-CTW1500 | ×4 | PASD | 19.15 ± 0.33 | 17.15 ± 0.29 | 0.1881 ± 0.0007 |
| SCUT-CTW1500 | ×4 | ResShift-v3 | 31.41 ± 0.15 | 23.75 ± 0.06 | 0.4730 ± 0.0003 |
| SCUT-CTW1500 | ×8 | GLYPH-SR | 8.65 ± 0.02 | 6.48 ± 0.01 | 0.4368 ± 0.0043 |
| SCUT-CTW1500 | ×8 | PASD | 3.68 ± 0.19 | 3.52 ± 0.18 | 0.1347 ± 0.0004 |
| SCUT-CTW1500 | ×8 | ResShift-v3 | 5.36 ± 0.17 | 4.19 ± 0.15 | 0.3811 ± 0.0004 |
| CUTE80 | ×4 | GLYPH-SR | 76.74 ± 0.30 | 48.54 ± 0.58 | 0.4813 ± 0.0009 |
| CUTE80 | ×4 | PASD | 54.93 ± 1.25 | 43.03 ± 0.79 | 0.2742 ± 0.0015 |
| CUTE80 | ×4 | ResShift-v3 | 78.86 ± 0.87 | 54.31 ± 1.44 | 0.3804 ± 0.0009 |
| CUTE80 | ×8 | GLYPH-SR | 66.49 ± 0.91 | 38.77 ± 0.71 | 0.4533 ± 0.0010 |
| CUTE80 | ×8 | PASD | 50.72 ± 1.00 | 40.84 ± 0.79 | 0.2418 ± 0.0021 |
| CUTE80 | ×8 | ResShift-v3 | 60.83 ± 0.57 | 41.40 ± 0.48 | 0.3561 ± 0.0007 |

The complete tables retain all three OCR evaluators and all IQA endpoints:
[exact-word F1](../evaluation/tmlr2026/tables/main_ocr.tex),
[word-matching F1](../evaluation/tmlr2026/tables/main_f1.tex),
[CER](../evaluation/tmlr2026/tables/main_cer.tex),
[exact-transcription accuracy](../evaluation/tmlr2026/tables/main_exact_accuracy.tex),
and [MANIQA/CLIP-IQA/MUSIQ](../evaluation/tmlr2026/tables/main_iqa.tex).
Different datasets, metrics, and scales favor different methods; the tables
include adverse and unresolved results.

## Appendix coverage and statistics

[evaluation/tmlr2026](../evaluation/tmlr2026) contains 780 aggregate metric entries,
all 732 paired tests in eight declared families, 23 LaTeX table files, and a
[CSV of every test](../evaluation/tmlr2026/all_statistics.csv). The 252 displayed
statistical rows are a subset of the 732 tests, with adjustment computed over
each complete family:

| Analysis family | All tests | Displayed comparisons | Measured tables |
| --- | ---: | ---: | --- |
| Main benchmark | 216 | 72 | [Statistics](../evaluation/tmlr2026/tables/statistics_main_all_datasets_scales_endpoints.tex) |
| Common guider | 102 | 42 | [Non-Latin](../evaluation/tmlr2026/tables/common_nonlatin.tex), [English OCR](../evaluation/tmlr2026/tables/common_english_ocr.tex), [English IQA](../evaluation/tmlr2026/tables/common_english_iqa.tex), [statistics](../evaluation/tmlr2026/tables/statistics_common_guider.tex) |
| SVT cues | 54 | 18 | [Scores](../evaluation/tmlr2026/tables/svt_cues.tex), [statistics](../evaluation/tmlr2026/tables/statistics_svt_cues.tex) |
| SVT scheduler | 18 | 6 | [Scores](../evaluation/tmlr2026/tables/svt_scheduler.tex), [statistics](../evaluation/tmlr2026/tables/statistics_svt_scheduler.tex) |
| SVT CFG | 72 | 24 | [Scores](../evaluation/tmlr2026/tables/svt_cfg.tex), [statistics](../evaluation/tmlr2026/tables/statistics_svt_cfg.tex) |
| SVT control strength | 54 | 18 | [Scores](../evaluation/tmlr2026/tables/svt_control.tex), [statistics](../evaluation/tmlr2026/tables/statistics_svt_control.tex) |
| SVT corpus | 72 | 24 | [Scores](../evaluation/tmlr2026/tables/svt_corpus.tex), [statistics](../evaluation/tmlr2026/tables/statistics_svt_corpus.tex) |
| Curved/dense layouts | 144 | 48 | [OCR](../evaluation/tmlr2026/tables/layout_ocr.tex), [IQA](../evaluation/tmlr2026/tables/layout_iqa.tex), [statistics](../evaluation/tmlr2026/tables/statistics_layout_subsets.tex) |

Paired 95% confidence intervals use 10,000 image-cluster bootstrap resamples,
retaining every seed of an image in one cluster. These intervals differ from
sample standard deviations over inference seeds. Holm adjustment applies within
each complete declared family. Differences are comparison minus reference.
English OCR F1 in the displayed statistical tables uses exact-word matching;
non-Latin F1 uses grapheme matching. For English evaluators, machine-readable `*_f1` fields retain word-matching
scores, while `*_strict_f1` fields identify exact-word scores.
See the [analysis plan](../evaluation/tmlr2026/analysis_plan.json) for all metrics,
fixed image memberships, seed sets, and family definitions.

The common-guider study uses fixed Chinese and Hindi crop sets at ×4/×16 and a
matched English control. SVT ablations use 1024/Wavelet settings; the control
sweep uses FP32 denoising and a BF16 VAE. Layout subsets use annotations alone:
curved text requires a 14-vertex polygon with normalized centerline bending at
least 0.05, and dense text requires at least 10 non-ignored instances. Subsets
may overlap. These tables retain their own image pools and protocol captions.
Other paper studies, including native-input evaluation and measured guider
latency, are described in the paper and belong to separate provenance bundles.

## Perceptual scores

The matched runs used `pyiqa==0.1.16`. MANIQA and CLIP-IQA use their native 0–1
scale, while MUSIQ uses its native scale. Older paper tables display some scores
on a 0–100 scale and use a different historical IQA protocol; multiplying by 100
does not establish protocol equivalence. A rerun should record evaluator weight
hashes, image bytes, resize/crop rules, and random seeds alongside package versions.

## Evidence verification

```bash
glyph-sr verify
```

Verification automatically locates evidence in a source checkout or an installed
wheel, including a `pip --target` installation. Use `--root /path/to/checkout`
to select a particular evidence copy. Installed wheel data lives under
`share/glyph_sr/` in its installation directory.

The [manifest](../evaluation/tmlr2026/manifest.json) records size and SHA-256 for
27 evidence files, original measured sources, and the camera-ready manuscript
commit used for table presentation. Table data rows match the recorded measured
sources. Aggregates and statistics retain their original bytes. Three provenance
paths in the packaged analysis plan have their original checkout prefix removed;
source and packaged hashes identify that transformation.

This package provides recorded aggregate evidence. The raw per-image OCR logs,
generated images, evaluator weights, and orchestration environment are not bundled.
No missing historical precision/recall/accuracy values have been inferred or added.
