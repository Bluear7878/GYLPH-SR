#!/usr/bin/env python3
"""Score OCR predictions with corrected and legacy GLYPH-SR metrics."""

from __future__ import annotations

import argparse
import collections
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from rapidfuzz.distance import Levenshtein


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-field", default="predicted")
    return parser.parse_args()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return " ".join(text.split())


def safe_ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def corrected_word_counts(gt: str, pred: str) -> tuple[int, int, int]:
    gt_counter = collections.Counter(normalize(gt).split())
    pred_counter = collections.Counter(normalize(pred).split())
    tp = sum((gt_counter & pred_counter).values())
    fn = sum((gt_counter - pred_counter).values())
    fp = sum((pred_counter - gt_counter).values())
    return tp, fp, fn


def legacy_counts(gt: str, pred: str) -> tuple[int, int, int]:
    """Replicate the submitted repository's unusual word-count rule."""
    gt_words = normalize(gt).split()
    pred_words = normalize(pred).split()
    used_pred = set()
    used_gt = set()
    tp = fp = fn = 0
    for i, gt_word in enumerate(gt_words):
        for j, pred_word in enumerate(pred_words):
            if i not in used_gt and j not in used_pred and gt_word == pred_word:
                tp += 1
                used_gt.add(i)
                used_pred.add(j)
                break
    for i, gt_word in enumerate(gt_words):
        if i in used_gt:
            continue
        candidates = [
            (SequenceMatcher(None, gt_word, word).ratio(), j)
            for j, word in enumerate(pred_words)
            if j not in used_pred
        ]
        if not candidates:
            fn += 1
            continue
        score, j = max(candidates)
        if score >= 0.5:
            fp += 1
            used_pred.add(j)
        else:
            fn += 1
    return tp, fp, fn


def aggregate_counts(rows: list[dict], counter) -> dict:
    tp = fp = fn = 0
    for row in rows:
        a, b, c = counter(row["ground_truth"], row["predicted"])
        tp += a
        fp += b
        fn += c
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": safe_ratio(2 * precision * recall, precision + recall) * 100,
    }


def main() -> int:
    args = parse_args()
    all_rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in all_rows:
        if args.prediction_field not in row:
            raise ValueError(
                f"missing prediction field {args.prediction_field!r} for {row.get('filename')}"
            )
        row["predicted"] = row[args.prediction_field]
    rows = [row for row in all_rows if row.get("ground_truth") is not None]
    if not rows:
        raise ValueError("no annotated rows to score")
    per_image = []
    total_edits = total_chars = 0
    exact = 0
    for row in rows:
        gt = normalize(row["ground_truth"])
        pred = normalize(row["predicted"])
        gt_chars = gt.replace(" ", "")
        pred_chars = pred.replace(" ", "")
        distance = Levenshtein.distance(gt_chars, pred_chars)
        total_edits += distance
        total_chars += len(gt_chars)
        exact += int(gt == pred)
        tp, fp, fn = corrected_word_counts(gt, pred)
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        per_image.append(
            {
                "filename": row["filename"],
                "ground_truth": row["ground_truth"],
                "predicted": row["predicted"],
                "word_tp": tp,
                "word_fp": fp,
                "word_fn": fn,
                "word_f1": safe_ratio(2 * precision * recall, precision + recall) * 100,
                "char_edits": distance,
                "char_count": len(gt_chars),
                "exact": gt == pred,
            }
        )

    payload = {
        "count": len(rows),
        "excluded_unannotated_count": len(all_rows) - len(rows),
        "corrected_word": aggregate_counts(rows, corrected_word_counts),
        "legacy_submitted_word": aggregate_counts(rows, legacy_counts),
        "character_error_rate": safe_ratio(total_edits, total_chars) * 100,
        "exact_image_accuracy": safe_ratio(exact, len(rows)) * 100,
        "per_image": per_image,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "per_image"},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
