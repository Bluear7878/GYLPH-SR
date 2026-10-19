"""Score user-supplied OCR predictions and verify recorded result/checkpoint files."""

import argparse
import collections
import hashlib
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


def _parse_score_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-field", default="predicted")
    return parser.parse_args(argv)


def normalize(text: str) -> str:
    """Normalize exact-word, CER, and exact-transcription inputs."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return " ".join(text.split())


def normalize_legacy(text: str) -> str:
    """Preserve the original word-matching evaluator's preprocessing."""
    return re.sub(r"[^\w\s]", "", text, flags=re.UNICODE).lower().strip()


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
    """Compute the paper's historical word-matching counts."""
    gt_words = normalize_legacy(gt).split()
    pred_words = normalize_legacy(pred).split()
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
        # The historical implementation retains the first prediction on a tie.
        score, j = max(candidates, key=lambda candidate: candidate[0])
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


def score(argv: list[str] | None = None) -> int:
    args = _parse_score_args(argv)
    from rapidfuzz.distance import Levenshtein

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


def resolve_evidence_root(root: Path | None = None) -> Path:
    """Locate evidence in a checkout, a wheel prefix, or a pip --target directory."""
    if root is not None:
        candidates = [Path(root).expanduser().resolve()]
    else:
        source_root = Path(__file__).resolve().parents[1]
        candidates = [
            source_root,
            Path(sys.prefix) / "share" / "glyph_sr",
            source_root / "share" / "glyph_sr",
        ]
    for candidate in dict.fromkeys(candidates):
        if (candidate / "docs" / "results" / "manifest.json").is_file():
            return candidate
    locations = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Evaluation evidence not found in {locations}. "
        "Use --root to select a checkout or installed share/glyph_sr directory."
    )


def check_file(path, expected):
    if not path.is_file():
        return f"missing: {path}"
    if path.stat().st_size != expected["bytes"]:
        return f"size mismatch: {path}"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected["sha256"]:
        return f"SHA-256 mismatch: {path}"
    return None


def verify(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        help="checkout or installation root containing docs/results/ (auto-detected by default)",
    )
    args = parser.parse_args(argv)
    root = resolve_evidence_root(args.root)
    manifest = json.loads((root / "docs/results/manifest.json").read_text())
    items = [(root / item["path"], item) for item in manifest["files"]]
    if args.checkpoint_dir is not None:
        models = json.loads((root / "docs/checkpoints.json").read_text())
        items += [(args.checkpoint_dir / item["path"], item) for item in models["assets"]]
    errors = [error for path, item in items if (error := check_file(path, item))]
    for error in errors:
        print(error)
    print(f"Verified {len(items) - len(errors)}/{len(items)} files")
    return int(bool(errors))
