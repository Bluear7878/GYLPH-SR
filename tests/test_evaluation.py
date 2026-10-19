"""Regression checks for the two deliberately distinct OCR protocols."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from glyph_sr.evaluation.ocr import (
    aggregate_counts,
    corrected_word_counts,
    legacy_counts,
    main,
)


class EvaluationTests(unittest.TestCase):
    def test_historical_partial_match_ties_keep_first_candidate(self):
        # Taking the last tied prediction changes the historical precision/recall.
        self.assertEqual(legacy_counts("a b c", "aa ab c"), (1, 2, 0))
        scores = aggregate_counts(
            [{"ground_truth": "a b c", "predicted": "aa ab c"}], legacy_counts
        )
        self.assertAlmostEqual(scores["precision"], 100 / 3)
        self.assertEqual(scores["recall"], 100)
        self.assertEqual(scores["f1"], 50)

    def test_historical_normalization_does_not_apply_nfkc_or_casefold(self):
        for reference, prediction in [("Ａ", "A"), ("Straße", "STRASSE")]:
            with self.subTest(reference=reference):
                self.assertEqual(corrected_word_counts(reference, prediction), (1, 0, 0))
                self.assertEqual(legacy_counts(reference, prediction)[0], 0)

    def test_word_matching_ignores_surplus_predictions_exact_scoring_counts_them(self):
        self.assertEqual(legacy_counts("STOP", "STOP EXTRA"), (1, 0, 0))
        self.assertEqual(corrected_word_counts("STOP", "STOP EXTRA"), (1, 1, 0))

    def test_cli_retains_json_fields_and_excludes_unannotated_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "predictions.jsonl"
            output = Path(temporary) / "scores.json"
            source.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {"filename": "one", "ground_truth": "STOP", "pred_text": "STOP EXTRA"},
                        {"filename": "two", "ground_truth": "A", "pred_text": ""},
                        {"filename": "unannotated", "ground_truth": None, "pred_text": "ignored"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "--input",
                            str(source),
                            "--output",
                            str(output),
                            "--prediction-field",
                            "pred_text",
                        ]
                    ),
                    0,
                )
            scores = json.loads(output.read_text())
            self.assertEqual(
                set(scores),
                {
                    "count",
                    "excluded_unannotated_count",
                    "corrected_word",
                    "legacy_submitted_word",
                    "character_error_rate",
                    "exact_image_accuracy",
                    "per_image",
                },
            )
            self.assertEqual(scores["count"], 2)
            self.assertEqual(scores["excluded_unannotated_count"], 1)
            self.assertEqual(scores["corrected_word"]["f1"], 50)
            self.assertAlmostEqual(scores["legacy_submitted_word"]["f1"], 200 / 3)
            self.assertEqual(scores["character_error_rate"], 120)
            self.assertEqual(scores["exact_image_accuracy"], 0)
            self.assertEqual(len(scores["per_image"]), 2)


if __name__ == "__main__":
    unittest.main()
