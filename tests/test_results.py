"""Check the completeness and internal consistency of recorded evidence."""

import contextlib
import hashlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glyph_sr import evaluation as verify
from glyph_sr.evaluation import check_file

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "results"


class EvidenceLocationTests(unittest.TestCase):
    def write_bundle(self, root):
        evidence = root / "docs" / "results"
        evidence.mkdir(parents=True)
        payload = b"recorded evidence\n"
        (evidence / "example.json").write_bytes(payload)
        (evidence / "manifest.json").write_text(
            json.dumps(
                {
                    "files": [
                        {
                            "path": "docs/results/example.json",
                            "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    ]
                }
            )
        )

    def test_checkout_evidence_precedes_installed_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkout, prefix = folder / "checkout", folder / "prefix"
            self.write_bundle(checkout)
            self.write_bundle(prefix / "share" / "glyph_sr")
            with patch.object(verify, "__file__", str(checkout / "glyph_sr/evaluation.py")):
                with patch.object(verify.sys, "prefix", str(prefix)):
                    self.assertEqual(verify.resolve_evidence_root(), checkout)

    def test_wheel_evidence_is_loaded_from_installation_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            installed = prefix / "lib/python3.12/site-packages"
            evidence = prefix / "share" / "glyph_sr"
            self.write_bundle(evidence)
            with patch.object(verify, "__file__", str(installed / "glyph_sr/evaluation.py")):
                with patch.object(verify.sys, "prefix", str(prefix)):
                    self.assertEqual(verify.resolve_evidence_root(), evidence)
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        self.assertEqual(verify.verify([]), 0)
                    self.assertIn("Verified 1/1 files", output.getvalue())

    def test_target_install_resolves_its_share_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            target = folder / "target"
            evidence = target / "share" / "glyph_sr"
            self.write_bundle(evidence)
            with patch.object(verify, "__file__", str(target / "glyph_sr/evaluation.py")):
                with patch.object(verify.sys, "prefix", str(folder / "unrelated-prefix")):
                    self.assertEqual(verify.resolve_evidence_root(), evidence)

    def test_explicit_root_is_honored_without_falling_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.write_bundle(folder / "selected")
            self.assertEqual(verify.resolve_evidence_root(folder / "selected"), folder / "selected")
            with self.assertRaisesRegex(FileNotFoundError, "Use --root"):
                verify.resolve_evidence_root(folder / "missing")


class RecordedResultsTests(unittest.TestCase):
    def test_manifest_covers_every_packaged_evidence_file(self):
        manifest = json.loads((EVIDENCE / "manifest.json").read_text())
        listed = {ROOT / item["path"] for item in manifest["files"]}
        actual = {path for path in EVIDENCE.rglob("*") if path.is_file()}
        actual.remove(EVIDENCE / "manifest.json")
        self.assertEqual(listed, actual)
        self.assertEqual(len(listed), 3)
        for item in manifest["files"]:
            with self.subTest(path=item["path"]):
                self.assertIsNone(check_file(ROOT / item["path"], item))

    def test_recorded_measurements_retain_their_original_bytes(self):
        originals = {
            "aggregates.json": "385d3f45d2efb0da26ae0e3b1d13d4128f0765aefdfb865875e43aedbc5974eb",
            "statistics.json": "9916fe82058a547b21221db0ba96a5fd8fcf8b14442a772c7dbca1993bc82aee",
        }
        for name, expected in originals.items():
            with self.subTest(file=name):
                self.assertEqual(
                    hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest(), expected
                )

    def test_all_declared_tests_and_aggregate_records_are_preserved(self):
        statistics = json.loads((EVIDENCE / "statistics.json").read_text())
        aggregates = json.loads((EVIDENCE / "aggregates.json").read_text())
        plan = json.loads((EVIDENCE / "analysis_plan.json").read_text())
        manifest = json.loads((EVIDENCE / "manifest.json").read_text())
        self.assertEqual(len(aggregates["results"]), 780)
        self.assertEqual(len(statistics["families"]), 8)
        self.assertEqual(sum(family["test_count"] for family in statistics["families"]), 732)
        self.assertEqual(
            {family["id"]: family["test_count"] for family in statistics["families"]},
            {family["id"]: family["test_count"] for family in plan["families"]},
        )
        self.assertEqual(
            manifest["coverage"],
            {
                "aggregate_metric_entries": 780,
                "paired_tests": 732,
                "analysis_families": 8,
            },
        )
        for family in statistics["families"]:
            self.assertEqual(len(family["results"]), family["test_count"])
            self.assertEqual(
                len({row["test_id"] for row in family["results"]}), family["test_count"]
            )

    def test_plan_keeps_cohort_counts_without_shipping_image_identifiers(self):
        plan = json.loads((EVIDENCE / "analysis_plan.json").read_text())
        cohorts = [
            *plan["registry"],
            *plan["precision_amendment"]["archived_original_control_cells"],
        ]
        self.assertEqual(len(cohorts), 53)
        for cohort in cohorts:
            images = cohort["image_count_by_seed"]
            ocr_images = cohort["ocr_image_count_by_seed"]
            self.assertEqual(set(images), set(ocr_images))
            for seed, count in images.items():
                self.assertGreater(count, 0)
                self.assertLessEqual(ocr_images[seed], count)
        self.assertEqual(sum(sum(c["image_count_by_seed"].values()) for c in cohorts), 26070)
        self.assertEqual(sum(sum(c["ocr_image_count_by_seed"].values()) for c in cohorts), 26029)

    def test_public_result_json_contains_no_per_image_material(self):
        forbidden = {
            "filenames_by_seed",
            "ocr_filenames_by_seed",
            "filename",
            "filenames",
            "ground_truth",
            "predicted",
            "transcription",
            "polygons",
            "bbox",
        }

        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)
            elif isinstance(value, str):
                self.assertIsNone(
                    re.search(r"\.(?:jpe?g|png|webp|bmp|tiff)(?:$|[?#])", value, re.I)
                )

        for path in EVIDENCE.glob("*.json"):
            with self.subTest(file=path.name):
                inspect(json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
