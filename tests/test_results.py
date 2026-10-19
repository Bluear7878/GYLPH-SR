"""Check the completeness and internal consistency of recorded evidence."""

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glyph_sr.evaluation import verify
from glyph_sr.evaluation.verify import check_file

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evaluation" / "tmlr2026"


class EvidenceLocationTests(unittest.TestCase):
    def write_bundle(self, root):
        evidence = root / "evaluation" / "tmlr2026"
        evidence.mkdir(parents=True)
        payload = b"recorded evidence\n"
        (evidence / "example.json").write_bytes(payload)
        (evidence / "manifest.json").write_text(
            json.dumps(
                {
                    "files": [
                        {
                            "path": "evaluation/tmlr2026/example.json",
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
            with patch.object(verify, "__file__", str(checkout / "glyph_sr/evaluation/verify.py")):
                with patch.object(verify.sys, "prefix", str(prefix)):
                    self.assertEqual(verify.resolve_evidence_root(), checkout)

    def test_wheel_evidence_is_loaded_from_installation_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            installed = prefix / "lib/python3.12/site-packages"
            evidence = prefix / "share" / "glyph_sr"
            self.write_bundle(evidence)
            with patch.object(verify, "__file__", str(installed / "glyph_sr/evaluation/verify.py")):
                with patch.object(verify.sys, "prefix", str(prefix)):
                    self.assertEqual(verify.resolve_evidence_root(), evidence)
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        self.assertEqual(verify.main([]), 0)
                    self.assertIn("Verified 1/1 files", output.getvalue())

    def test_target_install_resolves_its_share_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            target = folder / "target"
            evidence = target / "share" / "glyph_sr"
            self.write_bundle(evidence)
            with patch.object(verify, "__file__", str(target / "glyph_sr/evaluation/verify.py")):
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
        self.assertEqual(len(listed), 27)
        for item in manifest["files"]:
            with self.subTest(path=item["path"]):
                self.assertIsNone(check_file(ROOT / item["path"], item))
        self.assertEqual(
            manifest["manuscript_commit"],
            "eefbd46685e464c411aaad18ccb97cc2cec2d9c7",
        )

    def test_complete_statistics_csv_matches_recorded_json(self):
        statistics = json.loads((EVIDENCE / "statistics.json").read_text())
        aggregates = json.loads((EVIDENCE / "aggregates.json").read_text())
        plan = json.loads((EVIDENCE / "analysis_plan.json").read_text())
        self.assertEqual(len(aggregates["results"]), 780)
        self.assertEqual(len(statistics["families"]), 8)
        self.assertEqual(sum(family["test_count"] for family in statistics["families"]), 732)
        self.assertEqual(
            {family["id"]: family["test_count"] for family in statistics["families"]},
            {family["id"]: family["test_count"] for family in plan["families"]},
        )
        expected = {}
        fields = ("dataset", "scale", "subset", "comparison", "reference", "metric")
        for family in statistics["families"]:
            self.assertEqual(len(family["results"]), family["test_count"])
            for result in family["results"]:
                key = (family["id"], *(result.get(field) or "" for field in fields))
                self.assertNotIn(key, expected)
                expected[key] = result
        with (EVIDENCE / "all_statistics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 732)
        observed = set()
        for row in rows:
            key = (row["family"], *(row[field] for field in fields))
            self.assertNotIn(key, observed)
            observed.add(key)
            result = expected[key]
            for field in (
                "mean_difference",
                "ci_low",
                "ci_high",
                "two_sided_p",
                "holm_adjusted_p",
            ):
                self.assertEqual(float(row[field]), result[field])
            self.assertEqual(int(row["image_clusters"]), result["image_clusters"])
            self.assertEqual(json.loads(row["paired_seeds"]), result["paired_seeds"])
        self.assertEqual(observed, set(expected))

    def test_displayed_statistics_cover_all_eight_declared_families(self):
        expected_rows = {
            "common_guider": 42,
            "layout_subsets": 48,
            "main_all_datasets_scales_endpoints": 72,
            "svt_cfg": 24,
            "svt_control": 18,
            "svt_corpus": 24,
            "svt_cues": 18,
            "svt_scheduler": 6,
        }
        tables = list((EVIDENCE / "tables").glob("*.tex"))
        self.assertEqual(len(tables), 23)
        total = 0
        for family, count in expected_rows.items():
            table = EVIDENCE / "tables" / f"statistics_{family}.tex"
            rows = [
                line
                for line in table.read_text().splitlines()
                if " & " in line and not line.startswith("Comparison &")
            ]
            self.assertEqual(len(rows), count, family)
            total += len(rows)
        self.assertEqual(total, 252)


if __name__ == "__main__":
    unittest.main()
