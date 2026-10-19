# Validation

The public refactor preserves recorded experiment values and separates portable
OCR scoring from the GPU restoration runtime. Evaluation checks run on 19 October
2026 used Python 3.12 and RapidFuzz from the existing CPU validation environment.

| Check | Result |
| --- | --- |
| Full CPU software suite | 85 tests passed with PyTorch 2.8.0+cpu |
| First-party lint and formatting | Ruff checks passed |
| CLI startup | All 11 help paths work without site packages |
| Packaged evidence sizes and SHA-256 hashes | 27 / 27 files verified |
| Complete paired-test CSV against statistics JSON | All 732 records match, including estimates, intervals, adjusted p-values, image counts, and seeds |
| Statistical display coverage | 252 rows across all eight declared families |
| Aggregate and table coverage | 780 aggregate metric entries and 23 LaTeX files |
| OCR protocol regressions | Four tests passed: historical tie handling, distinct Unicode normalization, surplus predictions, and JSONL output compatibility |
| Evidence regression tests | Three tests passed: manifest completeness, full CSV/JSON consistency, and displayed family coverage |
| Evidence location tests | Four tests passed: checkout precedence, wheel-prefix lookup, target-directory lookup, and explicit-root handling |

During packaging, all table data rows were compared with their recorded measured
sources. The current camera-ready presentation changes captions and layout while
preserving those rows. Existing aggregate/statistics JSON files retain their
original bytes. The manifest identifies the measured sources and manuscript commit.

Run the portable checks from a checkout after installing the evaluation extra:

```bash
python -m pip install -e '.[evaluation]'
python -m unittest discover -s tests -p 'test_evaluation.py' -v
python -m unittest discover -s tests -p 'test_results.py' -v
glyph-sr verify
```

The full software test suite additionally uses the runtime and development dependencies:

```bash
python -m pip install -e '.[runtime,evaluation,dev]'
python -m pytest -q tests
```

These are software and evidence-consistency checks. GPU training, full restoration
inference, throughput/memory benchmarking, and new measurements of paper results
were not performed by this refactor. The raw generated images and OCR archives
are separate from the public aggregate package.
