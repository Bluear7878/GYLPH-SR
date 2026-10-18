# Release validation — 18 October 2026

The repository refresh was checked with a fresh Python 3.12 environment using
PyTorch 2.8.0 CPU wheels and the core dependency pins. The reference GPU recipe
remains Python 3.10 with the CUDA 12.8 wheel pair recorded in the README.

- Core dependencies installed successfully; `pip check` reported no conflicts.
- Model, data-loader, encoder and LLaVA imports passed with network access disabled.
- Ten CPU tests passed: checkpoint containers/key coverage, rejected incompatible
  weights, CLI help without site packages, required inputs, path overrides,
  Pillow resizing, corrected/legacy OCR behavior and JSONL scoring.
- All nine bundled evidence files passed size and SHA-256 checks. Aggregates,
  statistics and table numbers match their recorded source artifacts.
- `CITATION.cff` validated against the official CFF 1.2.0 schema.
- Relative documentation links and changed Python syntax were checked.
- Checkpoint links were checked through public file pages or one-byte range
  requests; the large checkpoint payloads were not downloaded again.

This is software/release validation. GPU training, full restoration inference,
performance benchmarking and remeasurement of the paper's results were not run
for this update. CPU checks do not establish GPU throughput, memory usage or
pixel-identical reproduction. Dependency deprecation notices and optional
video/xformers notices on CPU are retained as upstream compatibility notes.
