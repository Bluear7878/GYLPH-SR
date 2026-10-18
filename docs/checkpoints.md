# Checkpoints

Use the following components together for the submitted GLYPH-SR configuration.
The TS checkpoint is loaded with the `full_model` policy. Substituting a different
Juggernaut version or loading only text projections is a different configuration.

| Component | Source | Local path under `checkpoints/` |
| --- | --- | --- |
| Juggernaut XL v8 | [Pinned model file](https://huggingface.co/RunDiffusion/Juggernaut-XL-v8/resolve/5c38059bf38387bed17f70e5c1e5c00544caf41e/juggernautXL_v8Rundiffusion.safetensors) | `juggernautXL_v8Rundiffusion.safetensors` |
| SUPIR-v0Q | [Recorded download](https://weights.replicate.delivery/default/SUPIR-v0Q.ckpt), [upstream project](https://github.com/Fanghua-Yu/SUPIR) | `SUPIR-v0Q.ckpt` |
| Trained TS-ControlNet | [Google Drive file](https://drive.google.com/file/d/121NK-KAOzrKPgQ8XGIf57TQ3HqLFM3yb/view) | `TS-ControlNet_cp.ckpt` |
| OCR LoRA configuration | [Google Drive file](https://drive.google.com/file/d/1AvgzyFtdx0Q0K3O21qYaFLl1fb1m08JG/view) | `Llava-next_FT/adapter_config.json` |
| OCR LoRA weights | [Google Drive file](https://drive.google.com/file/d/1m7PIIE-_uScY085a48xKLy-10AdzsvEj/view) | `Llava-next_FT/adapter_model.safetensors` |
| LLaVA-NeXT base | [lmms-lab/llama3-llava-next-8b](https://huggingface.co/lmms-lab/llama3-llava-next-8b) | Hugging Face cache or a local snapshot |

The old README's pretrained-model folder returned HTTP 404 during the October
2026 update. The three individual Drive file pages above were accessible without
sign-in. This is a link check, not a fresh download/hash check of the large files.
Recorded sizes and SHA-256 identities are in [checkpoints.json](checkpoints.json).
After downloading, verify your local copies:

```bash
python tools/verify_release.py --checkpoint-dir ./checkpoints
```

The LLaVA base revision recorded in the review runs is
`6d6dcb8d7948364bebef9c11aba6b0e6c8835391`. For a pinned local snapshot:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "lmms-lab/llama3-llava-next-8b",
    revision="6d6dcb8d7948364bebef9c11aba6b0e6c8835391",
    local_dir="checkpoints/llama3-llava-next-8b",
)
```

Set `GLYPH_LLAVA_MODEL_PATH=./checkpoints/llama3-llava-next-8b` to use it.
The OCR LoRA was trained separately for OCR and then used to provide SR guidance;
`train_GLYPH_SR.py` trains the SR text-control branch, not that OCR adapter.

## Custom locations

All defaults are relative to the repository root. Set these variables before
starting Python, or pass the corresponding inference options where available.

| Variable | Meaning |
| --- | --- |
| `GLYPH_SDXL_CKPT` | Juggernaut XL v8 file, read by the model YAML |
| `GLYPH_SUPIR_Q_CKPT` | SUPIR-v0Q file, read by the model YAML |
| `GLYPH_TS_CKPT` | Trained TS file; also `--GYLPH_pretrained_ckpt` |
| `GLYPH_LLAVA_MODEL_PATH` | LLaVA base directory/model ID; also `--llava_model_path` |
| `GLYPH_LLAVA_ADAPTER_PATH` | OCR adapter directory; also `--VLM_pretrained_ckpt` |
| `GLYPH_CLIP1_PATH` | SDXL CLIP-L directory/model ID |
| `GLYPH_CLIP2_PATH` | OpenCLIP weight file or pretrained tag (`laion2b_s39b_b160k`) |

The LLaVA vision tower follows its model configuration and uses the Hugging Face
cache. The `LLAVA_CLIP_PATH` compatibility constant is not an inference override.
Model downloads retain their upstream terms; this repository does not relicense
them. Load trusted checkpoint files: the released Lightning format uses PyTorch
pickle deserialization.
