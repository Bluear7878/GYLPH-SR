# GYLPH-SR
![image](https://github.com/user-attachments/assets/2eab5b65-82d2-452e-8446-c3f57f482ac2)

GYLPH-SR is a framework for both high-quality super-resolution and high-fidelity text restoration via a VLM-guided generative diffusion model.
It injects explicit textual guidance (OCR strings, spatial coordinates, holistic scene captions) to harmonize readability with visual quality.
It achieves higher OCR metrics than other diffusion and GAN baselines.

## Running model
- Install dependent package
```
conda create -n GYLPH python=3.8 -y
conda activate GYLPH
pip install --upgrade pip
pip install -r requirements.txt
```

- Test model
```
python run_GLYPH_SR.py
```
Edit checkpoints in `CKPT_PTH.py`

- Train model
```
python train_GLYPH_SR.py
```

- Evaluate model
```
python metric/delta_metric.py --ori_dir [original image dataset] --sr_dir [SR image dataset] --res_path "./all_score"
```
