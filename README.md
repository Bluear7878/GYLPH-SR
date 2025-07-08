# GLYPH-SR
## Abstract
In real-world applications, image super-resolution (SR) underpins tasks such as autonomous driving, smart-city surveillance, and optical character recognition (OCR). Unfortunately, existing SR methods often fail to accurately recover text or chracters in images, as they primarily optimize SR quality using distortion or perceptual metrics (e.g. MANIQA, CLIP-IQA, MUSIQ) without explicitly considering text recovery. Achieving both high-quality SR and high-fidelity text restoration simultaneously remains technically challenging. To break through these limitations, we propose GLYPH-SR, a novel and high-performing framework for both image SR and text restoration via vision-language model (VLM)-guided generative diffusion model. To reconcile these conflicting goals, GLYPH-SR injects explicit textual guidance-token-wise OCR strings, their spatial coordinates, and a holistic scene caption-via a vision-language prompt pipeline. These guidance cues drive a _Text-SR Fusion ControlNet_, trained on a four-partition synthetic dataset, disentangling text-oriented from image-oriented guidance. A binary _ping-pong scheduler_ then alternates guidance cues along the diffusion trajectory, harmonizing readability with visual realism. Comprehensive experiments on SVT, SCUT-CTW 1500, and CUTE80 show that GLYPH-SR improves OCR _F<sub>1</sub>_ score by up to **12.0pp** over strong diffusion and GAN baselines, while matching or surpassing their perceptual metrics (MANIQA, CLIP-IQA, MUSIQ). Qualitative analyses confirm robustness even under severe LR conditions, where competing models collapse. Simulation code, pretrained models, datasets, and evaluation scripts are provided as supplementary materials.

## Architecture
![Architecture](https://github.com/user-attachments/assets/9b7f2f98-caec-4c75-a2a1-c368b38f435e)

## Results
### Visual Results
![Result](https://github.com/user-attachments/assets/4023afef-5311-4769-b7da-fcbea46bfefe)

### Quantitative Comparison


<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th rowspan="2">Model</th>
      <th colspan="3" style="text-align: center">SR metric</th>
      <th colspan="3" style="text-align: center">OCR metric(<i>F<sub>1</sub></i>)</th>
    </tr>
    <tr>
      <th style="width:100px; text-align: center;">MANIQA</th>
      <th style="width:100px; text-align: center;">CLIP-IQA</th>
      <th style="width:100px; text-align: center;">MUSIQ</th>
      <th style="width:100px; text-align: center;">OpenOCR</th>
      <th style="width:100px; text-align: center;">GOT-OCR</th>
      <th style="width:100px; text-align: center;">LLaVA-NeXT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td rowspan="6" style="text-align: center;">CUTE80(x4)</td>
        <td style="text-align: center;">BSRGAN</td>
        <td style="text-align: center;">36.89</td>
        <td style="text-align: center;">44.44</td>
        <td style="text-align: center;">55.22</td>
        <td style="text-align: center;">63.66</td>
        <td style="text-align: center;">48.38</td>
        <td style="text-align: center;">80.19</td>
    </tr>
    <tr>
        <td style="text-align: center;">Real-ESRGAN</td>
        <td style="text-align: center;">45.11</td>
        <td style="text-align: center;">69.21</td>
        <td style="text-align: center;">60.43</td>
        <td style="text-align: center;">66.84</td>
        <td style="text-align: center;">50.15</td>
        <td style="text-align: center;">79.34</td>
    </tr>
    <tr>
        <td style="text-align: center;">StableSR</td>
        <td style="text-align: center;">36.46</td>
        <td style="text-align: center;">60.84</td>
        <td style="text-align: center;">55.62</td>
        <td style="color: blue; text-align: center;"><b>67.53</b></td>
        <td style="color: red; text-align: center;"><b>53.14</b></td>
        <td style="color: red; text-align: center;"><b>86.02</b></td>
    </tr>
    <tr>
        <td style="text-align: center;">DiffBIR</td>
        <td style="color: blue; text-align: center;"><b>63.16</b></td>
        <td style="color: red; text-align: center;"><b>86.14</b></td>
        <td style="color: blue; text-align: center;"><b>66.78</b></td>
        <td style="text-align: center;">66.49</td>
        <td style="text-align: center;">45.65</td>
        <td style="color: blue; text-align: center;"><b>80.74</b></td>
    </tr>
    <tr>
        <td style="text-align: center;">SUPIR</td>
        <td style="text-align: center;">53.23</td>
        <td style="text-align: center;">76.41</td>
        <td style="text-align: center;">65.22</td>
        <td style="text-align: center;">64.74</td>
        <td style="color: red; text-align: center;"><b>53.14</b></td>
        <td style="text-align: center;">79.34</td>
    </tr>
    <tr>
        <td style="text-align: center;">GLYPH-SR</td>
        <td style="color: red; text-align: center;"><b>64.49</b></td>
        <td style="color: blue; text-align: center;"><b>81.23</b></td>
        <td style="color: red; text-align: center;"><b>69.67</b></td>
        <td style="color: red; text-align: center;"><b>68.54</b></td>
        <td style="color: blue; text-align: center;"><b>51.01</b></td>
        <td style="text-align: center;">80.47</td>
    </tr>
</tbody>
<tr>
<td colspan="8" style="border-bottom: 2px solid #888;"></td>
</tr>
  <tbody>
    <tr>
        <td rowspan="6" style="text-align: center;">CUTE80(x6)</td>
        <td style="text-align: center;">BSRGAN</td>
        <td style="text-align: center;">31.39</td>
        <td style="text-align: center;">43.95</td>
        <td style="text-align: center;">43.58</td>
        <td style="text-align: center;">48.82</td>
        <td style="text-align: center;">41.36</td>
        <td style="text-align: center;">69.21</td>
    </tr>
    <tr>
        <td style="text-align: center;">Real-ESRGAN</td>
        <td style="text-align: center;">45.84</td>
        <td style="text-align: center;">69.59</td>
        <td style="text-align: center;">56.90</td>
        <td style="text-align: center;">47.93</td>
        <td style="text-align: center;">35.26</td>
        <td style="text-align: center;">67.53</td>
    </tr>
    <tr>
        <td style="text-align: center;">StableSR</td>
        <td style="text-align: center;">24.38</td>
        <td style="text-align: center;">29.66</td>
        <td style="text-align: center;">39.45</td>
        <td style="text-align: center;">50.15</td>
        <td style="text-align: center;">38.87</td>
        <td style="text-align: center;">69.87</td>
    </tr>
    <tr>
        <td style="text-align: center;">DiffBIR</td>
        <td style="color: blue; text-align: center;"><b>59.75</b></td>
        <td style="color: red; text-align: center;"><b>82.71</b></td>
        <td style="color: blue; text-align: center;"><b>61.92</b></td>
        <td style="text-align: center;">52.30</td>
        <td style="text-align: center;">34.73</td>
        <td style="text-align: center;">67.53</td>
    </tr>
    <tr>
        <td style="text-align: center;">SUPIR</td>
        <td style="text-align: center;">49.87</td>
        <td style="text-align: center;">73.23</td>
        <td style="text-align: center;">61.11</td>
        <td style="color: blue; text-align: center;"><b>56.82</b></td>
        <td style="color: blue; text-align: center;"><b>42.33</b></td>
        <td style="color: blue; text-align: center;"><b>71.82</b></td>
    </tr>
    <tr>
        <td style="text-align: center;">GLYPH-SR</td>
        <td style="color: red; text-align: center;"><b>63.21</b></td>
        <td style="color: blue; text-align: center;"><b>81.17</b></td>
        <td style="color: red; text-align: center;"><b>68.95</b></td>
        <td style="color: red; text-align: center;"><b>60.70</b></td>
        <td style="color: red; text-align: center;"><b>45.65</b></td>
        <td style="color: red; text-align: center;"><b>74.33</b></td>
    </tr>
</tbody>
<tr>
<td colspan="8" style="border-bottom: 2px solid #888;"></td>
</tr>
  <tbody>
    <tr>
        <td rowspan="6" style="text-align: center;">CUTE80(x8)</td>
        <td style="text-align: center;">BSRGAN</td>
        <td style="text-align: center;">33.20</td>
        <td style="text-align: center;">38.41</td>
        <td style="text-align: center;">41.73</td>
        <td style="text-align: center;">36.31</td>
        <td style="text-align: center;">29.80</td>
        <td style="text-align: center;">51.87</td>
    </tr>
    <tr>
        <td style="text-align: center;">Real-ESRGAN</td>
        <td style="text-align: center;">46.44</td>
        <td style="text-align: center;">66.85</td>
        <td style="text-align: center;">54.75</td>
        <td style="text-align: center;">32.57</td>
        <td style="text-align: center;">24.57</td>
        <td style="text-align: center;">50.15</td>
    </tr>
    <tr>
        <td style="text-align: center;">StableSR</td>
        <td style="text-align: center;">19.61</td>
        <td style="text-align: center;">16.45</td>
        <td style="text-align: center;">28.38</td>
        <td style="text-align: center;">32.57</td>
        <td style="text-align: center;">23.97</td>
        <td style="text-align: center;">55.21</td>
    </tr>
    <tr>
        <td style="text-align: center;">DiffBIR</td>
        <td style="color: blue; text-align: center;"><b>54.68</b></td>
        <td style="color: blue; text-align: center;"><b>74.15</b></td>
        <td style="text-align: center;">54.18</td>
        <td style="color: blue; text-align: center;"><b>38.87</b></td>
        <td style="color: blue; text-align: center;"><b>33.12</b></td>
        <td style="color: blue; text-align: center;"><b>55.98</b></td>
    </tr>
    <tr>
        <td style="text-align: center;">SUPIR</td>
        <td style="text-align: center;">49.40</td>
        <td style="text-align: center;">70.92</td>
        <td style="color: blue; text-align: center;"><b>58.84</b></td>
        <td style="text-align: center;">34.19</td>
        <td style="text-align: center;">24.57</td>
        <td style="text-align: center;">55.58</td>
    </tr>
    <tr>
        <td style="text-align: center;">GLYPH-SR</td>
        <td style="color: red; text-align: center;"><b>62.01</b></td>
        <td style="color: red; text-align: center;"><b>79.69</b></td>
        <td style="color: red; text-align: center;"><b>67.92</b></td>
        <td style="color: red; text-align: center;"><b>48.82</b></td>
        <td style="color: red; text-align: center;"><b>38.36</b></td>
        <td style="color: red; text-align: center;"><b>62.20</b></td>
    </tr>
</tbody>
</table>

## Installation
```
conda create -n GYLPH python=3.8 -y
conda activate GYLPH
pip install --upgrade pip
pip install -r requirements.txt
```
## Downloading and Setup
[Download pretrained_model.zip (Google Drive)](https://drive.google.com/drive/folders/1hrZ5jRbVLcRSFpbL-uPxe9iLddylAFgk)
```
GLYPH-SR/
├── CKPT_PTH/
│   ├── Llava-next/
│   ├── GLYPH-SR-v0F.ckpt
│   ├── GLYPH-SR-v0Q.ckpt
│   ├── TS-ControlNet_cp.ckpt
│   └── ...
├── CKPT_PTH.py
├── README.md
├── run_GLYPH_SR.py
├── train_GLYPH_SR.py
└── ...
```
Edit Custom Path for Checkpoints
```
[CKPT_PTH.py] -> LLAVA_CLIP_PATH, LLAVA_MODEL_PATH, FINE_TUNED_LLAVA_MODEL_PATH, SDXL_CLIP1_PATH, SDXL_CLIP2_CKPT_PTH
```

## Running model
### Test model
```
python run_GLYPH_SR.py
```

Edit Custom Path : `img_path`, `save_dir`, `GLYPH_pretrained_ckpt`, `VLM_pretrained_ckpt`

### Train model
```
python train_GLYPH_SR.py
```
Edit Custom Path : `hq_meta_path`, `neg_meta_path`, `val_hp_meta_path`, `val_hq_meta_path`

### Evaluate model
```
python metric/delta_metric.py --ori_dir [original image path] --sr_dir [SR image path] --res_path "./all_score"
```
