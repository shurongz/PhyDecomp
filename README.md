# PELNet
PELNet is a general physics-guided deep learning framework for polarimetric SAR (PolSAR) target decomposition. It supports 3-, 4-, and 6-component decomposition schemes under a unified encoder–MoE–decoder architecture, integrating physical scattering models with data-driven learning.
# PELNet: A General Physics-Embedded Learning Framework for Polarimetric Target Decomposition

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Status-Under%20Review-yellow" />
</p>

> **PELNet** is a general physics-guided deep learning framework for polarimetric SAR (PolSAR) target decomposition. It supports 3-, 4-, and 6-component decomposition schemes under a unified encoder–MoE–decoder architecture, integrating physical scattering models with data-driven learning.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Usage](#usage)
- [Method](#method)
- [Citation](#citation)
- [License](#license)

---

## Overview

PolSAR target decomposition separates the backscattered signal into physically interpretable scattering mechanisms (surface, double-bounce, volume, helix, etc.). Traditional model-based methods rely on fixed physical assumptions, which limits their flexibility across diverse scenes.

**PELNet** addresses this by:
- Unifying 3-, 4-, and 6-component decomposition under one CNN-based autoencoder
- Incorporating a **Mixture-of-Experts (MoE)** bottleneck to adaptively select volume scattering models
- Combining **reconstruction loss**, **reference comparison loss** (against classical decompositions), and **smoothness loss** for physically consistent results
- Supporting optional de-orientation for 6-component decomposition

---

## Repository Structure

```
PELNet/
├── config.py                    # Global configuration (paths, hyperparameters)
├── data_import.py               # Data loading and preprocessing utilities
├── dataset.py                   # PolSAR PyTorch Dataset with patch sampling
├── component.py                 # Physical scattering component models
├── reconstruct.py               # 3/4/6-component reconstruction functions
├── UniversalPolarDecompAE.py    # Main CNN model (Encoder + MoE + Decoder)
├── loss.py                      # Loss functions and reference data loader
├── train.py                     # Training and inference entry point
├── checkpoints/                 # Pretrained model checkpoints
│   ├── checkpoint_3comp.pth
│   ├── checkpoint_4comp.pth
│   └── checkpoint_6comp.pth
├── sample_data/                 # Sample PolSAR data (San Francisco L-band)
│   ├── T11.bin
│   ├── T22.bin
│   ├── T33.bin
│   ├── T12_real.bin / T12_imag.bin
│   ├── T13_real.bin / T13_imag.bin
│   ├── T23_real.bin / T23_imag.bin
│   ├── T11.bin.hdr
│   └── reference/               # Classical decomposition reference results
│       ├── Freeman_Odd.bin / Freeman_Dbl.bin / Freeman_Vol.bin
│       ├── Yamaguchi4_Y4O_*.bin
│       └── Singh_i6SD_*.bin
└── requirements.txt
```

---

## Installation

**Requirements:** Python 3.8+, CUDA-compatible GPU recommended.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/PhyDecomp.git
cd PhyDecomp

# 2. Create a virtual environment (recommended)
conda create -n phydecomp python=3.8
conda activate phydecomp

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
torch>=2.0.0
torchvision
numpy
opencv-python
matplotlib
tqdm
```

---

## Data Preparation

PhyDecomp takes **Coherency Matrix (T3)** data in binary `.bin` format as input, which is the standard output of PolSARpro.

### Input Files

Each scene requires 9 binary files representing the T3 matrix elements:

| File | Description |
|------|-------------|
| `T11.bin` | T11 diagonal element |
| `T22.bin` | T22 diagonal element |
| `T33.bin` | T33 diagonal element |
| `T12_real.bin`, `T12_imag.bin` | Real/imaginary parts of T12 |
| `T13_real.bin`, `T13_imag.bin` | Real/imaginary parts of T13 |
| `T23_real.bin`, `T23_imag.bin` | Real/imaginary parts of T23 |
| `T11.bin.hdr` | Header file with image dimensions |

### Reference Data (Optional)

To use the reference comparison loss, place classical decomposition results (e.g., from PolSARpro) in a `reference/` subdirectory:

```
reference/
├── Freeman_Odd.bin          # For 3-component (Freeman–Durden)
├── Freeman_Dbl.bin
├── Freeman_Vol.bin
├── Yamaguchi4_Y4O_Odd.bin   # For 4-component (Yamaguchi)
├── Yamaguchi4_Y4O_Dbl.bin
├── Yamaguchi4_Y4O_Vol.bin
├── Yamaguchi4_Y4O_Hlx.bin
├── Singh_i6SD_Odd.bin       # For 6-component (Singh i6SD)
├── Singh_i6SD_Dbl.bin
├── Singh_i6SD_Vol.bin
├── Singh_i6SD_Hlx.bin
├── Singh_i6SD_OD.bin
└── Singh_i6SD_CD.bin
```

> If reference data is not available, the framework runs in **unsupervised mode** using only reconstruction and smoothness losses.

### Update Paths in `config.py`

```python
DATA_DIR       = '/path/to/your/data'
OUTPUT_DIR     = '/path/to/output'
CHECKPOINT_DIR = '/path/to/checkpoints'
REFERENCE_DIR  = '/path/to/your/data/reference'
```

## Usage

### Training

```bash
python train.py \
  --model_type 4comp \
  --epochs 100 \
  --batch_size 8 \
  --patch_size 128 \
  --lr 1e-4 \
  --recon_lambda 0.5 \
  --reference_lambda 1 \
  --smooth_lambda 0.3
```

### Inference Only

```bash
python train.py \
  --model_type 4comp \
  --inference
```

### Use a Specific ROI

```bash
python train.py \
  --model_type 6comp \
  --roi 0 512 0 512
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | `6comp` | Decomposition type: `3comp`, `4comp`, `6comp` |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `8` | Batch size |
| `--patch_size` | `128` | Training patch size |
| `--lr` | `1e-4` | Learning rate |
| `--recon_lambda` | `0.5` | Reconstruction loss weight |
| `--reference_lambda` | `1` | Reference comparison loss weight |
| `--smooth_lambda` | `0.3` | TV smoothness loss weight |
| `--reference_dir` | (config) | Path to classical decomposition references |
| `--inference` | `False` | Run inference only (no training) |
| `--roi` | `None` | Region of interest: `row_start row_end col_start col_end` |

### Output

Results are saved to `OUTPUT_DIR` as `.bin` files compatible with PolSARpro:

```
OUTPUT_DIR/
├── 4comp_ps.bin      # Surface scattering power
├── 4comp_pd.bin      # Double-bounce scattering power
├── 4comp_pv.bin      # Volume scattering power
├── 4comp_ph.bin      # Helix scattering power (4/6-comp only)
├── 4comp_beta_real.bin
├── 4comp_beta_imag.bin
├── 4comp_alpha_real.bin
└── 4comp_alpha_imag.bin
```

---


### Supported Decomposition Schemes

| Type | Components | Volume Models | Reference |
|------|-----------|---------------|-----------|
| 3-comp | Ps, Pd, Pv | 1 (standard) | Freeman–Durden |
| 4-comp | Ps, Pd, Pv, Ph | 3 (MoE) | Yamaguchi et al. |
| 6-comp | Ps, Pd, Pv, Ph, Pod, Pcd | 4 (MoE) | Singh et al. (i6SD) |



---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- PolSAR data processed using [PolSARpro](https://www.ietr.fr/polsarpro-bio/)
- Physical scattering models based on Freeman–Durden, Yamaguchi, and Singh decomposition frameworks
