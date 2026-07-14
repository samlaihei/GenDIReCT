# GenDIReCT

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

**GenDIReCT** (*Generative Deep learning for Interferometric Reconstruction of Closure invariant Textures*) is a deep learning framework for Very Long Baseline Interferometry (VLBI) radio interferometric imaging, specifically designed for Event Horizon Telescope (EHT) observations.

## Overview

GenDIReCT combines generative machine learning techniques with closure invariants to reconstruct high-fidelity astronomical images from sparse interferometric data. The method is particularly effective for imaging black holes and other compact astrophysical objects observed by the EHT.

### Key Features

- **Closure Invariant Processing**: Utilizes closure invariants for robust image reconstruction
- **Generative Models**: Implements conditional latent diffusion models for high-quality generative image synthesis
- **EHT**: Designed for Event Horizon Telescope datasets
- **GPU Acceleration**: CUDA/MPS support for fast inference

## Architecture

GenDIReCT employs a multi-stage architecture:

1. **Closure Invariant Computation**: Transforms interferometric data into closure invariants
2. **Generative Model**: Uses latent diffusion models conditioned on closure invariants
3. **Autoencoder**: Encoder-decoder architecture for latent space representation
4. **Convolutional Refinement**: Final CNN-based refinement for high-quality output

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Environment Setup
Tested on python==3.11.4.

1. Clone the repository:
```bash
git clone https://github.com/samlaihei/GenDIReCT.git
cd GenDIReCT
```

2. Create conda or virtual environment:
```bash
python -m venv .testenv
```

3. Install minimum dependencies:
```bash
pip install ehtim torch torchvision diffusers imagehash ipykernel
```
Install the ClosureInvariants package: https://github.com/nithyanandan/ClosureInvariants.

Expected installation time: few minutes

### Model Weights
Model weights are tracked via Git Large File Storage (LFS).

## Quick Start

### Basic Usage

```python
import torch
import ehtim as eh
from runGenDIReCT import GenDIReCT

# Set device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model
model = GenDIReCT(
    model_path='models/saved_models/Diffusion_M87_eht2022_230_thnoise_1.pt',
    autoencoder_path='models/saved_models/Autoencoders/DIReCT_v2_AEweights.pt',
    uvfits_path='Diffusion_M87_eht2022_230_thnoise_1.uvfits',
    device=device
)

# Load reference image
img = eh.image.load_fits('Images/s_sgra.fits')

# Generate reconstruction
model.image(img, N_images=1024, useObs=False)
```

Expected runtime: few minutes


## Model Components

### Pre-trained Models

The repository includes pre-trained models for:
- **M87**: Simulated M87* with trained weights for EHT 2022 data at 230 GHz
- **Autoencoders**: General-purpose autoencoder weights


## Dataset and Images

The `Images/` directory contains reference images for various objects:
- Geometric models (i.e. rings, crescents, disks)
- Model EHT targets (i.e. Sgr A*, Centaurus A)
- Einstein's face

## Citation

If you use GenDIReCT in your research, please cite:

```bibtex
@ARTICLE{2025PASA...42..148L,
       author = {{Lai}, Samuel and {Thyagarajan}, Nithyanandan and {Wong}, O. Ivy and {Diakogiannis}, Foivos},
        title = "{Very-long baseline interferometry imaging with closure invariants using conditional image diffusion}",
      journal = {\pasa},
     keywords = {Methods: data analysis, techniques: image processing, techniques: interferometric, Instrumentation and Methods for Astrophysics},
         year = 2025,
        month = nov,
       volume = {42},
          eid = {e148},
        pages = {e148},
          doi = {10.1017/pasa.2025.10110},
archivePrefix = {arXiv},
       eprint = {2510.12093},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025PASA...42..148L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Event Horizon Telescope Collaboration
- The `ehtim` library for interferometric imaging
- PyTorch and the open-source deep learning community

---

For questions and support, please open an issue or contact the maintainers.
    &middot;
    <a href="https://github.com/samlaihei/GenDIReCT/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>
