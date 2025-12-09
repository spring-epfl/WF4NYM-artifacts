# Artifact: Website Fingerprinting on Nym: Attacks and Defenses

**Paper**: Website Fingerprinting on Nym: Attacks and Defenses  
**Conference**: Privacy Enhancing Technologies Symposium (PETs) 2026  
**Authors**: Eric Jollès, Simon Wicky, Harry Halpin, Ania M. Piotrowska, Carmela Troncoso

**Artifact Badges**:
- ✓ **Available** - Publicly accessible on GitHub and Zenodo
- ✓ **Functional** - Complete, documented, and executable
- ✓ **Reproduced** - Main results reproducible

## Overview

This artifact provides a complete package for evaluating Website Fingerprinting (WF) attacks and the WTF4NYM defense mechanism on the Nym mixnet and Tor network. The artifact enables reproduction of all main experimental results from the paper.

## Directory Structure

### `captures/`
Complete pipeline for traffic capture, processing, and analysis:
- **`capture_scripts/`** - Automated website traffic capture using Firefox
- **`process_raw_packets/`** - Pipeline to convert raw PCAP captures into ML-ready traces
- **`setup_scripts/`** - System configuration for Tor and Nym network environments
- **`analysis/`** - Jupyter notebooks for generating WTF4NYM defense traces and computing overhead metrics

See `captures/README.md` for detailed workflow documentation.

### `data/`
Pre-captured traffic datasets across different network configurations:
- **`full_list/`** - Complete dataset with monitored and unmonitored websites for Tor and Nym
- **`reduced_list/`** - Various Nym mixnet configurations and defense mechanisms (33 configurations)
- **`traffic_captures/`** - Individual trace files for all configurations
- **`train_test_WF/`** - Pre-processed pickle files ready for WF attack training
- **`overheads/`** - Overhead analysis data (latency, bandwidth, traffic volume)

**Download from Zenodo**: https://doi.org/10.5281/zenodo.17840656 (~100GB total)

See `data/README.md` for complete dataset documentation.

### `correlation/`
Deep learning-based MixMatch flow correlation attack implementation:
- Neural network model for matching traffic flows at different observation points (proxy vs. network requester)
- Training scripts for scenarios with and without WTF4NYM defense
- ROC curve comparison tools for evaluating defense effectiveness

Based on the MixMatch drift classifier. See `correlation/README.md` for usage examples.

### `feature_importance/`
Feature importance analysis using Random Forest classifiers:
- Extracts and evaluates contribution of different feature groups (packet counts, timing, inter-arrival, n-grams, transposition, bursts)
- Adapted from reWeFDE framework for measuring information leakage
- Jupyter notebook pipeline for complete analysis

See `feature_importance/README.md` for details on feature groups and methodology.

### `WF_attacks/`
Website Fingerprinting attack implementations:
- Integration with ExplainWF framework
- Support for multiple classifiers: k-FP, Deep Fingerprinting (DF), Tik-Tok, SVM
- 5-fold cross-validation training pipeline
- Patch file for ExplainWF modifications

See `WF_attacks/README.md` for setup and usage instructions.

## Quick Start

### Option 1: Using Docker (Recommended)

**Prerequisites**: Docker and Docker Compose installed

1. **Clone the repository**:
   ```bash
   git clone https://github.com/spring-epfl/WF4NYM-artifacts.git
   cd Artifacts_PETs_WF4NYM
   ```

2. **Download datasets from Zenodo**:
   ```bash
   cd data/
   wget https://zenodo.org/records/17840656/files/full_list.zip
   wget https://zenodo.org/records/17840656/files/reduced_list.zip
   wget https://zenodo.org/records/17840656/files/traffic_captures.zip
   wget https://zenodo.org/records/17840656/files/train_test_WF.zip
   wget https://zenodo.org/records/17840656/files/overheads.zip
   
   unzip full_list.zip && unzip reduced_list.zip && unzip traffic_captures.zip
   unzip train_test_WF.zip && unzip overheads.zip
   cd ..
   ```

3. **Build and run Docker container**:
   ```bash
   docker-compose up --build
   ```

4. **Access Jupyter notebook**:
   - Open browser to: `http://localhost:8888`
   - No token required (for artifact evaluation convenience)

5. **Alternative: Interactive shell**:
   ```bash
   docker-compose run --rm wf4nym /bin/bash
   ```

### Option 2: Local Installation

**Prerequisites**: 
- Python 3.9
- 16GB+ RAM recommended
- GPU with CUDA support (optional, for faster training)

1. **Clone and setup environment**:
   ```bash
   git clone https://github.com/spring-epfl/WF4NYM-artifacts.git
   cd Artifacts_PETs_WF4NYM
   python3.9 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Download datasets** (as shown above in Docker option)

3. **Verify installation**:
   ```bash
   python3 -c "import torch; import numpy; import pandas; import sklearn; print('All dependencies installed successfully')"
   ```

## System Requirements

### Hardware Requirements

**Minimum** (for basic functionality):
- CPU: 4+ cores
- RAM: 8GB
- Storage: 120GB free disk space

**Recommended** (for full reproduction):
- CPU: 8+ cores (Intel Xeon or equivalent)
- RAM: 16GB+
- GPU: NVIDIA GPU with 8GB+ VRAM (CUDA-compatible)
- Storage: 120GB free disk space

### Software Requirements

**Operating System**:
- Tested on: Ubuntu 20.04 and 22.04 LTS
- Should work on: Any modern Linux distribution
- May work on: macOS with appropriate dependencies
- Not supported: Windows (use WSL2 or Docker)

**Python**: Python 3.9

**Dependencies** (see `requirements.txt`):
- PyTorch 2.5.1 (for neural networks and WF attacks)
- NumPy 1.26.4
- Pandas 2.2.3
- Scikit-learn 1.6.1 (for Random Forest classifier)
- Matplotlib 3.9.4
- Seaborn 0.13.2
- SciPy 1.13.1
- Selenium 4.26.1
- tqdm 4.67.1
- Jupyter (installed via Docker or pip)

## Data Download

## Datasets

The artifact includes pre-captured traffic datasets available on **Zenodo**: https://doi.org/10.5281/zenodo.17840656

### Download Instructions

**All datasets** (~100GB total):

```bash
cd data/
# Download all ZIP files
wget https://zenodo.org/records/17840656/files/full_list.zip
wget https://zenodo.org/records/17840656/files/reduced_list.zip
wget https://zenodo.org/records/17840656/files/traffic_captures.zip
wget https://zenodo.org/records/17840656/files/train_test_WF.zip
wget https://zenodo.org/records/17840656/files/overheads.zip

# Extract all datasets
unzip full_list.zip
unzip reduced_list.zip
unzip traffic_captures.zip
unzip train_test_WF.zip
unzip overheads.zip

cd ..
```

### Dataset Structure

After extraction, the `data/` directory contains:

- **`full_list/`** - Complete dataset with monitored and unmonitored websites
- **`reduced_list/`** - 33 different configurations testing various parameters:
  - Baseline: Tor, Nym (labnet/mainnet), no proxy
  - Defense mechanisms: FRONT, Tamaraw, WTF-PAD, WTF4NYM
  - Parameter variations: Loop rates, mix delays, Poisson rates
- **`traffic_captures/`** - Individual trace files (format: `<website_index>-<page_number>`)
- **`train_test_WF/`** - Pre-processed pickle files for WF attacks (60/40 train/test split)
- **`overheads/`** - Latency and bandwidth measurements

See `data/README.md` for detailed documentation of all configurations.

## Main Experiments

### Experiment 1: Website Fingerprinting Attacks

**Purpose**: Evaluate k-FP, Deep Fingerprinting, Tik-Tok, and SVM attacks  
**Paper Results**: Table 2, Table 4, Table 6, Table 7, Table 9  
**Location**: `WF_attacks/`  
**Documentation**: `WF_attacks/README.md`

**Quick Run** (single configuration, ~4-8 hours with GPU):
```bash
cd WF_attacks/explainwf-popets2023.github.io/ml/code
python train_test.py output/ ../../../data/train_test_WF/configuration00_default.pkl
```

### Experiment 2: Flow Correlation Attacks

**Purpose**: MixMatch-based traffic correlation at different observation points  
**Paper Results**: Figure 7  
**Location**: `correlation/`  
**Documentation**: `correlation/README.md`

**Quick Run** (~12-16 hours with GPU):
```bash
cd correlation
bash launch_training.sh
```

### Experiment 3: Feature Importance Analysis

**Purpose**: Identify most important features for WF attacks  
**Paper Results**: Table 3, Table 5, Table 8, Table 9  
**Location**: `feature_importance/`  
**Documentation**: `feature_importance/README.md`

**Quick Run** (Jupyter notebook, ~2-3 hours):
```bash
jupyter notebook feature_importance/feature_importance.ipynb
```

### Experiment 4: Defense Overhead Analysis

**Purpose**: Measure bandwidth and latency overhead of WTF4NYM  
**Paper Results**: Table 2, Table 4, Table 6, Table 7  
**Location**: `captures/analysis/`  
**Documentation**: `captures/analysis/README.md`

**Quick Run** (Jupyter notebook, ~30-45 minutes):
```bash
jupyter notebook captures/analysis/overheads.ipynb
```

## Testing the Environment

**Test 1: Python dependencies**
```bash
python3 -c "import torch; import numpy; import pandas; import sklearn; print('All dependencies OK')"
```

**Test 2: Dataset structure**
```bash
ls -lh data/train_test_WF/ | head -5
```
Expected: List of pickle files (e.g., `configuration00_default.pkl`, etc.)

**Test 3: ExplainWF integration** (after setup in `WF_attacks/README.md`)
```bash
cd WF_attacks/explainwf-popets2023.github.io/ml/code
python3 -c "import classifiers; import common; print('ExplainWF modules loaded')"
```

## Time and Storage Estimates

**Setup**: 20-30 minutes  
**Storage**: ~120GB (including datasets)

**Experiment Runtimes**:
- WF Attack (per config, 5-fold CV): 4-8 hours (GPU) / 20-40 hours (CPU)
- Flow correlation: 12-16 hours (GPU)
- Feature importance: 2-3 hours
- Overhead analysis: 30-45 minutes

## Reproducibility Notes

### What is Reproducible

All main experimental results can be reproduced:
1. ✓ WF attack accuracies (Table 2, 4, 6, 7, 9) - within 2-5% variance
2. ✓ Flow correlation ROC curves (Figure 7) - qualitatively reproducible
3. ✓ Feature importance rankings (Table 3, 5, 8, 9) - stable rankings
4. ✓ Defense overheads (Table 2, 4, 6, 7) - deterministic from datasets

### Limitations

1. **Traffic Collection**: Original capture scripts provided but not reproducible during evaluation
   - Requires Nym API keys and specific network setup
   - Network conditions vary over time
   - **Mitigation**: Complete pre-captured datasets provided

2. **Training Variability**: Neural networks are non-deterministic
   - Results may vary 2-5% across runs
   - GPU/CPU differences affect training
   - **Mitigation**: 5-fold cross-validation for robust evaluation

3. **Computational Resources**: GPU significantly accelerates training
   - CPU training is 4-5x slower but produces equivalent results
   - Full evaluation of all 33 configs is time-intensive

## Docker Usage Details

### Build Image
```bash
docker build -t wf4nym-artifact .
```

### Run Jupyter Notebook
```bash
docker-compose up
# Access at http://localhost:8888
```

### Interactive Shell
```bash
docker-compose run --rm wf4nym /bin/bash
```

### Run Specific Experiment
```bash
docker-compose run --rm wf4nym bash -c "cd WF_attacks/explainwf-popets2023.github.io/ml/code && python train_test.py output/ ../../../data/train_test_WF/configuration00_default.pkl"
```

### Mount External Data Directory
Edit `docker-compose.yml` to mount data from external location:
```yaml
volumes:
  - .:/workspace
  - /path/to/downloaded/data:/workspace/data
```

## Artifact Accessibility

**GitHub Repository**: https://github.com/spring-epfl/WF4NYM-artifacts  
**Zenodo Archive**: https://doi.org/10.5281/zenodo.17840656  
**License**: MIT (see individual component licenses in respective directories)

**Permanent Archive**: This artifact is permanently archived on Zenodo with a DOI for long-term accessibility.

## Complete Documentation

For complete artifact evaluation information including:
- Detailed badge requirements
- Step-by-step reproduction instructions
- Expected outputs and validation criteria
- Reusability guidelines

**See**: [ARTIFACT-APPENDIX.md](ARTIFACT-APPENDIX.md)

## Dependencies and External Code

This artifact integrates code from external projects:

**Defense Implementations**:
- WTF-PAD, FRONT, Tamaraw from: https://github.com/websitefingerprinting/WebsiteFingerprinting

**Attack Implementations**:
- ExplainWF framework: https://explainwf-popets2023.github.io/

**Flow Correlation**:
- Based on MixMatch: "MixMatch: Flow Matching for Mixnet Traffic" (Oldenburg et al.)

All external code retains its original licensing. Our modifications are provided as patches.

## Citation

If you use this artifact in your research, please cite:

```bibtex
@inproceedings{jolles2026WFonNym,
  author = {Jollès, Eric and Wicky, Simon and Halpin, Harry and Piotrowska, Ania M. and Troncoso, Carmela},
  title = {{Website Fingerprinting on Nym: Attacks and Defenses}},
  booktitle = {Privacy Enhancing Technologies Symposium (PETs)},
  year = {2026}
}
```

## Contact

For questions or issues regarding this artifact:
- **GitHub Issues**: https://github.com/spring-epfl/WF4NYM-artifacts/issues
- **Email**: eric.jolles@epfl.ch

## Acknowledgments

We thank the developers of the WebsiteFingerprinting and ExplainWF projects for making their code publicly available, and the Nym and Tor communities for their privacy-enhancing technologies.
