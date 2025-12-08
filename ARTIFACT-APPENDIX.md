# Artifact Appendix

Paper title: **Website Fingerprinting on Nym: Attacks and Defenses**

Authors: Eric Jollès, Simon Wicky, Harry Halpin, Ania M. Piotrowska, Carmela Troncoso

Requested Badge(s):
  - [x] **Available**
  - [x] **Functional**
  - [x] **Reproduced**

## Description

This artifact accompanies the paper "Website Fingerprinting on Nym: Attacks and Defenses" accepted at Privacy Enhancing Technologies Symposium (PETs) 2025.

The artifact provides:
1. **Traffic capture pipeline**: Scripts for capturing website traffic through Tor and Nym networks
2. **WTF4NYM defense implementation**: Defense with configurable parameters
3. **Flow correlation attack**: Deep learning model  for correlating traffic flows at proxy and network requester observation points. ML model from "MixMatch: Flow Matching for Mixnet Traffic", Oldenburg et al.  
4. **Feature importance analysis**: Tools for analyzing which traffic features contribute to website fingerprinting attacks
5. **Datasets**: Pre-captured traffic traces for monitored and unmonitored websites under various network configurations

The artifact enables reproduction of the main experimental results including:
- Flow correlation attack accuracy (ROC curves and AUC metrics)
- Defense effectiveness evaluation (correlation attack performance with/without WTF4NYM)
- Feature importance rankings
- Defense overhead measurements (bandwidth and time)

### Security/Privacy Issues and Ethical Concerns

**No security or privacy risks.** The artifact analyzes publicly accessible websites and does not include any personally identifiable information, vulnerable code, exploits, or security-disabling mechanisms. All website visits were automated and did not involve human subjects.

## Basic Requirements

### Hardware Requirements

**Recommended for full reproduction** (training models from scratch):
- CPU: 8+ cores
- RAM: 16GB or more
- GPU: NVIDIA GPU with 8GB+ VRAM (CUDA-compatible)
- Storage: 100GB free disk space

**Note**: The experiments reported in the paper were performed on:
- CPU: Intel Xeon processors with 16 cores
- RAM: 32GB
- GPU: NVIDIA Tesla V100 (16GB VRAM) or similar
- OS: Ubuntu 22.04 LTS

The artifact can run on commodity hardware without GPU (using CPU), but training will take significantly longer (~4-5x).

### Software Requirements

**Operating System**:
- Tested on: Ubuntu 20.04 and 22.04 LTS
- Should work on: Any modern Linux distribution
- May work on: macOS with appropriate dependencies
- Not supported: Windows (use WSL2)

**Python Environment**:
- Python 3.8 or higher (tested with Python 3.9, 3.10)

**Python Dependencies** (see `requirements.txt`):
- PyTorch >= 1.12.0 (for MixMatch neural network and WF attacks)
- TensorFlow >= 2.8.0 (for Deep Fingerprinting and Tik-Tok models)
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0 (for k-FP Random Forest classifier)
- Matplotlib >= 3.5.0
- Seaborn >= 0.11.0
- Jupyter >= 1.0.0
- Scapy >= 2.4.5 (for packet processing)
- tqdm >= 4.62.0

**Datasets**:
The artifact includes pre-captured traffic datasets available on [Zenodo](https://doi.org/10.5281/zenodo.17840656) and organized in the `data/` directory:
- `data/full_list/`: Complete dataset with monitored and unmonitored websites for Tor and Nym (labnet and mainnet)
- `data/reduced_list/`: Multiple configurations testing different WTF4NYM defense parameters
- `data/traffic_captures/`: Raw traffic captures organized by configuration
- `data/train_test_WF/`: Pre-processed pickle files ready for WF attack training (33 configurations, generated from `traffic_captures/` using the [`transform_to_ml.ipynb`](https://github.com/spring-epfl/WF4NYM-artifacts/blob/main/captures/analysis/transform_to_ml.ipynb) notebook)
- `data/overheads/`: Overhead analysis data including latency and bandwidth measurements

See `data/README.md` for complete dataset documentation.

### Estimated Time and Storage Consumption

**Time Estimates**:

*Setup and Environment*:
- Initial setup: 20-30 minutes

*Individual Experiments*:
- WF Attack training (per configuration, 5-fold CV): 4-8 hours with GPU, 20-40 hours CPU-only
- Flow correlation training: 12-16 hours with GPU
- Feature importance analysis: 2-3 hours
- Defense overhead analysis: 30-45 minutes

## Environment

### Accessibility

**Artifact Repository**: 
https://github.com/spring-epfl/WF4NYM-artifacts

The repository is publicly accessible without any restrictions.

**Datasets**:
Zenodo DOI: https://doi.org/10.5281/zenodo.17840656

The Zenodo repository contains the complete datasets as separate ZIP archives:
- `full_list.zip` - Complete dataset with monitored and unmonitored websites
- `reduced_list.zip` - Various Nym mixnet configurations and defense mechanisms
- `traffic_captures.zip` - Individual trace files for all configurations
- `train_test_WF.zip` - Pre-processed pickle files ready for WF attack training
- `overheads.zip` - Overhead analysis data (latency, bandwidth, traffic volume)

**Permanent Archive**:
The artifact is permanently archived on Zenodo with a specific DOI for long-term accessibility.

### Set up the environment

**Step 1: Clone the repository**

```bash
git clone git@github.com:spring-epfl/WF4NYM-artifacts.git
cd Artifacts_PETs_WF4NYM
```

**Step 2: Set up Python virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected output: All packages install successfully without errors.

**Step 4: Download datasets from Zenodo**

Download the dataset ZIP files from Zenodo (https://doi.org/10.5281/zenodo.17840656) and extract them into the `data/` directory:

```bash
cd data/
# Download and extract each dataset
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

Expected output: All ZIP files are downloaded and extracted successfully. The `data/` directory should contain approximately 100GB of data across all subdirectories.

### Testing the Environment

To verify that the environment is set up correctly, run the following basic functionality tests:

**Test 1: Verify Python dependencies**

```bash
python3 -c "import torch; import tensorflow; import sklearn; import numpy; import pandas; print('All core dependencies imported successfully')"
```

Expected output: `All core dependencies imported successfully`

**Test 2: Verify dataset structure**

```bash
ls -lh data/train_test_WF/ | head -5
```

Expected output: List of pickle files (e.g., `configuration00_default.pkl`, `configuration01_lqp10.pkl`, etc.)

**Test 3: Test ExplainWF integration (after setup)**

After completing the ExplainWF setup in `WF_attacks/README.md`:

```bash
cd WF_attacks/explainwf-popets2023.github.io/ml/code
python3 -c "import classifiers; import common; print('ExplainWF modules loaded successfully')"
```

Expected output: `ExplainWF modules loaded successfully`

If all tests pass, the environment is ready for artifact evaluation.

## Artifact Evaluation

### Results and Claims

The artifact supports reproduction of the following main claims from the paper:

#### Result 1: Website Fingerprinting Attacks on Nym/Tor

**Claim**: Website fingerprinting attacks (k-FP, DF, Tik-Tok) achieve high accuracy on traffic captured through Tor and Nym networks.

**Reproduced by**: WF Attack experiments using ExplainWF framework (see `WF_attacks/README.md`)

#### Result 2: Flow Correlation Attack

**Claim**: The MixMatch-based flow correlation attack can correlate traffic flows between different observation points in the Nym mixnet.

**Reproduced by**: Flow correlation experiments (see `correlation/README.md`)

#### Result 3: Feature Importance Analysis

**Claim**: Packet counts and timing features are the most important features for website fingerprinting attacks.

**Reproduced by**: Feature importance analysis using Random Forest (see `feature_importance/README.md`)

#### Result 4: Defense Effectiveness and Overhead

**Claim**: WTF4NYM defense reduces attack accuracy while introducing acceptable bandwidth and time overhead.

**Reproduced by**: Comparison of attack performance across defense configurations and overhead analysis (see `captures/analysis/README.md`)

### Experiments

The artifact is organized into modular components, each with detailed instructions in their respective README files:

#### Experiment 1: Website Fingerprinting Attacks

**Purpose**: Evaluate WF attacks (k-FP, DF, Tik-Tok, SVM) on captured traffic with different defense configurations.
This is used in Table 2, Table 4, Table 6, Table 7 and Table 9.

**Location**: `WF_attacks/`

**Documentation**: See `WF_attacks/README.md` for:
- Setup instructions (cloning ExplainWF framework and applying patches)
- Training models with 5-fold cross-validation
- Dataset format (pickle files in `data/train_test_WF/`)
- Expected outputs and evaluation metrics

**Key Command**:
```bash
cd WF_attacks/explainwf-popets2023.github.io/ml/code
python train_test.py <output_dir> <pickle_files...>
```

#### Experiment 2: Flow Correlation Attacks

**Purpose**: Train and evaluate MixMatch-based flow correlation models to match traffic at different observation points.
This is used in Figure 7.

**Location**: `correlation/`

**Documentation**: See `correlation/README.md` for:
- Data preparation pipeline
- Training correlation models with/without defense
- ROC curve generation and comparison
- Evaluation metrics

**Key Script**: `correlation/launch_training.sh`

#### Experiment 3: Feature Importance Analysis

**Purpose**: Analyze which traffic features are most important for WF attacks using Random Forest feature importance.
This is used in Table 3, Table 5, Table 8 and Table 9.

**Location**: `feature_importance/`

**Documentation**: See `feature_importance/README.md` for:
- Feature extraction methodology
- Random Forest training and feature importance computation
- Visualization of results

**Key Notebook**: `feature_importance/feature_importance.ipynb`

#### Experiment 4: Traffic Capture and Defense Overhead Analysis

**Purpose**: Capture traffic, apply WTF4NYM defense, and measure bandwidth/time overhead.
This is used in Table 2, Table 4, Table 6 and Table 7.

**Location**: `captures/`

**Documentation**: See `captures/README.md` and `captures/analysis/README.md` for:
- Traffic capture pipeline
- Defense implementation and parameter configuration
- Overhead computation methodology
- Data transformation for ML experiments

**Key Notebooks**:
- `captures/analysis/overheads.ipynb` - Defense overhead analysis
- `captures/analysis/transform_to_ml.ipynb` - Convert captures to ML format

#### Dataset Organization

**Location**: `data/`

**Documentation**: See `data/README.md` for:
- Complete dataset structure and organization
- Mapping between configurations and defense parameters
- Dataset sizes and descriptions
- Pre-processed pickle files for WF attacks

## Limitations

The following aspects should be noted when reproducing results:

1. **Traffic Collection**: 
   - The original traffic collection scripts in `captures/` are provided but are **not reproducible** for artifact evaluation
   - Requires specific network setup (Tor/Nym clients, network configuration) and **Nym API keys**
   - Traffic capture is a **long-running process** (days to weeks depending on the dataset size)
   - Network conditions, Tor circuit selection, and Nym network state vary over time, so recaptured traffic will differ from our datasets
   - **Mitigation**: We provide complete pre-captured datasets for all experiments, so reviewers do not need to run the capture scripts

2. **Training Variability**:
   - Neural network training is non-deterministic even with fixed random seeds due to GPU parallelism and framework differences
   - Results may vary by 2-5% across training runs
   - 5-fold cross-validation provides more robust evaluation but increases training time

3. **Computational Resources**:
   - WF attack training (especially DF and Tik-Tok models) benefits significantly from GPU acceleration
   - Training on CPU is possible but takes 4-5x longer
   - Full 5-fold CV on all 33 dataset configurations is time-intensive

4. **ExplainWF Framework**:
   - Requires cloning and patching the external ExplainWF repository
   - Our modifications are provided as a patch file in `WF_attacks/explainwf_modifications.patch`
   - See `WF_attacks/README.md` for detailed integration instructions

Despite these limitations, the artifact is **Functional** (all components can be executed) and **Reproduced** (main results can be validated within acceptable variance).

## Notes on Reusability

This artifact is designed for reuse and extension beyond the paper:

**Modularity**:
Each component can be used independently:
- `captures/`: Traffic collection pipeline and defense implementations
- `WF_attacks/`: Website fingerprinting attack evaluation framework
- `correlation/`: Flow correlation attack implementation
- `feature_importance/`: Feature analysis tools
- `data/`: Organized datasets with clear directory structure

**Extensibility**:

1. **Testing new defense configurations**:
   - Modify defense parameters in `captures/process_raw_packets/pipeline.py`
   - Generate new defended traces using `captures/analysis/transform_to_ml.ipynb`
   - Evaluate using existing WF attack and correlation frameworks

2. **Adding new WF attack models**:
   - The ExplainWF framework integration in `WF_attacks/` supports multiple classifiers
   - Add new models following the classifier interface in `explainwf-popets2023.github.io/ml/code/classifiers.py`
   - Use the same 5-fold CV evaluation pipeline via `train_test.py`

3. **Analyzing different traffic features**:
   - Feature extraction code in `feature_importance/extract.py`
   - Modify or extend feature sets
   - Re-run Random Forest importance analysis

4. **Working with new datasets**:
   - Follow the pickle format: `(X_train, X_test, y_train, y_test)`
   - See `captures/analysis/transform_to_ml.ipynb` for data preparation
   - Datasets must follow the sequence format (list of packet directions/sizes)

**Documentation**:
Each component directory contains a detailed README:
- `captures/README.md` - Traffic capture and processing pipeline
- `WF_attacks/README.md` - WF attack evaluation and ExplainWF integration
- `correlation/README.md` - Flow correlation experiments
- `feature_importance/README.md` - Feature analysis methodology
- `data/README.md` - Dataset organization and descriptions

## Contact

If there are questions about our tools or paper, please either file an issue or contact `eric.jolles (AT) epfl.ch`

## Research Paper

You can cite our work with the following BibTeX entry:

```bibtex
@inproceedings{jolles2026WFonNym,
 author = {Jollès, Eric and Wicky, Simon and Ania M., Piotrowsak and Harry, Halpin an Carmela, Troncoso},
 booktitle = {},
 title = {{Website Fingerprinting on Nym: Attacks and Defenses}},
 year = {2026},
}
```