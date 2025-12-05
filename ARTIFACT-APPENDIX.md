# Artifact Appendix

Paper title: **Website Fingerprinting on Nym: Attacks and Defenses**

Requested Badge(s):
  - [x] **Available**
  - [x] **Functional**
  - [x] **Reproduced**

## Description

This artifact accompanies the paper "Website Fingerprinting on Nym: Attacks and Defenses" accepted at Privacy Enhancing Technologies Symposium (PETs) 2025.

The artifact provides:
1. **Traffic capture pipeline**: Scripts for capturing website traffic through Tor and Nym networks
2. **WTF4NYM defense implementation**: Traffic morphing defense with configurable parameters
3. **MixMatch flow correlation attack**: Deep learning model (based on PoPETs 2024.2) for correlating traffic flows at proxy and network requester observation points
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

**Minimum requirements** (for running pre-trained models and analyzing results):
- CPU: Modern multi-core processor (Intel/AMD x86-64)
- RAM: 8GB minimum
- Storage: 60GB free disk space (50GB for datasets, 10GB for models and results)

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

**Optional** (for data collection only, not required for reproduction):
- Docker >= 20.10
- Firefox browser
- Tor Browser or standalone Tor
- Nym client

**Datasets**:
The artifact includes pre-captured traffic datasets organized in the `data/` directory:
- `data/full_list/`: Complete dataset with monitored and unmonitored websites for Tor and Nym (labnet and mainnet)
- `data/reduced_list/`: Multiple configurations testing different WTF4NYM defense parameters
- `data/traffic_captures/`: Raw traffic captures organized by configuration
- `data/train_test_WF/`: Pre-processed pickle files ready for WF attack training (33 configurations, generated from `traffic_captures/` using `captures/analysis/transform_to_ml.ipynb`)

See `data/README.md` for complete dataset documentation.

### Estimated Time and Storage Consumption

**Storage**:
- Artifact repository: ~100MB
- Datasets (included): See `data/README.md` for breakdown by configuration
- Generated results: ~5GB per WF attack experiment
- Temporary files during processing: ~10-20GB

**Time Estimates**:

*Setup and Environment*:
- Initial setup: 20-30 minutes
- Environment testing: 5 minutes

*Individual Experiments*:
- WF Attack training (per configuration, 5-fold CV): 4-8 hours with GPU, 20-40 hours CPU-only
- Flow correlation training: 12-16 hours with GPU
- Feature importance analysis: 2-3 hours
- Defense overhead analysis: 30-45 minutes

## Environment

### Accessibility

**Artifact Repository**: 
https://github.com/mixnet-correlation/Artifacts_PETs_WF4NYM

The repository is publicly accessible without any restrictions.

**Datasets and Models**:
Zenodo DOI: [To be added upon upload - will be in format https://doi.org/10.5281/zenodo.XXXXXXX]

**License**:
MIT License (see LICENSE file in repository)

**Permanent Archive**:
After acceptance, we will create a permanent release with a specific git tag and archive on Zenodo.

### Set up the environment

**Step 1: Clone the repository**

```bash
git clone https://github.com/mixnet-correlation/Artifacts_PETs_WF4NYM.git
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

**Step 4: Download datasets**

```bash
# Download from Zenodo (adjust URL when available)
wget https://zenodo.org/record/XXXXXXX/files/data.tar.gz

# Extract datasets
tar -xzf data.tar.gz

# Verify extraction
ls -lh data/
```

Expected output: `data/` directory should contain `full_list/` and `reduced_list/` subdirectories, totaling ~50GB.

**Step 5: (Optional) Download pre-trained models**

```bash
# Download pre-trained MixMatch models
wget https://zenodo.org/record/XXXXXXX/files/pretrained_models.tar.gz
tar -xzf pretrained_models.tar.gz -C correlation/
```

Expected output: Models placed in `correlation/models/`.



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

**Location**: `correlation/`

**Documentation**: See `correlation/README.md` for:
- Data preparation pipeline
- Training correlation models with/without defense
- ROC curve generation and comparison
- Evaluation metrics

**Key Script**: `correlation/launch_training.sh`

#### Experiment 3: Feature Importance Analysis

**Purpose**: Analyze which traffic features are most important for WF attacks using Random Forest feature importance.

**Location**: `feature_importance/`

**Documentation**: See `feature_importance/README.md` for:
- Feature extraction methodology
- Random Forest training and feature importance computation
- Visualization of results

**Key Notebook**: `feature_importance/feature_importance.ipynb`

#### Experiment 4: Traffic Capture and Defense Overhead Analysis

**Purpose**: Capture traffic, apply WTF4NYM defense, and measure bandwidth/time overhead.

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
   - The original traffic collection scripts are provided but require specific network setup (Tor/Nym clients, network configuration)
   - Network conditions, Tor circuit selection, and Nym network state vary over time, so recaptured traffic will differ
   - **Mitigation**: We provide complete pre-captured datasets for all experiments

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