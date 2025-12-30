# Artifact Appendix

Paper title: **Website Fingerprinting on Nym: Attacks and Defenses**

Authors: Eric Jollès, Simon Wicky, Ania M. Piotrowska, Harry Halpin, Carmela Troncoso

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

### Security/Privacy Issues and Ethical Concerns

**No security or privacy risks.** The artifact analyzes publicly accessible websites and does not include any personally identifiable information, vulnerable code, exploits, or security-disabling mechanisms. All website visits were automated and did not involve human subjects.

## Basic Requirements

### Hardware Requirements

The experiments reported in the paper were performed on:
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

**Python Environment**:
- Python 3.9

**Python Dependencies**:
See `requirements.txt`


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

There are two ways to setup the environment.
If you want to verify the functionality of the code on a subset of the data, we recommend you use our docker setup in [Testing the Environment](#testing-the-environment) otherwise here is the full stup for the pipeline.

**Step 1: Clone the repository**

```bash
git clone https://github.com/spring-epfl/WF4NYM-artifacts.git
cd WF4NYM-artifacts
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

Download the dataset ZIP files from Zenodo (https://doi.org/10.5281/zenodo.17840656) and extract them into the `data/` directory.

**Option 1: Automated download script** (recommended):

```bash
./download_data.sh
```

**Option 2: Manual download**:

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

The artifact can be tested using Docker to ensure a reproducible environment. The following steps demonstrate the complete workflow: processing raw packet captures (PCAP files) into machine learning format, then using this processed data for feature importance analysis and website fingerprinting attacks.

#### Quick Start with Docker

**Step 1: Build and launch the Docker container** (~10 minutes)

```bash
git clone https://github.com/spring-epfl/WF4NYM-artifacts.git
cd WF4NYM-artifacts
docker compose up --build
```

Expected output: Container builds successfully and Jupyter Lab starts on `http://localhost:8888`

**Step 2: Open Jupyter Lab and start a terminal**

1. Navigate to `http://localhost:8888/lab` in your web browser
2. Click "Terminal" to open a terminal session inside the container

**Step 3: Test traffic processing pipeline** (~1 minute)

In the Jupyter Lab terminal, run:

```bash
python3 /workspace/captures/process_raw_packets/pipeline.py \
    --pcap-folder /workspace/data/data_test \
    --datasets data-normal \
    --output-folder /workspace/data
```

Expected output: The pipeline processes the test dataset and creates the following populated directories in `http://localhost:8888/lab/workspaces/auto-D/tree/data`:
- `1_extracted_pcaps` - Extracted PCAP files
- `2_aggregated_websites` - Aggregated website traffic
- `3_ml_format` - Machine learning format files (including `data.pkl`)
- `4_individual_traces` - Individual trace files

**Step 4: Test feature importance analysis** (~2 minutes)

1. Navigate to `http://localhost:8888/lab/workspaces/auto-u/tree/feature_importance/feature_importance.ipynb` in your web browser
2. Run all cells in the notebook (Run -> Run all cells)

Expected output: The final cells display feature importance results with values close to 100%, since the test dataset contains two highly separable website classes.

**Step 5: Test website fingerprinting attacks** (~45 minutes)

In the Jupyter Lab terminal, run:

```bash
cd /workspace/WF_attacks/explainwf-popets2023.github.io/ml/code
python3 -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt  # ~3 minutes
python train_test.py /workspace/data/3_ml_format/data.pkl /workspace/output  # ~40 minutes
```

Expected output: 
- 5-fold cross-validation results are saved to `http://localhost:8888/lab/workspaces/auto-D/tree/output`
- Accuracy metrics should be close to 1.0 (100%) since the test dataset has two highly separable classes


## Artifact Evaluation

### Main Results and Claims

#### Main Result 1: Website Fingerprinting Attacks on Nym/Tor

Website fingerprinting attacks (k-FP, DF, Tik-Tok) achieve high accuracy on traffic captured through Tor and Nym networks. This claim is reproducible by executing [Experiment 1](#experiment-1-website-fingerprinting-attacks). We report these results in Table 2, Table 4, Table 6, Table 7 and Table 9 of our paper.

#### Main Result 2: Flow Correlation Attack

The MixMatch-based flow correlation attack can correlate traffic flows between different observation points in the Nym mixnet. This claim is reproducible by executing [Experiment 2](#experiment-2-flow-correlation-attacks). We report these results in Figure 7 of our paper.

#### Main Result 3: Feature Importance Analysis

Packet counts and timing features are the most important features for website fingerprinting attacks. This claim is reproducible by executing [Experiment 3](#experiment-3-feature-importance-analysis). We report these results in Table 3, Table 5, Table 8 and Table 9 of our paper.

#### Main Result 4: Defense Effectiveness and Overhead

WTF4NYM defense reduces attack accuracy while introducing acceptable bandwidth and time overhead. This claim is reproducible by executing [Experiment 4](#experiment-4-traffic-capture-and-defense-overhead-analysis). We report these results in Table 2, Table 4, Table 6 and Table 7 of our paper.

### Experiments

The artifact is organized into modular components, each with detailed instructions in their respective README files:

#### Experiment 1: Website Fingerprinting Attacks

- **Time**: 4-8 hours with GPU per configuration (5-fold CV), 20-40 hours CPU-only
- **Storage**: ~20GB per configuration

This experiment reproduces [Main Result 1](#main-result-1-website-fingerprinting-attacks-on-nymtor). Evaluate WF attacks (k-FP, DF, Tik-Tok, SVM) on captured traffic with different defense configurations.

**Location**: `WF_attacks/`

**Documentation**: See `WF_attacks/README.md` for detailed instructions on:
- Setup (cloning ExplainWF framework and applying patches)
- Training models with 5-fold cross-validation
- Dataset format (pickle files in `data/train_test_WF/`)
- Expected outputs and evaluation metrics

**Key Command**:
```bash
cd WF_attacks/explainwf-popets2023.github.io/ml/code
python train_test.py <output_dir> <pickle_files...>
```

#### Experiment 2: Flow Correlation Attacks

- **Time**: 12-16 hours with GPU
- **Storage**: ~15GB

This experiment reproduces [Main Result 2](#main-result-2-flow-correlation-attack). Train and evaluate MixMatch-based flow correlation models to match traffic at different observation points.

**Location**: `correlation/`

**Documentation**: See `correlation/README.md` for detailed instructions on:
- Data preparation pipeline
- Training correlation models with/without defense
- ROC curve generation and comparison
- Evaluation metrics

**Key Script**: `correlation/launch_training.sh`

#### Experiment 3: Feature Importance Analysis

- **Time**: 2-3 hours
- **Storage**: ~5GB

This experiment reproduces [Main Result 3](#main-result-3-feature-importance-analysis). Analyze which traffic features are most important for WF attacks using Random Forest feature importance.

**Location**: `feature_importance/`

**Documentation**: See `feature_importance/README.md` for detailed instructions on:
- Feature extraction methodology
- Random Forest training and feature importance computation
- Visualization of results

**Key Notebook**: `feature_importance/feature_importance.ipynb`

#### Experiment 4: Traffic Capture and Defense Overhead Analysis

- **Time**: 30-45 minutes (using pre-captured data)
- **Storage**: ~10GB

This experiment reproduces [Main Result 4](#main-result-4-defense-effectiveness-and-overhead). Analyze pre-captured traffic with WTF4NYM defense applied and measure bandwidth/time overhead.

**Location**: `captures/`

**Documentation**: See `captures/README.md` and `captures/analysis/README.md` for detailed instructions on:
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


**Traffic Collection**: 
- The original traffic collection scripts in `captures/` are provided but are **not reproducible** for artifact evaluation
- Requires specific network setup and **Nym API keys**
- Traffic capture is a **long-running process** (days to weeks depending on the dataset size)
- Network conditions, Tor circuit selection, and Nym network state vary over time, so recaptured traffic will differ from our datasets
- **Mitigation**: We provide complete pre-captured datasets for all experiments, so reviewers do not need to run the capture scripts

Despite this limitation, the artifact is **Functional** (all components can be executed) and **Reproduced** (main results can be validated within acceptable variance).

## Notes on Reusability

**Modularity**:
Each component can be used independently:
- `captures/`: Traffic collection pipeline and defense implementations
- `WF_attacks/`: Website fingerprinting attack evaluation framework
- `correlation/`: Flow correlation attack implementation
- `feature_importance/`: Feature analysis tools
- `data/`: Organized datasets with clear directory structure

**Documentation**:
Each component directory contains a detailed README:
- `captures/README.md` - Traffic capture and processing pipeline
- `WF_attacks/README.md` - WF attack evaluation and ExplainWF integration
- `correlation/README.md` - Flow correlation experiments
- `feature_importance/README.md` - Feature analysis methodology
- `data/README.md` - Dataset organization and descriptions

## License

This artifact is released under the MIT License. See the `LICENSE` file in the repository root for full license text.

## Contact

If there are questions about our tools or paper, please either file an issue or contact `eric.jolles (AT) epfl.ch`

## Research Paper

You can cite our work with the following BibTeX entry:

```bibtex
@inproceedings{jolles2026WFonNym,
 author = {Jollès, Eric and Wicky, Simon and Piotrowska, Ania M. and Halpin, Harry and Troncoso, Carmela},
 booktitle = {},
 title = {{Website Fingerprinting on Nym: Attacks and Defenses}},
 year = {2026},
}
```