# Artifacts for PETs WF4NYM

This repository contains the artifacts for the paper "Website Fingerprinting on Nym: Attacks and Defenses" submitted to Privacy Enhancing Technologies Symposium (PETs).

## Overview

This artifact package provides scripts, data, and analysis tools for evaluating Website Fingerprinting (WF) attacks and the WTF4NYM defense mechanism on the Nym mixnet and Tor network.

## Directory Structure

### `captures/`
Complete pipeline for traffic capture, processing, and analysis:
- **`capture_scripts/`** - Automated website traffic capture using Firefox
- **`process_raw_packets/`** - Pipeline to convert raw PCAP captures into ML-ready traces
- **`setup_scripts/`** - System configuration for Tor and Nym network environments
- **`analysis/`** - Jupyter notebooks for generating WTF4NYM defense traces and computing overhead metrics

See `captures/README.md` for detailed workflow documentation.

### `data/`
Traffic datasets captured across different network configurations (**Note: Not included by default, see [Data Download](#data-download) below**):

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

## Data Download

The `data/` directory is **not included by default** due to its large size. To download the dataset:

### Option 1: Download from Zenodo

1. Visit the Zenodo repository: [https://zenodo.org/record/XXXXXXX](https://zenodo.org/record/XXXXXXX) *(link to be added upon publication)*
2. Download the `data.tar.gz` archive
3. Extract it to this directory:
   ```bash
   tar -xzf data.tar.gz -C /path/to/Artifacts_PETs_WF4NYM/
   ```

### Option 2: Direct Download via Command Line

```bash
cd /path/to/Artifacts_PETs_WF4NYM/
wget https://zenodo.org/record/XXXXXXX/files/data.tar.gz
tar -xzf data.tar.gz
```

The extracted `data/` directory should contain:
- `full_list/` - ~XX GB
- `reduced_list/` - ~XX GB  
- `traffic_captures/` - ~XX GB
- `overheads/` - ~XX MB

## Dependencies

### Defense Code
We used the defense implementations from: https://github.com/websitefingerprinting/WebsiteFingerprinting

### Attack Code
We used the ML attack code from: https://explainwf-popets2023.github.io/

### Python Requirements
Install dependencies using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Artifact Evaluation

**For detailed artifact evaluation information, please see [ARTIFACT-APPENDIX.md](ARTIFACT-APPENDIX.md).**

The ARTIFACT-APPENDIX.md file contains:
- Complete installation and setup instructions
- Step-by-step reproduction instructions for all main results
- Expected outputs and validation criteria
- Information about artifact badges (Available, Functional, Reproduced)

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone https://github.com/[organization]/Artifacts_PETs_WF4NYM.git
   cd Artifacts_PETs_WF4NYM
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download data** (if not included):
   ```bash
   wget https://zenodo.org/record/XXXXXXX/files/data.tar.gz
   tar -xzf data.tar.gz
   ```

3. **Run quick validation test** (~30 minutes):
   ```bash
   cd correlation
   # [Follow instructions in ARTIFACT-APPENDIX.md for quick validation]
   ```

4. **For full reproduction** (~24-48 hours):
   ```bash
   # [Follow step-by-step instructions in ARTIFACT-APPENDIX.md]
   ```

## Citation

If you use this artifact, please cite:

```bibtex
@inproceedings{wf4nym2025,
  title={Website Fingerprinting on Nym: Attacks and Defenses},
  author={[Authors]},
  booktitle={Privacy Enhancing Technologies Symposium (PETs)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: Some components use code from external projects with their own licenses:
- Defense implementations: [WebsiteFingerprinting](https://github.com/websitefingerprinting/WebsiteFingerprinting) (MIT License)
- Attack implementations: [ExplainWF](https://explainwf-popets2023.github.io/) ([License])

## Contact

For questions or issues regarding this artifact, please:
- Open an issue on GitHub: [https://github.com/[organization]/Artifacts_PETs_WF4NYM/issues](https://github.com/[organization]/Artifacts_PETs_WF4NYM/issues)
- Contact the authors: [email@example.com]

## Acknowledgments

We thank the developers of the WebsiteFingerprinting and ExplainWF projects for making their code publicly available.