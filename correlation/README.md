# MixMatch Flow Correlation Analysis

Deep learning-based traffic correlation attack for website fingerprinting evaluation. This implementation is based on the MixMatch drift classifier from [1].

## Overview

This toolkit implements a neural network-based correlation attack that attempts to match traffic flows observed at different network observation points (proxy side and network requester side). It is designed to evaluate the effectiveness of traffic analysis defenses in anonymity networks like Nym.

## Prerequisites

### Python Dependencies
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn tqdm
```

### Input Data Format

Traffic capture files should contain one line per capture with the format:
```
<webpage_name> <capture_number> <total_bytes> <timestamp1:bytes1> <timestamp2:bytes2> ...
```

Example:
```
google.com 0 15234 100.5:1500 101.2:-800 102.3:2000 ...
```

Positive bytes indicate incoming packets, negative bytes indicate outgoing packets.

### Expected Directory Structure

The correlation datasets must follow this structure with separate directories for proxy-side and network-requester-side observations:

```

correlation_folder/
├── proxy/
│   ├── website1
│   ├── website2
│   └── ...
└── network_requester/
    ├── website1
    ├── website2
    └── ...
```

Each file contains traffic traces in the format described above, with one trace per line. The proxy and network_requester directories contain observations of the same websites from different vantage points in the Nym network.

Note : in our dataset we have 2 folders containing captures for correlation analysis: ```../data/reduced_list/correlation_without_defense``` and ```../data/reduced_list/correlation_with_defense```


### Training without defense:
```bash
./launch_training.sh \
    --input-dir-proxy ../data/reduced_list/correlation_without_defense/proxy/ \
    --input-dir-requester ../data/reduced_list/correlation_without_defense/network_requester/ \
    --output-dir data_without_defense \
    --websites 55 \
    --epochs 100
```

### Training with WTF4NYM defense:
```bash
./launch_training.sh \
    --input-dir-proxy ../data/reduced_list/correlation_with_defense/proxy/ \
    --input-dir-requester ../data/reduced_list/correlation_with_defense/network_requester/ \
    --output-dir data_with_defense \
    --websites 55 \
    --epochs 100
```

### Compare results:
```bash
python scripts/compare_roc_curves.py \
    --results-dirs data_without_defense/results data_with_defense/results \
    --labels "Without WTF4NYM" "With WTF4NYM" \
    --output-dir roc_comparison \
    --title "MixMatch Model Comparison: With vs Without defence"
```

[1] Lennart Oldenburg, Marc Juarez, Enrique Argones Rúa, Claudia Diaz (2023). MixMatch: Flow Matching for Mixnet Traffic. Privacy Enhancing Technologies Symposium (PETS) 2024.