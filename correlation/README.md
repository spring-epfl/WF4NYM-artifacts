# MixMatch Flow Correlation Analysis

Deep learning-based traffic correlation attack for website fingerprinting evaluation. This implementation is based on the MixMatch drift classifier from the PoPETs 2024.2 paper.

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
    --input-dir-proxy ../data/reduced_list/correlation_with_WTF4NYM/proxy/ \
    --input-dir-requester ../data/reduced_list/correlation_with_WTF4NYM/network_requester/ \
    --output-dir data_with_WTF4NYM \
    --websites 55 \
    --epochs 100
```

### Compare results:
```bash
python scripts/compare_roc_curves.py \
    --results-dirs data_without_defense/results data_with_WTF4NYM/results \
    --labels "Without WTF4NYM" "With WTF4NYM" \
    --output-dir roc_comparison \
    --title "MixMatch Model Comparison: With vs Without defence"
```
