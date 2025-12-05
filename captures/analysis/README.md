# Analysis Directory

This directory contains notebooks for creating WTF4NYM defense traces, computing overhead metrics, and converting traffic captures to ML-ready format on Nym mixnet and Tor network traffic.

## Notebooks

### `transform_to_ml.ipynb`
**Purpose**: Convert raw traffic captures into pickle format compatible with ExplainWF framework for Website Fingerprinting attacks.

**Input**: 
- Raw traffic traces from `../../data/traffic_captures/` or `../../data/reduced_list/`
- Format: `<url> <num_packets> <timestamp1>:<size1> <timestamp2>:<size2> ...`

**Output**:
- Pickle files in `processed_data/` directory
- Format: `(X_train, X_test, y_train, y_test)` tuple
- Compatible with ExplainWF ML classifiers (k-FP, DF, Tik-Tok, SVM)

**Usage**:
1. Set the `scenario` variable to the configuration name (e.g., `configuration00_default`)
2. Run the cells to process single configuration
3. Or uncomment batch processing cell to process all configurations

**Processing Details**:
- Takes up to 200 instances per website
- Filters websites with at least 20 instances
- 60/40 train/test split with stratification
- Converts to relative timestamps (seconds from start)
- Direction mapping: 0 = outgoing, 1 = incoming

### `cover_traffic.ipynb`
Notebook for generating WTF4NYM defense traffic traces by applying cover traffic padding mechanisms to captured network traces.

### `overheads.ipynb`
Notebook for computing latency and bandwidth overhead introduced by various defense mechanisms across different Nym configurations.

## Input Data
Processed traffic traces from `../../data/` directory in format: `<website_url> <num_packets> <timestamp1>:<size1> <timestamp2>:<size2> ...` where positive sizes are incoming packets and negative sizes are outgoing packets.
