# Analysis Directory

This directory contains notebooks for creating WTF4NYM defense traces and computing overhead metrics for Website Fingerprinting defenses on Nym mixnet and Tor network traffic.

## Notebooks

### `cover_traffic.ipynb`
Notebook for generating WTF4NYM defense traffic traces by applying cover traffic padding mechanisms to captured network traces.

### `overheads.ipynb`
Notebook for computing latency and bandwidth overhead introduced by various defense mechanisms across different Nym configurations.

## Input Data
Processed traffic traces from `../../data/` directory in format: `<website_url> <num_packets> <timestamp1>:<size1> <timestamp2>:<size2> ...` where positive sizes are incoming packets and negative sizes are outgoing packets.
