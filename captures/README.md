# Captures Directory

This directory contains scripts and notebooks for capturing, processing, and analyzing Website Fingerprinting traffic data.

## Directory Structure

### `setup_scripts/`
System configuration scripts for setting up the capture environment. Includes network namespace setup, service configurations for Nym proxy/requester components, and system initialization scripts.

### `capture_scripts/`
Scripts for automated website traffic capture using Firefox. Includes the browser automation script, network capture tools, and Firefox configuration files for consistent data collection across Tor and Nym networks.

### `process_raw_packets/`
Pipeline for converting raw PCAP captures into ML-ready traffic traces. Processes include TCP traffic extraction, website aggregation, and trace formatting. See the subdirectory README for detailed pipeline documentation.

### `analysis/`
Jupyter notebooks for generating WTF4NYM defense traces (`cover_traffic.ipynb`) and computing latency/bandwidth overhead metrics (`overheads.ipynb`) across different configurations.

## Workflow

1. **Setup** - Use `setup_scripts/` to configure the capture environment
2. **Capture** - Run `capture_scripts/` to collect raw network traffic
3. **Process** - Apply `process_raw_packets/` pipeline to convert PCAPs to traces
4. **Analyze** - Use `analysis/` notebooks to generate defenses and compute overheads
