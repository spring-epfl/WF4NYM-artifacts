# Traffic Processing Pipeline

End-to-end pipeline for converting raw pcap captures into training-ready website fingerprinting data.

## Overview

The pipeline consists of 4 main steps:

1. **PCAP Extraction** (`process_packets.py`) - Extract TCP traffic from pcap files
2. **Website Aggregation** (`webpages-to-websites.py`) - Aggregate individual pages into per-website files
3. **ML Format Conversion** (`tcp_to_ml.py`) - Optional format conversion (currently not implemented)
4. **Trace Extraction** (`trace_extraction.py`) - Extract individual timing/volume traces from aggregated data

## Quick Start

### Basic Usage

Process a folder of pcap files with default settings:

```bash
python pipeline.py --pcap-folder /path/to/pcaps --datasets dataset1
```

### Multiple Datasets

```bash
python pipeline.py --pcap-folder /path/to/pcaps \
                   --datasets dataset1 dataset2 dataset3 \
                   --output-folder /path/to/output
```

## Pipeline Steps

### Step 1: PCAP Extraction

Extracts TCP traffic from pcap files and creates tcpdump format files.

**Script:** `process_packets.py`

**Key Parameters:**
- `--src-ip` - Source IP address (default: `10.1.1.1`)
- `--dst-ips` - Destination IP addresses (default: `['139.162.200.242']`)
- `--ignore-port-9002` - Exclude packets on port 9002

**Output:**
- `1_extracted_pcaps/output-tcp/` - TCP traffic files per website

### Step 2: Website Aggregation

Aggregates individual page captures into per-website files, filtering by minimum sample count.

**Script:** `webpages-to-websites.py`

**Key Parameters:**
- `--min-count` - Minimum files per website (default: 10)
- `--min-valid-samples` - Minimum valid samples with 21 lines (default: 10)
- `--max-lines` - Maximum lines per file (default: 20)

**Output:**
- `2_aggregated_websites/` - Aggregated files per website

### Step 3: ML Format Conversion (Optional)

Converts data to ML-ready format. Currently skipped if `tcp_to_ml.py` is not implemented.

**Script:** `tcp_to_ml.py`

**Output:**
- `3_ml_format/` - ML-ready data (if implemented)

### Step 4: Trace Extraction

Extracts individual timing/volume traces from aggregated website files.

**Script:** `trace_extraction.py`

**Key Parameters:**
- `--max-samples` - Maximum samples per webpage (default: 20)

**Output:**
- `4_individual_traces/` - Individual traces
  - Format: `<website_index>-<page_number>`
  - Content: `timestamp\tsigned_volume` (one packet per line)

## Configuration Files

### Required Files

- `websites.txt` - List of website names (one per line)
- `final_urls.txt` - List of URLs for mapping (one per line)

Example `websites.txt`:
```
google.com
facebook.com
youtube.com
```

Example `final_urls.txt`:
```
https://google.com
https://www.facebook.com
https://youtube.com
```

## Advanced Usage

### Skip Specific Steps

```bash
# Skip PCAP extraction (use existing extracted data)
python pipeline.py --pcap-folder /path/to/pcaps \
                   --datasets dataset1 \
                   --skip-extraction

# Skip aggregation
python pipeline.py --pcap-folder /path/to/pcaps \
                   --datasets dataset1 \
                   --skip-aggregation

# Skip defense simulation
python pipeline.py --pcap-folder /path/to/pcaps \
                   --datasets dataset1 \
                   --skip-trace-extraction
```

### Custom Parameters

```bash
python pipeline.py --pcap-folder /path/to/pcaps \
                   --datasets dataset1 \
                   --output-folder ./my_output \
                   --websites-file my_websites.txt \
                   --urls-file my_urls.txt \
                   --src-ip 192.168.1.1 \
                   --dst-ips 10.0.0.1 10.0.0.2 \
                   --min-count 15 \
                   --min-valid-samples 15 \
                   --max-samples 25
```

## Output Structure

```
processed_output/
├── 1_extracted_pcaps/
│   └── output-tcp/
│       ├── google.com
│       ├── facebook.com
│       └── ...
├── 2_aggregated_websites/
│   ├── google.com
│   ├── facebook.com
│   └── ...
├── 3_ml_format/
│   └── (optional, if tcp_to_ml.py is implemented)
└── 4_individual_traces/
    ├── 0-0
    ├── 0-1
    ├── 1-0
    └── ...
```

## Data Format

### PCAP → Extracted TCP

Format: `<url> <num_packets> <timestamp1>:<size1> <timestamp2>:<size2> ...`

Example:
```
nym.com 150 1234567890.123456:-512 1234567890.234567:1024 ...
```

### Aggregated Website Files

Multiple lines of the above format, aggregated per website.

### Defense Simulation Output

Format: `<timestamp>\t<signed_volume>`

- Timestamp: Seconds relative to first packet
- Signed volume: Positive (incoming) or negative (outgoing) bytes

Example:
```
0.000	-512
0.111	1024
0.234	-256
```

