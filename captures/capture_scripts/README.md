# Website Fingerprinting Capture Scripts

Automated tools for collecting website traffic captures through Nym proxy for website fingerprinting experiments.

## Overview

These scripts automate the collection of network traffic while browsing websites through the Nym mixnet. They handle browser automation, traffic capture, error detection, and retry logic.

## Files

- **`firefox_query.py`** - Single URL capture with error handling
- **`foreground_capture.py`** - Batch capture script for processing URL lists
- **`prefs_non_quic_notracking.js`** - Firefox preferences (disables QUIC, tracking)
## Usage

### Single URL Capture

```bash
python firefox_query.py -u example.com -i veth0 -o ./output --nym_setup 1
```

**Arguments:**
- `--url, -u`: URL to capture (required)
- `--iface, -i`: Network interface to capture on (required)
- `--output, -o`: Output directory (default: `output`)
- `--nym_prefix`: Nym service prefix (default: `wfp`)
- `--nym_setup`: Nym setup index (default: `1`)
- `--no_nym`: Disable Nym proxy (direct connection)

**Output Files:**
- `capture.pcap` - Network traffic capture
- `sslkeys.txt` - SSL/TLS keys for decryption
- `index.html` - Captured page source
- `geckodriver.log` - Browser driver logs

### Batch Capture

```bash
python foreground_capture.py --urls urls.txt --nym_setup 1 --interface veth0
```

**Arguments:**
- `--urls`: Path to URL list file (default: `urls`)
- `--nym_setup`: Nym setup index (default: `1`)
- `--interface`: Network interface (default: `veth0`)
- `--background`: Capture mode for background traffic (1 sample per URL)

**URL File Format:**
```
example.com
https://google.com
wikipedia.org
```
- One URL per line
- Automatically adds `https://` if missing
- Whitespace is stripped

### Configuration Constants

In `foreground_capture.py`:
```python
SAMPLE_NB = 20   # Target successful captures per URL
TRIES_NB = 25    # Maximum attempts per URL
IFACE = "veth0"  # Default network interface
NYM_PREFIX = "wfp"  # Nym service prefix
```

## Output Structure

```
dataset/
└── dataset-20250101-120000/
    ├── example.com/
    │   ├── 0_0/          # Capture 0, error code 0 (success)
    │   │   ├── capture.pcap
    │   │   ├── sslkeys.txt
    │   │   └── index.html
    │   ├── 1_0/          # Capture 1, success
    │   ├── 2_3/          # Capture 2, error code 3 (CloudFlare)
    │   └── ...
    └── google.com/
        └── ...
```

**Directory naming:** `{index}_{error_code}/`
- `index`: Capture attempt number (0-based)
- `error_code`: Result code (see below)

## Error Codes

### Success
- **0** - Successful capture

### CloudFlare / Security
- **3** - CloudFlare CAPTCHA, "Just a moment...", Access Denied, Service Unavailable

### HTTP Errors
- **4** - 403 Forbidden, 404 Not Found, 410 Gone

### Server Errors
- **5** - 521 Web server down, 522 Connection timeout, 523 Origin unreachable

### Network Errors
- **10** - DNS not found, NSS failure, connection failure
- **11** - Network timeout
- **12** - Proxy connection failure
- **123** - Script timeout (150s)
- **19** - Unknown error