"""
Foreground Capture Script for Website Fingerprinting

Automated batch capture tool that fetches multiple URLs with network traffic recording.
Supports Nym proxy routing and handles retries for failed captures.

Features:
- Batch processing of URL lists
- Automatic retry logic (up to TRIES_NB attempts per URL)
- Organized output directory structure
- Error code tracking and file organization
- Configurable sample count (default: 20 successful captures per URL)

Output Structure:
    dataset/dataset-YYYYMMDD-HHMMSS/
        ├── example.com/
        │   ├── 0_0/          # index_errorcode (0=success)
        │   │   ├── capture.pcap
        │   │   ├── sslkeys.txt
        │   │   └── index.html
        │   ├── 1_0/
        │   └── ...
        └── another-site.com/
            └── ...
"""

import os
import time
import signal
import shutil
import argparse
import multiprocessing

from datetime import datetime
from firefox_query import *



URLS = "urls"
OUTPUT_PATH = "dataset"
NYM = True
NYM_PREFIX = "wfp"
NYM_SETUP = "1"
IFACE = "veth0"
FILES_TO_SAVE = ["capture.pcap", "sslkeys.txt", "index.html"]
SAMPLE_NB = 20  # Target number of successful captures per URL
TRIES_NB = 25   # Maximum attempts per URL


def get_urls(url_file):
    """
    Load and normalize URLs from file.
    
    Args:
        url_file: Path to file containing URLs (one per line)
    
    Returns:
        list: Normalized URLs with https:// prefix and trailing slash
    """
    f = open(url_file, "r")
    # separate elements that are on different lines
    urls = f.read().split("\n")
    # remove heading and trailing spaces and remove empty elements
    urls = list(filter(None, [url.strip() for url in urls]))
    # add https:// where it was forgotten
    urls = [f"https://{url}" if not url.startswith("https://") else url for url in urls]
    urls = [url if url.endswith("/") else url + "/" for url in urls]
    return urls    


def do_nym_capture(driver, url, output_path, iface, nym, nym_prefix, nym_setup):
    """
    Execute single capture with Nym proxy restart.
    
    Restarts Nym service before each capture to ensure clean proxy state.
    
    Args:
        driver: Firefox webdriver instance
        url: URL to fetch
        output_path: Output directory
        iface: Network interface
        nym: Whether to use Nym proxy
        nym_prefix: Nym service name prefix
        nym_setup: Nym setup identifier
    
    Returns:
        int: Error code from capture
    """

    stop_nym(nym_prefix, nym_setup)
    
    start_nym(nym_prefix, nym_setup)
    
    title, exception = capture(iface, driver, url, output_path)

    stop_nym(nym_prefix, nym_setup)

    error_code = 0
    
    if exception:
        error_code = handle_exception(title)
    else:
        error_code = handle_error(title)

    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] Done, quitting with exit code {error_code}.")
    
    return error_code


def process_error_code(error_code, url, index, output_path):
    """
    Organize captured files into directory structure.
    
    Moves capture files to organized directory: {url}/{index}_{error_code}/
    Files are made read-only after moving.
    
    Args:
        error_code: Exit code from capture
        url: Captured URL
        index: Capture attempt number
        output_path: Base output directory
    """
    path = f'{output_path}{url.replace("https://", "").replace("/", "")}/{index}_{error_code}/'
    print(path)
    os.makedirs(path, exist_ok=True) 
    for file in FILES_TO_SAVE:
        if os.path.isfile(f"{output_path}{file}"):
            shutil.move(f"{output_path}{file}", path)
            os.chmod(f"{path}{file}", 0o444)
            

def main():
    """
    Main capture loop - processes all URLs from list.
    
    For each URL:
    - Attempts up to TRIES_NB captures
    - Stops early if SAMPLE_NB successful captures achieved
    - Stops early if remaining attempts can't reach SAMPLE_NB
    
    Usage:
        python foreground_capture.py --urls urls.txt --nym_setup 1 --interface veth0
        python foreground_capture.py --background  # For background traffic (1 sample per URL)
    """

    parser = argparse.ArgumentParser(description='Make captures')
    parser.add_argument('--urls', dest='urls_list_path', default=URLS,
                        help='path to the urls list')
    parser.add_argument('--nym_setup', dest='nym_setup', default=NYM_SETUP,
                        help='path to the urls list')
    parser.add_argument('--interface', dest='iface', default=IFACE,
                        help='interface used')

    parser.add_argument("--background", action="store_true", 
                        help="Do backgroud capture") 

    
    args = parser.parse_args()

    urls = get_urls(args.urls_list_path)

    output_path = f'{OUTPUT_PATH}/dataset-{time.strftime("%Y%m%d-%H%M%S")}/'
    
    os.makedirs(output_path, exist_ok=True) 

    
    for url in urls:
        pages_done = 0
        for index in range(TRIES_NB if not args.background else 1):
            nym = NYM
            nym_prefix = NYM_PREFIX
            try:
                driver = setup_browser(output_path, nym=nym, nym_setup=args.nym_setup)
                error_code = do_nym_capture(driver, url, output_path, args.iface, nym, nym_prefix, args.nym_setup)
                process_error_code(error_code, url, index, output_path)
                pages_done = pages_done + (1 if error_code == 0 else 0)
            except Exception as e:
                print("error")
            if pages_done >= SAMPLE_NB or TRIES_NB - index + pages_done < SAMPLE_NB:
                break
            

if __name__ == "__main__":
    main()
