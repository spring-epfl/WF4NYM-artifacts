"""
Firefox Query Script for Website Fingerprinting Captures

This script automates web page fetching using Firefox with optional Nym proxy routing,
while capturing network traffic using dumpcap. It handles various error conditions and
timeouts to ensure robust data collection.

Key Features:
- Automated Firefox browser control via Selenium
- Network traffic capture using dumpcap
- Optional Nym proxy integration
- Error handling for network failures, CloudFlare blocks, and timeouts
- SSL/TLS key logging for decryption
"""

from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium import webdriver
import sys
import os
import subprocess
import time
import shutil
from datetime import datetime
import argparse

import contextlib 
import signal 
import time
from types import FrameType 
from typing import Generator


def start_dumpcap(interface, pcap_output):
    """
    Start dumpcap packet capture on specified interface.
    
    Args:
        interface: Network interface name (e.g., 'veth0')
        pcap_output: Output path for pcap file
    
    Returns:
        subprocess.Popen: Running dumpcap process
    """
    bin_path = shutil.which("dumpcap")
    proc = subprocess.Popen(
        [
            bin_path,
            "-i",
            interface,
            "-w",
            pcap_output,
        ]
    )
    time.sleep(1)
    return proc


def stop_dumpcap(proc):
    """Terminate dumpcap process gracefully."""
    try:
        proc.terminate()
    except Exception as e:
        t = datetime.now().strftime("%H:%M:%S")
        print(f"[{t}] Error while killing process, continuing anyway : {e}")


class TimeoutError(Exception):
    pass


def alarm_handler (signum: int, frame: FrameType) -> None:
    raise TimeoutError("Timeout")


@contextlib. contextmanager
def timeout(s: int)->Generator[None, None, None]:
    """
    Context manager for timeout enforcement using SIGALRM.
    
    Args:
        s: Timeout duration in seconds
    
    Raises:
        TimeoutError: If operation exceeds timeout
    """
    orig = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(s)
    try:
        yield 
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, orig)


def start_nym(prefix, index):
    """
    Start Nym proxy service via systemd.
    
    Args:
        prefix: Service name prefix (e.g., 'wfp')
        index: Service index (e.g., '1')
    """
    proxy_service = prefix + "-proxy" + index
    status_proxy = subprocess.check_output(f"systemctl start {proxy_service}", shell=True)
    time.sleep(5)
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] Nym started")


def stop_nym(prefix, index):
    """Stop Nym proxy service via systemd."""
    proxy_service = prefix + "-proxy" + index
    status_proxy = subprocess.check_output(f"systemctl stop {proxy_service}", shell=True)
    time.sleep(5)


def add_protocol(url):
    url = url.strip()
    if not url.startswith("https://"):
        url = "https://" + url
    return url


def setup_browser(output_path, nym=True, nym_setup=""):
    """
    Initialize Firefox browser with custom profile and proxy settings.
    
    Args:
        output_path: Directory for logs and SSL keys
        nym: Enable Nym SOCKS5 proxy if True
        nym_setup: Nym setup identifier for profile isolation
    
    Returns:
        webdriver.Firefox: Configured Firefox driver instance
    """
    profile_url = "./profiles" + nym_setup

    # nuke and recreate profile
    shutil.rmtree(profile_url, ignore_errors=True)
    os.makedirs(profile_url)
    shutil.copyfile("prefs_non_quic_notracking.js", profile_url + "/prefs.js")

    # Capture loop
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] Starting browser")

    os.environ["SSLKEYLOGFILE"] = output_path + "sslkeys.txt"

    options = FirefoxOptions()
    options.add_argument("-headless")
    options.add_argument("-devtools")
    options.add_argument("-profile")
    options.add_argument(profile_url)
    service = FirefoxService(log_path=output_path + "geckodriver.log")

    # Nym
    if nym:
        options.set_preference("network.proxy.type", 1)
        options.set_preference("network.proxy.socks", "127.0.0.1")
        options.set_preference("network.proxy.socks_port", 1080)

    driver = webdriver.Firefox(options=options, service=service)
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] Firefox started")

    return driver


def capture(iface, driver, url, output_path, subpath=""):
    """
    Perform network capture while fetching a URL.
    
    Args:
        iface: Network interface to capture on
        driver: Firefox webdriver instance
        url: URL to fetch
        output_path: Base output directory
        subpath: Optional subdirectory within output_path
    
    Returns:
        tuple: (page_title, exception_occurred)
    """
    # capture
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] Starting dumpcap")
    proc = start_dumpcap(iface, output_path + subpath + "capture.pcap")
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] Dumpcap started")

    # fetching url
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] Fetching url : ", url)
    exception = False
    
    try:
        with timeout(150):
            driver.get(url)
        with open(f"{output_path}index.html", "w", encoding='utf-8') as f:
            f.write(driver.page_source)
        title = driver.title
    except Exception as e:
        exception = True
        title = str(e)
    finally:
        t = datetime.now().strftime("%H:%M:%S")
        print(f"[{t}]  url fetched: ", url)
        # stopping capture and browser
        stop_dumpcap(proc)
        os.system('killall firefox')
        #driver.quit()
    return title, exception


def handle_exception(message):
    """
    Map exception messages to error codes.
    
    Returns:
        int: Error code (10=DNS/network fail, 11=timeout, 12=proxy fail, etc.)
    """
    if "dnsNotFound" in message:
        print("Network error : Website not found")
        return 10
    if "nssFailure2" in message:
        print("Network error : nssFailure2")
        return 10
    if "netTimeout" in message:
        print("Network error : Server takes too long")
        return 11
    if "connectionFailure" in message:
        print("Network Error : We can't connect to the server")
        return 10
    if "proxyConnectFailure" in message:
        print("Configured proxy is not on")
        return 12
    if "Timeout" in message:
        print("Timeout 150s")
        return 123
    print("Unknown Error : " + message)
    return 19


def handle_error(title):
    """
    Parse page title for common error conditions.
    
    Returns:
        int: Error code (0=success, 3=CloudFlare, 4=40x errors, 5=50x errors)
    """
    # CloudFlare errors
    if "Attention Required!" in title and "Cloudflare" in title:
        print("Cloudflare : Captcha : " + title)
        return 3
    if "Just a moment..." in title:
        print("CloudFlare : just a moment : " + title)
        return 3
    if "Access denied" in title:
        print("CloudFlare : Access Denied : " + title)
        return 3
    if "Service Unavailable" in title:
        print("Serivce Unavailable : " + title)
        return 3

    # 40x errors
    if "403 Forbidden" in title:
        print("Error 403 : " + title)
        return 4

    if (
        "Not found" in title
        or "Not Found" in title
        or "not found" in title
        or "404" in title
        or "The page you were looking for doesn't exist" in title
        or "This page could not be found" in title
        or "Sorry, page unavailable" in title
        or "Seite nicht gefunden" in title
        or "Page non trouvée" in title
    ):
        print("Error 404 : " + title)
        return 4

    if "410 Gone" in title:
        print("Error 410 : " + title)
        return 4

    # 50x errors
    if "521: Web server is down" in title:
        print("Error 521 : " + title)
        return 5

    if "522: Connection timed out" in title:
        print("Error 522 : " + title)
        return 5

    if "523: Origin is unreachable" in title:
        print("Error 523 : " + title)
        return 5

    return 0


def main():
    """
    Main entry point for single URL capture.
    
    Usage:
        python firefox_query.py -u example.com -i veth0 -o ./output --nym_setup 1
    
    Exit codes:
        0: Success
        3: CloudFlare block
        4: 404/403 errors
        5: 50x server errors
        10: Network/DNS failures
        11: Connection timeout
        12: Proxy connection failure
    """
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "--url", "-u", type=str, help="Url to capture", required=True
    )
    argParser.add_argument(
        "--iface", "-i", type=str, help="Iface to capture on", required=True
    )
    argParser.add_argument(
        "--output", "-o", type=str, help="Output directory", default="output"
    )
    argParser.add_argument(
        "--nym_prefix", type=str, help="Path to nym exec binary", default="wfp"
    )
    argParser.add_argument(
        "--nym_setup", type=str, help="Nym setup index (1/2)", default="1"
    )
    argParser.add_argument("--no_nym", action="store_true", help="Capture without nym")

    args = argParser.parse_args()

    url = args.url if args.url.startswith("https://") else "https://" + args.url
    iface = args.iface
    output_path = args.output if args.output.endswith("/") else args.output + "/"
    os.makedirs(output_path, exist_ok=True)
    nym = not args.no_nym

    # capture main part
    if nym:
        start_nym(args.nym_prefix, args.nym_setup)
    driver = setup_browser(output_path, nym=nym, nym_setup=args.nym_setup)
    title, exception = capture(iface, driver, url, output_path)
    if nym:
        stop_nym(args.nym_prefix, args.nym_setup)

    error_code = 0
    if exception:
        error_code = handle_exception(title)
    else:
        error_code = handle_error(title)

    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] Done, quitting with exit code {error_code}.")
    sys.exit(error_code)


if __name__ == "__main__":
    main()
