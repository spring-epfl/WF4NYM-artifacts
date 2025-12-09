#!/usr/bin/env python3
"""
Traffic Processing Pipeline

End-to-end pipeline for processing raw pcap captures into training-ready
website fingerprinting data. Orchestrates:
1. PCAP extraction (process_packets.py)
2. Webpage-to-website aggregation (webpages-to-websites.py)
3. Format conversion for ML (tcp_to_ml.py - if needed)
4. Defense simulation (trace_extraction.py)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """
    Execute a command and handle errors.
    
    Args:
        cmd: List of command arguments
        description: Human-readable description of the step
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'=' * 60}")
    print(f"[*] {description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"[ERROR] Command not found: {cmd[0]}")
        return False


def validate_inputs(pcap_folder, datasets, websites_file, urls_file):
    """
    Validate that all required input files exist.
    
    Args:
        pcap_folder: Base folder containing pcap files
        datasets: List of dataset subdirectories
        websites_file: Path to websites list
        urls_file: Path to URLs list
    
    Returns:
        bool: True if all inputs valid
    """
    print("[*] Validating inputs...")
    
    # Check pcap folder
    if not os.path.isdir(pcap_folder):
        print(f"[ERROR] PCAP folder not found: {pcap_folder}")
        return False
    
    # Check dataset subdirectories
    for dataset in datasets:
        dataset_path = os.path.join(pcap_folder, dataset)
        if not os.path.isdir(dataset_path):
            print(f"[WARNING] Dataset subdirectory not found: {dataset_path}")
    
    # Check websites file
    if not os.path.isfile(websites_file):
        print(f"[ERROR] Websites file not found: {websites_file}")
        return False
    
    # Check URLs file
    if not os.path.isfile(urls_file):
        print(f"[ERROR] URLs file not found: {urls_file}")
        return False
    
    print("[OK] Input validation passed")
    return True


def step1_extract_pcaps(script_dir, pcap_folder, intermediate_folder, datasets, 
                       src_ip, dst_ips, no_dst_filter, ignore_port_9002):
    """
    Step 1: Extract tcpdump data from pcap files.
    
    Returns:
        bool: Success status
    """
    cmd = [
        sys.executable,
        os.path.join(script_dir, "process_packets.py"),
        pcap_folder,
        intermediate_folder,
        *datasets,
        "--src_ip", src_ip,
        "--dst_ips", *dst_ips,
    ]
    
    if no_dst_filter:
        cmd.append("--no_dst_filter")
    
    if ignore_port_9002:
        cmd.append("--ignore_port_9002")
    
    return run_command(cmd, "Step 1: Extract PCAP data")


def step2_aggregate_websites(script_dir, intermediate_folder, aggregated_folder,
                             websites_file, urls_file, min_count, min_valid_samples, max_lines):
    """
    Step 2: Aggregate individual pages into per-website files.
    
    Returns:
        bool: Success status
    """
    source_folder = os.path.join(intermediate_folder, "output-tcp")
    
    cmd = [
        sys.executable,
        os.path.join(script_dir, "webpages-to-websites.py"),
        "--source-folder", source_folder,
        "--output-folder", aggregated_folder,
        "--websites-file", websites_file,
        "--urls-file", urls_file,
        "--min-count", str(min_count),
        "--min-valid-samples", str(min_valid_samples),
        "--max-lines", str(max_lines),
    ]
    
    return run_command(cmd, "Step 2: Aggregate webpages to websites")


def step3_convert_to_ml_format(script_dir, aggregated_folder, ml_folder):
    """
    Step 3: Convert to ML-ready format (if tcp_to_ml.py is implemented).
    
    Returns:
        bool: Success status
    """
    tcp_to_ml_script = os.path.join(script_dir, "tcp_to_ml.py")
    
    # Check if tcp_to_ml.py is implemented
    if not os.path.isfile(tcp_to_ml_script):
        print("\n" + "=" * 60)
        print("[*] Step 3: Convert to ML format")
        print("=" * 60)
        print("[WARNING] tcp_to_ml.py not found - skipping this step")
        print("[*] Using aggregated data directly for next step")
        return True
    
    cmd = [
        sys.executable,
        tcp_to_ml_script,
        "--input-folder", aggregated_folder,
        "--output-folder", ml_folder,
    ]
    
    return run_command(cmd, "Step 3: Convert to ML format")


def step4_extract_traces(script_dir, input_folder, output_folder,
                        websites_file, urls_file, max_samples):
    """
    Step 4: Extract individual traces from aggregated data (trace_extraction.py).
    
    Returns:
        bool: Success status
    """
    cmd = [
        sys.executable,
        os.path.join(script_dir, "trace_extraction.py"),
        "--input-path", input_folder,
        "--output-path", output_folder,
        "--websites-file", websites_file,
        "--webpages-file", urls_file,
        "--max-samples", str(max_samples),
    ]
    print(cmd)
    
    return run_command(cmd, "Step 4: Extract individual traces")


def create_output_structure(base_output_folder):
    """
    Create output directory structure.
    
    Args:
        base_output_folder: Base output directory
    
    Returns:
        dict: Paths for each processing stage
    """
    paths = {
        'base': base_output_folder,
        'intermediate': os.path.join(base_output_folder, '1_extracted_pcaps'),
        'aggregated': os.path.join(base_output_folder, '2_aggregated_websites'),
        'ml_format': os.path.join(base_output_folder, '3_ml_format'),
        'traces': os.path.join(base_output_folder, '4_individual_traces'),
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths


def main():
    """Command-line interface for the processing pipeline."""
    parser = argparse.ArgumentParser(
        description='End-to-end pipeline for processing pcap files to training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic usage with single dataset
  python pipeline.py --pcap-folder /path/to/pcaps --datasets dataset1
  
  # Multiple datasets with custom output
  python pipeline.py --pcap-folder /path/to/pcaps --datasets dataset1 dataset2 \\
                     --output-folder /path/to/output
  
  # Apply defense scenarios
  python pipeline.py --pcap-folder /path/to/pcaps --datasets dataset1 \\
                     --defense-scenarios lm0 lm20 lm100 --skip-defense false
  
  # Skip specific steps
  python pipeline.py --pcap-folder /path/to/pcaps --datasets dataset1 \\
                     --skip-extraction true --skip-aggregation true
        """
    )
    
    # Input/Output
    parser.add_argument('--pcap-folder', type=str, required=True,
                       help='Base folder containing pcap files')
    parser.add_argument('--datasets', nargs='+', required=True,
                       help='List of dataset subdirectories to process')
    parser.add_argument('--output-folder', type=str, default='./processed_output',
                       help='Base output folder for all processing stages')
    
    # Configuration files
    parser.add_argument('--websites-file', type=str, default='websites.txt',
                       help='File containing website names (one per line)')
    parser.add_argument('--urls-file', type=str, default='final_urls.txt',
                       help='File containing URLs (one per line)')
    
    # Step 1: PCAP extraction
    parser.add_argument('--src-ip', type=str, default='10.1.1.1',
                       help='Source IP address')
    parser.add_argument('--dst-ips', nargs='+', default=['139.162.200.242'],
                       help='Destination IP addresses')
    parser.add_argument('--no-dst-filter', action='store_true', default=True,
                       help='Disable destination IP filtering')
    parser.add_argument('--ignore-port-9002', action='store_true',
                       help='Exclude packets with port 9002')
    
    # Step 2: Aggregation
    parser.add_argument('--min-count', type=int, default=10,
                       help='Minimum files per website')
    parser.add_argument('--min-valid-samples', type=int, default=10,
                       help='Minimum valid samples (21 lines)')
    parser.add_argument('--max-lines', type=int, default=20,
                       help='Maximum lines per file')
    
    # Step 4: Trace extraction
    parser.add_argument('--max-samples', type=int, default=20,
                       help='Maximum samples per webpage')
    
    # Pipeline control
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip PCAP extraction step')
    parser.add_argument('--skip-aggregation', action='store_true',
                       help='Skip webpage aggregation step')
    parser.add_argument('--skip-ml-conversion', action='store_true',
                       help='Skip ML format conversion step')
    parser.add_argument('--skip-trace-extraction', action='store_true',
                       help='Skip individual trace extraction step')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve relative paths
    websites_file = os.path.join(script_dir, args.websites_file) if not os.path.isabs(args.websites_file) else args.websites_file
    urls_file = os.path.join(script_dir, args.urls_file) if not os.path.isabs(args.urls_file) else args.urls_file
    
    # Validate inputs
    if not validate_inputs(args.pcap_folder, args.datasets, websites_file, urls_file):
        print("\n[ERROR] Input validation failed")
        return 1
    
    # Create output structure
    print(f"\n[*] Creating output structure in {args.output_folder}...")
    paths = create_output_structure(args.output_folder)
    print(f"[OK] Output directories created")
    
    # Pipeline execution
    print("\n" + "=" * 60)
    print("STARTING PIPELINE")
    print("=" * 60)
    
    success = True
    
    # Step 1: Extract PCAPs
    if not args.skip_extraction:
        success = step1_extract_pcaps(
            script_dir, args.pcap_folder, paths['intermediate'], args.datasets,
            args.src_ip, args.dst_ips, args.no_dst_filter, args.ignore_port_9002
        )
        if not success:
            print("\n[ERROR] Pipeline failed at Step 1")
            return 1
    else:
        print("\n[*] Skipping Step 1: PCAP extraction")
    
    # Step 2: Aggregate websites
    if not args.skip_aggregation:
        success = step2_aggregate_websites(
            script_dir, paths['intermediate'], paths['aggregated'],
            websites_file, urls_file, args.min_count, args.min_valid_samples, args.max_lines
        )
        if not success:
            print("\n[ERROR] Pipeline failed at Step 2")
            return 1
    else:
        print("\n[*] Skipping Step 2: Website aggregation")
    
    # Step 3: Convert to ML format (optional)
    if not args.skip_ml_conversion:
        success = step3_convert_to_ml_format(
            script_dir, paths['aggregated'], paths['ml_format']
        )
        if not success:
            print("\n[ERROR] Pipeline failed at Step 3")
            return 1
    else:
        print("\n[*] Skipping Step 3: ML format conversion")
    
    # Step 4: Extract individual traces
    if not args.skip_trace_extraction:
        # Use aggregated data as input (or aggregated_websites if step 2 was executed)
        input_for_traces = paths['aggregated'] if not args.skip_ml_conversion else paths['aggregated']
        
        success = step4_extract_traces(
            script_dir, input_for_traces, paths['traces'],
            websites_file, urls_file, args.max_samples
        )
        if not success:
            print("\n[ERROR] Pipeline failed at Step 4")
            return 1
    else:
        print("\n[*] Skipping Step 4: Individual trace extraction")
    
    # Pipeline complete
    print("\n" + "=" * 60)
    print("[OK] PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nOutput structure:")
    print(f"  Base: {paths['base']}")
    if not args.skip_extraction:
        print(f"  Extracted PCAPs: {paths['intermediate']}")
    if not args.skip_aggregation:
        print(f"  Aggregated websites: {paths['aggregated']}")
    if not args.skip_ml_conversion:
        print(f"  ML format: {paths['ml_format']}")
    if not args.skip_trace_extraction:
        print(f"  Individual traces: {paths['traces']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
