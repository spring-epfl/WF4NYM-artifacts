#!/usr/bin/env python3
"""
Traffic Leakage Analysis Script

Processes captured traffic data to extract timing and volume information
for website fingerprinting analysis. Converts aggregated traffic captures
into individual page-level traces with normalized timestamps.
"""

import os
import argparse
import numpy as np 
from collections import Counter


def load_websites_list(websites_file):
    """
    Load list of websites to process.
    
    Args:
        websites_file: Path to file containing website names (one per line)
    
    Returns:
        list: Website names
    """
    with open(websites_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


def load_webpages_list(webpages_file):
    """
    Load list of valid webpages for filtering.
    
    Args:
        webpages_file: Path to file containing URLs (one per line)
    
    Returns:
        list: Normalized webpage names
    """
    with open(webpages_file) as f:
        return [line.rstrip('\n').replace("https://", "").replace("/", "") 
                for line in f.readlines()]


def process_scenario(tcp_path, output_path, final_websites, webpages, 
                     max_samples_per_page=20, direction_map=None):
    """
    Process traffic captures for a single scenario.
    
    Args:
        tcp_path: Directory containing aggregated traffic captures
        output_path: Directory to write processed individual traces
        final_websites: List of websites to process
        webpages: List of valid webpage names for filtering
        max_samples_per_page: Maximum samples to extract per webpage
        direction_map: Optional direction mapping (default: client-to-server=-1, server-to-client=1)
    
    Returns:
        dict: Processing statistics
    """
    if direction_map is None:
        direction_map = {"client-to-server": -1, "server-to-client": 1}
    
    os.makedirs(output_path, exist_ok=True)
    
    counter = Counter()
    stats = {
        'total_websites': len(final_websites),
        'processed_websites': 0,
        'total_traces': 0,
        'skipped_websites': 0
    }
    
    for website_index, website in enumerate(final_websites):
        print(f"[*] Processing {website_index + 1}/{len(final_websites)}: {website}")
        
        website_file = os.path.join(tcp_path, website)
        if not os.path.exists(website_file):
            print(f"    [WARNING] File not found, skipping")
            stats['skipped_websites'] += 1
            continue
        
        page_number = 0
        url_set = set()
        
        with open(website_file, "r") as f:
            for line in f:
                webpage = line.split(" ")[0]
                counter[webpage] += 1
                
                # Only process if within sample limit and webpage is valid
                if counter[webpage] <= max_samples_per_page and webpage in webpages:
                    line_content = process_line(line, direction_map, website)
                    
                    if line_content:
                        output_file = os.path.join(output_path, f"{website_index}-{page_number}")
                        with open(output_file, "w") as out_f:
                            out_f.write(line_content)
                        
                        page_number += 1
                        url_set.add(webpage)
                        stats['total_traces'] += 1
        
        print(f"    Extracted {page_number} traces from {len(url_set)} unique URLs")
        stats['processed_websites'] += 1
    
    return stats


def process_line(line, direction_map, website):
    """
    Process a single line of traffic data.
    
    Args:
        line: Line containing traffic data
        direction_map: Direction mapping dictionary
        website: Website name (for error reporting)
    
    Returns:
        str: Processed line content with normalized timestamps and directions
    """
    line_content = ""
    points = line.split(" ")[2:]
    
    if not points:
        return ""
    
    # Parse base time from first point
    try:
        base_time = float(points[0].split(":")[0])
    except (IndexError, ValueError):
        print(f"    [WARNING] Invalid first point in {website}")
        return ""
    
    for point in points:
        parts = point.split(":")
        if len(parts) != 2:
            print(f"    [WARNING] Invalid point format in {website}: {point}")
            continue
        
        try:
            time = float(parts[0])
            volume_raw = int(parts[1])
        except ValueError:
            print(f"    [WARNING] Invalid numeric values in {website}: {point}")
            continue
        
        # Normalize time to seconds relative to first packet
        rel_time = (time - base_time) / 1000.0
        
        # Determine direction: -1 is OUT (client-to-server), 1 is IN (server-to-client)
        direction = np.sign(volume_raw)
        direction = 1 if direction == direction_map["client-to-server"] else -1
        
        # Get absolute volume
        volume = np.abs(volume_raw)
        
        # Write: timestamp (seconds) \t signed_volume (direction * bytes)
        line_content += f"{rel_time}\t{int(direction * volume)}\n"
    
    return line_content


def process_from_aggregated(input_path, output_path, websites_file, webpages_file, max_samples_per_page=20):
    """
    Process aggregated website files into individual traces.
    
    Args:
        input_path: Directory containing aggregated website files
        output_path: Directory to write processed individual traces
        websites_file: Path to file containing website names (optional, will use all files if not restrictive)
        webpages_file: Path to file containing valid webpages
        max_samples_per_page: Maximum samples per webpage
    
    Returns:
        dict: Processing statistics
    """
    print("[*] Loading webpage list for filtering...")
    webpages = load_webpages_list(webpages_file)
    print(f"    Loaded {len(webpages)} valid webpages")
    
    # Get all website files from input directory (exclude pickle and other binary files)
    print(f"[*] Scanning input directory: {input_path}")
    all_files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    # Filter out pickle files and other non-text files
    all_website_files = [f for f in all_files if not f.endswith('.pkl') and not f.startswith('.')]
    print(f"    Found {len(all_website_files)} website files (excluding {len(all_files) - len(all_website_files)} non-text files)")
    
    print(f"\n[*] Processing aggregated data from: {input_path}")
    print("=" * 60)
    
    stats = process_scenario(
        tcp_path=input_path,
        output_path=output_path,
        final_websites=all_website_files,
        webpages=webpages,
        max_samples_per_page=max_samples_per_page
    )
    
    print(f"\n[OK] Processing complete:")
    print(f"    Processed: {stats['processed_websites']}/{stats['total_websites']} websites")
    print(f"    Total traces: {stats['total_traces']}")
    print(f"    Skipped: {stats['skipped_websites']} websites")
    
    return stats


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Process aggregated traffic captures into individual traces for website fingerprinting analysis'
    )
    
    parser.add_argument('--input-path', type=str, required=True,
                       help='Directory containing aggregated website files')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Directory to write processed individual traces')
    parser.add_argument('--websites-file', type=str,
                       default='websites.txt',
                       help='File containing website names to process')
    parser.add_argument('--webpages-file', type=str,
                       default='final_urls.txt',
                       help='File containing valid webpage URLs for filtering')
    parser.add_argument('--max-samples', type=int, default=20,
                       help='Maximum samples to extract per webpage (default: 20)')
    
    args = parser.parse_args()
    
    stats = process_from_aggregated(
        input_path=args.input_path,
        output_path=args.output_path,
        websites_file=args.websites_file,
        webpages_file=args.webpages_file,
        max_samples_per_page=args.max_samples
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("[OK] Processing complete!")
    print("=" * 60)
    print(f"Websites: {stats['processed_websites']}/{stats['total_websites']}")
    print(f"Traces: {stats['total_traces']}")
    
    return stats


if __name__ == "__main__":
    main()
                     
            