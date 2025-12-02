#!/usr/bin/env python3
"""
Traffic Data Aggregation Script

Aggregates individual page captures into per-website files for training.
Filters websites based on minimum sample count and valid URL mapping.
"""

import os
import argparse
import numpy as np
import shutil
from os import listdir
from os.path import isfile, join
from collections import defaultdict, Counter


def load_websites_list(websites_file):
    """
    Load website names from file.
    
    Args:
        websites_file: Path to file containing website names (one per line)
    
    Returns:
        list: Website names
    """
    with open(websites_file) as f:
        return [line.rstrip('\n') for line in f.readlines()]


def load_url_mapping(urls_file):
    """
    Load URL mapping from file.
    
    Args:
        urls_file: Path to file containing URLs (one per line)
    
    Returns:
        dict: Mapping from folder name to URL
    """
    folder_to_urls = {}
    with open(urls_file) as f:
        for line in f:
            url = line.rstrip('\n')
            folder_name = url.replace("https://", "").replace("/", "")[:100]
            folder_to_urls[folder_name] = url
    return folder_to_urls


def group_files_by_website(source_folder, site_array):
    """
    Group capture files by website.
    
    Args:
        source_folder: Directory containing capture files
        site_array: List of website names to match
    
    Returns:
        dict: Website name -> list of matching files
    """
    d = defaultdict(list)
    onlyfiles = [f for f in listdir(source_folder) if isfile(join(source_folder, f))]
    
    for file in onlyfiles:
        for site in site_array:
            if site in file:
                d[site].append(file)
                break
    
    return d


def filter_websites_by_count(website_files, min_count=10):
    """
    Filter websites by minimum sample count.
    
    Args:
        website_files: Dict of website -> list of files
        min_count: Minimum number of files required
    
    Returns:
        tuple: (websites_to_keep, websites_to_drop)
    """
    lengths = {website: len(files) for website, files in website_files.items()}
    
    websites_to_keep = [w for w, l in lengths.items() if l >= min_count]
    websites_to_drop = [(w, l) for w, l in lengths.items() if l < min_count]
    
    return websites_to_keep, websites_to_drop


def count_lines_per_file(source_folder, website_files, websites_to_keep):
    """
    Count lines in each file to ensure sufficient data.
    
    Args:
        source_folder: Directory containing capture files
        website_files: Dict of website -> list of files
        websites_to_keep: List of websites to process
    
    Returns:
        dict: website -> {filename: line_count}
    """
    websites = {}
    
    for website in websites_to_keep:
        websites[website] = {}
        for file in website_files[website]:
            line_count = 1
            with open(os.path.join(source_folder, file), "r") as f:
                for line in f:
                    line_count += 1
            websites[website][file] = min(21, line_count)
    
    return websites


def filter_by_url_mapping(websites, folder_to_urls):
    """
    Remove files that don't have valid URL mappings.
    
    Args:
        websites: Dict of website -> {filename: line_count}
        folder_to_urls: Dict of folder name -> URL
    
    Returns:
        dict: Filtered websites dict
    """
    for w, sp in websites.items():
        if len(sp) > 10:
            i = 0
            while len(sp) > 10 and i < len(sp):
                key = list(sp.keys())[i]
                if key not in folder_to_urls:
                    sp.pop(key)
                else:
                    i += 1
    
    return websites


def get_final_websites(websites, min_valid_samples=10):
    """
    Get websites with sufficient valid samples (21 lines each).
    
    Args:
        websites: Dict of website -> {filename: line_count}
        min_valid_samples: Minimum number of valid samples required
    
    Returns:
        tuple: (final_websites, rejected_websites)
    """
    counters = defaultdict(Counter)
    for w, p in websites.items():
        counters[w] = Counter(p.values())
    
    final_websites = [w for w, c in counters.items() if c.get(21, 0) >= min_valid_samples]
    rejected_websites = [w for w, c in counters.items() if c.get(21, 0) < min_valid_samples]
    
    return final_websites, rejected_websites


def aggregate_website_data(source_folder, output_folder, website_files, websites, final_websites, max_lines=20):
    """
    Aggregate individual page captures into per-website files.
    
    Args:
        source_folder: Directory containing individual capture files
        output_folder: Directory to write aggregated files
        website_files: Dict of website -> list of files
        websites: Dict of website -> {filename: line_count}
        final_websites: List of websites to process
        max_lines: Maximum lines to read from each file
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for website, pages in website_files.items():
        if website in final_websites:
            content = ""
            for page in pages:
                if website in websites and page in websites[website] and websites[website][page] == 21:
                    with open(os.path.join(source_folder, page), "r") as f:
                        # Read the first max_lines lines of the file
                        for i in range(max_lines):
                            line = f.readline()
                            if not line:
                                break  # Exit if there are fewer than max_lines
                            content += line
                else:
                    print(f"[WARNING] Skipping {website}/{page}")
            
            with open(os.path.join(output_folder, website), "a") as file:
                file.write(content)
            
            print(f"[OK] Aggregated {website}")


def process_traffic_data(source_folder, output_folder, websites_file, urls_file, 
                         min_count=10, min_valid_samples=10, max_lines=20):
    """
    Main function to process and aggregate traffic data.
    
    Args:
        source_folder: Directory containing individual capture files
        output_folder: Directory to write aggregated files
        websites_file: Path to file containing website names
        urls_file: Path to file containing URLs
        min_count: Minimum number of files per website
        min_valid_samples: Minimum number of valid samples (21 lines)
        max_lines: Maximum lines to read from each file
    
    Returns:
        dict: Statistics about the processing
    """
    print("[*] Loading website list...")
    site_array = load_websites_list(websites_file)
    print(f"    Loaded {len(site_array)} websites")
    
    print("[*] Loading URL mapping...")
    folder_to_urls = load_url_mapping(urls_file)
    print(f"    Loaded {len(folder_to_urls)} URL mappings")
    
    print("[*] Grouping files by website...")
    website_files = group_files_by_website(source_folder, site_array)
    print(f"    Found files for {len(website_files)} websites")
    
    print("[*] Filtering by minimum count...")
    websites_to_keep, websites_to_drop = filter_websites_by_count(website_files, min_count)
    print(f"    Keeping {len(websites_to_keep)} websites")
    print(f"    Dropping {len(websites_to_drop)} websites (insufficient samples)")
    
    print("[*] Counting lines per file...")
    websites = count_lines_per_file(source_folder, website_files, websites_to_keep)
    
    print("[*] Filtering by URL mapping...")
    websites = filter_by_url_mapping(websites, folder_to_urls)
    
    print("[*] Finding final websites with sufficient valid samples...")
    final_websites, rejected_websites = get_final_websites(websites, min_valid_samples)
    print(f"    Final: {len(final_websites)} websites")
    print(f"    Rejected: {len(rejected_websites)} websites (insufficient valid samples)")
    
    print(f"[*] Aggregating data to {output_folder}...")
    aggregate_website_data(source_folder, output_folder, website_files, websites, 
                          final_websites, max_lines)
    
    stats = {
        'total_websites': len(site_array),
        'websites_with_files': len(website_files),
        'websites_after_count_filter': len(websites_to_keep),
        'final_websites': len(final_websites),
        'rejected_websites': len(rejected_websites),
        'dropped_websites': len(websites_to_drop)
    }
    
    print("\n[OK] Processing complete!")
    print(f"    Total websites processed: {stats['total_websites']}")
    print(f"    Final aggregated websites: {stats['final_websites']}")
    
    return stats


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Aggregate individual page captures into per-website files'
    )
    
    parser.add_argument('--source-folder', type=str, 
                       default='processed_data/tor/output-tcp',
                       help='Directory containing individual capture files')
    parser.add_argument('--output-folder', type=str,
                       default='processed_data/tor/foreground/tcp',
                       help='Directory to write aggregated files')
    parser.add_argument('--websites-file', type=str,
                       default='websites.txt',
                       help='File containing website names (one per line)')
    parser.add_argument('--urls-file', type=str,
                       default='final_urls.txt',
                       help='File containing URLs (one per line)')
    parser.add_argument('--min-count', type=int, default=10,
                       help='Minimum number of files per website')
    parser.add_argument('--min-valid-samples', type=int, default=10,
                       help='Minimum number of valid samples (21 lines)')
    parser.add_argument('--max-lines', type=int, default=20,
                       help='Maximum lines to read from each file')
    
    args = parser.parse_args()
    
    stats = process_traffic_data(
        source_folder=args.source_folder,
        output_folder=args.output_folder,
        websites_file=args.websites_file,
        urls_file=args.urls_file,
        min_count=args.min_count,
        min_valid_samples=args.min_valid_samples,
        max_lines=args.max_lines
    )
    
    return stats


if __name__ == "__main__":
    main()

