#!/usr/bin/env python3
"""
Overhead Analysis Script
Compute latency and bandwidth overhead for WTF4NYM defense configurations.

This script can either:
1. Load pre-computed traffic info from a pickle file
2. Process raw traffic data from directories

Usage:
    # Using pre-computed pickle file
    python overheads.py --pickle ../../data/overheads/infos.pkl
    
    # Processing from directory
    python overheads.py --directory ../../data/reduced_list/no_proxy --name no_proxy
    
    # Compare multiple configurations
    python overheads.py --pickle ../../data/overheads/infos.pkl --compare configuration00_default configuration01_lqp10
"""

import os
import argparse
import pickle
import numpy as np
from collections import defaultdict
from tabulate import tabulate


def get_info(directory, timestamp_min_max=None, trim_last_seconds=0):
    """
    Extract traffic information from directory containing trace files.
    
    Args:
        directory: Path to directory containing traffic trace files
        timestamp_min_max: Optional dict mapping URLs to (start, end) timestamp tuples
        trim_last_seconds: Number of seconds to trim from end of each trace
    
    Returns:
        Tuple of (time_info, incoming_byte_info, outgoing_byte_info) dicts
    """
    time_info = defaultdict(list)
    incoming_byte_info = defaultdict(list)
    outgoing_byte_info = defaultdict(list)
    
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        return time_info, incoming_byte_info, outgoing_byte_info
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if os.path.isfile(filepath):
            print(f"Processing file: {filename}")

            with open(filepath, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    
                    if len(parts) < 3:
                        continue
                    
                    url = parts[0]
                    timestamps_bytes = parts[2:]
                    
                    timestamps = []
                    incoming_byte_counts = []
                    outgoing_byte_counts = []
                    
                    for tb in timestamps_bytes:
                        # Skip malformed entries that don't contain ':'
                        if ':' not in tb:
                            continue
                        
                        split_result = tb.split(':')
                        if len(split_result) != 2:
                            continue
                        
                        try:
                            timestamp, nb_of_bytes = split_result
                            timestamp = float(timestamp)
                            nb_of_bytes = float(nb_of_bytes)
                        except ValueError:
                            continue
                        
                        timestamps.append(timestamp)
                        
                        if nb_of_bytes > 0:  # Consider only incoming packets
                            incoming_byte_counts.append(nb_of_bytes)
                        else:
                            nb_of_bytes = int(np.abs(nb_of_bytes))
                            outgoing_byte_counts.append(nb_of_bytes)

                    # Skip lines with no valid timestamps
                    if not timestamps:
                        continue

                    timestamp_start = timestamps[0]
                    timestamp_end = timestamps[-1]
                    
                    # If trimming last seconds is requested
                    if trim_last_seconds > 0:
                        # timestamps are in ms, trim_last_seconds is in seconds
                        trim_cutoff = timestamp_end - trim_last_seconds * 1000
                        # Only keep those timestamps strictly before the cutoff
                        valid_indices = [i for i, t in enumerate(timestamps) if t < trim_cutoff]
                        if valid_indices:
                            last_valid_index = valid_indices[-1]
                            timestamp_end = timestamps[last_valid_index]
                        else:
                            # If all timestamps are beyond cutoff, skip this line
                            continue
                    
                    if timestamp_min_max:
                        for t_start, t_end in timestamp_min_max[url]:
                            if timestamp_start == t_start:
                                timestamp_end = t_end
                                break
                    
                    end_index = timestamps.index(timestamp_end)
                
                    time_info[url].append((timestamp_start, timestamp_end))
                    incoming_byte_info[url].append(sum(incoming_byte_counts[:end_index]))
                    outgoing_byte_info[url].append(sum(outgoing_byte_counts[:end_index]))
    
    return time_info, incoming_byte_info, outgoing_byte_info


def compute_proportional_overhead(base_info, other_info):
    """
    Compute proportional overhead metrics for each configuration.
    
    Args:
        base_info: Tuple of (time_info, incoming_byte_info, outgoing_byte_info) for baseline
        other_info: Dict mapping names to (time_info, incoming_byte_info, outgoing_byte_info) tuples
    
    Returns:
        Dict mapping configuration names to overhead metrics
    """
    base_time_info, base_incoming_byte_info, base_outgoing_byte_info = base_info
    
    websites_set = [set(base_time_info.keys())] + [set(time_info.keys()) for time_info, _, _ in other_info.values()]
    common_websites = set.intersection(*websites_set)
    print(len(common_websites))
    
    overheads = defaultdict(lambda: defaultdict(list))
    
    for website in common_websites:
        
        median_latency_base = np.median([end - start for start, end in base_time_info[website]])
        median_incoming_bytes_base = np.median(base_incoming_byte_info[website])
        median_outgoing_bytes_base = np.median(base_outgoing_byte_info[website])
        bandwidth_base = np.median(base_incoming_byte_info[website]+base_outgoing_byte_info[website])
    
        for source_name, (time_info, incoming_byte_info, outgoing_byte_info) in other_info.items():
            
            median_latency_source = np.median([end - start for start, end in time_info[website]])
            latency_overhead = (median_latency_source / median_latency_base) - 1 if median_latency_base != 0 else np.nan
            overheads[source_name]["latency"].append(latency_overhead)
    
            median_incoming_bytes_source = np.median(incoming_byte_info[website])
            incoming_byte_overhead = (median_incoming_bytes_source / median_incoming_bytes_base) - 1 if median_incoming_bytes_base != 0 else np.nan
            overheads[source_name]["incoming_bytes"].append(incoming_byte_overhead)
    
            median_outgoing_bytes_source = np.median(outgoing_byte_info[website])
            outgoing_byte_overhead = (median_outgoing_bytes_source / median_outgoing_bytes_base) - 1 if median_outgoing_bytes_base != 0 else np.nan
            overheads[source_name]["outgoing_bytes"].append(outgoing_byte_overhead)

            bandwidth_source = np.median(incoming_byte_info[website]+outgoing_byte_info[website])
            bandwidth_overhead = (bandwidth_source / bandwidth_base) - 1 if bandwidth_base != 0 else np.nan 
            overheads[source_name]["bandwidth"].append(bandwidth_overhead)
    
    return overheads


def print_overheads_table(overheads_final):
    """
    Print overhead metrics in a formatted table.
    
    Args:
        overheads_final: Dict mapping configuration names to overhead metrics
    """
    table = []
    headers = ['latency', 'incoming_bytes', 'outgoing_bytes', 'bandwidth']
    for source, metrics in overheads_final.items():
        row = [source]
        for key in headers:
            value = metrics.get(key, 'N/A')
            if isinstance(value, (int, float)):
                row.append(f"{value:.2f}")
            else:
                row.append(value)
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="grid"))


def compute_and_print_overheads(infos_dict, fields, baseline_key="no_proxy"):
    """
    Compute and print overhead table for given fields compared to baseline.
    
    Args:
        infos_dict: Dictionary containing traffic information
        fields: List of field names to compute overheads for
        baseline_key: Key for the baseline dataset (default: "no_proxy")

    Returns:
        Dict of overhead metrics
    """
    other_info = {n: f for n, f in infos_dict.items() if n in fields}
    overheads = compute_proportional_overhead(infos_dict[baseline_key], other_info)
    overheads_final = defaultdict(lambda: defaultdict(int))
    overheads_final = {key: overheads_final[key] for key in fields}
    for source, oh in overheads.items():
        for metric, values in oh.items():
            overheads_final[source][metric] = np.median(values)
    
    print_overheads_table(overheads_final)
    return overheads_final


def list_available_configurations(infos_dict):
    """
    List all available configurations in the loaded data.
    
    Args:
        infos_dict: Dictionary containing traffic information
    """
    print("Available configurations:")
    for i, config in enumerate(sorted(infos_dict.keys()), 1):
        print(f"  {i}. {config}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute overhead metrics for WTF4NYM defense configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load pre-computed pickle and show all configurations
  python overheads.py --pickle ../../data/overheads/infos.pkl --list
  
  # Compare specific configurations
  python overheads.py --pickle ../../data/overheads/infos.pkl \\
      --compare configuration00_default configuration01_lqp10 configuration02_lqp40
  
  # Process directory and compute info
  python overheads.py --directory ../../data/reduced_list/no_proxy --name no_proxy
  
  # Compare Tor and Nym networks
  python overheads.py --pickle ../../data/overheads/infos.pkl \\
      --compare tor_reduced nym_labnet_reduced nym_mainnet_reduced
        """
    )
    
    parser.add_argument('--pickle', type=str, help='Path to pickle file containing pre-computed traffic info')
    parser.add_argument('--directory', type=str, help='Path to directory containing traffic trace files')
    parser.add_argument('--name', type=str, help='Name for the configuration (required with --directory)')
    parser.add_argument('--baseline', type=str, default='no_proxy', 
                        help='Baseline configuration name (default: no_proxy)')
    parser.add_argument('--compare', nargs='+', help='List of configuration names to compare against baseline')
    parser.add_argument('--list', action='store_true', help='List all available configurations')
    parser.add_argument('--trim-last-seconds', type=int, default=0,
                        help='Number of seconds to trim from end of traces')
    
    args = parser.parse_args()
    
    infos_dict = {}
    
    # Load from pickle file
    if args.pickle:
        if not os.path.exists(args.pickle):
            print(f"Error: Pickle file {args.pickle} does not exist")
            return 1
        
        print(f"Loading data from {args.pickle}...")
        with open(args.pickle, 'rb') as f:
            infos_dict = pickle.load(f)
        print(f"Loaded {len(infos_dict)} configurations")
    
    # Process directory
    if args.directory:
        if not args.name:
            print("Error: --name is required when using --directory")
            return 1
        
        print(f"Processing directory: {args.directory}")
        info = get_info(args.directory, trim_last_seconds=args.trim_last_seconds)
        infos_dict[args.name] = info
        print(f"Processed configuration: {args.name}")
    
    if not infos_dict:
        print("Error: No data loaded. Use --pickle or --directory")
        parser.print_help()
        return 1
    
    # List configurations
    if args.list:
        list_available_configurations(infos_dict)
        return 0
    
    # Compare configurations
    if args.compare:
        print(f"\nComparing configurations against baseline: {args.baseline}")
        print("=" * 80)
        overheads = compute_and_print_overheads(infos_dict, args.compare, args.baseline)
        return 0
    
    # If no specific action, list available configurations
    if not args.list and not args.compare:
        list_available_configurations(infos_dict)
        print("\nUse --compare to analyze specific configurations")
        print("Use --list to see all available configurations")
    
    return 0


if __name__ == "__main__":
    exit(main())
