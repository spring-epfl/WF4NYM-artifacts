#!/usr/bin/env python3
"""
Data Extraction Script for MixMatch Training
Extracts and preprocesses traffic data from capture files.
"""

import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pickle
import json
import argparse
import warnings
warnings.filterwarnings('ignore')


def parse_traffic_file(filepath):
    """Parse traffic file: <webpage> <capture#> <total_bytes> <timestamp:bytes>..."""
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                webpage_name = parts[0]
                capture_number = parts[1]
                total_bytes = int(parts[2])
                
                timestamps = []
                byte_counts = []
                
                for i in range(3, len(parts)):
                    if ':' in parts[i]:
                        try:
                            timestamp, bytes_val = parts[i].split(':')
                            timestamps.append(float(timestamp))
                            byte_counts.append(int(bytes_val))
                        except ValueError:
                            continue
                
                if timestamps:
                    data.append({
                        'webpage_name': webpage_name,
                        'capture_number': capture_number,
                        'total_bytes': total_bytes,
                        'timestamps': timestamps,
                        'byte_counts': byte_counts,
                        'num_packets': len(timestamps),
                        'duration': max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0,
                        'avg_packet_size': np.mean(byte_counts) if byte_counts else 0
                    })
    except Exception as e:
        print(f"[ERROR] Parsing {filepath}: {e}")
    
    return data


def load_traffic_data(data_dir):
    """Load all traffic data from directory"""
    all_data = []
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory not found: {data_dir}")
        return []
    
    files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    print(f"[*] Processing {len(files)} files from {data_dir}")
    
    for i, filename in enumerate(files):
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i}/{len(files)} files...")
        
        filepath = join(data_dir, filename)
        file_data = parse_traffic_file(filepath)
        all_data.extend(file_data)
    
    print(f"[OK] Loaded {len(all_data)} traffic records")
    return all_data


def detect_traffic_bursts(record, min_burst_packets=5):
    """
    MAD-based burst detection for incoming traffic
    Uses MAD factor 3.0 for volume, 2.5 for density (50ms windows)
    """
    timestamps = np.array(record['timestamps'])
    byte_counts = np.array(record['byte_counts'])
    
    if len(timestamps) <= min_burst_packets:
        return record
    
    timestamps = timestamps - timestamps[0]  # Normalize to start from 0
    
    incoming_mask = byte_counts > 0
    incoming_timestamps = timestamps[incoming_mask]
    incoming_bytes = byte_counts[incoming_mask]
    
    if len(incoming_timestamps) < min_burst_packets:
        return record

    # Method 1: Volume-based burst detection
    window_size = max(5, len(incoming_timestamps) // 20)
    sliding_volume = []
    
    for i in range(len(incoming_timestamps) - window_size + 1):
        window_volume = np.sum(incoming_bytes[i:i+window_size])
        window_time = incoming_timestamps[i+window_size-1] - incoming_timestamps[i]
        volume_rate = window_volume / max(window_time, 0.1)
        sliding_volume.append(volume_rate)
    
    volume_starts, volume_ends = [], []
    if len(sliding_volume) > 0:
        median_volume = np.median(sliding_volume)
        mad = np.median(np.abs(np.array(sliding_volume) - median_volume))
        volume_threshold = median_volume + 3.0 * mad
        
        high_volume_mask = np.array(sliding_volume) > volume_threshold
        volume_changes = np.diff(np.concatenate([[False], high_volume_mask, [False]]).astype(int))
        volume_starts = np.where(volume_changes == 1)[0]
        volume_ends = np.where(volume_changes == -1)[0]
    
    # Method 2: Packet density analysis
    packet_densities = []
    for i in range(len(incoming_timestamps)):
        window_start = incoming_timestamps[i]
        window_end = window_start + 50.0  # 50ms window
        packets_in_window = np.sum((incoming_timestamps >= window_start) & 
                                 (incoming_timestamps <= window_end))
        packet_densities.append(packets_in_window)
    
    density_bursts = []
    if len(packet_densities) > 0:
        median_density = np.median(packet_densities)
        density_mad = np.median(np.abs(np.array(packet_densities) - median_density))
        density_threshold = median_density + 2.5 * density_mad
        
        high_density_indices = [i for i, density in enumerate(packet_densities) 
                               if density > density_threshold and density >= min_burst_packets]
        if high_density_indices:
            density_bursts = high_density_indices
    
    # Combine burst detections
    burst_indicators = set()
    
    for start, end in zip(volume_starts, volume_ends):
        start_idx = max(0, start)
        end_idx = min(len(incoming_timestamps), end + window_size)
        if end_idx - start_idx >= min_burst_packets:
            burst_indicators.update(range(start_idx, end_idx))
    
    if density_bursts:
        density_start = min(density_bursts)
        density_end = max(density_bursts) + 1
        if density_end - density_start >= min_burst_packets:
            burst_indicators.update(range(density_start, density_end))
    
    if not burst_indicators:
        return record
    
    # Map back to original traffic
    burst_indices = sorted(list(burst_indicators))
    first_incoming_idx = np.where(incoming_mask)[0][min(burst_indices)]
    last_incoming_idx = np.where(incoming_mask)[0][max(burst_indices)]
    
    burst_start_time = timestamps[first_incoming_idx]
    burst_end_time = timestamps[last_incoming_idx]
    
    # Include ±100ms margin
    margin = 100.0
    burst_mask = ((timestamps >= burst_start_time - margin) & 
                  (timestamps <= burst_end_time + margin))
    burst_indices_full = np.where(burst_mask)[0]
    
    if len(burst_indices_full) < min_burst_packets:
        return record
    
    trim_start = burst_indices_full[0]
    trim_end = burst_indices_full[-1] + 1
    
    # Create trimmed record
    trimmed_timestamps = record['timestamps'][trim_start:trim_end]
    trimmed_byte_counts = record['byte_counts'][trim_start:trim_end]
    trimmed_timestamps_arr = np.array(trimmed_timestamps)
    trimmed_byte_counts_arr = np.array(trimmed_byte_counts)
    
    return {
        'webpage_name': record['webpage_name'],
        'capture_number': record['capture_number'],
        'timestamps': trimmed_timestamps,
        'byte_counts': trimmed_byte_counts,
        'total_bytes': sum(abs(trimmed_byte_counts_arr)),
        'num_packets': len(trimmed_timestamps_arr),
        'duration': trimmed_timestamps_arr[-1] - trimmed_timestamps_arr[0] if len(trimmed_timestamps_arr) > 1 else 0,
        'avg_packet_size': np.mean(np.abs(trimmed_byte_counts_arr)) if len(trimmed_byte_counts_arr) > 0 else 0,
        'incoming_packets': np.sum(trimmed_byte_counts_arr > 0),
        'outgoing_packets': np.sum(trimmed_byte_counts_arr < 0),
        'incoming_bytes': np.sum(trimmed_byte_counts_arr[trimmed_byte_counts_arr > 0]),
        'outgoing_bytes': np.sum(np.abs(trimmed_byte_counts_arr[trimmed_byte_counts_arr < 0])),
        'original_length': len(record['timestamps']),
        'trimmed_length': trim_end - trim_start,
        'trim_start_idx': trim_start,
        'trim_end_idx': trim_end,
        'bursts_detected': len(set(burst_indices)),
        'volume_mad_detections': len(list(zip(volume_starts, volume_ends))),
        'density_mad_detections': len(density_bursts) if density_bursts else 0,
        'preprocessing_applied': True,
        'burst_detection_method': 'mad_based'
    }


def preprocess_traffic_data(traffic_data, side_name="unknown"):
    """Preprocess traffic data using MAD-based burst detection"""
    print(f"\n[*] MAD-BASED PREPROCESSING {side_name.upper()} SIDE DATA")
    print("="*60)
    
    preprocessed_data = []
    stats = {
        'total_records': len(traffic_data),
        'successfully_trimmed': 0,
        'no_change_needed': 0,
        'total_original_packets': 0,
        'total_trimmed_packets': 0,
        'total_volume_detections': 0,
        'total_density_detections': 0
    }
    
    for i, record in enumerate(traffic_data):
        if i % 200 == 0 and i > 0:
            print(f"  Processing record {i}/{len(traffic_data)}...")
        
        original_length = len(record['timestamps'])
        stats['total_original_packets'] += original_length
        
        trimmed_record = detect_traffic_bursts(record)
        
        if 'preprocessing_applied' in trimmed_record:
            stats['successfully_trimmed'] += 1
            stats['total_trimmed_packets'] += trimmed_record['trimmed_length']
            stats['total_volume_detections'] += trimmed_record.get('volume_mad_detections', 0)
            stats['total_density_detections'] += trimmed_record.get('density_mad_detections', 0)
        else:
            stats['no_change_needed'] += 1
            stats['total_trimmed_packets'] += original_length
        
        preprocessed_data.append(trimmed_record)
    
    # Calculate statistics
    if stats['total_original_packets'] > 0:
        stats['overall_reduction_percentage'] = float(
            (stats['total_original_packets'] - stats['total_trimmed_packets']) / 
            stats['total_original_packets'] * 100
        )
    else:
        stats['overall_reduction_percentage'] = 0.0
    
    print(f"\n[OK] MAD-BASED PREPROCESSING RESULTS:")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Trimmed: {stats['successfully_trimmed']}")
    print(f"  Unchanged: {stats['no_change_needed']}")
    print(f"  Packet reduction: {stats['overall_reduction_percentage']:.2f}%")
    print(f"  MAD volume detections: {stats['total_volume_detections']}")
    print(f"  MAD density detections: {stats['total_density_detections']}")
    
    return preprocessed_data, stats


def get_website_stats(data):
    """Get website statistics from data"""
    websites = {}
    for d in data:
        website = d.get('webpage_name', 'unknown')
        if website not in websites:
            websites[website] = 0
        websites[website] += 1
    # Convert to regular Python int for JSON serialization
    return {k: int(v) for k, v in websites.items()}


def main():
    parser = argparse.ArgumentParser(description='Extract and preprocess traffic data for MixMatch training')
    parser.add_argument('--input-dir-proxy', type=str, required=True,
                        help='Input directory containing proxy data')
    parser.add_argument('--input-dir-requester', type=str, required=True,
                        help='Input directory containing requester data')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output directory to save processed pickle files (default: ./data)')
    
    args = parser.parse_args()
    
    # Construct paths
    proxy_dir = args.input_dir_proxy
    requester_dir = args.input_dir_requester
    output_dir = args.output_dir
    
    print("[*] Data Extraction and Preprocessing")
    print("="*60)
    print(f"  Input proxy directory: {proxy_dir}")
    print(f"  Input requester directory: {requester_dir}")
    print(f"  Output directory: {output_dir}")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\n[*] Loading proxy side data...")
    proxy_data = load_traffic_data(proxy_dir)
    
    print("\n[*] Loading network requester side data...")
    requester_data = load_traffic_data(requester_dir)
    
    print(f"\n[*] Data Summary:")
    print(f"  Proxy records: {len(proxy_data)}")
    print(f"  Requester records: {len(requester_data)}")
    
    # Preprocess data
    print("\n[*] Processing proxy-side data...")
    processed_proxy_data, proxy_stats = preprocess_traffic_data(proxy_data, "proxy")
    
    print("\n[*] Processing requester-side data...")
    processed_requester_data, requester_stats = preprocess_traffic_data(requester_data, "requester")
    
    # Save processed data
    print("\n[*] Saving processed data...")
    
    with open(f"{output_dir}/mad_processed_proxy_data.pkl", 'wb') as f:
        pickle.dump(processed_proxy_data, f)
    print(f"[OK] Saved {len(processed_proxy_data)} MAD-processed proxy records")
    
    with open(f"{output_dir}/mad_processed_requester_data.pkl", 'wb') as f:
        pickle.dump(processed_requester_data, f)
    print(f"[OK] Saved {len(processed_requester_data)} MAD-processed requester records")
    
    # Save raw data
    with open(f"{output_dir}/raw_proxy_data.pkl", 'wb') as f:
        pickle.dump(proxy_data, f)
    print(f"[OK] Saved {len(proxy_data)} raw proxy records")
    
    with open(f"{output_dir}/raw_requester_data.pkl", 'wb') as f:
        pickle.dump(requester_data, f)
    print(f"[OK] Saved {len(requester_data)} raw requester records")
    
    # Create metadata
    proxy_websites = get_website_stats(proxy_data)
    requester_websites = get_website_stats(requester_data)
    processed_proxy_websites = get_website_stats(processed_proxy_data)
    processed_requester_websites = get_website_stats(processed_requester_data)
    
    metadata = {
        'proxy_samples': len(proxy_data),
        'requester_samples': len(requester_data),
        'mad_processed_proxy_samples': len(processed_proxy_data),
        'mad_processed_requester_samples': len(processed_requester_data),
        'extraction_timestamp': pd.Timestamp.now().isoformat(),
        'data_source': 'script_processing_with_mad',
        'input_directory_proxy': args.input_dir_proxy,
        'input_directory_requester': args.input_dir_requester,
        'mad_preprocessing': {
            'volume_mad_factor': 3.0,
            'density_mad_factor': 2.5,
            'density_window_ms': 50.0,
            'min_burst_packets': 5,
            'margin_ms': 100.0
        },
        'proxy_preprocessing_stats': proxy_stats,
        'requester_preprocessing_stats': requester_stats,
        'website_counts': {
            'raw_proxy': proxy_websites,
            'raw_requester': requester_websites,
            'processed_proxy': processed_proxy_websites,
            'processed_requester': processed_requester_websites,
            'common_websites_raw': list(set(proxy_websites.keys()) & set(requester_websites.keys())),
            'common_websites_processed': list(set(processed_proxy_websites.keys()) & set(processed_requester_websites.keys()))
        }
    }
    
    #with open(f"{output_dir}/extraction_metadata.json", 'w') as f:
    #    json.dump(metadata, f, indent=2)
    
    print(f"\n[OK] Final Summary:")
    print(f"  Raw data: {len(proxy_data)} proxy, {len(requester_data)} requester")
    print(f"  MAD-processed: {len(processed_proxy_data)} proxy, {len(processed_requester_data)} requester")
    print(f"  Common websites (raw): {len(metadata['website_counts']['common_websites_raw'])}")
    print(f"  Common websites (processed): {len(metadata['website_counts']['common_websites_processed'])}")
    print(f"  Proxy packet reduction: {proxy_stats['overall_reduction_percentage']:.2f}%")
    print(f"  Requester packet reduction: {requester_stats['overall_reduction_percentage']:.2f}%")
    print(f"  Files saved in: {output_dir}")
    print("[OK] Ready for MixMatch training!")


if __name__ == "__main__":
    main()
