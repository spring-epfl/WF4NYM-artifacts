#!/usr/bin/env python3
"""
Data Preparation Script for MixMatch Correlation Analysis
Saves the preprocessed traffic data from the notebook for training and testing.
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from sklearn.model_selection import train_test_split
import random

def load_notebook_data(data_dir):
    """
    Load processed traffic data from extracted files or notebook processing.
    Prioritizes MAD-processed data for better burst detection.
    """
    print("[*] Loading traffic data...")
    
    # First try to load MAD-processed data from notebook
    try:
        with open(f"{data_dir}/mad_processed_proxy_data.pkl", 'rb') as f:
            proxy_data = pickle.load(f)
        with open(f"{data_dir}/mad_processed_requester_data.pkl", 'rb') as f:
            requester_data = pickle.load(f)
        print(f"[OK] Loaded MAD-processed data: {len(proxy_data)} proxy, {len(requester_data)} requester samples")
        print("[*] Using burst-filtered data with MAD-based preprocessing")
        return proxy_data, requester_data
    except FileNotFoundError:
        print("[WARNING] MAD-processed data not found, trying raw extracted data...")
    
    # Fall back to raw extracted data
    try:
        with open(f"{data_dir}/raw_proxy_data.pkl", 'rb') as f:
            proxy_data = pickle.load(f)
        with open(f"{data_dir}/raw_requester_data.pkl", 'rb') as f:
            requester_data = pickle.load(f)
        print(f"[OK] Loaded raw extracted data: {len(proxy_data)} proxy, {len(requester_data)} requester samples")
        print("[WARNING] Using raw data without MAD preprocessing")
        return proxy_data, requester_data
    except FileNotFoundError:
        print("[WARNING] Extracted data not found, trying notebook data...")
    
    try:
        with open(f"{data_dir}/preprocessed_requester_data.pkl", 'rb') as f:
            requester_data = pickle.load(f)
        print(f"[OK] Loaded {len(requester_data)} requester data samples")
    except FileNotFoundError:
        print("[ERROR] Preprocessed requester data not found. Please run the notebook first.")
        return None, None
    
    return proxy_data, requester_data

def prepare_mixmatch_data(traffic_data, max_length=500):
    """
    Convert traffic data to sequence format for MixMatch.
    """
    if isinstance(traffic_data, dict):
        if 'timestamps' in traffic_data and 'byte_counts' in traffic_data:
            traffic_df = pd.DataFrame({
                'timestamp': traffic_data['timestamps'],
                'size': traffic_data['byte_counts']
            })
        else:
            # Find timestamp and size keys
            timestamp_key = None
            size_key = None
            
            for key in traffic_data.keys():
                if 'time' in key.lower() or 'stamp' in key.lower():
                    timestamp_key = key
                if any(x in key.lower() for x in ['size', 'bytes', 'byte_counts']):
                    size_key = key
            
            if timestamp_key and size_key:
                traffic_df = pd.DataFrame({
                    'timestamp': traffic_data[timestamp_key],
                    'size': traffic_data[size_key]
                })
            else:
                return np.zeros(max_length)
    else:
        return np.zeros(max_length)
    
    if traffic_df.empty:
        return np.zeros(max_length)
    
    # Subsample if too large
    if len(traffic_df) > 1000:
        traffic_df = traffic_df.sample(n=1000).sort_values('timestamp').reset_index(drop=True)
    
    # Convert timestamps
    if traffic_df['timestamp'].max() > 1e10:
        traffic_df['timestamp'] = traffic_df['timestamp'] / 1000.0
    
    # Normalize timestamps
    traffic_df = traffic_df.sort_values('timestamp').reset_index(drop=True)
    min_time = traffic_df['timestamp'].min()
    traffic_df['timestamp'] = traffic_df['timestamp'] - min_time
    
    # Create time bins
    max_time = traffic_df['timestamp'].max()
    if max_time == 0:
        max_time = 1.0
    
    # Use histogram for efficient binning
    bin_edges = np.linspace(0, max_time, max_length + 1)
    sequence, _ = np.histogram(traffic_df['timestamp'], bins=bin_edges, weights=traffic_df['size'])
    
    return sequence

def prepare_training_data(proxy_data, requester_data, num_websites=5, samples_per_website=50):
    """
    Prepare training data for MixMatch model with balanced website representation.
    """
    print(f"Preparing training data with {num_websites} websites...")
    
    # Group data by website
    proxy_by_website = defaultdict(list)
    requester_by_website = defaultdict(list)
    
    for i, data in enumerate(proxy_data):
        if 'webpage_name' in data:
            website = data['webpage_name']
            proxy_by_website[website].append((i, data))
    
    for i, data in enumerate(requester_data):
        if 'webpage_name' in data:
            website = data['webpage_name']
            requester_by_website[website].append((i, data))
    
    # Select top websites with most samples
    website_counts = [(website, len(samples)) for website, samples in proxy_by_website.items()]
    website_counts.sort(key=lambda x: x[1], reverse=True)
    selected_websites = [website for website, count in website_counts[:num_websites]]
    
    print(f"📊 Selected websites: {selected_websites}")
    
    # Prepare training pairs
    positive_pairs = []
    negative_pairs = []
    
    for website in selected_websites:
        proxy_samples = proxy_by_website[website][:samples_per_website]
        requester_samples = requester_by_website[website][:samples_per_website]
        
        print(f"   {website}: {len(proxy_samples)} proxy, {len(requester_samples)} requester samples")
        
        # Create positive pairs (same website)
        min_samples = min(len(proxy_samples), len(requester_samples))
        for i in range(min_samples):
            # Convert to sequences immediately
            proxy_seq = prepare_mixmatch_data(proxy_samples[i][1])
            requester_seq = prepare_mixmatch_data(requester_samples[i][1])
            positive_pairs.append((proxy_seq, requester_seq, 1, website))
        
        # Create negative pairs (different websites)
        for website2 in selected_websites:
            if website != website2:
                requester_samples2 = requester_by_website[website2][:samples_per_website//2]
                for i, proxy_sample in enumerate(proxy_samples[:samples_per_website//2]):
                    if i < len(requester_samples2):
                        proxy_seq = prepare_mixmatch_data(proxy_sample[1])
                        requester_seq = prepare_mixmatch_data(requester_samples2[i][1])
                        negative_pairs.append((proxy_seq, requester_seq, 0, f"{website}_vs_{website2}"))
    
    # Balance the dataset
    min_pairs = min(len(positive_pairs), len(negative_pairs))
    positive_pairs = positive_pairs[:min_pairs]
    negative_pairs = negative_pairs[:min_pairs]
    
    print(f"Training data prepared:")
    print(f"   Total pairs: {len(positive_pairs) + len(negative_pairs)}")
    print(f"   Positive pairs: {len(positive_pairs)}")
    print(f"   Negative pairs: {len(negative_pairs)}")
    
    return positive_pairs + negative_pairs, selected_websites

def save_processed_data(training_pairs, selected_websites, output_dir):
    """
    Save the processed training data and metadata.
    """
    print(f"Saving processed data to {output_dir}...")
    
    # Split data into train/validation/test
    train_data, temp_data = train_test_split(training_pairs, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Save datasets
    with open(f"{output_dir}/train_data.pkl", 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(f"{output_dir}/val_data.pkl", 'wb') as f:
        pickle.dump(val_data, f)
    
    with open(f"{output_dir}/test_data.pkl", 'wb') as f:
        pickle.dump(test_data, f)
    
    # Save metadata
    metadata = {
        'selected_websites': selected_websites,
        'total_pairs': len(training_pairs),
        'train_pairs': len(train_data),
        'val_pairs': len(val_data),
        'test_pairs': len(test_data),
        'sequence_length': 500,
        'positive_pairs': len([p for p in training_pairs if p[2] == 1]),
        'negative_pairs': len([p for p in training_pairs if p[2] == 0])
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved datasets:")
    print(f"   Training: {len(train_data)} pairs")
    print(f"   Validation: {len(val_data)} pairs")
    print(f"   Testing: {len(test_data)} pairs")
    print(f"   Metadata: {output_dir}/metadata.json")
    
    return metadata

def main():
    """
    Main function to prepare and save all training data.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare data for MixMatch training')
    parser.add_argument('--websites', type=int, default=5, 
                        help='Number of websites to use for training')
    parser.add_argument('--samples-per-website', type=int, default=40,
                        help='Samples per website')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save processed data (default: data_without_9002)')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Directory to load input data (default: same as output-dir or data_without_9002)')

    args = parser.parse_args()

    print("MixMatch Data Preparation Pipeline")
    print("="*50)
    print(f"Configuration:")
    print(f"  Websites: {args.websites}")
    print(f"  Samples per website: {args.samples_per_website}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Input dir: {args.input_dir if args.input_dir else (args.output_dir)}")
    print("="*50)

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Determine directories
    output_dir = args.output_dir 
    input_dir = args.input_dir if args.input_dir else output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data from notebook
    proxy_data, requester_data = load_notebook_data(input_dir)
    if proxy_data is None or requester_data is None:
        print("❌ Failed to load data. Exiting.")
        return

    # Prepare training data
    training_pairs, selected_websites = prepare_training_data(
        proxy_data=proxy_data,
        requester_data=requester_data,
        num_websites=args.websites,
        samples_per_website=args.samples_per_website
    )

    # Save processed data
    metadata = save_processed_data(training_pairs, selected_websites, output_dir)

    print("\nData preparation completed successfully!")
    print(f"Data saved in: {output_dir}")
    print(f"Ready for training with {metadata['total_pairs']} total pairs")

if __name__ == "__main__":
    main()
