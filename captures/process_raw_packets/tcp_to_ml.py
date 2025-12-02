#!/usr/bin/env python3
"""
TCP to ML Format Converter

Converts aggregated TCP traffic data into ML-ready pickle format with
train/test splits. Processes traffic captures and creates labeled datasets
for machine learning classifiers.
"""

import os
import argparse
import pickle
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split


def load_traffic_data(tcp_path, max_instances=200, time_filter=0):
    """
    Load and parse traffic data from aggregated website files.
    
    Args:
        tcp_path: Directory containing aggregated website files
        max_instances: Maximum number of instances per website
        time_filter: Time offset to filter cells (seconds)
    
    Returns:
        tuple: (data, labels) where data is list of dicts with 'cells' key
    """
    data = []
    labels = []
    
    DIRECTION = {"client-to-server": -1, "server-to-client": 1}
    
    files = [f for f in os.listdir(tcp_path) if os.path.isfile(os.path.join(tcp_path, f))]
    
    for index, file in enumerate(files, 1):
        print(f"[*] ({index}/{len(files)}) Reading: {file}")
        
        instance = 0
        
        with open(os.path.join(tcp_path, file), "r") as f:
            for line in f:
                if instance >= max_instances:
                    break
                
                points = line.split(" ")[2:]
                if not points:
                    continue
                
                # Parse base time
                base_time = points[0].split(":")[0]
                base_time = float(base_time) if "." in base_time else int(base_time)
                
                cells = []
                for point in points:
                    parts = point.split(":")
                    if len(parts) != 2:
                        continue
                    
                    # Parse timestamp
                    time = parts[0]
                    time = float(time) if "." in time else int(time)
                    rel_time = (time - base_time) / 1000.0
                    
                    # Parse direction: -1 is OUT (0), 1 is IN (1)
                    direction = np.sign(float(parts[1]))
                    direction = 0 if direction == DIRECTION["client-to-server"] else 1
                    
                    # Parse volume
                    volume = int(np.abs(float(parts[1])))
                    
                    cells.append([rel_time, direction, direction, volume])
                
                # Apply time filter if specified
                if time_filter > 0 and cells:
                    filtered_cells = [c for c in cells if c[0] <= cells[-1][0] - time_filter]
                else:
                    filtered_cells = cells
                
                if filtered_cells:
                    labels.append(file)
                    data.append({"cells": filtered_cells})
                    instance += 1
        
        if instance < max_instances:
            print(f"    [WARNING] {file} has only {instance} instances (expected {max_instances})")
    
    return data, labels


def filter_by_count(data, labels, min_count=20):
    """
    Filter data to only include classes with minimum sample count.
    
    Args:
        data: List of data samples
        labels: List of labels
        min_count: Minimum samples per class
    
    Returns:
        tuple: (filtered_data, filtered_labels, label_mapping)
    """
    label_counts = Counter(labels)
    
    # Create label to index mapping
    unique_labels = sorted(set(labels))
    label_to_idx = dict(zip(unique_labels, range(len(unique_labels))))
    
    # Filter by count
    filtered_data = []
    filtered_labels = []
    
    for d, l in zip(data, labels):
        if label_counts[l] >= min_count:
            filtered_data.append(d)
            filtered_labels.append(label_to_idx[l])
    
    print(f"[*] Filtering by minimum count ({min_count}):")
    print(f"    Total samples: {len(data)} -> {len(filtered_data)}")
    print(f"    Total classes: {len(unique_labels)} -> {len(set(filtered_labels))}")
    
    return filtered_data, filtered_labels, label_to_idx


def create_train_test_split(data, labels, train_size=0.6):
    """
    Split data into train and test sets.
    
    Args:
        data: List of data samples
        labels: List of labels
        train_size: Proportion of data for training
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        labels,
        train_size=train_size,
        shuffle=True,
        stratify=labels,
    )
    
    print(f"[*] Train/test split ({train_size:.0%} train):")
    print(f"    Train samples: {len(X_train)}")
    print(f"    Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def save_pickle(output_file, X_train, X_test, y_train, y_test):
    """
    Save train/test data to pickle file.
    
    Args:
        output_file: Path to output pickle file
        X_train: Training data
        X_test: Test data
        y_train: Training labels
        y_test: Test labels
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    
    print(f"[OK] Saved to: {output_file}")


def process_tcp_to_ml(input_folder, output_file, max_instances=200, 
                     min_count=20, train_size=0.6, time_filter=0):
    """
    Main processing function to convert TCP data to ML format.
    
    Args:
        input_folder: Directory containing aggregated website files
        output_file: Path to output pickle file
        max_instances: Maximum instances per website
        min_count: Minimum samples per class
        train_size: Proportion of data for training
        time_filter: Time offset to filter cells (seconds)
    
    Returns:
        dict: Label to index mapping
    """
    print(f"[*] Loading traffic data from: {input_folder}")
    data, labels = load_traffic_data(input_folder, max_instances, time_filter)
    
    print(f"[*] Loaded {len(data)} samples from {len(set(labels))} websites")
    
    filtered_data, filtered_labels, label_mapping = filter_by_count(
        data, labels, min_count
    )
    
    if not filtered_data:
        print("[ERROR] No data remaining after filtering")
        return None
    
    X_train, X_test, y_train, y_test = create_train_test_split(
        filtered_data, filtered_labels, train_size
    )
    
    save_pickle(output_file, X_train, X_test, y_train, y_test)
    
    return label_mapping


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Convert TCP traffic data to ML-ready pickle format'
    )
    
    parser.add_argument('--input-folder', type=str, required=True,
                       help='Directory containing aggregated website files')
    parser.add_argument('--output-folder', type=str, required=True,
                       help='Directory to write output pickle file')
    parser.add_argument('--output-name', type=str, default='data.pkl',
                       help='Name of output pickle file (default: data.pkl)')
    parser.add_argument('--max-instances', type=int, default=200,
                       help='Maximum instances per website (default: 200)')
    parser.add_argument('--min-count', type=int, default=20,
                       help='Minimum samples per class (default: 20)')
    parser.add_argument('--train-size', type=float, default=0.6,
                       help='Proportion of data for training (default: 0.6)')
    parser.add_argument('--time-filter', type=float, default=0,
                       help='Time offset to filter cells in seconds (default: 0)')
    
    args = parser.parse_args()
    
    output_file = os.path.join(args.output_folder, args.output_name)
    
    label_mapping = process_tcp_to_ml(
        input_folder=args.input_folder,
        output_file=output_file,
        max_instances=args.max_instances,
        min_count=args.min_count,
        train_size=args.train_size,
        time_filter=args.time_filter
    )
    
    if label_mapping:
        print(f"\n[OK] Processing complete!")
        print(f"    Output: {output_file}")
        print(f"    Classes: {len(label_mapping)}")
    else:
        print(f"\n[ERROR] Processing failed")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())