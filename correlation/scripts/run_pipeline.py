#!/usr/bin/env python3
"""
MixMatch Training Pipeline Master Script
Runs the complete MixMatch training and evaluation pipeline.
"""

import time
import subprocess
import argparse
import sys
import os
import json
import glob
from pathlib import Path
import torch
import torchvision
import numpy
import pandas
import matplotlib
import sklearn
import seaborn
import tqdm


def run_command(command, description, log_file=None):
    """
    Run a command with proper logging and error handling.
    """
    print(f"\n[*] {description}")
    print(f"Command: {command}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        if log_file:
            # Run with output redirection
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            print(f"[*] Output logged to: {log_file}")
        else:
            # Run with live output
            result = subprocess.run(
                command,
                shell=True,
                text=True
            )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[OK] {description} completed successfully in {duration:.2f}s")
            return True
        else:
            print(f"[ERROR] {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error running {description}: {e}")
        return False

def check_python_packages():
    """
    Check if required Python packages are installed.
    """
    print("[*] Checking Python dependencies...")
    print("[OK] All required packages are installed")
    return True

def setup_directories():
    """
    Create necessary directories for the pipeline.
    """
    print("Setting up directories...")
    
    # base_dir will be provided as an argument
    raise NotImplementedError("setup_directories() should not be used; use setup_output_directories(base_dir) instead.")

def setup_output_directories(base_dir):
    """
    Create necessary directories for the pipeline in the given base_dir.
    """
    print(f"Setting up directories in {base_dir}...")
    dirs = [
        f"{base_dir}/data",
        f"{base_dir}/models",
        f"{base_dir}/results",
        f"{base_dir}/logs"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"[*] Created: {dir_path}")
    return base_dir

def main():
    """
    Main pipeline execution function.
    """
    parser = argparse.ArgumentParser(description='MixMatch Training Pipeline')
    parser.add_argument('--skip-data-extraction', action='store_true', 
                        help='Skip data extraction step')
    parser.add_argument('--skip-training', action='store_true', 
                        help='Skip training step')
    parser.add_argument('--skip-testing', action='store_true', 
                        help='Skip testing step')
    parser.add_argument('--websites', type=int, default=5, 
                        help='Number of websites to use for training')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Training batch size')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Base output directory for all pipeline results (default: ./outputs/YYYYMMDD_HHMMSS)')
    parser.add_argument('--input-dir-proxy', type=str, required=True,
                        help='Input directory containing proxy traffic data')
    parser.add_argument('--input-dir-requester', type=str, required=True,
                        help='Input directory containing requester traffic data')
    parser.add_argument('--skip-pre-prepare', action='store_true',
                        help='Skip pre-preparation/extraction step')

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is not None:
        base_dir = args.output_dir
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        base_dir = f"./outputs/{timestamp}"

    print("MixMatch Training Pipeline")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Websites: {args.websites}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output directory: {base_dir}")
    print("=" * 50)

    # Check dependencies
    if not check_python_packages():
        print("Please install missing dependencies first")
        return False

    # Setup output directories
    setup_output_directories(base_dir)
    log_dir = f"{base_dir}/logs"

    # Store pipeline configuration
    config = {
        'websites': args.websites,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'output_dir': base_dir,
        'pipeline_start': time.time(),
        'steps_completed': []
    }
    
    success = True
    
    scripts_dir = "scripts"
    output_data_dir = f"{base_dir}/data"
    
    # Step 1: Pre-prepare Data (Extract and MAD preprocessing)
    if not args.skip_pre_prepare:
        # Check if pre-preparation is already done
        preprepare_files_exist = (
            os.path.exists(f"{output_data_dir}/mad_processed_proxy_data.pkl") and
            os.path.exists(f"{output_data_dir}/mad_processed_requester_data.pkl")
        )
        
        if preprepare_files_exist:
            print("[*] Skipping pre-preparation - MAD processed files already exist")
            config['steps_completed'].append('pre_preparation')
        else:
            step_success = run_command(
                f"python {scripts_dir}/pre-prepare_data_clean.py --input-dir-proxy {args.input_dir_proxy} --input-dir-requester {args.input_dir_requester} --output-dir {output_data_dir}",
                "Extracting and preprocessing traffic data (MAD burst detection)",
                f"{log_dir}/pre_prepare_data.log"
            )
            if step_success:
                config['steps_completed'].append('pre_preparation')
            else:
                success = False
    else:
        print("[*] Skipping pre-preparation")
    
    # Step 2: Prepare Data
    if success:
        # Check if data preparation is already done
        data_files_exist = (
            os.path.exists(f"{output_data_dir}/train_data.pkl") and
            os.path.exists(f"{output_data_dir}/train_labels.pkl") and
            os.path.exists(f"{output_data_dir}/test_data.pkl") and
            os.path.exists(f"{output_data_dir}/test_labels.pkl")
        )
    
    if data_files_exist:
        print("[*] Skipping data preparation - output files already exist")
        config['steps_completed'].append('data_preparation')
    else:
        step_success = run_command(
            f"python {scripts_dir}/prepare_data.py --websites {args.websites} --output-dir {output_data_dir} --input-dir {output_data_dir}",
            f"Preparing training data for {args.websites} websites",
            f"{log_dir}/prepare_data.log"
        )
        if step_success:
            config['steps_completed'].append('data_preparation')
        else:
            success = False
    
    # Step 3: Train Model
    if success and not args.skip_training:
        # Check if model training is already done
        model_files_exist = (
            os.path.exists(f"{base_dir}/models/final_model.pth") or
            len(glob.glob(f"{base_dir}/models/best_model_epoch_*.pth")) > 0
        )
        
        if model_files_exist:
            print("[*] Skipping training - model files already exist")
            config['steps_completed'].append('training')
        else:
            step_success = run_command(
                f"python {scripts_dir}/train_mixmatch.py --epochs {args.epochs} --batch_size {args.batch_size} --output_dir {base_dir}/models --data_dir {output_data_dir}",
                f"Training MixMatch model for {args.epochs} epochs",
                f"{log_dir}/train_mixmatch.log"
            )
            if step_success:
                config['steps_completed'].append('training')
            else:
                success = False
    elif args.skip_training:
        print("[*] Skipping training")
    
    # Step 4: Test Model
    if success and not args.skip_testing:
        # Check for the final model from training
        results_dir = f"{base_dir}/results"
        
        # Check if testing results already exist
        test_results_exist = (
            os.path.exists(f"{results_dir}/test_results.json") or
            os.path.exists(f"{results_dir}/confusion_matrix.png") or
            os.path.exists(f"{results_dir}/classification_report.txt")
        )
        
        if test_results_exist:
            print("[*] Skipping testing - results already exist")
            config['steps_completed'].append('testing')
        else:
            final_model_path = f"{base_dir}/models/final_model.pth"

            if os.path.exists(final_model_path):
                step_success = run_command(
                    f"python {scripts_dir}/test_mixmatch.py --model_path {final_model_path} --cpu --output_dir {results_dir} --data_dir {output_data_dir}",
                    "Testing trained MixMatch model",
                    f"{log_dir}/test_mixmatch.log"
                )
            else:
                # Fallback to searching for best model files
                best_model_files = glob.glob(f"{base_dir}/models/best_model_epoch_*.pth")
                if best_model_files:
                    best_model_path = max(best_model_files, key=os.path.getctime)
                    step_success = run_command(
                        f"python {scripts_dir}/test_mixmatch.py --model_path {best_model_path} --cpu --output_dir {results_dir} --data_dir {output_data_dir}",
                        "Testing trained MixMatch model",
                        f"{log_dir}/test_mixmatch.log"
                    )
                else:
                    print("[ERROR] No trained model found in models directory")
                    step_success = False

            if step_success:
                config['steps_completed'].append('testing')
            else:
                success = False
    elif args.skip_testing:
        print("[*] Skipping testing")
    
    # Save pipeline results
    config['pipeline_end'] = time.time()
    config['total_duration'] = config['pipeline_end'] - config['pipeline_start']
    config['success'] = success
    
    with open(f"{base_dir}/logs/pipeline_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 50)
    if success:
        print("Pipeline completed successfully!")
        print(f"Total time: {config['total_duration']:.2f} seconds")
        print(f"Steps completed: {', '.join(config['steps_completed'])}")
        print(f"Results available in: {base_dir}/results/")
    else:
        print("Pipeline failed!")
        print(f"Check logs in: {log_dir}/")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)