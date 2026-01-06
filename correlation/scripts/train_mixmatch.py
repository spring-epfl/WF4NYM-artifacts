#!/usr/bin/env python3
"""
MixMatch Training Script
Train the drift classifier on website correlation data.
"""

import sys
import os
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add models directory to path (relative to script location)
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.abspath(os.path.join(script_dir, '../models'))
if models_dir not in sys.path:
    sys.path.append(models_dir)
from mixmatch_model import SimplifiedDriftModel, MixMatchDataset, save_model

# Setup logging (will be initialized in main with output_dir)
logger = None
def setup_logging(output_dir):
    log_dir = output_dir if output_dir else os.path.join(script_dir, '../results')
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_training_data(data_dir):
    """
    Load preprocessed training data.
    """
    logger.info(f"Loading training data from {data_dir}")
    
    # Load datasets
    with open(f"{data_dir}/train_data.pkl", 'rb') as f:
        train_data = pickle.load(f)
    
    with open(f"{data_dir}/val_data.pkl", 'rb') as f:
        val_data = pickle.load(f)
    
    # Load metadata
    with open(f"{data_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"   Loaded data:")
    logger.info(f"   Training: {len(train_data)} pairs")
    logger.info(f"   Validation: {len(val_data)} pairs")
    logger.info(f"   Websites: {metadata['selected_websites']}")
    
    return train_data, val_data, metadata

def create_data_loaders(train_data, val_data, batch_size=16):
    """
    Create PyTorch data loaders.
    """
    train_dataset = MixMatchDataset(train_data)
    val_dataset = MixMatchDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (proxy_batch, requester_batch, label_batch) in enumerate(progress_bar):
        # Move to device
        proxy_batch = proxy_batch.to(device)
        requester_batch = requester_batch.to(device)
        label_batch = label_batch.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(proxy_batch, requester_batch)
        outputs = outputs.squeeze()
        
        # Compute loss
        loss = criterion(outputs, label_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = (outputs > 0.5).float()
        correct_predictions += (predictions == label_batch).sum().item()
        total_predictions += label_batch.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{correct_predictions/total_predictions:.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    """
    Validate for one epoch.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for proxy_batch, requester_batch, label_batch in val_loader:
            proxy_batch = proxy_batch.to(device)
            requester_batch = requester_batch.to(device)
            label_batch = label_batch.to(device)
            
            outputs = model(proxy_batch, requester_batch)
            outputs = outputs.squeeze()
            
            # Compute loss
            loss = criterion(outputs, label_batch)
            total_loss += loss.item()
            
            # Compute accuracy
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == label_batch).sum().item()
            total_predictions += label_batch.size(0)
            
            # Store for ROC calculation
            all_scores.extend(outputs.cpu().numpy())
            all_labels.extend(label_batch.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions
    
    # Calculate ROC AUC
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(all_labels, all_scores)
    except:
        roc_auc = 0.5
    
    return avg_loss, accuracy, roc_auc, all_scores, all_labels

def train_model(args):
    """
    Main training function.
    """
    logger.info("🚀 Starting MixMatch Training")
    logger.info("="*50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load data
    train_data, val_data, metadata = load_training_data(args.data_dir)
    train_loader, val_loader = create_data_loaders(train_data, val_data, args.batch_size)
    
    # Create model
    model = SimplifiedDriftModel(
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        sequence_length=metadata.get('sequence_length', 500)
    )
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_roc_auc': []
    }
    
    best_val_accuracy = 0.0
    best_model_path = None
    
    logger.info(f"   Training configuration:")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Learning rate: {args.learning_rate}")
    logger.info(f"   Model: {model.get_model_info()}")
    
    # Training loop
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc, val_roc_auc, val_scores, val_labels = validate_epoch(
            model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_roc_auc'].append(val_roc_auc)
        
        # Log progress
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, ROC AUC: {val_roc_auc:.4f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_path = f"{args.output_dir}/best_model_epoch_{epoch+1}.pth"
            save_model(model, best_model_path, {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_accuracy': val_acc,
                'val_roc_auc': val_roc_auc,
                'training_args': vars(args),
                'data_metadata': metadata
            })
            logger.info(f"New best model saved: {val_acc:.4f} accuracy")
        
        # Early stopping
        if args.early_stopping and epoch > 10:
            recent_val_acc = history['val_accuracy'][-5:]
            if len(recent_val_acc) == 5 and max(recent_val_acc) <= best_val_accuracy - 0.01:
                logger.info("Early stopping triggered")
                break
    
    # Save final model
    final_model_path = f"{args.output_dir}/final_model.pth"
    save_model(model, final_model_path, {
        'total_epochs': len(history['train_loss']),
        'final_val_accuracy': history['val_accuracy'][-1],
        'best_val_accuracy': best_val_accuracy,
        'training_history': history,
        'training_args': vars(args),
        'data_metadata': metadata
    })
    
    # Save training history
    with open(f"{args.output_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Create training plots
    create_training_plots(history, args.output_dir)
    
    logger.info("\n🎯 Training completed!")
    logger.info(f"   Best validation accuracy: {best_val_accuracy:.4f}")
    logger.info(f"   Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    logger.info(f"   Best model: {best_model_path}")
    logger.info(f"   Final model: {final_model_path}")

def create_training_plots(history, output_dir):
    """
    Create training visualization plots.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ROC AUC plot
    ax3.plot(epochs, history['val_roc_auc'], 'g-', label='Validation ROC AUC', linewidth=2)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Classifier')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('ROC AUC')
    ax3.set_title('Validation ROC AUC')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Learning curves comparison
    ax4.plot(epochs, history['train_accuracy'], 'b-', label='Train Acc', alpha=0.7)
    ax4.plot(epochs, history['val_accuracy'], 'r-', label='Val Acc', alpha=0.7)
    ax4.fill_between(epochs, history['train_accuracy'], history['val_accuracy'], 
                     alpha=0.2, color='orange', label='Overfitting Gap')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Overfitting Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Training plots saved to {output_dir}/training_plots.png")

def main():
    parser = argparse.ArgumentParser(description='Train MixMatch Drift Classifier')

    # Data arguments
    parser.add_argument('--data_dir', type=str,
                       default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')),
                       help='Directory containing training data')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../results')),
                       help='Directory to save results')

    # Model arguments
    parser.add_argument('--kernel_size', type=int, default=8,
                       help='Convolution kernel size')
    parser.add_argument('--pool_size', type=int, default=8,
                       help='Pooling size')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Hardware arguments
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')

    # Training options
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Setup logging with output_dir
    global logger
    logger = setup_logging(args.output_dir)

    # Start training
    train_model(args)

if __name__ == "__main__":
    main()
