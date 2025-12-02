#!/usr/bin/env python3
"""
MixMatch Testing and Evaluation Script
Evaluate trained models on test data and perform comprehensive analysis.
"""

import sys
import os
import pickle
import json
import torch
import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, 
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)
import seaborn as sns

from mixmatch_model import SimplifiedDriftModel, MixMatchDataset, load_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_data(data_dir):
    """
    Load test data and metadata.
    """
    logger.info(f"Loading test data from {data_dir}")
    
    with open(f"{data_dir}/test_data.pkl", 'rb') as f:
        test_data = pickle.load(f)
    
    with open(f"{data_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded {len(test_data)} test pairs")
    return test_data, metadata

def evaluate_model_comprehensive(model, test_data, device, metadata):
    """
    Comprehensive model evaluation.
    """
    logger.info("🔍 Running comprehensive model evaluation...")
    
    model.eval()
    
    all_scores = []
    all_labels = []
    all_predictions = []
    website_scores = {}
    
    # Group test data by website pairs for detailed analysis
    for proxy_seq, requester_seq, label, website_info in tqdm(test_data, desc="Evaluating"):
        with torch.no_grad():
            # Convert to tensors
            proxy_tensor = torch.from_numpy(proxy_seq).float().unsqueeze(0).to(device)
            requester_tensor = torch.from_numpy(requester_seq).float().unsqueeze(0).to(device)
            
            # Get prediction
            output = model(proxy_tensor, requester_tensor)
            score = output.cpu().numpy()[0, 0]
            prediction = 1 if score > 0.5 else 0
            
            all_scores.append(score)
            all_labels.append(label)
            all_predictions.append(prediction)
            
            # Store by website for detailed analysis
            if website_info not in website_scores:
                website_scores[website_info] = {'scores': [], 'labels': [], 'predictions': []}
            
            website_scores[website_info]['scores'].append(score)
            website_scores[website_info]['labels'].append(label)
            website_scores[website_info]['predictions'].append(prediction)
    
    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    roc_auc = roc_auc_score(all_labels, all_scores)
    avg_precision = average_precision_score(all_labels, all_scores)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    
    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_scores)
    
    # Find optimal threshold (maximize F1)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else 0.5
    
    # Recalculate predictions with optimal threshold
    optimal_predictions = (np.array(all_scores) >= optimal_threshold).astype(int)
    optimal_accuracy = accuracy_score(all_labels, optimal_predictions)
    
    # Generate classification report
    report = classification_report(all_labels, optimal_predictions, output_dict=True)
    
    # Website-specific analysis
    website_analysis = {}
    for website, data in website_scores.items():
        if len(data['scores']) > 0:
            website_analysis[website] = {
                'accuracy': accuracy_score(data['labels'], data['predictions']),
                'num_samples': len(data['scores']),
                'avg_score': np.mean(data['scores']),
                'score_std': np.std(data['scores'])
            }
    
    results = {
        'overall_metrics': {
            'accuracy': accuracy,
            'optimal_accuracy': optimal_accuracy,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'optimal_threshold': optimal_threshold
        },
        'curves': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'roc_thresholds': thresholds.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'pr_thresholds': pr_thresholds.tolist()
        },
        'classification_report': report,
        'website_analysis': website_analysis,
        'raw_data': {
            'scores': all_scores,
            'labels': all_labels,
            'predictions': all_predictions
        }
    }
    
    return results

def create_evaluation_plots(results, output_dir, model_name="MixMatch"):
    """
    Create comprehensive evaluation plots.
    """
    logger.info("📊 Creating evaluation plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. ROC Curve with LINEAR scale
    fpr = np.array(results['curves']['fpr'])
    tpr = np.array(results['curves']['tpr'])
    roc_auc = results['overall_metrics']['roc_auc']
    
    ax1.plot(fpr, tpr, color='blue', lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve (Linear Scale)')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # 2. ROC Curve with LOGARITHMIC scale
    # Avoid log(0) by setting minimum FPR to 1e-6
    fpr_log = np.maximum(fpr, 1e-6)
    
    ax2.semilogx(fpr_log, tpr, color='blue', lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Random classifier line: TPR = FPR (diagonal on linear scale, curve on log scale)
    random_fpr = np.logspace(-6, 0, 100)  # From 10^-6 to 1
    random_tpr = random_fpr  # For random classifier, TPR = FPR
    ax2.semilogx(random_fpr, random_tpr, color='gray', lw=1, linestyle='--', label='Random Classifier')
    
    ax2.set_xlim([1e-6, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate (log scale)')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve (Logarithmic Scale)')
    ax2.set_xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax2.set_xticklabels(['10⁻⁶', '10⁻⁵', '10⁻⁴', '10⁻³', '10⁻²', '10⁻¹', '1'])
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    precision = np.array(results['curves']['precision'])
    recall = np.array(results['curves']['recall'])
    avg_precision = results['overall_metrics']['average_precision']
    
    ax3.plot(recall, precision, color='blue', lw=2, label=f'{model_name} (AP = {avg_precision:.3f})')
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Classifier')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Score Distribution
    scores = results['raw_data']['scores']
    labels = results['raw_data']['labels']
    
    positive_scores = [s for s, l in zip(scores, labels) if l == 1]
    negative_scores = [s for s, l in zip(scores, labels) if l == 0]
    
    ax4.hist(negative_scores, bins=20, alpha=0.7, label='Negative Pairs', color='red', density=True)
    ax4.hist(positive_scores, bins=20, alpha=0.7, label='Positive Pairs', color='green', density=True)
    ax4.axvline(x=results['overall_metrics']['optimal_threshold'], color='black', 
                linestyle='--', label=f'Optimal Threshold = {results["overall_metrics"]["optimal_threshold"]:.3f}')
    ax4.set_xlabel('Prediction Score')
    ax4.set_ylabel('Density')
    ax4.set_title('Score Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 4. Score Distribution
    scores = results['raw_data']['scores']
    labels = results['raw_data']['labels']
    
    positive_scores = [s for s, l in zip(scores, labels) if l == 1]
    negative_scores = [s for s, l in zip(scores, labels) if l == 0]
    
    ax4.hist(negative_scores, bins=20, alpha=0.7, label='Negative Pairs', color='red', density=True)
    ax4.hist(positive_scores, bins=20, alpha=0.7, label='Positive Pairs', color='green', density=True)
    ax4.axvline(x=results['overall_metrics']['optimal_threshold'], color='black', 
                linestyle='--', label=f'Optimal Threshold = {results["overall_metrics"]["optimal_threshold"]:.3f}')
    ax4.set_xlabel('Prediction Score')
    ax4.set_ylabel('Density')
    ax4.set_title('Score Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)  # Add more padding between subplots
    plt.savefig(f"{output_dir}/evaluation_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a dedicated website performance plot
    website_analysis = results['website_analysis']
    if website_analysis:
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        websites = list(website_analysis.keys())[:15]  # Show more websites
        accuracies = [website_analysis[w]['accuracy'] for w in websites]
        sample_counts = [website_analysis[w]['num_samples'] for w in websites]
        
        # Create a color map based on sample counts
        colors = plt.cm.viridis(np.array(sample_counts) / max(sample_counts))
        
        bars = ax.bar(range(len(websites)), accuracies, color=colors, alpha=0.8)
        ax.set_xlabel('Website Pairs', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Per-Website Performance Analysis', fontsize=14)
        ax.set_xticks(range(len(websites)))
        ax.set_xticklabels([w[:30] + '...' if len(w) > 30 else w for w in websites], 
                          rotation=45, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add sample count labels
        for i, (bar, count) in enumerate(zip(bars, sample_counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'n={count}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout(pad=2.0)
        plt.savefig(f"{output_dir}/website_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a dedicated ROC comparison plot (linear vs log scale)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left: Linear scale ROC
    ax1.plot(fpr, tpr, color='blue', lw=3, label=f'{model_name} (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve (Linear Scale)', fontsize=14)
    ax1.legend(loc="lower right", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Right: Logarithmic scale ROC
    fpr_log = np.maximum(fpr, 1e-6)
    ax2.semilogx(fpr_log, tpr, color='blue', lw=3, label=f'{model_name} (AUC = {roc_auc:.3f})')
    random_fpr = np.logspace(-6, 0, 100)
    random_tpr = random_fpr
    ax2.semilogx(random_fpr, random_tpr, color='gray', lw=2, linestyle='--', label='Random Classifier')
    ax2.set_xlim([1e-6, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate (log scale)', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve (Logarithmic Scale)', fontsize=14)
    ax2.set_xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax2.set_xticklabels(['10⁻⁶', '10⁻⁵', '10⁻⁴', '10⁻³', '10⁻²', '10⁻¹', '1'], fontsize=11)
    ax2.legend(loc="lower right", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(f"{output_dir}/roc_comparison_linear_vs_log.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create confusion matrix plot
    cm = confusion_matrix(results['raw_data']['labels'], 
                         (np.array(results['raw_data']['scores']) >= 
                          results['overall_metrics']['optimal_threshold']).astype(int))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Plots saved to {output_dir}/")

def generate_attack_analysis(results, metadata, output_dir):
    """
    Generate privacy attack analysis report.
    """
    logger.info("📝 Generating attack analysis report...")
    
    accuracy = results['overall_metrics']['optimal_accuracy']
    roc_auc = results['overall_metrics']['roc_auc']
    
    # Privacy assessment
    if accuracy > 0.8:
        privacy_level = "🚨 CRITICAL VULNERABILITY"
        risk_score = 9
        recommendation = "URGENT: Current MAD protection is insufficient against trained ML attacks"
    elif accuracy > 0.7:
        privacy_level = "⚠️ HIGH VULNERABILITY"
        risk_score = 7
        recommendation = "RECOMMENDED: Implement additional privacy protections immediately"
    elif accuracy > 0.6:
        privacy_level = "🟡 MODERATE VULNERABILITY"
        risk_score = 5
        recommendation = "ADVISED: Consider enhanced protection mechanisms"
    else:
        privacy_level = "✅ ADEQUATE PROTECTION"
        risk_score = 3
        recommendation = "MAINTAIN: Current protections appear effective against trained attacks"
    
    # Calculate protection effectiveness
    protection_effectiveness = (1 - accuracy) * 100
    
    report = f"""
# MixMatch Privacy Attack Analysis Report

## Executive Summary
- **Attack Success Rate**: {accuracy*100:.2f}%
- **Privacy Protection Level**: {protection_effectiveness:.1f}%
- **Risk Assessment**: {privacy_level}
- **ROC AUC Score**: {roc_auc:.4f}

## Model Performance
- **Overall Accuracy**: {results['overall_metrics']['accuracy']*100:.2f}%
- **Optimized Accuracy**: {accuracy*100:.2f}%
- **Average Precision**: {results['overall_metrics']['average_precision']:.4f}
- **Optimal Threshold**: {results['overall_metrics']['optimal_threshold']:.4f}

## Dataset Analysis
- **Websites Analyzed**: {len(metadata['selected_websites'])}
- **Total Test Pairs**: {metadata['test_pairs']}
- **Positive Pairs**: {len([l for l in results['raw_data']['labels'] if l == 1])}
- **Negative Pairs**: {len([l for l in results['raw_data']['labels'] if l == 0])}

## Website-Specific Results
"""
    
    # Add website-specific results
    for website, analysis in results['website_analysis'].items():
        if analysis['num_samples'] > 0:
            report += f"- **{website[:50]}{'...' if len(website) > 50 else ''}**: "
            report += f"{analysis['accuracy']*100:.1f}% accuracy ({analysis['num_samples']} samples)\n"
    
    report += f"""

## Privacy Implications
- **Current MAD Protection Effectiveness**: {protection_effectiveness:.1f}%
- **Risk Score**: {risk_score}/10
- **Recommendation**: {recommendation}

## Technical Details
- **False Positive Rate**: {results['curves']['fpr'][np.argmax(results['curves']['tpr'] - np.array(results['curves']['fpr']))]*100:.2f}%
- **True Positive Rate**: {results['curves']['tpr'][np.argmax(results['curves']['tpr'] - np.array(results['curves']['fpr']))]*100:.2f}%
- **Precision**: {results['classification_report']['1']['precision']*100:.2f}%
- **Recall**: {results['classification_report']['1']['recall']*100:.2f}%
- **F1-Score**: {results['classification_report']['1']['f1-score']:.4f}

## Recommendations for Enhanced Protection
1. **Immediate Actions** (if high vulnerability):
   - Implement stronger traffic padding mechanisms
   - Add randomized timing delays
   - Consider burst pattern obfuscation

2. **Long-term Improvements**:
   - Deploy adaptive defense mechanisms
   - Implement multi-layer protection strategies
   - Regular security assessment updates

---
*Report generated by MixMatch Evaluation Script*
*Date: {torch.version.__version__}*
"""
    
    with open(f"{output_dir}/attack_analysis_report.md", 'w') as f:
        f.write(report)
    
    logger.info(f"📝 Analysis report saved to {output_dir}/attack_analysis_report.md")

def compare_models(model_paths, test_data, metadata, device, output_dir):
    """
    Compare multiple trained models.
    """
    logger.info(f"🔄 Comparing {len(model_paths)} models...")
    
    comparison_results = {}
    
    for model_path in model_paths:
        model_name = Path(model_path).stem
        logger.info(f"Evaluating {model_name}...")
        
        # Load model
        model, model_metadata = load_model(model_path, device)
        
        # Evaluate
        results = evaluate_model_comprehensive(model, test_data, device, metadata)
        comparison_results[model_name] = results
    
    # Create comparison plots
    plt.figure(figsize=(12, 8))
    
    # Plot ROC curves for all models with logarithmic x-axis
    plt.subplot(2, 2, 1)
    for model_name, results in comparison_results.items():
        fpr = np.array(results['curves']['fpr'])
        tpr = np.array(results['curves']['tpr'])
        roc_auc = results['overall_metrics']['roc_auc']
        
        # Avoid log(0) by setting minimum FPR to 1e-6
        fpr_log = np.maximum(fpr, 1e-6)
        plt.semilogx(fpr_log, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Random classifier line: TPR = FPR (diagonal on linear scale, curve on log scale)
    random_fpr = np.logspace(-6, 0, 100)  # From 10^-6 to 1
    random_tpr = random_fpr  # For random classifier, TPR = FPR
    plt.semilogx(random_fpr, random_tpr, 'k--', alpha=0.5, label='Random')
    
    plt.xlim([1e-6, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (log scale)')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    plt.gca().set_xticklabels(['10⁻⁶', '10⁻⁵', '10⁻⁴', '10⁻³', '10⁻²', '10⁻¹', '1'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy comparison
    plt.subplot(2, 2, 2)
    models = list(comparison_results.keys())
    accuracies = [comparison_results[m]['overall_metrics']['optimal_accuracy'] for m in models]
    plt.bar(models, [acc*100 for acc in accuracies], alpha=0.8)
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # ROC AUC comparison
    plt.subplot(2, 2, 3)
    roc_aucs = [comparison_results[m]['overall_metrics']['roc_auc'] for m in models]
    plt.bar(models, roc_aucs, alpha=0.8, color='orange')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison results
    comparison_summary = {}
    for model_name, results in comparison_results.items():
        comparison_summary[model_name] = {
            'accuracy': results['overall_metrics']['optimal_accuracy'],
            'roc_auc': results['overall_metrics']['roc_auc'],
            'average_precision': results['overall_metrics']['average_precision']
        }
    
    with open(f"{output_dir}/model_comparison.json", 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    logger.info(f"✅ Model comparison saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Evaluate MixMatch Drift Classifier')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data_dir', type=str,
                       default='/mnt/spring_scratch_pure/ejolles/correlation/data',
                       help='Directory containing test data')
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/spring_scratch_pure/ejolles/correlation/results',
                       help='Directory to save results')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--compare_models', nargs='+',
                       help='Compare multiple models')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load test data
    test_data, metadata = load_test_data(args.data_dir)
    
    if args.compare_models:
        # Compare multiple models
        compare_models(args.compare_models, test_data, metadata, device, args.output_dir)
    else:
        # Evaluate single model
        logger.info(f"Loading model from {args.model_path}")
        model, model_metadata = load_model(args.model_path, device)
        
        # Run evaluation
        results = evaluate_model_comprehensive(model, test_data, device, metadata)
        
        # Create plots and reports
        create_evaluation_plots(results, args.output_dir)
        generate_attack_analysis(results, metadata, args.output_dir)
        
        # Save detailed results
        with open(f"{args.output_dir}/detailed_results.json", 'w') as f:
            # Convert numpy types to JSON serializable types
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            json_results = convert_numpy_types(results)
            json.dump(json_results, f, indent=2)
        
        logger.info("\n🎯 Evaluation completed!")
        logger.info(f"   Attack Success Rate: {results['overall_metrics']['optimal_accuracy']*100:.2f}%")
        logger.info(f"   ROC AUC: {results['overall_metrics']['roc_auc']:.4f}")
        logger.info(f"   Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
