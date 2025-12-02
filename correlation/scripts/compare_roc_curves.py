#!/usr/bin/env python3
"""
ROC Curve Comparison Script
Compare ROC curves from multiple test result folders.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

def load_results(results_dir):
    """
    Load detailed results from a results directory.
    """
    results_path = Path(results_dir) / "detailed_results.json"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results

def create_linear_roc_plot(results_list, labels, output_path):
    """
    Create a linear scale ROC plot.
    """
    plt.figure(figsize=(12, 7), dpi=300)
    
    colors = ['#1f77b4', '#ff7f0e', 'green', 'red', 'purple', 'brown', 'pink', 'gray']  # Blue, Orange, then others
    
    # Linear scale ROC
    for i, (results, label) in enumerate(zip(results_list, labels)):
        fpr = np.array(results['curves']['fpr'])
        tpr = np.array(results['curves']['tpr'])
        roc_auc = results['overall_metrics']['roc_auc']
        
        color = colors[i % len(colors)]
        plt.plot(fpr, tpr, color=color, lw=3, 
                label=f'{label} (AUC = {roc_auc:.3f})')
    
    # Random classifier line
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', alpha=0.7, 
             label='Random Classifier (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=22)
    plt.ylabel('True Positive Rate', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="lower right", fontsize=22)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"Linear ROC plot saved to: {output_path}")

def create_log_roc_plot(results_list, labels, output_path):
    """
    Create a logarithmic scale ROC plot.
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', 'green', 'red', 'purple', 'brown', 'pink', 'gray']  # Blue, Orange, then others
    
    # Logarithmic scale ROC
    for i, (results, label) in enumerate(zip(results_list, labels)):
        fpr = np.array(results['curves']['fpr'])
        tpr = np.array(results['curves']['tpr'])
        roc_auc = results['overall_metrics']['roc_auc']
        
        # Avoid log(0) by setting minimum FPR to 1e-6
        fpr_log = np.maximum(fpr, 1e-6)
        
        color = colors[i % len(colors)]
        plt.semilogx(fpr_log, tpr, color=color, lw=3, 
                    label=f'{label} (AUC = {roc_auc:.3f})')
    
    # Random classifier line for log scale
    random_fpr = np.logspace(-6, 0, 100)
    random_tpr = random_fpr
    plt.semilogx(random_fpr, random_tpr, color='black', lw=2, linestyle='--', 
                alpha=0.7, label='Random Classifier (AUC = 0.500)')
    
    plt.xlim([1e-6, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (log scale)', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    plt.gca().set_xticklabels(['10⁻⁶', '10⁻⁵', '10⁻⁴', '10⁻³', '10⁻²', '10⁻¹', '1'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc="lower right", fontsize=16)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"Logarithmic ROC plot saved to: {output_path}")

def create_combined_roc_plot(results_list, labels, output_path, title="ROC Curve Comparison"):
    """
    Create separate ROC plots with both linear and logarithmic scales.
    """
    # Create separate plots instead of combined
    output_dir = Path(output_path).parent
    
    # Linear scale plot
    linear_path = output_dir / "roc_linear.pdf"
    create_linear_roc_plot(results_list, labels, linear_path)
    
    # Logarithmic scale plot
    log_path = output_dir / "roc_logarithmic.pdf"
    create_log_roc_plot(results_list, labels, log_path)

def create_metrics_comparison_table(results_list, labels, output_path):
    """
    Create a comparison table of key metrics.
    """
    metrics_data = []
    
    for results, label in zip(results_list, labels):
        metrics = results['overall_metrics']
        metrics_data.append({
            'Model': label,
            'Accuracy': f"{metrics['accuracy']*100:.2f}%",
            'Optimal Accuracy': f"{metrics['optimal_accuracy']*100:.2f}%",
            'ROC AUC': f"{metrics['roc_auc']:.4f}",
            'Average Precision': f"{metrics['average_precision']:.4f}",
            'Optimal Threshold': f"{metrics['optimal_threshold']:.4f}"
        })
    
    # Create a simple text table
    table_text = "# ROC Curve Comparison Results\n\n"
    table_text += "| Model | Accuracy | Optimal Accuracy | ROC AUC | Average Precision | Optimal Threshold |\n"
    table_text += "|-------|----------|------------------|---------|-------------------|-------------------|\n"
    
    for data in metrics_data:
        table_text += f"| {data['Model']} | {data['Accuracy']} | {data['Optimal Accuracy']} | {data['ROC AUC']} | {data['Average Precision']} | {data['Optimal Threshold']} |\n"
    
    table_text += "\n## Performance Summary\n\n"
    
    # Find best performing model for each metric
    best_auc = max(results_list, key=lambda x: x['overall_metrics']['roc_auc'])
    best_acc = max(results_list, key=lambda x: x['overall_metrics']['optimal_accuracy'])
    best_ap = max(results_list, key=lambda x: x['overall_metrics']['average_precision'])
    
    best_auc_idx = results_list.index(best_auc)
    best_acc_idx = results_list.index(best_acc)
    best_ap_idx = results_list.index(best_ap)
    
    table_text += f"- **Best ROC AUC**: {labels[best_auc_idx]} ({best_auc['overall_metrics']['roc_auc']:.4f})\n"
    table_text += f"- **Best Accuracy**: {labels[best_acc_idx]} ({best_acc['overall_metrics']['optimal_accuracy']*100:.2f}%)\n"
    table_text += f"- **Best Average Precision**: {labels[best_ap_idx]} ({best_ap['overall_metrics']['average_precision']:.4f})\n"
    
    # Privacy assessment comparison
    table_text += "\n## Privacy Protection Assessment\n\n"
    for results, label in zip(results_list, labels):
        accuracy = results['overall_metrics']['optimal_accuracy']
        protection_effectiveness = (1 - accuracy) * 100
        
        if accuracy > 0.8:
            risk_level = "🚨 CRITICAL"
        elif accuracy > 0.7:
            risk_level = "⚠️ HIGH"
        elif accuracy > 0.6:
            risk_level = "🟡 MODERATE"
        else:
            risk_level = "✅ ADEQUATE"
        
        table_text += f"- **{label}**: {protection_effectiveness:.1f}% protection ({risk_level} risk)\n"
    
    with open(output_path, 'w') as f:
        f.write(table_text)
    
    print(f"Comparison table saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare ROC curves from multiple test results')
    
    parser.add_argument('--results-dirs', nargs='+', required=True,
                       help='Directories containing detailed_results.json files')
    parser.add_argument('--labels', nargs='+', required=True,
                       help='Labels for each results directory')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Directory to save comparison plots')
    parser.add_argument('--title', type=str, default='ROC Curve Comparison',
                       help='Title for the comparison plot')
    
    args = parser.parse_args()
    
    if len(args.results_dirs) != len(args.labels):
        print("Error: Number of results directories must match number of labels")
        sys.exit(1)
    
    if len(args.results_dirs) < 2:
        print("Error: At least 2 results directories are required for comparison")
        sys.exit(1)
    
    print(f"🔍 Loading results from {len(args.results_dirs)} directories...")
    
    # Load all results
    results_list = []
    for i, results_dir in enumerate(args.results_dirs):
        try:
            results = load_results(results_dir)
            results_list.append(results)
            print(f"Loaded results from: {results_dir}")
            print(f"   - {args.labels[i]}: AUC = {results['overall_metrics']['roc_auc']:.4f}")
        except Exception as e:
            print(f"Error loading results from {results_dir}: {e}")
            sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create combined ROC plot
    roc_plot_path = output_dir / "roc_comparison.png"
    create_combined_roc_plot(results_list, args.labels, roc_plot_path, args.title)
    
    # Create metrics comparison table
    table_path = output_dir / "comparison_summary.md"
    create_metrics_comparison_table(results_list, args.labels, table_path)
    
    # Create a simple bar chart comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', 'green', 'red', 'purple', 'brown', 'pink', 'gray']  # Blue, Orange, then others
    
    # ROC AUC comparison
    aucs = [r['overall_metrics']['roc_auc'] for r in results_list]
    bars1 = ax1.bar(args.labels, aucs, color=colors[:len(args.labels)], alpha=0.8)
    ax1.set_ylabel('ROC AUC', fontsize=14)
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, auc in zip(bars1, aucs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Accuracy comparison
    accuracies = [r['overall_metrics']['optimal_accuracy']*100 for r in results_list]
    bars2 = ax2.bar(args.labels, accuracies, color=colors[:len(args.labels)], alpha=0.8)
    ax2.set_ylabel('Accuracy (%)', fontsize=14)
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars2, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Average Precision comparison
    avg_precisions = [r['overall_metrics']['average_precision'] for r in results_list]
    bars3 = ax3.bar(args.labels, avg_precisions, color=colors[:len(args.labels)], alpha=0.8)
    ax3.set_ylabel('Average Precision', fontsize=14)
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, ap in zip(bars3, avg_precisions):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ap:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Protection Effectiveness (inverse of accuracy)
    protection_rates = [(1 - r['overall_metrics']['optimal_accuracy'])*100 for r in results_list]
    bars4 = ax4.bar(args.labels, protection_rates, color=colors[:len(args.labels)], alpha=0.8)
    ax4.set_ylabel('Protection Effectiveness (%)', fontsize=14)
    ax4.set_ylim([0, 100])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, prot in zip(bars4, protection_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{prot:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Rotate x-axis labels if they're long
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout(pad=3.0)
    
    metrics_plot_path = output_dir / "metrics_comparison.pdf"
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"Metrics comparison plot saved to: {metrics_plot_path}")
    
    print("\nComparison completed!")
    print(f"All comparison files saved to: {output_dir}")
    print(f"Check the following files:")
    print(f"   - {output_dir / 'roc_linear.pdf'}")
    print(f"   - {output_dir / 'roc_logarithmic.pdf'}")
    print(f"   - {metrics_plot_path}")
    print(f"   - {table_path}")

if __name__ == "__main__":
    main()
