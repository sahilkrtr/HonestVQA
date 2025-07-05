
import argparse
import json
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import glob

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(results_dir: str) -> Dict:
    """Load all results from the results directory."""
    results = {}
    
    # Load evaluation results
    eval_files = glob.glob(os.path.join(results_dir, "*eval*.json"))
    for file_path in eval_files:
        with open(file_path, 'r') as f:
            results[os.path.basename(file_path)] = json.load(f)
    
    # Load cross-domain results
    cross_domain_files = glob.glob(os.path.join(results_dir, "*cross_domain*.json"))
    for file_path in cross_domain_files:
        with open(file_path, 'r') as f:
            results[os.path.basename(file_path)] = json.load(f)
    
    # Load ablation results
    ablation_files = glob.glob(os.path.join(results_dir, "*ablation*.json"))
    for file_path in ablation_files:
        with open(file_path, 'r') as f:
            results[os.path.basename(file_path)] = json.load(f)
    
    # Load computational results
    comp_files = glob.glob(os.path.join(results_dir, "*computational*.json"))
    for file_path in comp_files:
        with open(file_path, 'r') as f:
            results[os.path.basename(file_path)] = json.load(f)
    
    return results

def plot_metrics_comparison(results: Dict, output_dir: str):
    """Plot comparison of different metrics across models/datasets."""
    
    # Extract metrics data
    metrics_data = []
    
    for filename, data in results.items():
        if 'metrics' in data:
            metrics = data['metrics']
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                metrics_data.append({
                    'model': data.get('model_type', 'Unknown'),
                    'dataset': data.get('dataset_type', 'Unknown'),
                    'accuracy': metrics.get('accuracy', 0),
                    'macro_f1': metrics.get('macro_f1', 0),
                    'h_score': metrics.get('h_score', 0),
                    'eci': metrics.get('eci', 0),
                    'mean_entropy': metrics.get('mean_entropy', 0),
                    'mean_confidence': metrics.get('mean_confidence', 0)
                })
    
    if not metrics_data:
        print("No metrics data found for plotting")
        return
    
    df = pd.DataFrame(metrics_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('HonestVQA Metrics Comparison', fontsize=16, fontweight='bold')
    
    # Plot accuracy
    sns.barplot(data=df, x='model', y='accuracy', hue='dataset', ax=axes[0,0])
    axes[0,0].set_title('Accuracy')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot macro F1
    sns.barplot(data=df, x='model', y='macro_f1', hue='dataset', ax=axes[0,1])
    axes[0,1].set_title('Macro F1')
    axes[0,1].set_ylabel('Macro F1')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot H-Score
    sns.barplot(data=df, x='model', y='h_score', hue='dataset', ax=axes[0,2])
    axes[0,2].set_title('H-Score')
    axes[0,2].set_ylabel('H-Score')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # Plot ECI
    sns.barplot(data=df, x='model', y='eci', hue='dataset', ax=axes[1,0])
    axes[1,0].set_title('ECI')
    axes[1,0].set_ylabel('ECI')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot mean entropy
    sns.barplot(data=df, x='model', y='mean_entropy', hue='dataset', ax=axes[1,1])
    axes[1,1].set_title('Mean Entropy')
    axes[1,1].set_ylabel('Mean Entropy')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Plot mean confidence
    sns.barplot(data=df, x='model', y='mean_confidence', hue='dataset', ax=axes[1,2])
    axes[1,2].set_title('Mean Confidence')
    axes[1,2].set_ylabel('Mean Confidence')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_ablation_study(results: Dict, output_dir: str):
    """Plot ablation study results."""
    
    ablation_data = []
    
    for filename, data in results.items():
        if 'ablation_results' in data:
            for ablation_name, ablation_result in data['ablation_results'].items():
                if 'metrics' in ablation_result and 'error' not in ablation_result['metrics']:
                    metrics = ablation_result['metrics']
                    ablation_data.append({
                        'model': data.get('model_type', 'Unknown'),
                        'dataset': data.get('dataset_type', 'Unknown'),
                        'ablation': ablation_name,
                        'description': ablation_result.get('description', ''),
                        'accuracy': metrics.get('accuracy', 0),
                        'macro_f1': metrics.get('macro_f1', 0),
                        'h_score': metrics.get('h_score', 0),
                        'eci': metrics.get('eci', 0)
                    })
    
    if not ablation_data:
        print("No ablation data found for plotting")
        return
    
    df = pd.DataFrame(ablation_data)
    
    # Create ablation comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ablation Study Results', fontsize=16, fontweight='bold')
    
    # Plot accuracy by ablation
    sns.barplot(data=df, x='ablation', y='accuracy', hue='model', ax=axes[0,0])
    axes[0,0].set_title('Accuracy by Ablation')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot H-Score by ablation
    sns.barplot(data=df, x='ablation', y='h_score', hue='model', ax=axes[0,1])
    axes[0,1].set_title('H-Score by Ablation')
    axes[0,1].set_ylabel('H-Score')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot ECI by ablation
    sns.barplot(data=df, x='ablation', y='eci', hue='model', ax=axes[1,0])
    axes[1,0].set_title('ECI by Ablation')
    axes[1,0].set_ylabel('ECI')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot macro F1 by ablation
    sns.barplot(data=df, x='ablation', y='macro_f1', hue='model', ax=axes[1,1])
    axes[1,1].set_title('Macro F1 by Ablation')
    axes[1,1].set_ylabel('Macro F1')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_study.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_cross_domain_results(results: Dict, output_dir: str):
    """Plot cross-domain evaluation results."""
    
    cross_domain_data = []
    
    for filename, data in results.items():
        if 'cross_domain_results' in data:
            for dataset_name, dataset_result in data['cross_domain_results'].items():
                if 'metrics' in dataset_result and 'error' not in dataset_result['metrics']:
                    metrics = dataset_result['metrics']
                    cross_domain_data.append({
                        'trained_model': data.get('model_type', 'Unknown'),
                        'test_dataset': dataset_name,
                        'accuracy': metrics.get('accuracy', 0),
                        'macro_f1': metrics.get('macro_f1', 0),
                        'h_score': metrics.get('h_score', 0),
                        'eci': metrics.get('eci', 0)
                    })
    
    if not cross_domain_data:
        print("No cross-domain data found for plotting")
        return
    
    df = pd.DataFrame(cross_domain_data)
    
    # Create heatmap for cross-domain results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cross-Domain Evaluation Results', fontsize=16, fontweight='bold')
    
    # Accuracy heatmap
    acc_pivot = df.pivot(index='trained_model', columns='test_dataset', values='accuracy')
    sns.heatmap(acc_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0,0])
    axes[0,0].set_title('Accuracy')
    
    # H-Score heatmap
    h_pivot = df.pivot(index='trained_model', columns='test_dataset', values='h_score')
    sns.heatmap(h_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0,1])
    axes[0,1].set_title('H-Score')
    
    # ECI heatmap
    eci_pivot = df.pivot(index='trained_model', columns='test_dataset', values='eci')
    sns.heatmap(eci_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1,0])
    axes[1,0].set_title('ECI')
    
    # Macro F1 heatmap
    f1_pivot = df.pivot(index='trained_model', columns='test_dataset', values='macro_f1')
    sns.heatmap(f1_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1,1])
    axes[1,1].set_title('Macro F1')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_domain_results.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_computational_analysis(results: Dict, output_dir: str):
    """Plot computational analysis results."""
    
    comp_data = []
    
    for filename, data in results.items():
        if 'performance' in data and 'parameters' in data:
            comp_data.append({
                'model': data.get('model_type', 'Unknown'),
                'dataset': data.get('dataset_type', 'Unknown'),
                'total_params': data['parameters']['total'],
                'trainable_params': data['parameters']['trainable'],
                'inference_time': data['performance']['avg_inference_time'],
                'throughput': data['performance']['throughput_samples_per_sec'],
                'memory_mb': data['memory']['final_mb'],
                'estimated_flops': data['performance']['estimated_flops']
            })
    
    if not comp_data:
        print("No computational data found for plotting")
        return
    
    df = pd.DataFrame(comp_data)
    
    # Create computational comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Computational Analysis Results', fontsize=16, fontweight='bold')
    
    # Parameters comparison
    sns.barplot(data=df, x='model', y='total_params', hue='dataset', ax=axes[0,0])
    axes[0,0].set_title('Total Parameters')
    axes[0,0].set_ylabel('Parameters')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Inference time
    sns.barplot(data=df, x='model', y='inference_time', hue='dataset', ax=axes[0,1])
    axes[0,1].set_title('Inference Time')
    axes[0,1].set_ylabel('Time (seconds)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Throughput
    sns.barplot(data=df, x='model', y='throughput', hue='dataset', ax=axes[0,2])
    axes[0,2].set_title('Throughput')
    axes[0,2].set_ylabel('Samples/second')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # Memory usage
    sns.barplot(data=df, x='model', y='memory_mb', hue='dataset', ax=axes[1,0])
    axes[1,0].set_title('Memory Usage')
    axes[1,0].set_ylabel('Memory (MB)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # FLOPs comparison
    sns.barplot(data=df, x='model', y='estimated_flops', hue='dataset', ax=axes[1,1])
    axes[1,1].set_title('Estimated FLOPs')
    axes[1,1].set_ylabel('FLOPs')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Efficiency plot (throughput vs parameters)
    scatter = axes[1,2].scatter(df['total_params'], df['throughput'], 
                               c=df['inference_time'], s=100, alpha=0.7)
    axes[1,2].set_title('Efficiency: Throughput vs Parameters')
    axes[1,2].set_xlabel('Total Parameters')
    axes[1,2].set_ylabel('Throughput (samples/second)')
    plt.colorbar(scatter, ax=axes[1,2], label='Inference Time (s)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'computational_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(results: Dict, output_dir: str):
    """Create a summary table of all results."""
    
    summary_data = []
    
    for filename, data in results.items():
        if 'metrics' in data and isinstance(data['metrics'], dict):
            metrics = data['metrics']
            if 'accuracy' in metrics:
                summary_data.append({
                    'Experiment': filename.replace('.json', ''),
                    'Model': data.get('model_type', 'Unknown'),
                    'Dataset': data.get('dataset_type', 'Unknown'),
                    'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                    'Macro F1': f"{metrics.get('macro_f1', 0):.4f}",
                    'H-Score': f"{metrics.get('h_score', 0):.4f}",
                    'ECI': f"{metrics.get('eci', 0):.4f}",
                    'Mean Entropy': f"{metrics.get('mean_entropy', 0):.4f}",
                    'Mean Confidence': f"{metrics.get('mean_confidence', 0):.4f}"
                })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        
        # Save as CSV
        csv_file = os.path.join(output_dir, 'results_summary.csv')
        df.to_csv(csv_file, index=False)
        print(f"Summary table saved to {csv_file}")
        
        # Print table
        print("\n" + "="*100)
        print("RESULTS SUMMARY TABLE")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)
    
    return summary_data

def generate_latex_table(results: Dict, output_dir: str):
    """Generate LaTeX table for paper submission."""
    
    summary_data = []
    
    for filename, data in results.items():
        if 'metrics' in data and isinstance(data['metrics'], dict):
            metrics = data['metrics']
            if 'accuracy' in metrics:
                summary_data.append({
                    'Model': data.get('model_type', 'Unknown').upper(),
                    'Dataset': data.get('dataset_type', 'Unknown').upper(),
                    'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                    'Macro F1': f"{metrics.get('macro_f1', 0):.3f}",
                    'H-Score': f"{metrics.get('h_score', 0):.3f}",
                    'ECI': f"{metrics.get('eci', 0):.3f}"
                })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        
        # Generate LaTeX table
        latex_file = os.path.join(output_dir, 'results_table.tex')
        with open(latex_file, 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Experimental Results Comparison}\n")
            f.write("\\label{tab:results}\n")
            f.write("\\begin{tabular}{lccccc}\n")
            f.write("\\hline\n")
            f.write("Model & Dataset & Accuracy & Macro F1 & H-Score & ECI \\\\\n")
            f.write("\\hline\n")
            
            for _, row in df.iterrows():
                f.write(f"{row['Model']} & {row['Dataset']} & {row['Accuracy']} & {row['Macro F1']} & {row['H-Score']} & {row['ECI']} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"LaTeX table saved to {latex_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize HonestVQA experiment results.")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing result files')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory for plots')
    parser.add_argument('--plot_type', type=str, choices=['all', 'metrics', 'ablation', 'cross_domain', 'computational'], 
                       default='all', help='Type of plots to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Generate plots based on type
    if args.plot_type in ['all', 'metrics']:
        print("Generating metrics comparison plots...")
        plot_metrics_comparison(results, args.output_dir)
    
    if args.plot_type in ['all', 'ablation']:
        print("Generating ablation study plots...")
        plot_ablation_study(results, args.output_dir)
    
    if args.plot_type in ['all', 'cross_domain']:
        print("Generating cross-domain results plots...")
        plot_cross_domain_results(results, args.output_dir)
    
    if args.plot_type in ['all', 'computational']:
        print("Generating computational analysis plots...")
        plot_computational_analysis(results, args.output_dir)
    
    # Create summary table
    print("Creating summary table...")
    create_summary_table(results, args.output_dir)
    
    # Generate LaTeX table
    print("Generating LaTeX table...")
    generate_latex_table(results, args.output_dir)
    
    print(f"\nAll visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main() 