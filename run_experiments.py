#!/usr/bin/env python3
"""
Master script for running all HonestVQA experiments.
This script orchestrates the complete experimental pipeline.
"""

import argparse
import yaml
import os
import subprocess
import sys
from typing import List, Dict
import json

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Success!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("Failed!")
        print(f"Error: {e}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False

def run_training(config_path: str, epochs: int = 5) -> bool:
    """Run training experiment."""
    cmd = [
        sys.executable, "train.py",
        "--config", config_path
    ]
    return run_command(cmd, f"Training with config: {config_path}")

def run_evaluation(config_path: str, checkpoint_path: str = None) -> bool:
    """Run evaluation experiment."""
    cmd = [
        sys.executable, "eval.py",
        "--config", config_path
    ]
    if checkpoint_path:
        cmd.extend(["--checkpoint", checkpoint_path])
    return run_command(cmd, f"Evaluation with config: {config_path}")

def run_cross_domain_eval(config_path: str, test_configs: List[str], 
                         checkpoint_path: str = None, max_samples: int = 100) -> bool:
    """Run cross-domain evaluation."""
    cmd = [
        sys.executable, "cross_domain_eval.py",
        "--config", config_path,
        "--test_datasets"
    ] + test_configs + [
        "--max_samples", str(max_samples)
    ]
    if checkpoint_path:
        cmd.extend(["--checkpoint", checkpoint_path])
    return run_command(cmd, f"Cross-domain evaluation with {len(test_configs)} test datasets")

def run_ablation_study(config_path: str, checkpoint_path: str = None, 
                      max_samples: int = 100) -> bool:
    """Run ablation study."""
    cmd = [
        sys.executable, "ablation_study.py",
        "--config", config_path,
        "--max_samples", str(max_samples)
    ]
    if checkpoint_path:
        cmd.extend(["--checkpoint", checkpoint_path])
    return run_command(cmd, f"Ablation study with config: {config_path}")

def run_computational_analysis(config_path: str, checkpoint_path: str = None,
                              max_samples: int = 50) -> bool:
    """Run computational analysis."""
    cmd = [
        sys.executable, "computational_analysis.py",
        "--config", config_path,
        "--max_samples", str(max_samples)
    ]
    if checkpoint_path:
        cmd.extend(["--checkpoint", checkpoint_path])
    return run_command(cmd, f"Computational analysis with config: {config_path}")

def run_hyperparameter_sweep(config_path: str, output_dir: str = "hyperparameter_results") -> bool:
    """Run hyperparameter sensitivity analysis."""
    # Load base config
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Define hyperparameter ranges
    alpha_values = [0.1, 0.5, 1.0, 1.5, 2.0]
    margin_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    lambda_values = [0.3, 0.5, 0.7, 0.9, 1.1]
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    print(f"\n{'='*60}")
    print("Running Hyperparameter Sensitivity Analysis")
    print(f"{'='*60}")
    
    # Test alpha values
    print("\nTesting alignment loss alpha values...")
    for alpha in alpha_values:
        config = base_config.copy()
        config['alignment_alpha'] = alpha
        config_name = f"alpha_{alpha}"
        
        # Save modified config
        config_file = os.path.join(output_dir, f"{config_name}.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Run training and evaluation
        success = run_training(config_file, epochs=2)  # Short training for sweep
        if success:
            eval_success = run_evaluation(config_file)
            results[config_name] = {
                'alpha': alpha,
                'training_success': success,
                'evaluation_success': eval_success
            }
    
    # Test margin values
    print("\nTesting contrastive loss margin values...")
    for margin in margin_values:
        config = base_config.copy()
        config['contrastive_margin'] = margin
        config_name = f"margin_{margin}"
        
        config_file = os.path.join(output_dir, f"{config_name}.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        success = run_training(config_file, epochs=2)
        if success:
            eval_success = run_evaluation(config_file)
            results[config_name] = {
                'margin': margin,
                'training_success': success,
                'evaluation_success': eval_success
            }
    
    # Save results
    results_file = os.path.join(output_dir, "hyperparameter_sweep_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nHyperparameter sweep results saved to {results_file}")
    return True

def run_full_experiment_pipeline(config_path: str, test_configs: List[str] = None,
                                output_dir: str = "experiment_results",
                                max_samples: int = 100) -> Dict[str, bool]:
    """Run the complete experimental pipeline."""
    
    print(f"\n{'='*80}")
    print("HONESTVQA COMPLETE EXPERIMENTAL PIPELINE")
    print(f"{'='*80}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_type = config['model_type']
    dataset_type = config['dataset_type']
    
    print(f"Model: {model_type}")
    print(f"Dataset: {dataset_type}")
    print(f"Output directory: {output_dir}")
    
    results = {}
    
    # Step 1: Training
    print(f"\n{'='*40}")
    print("STEP 1: TRAINING")
    print(f"{'='*40}")
    results['training'] = run_training(config_path, epochs=5)
    
    # Step 2: Evaluation
    print(f"\n{'='*40}")
    print("STEP 2: EVALUATION")
    print(f"{'='*40}")
    results['evaluation'] = run_evaluation(config_path)
    
    # Step 3: Cross-domain evaluation (if test configs provided)
    if test_configs:
        print(f"\n{'='*40}")
        print("STEP 3: CROSS-DOMAIN EVALUATION")
        print(f"{'='*40}")
        results['cross_domain'] = run_cross_domain_eval(config_path, test_configs, max_samples=max_samples)
    
    # Step 4: Ablation study
    print(f"\n{'='*40}")
    print("STEP 4: ABLATION STUDY")
    print(f"{'='*40}")
    results['ablation'] = run_ablation_study(config_path, max_samples=max_samples)
    
    # Step 5: Computational analysis
    print(f"\n{'='*40}")
    print("STEP 5: COMPUTATIONAL ANALYSIS")
    print(f"{'='*40}")
    results['computational'] = run_computational_analysis(config_path, max_samples=50)
    
    # Step 6: Hyperparameter sweep
    print(f"\n{'='*40}")
    print("STEP 6: HYPERPARAMETER SENSITIVITY")
    print(f"{'='*40}")
    results['hyperparameter_sweep'] = run_hyperparameter_sweep(config_path, output_dir)
    
    # Save overall results
    results_file = os.path.join(output_dir, "pipeline_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENTAL PIPELINE SUMMARY")
    print(f"{'='*80}")
    for step, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{step.replace('_', ' ').title()}: {status}")
    
    print(f"\nDetailed results saved to {results_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Master script for HonestVQA experiments.")
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML')
    parser.add_argument('--test_configs', nargs='+', help='List of test dataset configs for cross-domain evaluation')
    parser.add_argument('--output_dir', type=str, default='experiment_results', help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=100, help='Maximum samples for testing')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    
    # Individual experiment options
    parser.add_argument('--train_only', action='store_true', help='Run only training')
    parser.add_argument('--eval_only', action='store_true', help='Run only evaluation')
    parser.add_argument('--cross_domain_only', action='store_true', help='Run only cross-domain evaluation')
    parser.add_argument('--ablation_only', action='store_true', help='Run only ablation study')
    parser.add_argument('--computational_only', action='store_true', help='Run only computational analysis')
    parser.add_argument('--hyperparameter_only', action='store_true', help='Run only hyperparameter sweep')
    
    args = parser.parse_args()
    
    # Check if running individual experiments
    if args.train_only:
        run_training(args.config)
    elif args.eval_only:
        run_evaluation(args.config, args.checkpoint)
    elif args.cross_domain_only:
        if not args.test_configs:
            print("Error: --test_configs required for cross-domain evaluation")
            return
        run_cross_domain_eval(args.config, args.test_configs, args.checkpoint, args.max_samples)
    elif args.ablation_only:
        run_ablation_study(args.config, args.checkpoint, args.max_samples)
    elif args.computational_only:
        run_computational_analysis(args.config, args.checkpoint, args.max_samples)
    elif args.hyperparameter_only:
        run_hyperparameter_sweep(args.config, args.output_dir)
    else:
        # Run full pipeline
        run_full_experiment_pipeline(args.config, args.test_configs, args.output_dir, args.max_samples)

if __name__ == "__main__":
    main() 