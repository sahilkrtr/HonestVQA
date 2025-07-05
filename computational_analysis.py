import argparse
import yaml
import torch
import torch.profiler
import time
import psutil
import os
from torch.utils.data import DataLoader
from data.dataset_loader import SpDocVQADataset, InfographicsVQADataset, SROIEDataset
from data.collate import LayoutLMv3Collator, UDOPCollator, DonutCollator
from models import setup_layoutlmv3, setup_udop, setup_donut
import json
from typing import Dict, List, Tuple
import numpy as np

MODEL_REGISTRY = {
    'layoutlmv3': {
        'setup': setup_layoutlmv3,
        'collator': LayoutLMv3Collator,
        'dataset': SpDocVQADataset
    },
    'udop': {
        'setup': setup_udop,
        'collator': UDOPCollator,
        'dataset': InfographicsVQADataset
    },
    'donut': {
        'setup': setup_donut,
        'collator': DonutCollator,
        'dataset': SROIEDataset
    }
}

DATASET_REGISTRY = {
    'spdocvqa': SpDocVQADataset,
    'infographicsvqa': InfographicsVQADataset,
    'sroie': SROIEDataset
}

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def measure_inference_time(model, dataloader, device, num_runs=10):
    """Measure average inference time per batch."""
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_runs:
                break
                
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            if 'decoder_input_ids' in batch and 'decoder_inputs_embeds' in batch:
                batch.pop('decoder_inputs_embeds')
            
            # Warm up
            if i == 0:
                for _ in range(3):
                    _ = model(**batch)
            
            # Measure time
            start_time = time.time()
            _ = model(**batch)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

def profile_model(model, dataloader, device, output_file=None):
    """Profile model using torch.profiler."""
    model.eval()
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 5: 
                    break
                    
                batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                if 'decoder_input_ids' in batch and 'decoder_inputs_embeds' in batch:
                    batch.pop('decoder_inputs_embeds')
                
                _ = model(**batch)
    
    # Print profiling results
    print("\n=== Profiling Results ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    if output_file:
        prof.export_chrome_trace(output_file)
        print(f"Chrome trace saved to {output_file}")
    
    return prof

def count_parameters(model):
    """Count trainable and total parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def estimate_flops(model, input_shape, device):
    """Estimate FLOPs for the model (simplified estimation)."""
    
    
    model.eval()
    with torch.no_grad():
        # Create dummy input
        if hasattr(model, 'config'):
            # For transformer models, estimate based on sequence length and hidden size
            seq_len = input_shape[1] if len(input_shape) > 1 else 512
            hidden_size = getattr(model.config, 'hidden_size', 768)
            num_layers = getattr(model.config, 'num_hidden_layers', 12)
            num_heads = getattr(model.config, 'num_attention_heads', 12)
            
            # Rough FLOP estimation for transformer
            # Self-attention: 4 * seq_len * hidden_size^2 per layer
            # FFN: 8 * seq_len * hidden_size^2 per layer
            attention_flops = 4 * seq_len * hidden_size * hidden_size * num_layers
            ffn_flops = 8 * seq_len * hidden_size * hidden_size * num_layers
            total_flops = attention_flops + ffn_flops
            
            return total_flops
        else:
            return 0  

def run_computational_analysis(config_path: str, checkpoint_path: str = None, 
                              output_dir: str = 'computational_results', 
                              max_samples: int = 50):
    """Run comprehensive computational analysis."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_type = config['model_type'].lower()
    dataset_type = config['dataset_type'].lower()
    dataset_path = config['dataset']
    images_dir = config.get('images_dir', None)
    batch_size = config.get('batch_size', 8)
    max_length = config.get('max_length', 512)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup model
    model_setup = MODEL_REGISTRY[model_type]['setup']
    if model_type == 'udop':
        model, tokenizer = model_setup()
        processor = tokenizer
    else:
        model, processor = model_setup()
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    # Setup dataset
    dataset_cls = DATASET_REGISTRY[dataset_type]
    if model_type == 'layoutlmv3':
        dataset = dataset_cls(dataset_path, images_dir, normalize_bboxes=True, max_samples=max_samples)
    else:
        dataset = dataset_cls(dataset_path, images_dir, max_samples=max_samples)
    
    collator_cls = MODEL_REGISTRY[model_type]['collator']
    if model_type == 'udop':
        collate_fn = collator_cls(processor, max_length=max_length)
    else:
        collate_fn = collator_cls(processor, max_length=max_length)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Running computational analysis for {model_type} on {dataset_type}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Max samples: {max_samples}")
    
    # Measure memory usage before
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Estimate FLOPs
    input_shape = (batch_size, max_length)
    estimated_flops = estimate_flops(model, input_shape, device)
    print(f"Estimated FLOPs per forward pass: {estimated_flops:,}")
    
    # Measure inference time
    print("\nMeasuring inference time...")
    avg_time, std_time = measure_inference_time(model, dataloader, device, num_runs=10)
    print(f"Average inference time per batch: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"Throughput: {batch_size/avg_time:.2f} samples/second")
    
    # Measure memory usage after
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Memory increase: {memory_increase:.2f} MB")
    
    # Profile model
    print("\nProfiling model...")
    profile_file = os.path.join(output_dir, f'profile_{model_type}_{dataset_type}.json')
    prof = profile_model(model, dataloader, device, profile_file)
    
    # Collect results
    results = {
        'model_type': model_type,
        'dataset_type': dataset_type,
        'checkpoint': checkpoint_path,
        'device': str(device),
        'batch_size': batch_size,
        'max_samples': max_samples,
        'parameters': {
            'total': total_params,
            'trainable': trainable_params
        },
        'performance': {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'throughput_samples_per_sec': batch_size/avg_time,
            'estimated_flops': estimated_flops
        },
        'memory': {
            'initial_mb': initial_memory,
            'final_mb': final_memory,
            'increase_mb': memory_increase
        },
        'profiling_file': profile_file
    }
    
    # Save results
    output_file = os.path.join(output_dir, f'computational_analysis_{model_type}_{dataset_type}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComputational analysis results saved to {output_file}")
    
    # Print summary
    print("\n=== Computational Analysis Summary ===")
    print(f"Model: {model_type}")
    print(f"Dataset: {dataset_type}")
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Inference time: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"Throughput: {batch_size/avg_time:.2f} samples/second")
    print(f"Memory usage: {final_memory:.2f} MB (+{memory_increase:.2f} MB)")
    print(f"Estimated FLOPs: {estimated_flops:,}")
    print("=====================================")
    
    return results

def compare_models(configs: List[str], output_dir: str = 'computational_comparison'):
    """Compare computational characteristics of multiple models."""
    
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    
    for config_path in configs:
        print(f"\n{'='*50}")
        print(f"Analyzing {config_path}")
        print(f"{'='*50}")
        
        try:
            results = run_computational_analysis(config_path, output_dir=output_dir)
            
            # Extract model and dataset type for key
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            key = f"{config['model_type']}_{config['dataset_type']}"
            all_results[key] = results
            
        except Exception as e:
            print(f"Error analyzing {config_path}: {str(e)}")
    
    # Save comparison results
    comparison_file = os.path.join(output_dir, 'model_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPUTATIONAL COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Params':<12} {'Time (s)':<10} {'Throughput':<12} {'Memory (MB)':<12}")
    print("-" * 80)
    
    for key, results in all_results.items():
        params = results['parameters']['total']
        time_val = results['performance']['avg_inference_time']
        throughput = results['performance']['throughput_samples_per_sec']
        memory = results['memory']['final_mb']
        
        print(f"{key:<20} {params:<12,} {time_val:<10.4f} {throughput:<12.2f} {memory:<12.2f}")
    
    print(f"\nDetailed results saved to {comparison_file}")

def main():
    parser = argparse.ArgumentParser(description="Computational analysis for HonestVQA experiments.")
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML')
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='computational_results', help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=50, help='Maximum samples for testing')
    parser.add_argument('--compare', nargs='+', help='List of configs to compare')
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.compare, args.output_dir)
    else:
        run_computational_analysis(args.config, args.checkpoint, args.output_dir, args.max_samples)

if __name__ == "__main__":
    main() 