import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from data.dataset_loader import SpDocVQADataset, InfographicsVQADataset, SROIEDataset
from data.collate import LayoutLMv3Collator, UDOPCollator, DonutCollator
from models import setup_layoutlmv3, setup_udop, setup_donut
from utils.metrics import evaluate_model_performance
import os
import json
from typing import Dict, List, Tuple

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

def load_model_and_checkpoint(model_type: str, checkpoint_path: str = None):
    """Load model and optionally load checkpoint."""
    model_setup = MODEL_REGISTRY[model_type]['setup']
    if model_type == 'udop':
        model, tokenizer = model_setup()
    else:
        model, processor = model_setup()
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    return model, processor if model_type != 'udop' else tokenizer

def evaluate_on_dataset(model, processor, model_type: str, dataset_type: str, 
                       dataset_path: str, images_dir: str = None, 
                       batch_size: int = 8, max_length: int = 512,
                       max_samples: int = None) -> Dict[str, float]:
    """Evaluate model on a specific dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Setup dataset and collator
    dataset_cls = DATASET_REGISTRY[dataset_type]
    if model_type == 'layoutlmv3':
        if dataset_type == 'sroie':
            dataset = dataset_cls(dataset_path, images_dir, max_samples=max_samples)
        else:
            dataset = dataset_cls(dataset_path, images_dir, normalize_bboxes=True, max_samples=max_samples)
    else:
        dataset = dataset_cls(dataset_path, images_dir, max_samples=max_samples)
    
    collator_cls = MODEL_REGISTRY[model_type]['collator']
    if model_type == 'udop':
        collate_fn = collator_cls(processor, max_length=max_length)
    else:
        collate_fn = collator_cls(processor, max_length=max_length)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            if 'decoder_input_ids' in batch and 'decoder_inputs_embeds' in batch:
                batch.pop('decoder_inputs_embeds')
            
            outputs = model(**batch)
            
            if hasattr(outputs, 'logits'):
                all_logits.append(outputs.logits.cpu())
            elif hasattr(outputs, 'start_logits'):
                all_logits.append(outputs.start_logits.cpu())
            
            if 'labels' in batch:
                all_labels.append(batch['labels'].cpu())
    
    if all_logits and all_labels:
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return evaluate_model_performance(all_logits, all_labels)
    else:
        return {'error': 'No predictions or labels found'}

def main():
    parser = argparse.ArgumentParser(description="Cross-domain evaluation for HonestVQA experiments.")
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML')
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to model checkpoint')
    parser.add_argument('--test_datasets', nargs='+', required=True, help='List of test dataset configs')
    parser.add_argument('--output_file', type=str, default='cross_domain_results.json', help='Output file for results')
    parser.add_argument('--max_samples', type=int, default=100, help='Maximum samples per dataset for testing')
    args = parser.parse_args()
    
    # Load main config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model_type = config['model_type'].lower()
    
    # Load model
    model, processor = load_model_and_checkpoint(model_type, args.checkpoint)
    
    results = {
        'model_type': model_type,
        'checkpoint': args.checkpoint,
        'cross_domain_results': {}
    }
    
    # Evaluate on each test dataset
    for test_config_path in args.test_datasets:
        with open(test_config_path, 'r') as f:
            test_config = yaml.safe_load(f)
        
        dataset_type = test_config['dataset_type'].lower()
        dataset_path = test_config.get('val_dataset', test_config.get('dataset'))
        images_dir = test_config.get('images_dir')
        batch_size = test_config.get('batch_size', 8)
        max_length = test_config.get('max_length', 512)
        
        print(f"\nEvaluating on {dataset_type}...")
        print(f"Dataset path: {dataset_path}")
        print(f"Images dir: {images_dir}")
        
        try:
            metrics = evaluate_on_dataset(
                model, processor, model_type, dataset_type,
                dataset_path, images_dir, batch_size, max_length, args.max_samples
            )
            
            results['cross_domain_results'][dataset_type] = {
                'dataset_path': dataset_path,
                'images_dir': images_dir,
                'metrics': metrics
            }
            
            if 'error' not in metrics:
                print(f"Results for {dataset_type}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
            else:
                print(f"Error evaluating {dataset_type}: {metrics['error']}")
                
        except Exception as e:
            print(f"Error evaluating {dataset_type}: {str(e)}")
            results['cross_domain_results'][dataset_type] = {
                'error': str(e)
            }
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_file}")
    
    # Print summary
    print("\n=== Cross-Domain Evaluation Summary ===")
    for dataset_type, result in results['cross_domain_results'].items():
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"{dataset_type}:")
            accuracy = metrics.get('accuracy', 'N/A')
            macro_f1 = metrics.get('macro_f1', 'N/A')
            h_score = metrics.get('h_score', 'N/A')
            eci = metrics.get('eci', 'N/A')
            print(f"  Accuracy: {accuracy}")
            print(f"  Macro F1: {macro_f1}")
            print(f"  H-Score: {h_score}")
            print(f"  ECI: {eci}")
        else:
            print(f"{dataset_type}: Error - {result.get('error', 'Unknown error')}")
    print("=======================================")

if __name__ == "__main__":
    main() 