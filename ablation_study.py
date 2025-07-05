import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset_loader import SpDocVQADataset, InfographicsVQADataset, SROIEDataset
from data.collate import LayoutLMv3Collator, UDOPCollator, DonutCollator
from models import setup_layoutlmv3, setup_udop, setup_donut
from utils.metrics import evaluate_model_performance, compute_entropy_and_confidence
from sentence_transformers import SentenceTransformer
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

def alignment_loss(y_pred, y_true, confidence, alpha=1.0, beta=0.5):
    """Alignment loss as per Equation (2) in the paper."""
    ce_loss = F.cross_entropy(y_pred, y_true)
    pred_labels = y_pred.argmax(dim=-1)
    penalty = alpha * (pred_labels != y_true).float() * confidence
    return beta * ce_loss + penalty.mean(), ce_loss.item(), penalty.mean().item()

def contrastive_loss_fn(anchor, positive, negative, margin=0.5):
    """Contrastive loss as per Equation (3) in the paper."""
    cos = torch.nn.CosineSimilarity(dim=-1)
    pos_sim = cos(anchor, positive)
    neg_sim = cos(anchor, negative)
    loss = torch.clamp(margin - pos_sim + neg_sim, min=0.0)
    return loss.mean(), pos_sim.mean().item(), neg_sim.mean().item()

def train_with_ablation(model, dataloader, optimizer, device, 
                       use_alignment_loss=True, use_contrastive_loss=True,
                       alpha=1.0, beta=0.5, lambda2=0.7, margin=0.5,
                       embedder=None, epochs=1):
    """Train model with specified ablation settings."""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_ce_loss = 0.0
        total_penalty = 0.0
        total_contrastive = 0.0
        all_entropy = []
        all_confidence = []
        
        for batch in dataloader:
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            optimizer.zero_grad()
            loss = None
            ce_loss_val = penalty_val = contrastive_val = 0.0
            
            if 'labels' in batch:
                outputs = model(**batch)
                logits = outputs.logits
                entropy, confidence = compute_entropy_and_confidence(logits)
                all_entropy.append(entropy.detach().cpu())
                all_confidence.append(confidence.detach().cpu())
                
                # Alignment loss
                if use_alignment_loss:
                    loss, ce_loss_val, penalty_val = alignment_loss(logits, batch['labels'], confidence, alpha=alpha, beta=beta)
                    total_ce_loss += ce_loss_val
                    total_penalty += penalty_val
                else:
                    loss = F.cross_entropy(logits, batch['labels'])
                
                # Contrastive loss
                if use_contrastive_loss and embedder and all(x in batch for x in ['anchor_answer', 'positive_answer', 'negative_answer']):
                    anchor_emb = torch.tensor(embedder.encode(batch['anchor_answer'], convert_to_numpy=True)).to(device)
                    pos_emb = torch.tensor(embedder.encode(batch['positive_answer'], convert_to_numpy=True)).to(device)
                    neg_emb = torch.tensor(embedder.encode(batch['negative_answer'], convert_to_numpy=True)).to(device)
                    contrastive, pos_sim, neg_sim = contrastive_loss_fn(anchor_emb, pos_emb, neg_emb, margin=margin)
                    loss = loss + lambda2 * contrastive
                    total_contrastive += contrastive.item()
            else:
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else None
                if hasattr(outputs, 'logits'):
                    entropy, confidence = compute_entropy_and_confidence(outputs.logits)
                    all_entropy.append(entropy.detach().cpu())
                    all_confidence.append(confidence.detach().cpu())
            
            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        if use_alignment_loss:
            avg_ce_loss = total_ce_loss / len(dataloader)
            avg_penalty = total_penalty / len(dataloader)
            print(f"  CE Loss: {avg_ce_loss:.4f}, Penalty: {avg_penalty:.4f}")
        if use_contrastive_loss:
            avg_contrastive = total_contrastive / len(dataloader)
            print(f"  Contrastive Loss: {avg_contrastive:.4f}")

def evaluate_model(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()
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

def run_ablation_study(config_path: str, checkpoint_path: str = None, 
                      output_dir: str = 'ablation_results', max_samples: int = 100):
    """Run comprehensive ablation study."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_type = config['model_type'].lower()
    dataset_type = config['dataset_type'].lower()
    dataset_path = config['dataset']
    images_dir = config.get('images_dir', None)
    batch_size = config.get('batch_size', 8)
    max_length = config.get('max_length', 512)
    learning_rate = float(config.get('learning_rate', 5e-5))
    
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
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    
    ablation_configs = [
        {
            'name': 'full_model',
            'use_alignment_loss': True,
            'use_contrastive_loss': True,
            'description': 'Full HonestVQA model with both losses'
        },
        {
            'name': 'baseline',
            'use_alignment_loss': False,
            'use_contrastive_loss': False,
            'description': 'Baseline model without HonestVQA losses'
        }
    ]
    
    results = {
        'model_type': model_type,
        'dataset_type': dataset_type,
        'checkpoint': checkpoint_path,
        'ablation_results': {}
    }
    
    # SentenceTransformer for contrastive loss
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    for ablation_config in ablation_configs:
        print(f"\n=== Running {ablation_config['name']} ===")
        print(f"Description: {ablation_config['description']}")
        
        # Reset model to checkpoint state
        if checkpoint_path and os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Train with ablation settings
        train_with_ablation(
            model, dataloader, optimizer, device,
            use_alignment_loss=ablation_config['use_alignment_loss'],
            use_contrastive_loss=ablation_config['use_contrastive_loss'],
            embedder=embedder if ablation_config['use_contrastive_loss'] else None,
            epochs=1  # Short training for ablation study
        )
        
        # Evaluate
        eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        metrics = evaluate_model(model, eval_dataloader, device)
        
        results['ablation_results'][ablation_config['name']] = {
            'description': ablation_config['description'],
            'config': ablation_config,
            'metrics': metrics
        }
        
        print(f"Results for {ablation_config['name']}:")
        if 'error' not in metrics:
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"  Error: {metrics['error']}")
    
    # Save results
    output_file = os.path.join(output_dir, f'ablation_study_{model_type}_{dataset_type}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAblation study results saved to {output_file}")
    
    # Print summary
    print("\n=== Ablation Study Summary ===")
    for config_name, result in results['ablation_results'].items():
        print(f"\n{config_name}:")
        print(f"  Description: {result['description']}")
        if 'metrics' in result and 'error' not in result['metrics']:
            metrics = result['metrics']
            print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"  Macro F1: {metrics.get('macro_f1', 'N/A'):.4f}")
            print(f"  H-Score: {metrics.get('h_score', 'N/A'):.4f}")
            print(f"  ECI: {metrics.get('eci', 'N/A'):.4f}")
        else:
            print(f"  Error: {result['metrics'].get('error', 'Unknown error')}")
    print("==============================")

def main():
    parser = argparse.ArgumentParser(description="Ablation study for HonestVQA experiments.")
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML')
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='ablation_results', help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=10, help='Maximum samples for testing')
    args = parser.parse_args()
    
    run_ablation_study(args.config, args.checkpoint, args.output_dir, args.max_samples)

if __name__ == "__main__":
    main() 