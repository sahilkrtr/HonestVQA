import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from data.dataset_loader import SpDocVQADataset, InfographicsVQADataset, SROIEDataset
from data.collate import LayoutLMv3Collator, UDOPCollator, DonutCollator
from models import setup_layoutlmv3, setup_udop, setup_donut
from utils.metrics import evaluate_model_performance, compute_entropy_and_confidence
import os

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

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation script for HonestVQA experiments.")
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML')
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to model checkpoint')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_type = config['model_type'].lower()
    dataset_type = config['dataset_type'].lower()
    dataset_path = config['val_dataset']
    images_dir = config.get('images_dir', None)
    batch_size = config.get('batch_size', 8)
    max_length = config.get('max_length', 512)

    # Setup model and processor/tokenizer
    model_setup = MODEL_REGISTRY[model_type]['setup']
    if model_type == 'udop':
        model, tokenizer = model_setup()
    else:
        model, processor = model_setup()

    # Load checkpoint if provided
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Setup dataset and collator
    dataset_cls = DATASET_REGISTRY[dataset_type]
    max_samples = None  # Use full dataset for evaluation
    if model_type == 'layoutlmv3':
        if dataset_type == 'sroie':
            dataset = dataset_cls(dataset_path, images_dir, max_samples=max_samples)
        else:
            dataset = dataset_cls(dataset_path, images_dir, normalize_bboxes=True, max_samples=max_samples)
    else:
        dataset = dataset_cls(dataset_path, images_dir, max_samples=max_samples)
    collator_cls = MODEL_REGISTRY[model_type]['collator']
    if model_type == 'udop':
        collate_fn = collator_cls(tokenizer, max_length=max_length)
    else:
        collate_fn = collator_cls(processor, max_length=max_length)

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print(f"Evaluating {model_type} on {dataset_type} with {len(dataset)} samples.")

    all_logits = []
    all_labels = []
    all_entropy = []
    all_confidence = []
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            if 'decoder_input_ids' in batch and 'decoder_inputs_embeds' in batch:
                batch.pop('decoder_inputs_embeds')
            if 'images' in batch:
                batch.pop('images')  # Remove images from batch for UDOP
            if model_type == 'donut':
                input_ids = batch['input_ids']
                labels = input_ids.clone()
                prompt_length = 10
                labels[:, :prompt_length] = -100
                encoder_outputs = model.encoder(pixel_values=batch["pixel_values"])
                outputs = model.decoder(
                    input_ids=input_ids,
                    attention_mask=batch.get("attention_mask", None),
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    labels=labels
                )
                # Generate predictions with shorter max_length for faster testing
                generated_ids = model.generate(
                    pixel_values=batch["pixel_values"],
                    max_length=128,  # Reduced from 512 for faster testing
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    do_sample=False,  # Use greedy decoding for speed
                    num_beams=1  # No beam search for speed
                )
                # Decode predictions and references
                predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
                # Filter out -100 tokens from labels before decoding
                valid_labels = []
                for label_seq in labels:
                    # Convert to list and filter out negative values
                    label_list = label_seq.tolist() if hasattr(label_seq, 'tolist') else list(label_seq)
                    valid_tokens = [token for token in label_list if token >= 0]
                    if valid_tokens:  # Only add if there are valid tokens
                        valid_labels.append(valid_tokens)
                    else:
                        valid_labels.append([processor.tokenizer.pad_token_id])  # Fallback
                references = processor.batch_decode(valid_labels, skip_special_tokens=True)
                all_predictions.extend(predictions)
                all_references.extend(references)
            else:
                outputs = model(**batch)
                # Handle different model outputs
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    all_logits.append(logits.cpu())
                    entropy, confidence = compute_entropy_and_confidence(logits)
                    all_entropy.append(entropy.cpu())
                    all_confidence.append(confidence.cpu())
                elif hasattr(outputs, 'start_logits') and hasattr(outputs, 'end_logits'):
                    start_logits = outputs.start_logits
                    end_logits = outputs.end_logits
                    all_logits.append(start_logits.cpu())
                    entropy, confidence = compute_entropy_and_confidence(start_logits)
                    all_entropy.append(entropy.cpu())
                    all_confidence.append(confidence.cpu())
                if 'labels' in batch:
                    all_labels.append(batch['labels'].cpu())
    
    # Combine all batches
    if model_type == 'donut':
        # String-based metrics for DONUT
        exact_matches = [int(p.strip() == r.strip()) for p, r in zip(all_predictions, all_references)]
        accuracy = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        print("\n=== DONUT Evaluation Results ===")
        print(f"Exact Match Accuracy: {accuracy:.4f}")
        print(f"Sample predictions:")
        for i in range(min(3, len(all_predictions))):
            print(f"  Pred: {all_predictions[i]}")
            print(f"  Ref : {all_references[i]}")
        print("========================\n")
    elif all_logits:
        all_logits = torch.cat(all_logits, dim=0)
        all_entropy = torch.cat(all_entropy, dim=0)
        all_confidence = torch.cat(all_confidence, dim=0)
        if all_labels:
            all_labels = torch.cat(all_labels, dim=0)
            metrics = evaluate_model_performance(all_logits, all_labels)
            print("\n=== Evaluation Results ===")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Macro F1: {metrics['macro_f1']:.4f}")
            print(f"H-Score: {metrics['h_score']:.4f}")
            print(f"ECI: {metrics['eci']:.4f}")
            print(f"Mean Entropy: {metrics['mean_entropy']:.4f}")
            print(f"Mean Confidence: {metrics['mean_confidence']:.4f}")
            print("========================\n")
        else:
            print(f"Predictions collected: {len(all_logits)}")
            print(f"Mean Entropy: {all_entropy.mean():.4f}")
            print(f"Mean Confidence: {all_confidence.mean():.4f}")
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    main() 