import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset_loader import SpDocVQADataset, InfographicsVQADataset, SROIEDataset
from data.sroie_dataset_loader import SROIEDatasetHF
from data.collate import LayoutLMv3Collator, UDOPCollator, DonutCollator
from models import setup_layoutlmv3, setup_udop, setup_donut
from sentence_transformers import SentenceTransformer, util as st_util
import os

# Map config model names to setup/collator/dataset
MODEL_REGISTRY = {
    'layoutlmv3': {
        'setup': setup_layoutlmv3,
        'collator': LayoutLMv3Collator,
        'dataset': SpDocVQADataset  # default, can be changed by config
    },
    'udop': {
        'setup': setup_udop,
        'collator': UDOPCollator,
        'dataset': InfographicsVQADataset  # default, can be changed by config
    },
    'donut': {
        'setup': setup_donut,
        'collator': DonutCollator,
        'dataset': SROIEDataset  # default, can be changed by config
    }
}

DATASET_REGISTRY = {
    'spdocvqa': SpDocVQADataset,
    'infographicsvqa': InfographicsVQADataset,
    'sroie': SROIEDatasetHF
}

def compute_entropy_and_confidence(logits):
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # shape: (batch, ...)
    confidence = torch.max(probs, dim=-1).values  # shape: (batch, ...)
    return entropy, confidence

def alignment_loss(y_pred, y_true, confidence, alpha=1.0, beta=0.5):
    ce_loss = F.cross_entropy(y_pred, y_true)
    pred_labels = y_pred.argmax(dim=-1)
    penalty = alpha * (pred_labels != y_true).float() * confidence
    return beta * ce_loss + penalty.mean(), ce_loss.item(), penalty.mean().item()

def contrastive_loss_fn(anchor, positive, negative, margin=0.5):
    # anchor, positive, negative: (batch, embed_dim)
    cos = torch.nn.CosineSimilarity(dim=-1)
    pos_sim = cos(anchor, positive)
    neg_sim = cos(anchor, negative)
    loss = torch.clamp(margin - pos_sim + neg_sim, min=0.0)
    return loss.mean(), pos_sim.mean().item(), neg_sim.mean().item()

def main():
    parser = argparse.ArgumentParser(description="Unified training script for HonestVQA experiments.")
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_type = config['model_type'].lower()
    dataset_type = config['dataset_type'].lower()
    dataset_path = config['dataset']
    images_dir = config.get('images_dir', None)
    batch_size = config.get('batch_size', 8)
    max_length = config.get('max_length', 512)
    epochs = config.get('epochs', 10)
    learning_rate = float(config.get('learning_rate', 5e-5))
    output_dir = config.get('output_dir', 'outputs/')
    os.makedirs(output_dir, exist_ok=True)
    # Alignment loss hyperparameters
    use_alignment_loss = config.get('use_alignment_loss', True)
    alpha = config.get('alignment_alpha', 1.0)
    beta = config.get('alignment_beta', 0.5)
    # Contrastive loss hyperparameters
    use_contrastive_loss = config.get('use_contrastive_loss', True)
    lambda2 = config.get('contrastive_lambda', 0.7)
    margin = config.get('contrastive_margin', 0.5)
    # SentenceTransformer for contrastive loss
    embedder = SentenceTransformer('all-MiniLM-L6-v2') if use_contrastive_loss else None

    # Setup model and processor
    model_setup = MODEL_REGISTRY[model_type]['setup']
    model, processor = model_setup()

    # Setup dataset and collator
    dataset_cls = DATASET_REGISTRY[dataset_type]
    max_samples = 10  # Limit for safe testing; increase or set to None for full run
    if model_type == 'layoutlmv3':
        if dataset_type == 'sroie':
            # Use Hugging Face SROIE dataset
            dataset = dataset_cls(split='train', max_samples=max_samples)
        else:
            dataset = dataset_cls(dataset_path, images_dir, normalize_bboxes=True, max_samples=max_samples)
    else:
        if dataset_type == 'sroie':
            dataset = dataset_cls(split='train', max_samples=max_samples)
        else:
            dataset = dataset_cls(dataset_path, images_dir, max_samples=max_samples)
    collator_cls = MODEL_REGISTRY[model_type]['collator']
    collate_fn = collator_cls(processor, max_length=max_length)

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Training {model_type} on {dataset_type} for {epochs} epochs ({len(dataset)} samples).")

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
            
            # Remove images from batch for UDOP
            if 'images' in batch:
                batch.pop('images')
            
            # Handle different model types
            if model_type == 'donut':
                # For DONUT, create labels for the decoder
                input_ids = batch['input_ids']
                labels = input_ids.clone()
                # Set prompt tokens to -100 (ignore in loss)
                prompt_length = 10  # Approximate length of "<s_docvqa><question>...</question><answer>"
                labels[:, :prompt_length] = -100
                
                # For vision-encoder-decoder, use separate encoder and decoder calls
                # First, encode the image
                encoder_outputs = model.encoder(pixel_values=batch["pixel_values"])
                
                # Then, decode with text inputs
                decoder_outputs = model.decoder(
                    input_ids=input_ids,
                    attention_mask=batch.get("attention_mask", None),
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    labels=labels
                )
                
                loss = decoder_outputs.loss if hasattr(decoder_outputs, 'loss') else None
                if hasattr(decoder_outputs, 'logits'):
                    entropy, confidence = compute_entropy_and_confidence(decoder_outputs.logits)
                    all_entropy.append(entropy.detach().cpu())
                    all_confidence.append(confidence.detach().cpu())
            elif 'labels' in batch:
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
                # Contrastive loss (scaffold: expects anchor/pos/neg answers in batch)
                if use_contrastive_loss and all(x in batch for x in ['anchor_answer', 'positive_answer', 'negative_answer']):
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
        if all_entropy:
            all_entropy = torch.cat(all_entropy)
            all_confidence = torch.cat(all_confidence)
            log_str = f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}"
            if use_alignment_loss:
                avg_ce_loss = total_ce_loss / len(dataloader)
                avg_penalty = total_penalty / len(dataloader)
                log_str += f" | CE: {avg_ce_loss:.4f} | Penalty: {avg_penalty:.4f}"
            if use_contrastive_loss:
                avg_contrastive = total_contrastive / len(dataloader)
                log_str += f" | Contrastive: {avg_contrastive:.4f}"
            log_str += f" | Entropy: {all_entropy.mean():.4f} | Confidence: {all_confidence.mean():.4f}"
            print(log_str)
        else:
            print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    main() 