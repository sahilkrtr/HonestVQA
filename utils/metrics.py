import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from typing import List, Tuple, Dict, Any

def compute_entropy_and_confidence(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute entropy and confidence from model logits.
    
    Args:
        logits: Model output logits [batch_size, num_classes]
        
    Returns:
        entropy: Entropy values [batch_size]
        confidence: Confidence values [batch_size]
    """
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    confidence = torch.max(probs, dim=-1).values
    return entropy, confidence

def compute_h_score(confidence: torch.Tensor, accuracy: torch.Tensor) -> float:
    """
    Compute H-Score (Honesty Score) as per Equation (8) in the paper.
    
    Args:
        confidence: Model confidence scores [batch_size]
        accuracy: Binary accuracy scores (1 for correct, 0 for incorrect) [batch_size]
        
    Returns:
        h_score: Honesty score between 0 and 1
    """
    h_score = 1 - torch.mean(torch.abs(confidence - accuracy))
    return h_score.item()

def compute_eci(confidence: torch.Tensor, accuracy: torch.Tensor) -> float:
    """
    Compute ECI (Expected Calibration Index) using ROC-AUC.
    
    Args:
        confidence: Model confidence scores [batch_size]
        accuracy: Binary accuracy scores (1 for correct, 0 for incorrect) [batch_size]
        
    Returns:
        eci: Expected Calibration Index
    """
    try:
        eci = roc_auc_score(accuracy.cpu().numpy(), confidence.cpu().numpy())
        return eci
    except ValueError:
        # Handle case where all predictions are the same class
        return 0.5

def compute_macro_f1(predictions: List[int], labels: List[int]) -> float:
    """
    Compute Macro F1 score.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        
    Returns:
        macro_f1: Macro F1 score
    """
    return f1_score(labels, predictions, average='macro')

def compute_accuracy(predictions: List[int], labels: List[int]) -> float:
    """
    Compute accuracy score.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        
    Returns:
        accuracy: Accuracy score
    """
    return accuracy_score(labels, predictions)

def evaluate_model_performance(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Comprehensive evaluation of model performance including novel metrics.
    
    Args:
        logits: Model output logits [batch_size, num_classes]
        labels: True labels [batch_size]
        
    Returns:
        metrics: Dictionary containing all evaluation metrics
    """
    # Get predictions
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Compute standard metrics
    accuracy = compute_accuracy(predictions, labels_np)
    macro_f1 = compute_macro_f1(predictions, labels_np)
    
    # Compute uncertainty metrics
    entropy, confidence = compute_entropy_and_confidence(logits)
    
    # Compute binary accuracy for each sample
    binary_accuracy = (predictions == labels_np).astype(float)
    binary_accuracy_tensor = torch.tensor(binary_accuracy, device=logits.device)
    
    # Compute novel metrics
    h_score = compute_h_score(confidence, binary_accuracy_tensor)
    eci = compute_eci(confidence, binary_accuracy_tensor)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'h_score': h_score,
        'eci': eci,
        'mean_entropy': entropy.mean().item(),
        'mean_confidence': confidence.mean().item()
    }

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] coordinates of first box
        box2: [x1, y1, x2, y2] coordinates of second box
        
    Returns:
        iou: IoU value between 0 and 1
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def compute_attention_iou(attention_weights: torch.Tensor, ocr_bboxes: List[List[float]], 
                         image_size: Tuple[int, int]) -> float:
    """
    Compute IoU between model attention and OCR bounding boxes.
    
    Args:
        attention_weights: Attention weights from model [seq_len]
        ocr_bboxes: List of OCR bounding boxes [x1, y1, x2, y2]
        image_size: (width, height) of the image
        
    Returns:
        mean_iou: Average IoU between attention and OCR boxes
    """
    if not ocr_bboxes:
        return 0.0
    
    # Normalize attention weights
    attention_weights = F.softmax(attention_weights, dim=-1)
    
    # For simplicity, we'll use the top-k attended tokens
    # In practice, you'd need to map tokens to spatial locations
    top_k = min(5, len(attention_weights))
    top_indices = torch.topk(attention_weights, top_k).indices
    
    # This is a simplified implementation
    # In a full implementation, you'd need to:
    # 1. Map token positions to spatial coordinates
    # 2. Create attention bounding boxes
    # 3. Compute IoU with OCR boxes
    
    # Placeholder: return average attention weight as proxy
    return attention_weights.mean().item() 