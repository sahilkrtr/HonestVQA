#!/usr/bin/env python3
"""
SROIE Dataset loader using Hugging Face datasets.
"""

import os
import json
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset

class SROIEDatasetHF(Dataset):
    """SROIE Dataset loader using Hugging Face datasets."""
    def __init__(self, split='train', max_samples=None):
        self.samples = []
        self.split = split
        
        # Load SROIE dataset from Hugging Face
        try:
            dataset = load_dataset("darentang/sroie", split=split)
            print(f"Loaded SROIE {split} dataset with {len(dataset)} samples")
            
            # Convert to our format
            for i, item in enumerate(dataset):
                if max_samples is not None and i >= max_samples:
                    break
                    
                # Extract image
                image = item['image']
                
                # Extract text and bounding boxes
                words = item['words']
                bboxes = item['bboxes']
                ner_tags = item.get('ner_tags', None)
                
                # Create sample
                sample = {
                    'image': image,
                    'words': words,
                    'bboxes': bboxes,
                    'ner_tags': ner_tags,
                    'id': str(i),
                    'question': 'What is the total amount?',  # Dummy question for VQA compatibility
                    'answers': ['test']  # Dummy answer for VQA compatibility
                }
                self.samples.append(sample)
                
        except Exception as e:
            print(f"Error loading SROIE dataset: {e}")
            print("Falling back to local JSONL file...")
            # Fallback to local file
            self._load_from_jsonl(split, max_samples)
    
    def _load_from_jsonl(self, split, max_samples):
        """Fallback to loading from local JSONL file."""
        jsonl_path = f"sroie/sroie/sroie_{split}_formatted.jsonl"
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"SROIE {split} file not found: {jsonl_path}")
            
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                entry = json.loads(line)
                
                # Create dummy image since we don't have the actual images
                image = Image.new('RGB', (800, 600), color='white')
                
                sample = {
                    'image': image,
                    'words': entry['words'],
                    'bboxes': entry['bboxes'],
                    'ner_tags': entry.get('ner_tags', None),
                    'id': entry['id'],
                    'question': 'What is the total amount?',  # Dummy question for VQA compatibility
                    'answers': ['test']  # Dummy answer for VQA compatibility
                }
                self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx] 