import os
import json
from torch.utils.data import Dataset
from PIL import Image
from .sroie_dataset_loader import SROIEDatasetHF

class SpDocVQADataset(Dataset):
    """Dataset loader for SpDocVQA."""
    def __init__(self, jsonl_path, images_dir, normalize_bboxes=False, max_samples=None):
        self.samples = []
        self.images_dir = images_dir
        self.normalize_bboxes = normalize_bboxes
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                entry = json.loads(line)
                # Expecting: image_path, words_bboxes, question, answers
                self.samples.append({
                    'image_path': entry['image_path'],
                    'words_bboxes': entry['words_bboxes'],
                    'question': entry['question'],
                    'answers': entry['answers']
                })
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        words = [wb['text'] for wb in sample['words_bboxes']]
        bboxes = [wb['bbox'] for wb in sample['words_bboxes']]
        if self.normalize_bboxes and bboxes:
            w, h = image.size
            bboxes = [
                [
                    int(1000 * max(0, min(x, w)) / w) if i % 2 == 0 else int(1000 * max(0, min(x, h)) / h)
                    for i, x in enumerate(bbox)
                ]
                for bbox in bboxes
            ]
        return {
            'image': image,
            'words': words,
            'bboxes': bboxes,
            'question': sample['question'],
            'answers': sample['answers']
        }

class InfographicsVQADataset(Dataset):
    """Dataset loader for InfographicsVQA."""
    def __init__(self, jsonl_path, images_dir, normalize_bboxes=False, max_samples=None):
        self.samples = []
        self.images_dir = images_dir
        self.normalize_bboxes = normalize_bboxes
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                entry = json.loads(line)
                self.samples.append({
                    'image_path': entry['image_path'],
                    'words_bboxes': entry['words_bboxes'],
                    'question': entry['question'],
                    'answers': entry['answers']
                })
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        words = [wb['text'] for wb in sample['words_bboxes']]
        bboxes = [wb['bbox'] for wb in sample['words_bboxes']]
        if self.normalize_bboxes and bboxes:
            w, h = image.size
            bboxes = [
                [
                    int(1000 * max(0, min(x, w)) / w) if i % 2 == 0 else int(1000 * max(0, min(x, h)) / h)
                    for i, x in enumerate(bbox)
                ]
                for bbox in bboxes
            ]
        return {
            'image': image,
            'words': words,
            'bboxes': bboxes,
            'question': sample['question'],
            'answers': sample['answers']
        }

class SROIEDataset(Dataset):
    """Dataset loader for SROIE."""
    def __init__(self, jsonl_path, images_dir=None, max_samples=None):
        self.samples = []
        self.images_dir = images_dir
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                entry = json.loads(line)
                # image_path is relative to the Hugging Face cache or provided images_dir
                image_path = entry['image_path']
                if images_dir is not None:
                    image_path = os.path.join(images_dir, os.path.basename(image_path))
                self.samples.append({
                    'image_path': image_path,
                    'words': entry['words'],
                    'bboxes': entry['bboxes'],
                    'ner_tags': entry.get('ner_tags', None),
                    'id': entry['id']
                })
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except (FileNotFoundError, OSError):
            # Create a dummy image if the file doesn't exist
            print(f"Warning: Image not found at {sample['image_path']}, creating dummy image")
            image = Image.new('RGB', (800, 600), color='white')
        return {
            'image': image,
            'words': sample['words'],
            'bboxes': sample['bboxes'],
            'ner_tags': sample['ner_tags'],
            'id': sample['id']
        } 