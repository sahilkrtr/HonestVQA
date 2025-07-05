import os
import json
from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'sroie')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the SROIE dataset from Hugging Face
ds = load_dataset('darentang/sroie')

for split in ['train', 'test']:
    output_file = os.path.join(OUTPUT_DIR, f'sroie_{split}_formatted.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for entry in ds[split]:
            # Verify required fields
            assert all(k in entry for k in ['id', 'words', 'bboxes', 'ner_tags', 'image_path']), f"Missing fields in entry: {entry}"
            out = {
                'id': entry['id'],
                'words': entry['words'],
                'bboxes': entry['bboxes'],
                'ner_tags': entry['ner_tags'],
                'image_path': entry['image_path']
            }
            f_out.write(json.dumps(out) + '\n')
    print(f"Saved {split} split to {output_file}") 