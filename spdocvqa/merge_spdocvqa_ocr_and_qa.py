import os
import json
from tqdm import tqdm

def normalize_filename(filename):
    # Remove directory, lowercase, ensure .png extension
    base = os.path.basename(filename).lower()
    if base.endswith('.json'):
        base = base.replace('.json', '.png')
    return base

# Paths (edit as needed)
OCR_JSONL = os.path.join(os.path.dirname(__file__), 'spdocvqa_ocr_extracted.jsonl')
QA_JSON = os.path.join(os.path.dirname(__file__), 'spdocvqa_qas/train_v1.0_withQT.json')
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'spdocvqa_images')
OUTPUT_JSONL = os.path.join(os.path.dirname(__file__), '../spdocvqa_merged_train.jsonl')

# Load OCR/box data
ocr_dict = {}
with open(OCR_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        key = normalize_filename(entry['file'])
        ocr_dict[key] = entry['words_bboxes']

# Load QA data
with open(QA_JSON, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)['data']

warned = set()
# Merge and write
with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out_f:
    for qa in tqdm(qa_data, desc='Merging QA and OCR'):
        image_name = normalize_filename(qa['image'])
        image_path = os.path.join(IMAGES_DIR, os.path.basename(qa['image']))
        if image_name in ocr_dict:
            merged_entry = {
                'image_path': image_path,
                'words_bboxes': ocr_dict[image_name],
                'question': qa['question'],
                'answers': qa['answers']
            }
            out_f.write(json.dumps(merged_entry, ensure_ascii=False) + '\n')
        else:
            if image_name not in warned:
                print(f"Warning: No OCR data found for {image_name}")
                warned.add(image_name)

print(f"Merged file written to {OUTPUT_JSONL}") 