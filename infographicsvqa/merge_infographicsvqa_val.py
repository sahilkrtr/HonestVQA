import os
import json
from tqdm import tqdm

# Paths (edit as needed)
QA_JSON = os.path.join(os.path.dirname(__file__), 'infographicsvqa_qas/infographicsVQA_val_v1.0_withQT.json')
OCR_DIR = os.path.join(os.path.dirname(__file__), 'infographicsvqa_ocr')
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'infographicsvqa_images')
OUTPUT_JSONL = os.path.join(os.path.dirname(__file__), '../infographicsvqa_merged_val.jsonl')

# Load QA data
with open(QA_JSON, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)['data']

# Merge and write
with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out_f:
    for qa in tqdm(qa_data, desc='Merging QA and OCR'):
        ocr_file = qa['ocr_output_file']
        ocr_path = os.path.join(OCR_DIR, ocr_file)
        image_name = qa['image_local_name']
        image_path = os.path.join(IMAGES_DIR, image_name)
        
        if os.path.exists(ocr_path):
            with open(ocr_path, 'r', encoding='utf-8') as ocr_f:
                ocr_data = json.load(ocr_f)
                words_bboxes = ocr_data.get('words_bboxes', [])
                
                merged_entry = {
                    'image_path': image_path,
                    'words_bboxes': words_bboxes,
                    'question': qa['question'],
                    'answers': qa['answers']
                }
                out_f.write(json.dumps(merged_entry, ensure_ascii=False) + '\n')
        else:
            print(f"Warning: No OCR file found for {ocr_file}")

print(f"Merged file written to {OUTPUT_JSONL}") 