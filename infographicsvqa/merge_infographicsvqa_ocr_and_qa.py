import os
import json
from tqdm import tqdm

# Paths (edit as needed)
QA_JSON = os.path.join(os.path.dirname(__file__), 'infographicsvqa_qas/infographicsVQA_train_v1.0.json')
OCR_DIR = os.path.join(os.path.dirname(__file__), 'infographicsvqa_ocr')
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'infographicsvqa_images')
OUTPUT_JSONL = os.path.join(os.path.dirname(__file__), '../infographicsvqa_merged_train.jsonl')

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
        # Load OCR/box data for this sample
        if os.path.exists(ocr_path):
            with open(ocr_path, 'r', encoding='utf-8') as ocr_f:
                ocr_json = json.load(ocr_f)
            words_bboxes = ocr_json.get('words_bboxes', [])
        else:
            words_bboxes = []
        out = {
            'image_path': image_path,
            'words_bboxes': words_bboxes,
            'question': qa['question'],
            'answers': qa['answers']
        }
        out_f.write(json.dumps(out) + '\n')
print(f"Merged file written to {OUTPUT_JSONL}") 