import os
import json
from tqdm import tqdm

OCR_DIR = os.path.join(os.path.dirname(__file__), 'spdocvqa_ocr')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'spdocvqa_ocr_extracted.jsonl')


def extract_words_and_bboxes(ocr_json):
    results = []
    for page in ocr_json.get('recognitionResults', []):
        for line in page.get('lines', []):
            for word in line.get('words', []):
                # Convert 8-point polygon to 4-point bbox (x0, y0, x2, y2)
                bbox = word['boundingBox']
                x_coords = bbox[::2]
                y_coords = bbox[1::2]
                x0, y0 = min(x_coords), min(y_coords)
                x1, y1 = max(x_coords), max(y_coords)
                results.append({
                    'text': word['text'],
                    'bbox': [x0, y0, x1, y1]
                })
    return results


def main():
    all_files = [f for f in os.listdir(OCR_DIR) if f.endswith('.json')]
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for fname in tqdm(all_files, desc='Processing OCR files'):
            fpath = os.path.join(OCR_DIR, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                ocr_json = json.load(f)
            words_bboxes = extract_words_and_bboxes(ocr_json)
            out = {
                'file': fname,
                'words_bboxes': words_bboxes
            }
            out_f.write(json.dumps(out) + '\n')
    print(f"Extraction complete. Output saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main() 