# HonestVQA

This repository provides a clean, research-focused pipeline for multimodal Visual Question Answering (VQA) on three datasets (SpDocVQA, InfographicsVQA, SROIE) using LayoutLMv3, DONUT, and UDOP models. All scripts are set up for real experiments on the full datasets—no sample/test logic remains.

## Quick Start

1. **Install dependencies**
   ```bash
   python3 -m venv venv310_new
   source venv310_new/bin/activate  # or .\venv310_new\Scripts\activate.bat on Windows
   pip install -r requirements.txt
   ```

2. **Train a model**
   ```bash
   python train.py --config configs/layoutlmv3_spdocvqa.yaml
   ```

3. **Evaluate a model**
   ```bash
   python eval.py --config configs/layoutlmv3_spdocvqa.yaml
   ```

4. **Run the full pipeline**
   ```bash
   python run_experiments.py --config configs/layoutlmv3_spdocvqa.yaml
   ```

5. **Visualize results**
   ```bash
   python visualize_results.py --results_dir experiment_results
   ```

## Key Features
- Full dataset training and evaluation (no sample/test code)
- HonestVQA framework: uncertainty, alignment, contrastive loss
- Novel metrics: H-Score, ECI, IoU
- Cross-domain, ablation, and computational analysis
- Modular configs for all models/datasets

## Configuration Example
```yaml
model_type: layoutlmv3
dataset_type: spdocvqa
dataset: spdocvqa_merged_train.jsonl
val_dataset: spdocvqa_merged_val.jsonl
images_dir: spdocvqa/spdocvqa_images
batch_size: 8
max_length: 512
epochs: 10
learning_rate: 5e-5
```

## Supported Models & Datasets

### Datasets
- **SpDocVQA**: [Access SpDocVQA](https://rrc.cvc.uab.es/?ch=17&com=downloads)
- **InfographicsVQA**: [Access InfographicsVQA](https://rrc.cvc.uab.es/?ch=17&com=downloads) 
- **SROIE**: [Access SROIE](https://rrc.cvc.uab.es/?ch=13)

### Models
- **LayoutLMv3**: [Download from Hugging Face](https://huggingface.co/microsoft/layoutlmv3-base)
- **DONUT**: [Download from Hugging Face](https://huggingface.co/naver-clova-ix/donut-base)
- **UDOP**: [Download from Hugging Face](https://huggingface.co/microsoft/udop-large)

🖥️ Note on Performance Variability: Evaluation results may vary by up to ±10% depending on your hardware configuration, especially GPU type, memory bandwidth, and compute environment. This margin reflects differences in numerical precision, runtime optimizations, and stability of training dynamics during model training and evaluation.

