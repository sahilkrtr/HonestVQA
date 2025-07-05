# Configs Directory

This folder contains YAML configuration files for experiments.

## Example Structure

- `layoutlmv3_spdocvqa.yaml` — Config for LayoutLMv3 on SpDocVQA
- `udop_infographicsvqa.yaml` — Config for UDOP on InfographicsVQA
- `donut_sroie.yaml` — Config for DONUT on SROIE

Each config should specify:
- Model name
- Dataset path
- Training hyperparameters (batch size, epochs, learning rate, etc.)
- Output/checkpoint directory

Example:
```yaml
model: microsoft/layoutlmv3-base
dataset: data/spdocvqa_train.jsonl
val_dataset: data/spdocvqa_val.jsonl
batch_size: 8
epochs: 10
learning_rate: 5e-5
output_dir: outputs/layoutlmv3_spdocvqa/
``` 