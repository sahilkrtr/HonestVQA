# HonestVQA

📚 Datasets

The following datasets can be accessed from their respective official sources:

    **SpDocVQA**
    A challenging benchmark for document visual question answering with complex layouts and text.
    [Access SpDocVQA](https://github.com/ibm/spdocvqa)

    **InfographicsVQA**
    A comprehensive dataset for infographic visual question answering with diverse visual elements.
    [Access InfographicsVQA](https://github.com/google-research-datasets/infographicvqa)

    **SROIE**
    A dataset focused on receipt understanding and information extraction from scanned documents.
    [Access SROIE](https://rrc.cvc.uab.es/?ch=13)

🧠 Multimodal Models

The following multimodal models can be downloaded from Hugging Face:

    **LayoutLMv3**
    [Download from Hugging Face](https://huggingface.co/microsoft/layoutlmv3-base)

    **DONUT**
    [Download from Hugging Face](https://huggingface.co/naver-clova-ix/donut-base)

    **UDOP**
    [Download from Hugging Face](https://huggingface.co/microsoft/udop-large)

🛠️ Usage: train.py

The script train.py is designed to train multimodal visual question answering models on the listed datasets. It can be used with any of the supported models to train and fine-tune, including:

    ✅ Model Training
    📊 Validation
    🧱 Checkpointing
    ⚙️ Hyperparameter Tuning

### How to Use

Simply pass your chosen model and dataset configuration to train.py to start training. The script supports:

    Any of the models listed above (e.g., LayoutLMv3, DONUT, UDOP)
    Any of the supported datasets (e.g., SpDocVQA, InfographicsVQA, SROIE)

🔁 Follow-up Processing and Evaluation

Once models have been trained using train.py, you can proceed with evaluation and analysis using any of the following scripts:

    `eval.py`
    `run_experiments.py`
    `visualize_results.py`

These scripts are designed to evaluate model performance and analyze results across different datasets and model configurations.

📈 Evaluation

After training any of the models, use eval.py to assess the performance of the trained models.

    ⚠️ Note: Make sure you have the appropriate access to the evaluation metrics and datasets used for assessment. These include:

* Exact Match Accuracy
* F1 Score
* H-Score (Honesty Score)
* ECI (Expected Calibration Index)
* IoU (Intersection over Union)

These evaluators are used to provide comprehensive assessment of model performance in terms of accuracy, calibration, and honesty.

🖥️ Note on Performance Variability: Evaluation results may vary by up to ±10% depending on your hardware configuration, especially GPU type, memory bandwidth, and compute environment. This margin reflects differences in numerical precision, runtime optimizations, and stability of training dynamics during model training and evaluation.
