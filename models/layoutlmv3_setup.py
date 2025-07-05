from transformers import AutoProcessor, LayoutLMv3ForQuestionAnswering

def setup_layoutlmv3(model_name="microsoft/layoutlmv3-base", cache_dir=None):
    """
    Checks for LayoutLMv3 model and processor, downloads if not present.
    Args:
        model_name (str): Hugging Face model name.
        cache_dir (str or None): Optional cache directory for Hugging Face models.
    Returns:
        model, processor
    """
    print(f"Checking and downloading LayoutLMv3 model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False, cache_dir=cache_dir)
    model = LayoutLMv3ForQuestionAnswering.from_pretrained(model_name, cache_dir=cache_dir)
    print("LayoutLMv3 model and processor are ready.")
    return model, processor

if __name__ == "__main__":
    setup_layoutlmv3() 