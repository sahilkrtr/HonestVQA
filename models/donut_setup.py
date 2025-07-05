from transformers import DonutProcessor, VisionEncoderDecoderModel

def setup_donut(model_name="naver-clova-ix/donut-base", cache_dir=None):
    """
    Checks for DONUT model and processor, downloads if not present.
    Args:
        model_name (str): Hugging Face model name.
        cache_dir (str or None): Optional cache directory for Hugging Face models.
    Returns:
        model, processor
    """
    print(f"Checking and downloading DONUT model: {model_name}")
    processor = DonutProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=cache_dir)
    print("DONUT model and processor are ready.")
    return model, processor

if __name__ == "__main__":
    setup_donut() 