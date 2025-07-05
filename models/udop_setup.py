from transformers import UdopForConditionalGeneration, AutoTokenizer

def setup_udop(model_name="microsoft/udop-large", cache_dir=None):
    """
    Checks for UDOP model and processor, downloads if not present.
    Args:
        model_name (str): Hugging Face model name.
        cache_dir (str or None): Optional cache directory for Hugging Face models.
    Returns:
        model, tokenizer
    """
    print(f"Checking and downloading UDOP model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=cache_dir)
    model = UdopForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
    print("UDOP model and tokenizer are ready.")
    return model, tokenizer

if __name__ == "__main__":
    setup_udop() 