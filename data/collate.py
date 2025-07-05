from torch.utils.data._utils.collate import default_collate

class LayoutLMv3Collator:
    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
    def __call__(self, batch):
        images = [item['image'] for item in batch]
        words = [item['words'] for item in batch]
        boxes = [item['bboxes'] for item in batch]
        encoding = self.processor(
            images,
            words,
            boxes=boxes,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoding

class UDOPCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __call__(self, batch):
        # Extract all components for UDOP multimodal input
        images = [item['image'] for item in batch]
        questions = [item['question'] for item in batch]
        words = [item['words'] for item in batch]
        boxes = [item['bboxes'] for item in batch]
        
        # Convert bounding boxes to float format for UDOP
        float_boxes = []
        for box_list in boxes:
            float_box_list = [[float(x) for x in box] for box in box_list]
            float_boxes.append(float_box_list)
        
        # UDOP expects text with corresponding bounding boxes
        encoding = self.tokenizer(
            text=questions,
            text_pair=words,  # Use words as text_pair
            boxes=float_boxes,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Add images to the encoding
        encoding['images'] = images
        return encoding

class DonutCollator:
    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
    def __call__(self, batch):
        images = [item['image'] for item in batch]
        questions = [item['question'] for item in batch]
        
        # Create prompts for DONUT
        prompts = []
        for question in questions:
            # Format question as a prompt for document understanding
            prompt = f"<s_docvqa><question>{question}</question><answer>"
            prompts.append(prompt)
        
        # Use the processor to handle both images and text
        encoding = self.processor(
            images,
            text=prompts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # For vision-encoder-decoder, we need to separate encoder and decoder inputs
        # The encoder only gets pixel_values, decoder gets text inputs
        return {
            "pixel_values": encoding["pixel_values"],
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding.get("attention_mask", None),
            "labels": encoding["labels"]
        } 