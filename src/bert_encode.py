# bert_encode.py - CORRECTED VERSION

import torch
import torch_neuronx
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BERTEncoder:
    def __init__(self, 
                 model_name="bert-base-cased-finetuned-mrpc",
                 sequence_0="The company HuggingFace is based in New York City",
                 sequence_1="Apples are especially bad for your health", 
                 sequence_2="HuggingFace's headquarters are situated in Manhattan"):
        
        self.name = model_name
        self.sequence_0 = sequence_0
        self.sequence_1 = sequence_1
        self.sequence_2 = sequence_2
        
        # Load once, reuse many times
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torchscript=True)
    
    def encode(self, *inputs, max_length=512, batch_size=4):  # FIXED: Proper defaults
        # Handle token_type_ids - some models (like RoBERTa) don't have them
        tokens = self.tokenizer.encode_plus(
            *inputs,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        if 'token_type_ids' not in tokens:
            # Create dummy token_type_ids (all zeros) for models that don't use them
            tokens['token_type_ids'] = torch.zeros_like(tokens['input_ids'])
        
        return (
            torch.repeat_interleave(tokens['input_ids'], batch_size, 0),
            torch.repeat_interleave(tokens['attention_mask'], batch_size, 0),
            torch.repeat_interleave(tokens['token_type_ids'], batch_size, 0),
        )