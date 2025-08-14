import torch
import torch_neuronx
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np

class RoBERTaNEREncoder:
    def __init__(self, 
                 model_name="Davlan/xlm-roberta-base-ner-hrl",
                 sample_text="The company HuggingFace is based in New York City"):

        self.name = model_name
        self.sample_text = sample_text

        # Load tokenizer and model for NER (Token Classification)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            torchscript=True,
            return_dict=False  # Important for tracing
        )

        # Set model to evaluation mode
        self.model.eval()

    def encode_for_ner(self, text, max_length=512, batch_size=1):
        """Encode text for NER task - different from sequence classification
        Returns input_ids and attention_mask (no token_type_ids for RoBERTa)
        """
        # Tokenize the input text
        tokens = self.tokenizer.encode_plus(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )

        # RoBERTa doesn't use token_type_ids, so we only need input_ids and attention_mask
        input_ids = torch.repeat_interleave(tokens['input_ids'], batch_size, 0)
        attention_mask = torch.repeat_interleave(tokens['attention_mask'], batch_size, 0)

        return input_ids, attention_mask

    def decode_ner_predictions(self, outputs, input_ids):
        """Convert model outputs to readable NER predictions"""
        
        # FIX: Extract logits from outputs (like your working script)
        if isinstance(outputs, dict):
            logits = outputs["logits"]  # If outputs is a dict
        else:
            logits = outputs[0]  # If outputs is a tuple, take first element
        
        # Get predictions (shape: batch_size, seq_len, num_labels)
        predictions = torch.argmax(logits, dim=-1)
        
        # Rest of your code stays the same...
        results = []
        for i, (pred_seq, input_seq) in enumerate(zip(predictions, input_ids)):
            tokens = self.tokenizer.convert_ids_to_tokens(input_seq)
            entities = []
            
            for j, (token, pred_id) in enumerate(zip(tokens, pred_seq)):
                if token in ['<s>', '</s>', '<pad>', '<unk>'] or token.startswith('<'):
                    continue
                    
                # Convert prediction ID to label
                label = self.model.config.id2label.get(pred_id.item(), 'O')
                if label != 'O':  # Only include actual entities
                    entities.append({
                        'token': token,
                        'label': label,
                        'position': j,
                        'confidence': torch.softmax(logits[i][j], dim=0)[pred_id].item()
                    })
            
            results.append(entities)
        return results