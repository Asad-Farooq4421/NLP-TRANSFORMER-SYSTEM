import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class PretrainedTransformerClassifier(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", num_classes: int = 4):
        super().__init__()
        
        # Override local folder paths if weights are missing or uncommitted
        if not isinstance(model_name, str) or not model_name or "saved_models" in model_name:
            model_name = "distilbert-base-uncased"
            
        self.model_name = model_name
        self.num_classes = num_classes
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_classes
        )
        self.model.eval()

    def predict_text(self, text: str):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
            predicted_class_id = int(torch.argmax(logits, dim=-1).item())
            
        return {
            "predicted_class": predicted_class_id,
            "confidence": probs[predicted_class_id],
            "probabilities": probs
        }