import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("bert_classifier_wrapper")

# AG News standard label mapping
AG_NEWS_ID2LABEL = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}


class PretrainedTransformerClassifier(nn.Module):
    """
    Modular PyTorch Wrapper for HuggingFace Pre-trained Classification Models 
    (BERT, DistilBERT, RoBERTa).
    """
    def __init__(self, model_name: str = "distilbert-base-uncased", num_classes: int = 4):
        super().__init__()
        self.config_cfg = load_config()
        self.model_name = model_name
        self.num_classes = num_classes

        logger.info(f"Loading pre-trained classifier backbone: {self.model_name}...")

        # Load Hugging Face Model & Tokenizer
        model_path = Path(self.model_name)
        if model_path.exists():
            # Load fine-tuned local checkpoint without overriding config
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        else:
            # Load base pre-trained model with new classification head
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_classes
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, labels: torch.Tensor = None):
        """
        Forward pass through pre-trained classifier.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def predict_text(self, text: str, device: str = "cpu") -> dict:
        """
        Inference helper method for raw text strings.
        """
        self.eval()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.config_cfg["data"]["max_length"]
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class].item()

        # Resolve string label if within 4-class AG News scope
        predicted_label = AG_NEWS_ID2LABEL.get(pred_class, str(pred_class))

        return {
            "predicted_class": predicted_label,
            "predicted_class_id": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": [round(p.item(), 4) for p in probs[0]]
        }


if __name__ == "__main__":
    classifier_wrapper = PretrainedTransformerClassifier(
        model_name="distilbert-base-uncased",
        num_classes=4
    )

    sample_text = "Wall Street stocks surged today as tech earnings exceeded analyst predictions across all quarters."
    result = classifier_wrapper.predict_text(sample_text)

    logger.info("✅ Pre-trained Classifier Wrapper test success!")
    logger.info(f"✅ Input Text: {sample_text}")
    logger.info(f"✅ Prediction Result: {result}")