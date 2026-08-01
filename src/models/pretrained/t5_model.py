import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("t5_model_wrapper")


class PretrainedT5Model(nn.Module):
    """
    Modular PyTorch Wrapper for HuggingFace Sequence-to-Sequence T5 Models 
    (used for Summarization and Translation).
    """
    def __init__(self, model_name: str = "t5-small"):
        super().__init__()
        self.config_cfg = load_config()
        self.model_name = model_name

        logger.info(f"Loading pre-trained T5 backbone: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def summarize(
        self,
        text: str,
        max_length: int = 60,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> str:
        """
        Summarizes input text using T5 prefix format.
        """
        self.model.to(device)
        self.eval()

        input_text = f"summarize: {text}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_length,
                min_length=15,
                num_beams=4,
                early_stopping=True
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def translate(
        self,
        text: str,
        target_language: str = "German",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> str:
        """
        Translates English text to target language (e.g., German) using T5 prefix format.
        """
        self.model.to(device)
        self.eval()

        input_text = f"translate English to German: {text}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=150,
                num_beams=4,
                early_stopping=True
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    t5_wrapper = PretrainedT5Model("t5-small")
    sample_text = "Transformer architectures have revolutionized natural language processing by enabling parallel token processing."
    
    summary = t5_wrapper.summarize(sample_text)
    translation = t5_wrapper.translate(sample_text)

    logger.info("✅ T5 Model Wrapper test success!")
    logger.info(f"✅ Summary Output: {summary}")
    logger.info(f"✅ Translation Output: {translation}")