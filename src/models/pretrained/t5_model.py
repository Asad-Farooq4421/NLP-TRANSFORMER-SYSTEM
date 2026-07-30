import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("t5_model_wrapper")


class PretrainedT5TaskModel:
    """
    Modular Wrapper for Google T5 (Text-to-Text Transfer Transformer) 
    supporting Text Summarization, Translation, and Multi-task execution.
    """
    def __init__(self, model_name: str = "t5-small"):
        self.config_cfg = load_config()
        self.model_name = model_name

        logger.info(f"Loading pre-trained T5 backbone: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def process_task(
        self,
        task_prefix: str,
        text_input: str,
        max_target_length: int = 100,
        device: str = "cpu"
    ) -> str:
        """
        Executes a sequence-to-sequence transformation task (e.g. summarize, translate).
        
        Args:
            task_prefix: Prefix like 'summarize: ' or 'translate English to German: '
            text_input: The input string to transform
        """
        self.model.to(device)
        self.model.eval()

        full_prompt = f"{task_prefix.strip()} {text_input.strip()}"
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_target_length,
                num_beams=4,
                early_stopping=True
            )

        transformed_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return transformed_text


if __name__ == "__main__":
    t5_wrapper = PretrainedT5TaskModel(model_name="t5-small")

    # Test 1: Summarization
    long_text = (
        "Transfer learning in natural language processing has allowed models trained on vast text corpora "
        "to achieve unprecedented state-of-the-art results across downstream tasks such as classification, "
        "question answering, and entity recognition with minimal fine-tuning effort."
    )
    summary = t5_wrapper.process_task("summarize:", long_text, max_target_length=30)

    # Test 2: Translation
    english_text = "Transformer architectures have revolutionized deep learning."
    german_translation = t5_wrapper.process_task("translate English to German:", english_text, max_target_length=30)

    logger.info("✅ T5 Model Wrapper test success!")
    logger.info(f"✅ Summarization Output: {summary}")
    logger.info(f"✅ Translation Output (DE): {german_translation}")