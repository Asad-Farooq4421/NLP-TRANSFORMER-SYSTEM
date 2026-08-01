import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("gpt_generator_wrapper")


class PretrainedGPTGenerator(nn.Module):
    """
    Modular PyTorch Wrapper for HuggingFace Autoregressive GPT Models.
    Leverages `gpt2-large` with sentence boundary truncation for high-quality generation.
    """
    def __init__(self, model_name: str = "gpt2-large"):
        super().__init__()
        self.config_cfg = load_config()
        self.model_name = model_name

        logger.info(f"Loading pre-trained GPT generator backbone: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Set padding token to end-of-sequence token if not already present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 70,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> str:
        """
        Generates creative, coherent text completions truncated cleanly at sentence boundaries.
        """
        self.model.to(device)
        self.eval()

        prompt = prompt.strip()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_tokens,
                temperature=temperature,       # Balanced for creative + factual output
                top_k=top_k,                     # Filter out weird low-probability tokens
                top_p=top_p,                     # Nucleus sampling for context retention
                do_sample=True,
                repetition_penalty=1.1,        # Prevents word & phrase loops
                no_repeat_ngram_size=3,         # Eliminates 3-word duplicate sequences
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode generated token IDs back to string
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Clean Sentence Truncation: Cut off at last complete sentence boundary (. ! ?)
        last_punct = max(full_text.rfind('.'), full_text.rfind('!'), full_text.rfind('?'))
        if last_punct > len(prompt):
            clean_text = full_text[:last_punct + 1]
        else:
            clean_text = full_text

        return clean_text


if __name__ == "__main__":
    generator = PretrainedGPTGenerator("gpt2-large")
    sample_prompt = "Imran Khan is a politician and is well known for"
    result = generator.generate_text(sample_prompt, max_tokens=70, temperature=0.7)
    
    logger.info("✅ GPT-2 Large Generator test success!")
    logger.info(f"✅ Generated Output:\n{result}")