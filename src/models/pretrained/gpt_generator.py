import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("gpt_generator_wrapper")


class PretrainedGPTGenerator:
    """
    Modular Text Generation Wrapper using GPT-2 supporting Greedy, Beam Search, 
    Top-K, and Top-P (Nucleus) Sampling.
    """
    def __init__(self, model_name: str = "gpt2"):
        self.config_cfg = load_config()
        self.model_name = model_name
        self.gen_cfg = self.config_cfg["generation"]

        logger.info(f"Loading pre-trained GPT generator backbone: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

        # Set pad token to eos token for GPT models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        num_beams: int = None,
        do_sample: bool = True,
        device: str = "cpu"
    ) -> str:
        """
        Generates text completion based on input prompt and decoding settings.
        """
        self.model.to(device)
        self.model.eval()

        # Fallback to YAML configuration if parameters aren't explicitly passed
        max_new_tokens = max_new_tokens or self.gen_cfg["max_new_tokens"]
        temperature = temperature or self.gen_cfg["temperature"]
        top_k = top_k or self.gen_cfg["top_k"]
        top_p = top_p or self.gen_cfg["top_p"]
        num_beams = num_beams or self.gen_cfg["num_beams"]

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                top_p=top_p if do_sample else 1.0,
                num_beams=num_beams if not do_sample else 1,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text


if __name__ == "__main__":
    generator = PretrainedGPTGenerator(model_name="gpt2")
    
    prompt = "Artificial Intelligence in modern healthcare is"
    logger.info(f"Prompt: '{prompt}'")
    
    # Test Top-P / Nucleus Sampling Generation
    output_text = generator.generate(prompt=prompt, max_new_tokens=40, do_sample=True)
    
    logger.info("✅ GPT-2 Generator test success!")
    logger.info("✅ Generated Output:\n" + output_text)