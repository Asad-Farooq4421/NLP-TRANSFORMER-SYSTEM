import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pretrained.gpt_generator import PretrainedGPTGenerator
from src.models.pretrained.t5_model import PretrainedT5TaskModel
from src.utils.logger import setup_logger

logger = setup_logger("generation_pipeline")


class GenerationPipeline:
    """
    Unified pipeline managing autoregressive text generation (GPT-2) 
    and task-based sequence-to-sequence generation (T5).
    """
    def __init__(self):
        logger.info("Initializing Generation Pipeline...")
        self.gpt_gen = PretrainedGPTGenerator(model_name="gpt2")
        self.t5_gen = PretrainedT5TaskModel(model_name="t5-small")

    def generate_gpt_completion(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        num_beams: int = 1,
        do_sample: bool = True
    ) -> str:
        """Runs GPT-2 text completion."""
        return self.gpt_gen.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            do_sample=do_sample
        )

    def summarize_text(self, text: str, max_length: int = 60) -> str:
        """Summarizes input text using T5."""
        return self.t5_gen.process_task("summarize:", text, max_target_length=max_length)

    def translate_english_to_german(self, text: str, max_length: int = 60) -> str:
        """Translates English text to German using T5."""
        return self.t5_gen.process_task("translate English to German:", text, max_target_length=max_length)


if __name__ == "__main__":
    pipeline = GenerationPipeline()
    
    # 1. Test GPT Completion
    prompt = "The future of deep learning is"
    completion = pipeline.generate_gpt_completion(prompt, max_tokens=30)
    logger.info(f"✅ GPT Completion Output:\n{completion}")

    # 2. Test T5 Summarization
    long_article = (
        "Transformers have replaced traditional Recurrent Neural Networks (RNNs) "
        "and Long Short-Term Memory networks (LSTMs) in natural language processing. "
        "Through self-attention mechanisms, they allow for parallel processing during training."
    )
    summary = pipeline.summarize_text(long_article, max_length=25)
    logger.info(f"✅ T5 Summary Output: {summary}")