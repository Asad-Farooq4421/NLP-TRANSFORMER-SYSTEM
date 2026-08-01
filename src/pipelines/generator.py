import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pretrained.gpt_generator import PretrainedGPTGenerator
from src.models.pretrained.t5_model import PretrainedT5Model
from src.utils.logger import setup_logger

logger = setup_logger("generation_pipeline")


class GenerationPipeline:
    """
    Unified Pipeline Manager for Generative Transformer Tasks:
    - Autoregressive Text Generation (GPT-2)
    - Abstractive Summarization (T5-Small)
    - Neural Machine Translation (T5-Small)
    """
    def __init__(self):
        logger.info("Initializing Generation Pipeline...")
        self.gpt_generator = PretrainedGPTGenerator(model_name="gpt2")
        self.t5_model = PretrainedT5Model(model_name="t5-small")

    def generate_gpt_completion(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """
        Generates text completions using GPT-2 model wrapper.
        """
        try:
            # Handle method name fallback (generate_text vs generate_gpt_completion)
            if hasattr(self.gpt_generator, "generate_text"):
                return self.gpt_generator.generate_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
            elif hasattr(self.gpt_generator, "generate_completion"):
                return self.gpt_generator.generate_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
            else:
                raise AttributeError("GPT Generator object missing text generation method.")
        except Exception as e:
            logger.error(f"Error generating GPT completion: {str(e)}")
            raise e

    def summarize_text(self, text: str, max_length: int = 60) -> str:
        """
        Summarizes input text using Google T5 model wrapper.
        """
        try:
            if hasattr(self.t5_model, "summarize"):
                return self.t5_model.summarize(text=text, max_length=max_length)
            elif hasattr(self.t5_model, "summarize_text"):
                return self.t5_model.summarize_text(text=text, max_length=max_length)
            else:
                raise AttributeError("T5 Model object missing summarization method.")
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            raise e

    def translate_english_to_german(self, text: str) -> str:
        """
        Translates English text to German using Google T5 model wrapper.
        """
        try:
            if hasattr(self.t5_model, "translate"):
                return self.t5_model.translate(text=text)
            elif hasattr(self.t5_model, "translate_english_to_german"):
                return self.t5_model.translate_english_to_german(text=text)
            else:
                raise AttributeError("T5 Model object missing translation method.")
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            raise e


if __name__ == "__main__":
    pipeline = GenerationPipeline()
    sample_text = "Data Science is"
    result = pipeline.generate_gpt_completion(sample_text)
    logger.info(f"✅ Test Generation Output: {result}")