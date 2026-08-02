import sys
import os
import requests
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("generation_pipeline")


class GenerationPipeline:
    """
    Lightweight Generation Pipeline using Hugging Face Serverless Inference API.
    Prevents Render Out-Of-Memory (OOM) errors by offloading model hosting.
    """
    def __init__(self):
        logger.info("Initializing Hugging Face API Generation Pipeline...")
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}

    def _query_api(self, model_id: str, payload: dict) -> dict:
        """Helper method to send POST requests to Hugging Face Router API."""
        url = f"https://router.huggingface.co/hf-inference/v1/models/{model_id}"
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"HF API Call failed for {model_id}: {str(e)}")
            raise e

    def generate_gpt_completion(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """Generates text completions via HF Inference API."""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": max(temperature, 0.01),
                "top_k": top_k,
                "top_p": top_p,
                "return_full_text": True
            }
        }
        res = self._query_api("gpt2", payload)
        
        if isinstance(res, list) and len(res) > 0:
            full_text = res[0].get("generated_text", prompt)
            # Truncate output at the last full sentence
            last_punct = max(full_text.rfind('.'), full_text.rfind('!'), full_text.rfind('?'))
            if last_punct > len(prompt):
                return full_text[:last_punct + 1]
            return full_text
        return prompt

    def summarize_text(self, text: str, max_length: int = 60) -> str:
        """Summarizes text via HF Inference API (T5-Small)."""
        payload = {
            "inputs": f"summarize: {text}",
            "parameters": {"max_length": max_length, "min_length": 15}
        }
        res = self._query_api("t5-small", payload)
        if isinstance(res, list) and len(res) > 0:
            return res[0].get("summary_text", res[0].get("generated_text", ""))
        return "Failed to generate summary."

    def translate_english_to_german(self, text: str) -> str:
        """Translates English to German via HF Inference API (T5-Small)."""
        payload = {
            "inputs": f"translate English to German: {text}"
        }
        res = self._query_api("t5-small", payload)
        if isinstance(res, list) and len(res) > 0:
            return res[0].get("translation_text", res[0].get("generated_text", ""))
        return "Failed to translate text."


if __name__ == "__main__":
    pipeline = GenerationPipeline()
    sample_text = "Data Science is"
    result = pipeline.generate_gpt_completion(sample_text)
    logger.info(f"✅ HF API Test Output: {result}")