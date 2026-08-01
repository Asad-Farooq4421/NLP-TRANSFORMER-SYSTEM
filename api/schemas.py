from pydantic import BaseModel, Field
from typing import List, Optional, Union

# --- Request Schemas ---

class TextClassificationRequest(BaseModel):
    text: str = Field(..., example="This movie was absolutely brilliant with great acting!")
    task: str = Field(default="sentiment", example="sentiment")  # "sentiment" or "topic"

class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., example="Artificial Intelligence in modern healthcare is")
    max_tokens: int = Field(default=50, ge=5, le=200)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=1, le=100)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)

class TextSummarizationRequest(BaseModel):
    text: str = Field(..., example="Transformers have revolutionized natural language processing by enabling parallel token processing.")
    max_length: int = Field(default=60, ge=10, le=200)

class TextTranslationRequest(BaseModel):
    text: str = Field(..., example="Transformer architectures have revolutionized deep learning.")
    target_language: str = Field(default="German", example="German")

# --- Response Schemas ---

class ClassificationResponse(BaseModel):
    text: str
    predicted_class: Union[str, int]  # Accepts category string (e.g., "Business") or integer ID (e.g., 2)
    confidence: float
    probabilities: List[float]

class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str

class SummarizationResponse(BaseModel):
    original_text: str
    summary: str

class TranslationResponse(BaseModel):
    original_text: str
    target_language: str
    translated_text: str

class HealthResponse(BaseModel):
    status: str
    version: str