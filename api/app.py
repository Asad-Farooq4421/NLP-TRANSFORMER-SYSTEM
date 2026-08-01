import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (
    TextClassificationRequest, ClassificationResponse,
    TextGenerationRequest, GenerationResponse,
    TextSummarizationRequest, SummarizationResponse,
    TextTranslationRequest, TranslationResponse,
    HealthResponse
)
from src.pipelines.generator import GenerationPipeline
from src.models.pretrained.bert_classifier import PretrainedTransformerClassifier
from src.utils.logger import setup_logger

logger = setup_logger("fastapi_app")

# Global pipeline dictionary
models_dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI Lifespan Context Manager for Model Loading & Teardown."""
    logger.info("Initializing API Model Pipelines...")
    
    models_dict["generation_pipeline"] = GenerationPipeline()
    
    saved_checkpoint = PROJECT_ROOT / "saved_models" / "fine_tuned_ag_news"
    if saved_checkpoint.exists():
        logger.info(f"Loading fine-tuned classifier from {saved_checkpoint}...")
        models_dict["classifier_ag_news"] = PretrainedTransformerClassifier(
            model_name=str(saved_checkpoint), num_classes=4
        )
    else:
        logger.info("Checkpoint not found. Loading default DistilBERT classifier...")
        models_dict["classifier_ag_news"] = PretrainedTransformerClassifier(
            model_name="distilbert-base-uncased", num_classes=4
        )
        
    logger.info("✅ All API models successfully loaded!")
    yield
    logger.info("Shutting down API and releasing resources...")
    models_dict.clear()


# Initialize FastAPI Application
app = FastAPI(
    title="NLP System with Transformer Models API",
    description="RESTful API providing Text Classification, Autoregressive Text Generation, Summarization, and Translation using Transformer Architectures.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint to verify system status."""
    return HealthResponse(status="online", version="1.0.0")


@app.post("/predict/classify", response_model=ClassificationResponse, tags=["NLP Tasks"])
def classify_text(request: TextClassificationRequest):
    """Classifies input text into categories using fine-tuned DistilBERT."""
    try:
        classifier = models_dict.get("classifier_ag_news")
        result = classifier.predict_text(request.text)
        
        # Pass the integer ID (0, 1, 2, 3) so frontend JS array mapping works seamlessly
        class_id = result.get("predicted_class_id", result.get("predicted_class"))
        
        return ClassificationResponse(
            text=request.text,
            predicted_class=class_id,
            confidence=result["confidence"],
            probabilities=result["probabilities"]
        )
    except Exception as e:
        logger.error(f"Classification Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/predict/generate", response_model=GenerationResponse, tags=["NLP Tasks"])
def generate_text(request: TextGenerationRequest):
    """Generates autoregressive text completions using GPT-2."""
    try:
        gen_pipeline = models_dict.get("generation_pipeline")
        completion = gen_pipeline.generate_gpt_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
        return GenerationResponse(prompt=request.prompt, generated_text=completion)
    except Exception as e:
        logger.error(f"Text Generation Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/predict/summarize", response_model=SummarizationResponse, tags=["NLP Tasks"])
def summarize_text(request: TextSummarizationRequest):
    """Summarizes text using Google T5."""
    try:
        gen_pipeline = models_dict.get("generation_pipeline")
        summary = gen_pipeline.summarize_text(request.text, max_length=request.max_length)
        return SummarizationResponse(original_text=request.text, summary=summary)
    except Exception as e:
        logger.error(f"Summarization Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@app.post("/predict/translate", response_model=TranslationResponse, tags=["NLP Tasks"])
def translate_text(request: TextTranslationRequest):
    """Translates English text to German using Google T5."""
    try:
        gen_pipeline = models_dict.get("generation_pipeline")
        translated = gen_pipeline.translate_english_to_german(request.text)
        return TranslationResponse(
            original_text=request.text,
            target_language=request.target_language,
            translated_text=translated
        )
    except Exception as e:
        logger.error(f"Translation Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)