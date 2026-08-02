import sys
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
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

models_dict = {}


def load_classifier_sync():
    """Synchronous loader executed in a background worker thread."""
    if "classifier_ag_news" not in models_dict:
        logger.info("Loading base DistilBERT classifier from Hugging Face Hub...")
        models_dict["classifier_ag_news"] = PretrainedTransformerClassifier(
            model_name="distilbert-base-uncased", num_classes=4
        )
    return models_dict["classifier_ag_news"]


async def get_classifier_async():
    """Asynchronous wrapper that offloads blocking model downloads off the main GIL loop."""
    return await asyncio.to_thread(load_classifier_sync)


def load_generator_sync():
    if "generation_pipeline" not in models_dict:
        logger.info("Loading Generation Pipeline...")
        models_dict["generation_pipeline"] = GenerationPipeline()
    return models_dict["generation_pipeline"]


async def get_generation_pipeline_async():
    return await asyncio.to_thread(load_generator_sync)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 API Server online!")
    yield
    models_dict.clear()


app = FastAPI(
    title="NLP System with Transformer Models API",
    version="1.0.0",
    lifespan=lifespan
)

# Open CORS configuration for cross-domain requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def root():
    return {"status": "online", "message": "Transformer Studio API is running live on Render!"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(status="online", version="1.0.0")


@app.post("/predict/classify", response_model=ClassificationResponse, tags=["NLP Tasks"])
async def classify_text(request: TextClassificationRequest):
    try:
        classifier = await get_classifier_async()
        result = classifier.predict_text(request.text)
        
        categories = ["World", "Sports", "Business", "Sci/Tech"]
        raw_pred = result.get("predicted_class", 0)
        class_id = raw_pred if isinstance(raw_pred, int) and 0 <= raw_pred < len(categories) else 0
        confidence = float(result.get("confidence", 0.0))
        probabilities = result.get("probabilities", [0.0, 0.0, 0.0, 0.0])

        return ClassificationResponse(
            text=request.text,
            predicted_class=class_id,
            confidence=confidence,
            probabilities=probabilities
        )
    except Exception as e:
        logger.error(f"Classification Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/predict/generate", response_model=GenerationResponse, tags=["NLP Tasks"])
async def generate_text(request: TextGenerationRequest):
    try:
        gen_pipeline = await get_generation_pipeline_async()
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
async def summarize_text(request: TextSummarizationRequest):
    try:
        gen_pipeline = await get_generation_pipeline_async()
        summary = gen_pipeline.summarize_text(request.text, max_length=request.max_length)
        return SummarizationResponse(original_text=request.text, summary=summary)
    except Exception as e:
        logger.error(f"Summarization Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@app.post("/predict/translate", response_model=TranslationResponse, tags=["NLP Tasks"])
async def translate_text(request: TextTranslationRequest):
    try:
        gen_pipeline = await get_generation_pipeline_async()
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