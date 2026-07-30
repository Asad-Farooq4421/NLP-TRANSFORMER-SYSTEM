import sys
from pathlib import Path
import math
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import evaluate

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("metrics_evaluator")

def compute_classification_metrics(eval_pred) -> dict:
    """
    Computes classification evaluation metrics for HuggingFace Trainer or PyTorch eval loop.
    
    Args:
        eval_pred: Tuple of (predictions, labels) or EvalPrediction object.
        
    Returns:
        dict: Accuracy, Precision, Recall, F1 score.
    """
    if isinstance(eval_pred, tuple):
        predictions, labels = eval_pred
    else:
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        
    if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
        preds = np.argmax(predictions, axis=1)
    else:
        preds = predictions

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4)
    }

def calculate_perplexity(loss: float) -> float:
    """
    Calculates Perplexity from cross-entropy loss.
    PPL = exp(loss)
    """
    try:
        return round(math.exp(loss), 4)
    except OverflowError:
        return float("inf")

def compute_bleu_score(predictions: list[str], references: list[list[str]]) -> dict:
    """
    Calculates BLEU score for text generation models using HuggingFace Evaluate.
    
    Args:
        predictions: List of generated sentence strings.
        references: List of lists of target sentence strings.
    """
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=references)
    return {"bleu": round(results["bleu"], 4)}

def get_confusion_matrix(y_true: list, y_pred: list) -> np.ndarray:
    """Generates confusion matrix array."""
    return confusion_matrix(y_true, y_pred)

if __name__ == "__main__":
    # Quick sanity check test
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    
    metrics = compute_classification_metrics((y_pred, y_true))
    ppl = calculate_perplexity(1.5)
    
    logger.info(f"✅ Classification Metrics Test Output: {metrics}")
    logger.info(f"✅ Perplexity Test (Loss=1.5): {ppl}")