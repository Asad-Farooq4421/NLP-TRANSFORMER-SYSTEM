import sys
import os
from pathlib import Path
import torch
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.utils.metrics import compute_classification_metrics
from src.data.dataset import DatasetManager
from src.models.pretrained.bert_classifier import PretrainedTransformerClassifier

logger = setup_logger("evaluator_pipeline")


class ModelEvaluator:
    """
    Evaluation pipeline for analyzing classification performance, 
    generating metrics reports, and computing confusion matrices.
    """
    def __init__(self, model_path: str = None, num_classes: int = 4):
        self.config = load_config()
        self.num_classes = num_classes
        self.dataset_manager = DatasetManager()
        
        target_path = Path(model_path) if model_path else None
        
        if target_path and target_path.exists():
            logger.info(f"Loading fine-tuned model checkpoint from: {target_path}")
            self.classifier = PretrainedTransformerClassifier(
                model_name=str(target_path),
                num_classes=num_classes
            )
        else:
            logger.info("Saved model checkpoint not found. Using default 'distilbert-base-uncased' backbone...")
            self.classifier = PretrainedTransformerClassifier(
                model_name="distilbert-base-uncased",
                num_classes=num_classes
            )

    def evaluate_dataset(self, dataset_name: str = "ag_news", split: str = "test[:50]"):
        """
        Runs inference over dataset split and computes classification metrics.
        """
        logger.info(f"Loading '{dataset_name}' dataset (split='{split}')...")

        if dataset_name == "ag_news":
            texts, labels = self.dataset_manager.load_ag_news(split=split)
        elif dataset_name == "imdb":
            texts, labels = self.dataset_manager.load_imdb(split=split)
        else:
            texts, labels = self.dataset_manager.load_sst2(split=split)

        logger.info(f"Running inference over {len(texts)} samples...")
        predictions = []
        self.classifier.eval()

        with torch.no_grad():
            for idx, text in enumerate(texts):
                result = self.classifier.predict_text(text)
                predictions.append(result["predicted_class"])

        logger.info("Calculating metrics and confusion matrix...")
        metrics = compute_classification_metrics((predictions, labels))
        matrix = confusion_matrix(labels, predictions)
        report = classification_report(labels, predictions, zero_division=0)

        logger.info("==================================================")
        logger.info("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info(f"✅ Accuracy & F1 Metrics: {metrics}")
        logger.info(f"✅ Confusion Matrix:\n{matrix}")
        logger.info(f"✅ Full Classification Report:\n{report}")
        logger.info("==================================================")

        return {
            "metrics": metrics,
            "confusion_matrix": matrix.tolist(),
            "classification_report": report
        }


if __name__ == "__main__":
    saved_checkpoint = PROJECT_ROOT / "saved_models" / "fine_tuned_ag_news"
    
    evaluator = ModelEvaluator(model_path=str(saved_checkpoint), num_classes=4)
    results = evaluator.evaluate_dataset(dataset_name="ag_news", split="test[:50]")