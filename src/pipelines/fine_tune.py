import sys
import os
from pathlib import Path
import torch
from transformers import TrainingArguments, Trainer

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.utils.metrics import compute_classification_metrics
from src.data.dataset import DatasetManager, NLPTextDataset
from src.models.pretrained.bert_classifier import PretrainedTransformerClassifier

logger = setup_logger("fine_tune_pipeline")


class FineTuningPipeline:
    """
    Pipeline to fine-tune pre-trained transformer models (BERT/DistilBERT) 
    on classification datasets (IMDB, AG News, SST-2).
    """
    def __init__(self, model_name: str = "distilbert-base-uncased", dataset_name: str = "ag_news"):
        self.config = load_config()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_manager = DatasetManager()
        
        # Determine number of classes based on dataset
        self.num_classes = 4 if dataset_name == "ag_news" else 2

    def run(self, train_samples: int = 10000, eval_samples: int = 2000):
        """
        Runs complete loading, tokenization, training, and evaluation workflow.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Starting fine-tuning pipeline for {self.model_name} on {self.dataset_name} using device: {device.upper()}...")

        # 1. Initialize Wrapper Model & Tokenizer
        classifier_wrapper = PretrainedTransformerClassifier(
            model_name=self.model_name,
            num_classes=self.num_classes
        )
        tokenizer = classifier_wrapper.tokenizer

        # 2. Load Raw Cleaned Data
        if self.dataset_name == "ag_news":
            train_texts, train_labels = self.dataset_manager.load_ag_news(split=f"train[:{train_samples}]")
            eval_texts, eval_labels = self.dataset_manager.load_ag_news(split=f"test[:{eval_samples}]")
        elif self.dataset_name == "imdb":
            train_texts, train_labels = self.dataset_manager.load_imdb(split=f"train[:{train_samples}]")
            eval_texts, eval_labels = self.dataset_manager.load_imdb(split=f"test[:{eval_samples}]")
        else:
            train_texts, train_labels = self.dataset_manager.load_sst2(split=f"train[:{train_samples}]")
            eval_texts, eval_labels = self.dataset_manager.load_sst2(split=f"validation[:{eval_samples}]")

        # 3. Create PyTorch Datasets
        train_dataset = NLPTextDataset(train_texts, train_labels, tokenizer=tokenizer)
        eval_dataset = NLPTextDataset(eval_texts, eval_labels, tokenizer=tokenizer)

        # 4. Output Directory for Checkpoints
        output_dir = PROJECT_ROOT / "saved_models" / f"fine_tuned_{self.dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        # 5. Define Hugging Face Training Arguments (Optimized for RTX 4070)
        use_fp16 = torch.cuda.is_available()  # Enable mixed precision training if CUDA GPU available
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config["fine_tuning"]["epochs"],
            per_device_train_batch_size=32,   # Optimized batch size for 8GB VRAM
            per_device_eval_batch_size=32,
            learning_rate=float(self.config["fine_tuning"]["learning_rate"]),
            weight_decay=float(self.config["fine_tuning"]["weight_decay"]),
            warmup_steps=self.config["fine_tuning"]["warmup_steps"],
            fp16=use_fp16,                    # Accelerated FP16 computation for NVIDIA RTX
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=str(PROJECT_ROOT / "logs"),
            logging_steps=20,
            report_to="none"
        )

        # 6. Instantiate Hugging Face Trainer
        trainer = Trainer(
            model=classifier_wrapper.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_classification_metrics
        )

        # 7. Execute Training Loop
        logger.info("Executing training loop on GPU...")
        trainer.train()

        # 8. Final Evaluation
        logger.info("Evaluating fine-tuned model...")
        eval_results = trainer.evaluate()
        
        logger.info(f"✅ Fine-Tuning Complete! Results: {eval_results}")
        
        # Save Final Model and Tokenizer
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        logger.info(f"✅ Best model checkpoint saved to: {output_dir}")


if __name__ == "__main__":
    # Execute full fine-tuning with 10,000 training samples and 2,000 evaluation samples
    pipeline = FineTuningPipeline(model_name="distilbert-base-uncased", dataset_name="ag_news")
    pipeline.run(train_samples=10000, eval_samples=2000)