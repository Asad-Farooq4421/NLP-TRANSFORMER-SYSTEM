import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.data.preprocessor import TextPreprocessor

logger = setup_logger("dataset_loader")

class NLPTextDataset(Dataset):
    """
    Custom PyTorch Dataset wrapper for tokenized text inputs.
    """
    def __init__(self, texts: list[str], labels: list[int], tokenizer=None, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            item = {key: val.squeeze(0) for key, val in encoding.items()}
            item["label"] = torch.tensor(label, dtype=torch.long)
            return item

        return {"text": text, "label": torch.tensor(label, dtype=torch.long)}


class DatasetManager:
    """
    Manages loading, cleaning, and creating DataLoaders for IMDB, AG News, and SST-2.
    """
    def __init__(self):
        self.config = load_config()
        self.preprocessor = TextPreprocessor()

    def load_imdb(self, split: str = "train"):
        """Loads and cleans IMDB dataset."""
        logger.info(f"Loading IMDB dataset split: {split}...")
        ds = load_dataset(self.config["data"]["datasets"]["imdb"], split=split)
        
        cleaned_texts = self.preprocessor.batch_clean(ds["text"])
        labels = ds["label"]
        return cleaned_texts, labels

    def load_ag_news(self, split: str = "train"):
        """Loads and cleans AG News dataset (Topic Classification)."""
        logger.info(f"Loading AG News dataset split: {split}...")
        ds = load_dataset(self.config["data"]["datasets"]["ag_news"], split=split)
        
        cleaned_texts = self.preprocessor.batch_clean(ds["text"])
        labels = ds["label"]
        return cleaned_texts, labels

    def load_sst2(self, split: str = "train"):
        """Loads and cleans SST-2 dataset from GLUE benchmark."""
        logger.info(f"Loading SST-2 dataset split: {split}...")
        ds = load_dataset(self.config["data"]["datasets"]["sst2"], "sst2", split=split)
        
        cleaned_texts = self.preprocessor.batch_clean(ds["sentence"])
        labels = ds["label"]
        return cleaned_texts, labels

    def create_dataloader(self, texts: list[str], labels: list[int], tokenizer=None, batch_size: int = 32, shuffle: bool = True):
        """Creates a PyTorch DataLoader."""
        dataset = NLPTextDataset(texts, labels, tokenizer=tokenizer, max_length=self.config["data"]["max_length"])
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    manager = DatasetManager()
    
    # Quick sanity test on a small subset of AG News
    texts, labels = manager.load_ag_news(split="train[:100]")
    
    logger.info(f"✅ Downloaded and processed {len(texts)} samples.")
    logger.info(f"✅ Sample 0 Text: {texts[0][:100]}...")
    logger.info(f"✅ Sample 0 Label: {labels[0]}")
    
    loader = manager.create_dataloader(texts, labels, batch_size=8)
    sample_batch = next(iter(loader))
    logger.info(f"✅ DataLoader Batch Keys: {list(sample_batch.keys())}")