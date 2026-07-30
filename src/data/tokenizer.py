import sys
import os
from pathlib import Path
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("custom_tokenizer")

class CustomBPETokenizer:
    """
    Custom Byte-Pair Encoding (BPE) & WordPiece Tokenizer implementation.
    """
    def __init__(self, vocab_size: int = 30000):
        self.config = load_config()
        self.vocab_size = vocab_size
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.tokenizer = None
        self.fast_tokenizer = None

    def build_and_train(self, texts: list[str], save_path: str = None):
        """
        Trains a BPE tokenizer from raw text iterator.
        """
        logger.info(f"Training Custom BPE Tokenizer with target vocab size {self.vocab_size}...")
        
        # Initialize BPE model with UNK token
        raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        
        # Pre-tokenization: split by whitespace and punctuation
        raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        raw_tokenizer.decoder = decoders.ByteLevel()
        
        # Trainer configuration
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2
        )
        
        # Train on text batch
        raw_tokenizer.train_from_iterator(texts, trainer=trainer)
        
        # Wrap into HuggingFace PreTrainedTokenizerFast for PyTorch compatibility
        self.fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw_tokenizer,
            pad_token="[PAD]",
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]"
        )
        
        logger.info("✅ BPE Tokenizer training complete!")

        if save_path:
            self.save(save_path)

    def save(self, save_directory: str):
        """Saves trained tokenizer to directory."""
        os.makedirs(save_directory, exist_ok=True)
        self.fast_tokenizer.save_pretrained(save_directory)
        logger.info(f"✅ Tokenizer saved to: {save_directory}")

    def load(self, load_directory: str):
        """Loads trained tokenizer from directory."""
        self.fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(load_directory)
        logger.info(f"✅ Tokenizer loaded successfully from: {load_directory}")
        return self.fast_tokenizer


if __name__ == "__main__":
    from src.data.dataset import DatasetManager
    
    # Quick sanity check training on small subset
    manager = DatasetManager()
    texts, _ = manager.load_ag_news("train[:500]")
    
    bpe_builder = CustomBPETokenizer(vocab_size=5000)
    save_dir = str(PROJECT_ROOT / "saved_models" / "custom_bpe_tokenizer")
    
    bpe_builder.build_and_train(texts, save_path=save_dir)
    
    # Test encoding
    sample_sentence = "Transformer models are revolutionary for NLP applications!"
    tokens = bpe_builder.fast_tokenizer.encode(sample_sentence)
    decoded = bpe_builder.fast_tokenizer.decode(tokens)
    
    logger.info(f"✅ Sample Text: {sample_sentence}")
    logger.info(f"✅ Token IDs: {tokens}")
    logger.info(f"✅ Decoded Text: {decoded}")