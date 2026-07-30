import sys
from pathlib import Path
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.scratch.encoder import TransformerEncoder
from src.models.scratch.decoder import TransformerDecoder
from src.utils.logger import setup_logger

logger = setup_logger("transformer_assembly")


class FullTransformer(nn.Module):
    """
    Complete Encoder-Decoder Transformer Architecture from Scratch.
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_layers=n_decoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        # Final projection layer to target vocabulary size
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            src: Source token tensor [batch_size, src_len]
            tgt: Target token tensor [batch_size, tgt_len]
        Returns:
            Logits over target vocabulary [batch_size, tgt_len, tgt_vocab_size]
        """
        memory = self.encoder(src, mask=src_mask)
        out = self.decoder(tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        logits = self.fc_out(out)
        return logits


class ScratchTransformerClassifier(nn.Module):
    """
    Encoder-Only Transformer Classifier built from scratch (BERT-style).
    """
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 2,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Token ID tensor [batch_size, seq_len]
        Returns:
            Class logits [batch_size, num_classes]
        """
        encoder_out = self.encoder(x, mask=mask)
        # Global mean pooling over sequence length
        pooled = torch.mean(encoder_out, dim=1)
        logits = self.classifier(pooled)
        return logits


if __name__ == "__main__":
    batch_size = 2
    src_len = 16
    tgt_len = 12
    vocab_size = 30000
    num_classes = 4  # e.g., AG News 4 classes

    # 1. Test Full Seq2Seq Transformer
    full_model = FullTransformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=256,
        n_encoder_layers=2,
        n_decoder_layers=2
    )
    src_tokens = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, tgt_len))
    
    seq2seq_out = full_model(src_tokens, tgt_tokens)
    
    logger.info("✅ Full Transformer (Seq2Seq) assembly test success!")
    logger.info(f"✅ Seq2Seq Output Shape: {seq2seq_out.shape}")

    # 2. Test Encoder-Only Classifier
    classifier_model = ScratchTransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=256,
        n_layers=2
    )
    classifier_out = classifier_model(src_tokens)

    logger.info("✅ Transformer Classifier test success!")
    logger.info(f"✅ Classifier Output Logits Shape: {classifier_out.shape}")