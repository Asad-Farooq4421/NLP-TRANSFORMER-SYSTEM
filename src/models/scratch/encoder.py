import sys
import math
from pathlib import Path
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.scratch.attention import MultiHeadAttention
from src.utils.logger import setup_logger

logger = setup_logger("encoder_module")

class PositionalEncoding(nn.Module):
    """
    Sine-Cosine Positional Encoding module.
    """
    def __init__(self, d_model: int = 256, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]

        # Register buffer so it is saved with model state but not treated as a trainable parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """
    Two-layer Position-wise Feed-Forward Network.
    """
    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer (MHA + FFN + Residual Norms).
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Sub-layer 1: Self-Attention + Residual & LayerNorm
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Sub-layer 2: Feed-Forward + Residual & LayerNorm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


class TransformerEncoder(nn.Module):
    """
    Full Stacked Transformer Encoder.
    """
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 4, 
                 n_heads: int = 8, d_ff: int = 1024, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # src shape: [batch_size, seq_len]
        x = self.embedding(src)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


if __name__ == "__main__":
    batch_size = 2
    seq_len = 16
    vocab_size = 30000

    # Initialize full encoder
    encoder = TransformerEncoder(vocab_size=vocab_size, d_model=256, n_layers=4, n_heads=8)
    
    # Dummy token ID batch
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output = encoder(dummy_input)
    
    logger.info("✅ Transformer Encoder forward pass success!")
    logger.info(f"✅ Token Input Shape: {dummy_input.shape}")
    logger.info(f"✅ Encoder Representation Shape: {output.shape}")