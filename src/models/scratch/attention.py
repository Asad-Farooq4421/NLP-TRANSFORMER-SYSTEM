import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("attention_module")

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Scaled Dot-Product Attention from scratch in PyTorch.
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Query, Key, Value
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output linear layer
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            query, key, value: Shape [batch_size, seq_len, d_model]
            mask: Optional tensor [batch_size, 1, seq_len, seq_len] for causal/padding masking
        Returns:
            output: Tensor [batch_size, seq_len, d_model]
            attention_weights: Tensor [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)

        # 1. Project Q, K, V and split into multiple heads
        # Shape: [batch_size, n_heads, seq_len, d_k]
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention: Scores = (Q @ K^T) / sqrt(d_k)
        # Shape: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. Apply mask (fill masked positions with -1e9 before softmax)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Softmax over last dimension to get attention probabilities
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. Multiply weights by Values: Output = Attn @ V
        # Shape: [batch_size, n_heads, seq_len, d_k]
        context = torch.matmul(attn_weights, V)

        # 6. Concatenate heads back together and pass through final linear layer
        # Shape: [batch_size, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)

        return output, attn_weights


if __name__ == "__main__":
    # Test parameters
    batch_size = 2
    seq_len = 10
    d_model = 256
    n_heads = 8

    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    
    # Dummy input sequence
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass without mask
    out, weights = mha(x, x, x)
    
    logger.info("✅ Multi-Head Attention forward pass success!")
    logger.info(f"✅ Input Shape: {x.shape}")
    logger.info(f"✅ Output Shape: {out.shape}")
    logger.info(f"✅ Attention Weights Shape: {weights.shape}")