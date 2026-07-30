import sys
from pathlib import Path
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.scratch.attention import MultiHeadAttention
from src.models.scratch.encoder import PositionalEncoding, PositionwiseFeedForward
from src.utils.logger import setup_logger

logger = setup_logger("decoder_module")

class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer (Masked Self-Attn + Cross-Attn + FFN + Residual Norms).
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            tgt: Target sequence embeddings [batch_size, tgt_seq_len, d_model]
            memory: Encoder outputs [batch_size, src_seq_len, d_model]
            tgt_mask: Causal mask for target sequence
            memory_mask: Mask for source/encoder sequence
        """
        # 1. Masked Causal Self-Attention
        attn1, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn1))

        # 2. Cross-Attention (Query from target, Keys & Values from encoder memory)
        if memory is not None:
            attn2, _ = self.cross_attn(tgt, memory, memory, memory_mask)
            tgt = self.norm2(tgt + self.dropout(attn2))

        # 3. Position-wise Feed-Forward
        ff_out = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_out))

        return tgt


class TransformerDecoder(nn.Module):
    """
    Full Stacked Transformer Decoder.
    """
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 4, 
                 n_heads: int = 8, d_ff: int = 1024, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generates lower-triangular causal mask to prevent attending to future tokens.
        Shape: [1, 1, seq_len, seq_len]
        """
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor = None, 
                tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None) -> torch.Tensor:
        
        # Auto-generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_causal_mask(tgt.size(1), tgt.device)

        x = self.embedding(tgt)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)

        return self.norm(x)


if __name__ == "__main__":
    batch_size = 2
    tgt_seq_len = 12
    src_seq_len = 16
    vocab_size = 30000

    decoder = TransformerDecoder(vocab_size=vocab_size, d_model=256, n_layers=4)
    
    dummy_tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
    dummy_memory = torch.randn(batch_size, src_seq_len, 256)
    
    output = decoder(dummy_tgt, dummy_memory)
    
    logger.info("✅ Transformer Decoder forward pass success!")
    logger.info(f"✅ Target Token Input Shape: {dummy_tgt.shape}")
    logger.info(f"✅ Decoder Representation Shape: {output.shape}")