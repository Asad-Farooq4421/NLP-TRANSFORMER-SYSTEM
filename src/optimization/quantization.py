import sys
import os
import time
from pathlib import Path
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.models.pretrained.bert_classifier import PretrainedTransformerClassifier

logger = setup_logger("quantization_module")


class ModelQuantizer:
    """
    Applies dynamic INT8 quantization to PyTorch Transformer models 
    to reduce size and boost CPU inference speed.
    """
    def __init__(self, model_wrapper: PretrainedTransformerClassifier):
        self.wrapper = model_wrapper
        self.original_model = model_wrapper.model

    def apply_dynamic_quantization(self) -> torch.nn.Module:
        """
        Dynamically quantizes linear (Dense) layers from FP32 to INT8.
        """
        logger.info("Applying Dynamic INT8 Quantization to model linear layers...")
        
        quantized_model = torch.ao.quantization.quantize_dynamic(
            self.original_model,
            {torch.nn.Linear},  # Quantize all linear layers
            dtype=torch.qint8
        )
        
        logger.info("✅ Dynamic quantization successfully applied!")
        return quantized_model

    def compare_model_sizes(self, quantized_model: torch.nn.Module) -> dict:
        """
        Compares disk/memory size of FP32 vs INT8 quantized model.
        """
        # Save temporary checkpoints to measure file sizes
        tmp_dir = PROJECT_ROOT / "saved_models" / "tmp_quant"
        os.makedirs(tmp_dir, exist_ok=True)

        fp32_path = tmp_dir / "fp32_model.pt"
        int8_path = tmp_dir / "int8_model.pt"

        torch.save(self.original_model.state_dict(), fp32_path)
        torch.save(quantized_model.state_dict(), int8_path)

        fp32_size_mb = round(os.path.getsize(fp32_path) / (1024 * 1024), 2)
        int8_size_mb = round(os.path.getsize(int8_path) / (1024 * 1024), 2)
        reduction_pct = round((1 - (int8_size_mb / fp32_size_mb)) * 100, 2)

        # Cleanup temporary files
        os.remove(fp32_path)
        os.remove(int8_path)
        if os.path.exists(tmp_dir) and not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)

        logger.info(f"✅ Original FP32 Size: {fp32_size_mb} MB")
        logger.info(f"✅ Quantized INT8 Size: {int8_size_mb} MB")
        logger.info(f"✅ Memory Reduction: {reduction_pct}%")

        return {
            "fp32_size_mb": fp32_size_mb,
            "int8_size_mb": int8_size_mb,
            "reduction_percentage": reduction_pct
        }


if __name__ == "__main__":
    saved_checkpoint = PROJECT_ROOT / "saved_models" / "fine_tuned_ag_news"
    
    if saved_checkpoint.exists():
        logger.info("Loading fine-tuned model for quantization test...")
        wrapper = PretrainedTransformerClassifier(model_name=str(saved_checkpoint), num_classes=4)
    else:
        logger.info("Using DistilBERT backbone for quantization test...")
        wrapper = PretrainedTransformerClassifier(model_name="distilbert-base-uncased", num_classes=4)

    quantizer = ModelQuantizer(wrapper)
    quantized_model = quantizer.apply_dynamic_quantization()
    size_comparison = quantizer.compare_model_sizes(quantized_model)