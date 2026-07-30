import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so 'src' imports work from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get_project_root, load_config

def setup_logger(name: str = "nlp_system") -> logging.Logger:
    """
    Sets up a logger that outputs to both console and a log file.
    
    Args:
        name (str): Name of the logger instance.
        
    Returns:
        logging.Logger: Configured logger.
    """
    config = load_config()
    root_dir = get_project_root()
    
    # Create logs directory if it doesn't exist
    logs_dir = root_dir / config["paths"]["logs_dir"]
    os.makedirs(logs_dir, exist_ok=True)
    
    # Log filename format: app_YYYY-MM-DD.log
    log_file = logs_dir / f"nlp_system_{datetime.now().strftime('%Y-%m-%d')}.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if logger is instantiated multiple times
    if logger.hasHandlers():
        return logger
        
    # Formatting rules
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File Handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    return logger

if __name__ == "__main__":
    test_logger = setup_logger("test_module")
    test_logger.info("✅ Logger setup successfully!")
    test_logger.warning("This is a test warning log.")