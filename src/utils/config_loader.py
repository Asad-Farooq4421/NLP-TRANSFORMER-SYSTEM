import os
import yaml
from pathlib import Path

def get_project_root() -> Path:
    """Returns the absolute path to the project root folder."""
    return Path(__file__).resolve().parent.parent.parent

def load_config(config_path: str = None) -> dict:
    """
    Loads YAML configuration file safely.
    
    Args:
        config_path (str, optional): Relative or absolute path to config file.
                                     Defaults to 'config/config.yaml'.
    Returns:
        dict: Parsed configuration dictionary.
    """
    root = get_project_root()
    
    if config_path is None:
        config_file = root / "config" / "config.yaml"
    else:
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = root / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")

    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config

if __name__ == "__main__":
    # Test loading the config
    cfg = load_config()
    print("✅ Configuration loaded successfully!")
    print(f"Device configured: {cfg['device']}")
    print(f"IMDB Dataset path/name: {cfg['data']['datasets']['imdb']}")