import sys
import re
import unicodedata
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("text_preprocessor")

class TextPreprocessor:
    """
    Standardized text cleaning pipeline for modern NLP tasks.
    """
    def __init__(self, lower_case: bool = True, remove_html: bool = True):
        self.lower_case = lower_case
        self.remove_html = remove_html
        
        # HTML tag regex pattern
        self.html_pattern = re.compile(r'<[^>]+>')
        # Multiple whitespace pattern
        self.whitespace_pattern = re.compile(r'\s+')

    def remove_html_tags(self, text: str) -> str:
        """Removes HTML markup (common in movie reviews like IMDB)."""
        return self.html_pattern.sub(' ', text)

    def normalize_unicode(self, text: str) -> str:
        """Normalizes unicode characters (e.g., non-breaking spaces)."""
        return unicodedata.normalize('NFKD', text)

    def clean_text(self, text: str) -> str:
        """
        Runs full cleaning pipeline on a raw text string.
        """
        if not isinstance(text, str):
            return ""

        # Normalize unicode
        text = self.normalize_unicode(text)

        # Remove HTML tags if enabled
        if self.remove_html:
            text = self.remove_html_tags(text)

        # Lowercase if enabled
        if self.lower_case:
            text = text.lower()

        # Collapse whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()

        return text

    def batch_clean(self, texts: list[str]) -> list[str]:
        """Cleans a list of text strings."""
        return [self.clean_text(t) for t in texts]


if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    dirty_sample = "  <br />This movie was <b>INCREDIBLE</b>!!!   The acting was top-tier.  "
    cleaned_sample = preprocessor.clean_text(dirty_sample)
    
    logger.info("✅ Raw Input: " + dirty_sample)
    logger.info("✅ Cleaned Output: " + cleaned_sample)