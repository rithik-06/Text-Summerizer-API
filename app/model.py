import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationModel:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the summarization model.
        Using BART-large-CNN for better quality summaries.
        Alternative: "t5-small" for faster inference on CPU
        """
        logger.info(f"Loading model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def summarize(
        self,
        text: str,
        max_length: int = 130,
        min_length: int = 30,
        do_sample: bool = False
    ) -> str:
        """
        Generate summary for the input text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            do_sample: Whether to use sampling (False = greedy decoding)
        
        Returns:
            Generated summary
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                do_sample=do_sample
            )
        
        # Decode and return
        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return summary
    
    def get_summary_lengths(self, summary_type: str) -> Dict[str, int]:
        """Map summary type to length parameters."""
        length_map = {
            "short": {"max_length": 80, "min_length": 20},
            "medium": {"max_length": 130, "min_length": 30},
            "detailed": {"max_length": 200, "min_length": 50}
        }
        return length_map.get(summary_type, length_map["medium"])


# Global model instance (loaded once at startup)
_model_instance = None


def get_model() -> SummarizationModel:
    """Dependency to get the model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = SummarizationModel()
    return _model_instance