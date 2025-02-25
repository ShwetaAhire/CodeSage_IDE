import torch
from pathlib import Path
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

class ModelConfig:
    def __init__(self):
        self.models_dir = Path("models")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.models = self.initialize_models()
         
    def initialize_models(self):
        try:
            model_name = "microsoft/codebert-base"
            return {
                'codebert': {
                    'tokenizer': AutoTokenizer.from_pretrained(model_name),
                    'model': AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=2,
                        ignore_mismatched_sizes=True
                    ).to(self.device)
                }
            }
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            return None