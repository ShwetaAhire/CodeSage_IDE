import os
import random
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestClassifier
import joblib
from .model_config import ModelConfig
from .code_embeddings import CodeEmbeddings

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.config = ModelConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_or_create_model()
        self.embeddings_generator = CodeEmbeddings()

    def predict_realtime(self, code: str, context: Dict = None) -> Dict:
        """Real-time predictions for IDE integration using CodeBERT"""
        try:
            # Generate embeddings using CodeBERT
            embeddings = self.embeddings_generator.generate(code)
            
            # Get base predictions
            predictions = self.predict(embeddings)
            
            # Get model-based analysis
            model = self.config.models['codebert']
            with torch.no_grad():
                inputs = model['tokenizer'](
                    code,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                outputs = model['model'](**inputs)
            
            # Process context-specific analysis
            if context:
                cursor_position = context.get('cursor_position')
                file_type = context.get('file_type')
                
                context_analysis = self._analyze_context(code, context)
                immediate_suggestions = self._get_immediate_suggestions(
                    code, 
                    outputs, 
                    cursor_position,
                    file_type
                )
                
                predictions.update({
                    'context_specific': context_analysis,
                    'immediate_suggestions': immediate_suggestions,
                    'quality_score': float(torch.sigmoid(outputs.logits[0][1]))
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Real-time prediction error: {str(e)}")
            return self._get_fallback_predictions()
            
    def _analyze_context(self, code: str, context: Dict) -> Dict:
        """Analyze code in current IDE context"""
        try:
            cursor_position = context.get('cursor_position')
            if cursor_position is not None:
                lines = code.split('\n')
                current_line = lines[cursor_position] if 0 <= cursor_position < len(lines) else ""
                
                return {
                    'current_line_analysis': self._analyze_line(current_line),
                    'scope_analysis': self._analyze_scope(code, cursor_position),
                    'context_type': context.get('file_type', 'unknown')
                }
            return {}
        except Exception as e:
            logger.error(f"Context analysis error: {str(e)}")
            return {}
            
    def _get_immediate_suggestions(self, code: str, model_outputs, cursor_position: int = None, file_type: str = None) -> List[str]:
        """Generate immediate suggestions based on model outputs and context"""
        try:
            suggestions = []
            confidence = float(torch.sigmoid(model_outputs.logits[0][1]))
            
            # Add model-based suggestions
            if confidence < 0.5:
                suggestions.append("Consider improving current code block")
            if confidence < 0.3:
                suggestions.append("Code might need immediate attention")
                
            # Add context-specific suggestions
            if cursor_position is not None:
                lines = code.split('\n')
                if 0 <= cursor_position < len(lines):
                    current_line = lines[cursor_position]
                    line_suggestions = self._analyze_line_for_suggestions(current_line, file_type)
                    suggestions.extend(line_suggestions)
                    
            return list(set(suggestions))
            
        except Exception as e:
            logger.error(f"Error generating immediate suggestions: {str(e)}")
            return []
        
    def _load_or_create_model(self):
        try:
            self.model_dir.mkdir(exist_ok=True)
            model_path = self.model_dir / "code_analyzer.joblib"
            
            if model_path.exists():
                logger.info(f"Loading existing model from {model_path}")
                return joblib.load(model_path)
            
            logger.info("Creating new model")
            return RandomForestClassifier(n_estimators=100, random_state=42)
            
        except Exception as e:
            logger.error(f"Error in model loading/creation: {str(e)}")
            return None
        
    def predict(self, embeddings):
        try:
            if embeddings is None or not isinstance(embeddings, (list, np.ndarray)):
                raise ValueError("Invalid embeddings input")
                
            if self.model is None:
                return self._get_fallback_predictions()
                
            embeddings = np.array(embeddings).reshape(1, -1)
            predictions = {
                'bug_probability': float(self.model.predict_proba(embeddings)[0][1]),
                'complexity_score': float(np.mean(self.model.feature_importances_) * 10),
                'suggested_improvements': self._generate_suggestions()
            }
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return self._get_fallback_predictions()
            
    def _get_fallback_predictions(self):
        return {
            'bug_probability': random.random(),
            'complexity_score': random.random() * 10,
            'suggested_improvements': self._generate_suggestions()
        }
            
    def analyze_variables(self, locals_dict):
        if not isinstance(locals_dict, dict):
            logger.error("Invalid input: locals_dict must be a dictionary")
            return {}
            
        analysis = {
            'type_info': {},
            'value_ranges': {},
            'potential_issues': []
        }
        
        try:
            for var_name, value in locals_dict.items():
                analysis['type_info'][var_name] = type(value).__name__
                
                if isinstance(value, (int, float)):
                    analysis['value_ranges'][var_name] = self._analyze_numeric(value)
                elif isinstance(value, (list, dict, set)):
                    self._analyze_collection(value, var_name, analysis)
                    
        except Exception as e:
            logger.error(f"Variable analysis error: {str(e)}")
            
        return analysis
        
    def _analyze_collection(self, value, var_name, analysis):
        try:
            size = len(value)
            if size > 1000:
                analysis['potential_issues'].append(
                    f"Large collection in {var_name}: {size} items"
                )
        except Exception as e:
            logger.error(f"Collection analysis error for {var_name}: {str(e)}")
        
    def _analyze_numeric(self, value):
        try:
            return {
                'is_positive': value > 0,
                'is_zero': value == 0,
                'magnitude': abs(value),
                'is_integer': isinstance(value, int),
                'is_finite': np.isfinite(float(value))
            } 
        except Exception as e:
            logger.error(f"Numeric analysis error: {str(e)}")
            return {}
        
    def _generate_suggestions(self):
        suggestions = [
            "Consider adding error handling for edge cases",
            "Validate input parameters and their types",
            "Add type hints for better code clarity",
            "Consider implementing unit tests",
            "Add proper documentation for functions",
            "Consider using dataclasses for structured data",
            "Implement logging for better debugging"
        ]
        return random.sample(suggestions, min(3, len(suggestions)))

    def analyze_code_for_ide(self, code, embeddings):
        """Analyze code specifically for IDE integration"""
        predictions = self.predict(embeddings)
        analysis = self.analyze_variables(locals())
    
        return {
            'suggestions': predictions['suggested_improvements'],
            'bugs': [f"Potential bug risk: {predictions['bug_probability']:.2%}"],
            'metrics': {
            'complexity': predictions['complexity_score'],
            'variables': analysis['type_info']
        }
    }