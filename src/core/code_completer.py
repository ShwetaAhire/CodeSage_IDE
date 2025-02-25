import logging
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..ai import CodeEmbeddings, MLPredictor, PatternRecognizer, ModelConfig

logger = logging.getLogger(__name__)

class CodeCompleter:
    def __init__(self):
        # Initialize AI components
        self.code_embeddings = CodeEmbeddings()
        self.ml_predictor = MLPredictor()
        self.pattern_recognizer = PatternRecognizer()
        self.config = ModelConfig()
        
        # Load completion model
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/codebert-base").to(self.device)
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            self.model = None

    def get_completions(self, code: str, cursor_position: int, file_type: str = None) -> Dict[str, Any]:
        """Get real-time code completions with AI analysis"""
        try:
            # Get context around cursor
            context = self._get_cursor_context(code, cursor_position)
            
            # Generate embeddings for context
            context_embeddings = self.code_embeddings.generate(context['current_line'])
            
            # Get ML predictions
            ml_analysis = self.ml_predictor.predict_realtime(
                code,
                {'cursor_position': cursor_position, 'file_type': file_type}
            )
            
            # Generate completions
            completions = self._generate_completions(context, ml_analysis)
            
            return {
                'completions': completions,
                'context': context,
                'analysis': ml_analysis,
                'quality_scores': self._get_completion_scores(completions, context)
            }
            
        except Exception as e:
            logger.error(f"Completion generation error: {str(e)}")
            return {'completions': [], 'context': {}, 'analysis': {}, 'quality_scores': []}

    def _generate_completions(self, context: Dict, analysis: Dict) -> List[str]:
        """Generate intelligent code completions"""
        try:
            if not self.model:
                return []

            input_text = context['prefix'][-100:]  # Last 100 chars for context
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_length=50,
                    num_return_sequences=5,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            completions = []
            for output in outputs:
                completion = self.tokenizer.decode(output[len(inputs['input_ids'][0]):], skip_special_tokens=True)
                if completion.strip():
                    completions.append(completion)
            
            return completions[:5]  # Return top 5 completions
            
        except Exception as e:
            logger.error(f"Completion generation error: {str(e)}")
            return []

    def _get_cursor_context(self, code: str, cursor_position: int) -> Dict[str, str]:
        """Extract context around cursor position"""
        try:
            lines = code.split('\n')
            current_line_no = code[:cursor_position].count('\n')
            current_line = lines[current_line_no] if current_line_no < len(lines) else ""
            
            # Get context before and after cursor
            prefix = code[:cursor_position]
            suffix = code[cursor_position:]
            
            return {
                'prefix': prefix,
                'suffix': suffix,
                'current_line': current_line,
                'line_number': current_line_no,
                'line_prefix': current_line[:cursor_position - len(code[:code.rfind('\n', 0, cursor_position)]) - 1]
            }
        except Exception as e:
            logger.error(f"Context extraction error: {str(e)}")
            return {'prefix': '', 'suffix': '', 'current_line': '', 'line_number': 0, 'line_prefix': ''}

    def _get_completion_scores(self, completions: List[str], context: Dict) -> List[float]:
        """Score completions using ML model"""
        try:
            scores = []
            for completion in completions:
                # Combine context and completion
                full_code = context['prefix'] + completion + context['suffix']
                
                # Get quality score from ML predictor
                analysis = self.ml_predictor.predict_realtime(full_code)
                scores.append(analysis.get('quality_score', 0.0))
                
            return scores
            
        except Exception as e:
            logger.error(f"Completion scoring error: {str(e)}")
            return [0.0] * len(completions)

    def get_realtime_suggestions(self, code: str, cursor_position: int) -> List[str]:
        """Get real-time coding suggestions"""
        try:
            context = self._get_cursor_context(code, cursor_position)
            pattern_analysis = self.pattern_recognizer.analyze_realtime(code, cursor_position)
            
            suggestions = []
            suggestions.extend(pattern_analysis.get('suggestions', []))
            
            # Add ML-based suggestions
            ml_analysis = self.ml_predictor.predict_realtime(code, {'cursor_position': cursor_position})
            suggestions.extend(ml_analysis.get('immediate_suggestions', []))
            
            return list(set(suggestions))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Suggestion generation error: {str(e)}")
            return []