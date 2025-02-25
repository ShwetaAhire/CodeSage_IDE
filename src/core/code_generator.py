import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
from transformers import pipeline
from ..ai import CodeEmbeddings, MLPredictor, PatternRecognizer, ModelConfig

logger = logging.getLogger(__name__)

class CodeGenerator:
    def __init__(self):
        # Initialize AI components
        self.code_embeddings = CodeEmbeddings()
        self.ml_predictor = MLPredictor()
        self.pattern_recognizer = PatternRecognizer()
        self.config = ModelConfig()
        
        # Set up code generation model
        try:
            models_dir = Path("./models")
            models_dir.mkdir(exist_ok=True)
            
            self.model = pipeline(
                "text2text-generation",
                model="Salesforce/codet5-base",
                device=0 if torch.cuda.is_available() else -1,
                cache_dir=str(models_dir)
            )
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            self.model = None

    def generate_code(self, prompt: str, context: Dict = None) -> Dict[str, Any]:
        """Generate code based on prompt with real-time analysis"""
        try:
            if not prompt.strip():
                return {"code": "", "analysis": {}}

            # Generate initial code
            generated_code = self._generate_with_model(prompt)
            
            # Analyze generated code
            embeddings = self.code_embeddings.generate(generated_code)
            code_analysis = self.ml_predictor.predict_realtime(generated_code, context)
            
            # Improve code based on analysis
            if code_analysis.get('quality_score', 0.0) < 0.7:
                generated_code = self._improve_code(generated_code, code_analysis)
            
            return {
                "code": generated_code,
                "analysis": code_analysis,
                "suggestions": self._get_code_suggestions(generated_code, code_analysis)
            }
            
        except Exception as e:
            logger.error(f"Code generation error: {str(e)}")
            return {"code": "", "analysis": {}, "suggestions": ["Generation failed"]}

    def generate_completion(self, code: str, cursor_position: int) -> Dict[str, Any]:
        """Real-time code completion with AI analysis"""
        try:
            # Get context around cursor
            context = self._get_cursor_context(code, cursor_position)
            
            # Generate completions
            completions = self._generate_with_model(
                context['prefix'],
                max_length=50,
                num_return_sequences=3
            )
            
            # Analyze completions
            analyzed_completions = []
            for completion in completions:
                full_code = context['prefix'] + completion + context['suffix']
                analysis = self.ml_predictor.predict_realtime(full_code, {'cursor_position': cursor_position})
                
                analyzed_completions.append({
                    'completion': completion,
                    'quality_score': analysis.get('quality_score', 0.0),
                    'suggestions': analysis.get('immediate_suggestions', [])
                })
            
            return {
                'completions': analyzed_completions,
                'context': context
            }
            
        except Exception as e:
            logger.error(f"Completion generation error: {str(e)}")
            return {'completions': [], 'context': {}}

    def _generate_with_model(self, prompt: str, **kwargs) -> str:
        """Generate code using the underlying model"""
        try:
            if not self.model:
                return ""
                
            response = self.model(prompt, **kwargs)
            return response[0]['generated_text'].strip() if response else ""
            
        except Exception as e:
            logger.error(f"Model generation error: {str(e)}")
            return ""

    def _improve_code(self, code: str, analysis: Dict) -> str:
        """Improve generated code based on analysis"""
        try:
            issues = analysis.get('issues', [])
            if not issues:
                return code
                
            # Create improvement prompt
            improvement_prompt = f"""Improve this code by fixing these issues:
            {' '.join(issues)}
            
            Code:
            {code}
            """
            
            improved_code = self._generate_with_model(improvement_prompt)
            return improved_code if improved_code else code
            
        except Exception as e:
            logger.error(f"Code improvement error: {str(e)}")
            return code

    def _get_cursor_context(self, code: str, cursor_position: int) -> Dict[str, str]:
        """Get code context around cursor"""
        try:
            prefix = code[:cursor_position]
            suffix = code[cursor_position:]
            
            return {
                'prefix': prefix,
                'suffix': suffix,
                'current_line': code.split('\n')[code[:cursor_position].count('\n')]
            }
        except Exception as e:
            logger.error(f"Context extraction error: {str(e)}")
            return {'prefix': '', 'suffix': '', 'current_line': ''}

    def _get_code_suggestions(self, code: str, analysis: Dict) -> List[str]:
        """Get suggestions for generated code"""
        try:
            suggestions = []
            quality_score = analysis.get('quality_score', 0.0)
            
            if quality_score < 0.7:
                suggestions.append("Consider improving code quality")
            if analysis.get('complexity_score', 0.0) > 0.7:
                suggestions.append("Code might be too complex")
            
            pattern_analysis = self.pattern_recognizer.analyze_code(code)
            suggestions.extend(pattern_analysis.get('suggestions', []))
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Suggestion generation error: {str(e)}")
            return []