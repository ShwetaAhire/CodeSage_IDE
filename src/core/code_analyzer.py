import ast
import logging
from typing import Dict, List, Any
from ..ai import CodeEmbeddings, MLPredictor, PatternRecognizer, ModelConfig

logger = logging.getLogger(__name__)

class CodeAnalyzer:
    def __init__(self):
        # Initialize AI components
        self.code_embeddings = CodeEmbeddings()
        self.ml_predictor = MLPredictor()
        self.pattern_recognizer = PatternRecognizer()
        self.config = ModelConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def analyze_code(self, code: str, cursor_position: int = None) -> Dict[str, Any]:
        """Comprehensive code analysis using AI models"""
        if not code.strip():
            return {"suggestions": ["No code to analyze"]}

        try:
            # Generate embeddings
            embeddings = self.code_embeddings.generate(code)
            
            # Get ML-based analysis
            context = {'cursor_position': cursor_position} if cursor_position else None
            ml_analysis = self.ml_predictor.predict_realtime(code, context)
            
            # Get pattern analysis
            pattern_analysis = self.pattern_recognizer.analyze_realtime(code, cursor_position)
            
            # Static analysis
            tree = ast.parse(code)
            static_analysis = self._perform_static_analysis(tree)
            
            return {
                'quality_metrics': {
                    'code_quality': ml_analysis.get('quality_score', 0.0),
                    'complexity': self._calculate_complexity(tree),
                    'maintainability': self._assess_maintainability(code, ml_analysis)
                },
                'suggestions': self._combine_suggestions(
                    static_analysis['suggestions'],
                    ml_analysis.get('immediate_suggestions', []),
                    pattern_analysis.get('suggestions', [])
                ),
                'issues': pattern_analysis.get('issues', []),
                'ml_insights': ml_analysis,
                'current_context': self._analyze_current_context(code, cursor_position, embeddings)
            }
            
        except SyntaxError:
            return {"suggestions": ["Syntax error in code"], "quality_metrics": {}}
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return {"suggestions": [f"Analysis error: {str(e)}"], "quality_metrics": {}}

    def _perform_static_analysis(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Enhanced static analysis"""
        suggestions = []
        
        self._check_complexity(tree, suggestions)
        self._check_naming(tree, suggestions)
        self._check_structure(tree, suggestions)
        
        return {'suggestions': suggestions}

    def _analyze_current_context(self, code: str, cursor_position: int, embeddings: Any) -> Dict:
        """Analyze code at current cursor position"""
        try:
            if cursor_position is None:
                return {}

            lines = code.split('\n')
            current_line = lines[cursor_position] if 0 <= cursor_position < len(lines) else ""
            
            # Get line-specific analysis
            line_analysis = self.pattern_recognizer._analyze_single_line(current_line, cursor_position)
            
            return {
                'current_line': current_line,
                'line_analysis': line_analysis,
                'context_suggestions': self._get_context_suggestions(current_line, embeddings)
            }
        except Exception as e:
            logger.error(f"Context analysis error: {str(e)}")
            return {}

    def _assess_maintainability(self, code: str, ml_analysis: Dict) -> float:
        """Calculate maintainability score using ML insights"""
        try:
            quality_score = ml_analysis.get('quality_score', 0.5)
            complexity = ml_analysis.get('complexity_score', 0.0)
            pattern_issues = len(self.pattern_recognizer.analyze_code(code).get('issues', []))
            
            # Weighted calculation
            maintainability = (
                quality_score * 0.4 +
                (1 - min(complexity, 1.0)) * 0.3 +
                (1 - min(pattern_issues / 10, 1.0)) * 0.3
            )
            
            return round(maintainability, 2)
            
        except Exception as e:
            logger.error(f"Maintainability assessment error: {str(e)}")
            return 0.5

    def _combine_suggestions(self, *suggestion_lists: List[str]) -> List[str]:
        """Combine and deduplicate suggestions"""
        combined = []
        for suggestions in suggestion_lists:
            combined.extend(suggestions)
        return list(set(combined))

    def _get_context_suggestions(self, line: str, embeddings: Any) -> List[str]:
        """Get context-aware suggestions using ML models"""
        try:
            predictions = self.ml_predictor.predict(embeddings)
            return predictions.get('suggested_improvements', [])
        except Exception as e:
            logger.error(f"Context suggestion error: {str(e)}")
            return []