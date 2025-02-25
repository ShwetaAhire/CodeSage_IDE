import logging
import threading
import time
import torch
from typing import Dict, List, Any, Callable
from ..ai import CodeEmbeddings, MLPredictor, PatternRecognizer, ModelConfig

logger = logging.getLogger(__name__)

class IDEAnalyzer:
    def __init__(self):
        self.code_embeddings = CodeEmbeddings()
        self.ml_predictor = MLPredictor()
        self.pattern_recognizer = PatternRecognizer()
        self.config = ModelConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._last_code = ""
        self.last_analysis = {}
        self.analysis_delay = 0.5  # seconds
        self._analysis_thread = None

    def analyze_realtime(self, code: str, cursor_position: int = None, file_type: str = None) -> Dict[str, Any]:
        """Real-time analysis for all IDE components"""
        if code == self._last_code and cursor_position is None:
            return self.last_analysis

        try:
            # Generate embeddings and context
            embeddings = self.code_embeddings.generate(code)
            context = {
                'cursor_position': cursor_position,
                'file_type': file_type
            }
            
            # Get various analyses
            ml_analysis = self.ml_predictor.predict_realtime(code, context)
            pattern_analysis = self.pattern_recognizer.analyze_realtime(code, cursor_position)
            
            # Comprehensive analysis results
            results = {
                # Editor panel
                'editor_analysis': {
                    'code_quality': ml_analysis.get('quality_score', 0.0),
                    'immediate_suggestions': ml_analysis.get('immediate_suggestions', []),
                    'current_context': self._analyze_current_context(code, cursor_position, embeddings)
                },
                
                # Suggestions panel
                'suggestions': {
                    'code_improvements': pattern_analysis.get('suggestions', []),
                    'best_practices': ml_analysis.get('suggested_improvements', []),
                    'refactoring': self._get_refactoring_suggestions(code, ml_analysis)
                },
                
                # Debug panel
                'debug_info': {
                    'potential_bugs': ml_analysis.get('bug_probability', 0.0),
                    'complexity_warnings': pattern_analysis.get('issues', []),
                    'runtime_considerations': self._analyze_runtime_aspects(code)
                },
                
                # Metrics
                'metrics': self._get_code_metrics(code, ml_analysis)
            }
            
            self._last_code = code
            self.last_analysis = results
            return results
            
        except Exception as e:
            logger.error(f"Real-time analysis error: {str(e)}")
            return self._get_fallback_analysis()

    def start_background_analysis(self, get_code_callback: Callable, update_ui_callback: Callable):
        """Start background analysis thread"""
        def analysis_loop():
            while self._analysis_thread:
                try:
                    current_code = get_code_callback()
                    cursor_position = getattr(get_code_callback, 'cursor_position', None)
                    file_type = getattr(get_code_callback, 'file_type', None)
                    
                    results = self.analyze_realtime(current_code, cursor_position, file_type)
                    update_ui_callback(results)
                    
                except Exception as e:
                    logger.error(f"Background analysis error: {str(e)}")
                finally:
                    time.sleep(self.analysis_delay)

        self._analysis_thread = threading.Thread(
            target=analysis_loop,
            daemon=True
        )
        self._analysis_thread.start()

    def stop_background_analysis(self):
        """Stop background analysis thread"""
        self._analysis_thread = None

    def _analyze_current_context(self, code: str, cursor_position: int, embeddings: Any) -> Dict:
        """Analyze current editing context"""
        try:
            if cursor_position is None:
                return {}

            lines = code.split('\n')
            current_line = lines[cursor_position] if 0 <= cursor_position < len(lines) else ""
            
            return {
                'line_analysis': self.pattern_recognizer._analyze_single_line(current_line, cursor_position),
                'context_suggestions': self._get_context_suggestions(current_line, embeddings),
                'scope': self._get_current_scope(code, cursor_position)
            }
        except Exception as e:
            logger.error(f"Context analysis error: {str(e)}")
            return {}

    def _get_refactoring_suggestions(self, code: str, ml_analysis: Dict) -> List[str]:
        """Get ML-based refactoring suggestions"""
        try:
            quality_score = ml_analysis.get('quality_score', 0.0)
            suggestions = []
            
            if quality_score < 0.7:
                pattern_issues = self.pattern_recognizer.analyze_code(code)
                suggestions.extend(pattern_issues.get('suggestions', []))
                
            return suggestions
        except Exception as e:
            logger.error(f"Refactoring suggestion error: {str(e)}")
            return []

    def _analyze_runtime_aspects(self, code: str) -> Dict:
        """Analyze potential runtime issues"""
        try:
            return {
                'complexity': self.pattern_recognizer._calculate_complexity(code),
                'memory_usage': self._estimate_memory_usage(code),
                'performance_hints': self._get_performance_hints(code)
            }
        except Exception as e:
            logger.error(f"Runtime analysis error: {str(e)}")
            return {}

    def _get_fallback_analysis(self) -> Dict:
        """Provide fallback analysis when ML analysis fails"""
        return {
            'editor_analysis': {},
            'suggestions': {'code_improvements': [], 'best_practices': [], 'refactoring': []},
            'debug_info': {'potential_bugs': 0.0, 'complexity_warnings': [], 'runtime_considerations': {}},
            'metrics': {}
        }