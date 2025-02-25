import sys
import bdb
import logging
import threading
from typing import Dict, Any, Set
from ..ai import CodeEmbeddings, MLPredictor, PatternRecognizer

logger = logging.getLogger(__name__)

class IDEDebugger(bdb.Bdb):
    def __init__(self):
        super().__init__() 
        # Initialize AI components
        self.code_embeddings = CodeEmbeddings()
        self.ml_predictor = MLPredictor()
        self.pattern_recognizer = PatternRecognizer()
        
        self.current_frame = None
        self.is_running = False
        self.debug_thread = None 
        self.breakpoints = set()
        self._current_file = None
        self._last_analysis = {}
        
    def analyze_debug_context(self, code: str, line_no: int = None) -> Dict[str, Any]:
        """Analyze current debug context using AI models"""
        try:
            # Generate embeddings for current code
            embeddings = self.code_embeddings.generate(code)
            
            # Get real-time predictions
            context = {'cursor_position': line_no} if line_no else None
            predictions = self.ml_predictor.predict_realtime(code, context)
            
            # Analyze current execution point
            if self.current_frame:
                current_line = code.split('\n')[self.current_frame.f_lineno - 1]
                line_analysis = self.pattern_recognizer._analyze_single_line(current_line, self.current_frame.f_lineno - 1)
                
                return {
                    'current_state': self.get_current_state(),
                    'code_quality': predictions.get('quality_score', 0.0),
                    'potential_issues': line_analysis.get('line_issues', []),
                    'runtime_suggestions': self._get_runtime_suggestions(predictions),
                    'variable_analysis': self._analyze_variables(self.current_frame.f_locals)
                }
            
            return self._last_analysis
            
        except Exception as e:
            logger.error(f"Debug analysis error: {str(e)}")
            return {}

    def _get_runtime_suggestions(self, predictions: Dict) -> List[str]:
        """Get runtime-specific suggestions"""
        suggestions = []
        if predictions.get('bug_probability', 0) > 0.3:
            suggestions.append("Consider adding error handling here")
        if predictions.get('complexity_score', 0) > 0.7:
            suggestions.append("This section might benefit from optimization")
        return suggestions

    def _analyze_variables(self, locals_dict: Dict) -> Dict[str, Any]:
        """Analyze variables in current scope using ML models"""
        try:
            analysis = {
                'type_info': {},
                'value_patterns': {},
                'potential_issues': []
            }
            
            for var_name, value in locals_dict.items():
                # Basic type analysis
                analysis['type_info'][var_name] = type(value).__name__
                
                # Value pattern analysis
                if isinstance(value, (int, float)):
                    analysis['value_patterns'][var_name] = self._analyze_numeric_pattern(value)
                elif isinstance(value, (list, dict, set)):
                    self._analyze_collection_pattern(value, var_name, analysis)
                    
            return analysis
            
        except Exception as e:
            logger.error(f"Variable analysis error: {str(e)}")
            return {}

    def start_debugging(self, filename: str):
        """Start debugging with AI integration"""
        if self.is_running:
            return
            
        self._current_file = filename
        self.is_running = True
        
        try:
            with open(filename, 'r') as f:
                code = f.read()
                # Pre-analyze code before debugging
                self._last_analysis = self.analyze_debug_context(code)
        except Exception as e:
            logger.error(f"Debug startup error: {str(e)}")
            
        self.debug_thread = threading.Thread(target=self._run_debugger, args=(filename,))
        self.debug_thread.start()

    def user_line(self, frame):
        """Called when we hit a line of code - now with AI analysis"""
        self.current_frame = frame
        try:
            with open(frame.f_code.co_filename, 'r') as f:
                code = f.read()
                self._last_analysis = self.analyze_debug_context(code, frame.f_lineno)
        except Exception as e:
            logger.error(f"Line analysis error: {str(e)}")