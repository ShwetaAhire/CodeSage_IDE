import ast
import logging
from typing import Dict, Any, List
from ..ai import CodeEmbeddings, MLPredictor, PatternRecognizer
from ..core import CodeAnalyzer, BugDetector

logger = logging.getLogger(__name__)
 
class CodeParser:
    def __init__(self):
        self.code_embeddings = CodeEmbeddings()
        self.ml_predictor = MLPredictor()
        self.pattern_recognizer = PatternRecognizer()
        self.code_analyzer = CodeAnalyzer()
        self.bug_detector = BugDetector()

         # Cache for performance optimization
        self._last_analysis = {}
        self._analysis_cache = {}

    def parse_realtime(self, code: str, cursor_position: int = None) -> Dict[str, Any]:
        """Parse code in real-time with AI analysis"""
        try:
            cache_key = f"{code}:{cursor_position}"
            if cache_key in self._analysis_cache:
                return self._analysis_cache[cache_key]

            # Parse AST
            tree = ast.parse(code)
            
            # Generate embeddings using CodeBERT
            embeddings = self.code_embeddings.generate(code)
            
            # Get ML predictions using CodeT5 and other models
            context = {
                'cursor_position': cursor_position,
                'last_analysis': self._last_analysis,
                'embeddings': embeddings
            }
            
            # Comprehensive analysis from all models
            ml_analysis = self.ml_predictor.predict_realtime(code, context)
            pattern_analysis = self.pattern_recognizer.analyze_realtime(code, cursor_position)
            code_quality = self.code_analyzer.analyze_code(code)
            potential_bugs = self.bug_detector.detect_bugs_realtime(code, cursor_position)

            # Extract code structure and symbols
            structure = self._analyze_structure(tree)
            symbols = self._extract_symbols(tree)
            current_context = self._get_current_context(tree, cursor_position, embeddings)

            # Combine all analyses
            analysis_result = {
                'ast': tree,
                'symbols': symbols,
                'structure': structure,
                'analysis': {
                    'quality_score': code_quality.get('score', 0.0),
                    'complexity': ml_analysis.get('complexity_score', 0.0),
                    'suggestions': ml_analysis.get('suggestions', []),
                    'patterns': pattern_analysis.get('patterns', []),
                    'bugs': potential_bugs,
                    'security_issues': ml_analysis.get('security_issues', []),
                    'performance_insights': ml_analysis.get('performance_insights', []),
                    'code_completions': ml_analysis.get('completions', [])
                },
                'current_context': current_context
            }

            # Cache the results
            self._analysis_cache[cache_key] = analysis_result
            self._last_analysis = analysis_result

            return analysis_result

        except SyntaxError:
            return self._get_error_response('syntax_error')
        except Exception as e:
            logger.error(f"Parse error: {str(e)}")
            return self._get_error_response('general_error')

    def _get_context_suggestions(self, node: ast.AST, embeddings: Any) -> List[str]:
        """Get context-aware suggestions using ML models"""
        try:
            code_context = ast.unparse(node)
            
            # Get predictions from CodeT5
            context_analysis = self.ml_predictor.predict_realtime(
                code_context, 
                {
                    'embeddings': embeddings,
                    'model_type': 'code_completion',
                    'max_suggestions': 5
                }
            )
            
            # Get pattern-based suggestions
            pattern_suggestions = self.pattern_recognizer.get_suggestions(code_context)
            
            # Combine and rank suggestions
            all_suggestions = context_analysis.get('suggestions', []) + pattern_suggestions
            return self._rank_suggestions(all_suggestions, embeddings)
            
        except Exception as e:
            logger.error(f"Context suggestion error: {str(e)}")
            return []

    def _rank_suggestions(self, suggestions: List[str], embeddings: Any) -> List[str]:
        """Rank suggestions using ML models"""
        try:
            ranked_suggestions = self.ml_predictor.rank_suggestions(
                suggestions,
                embeddings,
                max_suggestions=5
            )
            return ranked_suggestions
        except Exception:
            return suggestions[:5]


    def _analyze_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code structure using AST and ML insights"""
        try:
            structure = {
                'imports': self._analyze_imports(tree),
                'classes': self._analyze_classes(tree),
                'functions': self._analyze_functions(tree),
                'complexity_metrics': self._get_complexity_metrics(tree)
            }
            return structure
        except Exception as e:
            logger.error(f"Structure analysis error: {str(e)}")
            return {}

    def _extract_methods(self, node: ast.ClassDef) -> List[Dict]:
        """Extract method information from class definition"""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append({
                    'name': item.name,
                    'lineno': item.lineno,
                    'complexity': self._calculate_complexity(item)
                })
        return methods

    def _extract_import_info(self, node: ast.AST) -> Dict:
        """Extract import information"""
        if isinstance(node, ast.Import):
            return {'type': 'import', 'names': [n.name for n in node.names]}
        elif isinstance(node, ast.ImportFrom):
            return {
                'type': 'from_import',
                'module': node.module,
                'names': [n.name for n in node.names]
            }
        return {}

    def _find_node_at_position(self, tree: ast.AST, position: int) -> Optional[ast.AST]:
        """Find AST node at given position"""
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                if node.lineno == position:
                    return node
        return None

    def _get_scope(self, node: ast.AST) -> Dict[str, Any]:
        """Get scope information for current node"""
        try:
            scope = {
                'variables': self._get_scope_variables(node),
                'parent_function': self._find_parent_function(node),
                'parent_class': self._find_parent_class(node)
            }
            return scope
        except Exception as e:
            logger.error(f"Scope analysis error: {str(e)}")
            return {}

    def _get_context_suggestions(self, node: ast.AST, embeddings: Any) -> List[str]:
        """Get context-aware suggestions using ML models"""
        try:
            code_context = ast.unparse(node)
            context_analysis = self.ml_predictor.predict_realtime(
                code_context, 
                {'embeddings': embeddings}
            )
            return context_analysis.get('suggestions', [])
        except Exception:
            return []

    def _extract_symbols(self, tree: ast.AST) -> Dict[str, List]:
        """Extract and analyze code symbols"""
        symbols = {
            'functions': [],
            'classes': [],
            'variables': [],
            'imports': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                symbols['functions'].append({
                    'name': node.name,
                    'lineno': node.lineno,
                    'complexity': self._calculate_complexity(node)
                })
            elif isinstance(node, ast.ClassDef):
                symbols['classes'].append({
                    'name': node.name,
                    'lineno': node.lineno,
                    'methods': self._extract_methods(node)
                })
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                symbols['variables'].append({
                    'name': node.id,
                    'lineno': node.lineno
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                symbols['imports'].append(self._extract_import_info(node))
                
        return symbols

    def _get_current_context(self, tree: ast.AST, cursor_position: int, embeddings: Any) -> Dict:
        """Get context at current cursor position"""
        if cursor_position is None:
            return {} 
            
        try:
            current_node = self._find_node_at_position(tree, cursor_position)
            if not current_node:
                return {}
                
            return {
                'node_type': type(current_node).__name__,
                'scope': self._get_scope(current_node),
                'suggestions': self._get_context_suggestions(current_node, embeddings),
                'related_symbols': self._find_related_symbols(current_node)
            }
            
        except Exception as e:
            logger.error(f"Context analysis error: {str(e)}")
            return {}

    def _calculate_complexity(self, node: ast.AST) -> float:
        """Calculate code complexity using ML insights"""
        try:
            code = ast.unparse(node)
            analysis = self.ml_predictor.predict_realtime(code)
            return analysis.get('complexity_score', 0.0)
        except Exception:
            return 0.0

    def _get_error_response(self, error_type: str) -> Dict:
        """Generate error response"""
        return {
            'ast': None,
            'symbols': {},
            'structure': {},
            'analysis': {
                'error': error_type,
                'quality_score': 0.0,
                'complexity': 0.0,
                'suggestions': []
            },
            'current_context': {}
        }