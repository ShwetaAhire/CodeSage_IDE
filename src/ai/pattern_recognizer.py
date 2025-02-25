import re
import ast
import logging
from typing import List, Dict, Optional
import numpy as np
import torch
from .code_models import CodeAnalyzer
from .code_embeddings import CodeEmbeddings
from .model_config import ModelConfig

logger = logging.getLogger(__name__)

class PatternRecognizer:
    def __init__(self): 
        self.patterns = {
            'unused_import': r'import\s+\w+(?:\s*,\s*\w+)*(?!\s*\w)',
            'empty_block': r'\s*(?:if|while|for|def)\s+.*:\s*(?:\n\s*)+pass\b',
            'complex_condition': r'if.*(?:and|or).*(?:and|or).*:',
            'long_line': r'^.{80,}$',
            'nested_loops': r'\s*for.*:\s*\n\s*for.*:',
            'magic_numbers': r'\b(?!-?[01](?:\.0+)?)[0-9]+(?:\.[0-9]+)?\b(?!\s*[=<>])',
            'redundant_else': r'if.*:\s*\n\s*return.*\n\s*else:'
        }

    def _analyze_single_line(self, line: str, line_number: int) -> Dict:
        """Analyze a single line for real-time feedback"""
        issues = []
        suggestions = []
        
        # Basic pattern matching
        for pattern_name, pattern in self.patterns.items():
            if re.search(pattern, line):
                issues.append(f"Line {line_number + 1}: Potential {pattern_name.replace('_', ' ')}")
                suggestions.append(self._get_suggestion(pattern_name))
        
        # Get model-based analysis
        model_analysis = self._get_model_analysis(line)
        if model_analysis:
            issues.extend(model_analysis.get('issues', []))
            suggestions.extend(model_analysis.get('suggestions', []))
                
        return {
            'line_issues': issues,
            'line_suggestions': list(set(suggestions)),
            'line_complexity': self._calculate_line_complexity(line),
            'model_score': model_analysis.get('quality_score', 0.0) if model_analysis else 0.0
        }
        
    def _get_model_analysis(self, line: str) -> Dict:
        """Get analysis from CodeBERT model"""
        try:
            from .code_models import CodeAnalyzer
            analyzer = CodeAnalyzer()
            return analyzer.analyze_code(line)
        except Exception as e:
            logger.error(f"Model analysis error: {str(e)}")
            return {}

    def analyze_realtime(self, code: str, cursor_position: int = None) -> Dict:
        """Real-time code analysis for IDE integration"""
        try:
            # Initialize AI components
            code_analyzer = CodeAnalyzer()
            code_embeddings = CodeEmbeddings()
            
            # Get AI-powered analysis
            embeddings = code_embeddings.generate(code)
            model_analysis = code_analyzer.analyze_code(code)
            
            # Get pattern-based analysis
            pattern_analysis = self.analyze_code(code)
            
            if cursor_position is not None:
                lines = code.split('\n')
                current_line = lines[cursor_position] if 0 <= cursor_position < len(lines) else ""
                line_analysis = self._analyze_single_line(current_line, cursor_position)
                
                return {
                    'embeddings': embeddings,
                    'model_analysis': model_analysis,
                    'pattern_analysis': pattern_analysis,
                    'current_line': {
                        'analysis': line_analysis,
                        'model_score': model_analysis.get('quality_score', 0.0),
                        'suggestions': line_analysis.get('line_suggestions', [])
                    }
                }
            
            return {
                'embeddings': embeddings,
                'model_analysis': model_analysis,
                'pattern_analysis': pattern_analysis
            }
            
        except Exception as e:
            logger.error(f"Real-time analysis error: {str(e)}")
            return {}
    def analyze_code(self, code: str) -> Dict[str, List[str]]:
        try:
            issues = []
            suggestions = []
            lines = code.split('\n')
            
            for i, line in enumerate(lines, 1):
                for pattern_name, pattern in self.patterns.items():
                    if re.search(pattern, line):
                        issues.append(f"Line {i}: Potential {pattern_name.replace('_', ' ')}")
                        suggestions.append(self._get_suggestion(pattern_name))
                        
            complexity_score = self._calculate_complexity(code)
            
            return {
                'issues': issues,
                'suggestions': list(set(suggestions)),
                'complexity_score': complexity_score,
                'metrics': self._get_code_metrics(code)
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis error: {str(e)}")
            return {'issues': [], 'suggestions': [], 'complexity_score': 0, 'metrics': {}}
            
    def _calculate_complexity(self, code: str) -> float:
        try:
            complexity = 0
            lines = code.split('\n')
            
            # Count control structures
            control_structures = len(re.findall(r'\b(if|for|while|def|class)\b', code))
            nested_depth = max(len(line) - len(line.lstrip()) for line in lines) // 4
            
            complexity = (control_structures * 0.5) + (nested_depth * 1.5)
            return min(10, complexity)  # Scale from 0-10
            
        except Exception as e:
            logger.error(f"Complexity calculation error: {str(e)}")
            return 0
            
    def _get_code_metrics(self, code: str) -> Dict[str, int]:
        return {
            'total_lines': len(code.split('\n')),
            'blank_lines': len([l for l in code.split('\n') if not l.strip()]),
            'comment_lines': len([l for l in code.split('\n') if l.strip().startswith('#')]),
            'control_structures': len(re.findall(r'\b(if|for|while|def|class)\b', code))
        }
        
    def _get_suggestion(self, pattern_name: str) -> str:
        suggestions = {
            'unused_import': "Remove unused imports to improve code clarity",
            'empty_block': "Implement meaningful logic instead of using 'pass'",
            'complex_condition': "Consider breaking down complex conditions",
            'long_line': "Break long lines to improve readability (max 79 chars)",
            'nested_loops': "Consider extracting nested loops into separate functions",
            'magic_numbers': "Replace magic numbers with named constants",
            'redundant_else': "Consider removing redundant else after return"
        }
        return suggestions.get(pattern_name, "Review this section for potential improvements")