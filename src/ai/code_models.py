import ast
import torch
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .model_config import ModelConfig
from .code_embeddings import CodeEmbeddings

logger = logging.getLogger(__name__)

class CodeAnalyzer:
    def __init__(self):
        self.config = ModelConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.config.models['codebert']['model'].to(self.device)
        self.tokenizer = self.config.models['codebert']['tokenizer']

    def analyze_for_ide(self, code: str, cursor_position: int = None) -> Dict:
        """Analyze code for IDE integration"""
        try:
            analysis = {
                'quality': self._analyze_quality(code),
                'structure': self._analyze_structure(code),
                'suggestions': self._generate_suggestions(code)
            }
            
            if cursor_position is not None:
                analysis['cursor_context'] = self._analyze_cursor_context(
                    code, cursor_position
                )
                
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return {}

    def analyze_code(self, code_snippet: str) -> Dict[str, Any]:
        try:
            if not code_snippet.strip():
                return {}
                
            model = self.config.models['codebert']
            inputs = model['tokenizer'](
                code_snippet,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model['model'](**inputs)
                
            analysis_result = {
                'quality_score': float(torch.sigmoid(outputs.logits[0][1])),
                'complexity': self._analyze_complexity(code_snippet),
                'suggestions': self._generate_suggestions(outputs),
                'structure': self._analyze_structure(code_snippet),
                'metrics': self._calculate_metrics(code_snippet)
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Code analysis error: {str(e)}")
            return {}

    def _analyze_quality(self, code: str) -> Dict:
        """Analyze code quality"""
        try:
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                quality_score = float(torch.sigmoid(outputs.logits[0][1]))
            
            return {
                'score': quality_score,
                'readability': self._assess_readability(code),
                'complexity': self._analyze_complexity(code),
                'maintainability': quality_score * 0.7 + self._assess_readability(code) * 0.3
            }
        except Exception as e:
            logger.error(f"Quality analysis error: {str(e)}")
            return {}

    def _analyze_complexity(self, code: str) -> float:
        try:
            tree = ast.parse(code)
            complexity = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While)):
                    complexity += 1
                elif isinstance(node, ast.FunctionDef):
                    complexity += 2
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
                    
            lines = code.split('\n')
            indentation_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
            avg_indentation = sum(indentation_levels) / (len(indentation_levels) or 1)
            
            return (complexity * 0.7 + avg_indentation * 0.3) / 10
            
        except Exception as e:
            logger.error(f"Complexity analysis error: {str(e)}")
            return 0.0

    def _assess_readability(self, code: str) -> float:
        """Assess code readability"""
        try:
            lines = code.split('\n')
            metrics = {
                'avg_line_length': sum(len(l.strip()) for l in lines) / len(lines),
                'comment_ratio': len([l for l in lines if l.strip().startswith('#')]) / len(lines),
                'blank_line_ratio': len([l for l in lines if not l.strip()]) / len(lines)
            }
            
            # Calculate readability score (0-1)
            score = 1.0
            if metrics['avg_line_length'] > 80:
                score *= 0.8
            if metrics['comment_ratio'] < 0.1:
                score *= 0.9
            if metrics['blank_line_ratio'] < 0.05:
                score *= 0.9
                
            return score
            
        except Exception as e:
            logger.error(f"Readability assessment error: {str(e)}")
            return 0.5

    def _analyze_structure(self, code: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(code)
            structure = {
                'functions': [],
                'classes': [],
                'imports': [],
                'variables': [],
                'metrics': {
                    'depth': 0,
                    'branches': 0,
                    'loops': 0
                }
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    structure['functions'].append({
                        'name': node.name,
                        'args': len(node.args.args),
                        'line': node.lineno,
                        'complexity': self._calculate_function_complexity(node)
                    })
                elif isinstance(node, ast.ClassDef):
                    structure['classes'].append({
                        'name': node.name,
                        'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                        'line': node.lineno
                    })
                elif isinstance(node, ast.Import):
                    structure['imports'].extend([n.name for n in node.names])
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    structure['variables'].append(node.id)
                    
            return structure
        except Exception as e:
            logger.error(f"Structure analysis error: {str(e)}")
            return {}

    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate complexity of a function"""
        complexity = 1  # Base complexity
        
        for n in ast.walk(node):
            if isinstance(n, (ast.If, ast.For, ast.While)):
                complexity += 1
            elif isinstance(n, ast.BoolOp):
                complexity += len(n.values) - 1
                
        return complexity

    def _analyze_cursor_context(self, code: str, position: int) -> Dict:
        """Analyze code context at cursor position"""
        try:
            lines = code.split('\n')
            current_line_no = 0
            current_pos = 0
            
            # Find current line number
            for i, line in enumerate(lines):
                if current_pos + len(line) + 1 >= position:
                    current_line_no = i
                    break
                current_pos += len(line) + 1
            
            current_line = lines[current_line_no]
            
            context = {
                'line_number': current_line_no + 1,
                'line_content': current_line,
                'scope': self._determine_scope(lines, current_line_no),
                'suggestions': self._get_context_suggestions(current_line),
                'symbols': self._get_nearby_symbols(lines, current_line_no)
            }
            
            return context
        except Exception as e:
            logger.error(f"Cursor context analysis error: {str(e)}")
            return {}

    def _determine_scope(self, lines: List[str], line_no: int) -> Dict:
        """Determine code scope at given line"""
        scope = {'type': 'global', 'name': None}
        current_indent = len(lines[line_no]) - len(lines[line_no].lstrip())
        
        for i in range(line_no - 1, -1, -1):
            line = lines[i]
            indent = len(line) - len(line.lstrip())
            if indent < current_indent:
                if line.lstrip().startswith('def '):
                    scope = {'type': 'function', 'name': line.split('def ')[1].split('(')[0]}
                elif line.lstrip().startswith('class '):
                    scope = {'type': 'class', 'name': line.split('class ')[1].split('(')[0]}
                break
                
        return scope

    def _get_context_suggestions(self, line: str) -> List[str]:
        """Generate context-aware suggestions"""
        suggestions = []
        
        if 'def ' in line:
            suggestions.append("Add function documentation")
            suggestions.append("Consider adding type hints")
        elif 'class ' in line:
            suggestions.append("Add class documentation")
            suggestions.append("Consider adding method docstrings")
        elif '=' in line:
            suggestions.append("Consider adding type annotations")
        
        return suggestions

    def _get_nearby_symbols(self, lines: List[str], line_no: int) -> Dict:
        """Get nearby symbol definitions"""
        symbols = {'variables': [], 'functions': [], 'classes': []}
        start = max(0, line_no - 5)
        end = min(len(lines), line_no + 5)
        
        for line in lines[start:end]:
            if '=' in line and not line.strip().startswith('#'):
                var = line.split('=')[0].strip()
                if var.isidentifier():
                    symbols['variables'].append(var)
            elif 'def ' in line:
                func = line.split('def ')[1].split('(')[0]
                symbols['functions'].append(func)
            elif 'class ' in line:
                cls = line.split('class ')[1].split('(')[0]
                symbols['classes'].append(cls)
                
        return symbols

    def _calculate_metrics(self, code: str) -> Dict[str, Any]:
        try:
            lines = code.split('\n')
            return {
                'total_lines': len(lines),
                'code_lines': len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
                'comment_lines': len([l for l in lines if l.strip().startswith('#')]),
                'blank_lines': len([l for l in lines if not l.strip()]),
                'avg_line_length': sum(len(l) for l in lines) / (len(lines) or 1)
            }
        except Exception as e:
            logger.error(f"Metrics calculation error: {str(e)}")
            return {}