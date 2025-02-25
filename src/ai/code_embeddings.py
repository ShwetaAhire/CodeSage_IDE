import ast
import torch
import logging
import numpy as np
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel
from .model_config import ModelConfig

logger = logging.getLogger(__name__)

class CodeEmbeddings: 
    def __init__(self):
        self.config = ModelConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.config.models['codebert']
        self.tokenizer = model['tokenizer']
        self.model = model['model'].to(self.device)
        
    def generate(self, code):
        """Generate embeddings for the given code"""
        try:
            if not code.strip():
                return np.zeros((1, 768))
                
            # Tokenize code
            inputs = self.tokenizer(
                code, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
            return embeddings.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return np.zeros((1, 768))  # Return zero embeddings on error

    # ... existing code ...

    def _analyze_current_line(self, line: str) -> Dict:
        """Analyze current line features"""
        features = {
            'type': None,
            'context': {},
            'suggestions': []
        }
        
        try:
            # Analyze line type
            if 'def ' in line:
                features['type'] = 'function_definition'
            elif 'class ' in line:
                features['type'] = 'class_definition'
            elif 'import ' in line:
                features['type'] = 'import'
            elif '=' in line:
                features['type'] = 'assignment'
            elif any(keyword in line for keyword in ['if', 'for', 'while']):
                features['type'] = 'control_flow'
                
            # Get line context
            tree = ast.parse(line)
            features['context'] = self._extract_features(tree)
            
        except Exception as e:
            logger.error(f"Error analyzing line: {str(e)}")
            
        return features

# ... rest of the code remains the same ...

    def analyze_realtime(self, code: str, cursor_position: int = None) -> Dict:
        """Real-time code analysis for IDE integration"""
        try:
            # Generate embeddings using CodeBERT
            embeddings = self.generate(code)
            
            # Get code structure analysis
            features = self.analyze_code_structure(code)
            
            if cursor_position is not None:
                lines = code.split('\n')
                if 0 <= cursor_position < len(lines):
                    current_line = lines[cursor_position]
                    # Get embeddings for current line
                    line_embeddings = self.generate(current_line)
                    # Analyze current line structure
                    line_features = self._analyze_current_line(current_line)
                    features.update({
                        'current_line': {
                            'embeddings': line_embeddings,
                            'features': line_features,
                            'position': cursor_position
                        }
                    })
                
            return {
                'embeddings': embeddings,
                'features': features,
                'model_info': {
                    'device': self.device,
                    'model_type': 'codebert'
                }
            }
        except Exception as e:
            logger.error(f"Real-time analysis error: {str(e)}")
            return {}
            
    def analyze_code_structure(self, code):
        """Analyze code structure using AST"""
        try:
            tree = ast.parse(code)
            features = self._extract_features(tree)
            features['complexity'] = self._calculate_complexity(tree)
            return features
        except Exception as e:
            logger.error(f"Error analyzing code structure: {str(e)}")
            return {}
            
    def _extract_features(self, tree):
        """Extract code features from AST"""
        features = {
            'functions': [],
            'classes': [],
            'imports': [],
            'complexity': 0,
            'variables': [],
            'loops': 0,
            'conditionals': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                features['functions'].append({
                    'name': node.name,
                    'args': len(node.args.args),
                    'line_no': node.lineno
                })
            elif isinstance(node, ast.ClassDef):
                features['classes'].append({
                    'name': node.name,
                    'line_no': node.lineno
                })
            elif isinstance(node, ast.Import):
                features['imports'].extend(n.name for n in node.names)
            elif isinstance(node, (ast.For, ast.While)):
                features['loops'] += 1
            elif isinstance(node, ast.If):
                features['conditionals'] += 1
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                features['variables'].append(node.id)
            
        return features
        
    def _calculate_complexity(self, tree):
        """Calculate code complexity"""
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.FunctionDef)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity