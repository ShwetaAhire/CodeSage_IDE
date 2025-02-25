import ast
import logging
from typing import Dict, List, Any
import torch
from ..ai import CodeEmbeddings, MLPredictor, PatternRecognizer, ModelConfig

logger = logging.getLogger(__name__)

class BugDetector:
    def __init__(self):
         # Initialize AI components
        self.code_embeddings = CodeEmbeddings()
        self.ml_predictor = MLPredictor()
        self.pattern_recognizer = PatternRecognizer()
        self.config = ModelConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        
        # Use models from ModelConfig instead of direct initialization
        try:
            model = self.config.models['codebert']
            self.tokenizer = model['tokenizer']
            self.model = model['model'].to(self.device)
            self.code_smell_detector = model.get('smell_detector')
            self.security_detector = model.get('security_detector')
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            self.model = None

    def detect_bugs_realtime(self, code: str, cursor_position: int = None) -> Dict[str, Any]:
        """Real-time bug detection with AI analysis"""
        if not code.strip():
            return {"bugs": [], "analysis": {}}

        try:
            # Get embeddings and ML predictions
            embeddings = self.code_embeddings.generate(code)
            context = {'cursor_position': cursor_position} if cursor_position else None
            ml_analysis = self.ml_predictor.predict_realtime(code, context)
            pattern_analysis = self.pattern_recognizer.analyze_realtime(code, cursor_position)
            
            # Combine analyses
            bugs = []
            tree = ast.parse(code)
            
            # Static Analysis
            static_bugs = self._static_bug_detection(tree)
            bugs.extend(static_bugs)
            
            # ML-based Analysis
            ml_bugs = self._ai_bug_detection(code, embeddings, ml_analysis)
            bugs.extend(ml_bugs)
            
            # Pattern Analysis
            bugs.extend(pattern_analysis.get('issues', []))
            
            return {
                "bugs": list(set(bugs)),
                "analysis": {
                    "quality_score": ml_analysis.get('quality_score', 0.0),
                    "bug_probability": ml_analysis.get('bug_probability', 0.0),
                    "immediate_fixes": self._generate_quick_fixes(bugs, code, cursor_position),
                    "security_issues": self._detect_security_issues(code, ml_analysis)
                }
            }
    def detect_bugs(self, code):
        """Detect potential bugs using both static analysis and AI"""
        if not code.strip():
            return []

        try:
            bugs = []
            
            # Static Analysis
            tree = ast.parse(code)
            static_bugs = self._static_bug_detection(tree)
            bugs.extend(static_bugs)
            
            # AI Analysis
            ai_bugs = self._ai_bug_detection(code)
            bugs.extend(ai_bugs)
            
            # Code Smell Detection
            smell_bugs = self._detect_code_smells(code)
            bugs.extend(smell_bugs)
            
            # Security Vulnerability Detection
            security_bugs = self._detect_security_issues(code)
            bugs.extend(security_bugs)
            
            return list(set(bugs))  # Remove duplicates
            
        except SyntaxError:
            return ["Syntax error in code"]
        except Exception as e:
            return [f"Error analyzing code: {str(e)}"]

    def _static_bug_detection(self, tree):
        """Perform static bug detection"""
        bugs = []
        
        # Check for common issues
        self._check_exception_handling(tree, bugs)
        self._check_resource_management(tree, bugs)
        self._check_undefined_variables(tree, bugs)
        self._check_infinite_loops(tree, bugs)
        self._check_unreachable_code(tree, bugs)
        
        return bugs

    def _ai_bug_detection(self, code):
        """Perform AI-based bug detection"""
        try:
            # Tokenize code
            inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            bugs = []
            if predictions[0][1] > 0.7:  # High confidence threshold
                # Analyze specific code segments
                for chunk in self._split_code_chunks(code):
                    chunk_result = self._analyze_code_chunk(chunk)
                    if chunk_result:
                        bugs.extend(chunk_result)
            
            return bugs
            
        except Exception as e:
            return [f"AI Bug Detection Error: {str(e)}"]

    def _detect_code_smells(self, code):
        """Detect code smells using AI"""
        try:
            results = self.code_smell_detector(code, max_length=512)
            smells = []
            for result in results:
                if result['score'] > 0.7:
                    smells.append(f"Code smell detected: {result['label']}")
            return smells
        except Exception:
            return []

    def _detect_security_issues(self, code):
        """Detect security vulnerabilities using AI"""
        try:
            results = self.security_detector(code, max_length=512)
            issues = []
            for result in results:
                if result['score'] > 0.7:
                    issues.append(f"Security vulnerability detected: {result['label']}")
            return issues
        except Exception:
            return []

    def _analyze_code_chunk(self, chunk):
        """Analyze individual code chunk for bugs"""
        bugs = []
        try:
            # Tokenize chunk
            inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            if predictions[0][1] > 0.8:  # Higher threshold for specific chunks
                bugs.append(f"Potential bug in code segment: {chunk[:100]}...")
                
        except Exception:
            pass
        return bugs

    def _check_exception_handling(self, tree, bugs):
        """Check for proper exception handling"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                if not node.handlers:
                    bugs.append("Empty try block without exception handlers")
                for handler in node.handlers:
                    if handler.type is None or (hasattr(handler.type, 'id') and handler.type.id == 'Exception'):
                        bugs.append("Too broad exception handling detected")

    def _check_resource_management(self, tree, bugs):
        """Check for proper resource management"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if hasattr(node.func, 'id'):
                    if node.func.id in ['open', 'socket']:
                        if not self._is_within_with_block(node):
                            bugs.append(f"Resource '{node.func.id}' should be used with 'with' statement")

    def _check_undefined_variables(self, tree, bugs):
        """Check for undefined variables"""
        defined_vars = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    defined_vars.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    if node.id not in defined_vars and node.id not in dir(__builtins__):
                        bugs.append(f"Potentially undefined variable: {node.id}")

    def _check_infinite_loops(self, tree, bugs):
        """Check for potential infinite loops"""
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and node.test.value:
                    if not self._has_break_statement(node):
                        bugs.append("Potential infinite loop detected")

    def _check_unreachable_code(self, tree, bugs):
        """Check for unreachable code"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                next_stmt = self._get_next_statement(node)
                if next_stmt:
                    bugs.append("Unreachable code detected after return statement")

    def _is_within_with_block(self, node):
        """Check if a node is within a with block"""
        parent = node
        while hasattr(parent, 'parent'):
            parent = parent.parent
            if isinstance(parent, ast.With):
                return True
        return False

    def _has_break_statement(self, node):
        """Check if a loop has a break statement"""
        for child in ast.walk(node):
            if isinstance(child, ast.Break):
                return True
        return False

    def _get_next_statement(self, node):
        """Get the next statement after a node"""
        try:
            parent = node.parent
            for i, child in enumerate(parent.body):
                if child == node and i < len(parent.body) - 1:
                    return parent.body[i + 1]
        except AttributeError:
            pass
        return None

    def _split_code_chunks(self, code, chunk_size=256):
        """Split code into manageable chunks for AI analysis"""
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            if current_size + len(line) > chunk_size:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(line)
            current_size += len(line)
            
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks 