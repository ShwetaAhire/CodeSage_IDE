import ast
import logging
from pathlib import Path
from transformers import pipeline
from ..ai import CodeEmbeddings, MLPredictor, PatternRecognizer

logger = logging.getLogger(__name__)

class TestGenerator:
    def __init__(self):
        # Initialize AI components
        self.code_embeddings = CodeEmbeddings()
        self.ml_predictor = MLPredictor()
        self.pattern_recognizer = PatternRecognizer()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Create models directory for caching
        models_dir = Path("./models")
        models_dir.mkdir(exist_ok=True)
        
        try:
            self.model = pipeline(
                "text2text-generation",
                model="Salesforce/codet5-base",
                device=-1,  # Force CPU usage
                cache_dir=str(models_dir),
                local_files_only=False 
            )
        except Exception as e:
            print(f"Warning: AI initialization failed: {str(e)}")
            self.model = None


    def generate_tests(self, code, context=None):
        """Generate test cases for the given code with real-time analysis"""
        if not code.strip():
            return {"tests": ["No code to generate tests for"]}

        try:
            # Get code embeddings and analysis
            embeddings = self.code_embeddings.generate(code)
            code_analysis = self.ml_predictor.predict_realtime(code, context)
            pattern_analysis = self.pattern_recognizer.analyze_realtime(code)
            
            tests = []
            tree = ast.parse(code)
            functions = self._extract_functions(tree)
            
            for func in functions:
                # Static analysis based tests
                static_tests = self._generate_static_tests(func)
                tests.extend(static_tests)
                
                # AI-based test generation
                if self.model:
                    ai_tests = self._generate_ai_tests(func, embeddings, code_analysis)
                    tests.extend(ai_tests)
            
            return {
                "tests": tests,
                "analysis": {
                    "code_quality": code_analysis.get('quality_score', 0.0),
                    "patterns": pattern_analysis.get('issues', []),
                    "suggestions": code_analysis.get('immediate_suggestions', [])
                }
            }
            
        except SyntaxError:
            return {"tests": ["Syntax error in code"], "analysis": {}}
        except Exception as e:
            logger.error(f"Test generation error: {str(e)}")
            return {"tests": [f"Test generation error: {str(e)}"], "analysis": {}}

    def _generate_ai_tests(self, func, embeddings, code_analysis):
        """Generate tests using AI with enhanced context"""
        try:
            if not self.model:
                return []

            func_code = ast.unparse(func)
            
            # Enhanced prompt using code analysis
            prompt = f"""Generate pytest test cases for this function with quality score {code_analysis.get('quality_score', 0.0):.2f}:
            {func_code}
            Consider these aspects:
            - Function complexity
            - Edge cases
            - Error handling
            """
            
            response = self.model(
                prompt, 
                max_length=200, 
                num_return_sequences=1,
                temperature=0.7
            )
            
            tests = []
            for r in response:
                test_code = r['generated_text'].strip()
                if test_code:
                    # Analyze generated test code
                    test_analysis = self.pattern_recognizer.analyze_code(test_code)
                    if test_analysis.get('complexity_score', 0) < 8:  # Filter overly complex tests
                        tests.append(self._format_test(test_code))
            
            return tests
            
        except Exception as e:
            logger.error(f"AI test generation failed: {str(e)}")
            return []


    # Add real-time test suggestion method
    def suggest_tests_realtime(self, code: str, cursor_position: int = None) -> Dict:
        """Provide real-time test suggestions as user types"""
        try:
            # Get real-time analysis
            embeddings = self.code_embeddings.generate(code)
            context = {'cursor_position': cursor_position} if cursor_position is not None else None
            code_analysis = self.ml_predictor.predict_realtime(code, context)
            
            # Get current function under cursor
            tree = ast.parse(code)
            current_function = self._get_function_at_cursor(tree, cursor_position) if cursor_position else None
            
            suggestions = []
            if current_function:
                ai_tests = self._generate_ai_tests(current_function, embeddings, code_analysis)
                suggestions.extend(ai_tests)
            
            return {
                'suggestions': suggestions,
                'analysis': code_analysis,
                'quality_score': code_analysis.get('quality_score', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Real-time test suggestion error: {str(e)}")
            return {'suggestions': [], 'analysis': {}, 'quality_score': 0.0}

    def _get_function_at_cursor(self, tree, cursor_position: int):
        """Find function definition at cursor position"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.lineno <= cursor_position <= node.end_lineno:
                    return node
        return None
        
    def _extract_functions(self, tree):
        """Extract functions and methods from the code"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node)
        return functions

    def _generate_static_tests(self, func):
        """Generate tests based on static analysis"""
        tests = []
        
        # Basic function test
        tests.append(f"def test_{func.name}():")
        
        # Check function parameters
        if func.args.args:
            params = [arg.arg for arg in func.args.args]
            tests.append(f"    # Test with valid parameters: {', '.join(params)}")
        
        # Check return values
        returns = [n for n in ast.walk(func) if isinstance(n, ast.Return)]
        if returns:
            tests.append("    # Test return value")
        
        # Check exceptions
        try_blocks = [n for n in ast.walk(func) if isinstance(n, ast.Try)]
        if try_blocks:
            tests.append("    # Test exception handling")
        
        return tests


    def _format_test(self, test_code):
        """Format the generated test code"""
        try:
            tree = ast.parse(test_code)
            return ast.unparse(tree)
        except:
            return test_code