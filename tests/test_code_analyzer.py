import unittest
import torch
from pathlib import Path
from ..ai import ModelManager, CodeEmbeddings, MLPredictor, PatternRecognizer
from ..core import CodeAnalyzer, BugDetector
from ..utils import CodeParser
from . import TestBase

class TestCodeAnalyzer(TestBase):
    @classmethod
    def setUpClass(cls):
        """Initialize models and components"""
        super().setUpClass()
        cls.model_paths = {
            'codebert': {
                'model': 'models/codebert-base',
                'tokenizer': 'models/codebert-tokenizer'
            },
            'codet5': {
                'model': 'models/codet5-base',
                'tokenizer': 'models/codet5-tokenizer'
            }
        }
        
        # Load models with GPU support
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model_manager.load_models(cls.model_paths, cls.device)

    def setUp(self):
        """Set up test components"""
        super().setUp()
        self.code_embeddings = CodeEmbeddings(self.model_manager)
        self.ml_predictor = MLPredictor(self.model_manager)
        self.pattern_recognizer = PatternRecognizer(self.model_manager)
        self.code_analyzer = CodeAnalyzer(
            self.code_embeddings,
            self.ml_predictor,
            self.pattern_recognizer
        )

    def test_model_loading(self):
        """Test if models are loaded correctly"""
        self.assertTrue(self.model_manager.is_model_loaded('codebert'))
        self.assertTrue(self.model_manager.is_model_loaded('codet5'))
        self.assertEqual(self.model_manager.get_device(), self.device)

    def test_codebert_embeddings(self):
        """Test CodeBERT embeddings generation"""
        test_code = "def test_function(): pass"
        embeddings = self.code_embeddings.generate(test_code)
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.device, self.device)
        self.assertEqual(len(embeddings.shape), 2)

    def test_codet5_completion(self):
        """Test CodeT5 code completion"""
        test_code = "def calculate_average(numbers):\n    total = sum("
        completions = self.ml_predictor.predict_completion(
            test_code,
            max_length=50,
            num_return_sequences=5
        )
        self.assertGreater(len(completions), 0)
        self.assertTrue(any('numbers' in c for c in completions))

    def test_realtime_analysis(self):
        """Test real-time code analysis"""
        test_code = """
        def process_data(data):
            results = []
            for item in data:
                results.append(item * 2)
            return results
        """
        analysis = self.code_analyzer.analyze_realtime(
            test_code,
            cursor_position=len(test_code)
        )
        
        self.assertIn('quality_score', analysis)
        self.assertIn('suggestions', analysis)
        self.assertIn('bugs', analysis)
        self.assertGreaterEqual(analysis['quality_score'], 0.0)

    def test_error_handling(self):
        """Test error handling in analysis"""
        invalid_code = "def invalid_function("
        analysis = self.code_analyzer.analyze_code(invalid_code)
        self.assertIn('error', analysis)
        self.assertIn('syntax_error', analysis['error'])

    def test_model_integration(self):
        """Test integration between different models"""
        test_code = """
        def fibonacci(n):
            if n <= 1: return n
            return fibonacci(n-1) + fibonacci(n-2)
        """
        
        # Get embeddings from CodeBERT
        embeddings = self.code_embeddings.generate(test_code)
        
        # Use embeddings for CodeT5 analysis
        analysis = self.ml_predictor.predict_with_embeddings(
            test_code,
            embeddings
        )
        
        self.assertIn('suggestions', analysis)
        self.assertIn('performance', analysis)
        self.assertTrue(any('recursive' in s for s in analysis['suggestions']))

    def test_batch_processing(self):
        """Test batch processing capabilities"""
        test_codes = [
            "def func1(): pass",
            "def func2(): return None",
            "class TestClass: pass"
        ]
        
        batch_results = self.code_analyzer.analyze_batch(test_codes)
        self.assertEqual(len(batch_results), len(test_codes))
        for result in batch_results:
            self.assertIn('quality_score', result)

    def tearDown(self):
        """Clean up resources"""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == '__main__':
    unittest.main()