import unittest
import torch
from pathlib import Path
from ..ai import ModelManager, CodeGenerator, MLPredictor
from ..core import CodeAnalyzer, TestGenerator
from ..utils import CodeParser, FileHandler
from . import TestBase

class TestCodeGenerator(TestBase):
    @classmethod
    def setUpClass(cls):
        """Initialize test environment with models"""
        super().setUpClass()
        cls.code_generator = CodeGenerator(cls.model_manager)
        cls.test_generator = TestGenerator(cls.model_manager)
        cls.file_handler = FileHandler()

    def test_function_generation(self):
        """Test function code generation"""
        prompt = "Generate a function to calculate fibonacci sequence"
        generated_code = self.code_generator.generate_function(
            prompt,
            max_length=200,
            num_return_sequences=3
        )
        
        self.assertIsNotNone(generated_code)
        for code in generated_code:
            # Verify code quality
            analysis = self.code_analyzer.analyze_code(code)
            self.assertGreaterEqual(analysis['quality_score'], 0.7)
            
            # Check for basic function structure
            self.assertIn('def', code)
            self.assertIn('fibonacci', code.lower())

    def test_test_case_generation(self):
        """Test automatic test case generation"""
        source_code = """
        def calculate_average(numbers):
            return sum(numbers) / len(numbers)
        """
        
        test_cases = self.test_generator.generate_test_cases(
            source_code,
            num_cases=3
        )
        
        self.assertGreaterEqual(len(test_cases), 1)
        for test_case in test_cases:
            self.assertIn('def test_', test_case)
            self.assertIn('calculate_average', test_case)
            self.assertIn('assert', test_case)

    def test_code_completion(self):
        """Test code completion generation"""
        incomplete_code = """
        def sort_list(items):
            if not items:
                return []
            pivot = items[0]
        """
        
        completions = self.code_generator.complete_code(
            incomplete_code,
            max_tokens=100
        )
        
        self.assertGreater(len(completions), 0)
        for completion in completions:
            self.assertIn('return', completion)
            analysis = self.code_analyzer.analyze_code(completion)
            self.assertNotIn('error', analysis)

    def test_docstring_generation(self):
        """Test docstring generation"""
        code_without_docs = """
        def process_data(data, threshold=0.5):
            filtered = [x for x in data if x > threshold]
            return sum(filtered) / len(filtered) if filtered else 0
        """
        
        docstring = self.code_generator.generate_docstring(code_without_docs)
        self.assertIn('Args:', docstring)
        self.assertIn('Returns:', docstring)
        self.assertIn('threshold', docstring)
        self.assertIn('data', docstring)

    def test_code_optimization(self):
        """Test code optimization suggestions"""
        unoptimized_code = """
        def find_duplicates(items):
            duplicates = []
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if items[i] == items[j] and items[i] not in duplicates:
                        duplicates.append(items[i])
            return duplicates
        """
        
        optimized_versions = self.code_generator.suggest_optimizations(
            unoptimized_code,
            num_suggestions=2
        )
        
        self.assertGreater(len(optimized_versions), 0)
        for optimized in optimized_versions:
            # Verify performance improvement
            self.assertIn('set()', optimized.lower())
            analysis = self.code_analyzer.analyze_performance(optimized)
            self.assertGreater(analysis['performance_score'], 0.7)

    def test_error_handling_generation(self):
        """Test error handling code generation"""
        base_code = """
        def divide_numbers(a, b):
            return a / b
        """
        
        safe_code = self.code_generator.add_error_handling(base_code)
        self.assertIn('try:', safe_code)
        self.assertIn('except', safe_code)
        self.assertIn('ZeroDivisionError', safe_code)

    def test_type_hint_generation(self):
        """Test type hint generation"""
        code_without_types = """
        def process_user_data(name, age, scores):
            return {
                'name': name.strip(),
                'age': int(age),
                'average_score': sum(scores) / len(scores)
            }
        """
        
        code_with_types = self.code_generator.add_type_hints(code_without_types)
        self.assertIn('str', code_with_types)
        self.assertIn('List[', code_with_types)
        self.assertIn('Dict[', code_with_types)

    def tearDown(self):
        """Clean up resources"""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == '__main__':
    unittest.main() 