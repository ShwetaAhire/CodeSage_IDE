import unittest
import torch
from ..ai import ModelManager, CodeBERTModel, CodeT5Model
from ..core import CodeAnalyzer, BugDetector
from .test_runner import TestRunner
from .test_generator import TestGenerator
from .test_utils import TestUtils

# Initialize test model manager
test_model_manager = ModelManager()
test_model_manager.initialize_models({
    'codebert': CodeBERTModel(),
    'codet5': CodeT5Model(),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
})

class TestBase(unittest.TestCase):
    """Base class for all test cases with ML model support"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with ML models"""
        cls.model_manager = test_model_manager
        cls.test_runner = TestRunner(cls.model_manager)
        cls.test_generator = TestGenerator(cls.model_manager)
        cls.test_utils = TestUtils()

    def setUp(self):
        """Set up individual test with necessary components"""
        self.code_analyzer = CodeAnalyzer(self.model_manager)
        self.bug_detector = BugDetector(self.model_manager)

    def assertCodeQuality(self, code: str, min_score: float = 0.7):
        """Assert code quality using ML models"""
        quality_score = self.code_analyzer.analyze_code_quality(code)
        self.assertGreaterEqual(quality_score, min_score)

    def assertNoBugs(self, code: str):
        """Assert no critical bugs using ML detection"""
        bugs = self.bug_detector.detect_bugs_realtime(code)
        critical_bugs = [bug for bug in bugs if bug.severity == 'critical']
        self.assertEqual(len(critical_bugs), 0)

    def assertValidSuggestions(self, code: str):
        """Assert valid code suggestions from ML models"""
        suggestions = self.code_analyzer.get_suggestions(code)
        self.assertIsNotNone(suggestions)
        self.assertGreater(len(suggestions), 0)

__all__ = [
    'TestBase',
    'TestRunner',
    'TestGenerator',
    'TestUtils',
    'test_model_manager'
]