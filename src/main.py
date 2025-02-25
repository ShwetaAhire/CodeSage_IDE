import sys
import logging
import torch
from pathlib import Path
from typing import Optional

from . import model_manager
from .ui import MainWindow
from .ai import CodeEmbeddings, MLPredictor, PatternRecognizer
from .core import ConfigManager, CodeAnalyzer, BugDetector, IDEAnalyzer
from .utils import Logger, ErrorHandler
from .tests import TestRunner

class AIIntelligentIDE:
    def __init__(self):
        self.logger = Logger.setup_logger()
        self.error_handler = ErrorHandler()
        
        try:
            # Initialize AI Model Components
            self.model_manager = ModelManager()
            self.model_manager.initialize_models({
                'codebert': {
                    'model_path': 'models/codebert-base',
                    'tokenizer_path': 'models/codebert-tokenizer',
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
                },
                'codet5': {
                    'model_path': 'models/codet5-base',
                    'tokenizer_path': 'models/codet5-tokenizer',
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
                }
            })
            
            # Initialize AI Components
            self.code_embeddings = CodeEmbeddings(self.model_manager)
            self.ml_predictor = MLPredictor(self.model_manager)
            self.pattern_recognizer = PatternRecognizer(self.model_manager)
            
            # Initialize Core Components
            self.code_analyzer = CodeAnalyzer(
                self.code_embeddings,
                self.ml_predictor,
                self.pattern_recognizer
            )
            self.bug_detector = BugDetector(
                self.code_embeddings,
                self.ml_predictor
            )
            self.ide_analyzer = IDEAnalyzer(
                self.code_analyzer,
                self.bug_detector
            )
            
            # Load configuration
            self.config = ConfigManager()
            self.config.load_config()
            
            # Initialize test components
            self.test_runner = TestRunner(self.ml_predictor)
            
            # Create main window with AI components
            self.main_window = MainWindow(
                analyzer=self.ide_analyzer,
                code_analyzer=self.code_analyzer,
                bug_detector=self.bug_detector,
                model_manager=self.model_manager
            )
            
            self.error_handler.set_main_window(self.main_window)
            
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            sys.exit(1)

    def _start_realtime_analysis(self):
        """Initialize real-time code analysis with ML models"""
        try:
            self.main_window.analyzer.set_model_manager(self.model_manager)
            self.main_window.analyzer.configure_analysis({
                'embedding_batch_size': 32,
                'prediction_threshold': 0.85,
                'max_suggestions': 10,
                'analysis_interval': 1.0,  # seconds
                'use_cache': True
            })
            
            # Start background analysis with ML components
            self.main_window.analyzer.start_background_analysis(
                self.main_window.get_current_code,
                self.main_window.update_suggestions,
                self.code_embeddings,
                self.ml_predictor
            )
            
        except Exception as e:
            self.logger.error(f"Analysis initialization error: {str(e)}")

    def _apply_initial_config(self):
        """Apply initial IDE configuration with ML settings"""
        try:
            theme = self.config.get('theme', 'Default')
            self.main_window.apply_theme(theme)
            
            # Configure ML models
            model_config = {
                'batch_size': self.config.get('batch_size', 32),
                'max_sequence_length': self.config.get('max_sequence_length', 512),
                'prediction_threshold': self.config.get('prediction_threshold', 0.85),
                'use_gpu': self.config.get('use_gpu', True) and torch.cuda.is_available()
            }
            self.model_manager.configure_models(model_config)
            
            # Set up file associations and analysis settings
            file_types = self.config.get('file_types', ['.py', '.js', '.java'])
            self.main_window.set_file_types(file_types)
            
        except Exception as e:
            self.logger.error(f"Configuration error: {str(e)}")