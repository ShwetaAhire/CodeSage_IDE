import torch
from .ai import (
    CodeEmbeddings, 
    MLPredictor, 
    PatternRecognizer, 
    ModelManager,
    CodeBERTModel,
    CodeT5Model
)
from .core import (
    CodeAnalyzer, 
    BugDetector, 
    IDEAnalyzer,
    ConfigManager
)
from .ui import (
    MainWindow, 
    CodeEditor, 
    SuggestionsPanel, 
    DebugPanel
)
from .utils import (
    CodeParser,
    FileHandler,
    IDEThemes,
    Logger,
    ErrorHandler
)
from .tests import TestRunner, TestGenerator

__version__ = '1.0.0'
__author__ = 'AI Intelligent IDE Team'

# Initialize global model manager
model_manager = ModelManager()
model_manager.initialize_models({
    'codebert': CodeBERTModel(),
    'codet5': CodeT5Model(),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
})

__all__ = [
    # AI Components
    'CodeEmbeddings',
    'MLPredictor',
    'PatternRecognizer',
    'ModelManager',
    'CodeBERTModel',
    'CodeT5Model',
    
    # Core Components
    'CodeAnalyzer',
    'BugDetector',
    'IDEAnalyzer',
    'ConfigManager',
    
    # UI Components
    'MainWindow',
    'CodeEditor',
    'SuggestionsPanel',
    'DebugPanel',
    
    # Utility Components
    'CodeParser',
    'FileHandler',
    'IDEThemes',
    'Logger',
    'ErrorHandler',
    
    # Testing Components
    'TestRunner',
    'TestGenerator',
    
    # Global Instances
    'model_manager'
]