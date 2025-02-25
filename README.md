# CodeSage IDE

An intelligent integrated development environment powered by state-of-the-art AI models (CodeBERT and CodeT5) that provides real-time code analysis, suggestions, and assistance.

## Key Features

### Real-time AI-Powered Analysis
- Live code quality assessment using CodeBERT embeddings
- Intelligent code completion with CodeT5
- Pattern-based code analysis and suggestions
- Real-time bug detection and security vulnerability scanning
- Performance optimization recommendations

### Smart Code Generation
- Context-aware code completion
- Automated test case generation
- Intelligent docstring generation
- Type hint suggestions
- Error handling code generation

### Advanced IDE Features
- Multi-language support (Python, JavaScript, Java, C++)
- Multiple theme options (Default, Dark, Monokai, Nord, Solarized)
- Split-view editing
- Integrated debugging panel
- Real-time syntax highlighting
- Tab management with context menus

### Test Generation Features
- Automatic unit test generation using CodeT5
- Test case coverage analysis
- Edge case detection
- Integration test suggestions
- Test documentation generation

### Developer Productivity Tools
- Automated code refactoring suggestions
- Performance analysis with optimization tips
- Security vulnerability detection
- Code quality metrics
- Intelligent error handling suggestions

## Unique Advantages

- **AI-First Approach**: Unlike traditional IDEs, our solution integrates AI at every level, from code completion to bug detection
- **Real-time Analysis**: Continuous code analysis without impacting performance
- **Context-Aware**: Suggestions based on your coding style and project context
- **Performance Focused**: Built-in performance analysis and optimization suggestions
- **Modern UI**: Clean, customizable interface with multiple theme options

## Technical Stack

- **AI Models**: 
  - CodeBERT for code embeddings and analysis
  - CodeT5 for code generation and completion
  
- **Core Technologies**:
  - Python 3.8+
  - PyTorch for AI model inference
  - Tkinter for UI
  - FastAPI for backend services


## Requirements

- torch>=1.9.0
- transformers>=4.15.0
- fastapi>=0.68.0
- uvicorn>=0.15.0
- tkinter>=8.6
- numpy>=1.19.5
- pandas>=1.3.0
- pytest>=6.2.5
- black>=21.5b2
- pylint>=2.8.2
- coverage>=5.5
- python-dotenv>=0.19.0
- colorama>=0.4.4
- tqdm>=4.62.0

Optional Dependencies:
- CUDA Toolkit 11.1+ (for GPU acceleration)
- Git LFS (for downloading model files)
Note: Some packages might require additional system-level dependencies. Please refer to their respective documentation for complete installation instructions.

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/ShwetaAhire/CodeSage_IDE.git
cd ai_intelligent_ide
```

2. Install dependencies:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -e .
```

3. Download AI models:
```bash
python scripts/download_models.py
```

4. Run the IDE:
```bash
python -m src.main
```

# System Requirements
- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- NVIDIA GPU with CUDA support (recommended for optimal performance)
- Windows 10/11

