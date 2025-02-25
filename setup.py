from setuptools import setup, find_packages

setup(
    name="ai_intelligent_ide",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        # AI Model Dependencies
        'torch>=2.0.1',
        'transformers>=4.30.2',
        'tokenizers>=0.13.3',
        'sentencepiece>=0.1.99',
        'numpy>=1.24.3',
        
        # IDE Core Dependencies
        'python-language-server>=0.36.2',
        'pylint>=2.17.4',
        'jedi>=0.18.2',
        'rope>=1.9.0',
        'autopep8>=2.0.2',
        
        # Testing and Quality
        'pytest>=7.3.1',
        'pytest-cov>=4.1.0',
        'black>=23.3.0',
        
        # Backend Services
        'fastapi>=0.100.0',
        'uvicorn>=0.22.0',
        'websockets>=11.0.3',
        
        # UI Requirements
        'pillow>=10.0.0',
        
        # Performance Monitoring
        'psutil>=5.9.5',
    ],
    extras_require={
        'gpu': [
            'torch>=2.0.1; platform_system=="Windows"'
        ]
    },
    entry_points={
        'console_scripts': [
            'ai-ide=src.main:main',
        ],
    },
    package_data={
        'ai_intelligent_ide': [
            'models/*.bin',
            'models/*.json',
            'config/*.yaml',
        ]
    },
    author="Shweta Ahire",
    author_email="ahireshweta99@gmail.com",
    description="An AI-powered Intelligent IDE with CodeBERT and CodeT5",
    keywords="IDE, AI, CodeBERT, CodeT5, code generation, debugging",
    python_requires='>=3.8',
)