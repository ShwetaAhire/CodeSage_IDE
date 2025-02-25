import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from ..ai import CodeEmbeddings, MLPredictor, PatternRecognizer
from ..core import CodeAnalyzer, BugDetector
from ..utils.code_parser import CodeParser
 
logger = logging.getLogger(__name__)

class FileHandler:
    def __init__(self):
        # Initialize AI components
        self.code_embeddings = CodeEmbeddings()
        self.ml_predictor = MLPredictor()
        self.pattern_recognizer = PatternRecognizer()
        self.code_analyzer = CodeAnalyzer()
        self.bug_detector = BugDetector()
        self.code_parser = CodeParser()
        
        # Cache for file analysis
        self._file_cache = {}
        self._embedding_cache = {}

    def read_file(self, file_path: str) -> Dict[str, Any]:
        """Read and analyze file with AI insights"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Generate file embeddings
            if file_path not in self._embedding_cache:
                self._embedding_cache[file_path] = self.code_embeddings.generate(content)
                
            # Get real-time analysis
            analysis = self._analyze_file_content(
                content, 
                file_path, 
                self._embedding_cache[file_path]
            )
            
            return {
                'content': content,
                'analysis': analysis,
                'file_info': self._get_file_info(file_path)
            }
            
        except Exception as e:
            logger.error(f"File read error: {str(e)}")
            return {'error': str(e)}

    def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Write file with pre-save analysis"""
        try:
            # Pre-save analysis
            embeddings = self.code_embeddings.generate(content)
            analysis = self._analyze_file_content(content, file_path, embeddings)
            
            # Check for potential issues
            if analysis['critical_issues']:
                return {
                    'success': False,
                    'warnings': analysis['critical_issues']
                }
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
                
            # Update caches
            self._embedding_cache[file_path] = embeddings
            self._file_cache[file_path] = {
                'content': content,
                'analysis': analysis
            }
            
            return {
                'success': True,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"File write error: {str(e)}")
            return {'error': str(e)}

    def _analyze_file_content(self, content: str, file_path: str, embeddings: Any) -> Dict[str, Any]:
        """Analyze file content using ML models"""
        try:
            # Get file type specific analysis
            file_type = Path(file_path).suffix.lower()
            context = {
                'file_type': file_type,
                'embeddings': embeddings,
                'file_path': file_path
            }
            
            # Get comprehensive analysis
            ml_analysis = self.ml_predictor.predict_realtime(content, context)
            pattern_analysis = self.pattern_recognizer.analyze_realtime(content)
            code_quality = self.code_analyzer.analyze_code(content)
            potential_bugs = self.bug_detector.detect_bugs_realtime(content)
            
            # Parse code structure
            parsed_code = self.code_parser.parse_realtime(content)
            
            return {
                'quality_score': code_quality.get('score', 0.0),
                'complexity': ml_analysis.get('complexity_score', 0.0),
                'suggestions': ml_analysis.get('suggestions', []),
                'patterns': pattern_analysis.get('patterns', []),
                'bugs': potential_bugs,
                'security_issues': ml_analysis.get('security_issues', []),
                'performance_insights': ml_analysis.get('performance_insights', []),
                'code_structure': parsed_code.get('structure', {}),
                'critical_issues': self._get_critical_issues(ml_analysis, potential_bugs)
            }
            
        except Exception as e:
            logger.error(f"Content analysis error: {str(e)}")
            return {}

    def _get_critical_issues(self, ml_analysis: Dict, bugs: List) -> List[str]:
        """Identify critical issues that need attention"""
        critical_issues = []
        
        # Check for security vulnerabilities
        if 'security_issues' in ml_analysis:
            critical_issues.extend([
                issue for issue in ml_analysis['security_issues']
                if issue.get('severity', 'low') == 'high'
            ])
            
        # Check for critical bugs
        critical_issues.extend([
            bug for bug in bugs
            if bug.get('severity', 'low') == 'high'
        ])
        
        return critical_issues

    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information with ML insights"""
        try:
            stats = os.stat(file_path)
            return {
                'size': stats.st_size,
                'modified': stats.st_mtime,
                'created': stats.st_ctime,
                'type': Path(file_path).suffix.lower(),
                'name': Path(file_path).name
            }
        except Exception as e:
            logger.error(f"File info error: {str(e)}")
            return {}

    def get_file_suggestions(self, file_path: str) -> List[str]:
        """Get file-specific suggestions using ML models"""
        try:
            if file_path in self._file_cache:
                cached_analysis = self._file_cache[file_path]['analysis']
                return cached_analysis.get('suggestions', [])
            return []
        except Exception:
            return []