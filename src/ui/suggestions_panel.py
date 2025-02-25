import tkinter as tk
from tkinter import ttk
from src.utils.themes import IDEThemes 
from ..ai import CodeEmbeddings, MLPredictor, PatternRecognizer
from ..core.code_analyzer import CodeAnalyzer
import logging

logger = logging.getLogger(__name__)

class SuggestionsPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
         # Initialize AI components
        self.code_embeddings = CodeEmbeddings()
        self.ml_predictor = MLPredictor()
        self.pattern_recognizer = PatternRecognizer()
        self.code_analyzer = CodeAnalyzer()
        self.themes = IDEThemes()
        
        # Create notebook for different types of suggestions
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both')
        
         # Create enhanced tabs with real-time analysis
        self.suggestions_text = self._create_suggestions_tab("AI Suggestions (0)")
        self.bugs_text = self._create_suggestions_tab("ML Bugs (0)")
        self.tests_text = self._create_suggestions_tab("Auto Tests (0)")
        self.performance_text = self._create_suggestions_tab("Performance (0)")
        
        # Store tab indices with new performance tab
        self.tab_indices = {
            "Suggestions": 0,
            "Bugs": 1,
            "Tests": 2,
            "Performance": 3
        }

     def update_suggestions(self, analysis, bugs=None, tests=None):
        """Update suggestions with AI-powered analysis"""
        try:
            # Process code with ML models
            if isinstance(analysis, dict):
                code = analysis.get('code', '')
                cursor_position = analysis.get('cursor_position', None)
                
                # Generate embeddings
                embeddings = self.code_embeddings.generate(code)
                
                # Get ML predictions
                ml_predictions = self.ml_predictor.predict_realtime(code, {
                    'embeddings': embeddings,
                    'cursor_position': cursor_position
                })
                
                # Get pattern analysis
                patterns = self.pattern_recognizer.analyze_realtime(code, cursor_position)
                
                # Update suggestions with ML insights
                self._update_suggestions_tab(ml_predictions)
                self._update_bugs_tab(ml_predictions, patterns)
                self._update_tests_tab(ml_predictions)
                self._update_performance_tab(ml_predictions)
                
            else:
                logger.warning("Invalid analysis format received")
                
        except Exception as e:
            logger.error(f"Suggestion update error: {str(e)}")

    def _update_suggestions_tab(self, ml_predictions):
        """Update suggestions with ML-powered insights"""
        suggestions = []
        
        # Add code completions
        if 'completions' in ml_predictions:
            suggestions.extend(ml_predictions['completions'])
            
        # Add refactoring suggestions
        if 'refactoring' in ml_predictions:
            suggestions.extend(ml_predictions['refactoring'])
            
        # Add best practices
        if 'best_practices' in ml_predictions:
            suggestions.extend(ml_predictions['best_practices'])
            
        self._update_text(self.suggestions_text, suggestions)
        self.update_tab_counter("Suggestions", len(suggestions))

    def _update_bugs_tab(self, ml_predictions, patterns):
        """Update bugs tab with ML-detected issues"""
        bugs = []
        
        # Add ML-detected bugs
        if 'potential_bugs' in ml_predictions:
            bugs.extend(ml_predictions['potential_bugs'])
            
        # Add pattern-based issues
        if 'issues' in patterns:
            bugs.extend(patterns['issues'])
            
        self._update_text(self.bugs_text, bugs)
        self.update_tab_counter("Bugs", len(bugs))

    def _update_performance_tab(self, ml_predictions):
        """Update performance insights"""
        insights = []
        
        if 'performance' in ml_predictions:
            perf_data = ml_predictions['performance']
            
            # Add complexity analysis
            if 'complexity' in perf_data:
                insights.append(f"Complexity Score: {perf_data['complexity']:.2f}")
                
            # Add optimization suggestions
            if 'optimizations' in perf_data:
                insights.extend(perf_data['optimizations'])
                
        self._update_text(self.performance_text, insights)
        self.update_tab_counter("Performance", len(insights))
        
    def _create_suggestions_tab(self, title):
        """Create a new tab for suggestions"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        
        # Create Text widget with scrollbar
        text_widget = tk.Text(frame, wrap=tk.WORD, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        return text_widget

    def clear_all_tabs(self):
        """Clear all suggestion tabs"""
        self._update_text(self.suggestions_text, [])
        self._update_text(self.bugs_text, [])
        self._update_text(self.tests_text, [])
        
    def update_tab_counter(self, tab_name, count):
        """Update the tab title with item count"""
        if tab_name in self.tab_indices:
            index = self.tab_indices[tab_name]
            self.notebook.tab(index, text=f"{tab_name} ({count})")

    def _update_text(self, text_widget, items):
        """Update a text widget with new items"""
        text_widget.config(state=tk.NORMAL)
        text_widget.delete('1.0', tk.END)
        for item in items:
            text_widget.insert(tk.END, f"• {item}\n")
        text_widget.config(state=tk.DISABLED)

    def apply_theme(self, theme_name):
        """Apply theme to suggestions panel"""
        theme = self.themes.get_theme(theme_name)
        # Apply theme to all text widgets
        for text_widget in [self.suggestions_text, self.bugs_text, self.tests_text]:
            text_widget.configure(
                background=theme['background'],
                foreground=theme['foreground'],
                insertbackground=theme['cursor'],
                selectbackground=theme['selection_bg'],
                selectforeground=theme['foreground']
            )

    def update_suggestions(self, analysis, bugs=None, tests=None):
        """Update the suggestions panel with new suggestions"""
        # Update suggestions tab
        self.suggestions_text.config(state=tk.NORMAL)
        self.suggestions_text.delete(1.0, tk.END)
    
        if isinstance(analysis, dict) and 'suggestions' in analysis:
            suggestions = analysis['suggestions']
        elif isinstance(analysis, list):
            suggestions = analysis
        else:
            suggestions = []
        
        for suggestion in suggestions:
            self.suggestions_text.insert(tk.END, f"• {suggestion}\n")
        self.suggestions_text.config(state=tk.DISABLED)
    
    # Update bugs tab if provided
        if bugs:
            self.bugs_text.config(state=tk.NORMAL)
            self.bugs_text.delete(1.0, tk.END)
            for bug in bugs:
                self.bugs_text.insert(tk.END, f"• {bug}\n")
            self.bugs_text.config(state=tk.DISABLED)
            self.update_tab_counter("Bugs", len(bugs))
    
    # Update tests tab if provided
        if tests:
            self.tests_text.config(state=tk.NORMAL)
            self.tests_text.delete(1.0, tk.END) 
            for test in tests:
                self.tests_text.insert(tk.END, f"• {test}\n")
            self.tests_text.config(state=tk.DISABLED)
            self.update_tab_counter("Tests", len(tests))
    
    # Update counters
        self.update_tab_counter("Suggestions", len(suggestions))