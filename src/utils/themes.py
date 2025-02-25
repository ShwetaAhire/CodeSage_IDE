from typing import Dict, Any
from tkinter import ttk, Text, Listbox, Menu
from ..ai import MLPredictor, CodeEmbeddings
import logging


class IDEThemes:  
    def __init__(self):
        self.ml_predictor = MLPredictor()
        self.code_embeddings = CodeEmbeddings()
        self.themes = {
            'Default': {
                'background': '#ffffff',
                'foreground': '#2e3440',
                'keywords': '#0000FF',
                'strings': '#008000',
                'comments': '#808080',
                'functions': '#800080',
                'line_numbers_bg': '#f0f0f0',
                'line_numbers_fg': '#606060',
                'selection_bg': '#add6ff',
                'cursor': '#000000',
                'menu_bg': '#f0f0f0',
                'menu_fg': '#000000',
                'selection_fg': '#000000',  # Added
                'button_bg': '#f0f0f0',  # Added
                'button_fg': '#000000',  # Added
                'border': '#d4d4d4', # Added
                'button_active_bg': '#add6ff',  # Added
                'button_active_fg': '#000000',  # Added
                'border': '#d4d4d4'
            },
            'Dark': {
                'background': '#1e1e1e',
                'foreground': '#d4d4d4',
                'keywords': '#569cd6',
                'strings': '#ce9178',
                'comments': '#6a9955',
                'functions': '#dcdcaa',
                'line_numbers_bg': '#252526',
                'line_numbers_fg': '#858585',
                'selection_bg': '#264f78',
                'cursor': '#ffffff',
                'menu_bg': '#333333',
                'menu_fg': '#ffffff',
                'selection_fg': '#ffffff',  # Added
                'button_bg': '#333333',  # Added
                'button_fg': '#ffffff',  # Added
                'button_active_bg': '#264f78',  # Added
                'button_active_fg': '#ffffff',  # Added
                'border': '#474747'  # Added
            },
            'Monokai': {
                'background': '#272822',
                'foreground': '#f8f8f2',
                'keywords': '#f92672',
                'strings': '#e6db74',
                'comments': '#75715e',
                'functions': '#a6e22e',
                'line_numbers_bg': '#1e1f1c',
                'line_numbers_fg': '#90908a',
                'selection_bg': '#49483e',
                'cursor': '#f8f8f0',
                'menu_bg': '#1e1f1c',
                'menu_fg': '#f8f8f2',
                'selection_fg': '#f8f8f2',  # Added
                'button_bg': '#1e1f1c',  # Added
                'button_fg': '#f8f8f2',  # Added
                'button_active_bg': '#49483e',  # Added
                'button_active_fg': '#f8f8f2',  # Added
                'border': '#3e3d32'  # Added
            },
            'Nord': {
                'background': '#2e3440',
                'foreground': '#d8dee9',
                'keywords': '#81a1c1',
                'strings': '#a3be8c',
                'comments': '#616e88',
                'functions': '#88c0d0',
                'line_numbers_bg': '#3b4252',
                'line_numbers_fg': '#616e88',
                'selection_bg': '#434c5e',
                'cursor': '#d8dee9',
                'menu_bg': '#3b4252',
                'menu_fg': '#d8dee9',
                # ... existing colors ...
                'selection_fg': '#d8dee9',  # Added
                'button_bg': '#3b4252',  # Added
                'button_fg': '#d8dee9',  # Added
                'button_active_bg': '#434c5e',  # Added
                'button_active_fg': '#d8dee9',  # Added
                'border': '#4c566a'  # Added
            },
            'Solarized': {
                'background': '#002b36',
                'foreground': '#839496',
                'keywords': '#859900',
                'strings': '#2aa198',
                'comments': '#586e75',
                'functions': '#268bd2',
                'line_numbers_bg': '#073642',
                'line_numbers_fg': '#586e75',
                'selection_bg': '#073642',
                'cursor': '#839496',
                'menu_bg': '#073642',
                'menu_fg': '#839496',
                'selection_fg': '#839496',  # Added
                'button_bg': '#073642',  # Added
                'button_fg': '#839496',  # Added
                'button_active_bg': '#073642',  # Added
                'button_active_fg': '#93a1a1',  # Added
                'border': '#586e75'  # Added
            }
        }

        # Add AI-enhanced theme features
        self.syntax_cache = {}
        self.context_themes = {}


    def analyze_code_context(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code context for theme optimization"""
        try:
            embeddings = self.code_embeddings.generate(code)
            context = {
                'language': language,
                'embeddings': embeddings,
                'code_length': len(code)
            }
            
            return self.ml_predictor.analyze_visual_context(context)
        except Exception as e:
            logging.error(f"Theme context analysis error: {str(e)}")
            return {}

    def get_optimized_theme(self, code: str, language: str, base_theme: str) -> Dict[str, str]:
        """Get AI-optimized theme based on code context"""
        try:
            context_analysis = self.analyze_code_context(code, language)
            base_colors = self.get_theme(base_theme)
            
            if not context_analysis:
                return base_colors
                
            # Adjust colors based on ML recommendations
            optimized_colors = base_colors.copy()
            if 'color_adjustments' in context_analysis:
                for key, adjustment in context_analysis['color_adjustments'].items():
                    if key in optimized_colors:
                        optimized_colors[key] = self._adjust_color(
                            optimized_colors[key],
                            adjustment
                        )
                        
            return optimized_colors
            
        except Exception:
            return self.get_theme(base_theme)

    def _adjust_color(self, color: str, adjustment: Dict[str, float]) -> str:
        """Adjust color based on ML recommendations"""
        try:
            # Color adjustment logic here
            return color  # Placeholder
        except Exception:
            return color

    def apply_theme(self, widget: Any, theme_name: str, code_context: str = None) -> None:
        """Apply theme with AI optimizations"""
        if code_context:
            language = self._detect_language(code_context)
            theme_colors = self.get_optimized_theme(code_context, language, theme_name)
        else:
            theme_colors = self.get_theme(theme_name)
            
        # Rest of existing apply_theme implementation...

    def _detect_language(self, code: str) -> str:
        """Detect code language using ML"""
        try:
            embeddings = self.code_embeddings.generate(code)
            prediction = self.ml_predictor.predict_language(embeddings)
            return prediction.get('language', 'Python')
        except Exception:
            return 'Python'

    def apply_syntax_highlighting(self, text_widget: Text, theme_name: str, code: str = None) -> None:
        """Apply AI-enhanced syntax highlighting"""
        if code:
            # Get ML-optimized syntax colors
            language = self._detect_language(code)
            embeddings = self.code_embeddings.generate(code)
            syntax_analysis = self.ml_predictor.analyze_syntax_patterns(code, embeddings)
            
            # Apply optimized highlighting
            if syntax_analysis and 'syntax_patterns' in syntax_analysis:
                self._apply_advanced_highlighting(
                    text_widget,
                    syntax_analysis['syntax_patterns'],
                    theme_name
                )
            else:
                # Fallback to standard highlighting
                super().apply_syntax_highlighting(text_widget, theme_name)
        else:
            super().apply_syntax_highlighting(text_widget, theme_name)

    def _apply_advanced_highlighting(self, text_widget: Text, patterns: Dict, theme_name: str):
        """Apply advanced syntax highlighting based on ML patterns"""
        colors = self.get_syntax_colors(theme_name)
        for pattern_type, pattern_info in patterns.items():
            if pattern_type in colors:
                text_widget.tag_configure(
                    pattern_type,
                    foreground=colors[pattern_type],
                    **pattern_info.get('style', {})
                ) 

    def get_theme(self, theme_name: str) -> Dict[str, str]:
        """Get theme colors by name"""
        return self.themes.get(theme_name, self.themes['Default'])

    def apply_theme(self, widget: Any, theme_name: str) -> None:
        """Apply theme to a widget and all its children"""
        theme = self.get_theme(theme_name)
        style = ttk.Style()
        
        # Configure ttk styles
        style.configure('TFrame', background=theme['background'])
        style.configure('TLabel', 
            background=theme['background'],
            foreground=theme['foreground']
        )
        style.configure('TNotebook', 
            background=theme['background'],
            tabmargins=[2, 5, 2, 0]
        )
        style.configure('TNotebook.Tab',
            background=theme['menu_bg'],
            foreground=theme['menu_fg'],
            padding=[10, 2]
        )
        style.map('TNotebook.Tab',
            background=[('selected', theme['selection_bg'])],
            foreground=[('selected', theme['foreground'])]
        )
        
        # Debug panel specific styles
        style.configure('Debug.TButton',
            background=theme.get('button_bg', theme['menu_bg']),
            foreground=theme.get('button_fg', theme['menu_fg']),
            padding=1,
            borderwidth=1,
            relief='raised'
        )

        style.map('Debug.TButton',
            background=[('active', theme.get('button_active_bg', theme['selection_bg']))],
            foreground=[('active', theme.get('button_active_fg', theme['foreground']))]
        )

        style.configure('Treeview',
            background=theme['background'],
            foreground=theme['foreground'],
            fieldbackground=theme['background']
        )
        style.map('Treeview',
            background=[('selected', theme['selection_bg'])],
            foreground=[('selected', theme['foreground'])]
        )
        
        # Apply theme to specific widget types
        if isinstance(widget, ttk.Widget):
            pass  # Style already applied via ttk.Style
        elif isinstance(widget, Text):
            widget.configure(
                background=theme['background'],
                foreground=theme['foreground'],
                insertbackground=theme['cursor'],
                selectbackground=theme['selection_bg'],
                selectforeground=theme['foreground']
            )
        elif isinstance(widget, Menu):
            widget.configure(
                background=theme['menu_bg'],
                foreground=theme['menu_fg'],
                activebackground=theme['selection_bg'],
                activeforeground=theme['foreground']
            )

        # Apply theme to all children widgets recursively
        for child in widget.winfo_children():
            self.apply_theme(child, theme_name)

    def apply_syntax_highlighting(self, text_widget: Text, theme_name: str) -> None:
        """Apply syntax highlighting colors to a text widget"""
        colors = self.get_syntax_colors(theme_name)
        text_widget.tag_configure('keyword', foreground=colors['keyword'])
        text_widget.tag_configure('string', foreground=colors['string'])
        text_widget.tag_configure('comment', foreground=colors['comment'])
        text_widget.tag_configure('function', foreground=colors['function'])

    def get_syntax_colors(self, theme_name: str) -> Dict[str, str]:
        """Get syntax highlighting colors for a theme"""
        theme = self.get_theme(theme_name)
        return {
            'keyword': theme['keywords'],
            'string': theme['strings'],
            'comment': theme['comments'],
            'function': theme['functions']
        }

    def get_menu_colors(self, theme_name: str) -> Dict[str, str]:
        """Get menu colors for a theme"""
        theme = self.get_theme(theme_name)
        return {
            'background': theme['menu_bg'],
            'foreground': theme['menu_fg']
        }

    def get_line_numbers_colors(self, theme_name: str) -> Dict[str, str]:
        """Get line numbers colors for a theme"""
        theme = self.get_theme(theme_name)
        return {
            'background': theme['line_numbers_bg'],
            'foreground': theme['line_numbers_fg']
        }

    def apply_ide_theme(self, root_widget: Any, theme_name: str) -> None:
        """Apply theme to entire IDE"""
        theme = self.get_theme(theme_name)
        root_widget.configure(background=theme['background'])
        self.apply_theme(root_widget, theme_name)

    @property
    def available_themes(self) -> list:
        """Get list of available themes"""
        return list(self.themes.keys())