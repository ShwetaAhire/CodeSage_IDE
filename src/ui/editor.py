import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import re
import keyword
import builtins
from src.utils.themes import IDEThemes
from ..ai import CodeEmbeddings, MLPredictor, PatternRecognizer, ModelConfig
import logging

logger = logging.getLogger(__name__)

class CodeEditor(ttk.Frame):
    def __init__(self, parent, code_analyzer, bug_detector):
        super().__init__(parent)
        self.code_embeddings = CodeEmbeddings()
        self.ml_predictor = MLPredictor()
        self.pattern_recognizer = PatternRecognizer()
        self.config = ModelConfig()
        self.code_analyzer = code_analyzer
        self.bug_detector = bug_detector
        self.suggestions_callback = None
        self.current_file = None
        self.themes = IDEThemes()
        self._modified = False
        self.breakpoints = set()
        
        # Create main editor container first
        editor_container = ttk.Frame(self)
        editor_container.pack(fill=tk.BOTH, expand=True)
        
        # Create text editor with line numbers
        self.line_numbers = tk.Text(editor_container, width=4, padx=3, takefocus=0, border=0,
                                  background='lightgray', state='disabled')
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # Create editor frame for proper scrollbar placement
        editor_frame = ttk.Frame(editor_container)
        editor_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create editor before configuring tags
        self.editor = ScrolledText(editor_frame, wrap=tk.NONE, undo=True)
        self.editor.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Configure tags for syntax highlighting and debugging
        self.setup_tags()
        self.editor.tag_configure('breakpoint', background='#FF9999')
        self.editor.tag_configure('current_line', background='#E8E8FF')
        
        # Horizontal scrollbar
        h_scroll = ttk.Scrollbar(editor_frame, orient=tk.HORIZONTAL, command=self.editor.xview)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.editor.configure(xscrollcommand=h_scroll.set)
        
        # Create toolbar last
        self.create_toolbar()
        
        # Bind events
        self.editor.bind('<KeyRelease>', self.on_key_release)
        self.editor.bind('<Return>', self.on_return)
        self.editor.bind('<Tab>', self.on_tab)
        self.editor.bind('<<Modified>>', self._on_modify)
        self.editor.bind('<Button-3>', self.show_context_menu)

         # Initialize syntax patterns
        self.setup_syntax_patterns()
        
        # Apply default theme
        self.apply_theme('Default')
        self.editor.edit_modified(False)

    def check_code(self):
        """Real-time code analysis with AI integration"""
        try:
            code = self.get('1.0', tk.END)
            cursor_position = self.get_cursor_position()
        
        # Real-time code embeddings
            embeddings = self.code_embeddings.generate(code)
        
        # Get ML predictions with context
            context = {
                'cursor_position': cursor_position,
                'language': self.language_var.get(),
                'file_type': self.current_file.split('.')[-1] if self.current_file else 'py'
            }
            ml_analysis = self.ml_predictor.predict_realtime(code, context)
        
        # Pattern recognition
            pattern_analysis = self.pattern_recognizer.analyze_realtime(code, cursor_position)
        
        # Real-time bug detection
            bugs = self.bug_detector.detect_bugs_realtime(code, cursor_position)
        
        # Combine all analyses
            analysis_results = {
                'code_quality': ml_analysis.get('quality_score', 0.0),
                'suggestions': ml_analysis.get('immediate_suggestions', []),
                'patterns': pattern_analysis.get('patterns', []),
                'improvements': self._get_code_improvements(code, ml_analysis),
                'bugs': bugs,
                'security_issues': ml_analysis.get('security_issues', [])
            }
        
        # Apply real-time highlighting
            self._apply_ai_highlights(code, ml_analysis, pattern_analysis)
        
        # Update suggestions
            if self.suggestions_callback:
                self.suggestions_callback(analysis_results)
            
        except Exception as e:
        logger.error(f"Real-time analysis error: {str(e)}")

    def on_key_release(self, event):
        """Enhanced key release handler with real-time AI analysis"""
        self.update_line_numbers()
        self.highlight_syntax()
    
    # Debounced real-time analysis
        if hasattr(self, '_check_timer'):
            self.after_cancel(self._check_timer)
        self._check_timer = self.after(500, self.check_code)  # 500ms delay for performance

    def _apply_ai_highlights(self, code: str, ml_analysis: dict, pattern_analysis: dict):
        """Apply AI-based code highlighting"""
        try:
            # Clear previous AI highlights
            self.editor.tag_remove('ai_warning', '1.0', tk.END)
            self.editor.tag_remove('ai_suggestion', '1.0', tk.END)
            self.editor.tag_remove('ai_optimization', '1.0', tk.END)
            
            # Apply new highlights based on AI analysis
            for issue in ml_analysis.get('issues', []):
                if 'line' in issue:
                    self.editor.tag_add('ai_warning', 
                                      f"{issue['line']}.0", 
                                      f"{issue['line']}.end")
                    
            for pattern in pattern_analysis.get('patterns', []):
                if 'start' in pattern and 'end' in pattern:
                    self.editor.tag_add('ai_suggestion',
                                      f"{pattern['start']}.0",
                                      f"{pattern['end']}.end")
                    
        except Exception as e:
            logger.error(f"Highlight application error: {str(e)}")

    def _get_code_improvements(self, code: str, ml_analysis: dict) -> List[str]:
        """Get AI-suggested code improvements"""
        try:
            improvements = []
            if ml_analysis.get('quality_score', 1.0) < 0.7:
                improvements.extend(ml_analysis.get('suggested_improvements', []))
                
            # Add performance optimization suggestions
            if ml_analysis.get('complexity_score', 0.0) > 0.7:
                improvements.append("Consider optimizing code complexity")
                
            return improvements
            
        except Exception as e:
            logger.error(f"Code improvement analysis error: {str(e)}")
            return []

    def get_cursor_position(self) -> Dict[str, int]:
        """Get current cursor position details"""
        try:
            cursor_index = self.editor.index(tk.INSERT)
            line, col = map(int, cursor_index.split('.'))
            return {
                'line': line,
                'column': col,
                'absolute': len(self.editor.get('1.0', cursor_index))
            }
        except Exception as e:
            logger.error(f"Cursor position error: {str(e)}")
            return {'line': 1, 'column': 0, 'absolute': 0}

    def setup_tags(self):
        """Setup enhanced tag configurations with AI highlights"""
        # Existing tags remain the same...
        
        # Add AI-specific tags
        self.editor.tag_configure('ai_warning', background='#FFE8E8')
        self.editor.tag_configure('ai_suggestion', background='#E8FFE8')
        self.editor.tag_configure('ai_optimization', background='#E8E8FF')
        

    def setup_syntax_patterns(self):
        """Initialize syntax highlighting patterns for all supported languages"""
        self.syntax_patterns = {
            'Python': {
                'keywords': set(keyword.kwlist),
                'builtins': set(dir(builtins)),
                'strings': r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|".*?"|\'.*?\')',
                'comments': r'(#.*$)',
                'numbers': r'\b(?:0[xX][0-9a-fA-F]+|0[bB][01]+|\d+\.?\d*|\.\d+)\b',
                'functions': r'\bdef\s+([a-zA-Z_]\w*)',
                'classes': r'\bclass\s+([a-zA-Z_]\w*)',
                'decorators': r'@([a-zA-Z_]\w*)'
            },
            'JavaScript': {
                'keywords': set(['async', 'await', 'break', 'case', 'catch', 'class', 'const', 
                               'continue', 'debugger', 'default', 'delete', 'do', 'else', 'export',
                               'extends', 'finally', 'for', 'function', 'if', 'import', 'in',
                               'instanceof', 'new', 'return', 'super', 'switch', 'this', 'throw',
                               'try', 'typeof', 'var', 'void', 'while', 'with', 'yield', 'let']),
                'strings': r'(`[\s\S]*?`|".*?"|\'.*?\')',
                'comments': r'(\/\/.*$|\/\*[\s\S]*?\*\/)',
                'numbers': r'\b(?:0[xX][0-9a-fA-F]+|0[bB][01]+|\d+\.?\d*|\.\d+)\b',
                'functions': r'\bfunction\s+([a-zA-Z_]\w*)',
                'classes': r'\bclass\s+([a-zA-Z_]\w*)',
                'methods': r'\b([a-zA-Z_]\w*)\s*\('
            },
            'HTML': {
                'tags': r'<[^>]+>',
                'attributes': r'\b([a-zA-Z_]\w*)\s*=\s*("[^"]*"|\'[^\']*\')',
                'strings': r'("[^"]*"|\'[^\']*\')',
                'comments': r'<!--[\s\S]*?-->',
                'entities': r'&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;'
            },
            'CSS': {
                'selectors': r'[^{}\n]+(?=\s*\{)',
                'properties': r'([a-zA-Z-]+)\s*:',
                'values': r':\s*([^;]+);',
                'comments': r'\/\*[\s\S]*?\*\/',
                'colors': r'#[0-9a-fA-F]{3,6}|\b(?:rgb|hsl)a?\([^)]*\)',
                'units': r'\b\d+(?:px|em|rem|%|vh|vw|pt|pc|in|cm|mm|ex|ch)\b'
            }
        }

    def _highlight_python(self, content):
        """Apply Python-specific syntax highlighting"""
        patterns = self.syntax_patterns['Python']
        
        # Clear existing tags
        for tag in ['keyword', 'builtin', 'string', 'comment', 'number', 
                   'function', 'class', 'decorator']:
            self.editor.tag_remove(tag, '1.0', 'end')
        
        # Apply highlighting for each pattern
        for keyword in patterns['keywords']:
            self._highlight_words(keyword, 'keyword')
            
        for builtin in patterns['builtins']:
            self._highlight_words(builtin, 'builtin')
            
        self._highlight_regex(patterns['strings'], 'string')
        self._highlight_regex(patterns['comments'], 'comment')
        self._highlight_regex(patterns['numbers'], 'number')
        self._highlight_regex(patterns['functions'], 'function')
        self._highlight_regex(patterns['classes'], 'class')
        self._highlight_regex(patterns['decorators'], 'decorator')

    def _highlight_words(self, word, tag):
        """Highlight specific words with given tag"""
        start = '1.0'
        while True:
            start = self.editor.search(r'\b' + word + r'\b', start, 'end', regexp=True)
            if not start:
                break
            end = f"{start}+{len(word)}c"
            self.editor.tag_add(tag, start, end)
            start = end

    def _highlight_regex(self, pattern, tag):
        """Highlight regex pattern with given tag"""
        content = self.editor.get('1.0', 'end-1c')
        for match in re.finditer(pattern, content, re.MULTILINE):
            start = f"1.0+{match.start()}c"
            end = f"1.0+{match.end()}c"
            self.editor.tag_add(tag, start, end)

    def show_context_menu(self, event):
        """Show context menu on right click"""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Cut", command=lambda: self.editor.event_generate("<<Cut>>"))
        menu.add_command(label="Copy", command=lambda: self.editor.event_generate("<<Copy>>"))
        menu.add_command(label="Paste", command=lambda: self.editor.event_generate("<<Paste>>"))
        menu.add_separator()
        menu.add_command(label="Select All", command=lambda: self.editor.tag_add('sel', '1.0', 'end'))
        menu.post(event.x_root, event.y_root)

    def toggle_breakpoint(self, line_number):
        """Toggle breakpoint at the specified line"""
        if line_number in self.breakpoints:
            self.breakpoints.remove(line_number)
            self.editor.tag_remove('breakpoint', f"{line_number}.0", f"{line_number}.end")
        else:
            self.breakpoints.add(line_number)
            self.editor.tag_add('breakpoint', f"{line_number}.0", f"{line_number}.end")
            
    def get_breakpoints(self):
        """Get the set of active breakpoints"""
        return self.breakpoints.copy()

    def highlight_current_line(self, line_number):
        """Highlight the current debugging line"""
        self.editor.tag_remove('current_line', '1.0', tk.END)
        self.editor.tag_add('current_line', f"{line_number}.0", f"{line_number}.end")
        
        # Ensure the line is visible
        self.editor.see(f"{line_number}.0")
    
    def on_key_release(self, event):
        """Handle key release events"""
        self.update_line_numbers()
        self.highlight_syntax()
        self.check_code()
        
    def on_return(self, event):
        """Handle return key press"""
        current_line = self.editor.get("insert linestart", "insert")
        indentation = re.match(r'^\s*', current_line).group()
        if current_line.rstrip().endswith(':'):
            indentation += '    '
        self.editor.insert('insert', '\n' + indentation)
        return 'break'
        
    def on_tab(self, event):
        """Handle tab key press"""
        self.editor.insert('insert', '    ')
        return 'break'
        
    def update_line_numbers(self):
        """Update the line numbers display"""
        editor_lines = self.editor.get('1.0', tk.END).split('\n')
        line_count = len(editor_lines)
        line_numbers_text = '\n'.join(str(i) for i in range(1, line_count + 1))
        self.line_numbers.config(state='normal')
        self.line_numbers.delete('1.0', tk.END)
        self.line_numbers.insert('1.0', line_numbers_text)
        self.line_numbers.config(state='disabled')
        
    def highlight_syntax(self):
        """Highlight syntax based on current language"""
        if not hasattr(self, 'current_syntax'):
            return
            
        content = self.editor.get('1.0', tk.END)
        for tag in ['keyword', 'string', 'comment']:
            self.editor.tag_remove(tag, '1.0', tk.END)
            
        if hasattr(self, 'current_syntax'):
            self.highlight_pattern(self.current_syntax['keywords'], 'keyword')
            self.highlight_pattern(self.current_syntax['strings'], 'string')
            self.highlight_pattern(self.current_syntax['comments'], 'comment')
        
    def highlight_pattern(self, pattern, tag):
        """Apply highlighting pattern"""
        content = self.editor.get('1.0', tk.END)
        for match in re.finditer(pattern, content, re.MULTILINE):
            start = f"1.0+{match.start()}c"
            end = f"1.0+{match.end()}c"
            self.editor.tag_add(tag, start, end)

    
    def highlight_syntax(self):
        """Enhanced syntax highlighting based on current language"""
        if not hasattr(self, 'syntax_patterns'):
            return
            
        language = self.language_var.get()
        if language not in self.syntax_patterns:
            return
            
        # Clear all existing tags
        for tag in ['keyword', 'builtin', 'string', 'comment', 'number', 
                   'function', 'class', 'decorator', 'tag', 'attribute', 
                   'selector', 'property', 'value']:
            self.editor.tag_remove(tag, '1.0', 'end')
        
        # Apply language-specific highlighting
        if language == 'Python':
            self._highlight_python(self.editor.get('1.0', 'end-1c'))
        elif language == 'JavaScript':
            self._highlight_javascript(self.editor.get('1.0', 'end-1c'))
        elif language == 'HTML':
            self._highlight_html(self.editor.get('1.0', 'end-1c'))
        elif language == 'CSS':
            self._highlight_css(self.editor.get('1.0', 'end-1c'))
            
        
    def create_toolbar(self):
        """Enhanced toolbar with language and theme selection"""
        toolbar = ttk.Frame(self)
        toolbar.pack(fill='x', pady=2)
        
        # File location label
        self.file_label = ttk.Label(toolbar, text="<unsaved>")
        self.file_label.pack(side='left', padx=5)
        
        # Language selector
        language_frame = ttk.Frame(toolbar)
        language_frame.pack(side='right', padx=5)
        ttk.Label(language_frame, text="Language:").pack(side='left')
        languages = ['Python', 'JavaScript', 'HTML', 'CSS']
        self.language_var = tk.StringVar(value='Python')
        language_menu = ttk.OptionMenu(language_frame, self.language_var, 'Python', *languages)
        language_menu.pack(side='left', padx=5)
        
        # Theme selector
        theme_frame = ttk.Frame(toolbar)
        theme_frame.pack(side='right', padx=5)
        ttk.Label(theme_frame, text="Theme:").pack(side='left')
        self.theme_var = tk.StringVar(value='Default')
        # Use a default list of themes
        themes = ['Default', 'Dark', 'Light', 'Monokai']
        theme_menu = ttk.OptionMenu(theme_frame, self.theme_var, 'Default', *themes)
        theme_menu.pack(side='left', padx=5)
        
        # Bind change events
        self.language_var.trace('w', lambda *args: self.on_language_change(None))
        self.theme_var.trace('w', lambda *args: self.on_theme_change(None))

    def on_language_change(self, event):
        """Handle language change and update syntax highlighting"""
        language = self.language_var.get()
        
        # Update file extension for save dialogs
        self.file_extensions = {
            'Python': '.py',
            'JavaScript': '.js',
            'Java': '.java',
            'C++': '.cpp',
            'HTML': '.html',
            'CSS': '.css'
        }
        
        # Update syntax highlighting patterns
        syntax_patterns = {
            'Python': {
                'keywords': r'\b(def|class|if|else|while|for|return|import|from)\b',
                'strings': r'(\".*?\"|\'.*?\')',
                'comments': r'(#.*$)'
            },
            'JavaScript': {
                'keywords': r'\b(function|var|let|const|if|else|while|for|return|import|class)\b',
                'strings': r'(\".*?\"|\'.*?\'|`.*?`)',
                'comments': r'(\/\/.*$|\/\*[\s\S]*?\*\/)'
            },
            'Java': {
                'keywords': r'\b(class|public|private|protected|void|int|String|if|else|while|for|return)\b',
                'strings': r'(\".*?\")',
                'comments': r'(\/\/.*$|\/\*[\s\S]*?\*\/)'
            },
            'C++': {
                'keywords': r'\b(class|struct|void|int|char|if|else|while|for|return)\b',
                'strings': r'(\".*?\")',
                'comments': r'(\/\/.*$|\/\*[\s\S]*?\*\/)'
            },
            'HTML': {
                'keywords': r'(<[^>]*>)',
                'strings': r'(\".*?\")',
                'comments': r'(<!--[\s\S]*?-->)'
            },
            'CSS': {
                'keywords': r'([{}\[\];:])',
                'strings': r'(\".*?\")',
                'comments': r'(\/\*[\s\S]*?\*\/)'
            }
        }
        
        if language in syntax_patterns:
            self.current_syntax = syntax_patterns[language]
            self.highlight_syntax()
        
    def _on_modify(self, event=None):
        if self.editor.edit_modified():
            self._modified = True
            self.editor.edit_modified(False)
            
    def is_modified(self):
        return self._modified
        
    def set_modified(self, state):
        self._modified = state
        self.editor.edit_modified(False)
        
    def set_current_file(self, path):
        self.current_file = path
        self._modified = False
        self.file_label.config(text=path if path else "<unsaved>")
        
    def insert_text(self, content):
        self.editor.delete('1.0', tk.END)
        self.editor.insert('1.0', content)
        self._modified = False
        self.editor.edit_modified(False)

    def apply_theme(self, theme_name):
        theme = self.themes.get_theme(theme_name)
        self.theme_var.set(theme_name)  # Update theme variable
        
        # Apply to editor
        self.editor.configure(
            background=theme['background'],
            foreground=theme['foreground'],
            insertbackground=theme['cursor'],
            selectbackground=theme['selection_bg'],
            selectforeground=theme['foreground']
        )
        
        # Apply to line numbers
        line_colors = self.themes.get_line_numbers_colors(theme_name)
        self.line_numbers.configure(
            background=line_colors['background'],
            foreground=line_colors['foreground']
        )
        
        # Apply to file label (using ttk style instead of direct configuration)
        style = ttk.Style()
        style.configure('Custom.TLabel',
            background=theme['background'],
            foreground=theme['foreground']
        )
        self.file_label.configure(style='Custom.TLabel')
        
        # Update syntax highlighting colors
        syntax_colors = self.themes.get_syntax_colors(theme_name)
        for tag, color in syntax_colors.items():
            self.editor.tag_configure(tag, foreground=color)
            
    def on_theme_change(self, event):
        self.apply_theme(self.theme_var.get())
        
    def setup_tags(self):
        """Setup initial tag configurations"""
        self.editor.tag_configure('keyword', foreground='#FF6B6B')
        self.editor.tag_configure('builtin', foreground='#C678DD')
        self.editor.tag_configure('string', foreground='#98C379')
        self.editor.tag_configure('comment', foreground='#5C6370')
        self.editor.tag_configure('number', foreground='#D19A66')
        self.editor.tag_configure('function', foreground='#61AFEF')
        self.editor.tag_configure('class', foreground='#E5C07B')
        self.editor.tag_configure('decorator', foreground='#56B6C2')

    def get(self, start, end):
        """Get text content from the editor"""
        return self.editor.get(start, end)
        
    def delete(self, start, end):
        """Delete text content from the editor"""
        return self.editor.delete(start, end)
        
    def insert(self, index, chars):
        """Insert text content into the editor"""
        return self.editor.insert(index, chars)
        
    def check_code(self):
        """Check code and update suggestions"""
        code = self.get('1.0', tk.END)
        analysis = self.code_analyzer.analyze(code)
        bugs = self.bug_detector.detect(code) 
    
        if self.suggestions_callback:
            self.suggestions_callback(analysis, bugs)
            
    def bind_suggestions_update(self, callback):
        self.suggestions_callback = callback
        
    def get_code(self):
        return self.editor.get('1.0', tk.END)
        
    def clear(self):
        self.editor.delete('1.0', tk.END)
        self._modified = False
        self.editor.edit_modified(False)
        self.current_file = None
        
    def new_file(self):
        if self._modified:
            response = messagebox.askyesnocancel("Save Changes", 
                "Do you want to save changes before creating new file?")
            if response is None:  # Cancel
                return
            if response:  # Yes
                if not self.save_file():
                    return
        self.clear()
        
    def open_file(self):
        if self._modified:
            response = messagebox.askyesnocancel("Save Changes", 
                "Do you want to save changes before opening another file?")
            if response is None:  # Cancel
                return
            if response:  # Yes
                if not self.save_file():
                    return
                    
        file_path = filedialog.askopenfilename(
            filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                self.editor.delete('1.0', tk.END)
                self.editor.insert('1.0', content)
                self.current_file = file_path
                self._modified = False
                self.editor.edit_modified(False)
                self.update_line_numbers()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {str(e)}")
                
    def save_file(self):
        if not self.current_file:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".py",
                filetypes=[("Python files", "*.py"), ("All files", "*.*")])
            if file_path:
                self.current_file = file_path
            else:
                return False
                
        try:
            with open(self.current_file, 'w', encoding='utf-8') as file:
                content = self.editor.get('1.0', tk.END)
                file.write(content)
            self._modified = False
            self.editor.edit_modified(False)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")
            return False
            
    def save_as_file(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if file_path:
            self.current_file = file_path
            return self.save_file()
        return False
        
    def exit_editor(self):
        if self._modified:
            response = messagebox.askyesnocancel("Save Changes", 
                "Do you want to save changes before exiting?")
            if response is None:  # Cancel
                return False
            if response:  # Yes
                if not self.save_file():
                    return False
        return True
        
    def get_current_line_number(self):
        return int(self.editor.index('insert').split('.')[0])
        
    def get_current_file(self):
        return self.current_file or '<unsaved>'

 