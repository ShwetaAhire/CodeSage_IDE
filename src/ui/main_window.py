import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from src.ui.editor import CodeEditor
from src.ui.suggestions_panel import SuggestionsPanel
from src.ui.debug_panel import DebugPanel
from src.core.code_analyzer import CodeAnalyzer
from src.core.bug_detector import BugDetector
from src.utils.themes import IDEThemes
from src.core.ide_analyzer import IDEAnalyzer
from ..ai import CodeEmbeddings, MLPredictor, PatternRecognizer, ModelConfig
import logging

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.code_embeddings = CodeEmbeddings()  # CodeBERT integration
        self.ml_predictor = MLPredictor()        # CodeT5 integration
        self.pattern_recognizer = PatternRecognizer()
        self.config = ModelConfig()
        self.analyzer = IDEAnalyzer()
        self.title("AI Intelligent IDE")
        self.geometry("1400x900")
        
        # Initialize components
        self.code_analyzer = CodeAnalyzer()
        self.bug_detector = BugDetector()
        self.themes = IDEThemes()
        
        # Create menu bar
        self.create_menu()

         # Start background analysis
        self.analyzer.start_background_analysis(
            self.get_current_code,
            self.update_suggestions
        )

        # Create main vertical container
        self.main_vertical = ttk.PanedWindow(self, orient=tk.VERTICAL)
        self.main_vertical.pack(fill=tk.BOTH, expand=True)
        
        # Create horizontal container for editor and suggestions
        self.main_horizontal = ttk.PanedWindow(self.main_vertical, orient=tk.HORIZONTAL)
        self.main_vertical.add(self.main_horizontal)
        
        # Create notebook for multiple editors
        self.editor_notebook = ttk.Notebook(self.main_horizontal)
        self.main_horizontal.add(self.editor_notebook)

        # Create suggestions panel
        self.suggestions = SuggestionsPanel(self.main_horizontal)
        self.main_horizontal.add(self.suggestions)

        # Create debug panel
        self.debug_panel = DebugPanel(self.main_vertical)
        self.main_vertical.add(self.debug_panel, weight=1)

        # Dictionary to store all editors
        self.editors = {}

        # Create initial editor
        self.create_new_editor()

        # Bind tab close event
        self.editor_notebook.bind('<ButtonRelease-3>', self.show_tab_context_menu)
        
        # Apply initial theme
        self.themes.apply_theme(self, 'Default')

    def get_current_code(self):
        """Get current code from editor"""
        if self.current_editor:
            return self.current_editor.get('1.0', tk.END)
        return ""

    def update_suggestions(self, results):
        """Update all suggestion panels with AI insights"""
        try:
            code = self.get_current_code()
            cursor_position = self.current_editor.get_cursor_position() if self.current_editor else None
            
            # Generate embeddings using CodeBERT
            embeddings = self.code_embeddings.generate(code)
            
            # Get ML predictions using CodeT5
            context = {
                'cursor_position': cursor_position,
                'language': self.current_editor.language_var.get() if self.current_editor else 'Python',
                'embeddings': embeddings
            }
            
            ml_analysis = self.ml_predictor.predict_realtime(code, context)
            pattern_analysis = self.pattern_recognizer.analyze_realtime(code, cursor_position)
            
            # Combine all analyses
            combined_results = {
                'suggestions': self._process_ml_suggestions(ml_analysis),
                'bugs': self.bug_detector.detect_bugs_realtime(code, embeddings),
                'tests': self._generate_test_suggestions(code, ml_analysis),
                'performance': ml_analysis.get('performance_insights', []),
                'security': ml_analysis.get('security_analysis', [])
            }
            
            # Update UI with real-time insights
            self.update_suggestion_panel(combined_results['suggestions'])
            self.update_bug_panel(combined_results['bugs'])
            self.update_test_panel(combined_results['tests'])
            
            # Update debug panel with AI insights
            if hasattr(self, 'debug_panel'):
                self.debug_panel.update_debug_state(code, cursor_position, embeddings)
                
        except Exception as e:
            logger.error(f"Real-time analysis error: {str(e)}")

    def _process_ml_suggestions(self, ml_analysis):
        """Process ML model suggestions"""
        suggestions = []
        if 'code_completions' in ml_analysis:
            suggestions.extend(ml_analysis['code_completions'])
        if 'refactoring_suggestions' in ml_analysis:
            suggestions.extend(ml_analysis['refactoring_suggestions'])
        if 'best_practices' in ml_analysis:
            suggestions.extend(ml_analysis['best_practices'])
        return suggestions

    def create_new_editor(self, file_path=None):
        """Create new editor with AI integration"""
        editor = CodeEditor(
            self.editor_notebook,
            self.code_analyzer,
            self.bug_detector,
            ml_predictor=self.ml_predictor,
            pattern_recognizer=self.pattern_recognizer,
            code_embeddings=self.code_embeddings
        )

    def update_suggestion_panel(self, suggestions):
        """Update the code suggestions tab"""
        if not hasattr(self.suggestions, 'suggestions_text'):
            return
            
        self.suggestions.suggestions_text.delete('1.0', tk.END)
        if suggestions:
            for suggestion in suggestions:
                self.suggestions.suggestions_text.insert(tk.END, f"• {suggestion}\n")
                
        # Update tab counter
        count = len(suggestions) if suggestions else 0
        self.suggestions.update_tab_counter("Suggestions", count)

    def update_bug_panel(self, bugs):
        """Update the bug detection tab"""
        if not hasattr(self.suggestions, 'bugs_text'):
            return
            
        self.suggestions.bugs_text.delete('1.0', tk.END)
        if bugs:
            for bug in bugs:
                self.suggestions.bugs_text.insert(tk.END, f"• {bug}\n")
                
        # Update tab counter
        count = len(bugs) if bugs else 0
        self.suggestions.update_tab_counter("Bugs", count)

    def update_test_panel(self, tests):
        """Update the test suggestions tab"""
        if not hasattr(self.suggestions, 'tests_text'):
            return
            
        self.suggestions.tests_text.delete('1.0', tk.END)
        if tests:
            for test in tests:
                self.suggestions.tests_text.insert(tk.END, f"• {test}\n")
                
        # Update tab counter
        count = len(tests) if tests else 0
        self.suggestions.update_tab_counter("Tests", count)

    def cleanup(self):
        """Clean up resources before closing"""
        # Stop background analysis
        if hasattr(self, 'analyzer'):
            self.analyzer.stop_background_analysis()
        
        # Save any unsaved changes
        for editor in self.editors.keys():
            if editor.is_modified():
                self.check_save_changes(editor)
        
        # Destroy the window
        self.quit()
        
    def create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Open", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As", command=self.save_as_file, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Close Tab", command=self.close_current_tab, accelerator="Ctrl+W")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # Language menu
        language_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Language", menu=language_menu)
        languages = ['Python', 'JavaScript', 'Java', 'C++', 'HTML', 'CSS']
        for lang in languages:
            language_menu.add_command(
                label=lang,
                command=lambda l=lang: self.change_language(l)
            )

        # Theme menu
        theme_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Theme", menu=theme_menu)
        for theme in self.themes.available_themes:
            theme_menu.add_command(
                label=theme,
                command=lambda t=theme: self.apply_theme(t)
            )

        # Add keyboard shortcuts
        self.bind_all("<Control-n>", lambda e: self.new_file())
        self.bind_all("<Control-o>", lambda e: self.open_file())
        self.bind_all("<Control-s>", lambda e: self.save_file())
        self.bind_all("<Control-S>", lambda e: self.save_as_file())
        self.bind_all("<Control-w>", lambda e: self.close_current_tab())

    def apply_theme(self, theme_name):
        """Apply theme to all components"""
        # Apply to all editors
        for editor in self.editors.keys():
            editor.apply_theme(theme_name)
        
        # Apply to suggestions panel
        self.suggestions.apply_theme(theme_name)
        
        # Apply to debug panel
        self.debug_panel.apply_theme(theme_name)
    
    def change_language(self, language):
        """Change the current editor's language"""
        if self.current_editor:
            self.current_editor.language_var.set(language)
            self.current_editor.on_language_change(None)
        
    def create_new_editor(self, file_path=None): 
        """Create a new editor tab"""
        editor = CodeEditor(self.editor_notebook, self.code_analyzer, self.bug_detector)
        tab_name = file_path if file_path else "Untitled"
        self.editor_notebook.add(editor, text=self._get_tab_name(tab_name))
        self.editors[editor] = tab_name
        editor.bind_suggestions_update(self.suggestions.update_suggestions)
        self.editor_notebook.select(self.editor_notebook.index('end')-1)
        return editor
        
    def _get_tab_name(self, path):
        """Get shortened name for tab"""
        return path.split('/')[-1] if '/' in path else path.split('\\')[-1] if '\\' in path else path
        
    @property
    def current_editor(self):
        """Get the currently active editor"""
        current = self.editor_notebook.select()
        if current:
            return self.editor_notebook.children[current.split('.')[-1]]
        return None
        
    def show_tab_context_menu(self, event):
        """Show context menu for tabs"""
        clicked_tab = self.editor_notebook.tk.call(self.editor_notebook._w, "identify", "tab", event.x, event.y)
        if clicked_tab is not None:
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="Close", command=lambda: self.close_tab(clicked_tab))
            menu.add_command(label="Close Others", command=lambda: self.close_other_tabs(clicked_tab))
            menu.add_command(label="Close All", command=self.close_all_tabs)
            menu.tk_popup(event.x_root, event.y_root)
            
    def close_tab(self, tab_id):
        """Close specific tab"""
        editor = self.editor_notebook.children[self.editor_notebook.tabs()[tab_id].split('.')[-1]]
        if self.check_save_changes(editor):
            self.editor_notebook.forget(tab_id)
            if len(self.editor_notebook.tabs()) == 0:
                self.create_new_editor()
                
    def close_other_tabs(self, current_tab):
        """Close all tabs except the specified one"""
        for tab in range(len(self.editor_notebook.tabs())-1, -1, -1):
            if tab != current_tab:
                editor = self.editor_notebook.children[self.editor_notebook.tabs()[tab].split('.')[-1]]
                if self.check_save_changes(editor):
                    self.editor_notebook.forget(tab)
                    
    def close_all_tabs(self):
        """Close all tabs"""
        for tab in range(len(self.editor_notebook.tabs())-1, -1, -1):
            editor = self.editor_notebook.children[self.editor_notebook.tabs()[tab].split('.')[-1]]
            if self.check_save_changes(editor):
                self.editor_notebook.forget(tab)
        self.create_new_editor()
        
    def close_current_tab(self, event=None):
        """Close currently active tab"""
        current = self.editor_notebook.select()
        if current:
            tab_id = self.editor_notebook.index(current)
            self.close_tab(tab_id)
            
    def check_save_changes(self, editor):
        """Check if changes need to be saved"""
        if hasattr(editor, 'is_modified') and editor.is_modified():
            response = messagebox.askyesnocancel(
                "Save Changes",
                f"Save changes to {self._get_tab_name(self.editors[editor])}?"
            )
            if response is None:  # Cancel
                return False
            if response:  # Yes
                return self.save_file(editor)
        return True
        
    def new_file(self, event=None):
        editor = self.create_new_editor()
        editor.set_current_file('<unsaved>')
        editor.set_modified(False)
        return editor
        
    def open_file(self, event=None):
        file_paths = filedialog.askopenfilenames(
            defaultextension=".py",
            filetypes=[
                ("Python Files", "*.py"),
                ("All Files", "*.*")
            ]
        )
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                editor = self.create_new_editor(file_path)
                editor.delete('1.0', tk.END)
                editor.insert('1.0', content)
                editor.set_current_file(file_path)
                editor.set_modified(False)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {str(e)}")
                
    def save_file(self, editor=None, event=None):
        editor = editor or self.current_editor
        if not editor:
            return False
            
        current_file = editor.get_current_file()
        if not current_file or current_file == '<unsaved>':
            return self.save_as_file(editor)
        
        try:
            content = editor.get('1.0', 'end-1c')
            with open(current_file, 'w', encoding='utf-8') as file:
                file.write(content)
            editor.set_modified(False)
            self.editor_notebook.tab(editor, text=self._get_tab_name(current_file))
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")
            return False
            
    def save_as_file(self, editor=None, event=None):
        editor = editor or self.current_editor
        if not editor:
            return False
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[
                ("Python Files", "*.py"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            editor.set_current_file(file_path)
            self.editors[editor] = file_path
            return self.save_file(editor)
        return False