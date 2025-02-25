import tkinter as tk
from tkinter import ttk
from src.utils.themes import IDEThemes
from src.core.debugger import IDEDebugger
from ..ai import CodeEmbeddings, MLPredictor, PatternRecognizer

logger = logging.getLogger(__name__)

class DebugPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        # Initialize AI components
        self.code_embeddings = CodeEmbeddings()
        self.ml_predictor = MLPredictor()
        self.pattern_recognizer = PatternRecognizer()
        
        self.themes = IDEThemes()
        self.debugger = IDEDebugger()
        self.current_file = None
        self.last_analysis = None
        
        # Configure styles
        style = ttk.Style()
        style.layout('Debug.TNotebook.Tab', [
            ('Debug.TNotebook.Tab', {
                'sticky': 'nswe',
                'children': [
                    ('Debug.TNotebook.Tab.padding', {
                        'side': 'top',
                        'sticky': 'nswe',
                        'children': [
                            ('Debug.TNotebook.Tab.label', {'side': 'left', 'sticky': ''})
                        ]
                    })
                ]
            })
        ])
        
        # Create debug toolbar
        self.create_toolbar()
        
        # Create debug notebook
        self.create_notebook()

        # Pack the debug panel itself
        self.pack(fill=tk.BOTH, expand=True)

        # Apply initial theme
        self.apply_theme('Default')


     def update_debug_state(self, code: str, line_no: int = None):
        """Update debug state with AI insights"""
        try:
            # Get real-time analysis
            embeddings = self.code_embeddings.generate(code)
            context = {'cursor_position': line_no} if line_no else None
            ml_analysis = self.ml_predictor.predict_realtime(code, context)
            pattern_analysis = self.pattern_recognizer.analyze_realtime(code, line_no)
            
            # Update variable inspector with ML insights
            debug_state = self.debugger.get_current_state()
            if debug_state:
                self._update_variable_inspector(debug_state['locals'], ml_analysis)
                self._update_callstack(debug_state, pattern_analysis)
                self._update_ai_insights(code, line_no, ml_analysis, pattern_analysis)
            
            self.last_analysis = ml_analysis
            
        except Exception as e:
            logger.error(f"Debug state update error: {str(e)}")

    def _update_variable_inspector(self, variables: dict, ml_analysis: dict):
        """Update variable inspector with ML-enhanced insights"""
        self.var_tree.delete(*self.var_tree.get_children())
        
        for name, value in variables.items():
            var_type = type(value).__name__
            risk_level = self._analyze_variable_risk(name, value, ml_analysis)
            
            # Add variable with color coding based on risk level
            self.var_tree.insert('', 'end', values=(name, str(value), var_type),
                               tags=(risk_level,))
            
        # Configure tag colors
        self.var_tree.tag_configure('high_risk', foreground='red')
        self.var_tree.tag_configure('medium_risk', foreground='orange')
        self.var_tree.tag_configure('low_risk', foreground='green')

    def _update_ai_insights(self, code: str, line_no: int, ml_analysis: dict, pattern_analysis: dict):
        """Update AI insights panel with real-time analysis"""
        self.debug_suggestions_text.delete('1.0', tk.END)
        
        insights = [
            "🔍 AI Debug Insights:",
            f"Current Line Quality: {ml_analysis.get('quality_score', 0.0):.2f}",
            f"Bug Probability: {ml_analysis.get('bug_probability', 0.0):.2f}",
            "\n🛠 Suggestions:",
            *ml_analysis.get('immediate_suggestions', []),
            "\n⚠ Potential Issues:",
            *pattern_analysis.get('issues', []),
            "\n💡 Runtime Considerations:",
            *self._get_runtime_insights(code, line_no)
        ]
        
        self.debug_suggestions_text.insert('1.0', '\n'.join(insights))

    def _analyze_variable_risk(self, name: str, value: Any, ml_analysis: dict) -> str:
        """Analyze variable risk level using ML insights"""
        try:
            # Check for known risky patterns
            risk_score = 0
            
            # Value-based risks
            if isinstance(value, (list, dict, set)) and len(value) > 1000:
                risk_score += 0.3  # Large collections
            
            # Name-based risks from ML analysis
            var_risks = ml_analysis.get('variable_risks', {})
            if name in var_risks:
                risk_score += var_risks[name]
            
            # Determine risk level
            if risk_score > 0.7:
                return 'high_risk'
            elif risk_score > 0.3:
                return 'medium_risk'
            return 'low_risk'
            
        except Exception as e:
            logger.error(f"Variable risk analysis error: {str(e)}")
            return 'low_risk'

    def _get_runtime_insights(self, code: str, line_no: int) -> List[str]:
        """Get ML-based runtime insights"""
        try:
            insights = []
            if self.last_analysis:
                complexity = self.last_analysis.get('complexity_score', 0.0)
                if complexity > 0.7:
                    insights.append("⚠ High computational complexity detected")
                
                memory_usage = self.last_analysis.get('memory_usage', 0.0)
                if memory_usage > 0.7:
                    insights.append("⚠ Potential memory intensive operation")
            
            return insights
            
        except Exception as e:
            logger.error(f"Runtime insights error: {str(e)}")
            return []

    def start_debugging(self):
        """Start debugging session"""
        if self.current_file:
            self.debugger.start_debugging(self.current_file)
        
    def stop_debugging(self):
        """Stop debugging session"""
        self.debugger.stop_debugging()
        
    def step_next(self):
        """Step to next line"""
        self.debugger.step_over()
        
    def step_over(self):
        """Step over current function"""
        self.debugger.step_over()
        
    def step_into(self):
        """Step into function call"""
        self.debugger.step_into()
        
    def step_out(self):
        """Step out of current function"""
        self.debugger.step_out()
        
    def set_current_file(self, filename):
        """Set the current file for debugging"""
        self.current_file = filename

    def create_toolbar(self):
        """Create debug toolbar with control buttons"""
        self.debug_toolbar = ttk.Frame(self, style='Debug.TFrame')
        self.debug_toolbar.pack(fill=tk.X, padx=3, pady=1)
        
        # Debug control buttons with explicit font and padding
        button_configs = [
            ("▶", self.start_debugging, 2),
            ("⏹", self.stop_debugging, 2),
            (None, None, None),  # Separator
            ("Step", self.step_next, 4),
            ("Over", self.step_over, 4),
            ("Into", self.step_into, 4),
            ("Out", self.step_out, 4)
        ]
        
        self.debug_buttons = []  # Store button references
        for text, command, width in button_configs:
            if text is None:  # Separator
                ttk.Separator(self.debug_toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
            else:
                btn = ttk.Button(self.debug_toolbar, text=text, command=command, 
                               style='Debug.TButton', width=width)
                btn.pack(side=tk.LEFT, padx=1)
                self.debug_buttons.append(btn)

    def create_notebook(self):
        """Create debug notebook with various panels"""
        self.debug_notebook = ttk.Notebook(self, style='Debug.TNotebook')
        self.debug_notebook.pack(fill=tk.BOTH, expand=True, padx=3, pady=1)
        
        # Variables panel
        self.variables_frame = ttk.Frame(self.debug_notebook, style='Debug.TFrame')
        self.debug_notebook.add(self.variables_frame, text="Variables")
        self.create_variable_inspector(self.variables_frame)
        
        # Call Stack panel
        self.callstack_frame = ttk.Frame(self.debug_notebook, style='Debug.TFrame')
        self.debug_notebook.add(self.callstack_frame, text="Call Stack")
        self.create_callstack_view(self.callstack_frame)
        
        # Breakpoints panel
        self.breakpoints_frame = ttk.Frame(self.debug_notebook, style='Debug.TFrame')
        self.debug_notebook.add(self.breakpoints_frame, text="Breakpoints")
        self.create_breakpoints_view(self.breakpoints_frame)
        
        # AI Insights panel
        self.insights_frame = ttk.Frame(self.debug_notebook, style='Debug.TFrame')
        self.debug_notebook.add(self.insights_frame, text="AI Insights")
        self.create_insights_view(self.insights_frame)

    def create_variable_inspector(self, parent_frame):
        """Create the variable inspector panel"""
        columns = ('Name', 'Value', 'Type')
        self.var_tree = ttk.Treeview(parent_frame, columns=columns, show='headings', style='Debug.Treeview')
        
        # Configure columns
        for col in columns:
            self.var_tree.heading(col, text=col)  # Removed style parameter
            self.var_tree.column(col, width=100)
            
        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=self.var_tree.yview, style='Debug.Vertical.TScrollbar')
        self.var_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.var_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def create_callstack_view(self, parent_frame):
        """Create the call stack view"""
        self.callstack_text = tk.Text(parent_frame, wrap=tk.WORD, height=10)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=self.callstack_text.yview, style='Debug.Vertical.TScrollbar')
        self.callstack_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.callstack_text.pack(fill=tk.BOTH, expand=True)

    def create_breakpoints_view(self, parent_frame):
        """Create the breakpoints view"""
        self.breakpoints_text = tk.Text(parent_frame, wrap=tk.WORD, height=10)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=self.breakpoints_text.yview, style='Debug.Vertical.TScrollbar')
        self.breakpoints_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.breakpoints_text.pack(fill=tk.BOTH, expand=True)

    def create_insights_view(self, parent_frame):
        """Create the AI insights view"""
        self.debug_suggestions_text = tk.Text(parent_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=self.debug_suggestions_text.yview, style='Debug.Vertical.TScrollbar')
        self.debug_suggestions_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.debug_suggestions_text.pack(fill=tk.BOTH, expand=True)

    def apply_theme(self, theme_name):
        """Apply theme to debug panel components"""
        theme = self.themes.get_theme(theme_name) 
        style = ttk.Style()
        
        # Configure debug panel specific styles
        style.configure('Debug.TFrame',
            background=theme['background']
        )
        
        # Configure notebook
        style.configure('Debug.TNotebook',
            background=theme['background']
        )
        style.configure('Debug.TNotebook.Tab',
            background=theme['button_bg'],
            foreground=theme['button_fg'],
            padding=[5, 2],
            font=('TkDefaultFont', 9)
        )
        style.map('Debug.TNotebook.Tab',
            background=[('selected', theme['button_active_bg']),
                       ('active', theme['button_active_bg'])],
            foreground=[('selected', theme['button_active_fg']),
                       ('active', theme['button_active_fg'])]
        )
        
        # Configure buttons with enhanced visibility
        style.configure('Debug.TButton',
            background=theme['button_bg'],
            foreground=theme['button_fg'],
            padding=2,
            relief='raised',
            font=('TkDefaultFont', 9, 'bold')
        )
        style.map('Debug.TButton',
            background=[('pressed', theme['button_active_bg']),
                       ('active', theme['button_active_bg'])],
            foreground=[('pressed', theme['button_active_fg']),
                       ('active', theme['button_active_fg'])],
            relief=[('pressed', 'sunken')]
        )
        
        # Configure treeview and its headers
        style.configure('Debug.Treeview',
            background=theme['background'],
            foreground=theme['foreground'],
            fieldbackground=theme['background'],
            font=('TkDefaultFont', 9)
        )
        style.map('Debug.Treeview',
            background=[('selected', theme['selection_bg'])],
            foreground=[('selected', theme['selection_fg'])]
        )
        
        # Enhanced treeview header styling
        style.configure('Debug.Treeview.Heading',
            background=theme['button_bg'],
            foreground=theme['button_fg'],
            relief='raised',
            font=('TkDefaultFont', 9, 'bold'),
            padding=2
        )
        style.map('Debug.Treeview.Heading',
            background=[('active', theme['button_active_bg'])],
            foreground=[('active', theme['button_active_fg'])]
        )
        
        # Configure scrollbar
        style.configure('Debug.Vertical.TScrollbar',
            background=theme['button_bg'],
            troughcolor=theme['background'],
            arrowcolor=theme['button_fg'],
            bordercolor=theme['border'],
            lightcolor=theme['button_bg'],
            darkcolor=theme['button_bg'],
            relief='raised',
            width=12
        )
        style.map('Debug.Vertical.TScrollbar',
            background=[('active', theme['button_active_bg'])],
            arrowcolor=[('active', theme['button_active_fg'])]
        )
        
        # Apply to all text widgets
        for text_widget in [self.debug_suggestions_text, self.callstack_text, self.breakpoints_text]:
            text_widget.configure(
                background=theme['background'],
                foreground=theme['foreground'],
                insertbackground=theme['cursor'],
                selectbackground=theme['selection_bg'],
                selectforeground=theme['selection_fg'],
                font=('TkDefaultFont', 9),
                borderwidth=1,
                relief='solid',
                highlightthickness=1,
                highlightbackground=theme['border'],
                highlightcolor=theme['border']
            )
        
        # Force treeview header update
        if hasattr(self, 'var_tree'):
            style.configure('Treeview.Heading',
                background=theme['button_bg'],
                foreground=theme['button_fg']
            )
            self.var_tree.configure(style='Debug.Treeview')
            
            # Refresh headers
            for col in self.var_tree['columns']:
                self.var_tree.heading(col, text=self.var_tree.heading(col)['text'])
        
        # Update button states
        if hasattr(self, 'debug_buttons'):
            for btn in self.debug_buttons:
                btn.configure(style='Debug.TButton')
        
        # Force frame updates
        self.configure(style='Debug.TFrame')
        self.debug_toolbar.configure(style='Debug.TFrame')
        for frame in [self.variables_frame, self.callstack_frame, 
                     self.breakpoints_frame, self.insights_frame]:
            frame.configure(style='Debug.TFrame')