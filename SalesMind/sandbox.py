import sys
import io
import contextlib
import ast
import time
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from config import settings

class SafeCodeExecutor:
    """Safe execution environment for AI-generated code."""
    
    ALLOWED_MODULES = {
        'pandas', 'pd', 'numpy', 'np', 'plotly', 'math', 'datetime',
        'plotly.express', 'plotly.graph_objects'
    }
    
    FORBIDDEN_FUNCTIONS = {
        'open', 'file', 'input', 'raw_input', 'exec', 'eval', 'compile',
        '__import__', 'reload', 'exit', 'quit', 'help', 'license', 'credits',
        'copyright', 'dir', 'vars', 'locals', 'globals'
    }

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.start_time = None

    def is_safe_code(self, code: str) -> Tuple[bool, str]:
        """Check if code is safe to execute."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        for node in ast.walk(tree):
            # Check for forbidden function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.FORBIDDEN_FUNCTIONS:
                        return False, f"Forbidden function: {node.func.id}"
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in self.FORBIDDEN_FUNCTIONS:
                        return False, f"Forbidden function: {node.func.attr}"

            # Check for file operations
            if isinstance(node, ast.Attribute):
                if node.attr in ['open', 'read', 'write', 'remove', 'delete']:
                    return False, f"Forbidden file operation: {node.attr}"

            # Check for imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.ALLOWED_MODULES:
                        return False, f"Forbidden import: {alias.name}"

            if isinstance(node, ast.ImportFrom):
                if node.module and node.module not in self.ALLOWED_MODULES:
                    return False, f"Forbidden import from: {node.module}"

        return True, "Code is safe"

    def execute_code(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute code safely and return results."""
        self.start_time = time.time()
        
        # Check if code is safe
        is_safe, message = self.is_safe_code(code)
        if not is_safe:
            return {
                'success': False,
                'error': f"Unsafe code detected: {message}",
                'output': '',
                'result': None
            }

        # Prepare execution environment
        safe_globals = {
            'pd': pd,
            'pandas': pd,
            'np': np,
            'numpy': np,
            'px': px,
            'go': go,
            'df': df,
            '__builtins__': self._get_safe_builtins(),
            'print': self._safe_print,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'sum': sum,
            'max': max,
            'min': min,
            'abs': abs,
            'round': round,
            'sorted': sorted,
        }
        
        safe_locals = {}
        
        # Capture output
        output_buffer = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(output_buffer):
                with contextlib.redirect_stderr(output_buffer):
                    # Execute with timeout check
                    exec(code, safe_globals, safe_locals)
                    
                    # Check execution time
                    if time.time() - self.start_time > self.timeout:
                        return {
                            'success': False,
                            'error': 'Execution timeout',
                            'output': output_buffer.getvalue(),
                            'result': None
                        }
            
            # Get output
            output = output_buffer.getvalue()
            
            # Limit output size
            if len(output) > settings.max_output_size:
                output = output[:settings.max_output_size] + "\n... (output truncated)"
            
            # Try to get meaningful results
            result = self._extract_results(safe_locals)
            
            return {
                'success': True,
                'error': None,
                'output': output,
                'result': result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'output': output_buffer.getvalue(),
                'result': None
            }

    def _get_safe_builtins(self) -> Dict[str, Any]:
        """Get safe builtin functions."""
        safe_builtins = {}
        
        # Allow only safe builtins
        allowed_builtins = [
            'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
            'range', 'enumerate', 'zip', 'sum', 'max', 'min', 'abs', 'round',
            'sorted', 'isinstance', 'type', 'hasattr', 'getattr'
        ]
        
        for name in allowed_builtins:
            if hasattr(__builtins__, name):
                safe_builtins[name] = getattr(__builtins__, name)
        
        return safe_builtins

    def _safe_print(self, *args, **kwargs):
        """Safe print function with size limits."""
        output = ' '.join(str(arg) for arg in args)
        if len(output) > 1000:  # Limit individual print statements
            output = output[:1000] + "... (truncated)"
        print(output, **kwargs)

    def _extract_results(self, locals_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract meaningful results from execution."""
        results = {}
        
        for name, value in locals_dict.items():
            if not name.startswith('_'):  # Skip private variables
                try:
                    if isinstance(value, (pd.DataFrame, pd.Series)):
                        # Convert pandas objects to dict
                        if isinstance(value, pd.DataFrame):
                            results[name] = {
                                'type': 'dataframe',
                                'shape': value.shape,
                                'columns': list(value.columns),
                                'head': value.head().to_dict('records') if len(value) > 0 else []
                            }
                        else:
                            results[name] = {
                                'type': 'series',
                                'length': len(value),
                                'head': value.head().to_dict() if len(value) > 0 else {}
                            }
                    elif isinstance(value, (go.Figure, dict)):
                        # Handle plotly figures
                        results[name] = {
                            'type': 'figure',
                            'data': str(type(value))
                        }
                    elif isinstance(value, (int, float, str, bool, list)):
                        # Basic types
                        results[name] = {
                            'type': type(value).__name__,
                            'value': value
                        }
                except Exception:
                    # Skip problematic values
                    continue
        
        return results

# Global executor instance
executor = SafeCodeExecutor(timeout=settings.sandbox_timeout)
