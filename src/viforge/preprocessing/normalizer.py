"""
ViForge Code AST Normalizer & Comment Formatter.
"""

import ast
import re
from typing import Tuple


class CodeNormalizer:
    """Normalizes code syntax, strips comments, and verifies AST parseability."""

    @classmethod
    def strip_comments(cls, code: str) -> str:
        """Strip single-line and multi-line comments."""
        # Strip single line comments (# ...)
        code_no_comments = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
        # Strip multi-line docstrings
        code_no_docstrings = re.sub(r'""".*?"""', "", code_no_comments, flags=re.DOTALL)
        code_clean = re.sub(r"'''.*?'''", "", code_no_docstrings, flags=re.DOTALL)
        # Normalize whitespace
        return "\n".join(line.rstrip() for line in code_clean.splitlines() if line.strip())

    @classmethod
    def validate_ast(cls, code: str) -> Tuple[bool, str]:
        """Validate if Python snippet is syntactically correct."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)
