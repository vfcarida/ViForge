"""
AST-based Alpha-Equivalence Code Normalizer for Contamination Shielding.
Identifies syntactically and semantically identical code under variable, function,
and argument renames to prevent benchmark leakage evasion.
"""

import ast
import builtins
import hashlib
import re
from typing import Dict, List, Optional, Set

# Built-in keywords and identifiers that should not be alpha-renamed
STANDARD_RESERVED: Set[str] = set(dir(builtins)) | {
    "self",
    "cls",
    "args",
    "kwargs",
    "__init__",
    "__str__",
    "__repr__",
    "__call__",
    "__enter__",
    "__exit__",
    "__getitem__",
    "__setitem__",
    "__len__",
    "__iter__",
    "__next__",
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
}


class _AlphaTransformer(ast.NodeTransformer):
    """
    AST transformer that strips docstrings and performs canonical alpha-renaming
    on user-defined variables, functions, and arguments.
    """

    def __init__(self) -> None:
        super().__init__()
        self.var_map: Dict[str, str] = {}
        self.counter: int = 0

    def get_canonical(self, name: str) -> str:
        if name in STANDARD_RESERVED or name.startswith("__"):
            return name
        if name not in self.var_map:
            self.var_map[name] = f"v_{self.counter}"
            self.counter += 1
        return self.var_map[name]

    def _strip_docstring(self, body: List[ast.stmt]) -> None:
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body.pop(0)

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._strip_docstring(node.body)
        node.name = self.get_canonical(node.name)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self._strip_docstring(node.body)
        node.name = self.get_canonical(node.name)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self._strip_docstring(node.body)
        node.name = self.get_canonical(node.name)
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg not in ("self", "cls"):
            node.arg = self.get_canonical(node.arg)
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self.get_canonical(node.id)
        return node


class ASTAlphaNormalizer:
    """
    Normalizes code into canonical AST representations to detect alpha-equivalent
    code snippets regardless of variable renaming, docstring edits, or formatting.
    """

    @classmethod
    def extract_code_blocks(cls, text: str) -> List[str]:
        """
        Extract code from markdown code fences or return raw text if no fences found.
        """
        pattern = r"```(?:python)?\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return [m.strip() for m in matches if m.strip()]
        stripped = text.strip()
        return [stripped] if stripped else []

    @classmethod
    def normalize_code(cls, code: str) -> str:
        """
        Convert Python code into a canonical, alpha-renamed, docstring-stripped representation.
        If parsing fails (e.g. invalid syntax or non-Python snippet), gracefully falls back
        to line-level comment stripping and whitespace normalization.
        """
        if not code or not code.strip():
            return ""

        try:
            tree = ast.parse(code)
            transformer = _AlphaTransformer()
            canonical_tree = transformer.visit(tree)
            ast.fix_missing_locations(canonical_tree)
            unparsed = ast.unparse(canonical_tree)
            lines = [line.strip() for line in unparsed.splitlines() if line.strip()]
            return "\n".join(lines)
        except SyntaxError:
            # Fallback: strip line comments and normalize whitespace
            cleaned_lines = []
            for line in code.splitlines():
                no_comment = re.sub(r"#.*$", "", line).strip()
                if no_comment:
                    cleaned_lines.append(re.sub(r"\s+", " ", no_comment))
            return "\n".join(cleaned_lines)

    @classmethod
    def compute_ast_hash(cls, code: str) -> str:
        """
        Compute SHA-256 hash of the alpha-normalized code representation.
        """
        normalized = cls.normalize_code(code)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @classmethod
    def is_alpha_equivalent(cls, code_a: str, code_b: str) -> bool:
        """
        Check if two code snippets are AST alpha-equivalent (same logic under variable renames).
        """
        norm_a = cls.normalize_code(code_a)
        norm_b = cls.normalize_code(code_b)
        if not norm_a or not norm_b:
            return False
        return norm_a == norm_b
