"""Static-only Python AST / Kotlin and TypeScript Tree-sitter parsing."""
import ast
from functools import lru_cache

from tree_sitter import Language, Parser
import tree_sitter_kotlin
import tree_sitter_typescript


@lru_cache
def parser(language: str, tsx: bool = False):
    grammar = tree_sitter_kotlin.language() if language == 'kotlin' else (
        tree_sitter_typescript.language_tsx() if tsx else tree_sitter_typescript.language_typescript())
    return Parser(Language(grammar))


def analyze(text: str, language: str, suffix: str = '') -> dict:
    """No import, eval, compilation, build hooks, or source execution occurs."""
    if language == 'python':
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            return {'valid': False, 'symbols': [], 'imports': []}
        symbols, imports = [], []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
                symbols.append({'symbol': node.name, 'kind': 'class' if isinstance(node, ast.ClassDef) else 'function',
                                'start_line': start, 'end_line': node.end_lineno, 'valid': True})
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.get_source_segment(text, node) or '')
        return {'valid': True, 'symbols': symbols, 'imports': imports}
    data = text.encode('utf-8')
    tree = parser(language, suffix == '.tsx').parse(data)
    symbols, imports = [], []
    wanted = {'function_declaration', 'class_declaration', 'object_declaration', 'interface_declaration',
              'method_definition', 'function_expression'}
    def walk(node):
        if node.type in ('import_header', 'import_declaration', 'import_statement'):
            imports.append(data[node.start_byte:node.end_byte].decode())
        if node.type in wanted:
            name = node.child_by_field_name('name')
            if name:
                symbols.append({'symbol': data[name.start_byte:name.end_byte].decode(), 'kind': node.type,
                                'start_line': node.start_point.row + 1, 'end_line': node.end_point.row + 1,
                                'valid': not node.has_error})
        if node.type == 'variable_declarator':
            value, name = node.child_by_field_name('value'), node.child_by_field_name('name')
            if value and name and value.type in ('arrow_function', 'function_expression'):
                symbols.append({'symbol': data[name.start_byte:name.end_byte].decode(), 'kind': 'arrow_function',
                                'start_line': node.start_point.row + 1, 'end_line': node.end_point.row + 1,
                                'valid': not node.has_error})
        for child in node.named_children:
            walk(child)
    walk(tree.root_node)
    return {'valid': not tree.root_node.has_error, 'symbols': symbols, 'imports': imports}


def syntax_valid(text: str, language: str) -> bool:
    return bool(text.strip()) and analyze(text, language)['valid']
