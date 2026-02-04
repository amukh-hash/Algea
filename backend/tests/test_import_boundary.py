import os
import sys
import pytest
import ast
from pathlib import Path

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
APP_ROOT = BACKEND_ROOT / "app"

def get_python_files(directory):
    """Recursively yield all Python files in a directory."""
    for root, dirs, files in os.walk(directory):
        if "legacy" in root: # Skip checking legacy itself? No, we check strict imports inside app
            continue
        for file in files:
            if file.endswith(".py"):
                yield Path(root) / file

def check_imports(file_path):
    """Parse file and check for forbidden imports."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError:
            return [] # Skip syntax errors, likely template files or broken code

    forbidden = []
    for node in ast.walk(tree):
        # Check 'import X'
        if isinstance(node, ast.Import):
            for alias in node.names:
                if is_forbidden(alias.name):
                    forbidden.append((node.lineno, alias.name))
        
        # Check 'from X import Y'
        elif isinstance(node, ast.ImportFrom):
            if node.module and is_forbidden(node.module):
                forbidden.append((node.lineno, node.module))

    return forbidden

def is_forbidden(module_name):
    """Check if module name references legacy or forbidden paths."""
    forbidden_terms = ["legacy", "Legacy_Algaie_2", "Algaie_2.0", "backend.app.features", "backend.app.targets", "backend.app.xai"]
    # We might want to be strict about 'features', 'targets' if they are top level, but they might be used if ported.
    # The user specifically said: "Ensure you didn’t accidentally keep 2.0 namespaces inside backend/app/"
    # So if code imports 'app.features' it implies it exists or is being used.
    # But this test checks *imports*, not existence.
    # To check existence, we can just fail if the folder exists.
    
    for term in forbidden_terms:
         if term in module_name:
             return True
    return False

def test_no_legacy_imports():
    """Ensure no file in backend/app imports from legacy locations."""
    errors = []
    
    for py_file in get_python_files(APP_ROOT):
        forbidden = check_imports(py_file)
        if forbidden:
            for lineno, name in forbidden:
                errors.append(f"{py_file}:{lineno} Imports forbidden module '{name}'")
    
    if errors:
        pytest.fail("\n".join(errors))

def test_clean_namespaces():
    """Ensure legacy 2.0 namespaces do not exist in backend/app."""
    forbidden_dirs = ["features", "targets", "xai", "api"]
    found = []
    
    for name in forbidden_dirs:
        path = APP_ROOT / name
        if path.exists():
            found.append(f"Forbidden directory exists: {path}")
            
    if found:
        pytest.fail("\n".join(found))

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
