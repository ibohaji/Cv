import sys
import os

def add_project_root_to_path():
    """
    Adds the root directory of the project to the system path.
    This enables the import of modules from the project, regardless of the current working directory.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
    if project_root not in sys.path:
        sys.path.append(project_root)
