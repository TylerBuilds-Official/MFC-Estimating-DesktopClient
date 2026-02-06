"""
Entry point for running desktop_tooling as a module.

Usage:
    python -m desktop_tooling <pdf_path> [OPTIONS]
"""
# Use absolute import for PyInstaller compatibility
from desktop_tooling.cli import main

if __name__ == "__main__":
    main()
