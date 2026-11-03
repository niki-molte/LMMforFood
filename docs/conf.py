"""Sphinx configuration for the LLMforFood documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../datasets/Nutrition5k"))

project = "LLMforFood"
author = "LLMforFood"
language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_mock_imports = [
    "torch", "transformers", "sentence_transformers", "ultralytics", "cv2",
    "optuna", "ollama", "google", "huggingface_hub", "seaborn", "sklearn",
    "scipy", "matplotlib", "PIL", "tabulate",
]
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

exclude_patterns = ["_build", "report_codebase_audit.md"]
html_theme = "furo"
html_title = "LLMforFood"
