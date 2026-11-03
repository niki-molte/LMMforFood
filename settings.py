"""
Centralized configuration of the LLMforFood project.

This module is the only place where the ``.env`` file located in the project
root is loaded. All scripts that need API keys, model names or local paths
(e.g. the external disk holding the datasets) must read them through
:func:`get_env` instead of hard-coding them.

Usage example::

    from settings import get_env, get_path

    model = get_env("OLLAMA_GEMMA_4_E2B_Q4")
    images_dir = get_path("NUTRITION5K_DATASET_DIR")

See ``.env.example`` for the full list of expected variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

#: Project root directory (the one containing ``.env``).
PROJECT_ROOT: Path = Path(__file__).resolve().parent

#: Path of the ``.env`` file loaded when this module is imported.
ENV_FILE: Path = PROJECT_ROOT / ".env"

# The loading uses an absolute path, so it works regardless of the working
# directory the script is launched from. Variables already present in the
# shell environment take precedence (override=False).
load_dotenv(ENV_FILE, override=False)


class MissingEnvVarError(RuntimeError):
    """Raised when a required variable is not defined in ``.env``."""


def get_env(name: str, default: str | None = None, required: bool = True) -> str | None:
    """
    Returns the value of an environment variable defined in ``.env``.

    Args:
        name: Variable name (e.g. ``"GOOGLE_API_KEY"``).
        default: Value returned if the variable is not defined. If provided,
            the variable is considered optional.
        required: If ``True`` (default) and the variable is missing without a
            ``default``, :class:`MissingEnvVarError` is raised.

    Returns:
        The variable value, ``default`` or ``None``.

    Raises:
        MissingEnvVarError: if the variable is required and not defined.
    """
    value = os.environ.get(name)
    if value is not None and value.strip() != "":
        return value.strip()
    if default is not None:
        return default
    if required:
        raise MissingEnvVarError(
            f"Variable '{name}' not found. Add it to the file {ENV_FILE} "
            f"(see .env.example for the format)."
        )
    return None


def get_path(name: str, required: bool = True) -> str | None:
    """Like :func:`get_env`, but expands ``~`` and returns an absolute path."""
    value = get_env(name, required=required)
    return os.path.abspath(os.path.expanduser(value)) if value else None
