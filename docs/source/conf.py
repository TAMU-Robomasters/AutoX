# Configuration file for the Sphinx documentation builder.  # noqa: D100
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from importlib.metadata import version as get_version

sys.path.insert(0, os.path.abspath("../../"))  # Adjust path to your source code

project = "AutoX"
copyright = "2026, Texas Aimbots"
author = "Texas Aimbots"
release = get_version("AutoX")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Autodoc configuration ---------------------------------------------------
# Mock modules that have hardware dependencies or cause side effects on import.
# This allows Sphinx to generate API docs without needing GPUs, cameras, etc.
autodoc_mock_imports = [
    # Hardware / GPU
    "torch",
    "tensorrt",
    "pycuda",
    "pyrealsense2",
    "iceoryx2",
    "cupy",
    "serial",
    # Config (executes find_and_load at import time, parses CLI args)
    "quik_config",
    "super_map",
    # ML frameworks
    "ultralytics",
    "onnx",
    "onnxruntime",
    # Vision / media
    "cv2",
    # Utility
    "file_system_py",
    # Bare-path aliases used by some legacy modules (from toolbox.globals, etc.)
    "toolbox",
    "subsystems",
    "plugins",
]

# Pre-populate sys.modules with mock versions of internal modules that have
# side effects at import time (e.g. reading YAML config, opening hardware).
# This must happen before Sphinx's autodoc tries to import any src.* modules.
from unittest.mock import MagicMock

_internal_mocks = [
    # globals.py executes find_and_load() and parses CLI args at module level
    "src.toolbox.globals",
    # video_stream.py calls create_video_stream() at module level
    "src.subsystems.video_streaming.video_stream",
    # display.py creates Display() singleton at module level
    "src.subsystems.display",
    # communicate.py opens serial ports at module level
    "src.subsystems.communicate",
    # selection.py reads config at module level
    "src.subsystems.selection",
    # log.py reads config and has broken imports
    "src.subsystems.log",
    # pnp.py calls video_stream.get_intrinsics() at module level
    "src.subsystems.vision.classical_detector.pnp",
    # icon_detection.py loads images from disk at module level
    "src.subsystems.vision.classical_detector.icon_detection",
    # realsense.py creates rs.align() at module level
    "src.subsystems.video_streaming.realsense",
    # types/ has no __init__.py; submodules use video_stream at class level
    "src.types",
    "src.types.ipc",
    "src.types.autoaim",
    "src.types.world_model",
    # modeling plugins load native .so files at module level
    "src.subsystems.modeling.model.plugins",
]

for _mod in _internal_mocks:
    sys.modules[_mod] = MagicMock()

autosummary_generate = True

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
