# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
project = 'tfilterspy'
copyright = '2025, Thabang Mashinin-Sekhoto, \
Lebogang Mashinini-Sekhoto, Palesa Mashinini-Sekhoto'
author = 'Thabang Mashinin-Sekhoto, Lebogang Mashinini-Sekhoto, Palesa Mashinini-Sekhoto'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',           # Automatically document modules
    'sphinx.ext.napoleon',          # Support for Google/NumPy-style docstrings
    'sphinx.ext.intersphinx',       # Cross-reference external projects
    'sphinx.ext.viewcode',          # Link source code in the docs
     'nbsphinx',                    # Add this line to use nbsphinx
     'sphinx.ext.githubpages',      # For GitHub Pages integration
     'sphinx.ext.mathjax',         # For rendering LaTeX math
]
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'collapse_navigation': True,
}

html_static_path = ['_static']
