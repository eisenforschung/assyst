# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ASSYST'
copyright = '2025, Max-Planck-Institute for Sustainable Materials'
author = 'Marvin Poul'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'myst_nb',
]

nb_execution_mode = "off"

import shutil
import os

# Copy notebooks from root to docs/notebooks
# This ensures that notebooks are available to sphinx regardless of where the build is run
# and avoids having to hardcode them in index.rst
nb_source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))
nb_dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'notebooks'))

if os.path.exists(nb_source_dir):
    if not os.path.exists(nb_dest_dir):
        os.makedirs(nb_dest_dir)
    for file in os.listdir(nb_source_dir):
        if file.endswith('.ipynb'):
            shutil.copy(os.path.join(nb_source_dir, file), os.path.join(nb_dest_dir, file))

intersphinx_mapping = {
        'ase': ('https://wiki.fysik.dtu.dk/ase', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'notebooks/.*']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
