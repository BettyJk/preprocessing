import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Data Preprocessing Pro'
author = 'Bouthayna Jouak , Hajar El Hadri'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Support Markdown
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
