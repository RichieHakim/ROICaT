# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ROICaT'
copyright = '2023, Rich Hakim, Joshua Zimmer, Janet Berrios'
author = 'Rich Hakim, Joshua Zimmer, Janet Berrios'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',  # allows automatic parsing of docstrings
              'sphinx.ext.mathjax',  # allows mathjax in documentation
              'sphinx.ext.viewcode',  # links documentation to source code
              'sphinx.ext.githubpages',  # allows integration with github
              'sphinx.ext.napoleon']  # parsing of different docstring styles

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

exclude_patterns = ['_build']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# html_theme = 'classic'
# html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'ROICaT-doc'
