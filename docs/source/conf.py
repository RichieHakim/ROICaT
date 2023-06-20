# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path

## Get version number
_dir_parent = Path(__file__).parent.parent.parent

with open(str(_dir_parent / "roicat" / "__init__.py"), "r") as _f:
    for _line in _f:
        if _line.startswith("__version__"):
            _version = _line.split("=")[1].strip().replace("\"", "").replace("\'", "")
            break

project = 'ROICaT'
copyright = '2023, Rich Hakim, Joshua Zimmer, Janet Berrios, Gyu Heo'
author = 'Rich Hakim, Joshua Zimmer, Janet Berrios, Gyu Heo'
release = str(_version)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # allows automatic parsing of docstrings
    'sphinx.ext.mathjax',  # allows mathjax in documentation
    'sphinx.ext.viewcode',  # links documentation to source code
    'sphinx.ext.githubpages',  # allows integration with github
    'sphinx.ext.napoleon',  # parsing of different docstring styles
    'sphinx.ext.coverage',  # allows coverage of docstrings
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

exclude_patterns = ['_build']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# import sphinx_rtd_theme
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme = 'sphinx_rtd_theme'  ## Theme for documentation
# html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'ROICaT-doc'

html_favicon = '../media/favicon_grayOnWhite.png'
html_logo = '../media/favicon_grayOnWhite.png'
# html_logo = '../../media/logo1.png'

# def setup(app):
#     app.add_stylesheet('css/custom.css')