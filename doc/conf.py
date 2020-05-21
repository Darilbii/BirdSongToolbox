# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# import os
# from os.path import dirname as up

from datetime import date

import sphinx_gallery
import sphinx_bootstrap_theme
from sphinx_gallery.sorting import FileNameSortKey, ExplicitOrder



# -- Project information -----------------------------------------------------

project = 'BirdSongToolbox'
copyright = '2020, Daril E. Brown II'
author = 'Daril E. Brown II'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Set to generate sphinx docs for class members (methods)
autodoc_default_options = {
    'members': None,
    'inherited-members': None,
}

# generate autosummary even if no references
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'

# Set the theme path explicitly
#   This isn't always needed, but is useful so bulding docs doesn't fail on
#   operating systems which don't have bootstrap on theme path
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Theme options to customize the look and feel, which are theme-specific.
html_theme_options = {
    'navbar_sidebarrel': False,
    'navbar_links': [
        ("Installation", 'installation'),
        ("Overview", "overview/index"),
        ("Tutorials", "auto_tutorials/index"),
        ("Glossary", "glossary"),
        ("FAQ", "faq"),
        ("API", "api"),
        ("Examples", "auto_examples/index"),
        ("GitHub", "https://github.com/Darilbii/BirdSongToolbox", True),
    ],

    # Set the page width to not be restricted to hardset value
    'body_max_width': None,

    # Bootswatch (http://bootswatch.com/) theme to apply.
    'bootswatch_theme': "flatly",

    # Render the current pages TOC in the navbar
    'navbar_pagenav': False,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Configurations for sphinx gallery
sphinx_gallery_conf = {
    'examples_dirs': ['../examples', '../tutorials'],
    'gallery_dirs': ['auto_examples', 'auto_tutorials'],
    'within_subsection_order': FileNameSortKey,
    'default_thumb_file': 'img/bird_web_version.png',
    'backreferences_dir': 'generated',   # Where to drop linking files between examples & API
    'doc_module': ('BirdSongToolbox',),
    'reference_url': {'BirdSongToolbox': None},
    'remove_config_comments': True,
}