# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Pyter"
copyright = "2024, Dylan H. Morris (dylanhmorris.com)"
author = "Dylan H. Morris (dylanhmorris.com)"
release = "0.3.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.inheritance_diagram",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/CDCgov/multisignal-epi-inference",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_download_button": True,
}

html_static_path = ["_static"]

html_sidebars = {
    "**": [
        "navbar-logo.html",
        "search-field.html",
        "sbt-sidebar-nav.html",
    ]
}

autosummary_generate = True
autosummary_imported_members = True
autosummary_ignore_module_all = False
autodoc_typehints_format = "short"
autodoc_typehints = "signature"
autodoc_type_aliases = {
    "ArrayLike": ":class:`numpy.typing.ArrayLike`",
    ":class:`numpy.typing.ArrayLike`": "ArrayLike",
}


intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numpyro": ("https://num.pyro.ai/en/latest/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

master_doc = "contents"
