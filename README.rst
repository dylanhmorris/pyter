
PyTer: Bayesian quantitative virology with Python and Numpyro
=================================

PyTer is an open-source `Python`__ package for analyzing virological data. It allows for flexible `Bayesian inference`__ of virus titers and virus environmental halflives from raw data on cell infection. Both endpoint titration and plaque assays are supported.

__ https://www.python.org/
__ https://xcelab.net/rm/statistical-rethinking/


PyTer is designed to be fairly plug-and-play for the scientist who has a relatively standard problem but is a novice coder, but the API also permits a more experienced user to build their own custom models without reinventing the wheel.

PyTer uses `Numpyro`__ to specify models and perform inference from them.

__ https://pyro.ai/numpyro/


=================
Installing PyTer
=================

PyTer is written in `Python`__, and so to install it you will need a working Python 3 installation along with the standard package manager `pip`__.

__ https://docs.python-guide.org/
__ https://pip.pypa.io/en/stable/

Linux, macOS, and other Unix-like systems
-----------------------------------------

You can use ``pip`` to install PyTer directly from the project GitHub::

   $ pip install git+https://github.com/dylanhmorris/pyter.git

Alternatively, you can download the repository from GitHub, navigate to the top-level directory (containing ``pyproject.toml``, and run::

   $ pip install .
