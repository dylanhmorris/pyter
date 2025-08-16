# Pyter: Bayesian quantitative virology with Python and Numpyro

Pyter is an open-source [Python](https://www.python.org/) package for analyzing virological data. It allows for flexible [Bayesian inference](https://xcelab.net/rm/statistical-rethinking/) of virus titers and virus environmental halflives from raw data on cell infection. Both endpoint titration and plaque assays are supported.


Pyter is designed to be fairly plug-and-play for the scientist who has a relatively standard problem but is a novice coder, but the API also permits a more experienced user to build their own custom models without reinventing the wheel.

PyTer uses [Numpyro](https://pyro.ai/numpyro/) to specify models and perform inference from them.



## Installing Pyter

Pyter is written in [Python](https://docs.python-guide.org/), and so to install it you will need a working Python 3 installation along with the standard package manager [pip](https://pip.pypa.io/en/stable/)

### Linux, macOS, and other Unix-like systems

You can use `pip` to install Pyter directly from the project GitHub::

```bash
pip install git+https://github.com/dylanhmorris/pyter.git
```

Alternatively, you can download the repository from GitHub, navigate to the top-level directory (containing `pyproject.toml`, and run:

```bash
pip install .
```
