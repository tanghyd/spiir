# SPIIR Library Development Guide

## Documentation

### NumPy Style Guide

All documentation should follow the 
[NumPy documentation style guide](https://numpydoc.readthedocs.io/en/latest/format.html).
The NumPy style guide is a derivation of the Google Developer documentation style guide.
All docstrings should follow the style and order of sections as defined in the links above.

#### Docstring Conventions

Further information about writing Python docstrings can be found in the 
[specifications from PEP 257](https://peps.python.org/pep-0257/#multi-line-docstrings) 
detailing docstring conventions for scripts, modules, functions, and classes. For an 
example of NumPy Style Python docstrings we refer to an 
[example from Sphinx](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy).
Note that this NumPy example starts with a UTF-8 encoding declaration that we have 
chosen to omit in our package documentation because UTF-8 is already the default 
encoding, as described in [lexical analysis](https://docs.python.org/3/reference/lexical_analysis.html#encoding-declarations) section of the Python documentation.

#### Type Hints

When type hints are written into the documentation, we strongly recommend writing them 
in the style of Python's most modern type hint style, which leverages Python types 
directly (i.e. `list`, `tuple`, `dict` instead of `List`, `Tuple`, `Dict` from the 
`typing` module) as well as the new typing operators for union (`|` instead of `Union`) 
and optionals (`| None` instead of `Optional`).

### Building Documentation With Sphinx

The documentation for this package can be built using Sphinx by calling `make docs` in 
the `docsrc/` directory. A number of packages will first need to be installed to run 
Sphinx - we recommend creating a virtual environment with the packages specified in the 
`requirements-docs.txt` file.

The `docs` target will run a series of commands that auto-generate .rst files for the 
documentation from the package using `sphinx-apidoc`, render the documentation as HTML 
files, and prepare them for hosting. After being built, the repository changes can be 
committed and pushed to GitHub where the documentation will be hosted via GitHub Pages.

This documentation was originally produced by calling `sphinx-quickstart` in the 
`docsrc/` directory, and making the necessary changes to `docsrc/source/conf.py`, as 
well as adding a `make docs` command to `docsrc/Makefile`.