# SPIIR Library Development Guide

## Documentation

### NumPy Style Guide

All documentation should follow the [NumPy Documentation Style Guide].
The NumPy style guide is a derivation of the Google Developer documentation style guide.
All docstrings should follow the style and order of sections as defined in the links above.

#### Docstring Conventions

Further information about writing Python docstrings can be found in the multi-line
docstring specifications from [PEP 257] detailing docstring conventions for scripts,
modules, functions, and classes. For an example of NumPy Style Python docstrings we
refer to an [example from Sphinx].

Note that this NumPy example starts with a UTF-8 encoding declaration that we have
chosen to omit in our package documentation because UTF-8 is already the default
encoding, as described in [lexical analysis] section of the Python documentation.

### Building Documentation With Sphinx

The source files for the documentation were originally produced by calling
`sphinx-quickstart` in the `docs/` directory, and making the necessary changes to
`docs/source/conf.py`, as well as adding a `make docs` command to `docs/Makefile`.

Next, the documentation can be built using Sphinx by calling `make docs` in the `docs/`
directory. A number of packages will first need to be installed to run Sphinx - we
recommend creating a virtual environment with the packages specified in the
`docs/requirements.txt` file.

### Hosting Documentation with GitHub Pages

The `make docs` target will run a series of commands that auto-generate .rst files from
the package using `sphinx-apidoc`, render the documentation as HTML files, and prepare
them for hosting. After being built, the repository changes can be committed and pushed
to GitHub where the documentation will be hosted via GitHub Pages. The GitHub Actions
script in `.github/workflows/sphinx.yaml` will automatically run `make docs` for you
and move the rendered HTML files from `docs/build/html/` to the root directory of a
stand-alone branch called `gh-pages` for hosting the documentation.

For more information, see the [Sphinx GitHub Pages Deployment Tutorial] from Sphinx.

<!-- # References -->

[example from sphinx]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy
[lexical analysis]: https://docs.python.org/3/reference/lexical_analysis.html#encoding-declarations
[numpy documentation style guide]: https://numpydoc.readthedocs.io/en/latest/format.html
[pep 257]: https://peps.python.org/pep-0257/#multi-line-docstrings
[sphinx github pages deployment tutorial]: https://www.sphinx-doc.org/en/master/tutorial/deploying.html#id5
