<<<<<<< HEAD
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
directly (i.e. list, tuple, dict instead of typing.List, typing.Tuple, typing.Dict) 
=======
# SPIIR Python Library Development Guide

## Documentation

### NumPy Style

All documentation should follow the NumPy documentation style guide which can be found 
[here](https://numpydoc.readthedocs.io/en/latest/format.html). The NumPy style guide is 
a derivation of the Google Developer documentation style guide. All function docstrings 
must follow the style and order of sections as defined in the link above.

When type hints are written into the documentation, we strongly recommend writing them 
<<<<<<< HEAD
in the style of Python most modern type hint style, which leverages Python types 
directly  (i.e. list, tuple, dict instead  of typing.List, typing.Tuple, typing.Dict) 
>>>>>>> Add sphinx quick start
=======
in the style of Python's most modern type hint style, which leverages Python types 
directly (i.e. list, tuple, dict instead of typing.List, typing.Tuple, typing.Dict) 
>>>>>>> Minor edits to CONTRIBUTING.md
as well as the new typing operators for union (`|` instead of `typing.Union`) and 
optionals (`| None` instead of `typing.Optional`). However, currently this package is 
being written in Python3.8, and so the latest Python3.9 and Python3.10 typing features 
won't be available in the actual code itself. Nevertherless, the new approach is more 
readable and will be suitable for future updates when the library is eventually ported 
to Python3.10.

<<<<<<< HEAD
<<<<<<< HEAD
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

## Development To Do List

- Add tests using PyTest.
- Implement a formal logging system with the logging module.
- Add documentation with Sphinx.

### spiir/distribution/distribution.py

- Allow JointDistribution to take JointDistributions as arguments, or
  - refactor a single class for both Distribution and JointDistribution, or
  - simply use JointDistribution((masses.distributions, spins.distributions))
- Add constraints to Distribution objects, rather than only JointDistributions.
- Add more robust input argument and type checking to custom calsses.
- Consider necessity of tuples over lists for class input arguments
- Review NumPy random number generation:
  - see: https://albertcthomas.github.io/good-practices-random-number-generators/

### spiir/inspiral/waveform.py

- Confirm input argument descriptions to lalsimulation/lalinference related functions.
- Implement multi-process parameter sampling tools for waveform generation functions.
- Review Bilby implementation: https://github.com/lscsoft/bilby/blob/master/bilby/gw/conversion.py#L73

### spiir/utils/data.py

- Enable iteration over other axes for chunk_iterable:
  - Try iterator = iter(np.moveaxis(iterable, 0, axis)).
  - See posts [here](https://stackoverflow.com/a/5923332) and [here](https://numpy.org/doc/stable/reference/generated/numpy.nditer.html).

### Requests from Damon

Package functionality requests:

- Option for astrophysical distributions/importing LIGO format injection files (shouldn't be difficult to import these)
- I want to be able to use any amount of detectors and use either provided PSDs for each or use real noise from an operating run
- Single detector operation is key at this stage for detection, as we are focusing on single detector performance and can do coincidence after
- We do want multiple detectors though for things like parameter estimation
- Just generally we want outputs of either strain or SNR time-series. I think currently most of us are focusing on that
- For the matched filtering, we also want to be able to specify a negative latency for producing datasets of early warning injections
- I've got most of the code required for acquiring regions of real data that we can use to generate datasets, but am yet to implement real data noise
- The ability to reproduce datasets would be good, so just specifying a global seed at runtime essentially
- We already have this, but utilizing SPIIR/GstLAL template banks as the templates to use in matched filtering (template selection is normally distributed around the best template match to the injection by chirp mass)
- I guess ability to produce frequency series or time series data on request would be cool, but I don't think anyone is using frequency series data at the moment

The current issues holding things up include:

- PyCBC matched filtering results in peak SNRs that aren't equal to what we scale the injection to have, it always seems to be distributed around the desired value
- I realised the current code I've been working on is all more astrophysical and multiple detectors, when I desire mostly single detector as mentioned before
- I think I've fixed injections being projected properly for multiple detectors, but I will list it here in case
=======
### Sphinx
=======
### Building With Sphinx
>>>>>>> Rerun make github again

The documentation for this package can be built using Sphinx by running the following 
commands in the `docs/` directory as follows:

    sphinx-quickstart
<<<<<<< HEAD
    sphinx-apidoc -f -o ./source ../src  # generates .rst files from code
    make html                            # use make clean html to remove old build files
>>>>>>> Add sphinx quick start
=======
    sphinx-apidoc -M -f -o ./source ../src/spiir  # generates .rst files from code
    make github                          # use make clean to remove old build files
>>>>>>> Rerun make github again
