# SPIIR Python Library

This repository is an `pip` importable Python package that provides comprehensive data generation and processing tools for research and development with the OzGrav UWA node and its low latency gravitational wave search pipeline called SPIIR.

## Installation

The instructions below detail how to install this package 

### OzStar

If installing on the OzStar supercomputer, we recommend the following environment
modules be loaded first:

  ```
  module load gcc/9.2.0 openmpi/4.0.2
  module load git/2.18.0
  module load python/3.8.5
  ```

Then proceed with the instructions for a installation with Virtualenv.

### Virtualenv
  
Installation with these commands requires an existing installation of Python3.8.

  ```
  # you can change venv to "spiir"
  python -m venv venv
  source venv/bin/activate

  # if a "conda base" env is activated, the wrong pip may be used -> check "which pip".
  pip install --upgrade pip setuptools wheel  # useful for ozstar updates
  pip install -r requirements.txt

  # install local packages in editable mode
  pip install -e .
  ```

#### Optional Dependencies

Sometimes optional dependencies may be required by the user, such as certain distribution implementatinos by PyCBC. To install these, we can instead add an extra tag to our install as follows:

  ```
  pip install -e .[pycbc]
  ```

#### Jupyter Notebook

If the user would like to use this virtual environment in a Jupyter notebook kernel, we can execute the following:

  ```
  # add virtual environment to jupyter notebook kernels (can change --name)
  python -m ipykernel install --user --name=spiir
  ```

## Current Problems

### Python3.10 and legacy 'ilwd:char' LIGO_LW type incompatibility

Using the GWPy XML reader functionality (i.e. `gwpy.table.EventTable`) is a very useful tool for our LIGO_LW XML document handling. Unfortunately, the XML reader used to parse legacy ilwd:char types seems to break on Python3.10 with the following `SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats`. 

As a work-around, we intend to develop this package in Python3.8 until a more appropriate processing tool is used to convert ilwd:char types (or standalone conversion script), or the ilwd:char type is removed from the SPIIR research and development workflow entirely. Note that backporting from Python3.10 to Python3.9 should be sufficient but it is not installed as an environment module in SPIIR's main R&D environments on OzStar.