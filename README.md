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
  ```

Next we need to change directories to where this repository is installed. For example:

  ```
  # download spiir repository and change working directory
  git clone https://github.com/tanghyd/spiir.git
  cd spiir

  # then we can install our local "spiir" package
  # this will also install required dependencies
  pip install .
  ```

#### Optional Dependencies

Sometimes optional dependencies may be required by the user, such as certain distribution implementations by PyCBC. To install these, we can instead add an extra tag to our install as follows:

  ```
  pip install .[pycbc]
  ```

#### Jupyter Notebook

If the user would like to use this virtual environment in a Jupyter notebook kernel, we can execute the following:

  ```
  # add virtual environment to jupyter notebook kernels (can change --name)
  pip install ipykernel
  python -m ipykernel install --user --name=spiir
  ```
