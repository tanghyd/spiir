# SPIIR Python Library

This repository is an `pip` importable Python package that provides comprehensive data generation and processing tools for research and development with the OzGrav UWA node and its low latency gravitational wave search pipeline called SPIIR.

## Installation

The instructions below detail how to install this package.

### OzStar

For example, if one was to install this package on the OzStar supercomputer with
Python 3.10.4, we recommend loading the following environment modules:

```
module load gcc/9.2.0 openmpi/4.0.2
module load git/2.18.0
module load python/3.10.4  # or python/3.8.5 for example
```

Then proceed with the instructions for a installation with Virtualenv.

### Virtualenv

Installation with these commands requires an existing installation of Python >= 3.8.

```
# you can change the last "venv" to any name you like
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
pip install .  # optionally add the -e flag for --editable mode
```

#### Optional Dependencies

Sometimes optional dependencies may be required by the user, such as certain code used by PyCBC, or infrequently used packages such as IGWNAlert.

To install these, we can instead add an extra tag to our install as follows:

```
pip install .[p-astro]  # algorithms for p_astro classification
pip install .[igwn-alert]  # utilities for consuming IGWNAlert payloads
pip install .[all]  # all optional dependencies
```

Note that when running on zsh, we have observed that quotation marks might be required for these tags.
The solution to this error may look something like the following:

```
pip install '.[all]'
```

### Jupyter Notebook

If the user would like to use this virtual environment as a Jupyter notebook kernel called "spiir-py3x", then we can execute the following:

```
# add virtual environment to jupyter notebook kernels (you can rename "spiir-py3x" to anything)

pip install ipykernel
python -m ipykernel install --user --name=spiir-py3x
```
