## Installation

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

  # add virtual environment to jupyter notebook kernels (can change --name)
  python -m ipykernel install --user --name=spiir
  ```

### Conda with LALSuite New Waveform Interface

This branch as not been tested and requires installation via Conda, following the
instructions from: https://git.ligo.org/waveforms/reviews/lalsuite/-/tree/new-interface.

  ```
  conda create -n lalsuite-dev -c conda-forge python=3.8.5

  # install new lalsuite waveform interface
  git clone -b new-interface https://git.ligo.org/waveforms/reviews/lalsuite.git
  cd lalsuite

  # direct installation from environment.yml will create env in python3.9
  #  conda env create -f conda/environment.yml

  conda activate lalsuite-dev
  
  # installs packages in pre-made python3.10 environment above
  conda install --yes --file conda/environment.yml  
  
  # install lalsuite-dev
  ./00boot
  ./configure --prefix=${CONDA_PREFIX}
  make    # may need sudo apt install gcc-multilib on ubuntu machines
  make install

  # install spiir dev dependencies
  conda install -c conda-forge pycbc pandas tqdm ipykernel black isort mypy flake8 gwpy

  # install local packages in editable mode
  pip install -e .     

  # add virtual environment to jupyter notebook kernels
  python -m ipykernel install --user --name=lalsuite-dev
  ```

## Notes from Damon

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

## Module Development To Do List

- Add tests using PyTest.
- Implement a formal logging system with the logging module.
- Add documentation with Sphinx.

### spiir/distribution/distribution.py

- Allow JointDistribution to take JointDistributions as arguments, or
  - refactor a single class for both Distribution and JointDistribution, or
  - simply use JointDistribution((*masses.distributions, *spins.distributions))
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
  - Try `iterator = iter(np.moveaxis(iterable, 0, axis))`.
  - See posts [here](https://stackoverflow.com/a/5923332) and [here](https://numpy.org/doc/stable/reference/generated/numpy.nditer.html).

## Problems

### Python3.10 and legacy 'ilwd:char' LIGO_LW type incompatibility

Using the GWPy XML reader functionality (i.e. `gwpy.table.EventTable`) is a very useful tool for our LIGO_LW XML document handling. Unfortunately, the XML reader used to parse legacy ilwd:char types seems to break on Python3.10 with the following `SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats`. 

As a work-around, we intend to develop this package in Python3.8 until a more appropriate processing tool is used to convert ilwd:char types (or standalone conversion script), or the ilwd:char type is removed from the SPIIR research and development workflow entirely. Note that backporting from Python3.10 to Python3.9 should be sufficient but it is not installed as an environment module in SPIIR's main R&D environments on OzStar.