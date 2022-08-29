# SPIIR Development TODO List

## Installation

SPIIR depends heavily on a codebase written in C, in particular leveraging the LIGO 
Algorithm Library (LAL) and Gstreamer (Gst) for its GPU accelerated pipeline. However, 
it is also a hybrid codebase that contains a large amount of Python, and recent 
research and development continues to lean heavily into Python code. As a result, we 
would like to develop an installation workflow that incorporates all of SPIIR's C code 
and make it conveniently available to its core users that primarily write in python.
Developing modular Python/C interoperability for core SPIIR modules would allow 
researchers and students to directly access the optimised science code used in 
production with the convenience of Python.

## Python

The preferred installation workflow from the perspective of a Python user should be:

    pip install spiir

Therefore, all installation procedures should be encapsulated in a `setup.py` file, or a
related `setup.cfg` or `pyproject.toml` configuration file accompanying the main Python 
install script.

### Building C Extensions

Ideally, we would like to be able to install dependent C libraries (i.e. LAL, GSL, Gst) 
as an automatic part of the setup script, such that can be used by the SPIIR C codebase.
After all C code has been properly installed, we want some Python/C interoperable 
modules to be able to use the functions defined in the C code (for example, IIR 
template waveform (Qi Chu) and skymap generation (Qian Hu)).

In the ideal case, we would like to leverage any pre-installed versions of C library 
dependencies if they are detectable in path, otherwise we want to provide a means to 
download and install them automatically (and/or include an override to force install). 
Developing the build script in this way should hopefully generalise the SPIIR bulid 
process to multiple working environments (i.e. CIT, OzStar) where pre-existing 
installations can be leveraged, but also enable a complete, self-contained installation 
when no prior dependencies are installed (i.e. Dockerfile, user local computers).
