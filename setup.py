from pathlib import Path
from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize
import numpy as np


# TODO: Investigate best method to handle installing C extension dependencies
#   see: --no-build-isolation flag ? discussion: https://github.com/pypa/pip/issues/6144
try:
    # get include/lib from pre-installed location
    # INCLUDE_DIR = 
    # LIB_DIR =
    raise NotImplementedError
except NotImplementedError as e:
    # install LAL/GSL in repo directory
    # INCLUDE = Path(__file__).parent
    # LIB_DIR = Path(__file__).parent
    # install_dependencies()
    pass

INCLUDE_DIR = (Path(__file__).parent / "include").resolve()
LIB_DIR = (Path(__file__).parent / "lib").resolve()

extensions = [
    "src/spiir/waveform/iir/_optimizer.pyx",
    Extension(
        "spiir.waveform.iir._spiir_decomp",
        sources=["src/spiir/waveform/iir/_spiir_decomp.c"],
        include_dirs=[np.get_include(), str(INCLUDE_DIR), "/usr/local/include"],
        library_dirs=[str(LIB_DIR), "/usr/local/lib"],
        libraries=["lal", "gsl", "lalinspiral"],
        extra_compile_args=['-Wall']
    ),
]

install_requirements = [
    "lalsuite",
    "ligo.skymap",
    "astropy",
    "python-ligo-lw==1.8.1",
    "igwn-alert",
    "ligo-gracedb",
    "toml",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib"
]

extras_requirements = {
    "tensorflow": ["tensorflow>=2.8", "tensorflow-probability", "scikit-learn"],
    "torch": ["torch", "torchaudio", "scikit-learn"],
    "pycbc": ["pycbc"],
}

setup(
    name="spiir",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    setup_requires=["wheel", "setuptools", "Cython"],
    install_requires=install_requirements,
    extras_require=extras_requirements,
    # ext_modules=cythonize(extensions, language_level = "3"),
    include_package_data=True,
    description="A Python library for the SPIIR gravitational wave science pipeline.",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Daniel Tang",
    author_email="daniel.tang@uwa.edu.au",
)
