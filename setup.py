from pathlib import Path
from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize
import numpy as np


install_requirements = [
    "wheel",
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
    "matplotlib",
    "Cython"
]

extras_requirements = {
    "tensorflow": ["tensorflow>=2.8", "tensorflow-probability", "scikit-learn"],
    "torch": ["torch", "torchaudio", "scikit-learn"],
    "pycbc": ["pycbc"],
}

extensions = [
    "src/spiir/waveform/iir/_optimizer.pyx",
    Extension(
        "spiir.waveform.iir._spiir_decomp",
        sources=["src/spiir/waveform/iir/_spiir_decomp.c"],
        include_dirs=[np.get_include()],
        libraries=["lal", "gsl", "lalinspiral"],
        extra_compile_args=['-Wall']
    ),
]

setup(
    name="spiir",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    seutp_requires=["setuptools"],
    install_requires=install_requirements,
    extras_require=extras_requirements,
    ext_modules=cythonize(extensions, language_level = "3"),
    include_package_data=True,
    description="A Python library for the SPIIR gravitational wave science pipeline.",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Daniel Tang",
    author_email="daniel.tang@uwa.edu.au",
)
