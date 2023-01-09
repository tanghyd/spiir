from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

# configure C extensions

INCLUDE_DIR = (Path(__file__).parent / "include").resolve()
LIB_DIR = (Path(__file__).parent / "lib").resolve()

extensions = [
    "src/spiir/data/waveform/iir/_optimizer.pyx",
    Extension(
        "spiir.data.waveform.iir._spiir_decomp",
        sources=["src/spiir/data/waveform/iir/_spiir_decomp.c"],
        include_dirs=[np.get_include(), str(INCLUDE_DIR), "/usr/local/include"],
        library_dirs=[str(LIB_DIR), "/usr/local/lib"],
        libraries=["lal", "gsl", "lalinspiral"],
        extra_compile_args=['-Wall'],
    ),
]

# specify optional dependencies
igwn_alert_requires = ["igwn-alert", "toml"]
p_astro_requires = [
    "scikit-learn>=1.0",
    "pycbc @ git+https://github.com/gwastro/pycbc.git@master#egg=pycbc",
    "p_astro @ git+https://git.ligo.org/spiir-group/p-astro.git@feature/enable_pickle_compat#egg=p_astro",
]

extras_require = {
    "p-astro": p_astro_requires,
    "igwn-alert": igwn_alert_requires,
}

extras_require["all"] = [pkg for pkgs in extras_require.values() for pkg in pkgs]

# install package
setup(
    name="spiir",
    version="0.0.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "wheel",
        "setuptools",
        "lalsuite",
        "astropy",
        "python-ligo-lw>=1.8.1",
        "ligo.skymap",
        "ligo-gracedb",
        "numpy>=1.23",
        "scipy",
        "pandas",
        "matplotlib",
        "click",
    ],
    extras_require=extras_require,
    ext_modules=cythonize(extensions, language_level="3"),
    include_package_data=True,
    description="A Python library for the SPIIR gravitational wave science pipeline.",
    author="Daniel Tang",
    author_email="daniel.tang@uwa.edu.au",
)
