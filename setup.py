from setuptools import find_packages, setup

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
    description="A Python library for the SPIIR gravitational wave science pipeline.",
    author="Daniel Tang",
    author_email="daniel.tang@uwa.edu.au",
)
