"""SPIIR subpackage for astrophysical source classification, or "p_astro".

Each module in the SPIIR p_astro subpackage separates a particular concern or
approach in estimating astrophysical source classification. Typically, we aim to
provide the relevant operations for each approach as a set of functions in their
respective modules.

When models are composed together or fit with a set of default values, we construct one
or more classes inside a submodule of p_astro.models that have the approach `.fit()`
and `.predict()` methods, so they can be efficiently and interchangeably with eachother
during testing and deployment.
"""

from . import bayes_factor, mchirp_area, models
