"""A work-in-progress subpackage providing common tools for SPIIR R&D workflows.

This subpackage intends to define a collect of convenience functions and classes for 
the analysis and visualisation of research results produced by the SPIIR gravitational 
wave search pipeline.

At present, additional work must be done in order to streamline the end-to-end analysis
workflow of producing and processing data that may vary between sources (such as 
zerolags, injection data sets). As an example, functions should be careful to assume 
specific column names for input dataframes that are not controlled directly by the 
SPIIR pipeline. Alternatively, if the data is sourced from the SPIIR pipeline itself, 
this codebase will have to be kept up to date in lock step with any schema changes made 
to the pipeline data outputs as well, which introduces its own challenges as well.

For this reason, a number of these functions and classes will be subject to change as 
this library structure as well as pipeline functionality changes considerably before O4.
We recommend users use the code in this subpackage at their own risk as things may 
change without warning.
"""

from . import analysis, visualisation
