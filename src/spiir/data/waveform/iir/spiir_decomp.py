"""
This module is a wrapper of the _spiir module, supplementing the C
code in that module with additional features that are more easily
implemented in Python.  It is recommended that you import this module
rather than importing _spiir directly.

Original author: Qi Chu
"""


from ._spiir_decomp import iir, iirinnerproduct, iirresponse
