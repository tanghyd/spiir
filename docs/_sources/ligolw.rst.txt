=================
LIGO Light-Weight
=================

Version |version|

The LIGO LW XML Format
======================

LIGO Light-Weight (LIGO LW) is the name of a XML-based data format used across the LIGO 
Scientific Collaboration.

.. note::
   
   This documentation is a work in progress! Hold tight...


Legacy ilwd:char Types
======================

LIGO Light Weight XML “ilwd:char” types are a legacy data type used to represent unique 
IDs as strings of the form “table:column:integer” - for example 
``“process:process_id:10”`` or  ``"postcoh:event_id:1"``. Large LIGO LW XML ocuments may 
have many millions of these  strings and as a result are an extremely memory 
inefficient method of tracking ids. However, a number of projects within the LIGO 
Scientific Collaboration (LSC) still use ilwd:char types, especially if they are 
dependent on data generated or collected from previous science observation runs.

There has been a push to deprecate all usage of ilwd:char types, with a number of 
scripts and packages offering functionality to convert ilwd:char types to a more simple 
integer based id format. For example, the `python-ligo-lw`_ package offers utilities to 
convert LIGO LW XML documents with ilwd:char types to their corresponding modern 
integer version, and the Python gravitational wave astrophysics library `gwpy`_ also 
offers automatic ilwd:char compatibility for a standard set of LSC LIGO LW Table 
schemas as well.

However, collaboration-wide support for this this legacy data type will not be 
maintained forever, with some packages deprecating support for any ilwd:char types as
of Python3.10 and onwards - making it imperative for users to use the updated formats 
in their research and development workflow as soon as possible.

In the case of SPIIR, a number of LIGO LW XML documents generated from the Python2.7 
version of the pipeline may still contain ilwd:char types. To complicate matters, SPIIR 
uses its own custom LIGO LW Table format called ``PostcohInspiralTable`` to record 
post-coherence coincident trigger data as candidate events for the pipeline, and so it 
may be the case that LIGO LW XML Documents with this ``"postcoh"`` table may still have 
ilwd:char types as well. However, as it is a custom type, it is not automatically 
supported out of the box by modern packages such as `python-ligo-lw`_ or `gwpy`_, and 
they must be handled as a special case.

.. _python-ligo-lw: https://git.ligo.org/kipp.cannon/python-ligo-lw
.. _gwpy: https://gwpy.github.io/


Legacy Compatibility
====================

This package provides the tools to help parse custom `PostcohInspiralTable` table 
formats with automatic compatibility support for legacy ilwd:char types and/or table 
schemas (i.e. changing column names) during SPIIR's ongoing development updates.