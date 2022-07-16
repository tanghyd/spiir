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
`“process:process_id:10”`` or  `"postcoh:event_id:1"`. Large LIGO LW XML ocuments may 
have many millions of these  strings and as a result are an extremely memory 
inefficient method of tracking ids. However, a number of projects within the LIGO 
Scientific Collaboration (LSC) still use ilwd:char types, especially if they are 
dependent on data generated or collected from previous science observation runs.

There has been a push to deprecate all usage of ilwd:char types, with a number of 
scripts and packages offering functionality to convert ilwd:char types to a more simple 
integer based id format. For example, the `python-ligo-lw`` package offers scripts to 
convert any LIGO LW XML document with ilwd:char types to corresponding integer types, 
and the Python gravitational wave data processing library `gwpy` also offers legacy 
ilwd:char compatibility handling as well.

In the case of SPIIR, a number of LIGO LW XML documents generated from the Python2.7 
version of the pipeline may still contain ilwd:char types. To complicate matters, SPIIR 
uses its own custom LIGO LW Table format called `PostcohInspiralTable` to record 
post-coherence coincident trigger data as candidate events for the pipeline, and so it 
may be the case that LIGO LW XML Documents with this `"postcoh"` table may still have 
ilwd:char types as well. However, as it is a custom type, it is not automatically 
supported out of the box by modern packages such as `python-ligo-lw` or `gwpy`, and 
they must be handled as a special case.

SPIIR Legacy Compatibility
==========================

Thankfully, this package provides the tools to assist with parsing SPIIR's custom 
`PostcohInspiralTable` formats with support for ilwd:char types as well as any changing 
column data schemas that may occur during the ongoing development work of porting its 
legacy Python2.7 codebase to Python3.