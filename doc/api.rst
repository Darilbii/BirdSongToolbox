.. _api_documentation:

=================
API Documentation
=================

API reference for the BirdSongToolbox module.

Table of Contents
=================

.. contents::
   :local:
   :depth: 2

.. currentmodule:: BirdSongToolbox

Import Data Objects
-------------------

Classes that import data and return objects that manage both neural and behavioral data and its corresponding metadata.

Chunked Data Object
~~~~~~~~~~~~~~~~~~~

The ImportData class imports the Chunked Data format and returns it as a object for analysis.

.. autosummary::
   :toctree: generated/

   import_data.ImportData

Epoched Data Object
~~~~~~~~~~~~~~~~~~~

The Import_PrePd_Data object imports the Epoched Data format.

.. autosummary::
   :toctree: generated/

   ImportClass.Import_PrePd_Data

.. currentmodule:: BirdSongToolbox.file_utility_functions

File I/O
--------
There Are internal Mechanisms for loading and saving files.


.. currentmodule:: BirdSongToolbox.preprocess



Preprocessing: Chunks
---------------------
Preprocessing functions for the Chunked Data Format must be selected based on the structure of the dataset used.
Always check the shape information in the function documentation.

Rereference
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   common_average_reference

Filter
~~~~~~

.. autosummary::
   :toctree: generated/

   bandpass_filter
   bandpass_filter_epochs
   multi_bpf_epochs
   multi_bpf

Hilbert Transform
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   hilbert_module


.. currentmodule:: BirdSongToolbox.PreProcessClass

Preprocessing: Epochs
---------------------

The Epoched data format uses object-oriented interface which primarily uses the Pipeline class. You must first make a
Pipeline instance, which is a hard copy of the imported data. You can now pre-process the data in place using the
preprocessing methods. Once you have completed the pipeline you can close the pipe which will prevent the data from
being preprocessed further by accident.

Pipeline Object
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

    Pipeline


.. currentmodule:: BirdSongToolbox.free_epoch_tools

Song Behavior
-------------
Suite of Functions for working with Song Behavioral Labels. To use these functions on the handlabels you must first
convert the labels into a different format using this helper function.


.. autosummary::
   :toctree: generated/

    get_chunk_handlabels

Basic Label Extraction
~~~~~~~~~~~~~~~~~~~~~~
The most basic operations to operate on the handlabels are as follows.

.. autosummary::
   :toctree: generated/

    label_focus_chunk
    label_group_chunk
    label_extractor

.. currentmodule:: BirdSongToolbox.context_hand_labeling

Contextual Labels
~~~~~~~~~~~~~~~~~
More advance Contextual functions can be added to the handlabels using a properly initialized instance of this object.


.. autosummary::
   :toctree: generated/

    ContextLabels

**Contextual Label Operations**

.. autosummary::
   :toctree: generated/

    label_focus_context
    get_motif_identifier

.. currentmodule:: BirdSongToolbox.free_epoch_tools

**Long Silence Finder**
Helpful for finding long periods of silence

.. autosummary::
   :toctree: generated/

   long_silence_finder


.. currentmodule:: BirdSongToolbox.free_epoch_tools

Restructure Data using Behavior
-------------------------------
Using a selection of behavior event times you can restructure neural or behavioral data to run event related analysis.

.. autosummary::
   :toctree: generated/

    get_event_related_nd
    get_event_related
    event_clipper
    event_clipper_freqs
    get_event_related_nd_chunk
    event_clipper_nd
    event_shape_correction