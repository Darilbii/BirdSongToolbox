===============
BirdSongToolbox
===============

|ProjectStatus|_ |Version|_ |BuildStatus|_ |codecov|_ |License|_ |PythonVersions|_ |Preprint|_


.. |ProjectStatus| image:: https://www.repostatus.org/badges/latest/wip.svg
.. _ProjectStatus: https://www.repostatus.org/#wip

.. |Version| image:: https://img.shields.io/badge/version-0.1.0-blue
.. _Version: https://img.shields.io/badge/version-0.1.0-blue

.. |BuildStatus| image:: https://travis-ci.com/Darilbii/BirdSongToolbox.svg?token=ZTfpA5S7XqS8CnSq7qLL&branch=master
.. _BuildStatus: https://travis-ci.com/Darilbii/BirdSongToolbox

.. |codecov| image:: https://codecov.io/gh/Darilbii/BirdSongToolbox/branch/master/graph/badge.svg?token=GrXRs2VvMo
.. _codecov : https://codecov.io/gh/Darilbii/BirdSongToolbox

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
.. _License: https://github.com/Darilbii/BirdSongToolbox/blob/master/LICENSE

.. |PythonVersions| image:: https://img.shields.io/badge/python-3.5%7C3.6%7C3.7-blue.svg
.. _PythonVersions: https://www.python.org/

.. |Preprint| image:: https://img.shields.io/badge/preprint-to%20be%20submitted-red
.. _Preprint: https://img.shields.io/badge/preprint-to%20be%20submitted-red


A Package of Tools for Analyzing Electrophysiology Data in Free Singing birds, this package is intended to make Analysis of Sequential Vocal Motor Behavior Easier

Documentation
-------------

Documentation for the BirdSongToolbox module is underdevelopment, however it can be found at the
`documentation site <https://darilbii.github.io/BirdSongToolbox/index.html>`_

The documentation intends to include a full set of *tutorials* covering the functionality of BirdSongToolbox.


This documentation includes:

- `Overview <https://darilbii.github.io/BirdSongToolbox/overview/index.html>`_:
  with information that both motivates the importance of behavioral context and descriptions of
  important concepts when using BirdSongToolbox to analyze sequential vocal-motor behavior.
- `Tutorials <https://darilbii.github.io/BirdSongToolbox/auto_tutorials/index.html>`_:
  with a step-by-step guide through the various uses of the package
- `Glossary <https://darilbii.github.io/BirdSongToolbox/glossary.html>`_:
  which defines all the key terms that are useful in understanding the package
- `FAQ <https://darilbii.github.io/BirdSongToolbox/faq.html>`_:
  answering frequently asked questions
- `API <https://darilbii.github.io/BirdSongToolbox/api.html>`_:
  which lists and describes all the code and functionality available in the module
- `Examples <https://darilbii.github.io/BirdSongToolbox/auto_examples/index.html>`_:
  demonstrating example analyses and use cases, and other functionality



At present BirdSongToolbox has infrastructure and documentation of two versions of derived data. The older
version will likely be deprecated in the future however there is a early tutorial showing how to work with
this data format

A **Basic Introduction Tutorial** can be found `here <https://github.com/Darilbii/BirdSongToolbox/blob/master/Tutorial/1-Introduction_to_BirdSongToolbox.ipynb>`_

If you have a question about using BirdSongToolbox that doesn't seem to be covered by the documentation, feel free to
open an `issue <https://github.com/Darilbii/BirdSongToolbox/issues>`_ and ask!

Dependencies
------------

BirdSongToolbox is written in Python, and requires Python >= 3.5 to run.

It has the following required dependencies:

- `neurodsp <https://github.com/neurodsp-tools/neurodsp>`_
- `numpy <https://github.com/numpy/numpy>`_
- `scipy <https://github.com/scipy/scipy>`_ >= 0.19
- `h5py <https://github.com/h5py/h5py>`_
- `decorator <https://github.com/micheles/decorator>`_
- `PyYAML <https://github.com/yaml/pyyaml>`_
- `praatio <https://github.com/timmahrt/praatIO>`_
- `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_
- `pandas <https://github.com/pandas-dev/pandas>`_
- `matplotlib <https://github.com/matplotlib/matplotlib>`_ is needed to visualize data and behavior


There are also optional dependencies, which are not required for package to work, but offer extra functionality:

**Alternative Signal Processing Back-end**

- `mne <https://github.com/mne-tools/mne-python>`_ is a alternative back-end for signal processing.

**For Local Testing**

- `pytest <https://github.com/pytest-dev/pytest>`_ is needed to run the test suite locally
- `pytest-ordering <https://github.com/ftobia/pytest-ordering>`_ is needed for test suite to run in order locally
- `googledrivedownloader <https://github.com/ndrplz/google-drive-downloader>`_ is needed to download test data locally


We recommend using the `Anaconda <https://www.anaconda.com/distribution/>`_ distribution to manage these requirements.

Install
-------

**Stable Release Version**

**Note:** At Present there is no official Stable Release, however once there is, the below should be true
To install the latest release of BirdSongToolbox, you can install from pip:

.. code-block:: shell

    $ pip install BirdSongToolbox

**Development Version**

To get the development version (updates that are not yet published to pip), you can clone this repo.

.. code-block:: shell

    $ git clone https://github.com/Darilbii/BirdSongToolbox.git

To install this cloned copy of BirdSongToolbox, move into the directory you just cloned, and run:

.. code-block:: shell

    $ pip install .

**Editable Version**

If you want to install an editable version, for making contributions, download the development version as above, and run:

.. code-block:: shell

    $ pip install -e .

It is recommended that if you are using conda virtual environments, to first activate the specific environment you will be developing contributions on prior to running the above line

Bug Reports
-----------

Please use the `Github issue tracker <https://github.com/Darilbii/BirdSongToolbox/issues>`_ to file bug reports and/or ask questions about this project.

Contribute
----------

`BirdSongToolbox` welcomes and encourages contributions from the community!

If you have an idea of something to add to BirdSongToolbox, please start by opening an `issue <https://github.com/Darilbii/BirdSongToolbox/issues>`_.

When writing code to add to BirdSongToolbox, please follow the `Contribution Guidelines <https://github.com/Darilbii/BirdSongToolbox/blob/master/CONTRIBUTING.md>`_, and also make sure to follow our
`Code of Conduct <https://github.com/Darilbii/BirdSongToolbox/blob/master/CODE_OF_CONDUCT.md>`_.

Reference
---------

If you use this code in your project, please cite:

```
Someday we shall get published and that reference will go here!
```



Acknowledgements
----------------

Special Thanks to `Tom Donoghue <https://tomdonoghue.github.io/>`_ and the `Voytek Lab <https://voyteklab.com/>`_ who were heavily influential in the development of this readme

Contact
-------
debrown@ucsd.edu