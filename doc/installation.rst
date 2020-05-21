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

Installation
------------
**Stable Release Version**

**Note:** At Present there is no official Stable Release, however once there is, the below should be true:

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

If you want to install an editable version, for making contributions, download the development version as above,
and run:

.. code-block:: shell

    $ pip install -e .

It is recommended that if you are using conda virtual environments, to first activate the specific environment you
will be developing contributions on prior to running the above line

Configuration
-------------

BirdSongToolbox can handle choosing the proper path to import and save data. To do this you must first configure
the config.yaml file. By default this file is empty and structured as shown below:

.. code-block:: yaml

    Chunked_Data_Path: ''
    Intermediate_Path: ''
    PrePd_Data_Path: ''
    Raw_Data_Path: ''
    User_Defined_Paths: {}

Although not necesary to run analysis on already imported and properly formated data it is convenient to use native
import functions to ensure all data and meta-data are available for analysis. If this file is not configure
BirdSongToolbox will give the user warnings when using functions that need thiss configuration. This configuration file
will be used by BirdSongToolbox to determine where to read and write data. This file only needs to be configured one
time to work, and there are several helper functions that will allow the user to update the paths.

There are 5 default paths, which are as follows:

* **Chunked_Data_Path**: Default path to the directory that contains derived data in the *Chunked* data format.
* **Intermediate_Path**: Default path that BirdSongtoolbox will use when saving results or custom files. This
  path is for convenience and is not necessary to work with BirdSongToolbox.
* **PrePd_Data_Path**: Default path to the directory that contains derived data in the *Epoched* Data format.
* **Raw_Data_Path**: The default path to the directory that contains the contains raw data.
* **User_Defined_Paths**: Dictionary of paths that the users can use to define there own set of custom paths. These
  paths are to the users discretion and for their own convenience.

**Example Configuration Steps:**

.. code-block:: python

    # Import the helper functions for altering the config.yaml
    from BirdSongToolbox.config.utils import update_config_path

    # Set the New Path
    update_config_path(specific_path="Chunked_Data_Path", new_path='<Your-Default-Chunk-Path>' )

Replace ``<Your-Default-Chunk-Path>`` with the path to the location of the directory containing derived data in the
*Chunked* data format. If desired you can use the ``update_config_path()`` function for the other default paths. Once done
restart your python kernel. BirdSongToolbox should now be properly configured. A simple test would be to print out the
configuration's current default paths. Such as in the lines below.

.. code-block:: python

    # Import the helper functions for altering the config.yaml
    from BirdSongToolbox.config.utils import get_spec_config_path

    # Get the current path configuration for PrePd_Data_Path and print it
    print(get_spec_config_path("PrePd_Data_Path"))

.. note:: If you face difficulty with this configuration

    Please use the `Github issue tracker <https://github.com/Darilbii/BirdSongToolbox/issues>`_ to file bug reports
    and/or ask questions about this project.





