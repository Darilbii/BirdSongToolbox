"""
Import and Pre-process Data
===========================

This example shows how to import Data and do a basic preprocess pipeline
"""

###############################################################################
from BirdSongToolbox.import_data import ImportData
from BirdSongToolbox.preprocess import multi_bpf, hilbert_module, common_average_reference_array

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import inspect

###############################################################################
# Basic Workflow
# --------------
#
# When using BirdSongToolbox you will need to have data in which you want to work on
#
# Although not required, it is convenient to use the Import Classes to import properly formated
# derived data this way you can guarantee that you have all of the data and meta data needed and
# that it is all synchronized with each other
#
# As this example assumes that you have not yet configured BirdSongToolbox to automatically know
# where to look for data we will go through the optional step of telling the toolbox where we
# would like it to impor from. In this case we will import data used for testing the package.
# (Don't worry about this if you are unfamiliar with pytest)
#

###############################################################################


# Select bird_id and session
bird_id = 'z007'
session = 'day-2016-09-09'

###############################################################################

src_file_path = inspect.getfile(lambda: None)
project_dir = Path(src_file_path).resolve().parents[1]
print(project_dir)
data_dir = project_dir / "BirdSongToolbox" / "data" / "Chunk_Data_Demo"

# Import Data
zdata = ImportData(bird_id=bird_id, session=session, location=data_dir)

###############################################################################
# The neural and audio data files BirdSongToolbox works with are almost alway numpy ndarrays with
# their last dimension being samples. This allows for versitility in functionality and computational
# simplicity. However, to Demonstrate the most basic api functionality we are going to select a
# single chunk to preprocess.
#
# *Note:* This is not always necessary, as their are wrapper functions that can work the most common
# versions of various preprocessed data



###############################################################################

neural_data = zdata.song_neural[0]

###############################################################################
# Common Average Reference
# ------------------------
#
# Typically it is useful to rereference neural data to remove noise or motion artifacts. For the test
# data it is convenient to use a Common Average Reference (CAR). Basically you subtract the mean of all
# of your neural channels from each channels. The logic here is that most of what is commone accross all
# of your channels would be dominated by noise and this would improve the Signal to Noise ratio.
#
# When doing a CAR it is best to exclude channels that are known to be noisy or unrealiable to prevent
# introducing noise into your other channels. So we are going to declare which channels to exclude from
# the CAR.

###############################################################################

# Common Average Reference
bad_channels = [24, 28]  # Define Bad Channels
car_data = common_average_reference_array(neural_data=neural_data, bad_channels=bad_channels)  # CAR

###############################################################################
# Band Pass Filter
# ----------------
#
# Next we can bandpass filter the data to get out frequencies that we want. Depending on your own preference
# this filtering will be done using either neurodsp or mne (optional dependency). For this example we will
# be using mne.
#
# When using the multi_bpf function you can make as many band pass filters as you want. Each filtered data
# ndarray will be returned in a list
#
# No matter your preferred backend you can print out detailed reports on the filters used using the verbose
# parameter

###############################################################################
fs = 1000  # Neural Data is sampled at 1 kHz
l_freqs = [10, 100]  # The lower bound of the band pass filters
h_freqs = [20, 200]  # The Upper bound of the band pass filter

filtered_data = multi_bpf(chunk_neural_data=car_data, fs=fs, l_freqs=l_freqs, h_freqs=h_freqs,
                          verbose=True)
filtered_data = np.asarray(filtered_data)  # Make a view as a ndarray



