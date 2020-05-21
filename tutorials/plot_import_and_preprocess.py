"""
Import and Pre-process Data
===========================

This example shows how to import Data and do a basic preprocess pipeline
"""

###############################################################################
from BirdSongToolbox.import_data import ImportData
import BirdSongToolbox.free_epoch_tools as fet
from BirdSongToolbox.context_hand_labeling import ContextLabels, label_focus_context
from BirdSongToolbox.behave.behave_utils import event_array_maker_chunk, get_events_rasters, repeat_events

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