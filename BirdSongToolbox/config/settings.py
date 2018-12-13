""" This python file handles switching between lab servers or local paths to data"""
import platform
from pathlib import Path

## Check what computer is being used
## if computer is one of the lab servers
##     set the data path to be the corresponding data path
## elif the computer isn't one of the lab computers
##     check if their is a file called local_config.py
#         if the file exists
#             read the files contents and add them to the global scope
#         else
#             tell the user that they need to create this file
#             ask if they want to make this file now
#                 if yes
#                      tell them that you are going to make this file now
#                      ask the user what the data file path is for the local computer
#                 if no
#                     inform the user that the package will import but will not be able to import new data

#TODO: handle paths for lintu and txori
lab_servers = ['crunch', 'lintu', 'txori']
lab_paths = {'crunch': '/net/expData/birdSong/'}

HOSTNAME = platform.uname()[1]

for host in lab_servers:
    if host in HOSTNAME:
        DATAPATH = lab_paths[host]
    else:
        local_config = Path('BirdSongToolbox/config/local_config.py')
        local_config.resolve()
        if local_config.exists():
            # import local path
            pass






