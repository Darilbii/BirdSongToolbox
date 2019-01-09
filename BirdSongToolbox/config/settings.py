""" This python file handles switching between lab servers or local paths to data"""
import platform
import pickle as pk
from pathlib import Path

# TODO: handle paths for lintu and txori
lab_servers = ['crunch', 'lintu', 'txori']
lab_paths = {'crunch': '/net/expData/birdSong/'}  # Dictionary of Lab Servers

HOSTNAME = platform.uname()[1]  # Get the identity of the host computer
DEFAULT_PATH = Path(__file__)
DEFAULT_PATH = DEFAULT_PATH.parent / 'local_config.pckl'
DATA_PATH = ''


def using_lab_server():
    """ Check if using one of the lab servers """
    for host in lab_servers:
        if host in HOSTNAME.lower():
            return True
    return False


def handle_lab_data_path():
    """ Function for handling package settings on lab computers"""
    for host in lab_servers:
        if host in HOSTNAME.lower():
            return lab_paths[host]


def handle_local_data_path():
    """ Function for handling package settings on non-lab computers"""

    local_config = DEFAULT_PATH

    if local_config.exists():
        # import local path
        data_path = load_local_data_path()  # return the data path from the local_config.pckl file

    else:

        print('You are not currently working on a lab server and there is no local_config.pckl in your package')
        interacting = True

        while interacting:

            response = input('Would you like to create one now? (Y/N)')

            if response == 'y' or response == 'Y':

                # Give User Instructions to create path
                print(
                    "To make your local config file to enable full functionality of this package you need to find where ",
                    "Birdsong Data is located on your host computer. \n Once this is done determine the full path to its" +
                    "location. \n Examples: \n Linux/Mac: /Users/Username/Documents/Data \n Windows: c:/Program Files/Data")

                create_local_config()  # Create the local_config.pckl file
                data_path = load_local_data_path()  # return the data path from the local_config.pckl file
                interacting = False

            elif response == 'n' or response == 'N':
                print('Ok then...')
                print('Note: Without a local_config.py this package will not be able to import data on this system')
                data_path = ''
                interacting = False

            else:
                print('Did not receive a Y or N. Would you like to create one now? (Y/N)')

    return data_path


def create_local_config():
    """ Create a local_config file from user input
    """

    default_path = DEFAULT_PATH

    # Make local Variables for while loop
    making = True  # Currently Making Local Config File
    counter = 0  # Counts the number of attempts to verify the data path

    # Main Section of Function
    while making:

        local_data_path = input("What is the path to the data folder on your local computer?)")

        # Verify that this local path exists
        verify = Path(local_data_path)
        verify.resolve()

        if verify.exists():
            # Create the setting.pckl file
            default_path.resolve()
            with default_path.open(mode='wb') as settings_file:
                pk.dump(local_data_path, settings_file,
                        protocol=0)  # Protocol 0 is human readable and backwards compatible
            making = False

        else:
            # Prevent While loop from running infinitely
            counter += 1
            if counter > 3:
                print("Too many Attempts")
                making = False

            print("That path doesn't work. Try again. Number of Attempts left: ", str(4 - counter))


def load_local_data_path():
    """ Reads the local_config.pckl file and Loads the local Data path"""

    default_path = DEFAULT_PATH

    with default_path.open(mode='rb') as settings_file:
        local_path = pk.load(settings_file)
        # local_path = ''

    return local_path


# Main Function for Determining Settings on Import
def main():
    """ Determines the host computer being used and determines where the data directory is on the host"""

    if using_lab_server():
        return handle_lab_data_path()
    else:
        return handle_local_data_path()


# TODO: Find a less hacky way to make this a global variable or determine if this way isn't that hacky Resource https://www.python-course.eu/python3_global_vs_local_variables.php
DATA_PATH = main()  # This is a Hacky way to get the variable to be a global variable usefule to ImportClass.py

if __name__ == main():
    main()
