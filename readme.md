# BirdSongToolbox
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Build Status](https://travis-ci.com/Darilbii/BirdSongToolbox.svg?token=ZTfpA5S7XqS8CnSq7qLL&branch=master)](https://travis-ci.com/Darilbii/BirdSongToolbox)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A Package of Tools for Analysing Electrophysiology Data in Free Singing birds, this package is intended to make Analysis of Sequential Vocal Motor Behavior Easier

## Documentation

Documentation for the BirdSongToolbox module is underdevelopment.

The documentation also intends to include a full set of *tutorials* covering the functionality of BirdSongToolbox.

A **Basic Introduction Tutorial** can be found [here](https://github.com/Darilbii/BirdSongToolbox/blob/master/Tutorial/1-Introduction_to_BirdSongToolbox.ipynb)

If you have a question about using NeuroDSP that doesn't seem to be covered by the documentation, feel free to
open an [issue](https://github.com/Darilbii/BirdSongToolbox/issues) and ask!

## Install

**Stable Release Version**

**Note:** At Present there is no official Stable Release, however once there is, the below should be true
To install the latest release of BirdSongToolbox, you can install from pip:

`$ pip install BirdSongToolbox`

**Development Version**

To get the development version (updates that are not yet published to pip), you can clone this repo.

`$ git clone https://github.com/Darilbii/BirdSongToolbox.git`

To install this cloned copy of neurodsp, move into the directory you just cloned, and run:

`$ pip install .`

**Editable Version**

If you want to install an editable version, for making contributions, download the development version as above, and run:

`$ pip install -e .`

It is recommended that if you are using conda virtual environments, to first activate the specific environment you will be developing contributions on prior to running the above line

## Bug Reports

Please use the [Github issue tracker](https://github.com/Darilbii/BirdSongToolbox/issues) to file bug reports and/or ask questions about this project.

## Contribute

`BirdSongToolbox` welcomes and encourages contributions from the community!

If you have an idea of something to add to BirdSongToolbox, please start by opening an [issue](https://github.com/Darilbii/BirdSongToolbox/issues).

When writing code to add to NeuroDSP, please follow the [Contribution Guidelines](https://github.com/Darilbii/BirdSongToolbox/blob/master/CONTRIBUTING.md), and also make sure to follow our
[Code of Conduct](https://github.com/Darilbii/BirdSongToolbox/blob/master/CODE_OF_CONDUCT.md).

## Reference

If you use this code in your project, please cite:

```
Someday we shall get published and that reference will go here!
```

## Acknowledgements

Special Thanks to [Tom Donoghue](https://tomdonoghue.github.io/) and the [Voytek Lab](https://voyteklab.com/) who were heavily influential in the development of this readme

## Contact
debrown@ucsd.edu

----------------------------------
# Stuff Below was dumped here a long time ago
----------------------------------

## General Workflow of this Package:

Import Class: Import data for use
PreProcessClass.py: Pre-process Data as desired

Third Do analysis on Data

----------------------------------
Dumping This Here as Test...

# 1. Benefits for Annotations:

## Python Shorthands that make reading documentation easy
- Type the function name Func() then within the parenthesis hit: Shift + Tab to show discription of Function
- Type Func? then hit Shift + Enter to show parameters of Function
- Type Func?? then hit Shift + Enter to show entire Documentation and Function Code

# Default Neural Oscillation Frequency bands from literature (WIkipedia)
## Brain waves
- Delta wave – (0.2 – 3 Hz)
- Theta wave – (4 – 7 Hz)
- Alpha wave – (8 – 13 Hz)
- Mu wave – (7.5 – 12.5 Hz)
- SMR wave – (12.5 – 15.5 Hz)
- Beta wave – (16 – 31 Hz)
- Gamma wave – (32 – 100 Hz)

#### Others: delta (1–4 Hz), theta (4–8 Hz), beta (13–30 Hz),low gamma (30–70 Hz), and high gamma (70–150 Hz)


For Class Pre-Processing:

The Processing Functions all follow the same general Steps:
    - [1] Validate proper steps have been made and Necessary Object Instances exist
        - [1.1] Check Pipeline is still Open
        - [1.2] Check Dependencies Exist
    - [2] Back-up Neural Data in case of Mistake [Make_Backup(self)]
    - [3] Do User Specified Processing on Song Neural Data
    - [4] Do User Specified Processing on Silence Neural Data
    - [5] Update the Process Log with User Defined Steps (Done Last Incase of Error)



