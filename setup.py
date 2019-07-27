from setuptools import setup, find_packages

"""
===============
BirdSongToolbox
===============

"""

setup(
    name='BirdSongToolbox',
    version='0.1.0dev',
    packages=find_packages(),
    url='',
    license='Apache License, 2.0',
    author='Daril Brown II',
    author_email='',
    description='',
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'h5py', 'scikit-learn', 'decorator', 'praatio'],
    keywords=['neuroscience', 'neural oscillations', 'bird song', 'spectral analysis', 'LFP'],
    tests_require=['pytest', 'googledrivedownloader', 'pytest-ordering'],
    extra_requires={
        'mne': ['mne']
    }
)