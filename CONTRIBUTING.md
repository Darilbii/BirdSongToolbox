# Contributing to BirdSongToolbox

Thank you for your interest in contributing to BirdSongToolbox!

This documents is a set of guidelines for contributing to BirdSongToolbox on GitHub. This guide is meant to make it easy for you to get involved. Before contributing, please take a moment to read over this page to get a sense of the scope of the toolbox, and the preferred procedures for requesting features, reporting issues, and making contributions. 

* [Participation guidelines](#participation-guidelines)
* [How to report bugs](#how-to-report-bugs)
* [How to suggest changes or updates](#how-to-suggest-changes-or-updates)
* [How to submit code](#how-to-submit-code)
* [Style guidelines](#style-guidelines)

## Participation guidelines

Please note that this project adheres to a [code of conduct](https://github.com/Darilbii/BirdSongToolbox/blob/master/CODE_OF_CONDUCT.md), that you are expected to uphold when participating in this project. 

## How to report bugs

If you are reporting a bug, please submit it to our [issue tracker](https://github.com/Darilbii/BirdSongToolbox/issues). Please do your best to include the following:

1. A short, top-level summary of the bug (usually 1-2 sentences)
2. A short, self-contained code snippet to reproduce the bug, ideally allowing a simple copy and paste to reproduce. Please do your best to reduce the code snippet to the minimum required.
3. The actual outcome of the code snippet
4. The expected outcome of the code snippet

## How to suggest changes or updates

If there is a feature you would like to see, please submit it as an [issue](https://github.com/Darilbii/BirdSongToolbox/issues), with a brief description of what you would like to see added / changed, and why. If it is feature that you would be willing to implement, please indicate that and, and follow the guidelines below for making a contribution. 

## How to submit code

If there is a feature you would like to add, or an issue you saw that you think you can help with, you are ready to make a code submission to the project!

If you are working on a feature, please indicate so in the relevant issue, so that we can keep track of who is working on what. 

Once you're ready to start working on your contribution, do the following:

1. **[Fork](https://help.github.com/articles/fork-a-repo/) this repository**. This makes your own version of this project you can edit and use.
2. **[Make your changes](https://guides.github.com/activities/forking/#making-changes)**, adding code that add the desired functionality.
3. Check through the code to make sure it follows the projects **[style guidelines](#style-guidelines)**
4. **Submit a [pull request](https://help.github.com/articles/proposing-changes-to-a-project-with-pull-requests/)**.

If it's your first time contributing to open source, check out this free resource on [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github).

## Style guidelines

All code that is to be added to BirdSongToolbox must follow the code conventions of the project. 

BirdSongToolbox follows the following conventions:
- Code style should follow [PEP8](https://www.python.org/dev/peps/pep-0008/)
  - Merge candidate code will be checked using [pylint](https://www.pylint.org)
  - Max line length is 100 characters
- All functions should be unit tested, using [pytest](https://docs.pytest.org/en/latest/)
  - Merge candidates must pass all existing tests, and add new tests such as to not reduce test coverage
- All code shouod be documented, following the [numpy docs](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) format

For more guidelines on how to write well formated and organized code, check out the [Python API Checklist](http://python.apichecklist.com).

Special Thanks to [Tom Donoghue](https://github.com/TomDonoghue) who wrote the inspiration for this document.