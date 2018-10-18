# Mathematical Modelling Project
[![Build Status](https://travis-ci.com/JakobGM/mathematical-modelling.svg?branch=master)](https://travis-ci.com/JakobGM/mathematical-modelling)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Project in Mathematical Modelling (TMA4195) at NTNU


## Project setup

### Project dependencies

To install project dependencies, run

```
$ pip install -r requirements.txt
```

from the project repository root path.


### Pre-commit hooks

The [pre-commit](https://pre-commit.com/) tool (part of the project dependencies installed above) is used in order to install pre-commit hooks into the git repository.

The hooks ensure that all tests, including linters, pass before you commit.

To install these hooks, run

```
$ pre-commit install
```


## Development

### Code style

The [black](https://github.com/ambv/black) python code formatter is used in order to enforce a consistent style.
The test suite will fail if code does not adhere to the black style.

It is recommended to enable automatic (on-save) black formatting for your editor-of-choice. Here are some editor instructions:

* [PyCharm](https://github.com/ambv/black#pycharm).
* [Vim](https://github.com/ambv/black#vim).
