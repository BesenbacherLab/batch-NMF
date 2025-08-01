# pytorch-nmf

Train NMF (and extensions of NMF) using pytorch

## Requirements

pytorch-nmf requires Python 3.9 or above.

## Development:

For development we use [poetry](https://python-poetry.org) to handle dependencies.
If poetry is not installed on your system you can install poetry using pipx:
```
pipx install poetry
```
Once poetry is installed you can install the current project including all dependencies using the following command:
```
poetry install
```

### Using poetry to run the software

poetry creates a virtual environment so that the current project is isolated from the rest of the system. To run a command in the poetry virtual environment you type `poetry run {command}`. So to run the bbq tool you can type:
```
poetry run pytorch-nmf
```
This only works if you are in the current directory. If you want to run the code from a different directory you can first make sure that the current version is installed in the poetry environment by running `poetry install` and then using the `which` command to get the full path of the installed version:
```
poetry install
poetry run which pytorch-nmf
```
That will return a full path to an executable that can be run from any directory.


### Using poetry to add dependencies

If we want to use numpy in our program we can add it as a dependency using the command:
```
poetry add numpy
```
poetry will then find a version of numpy that fits with the requirements of other packages and install it.


### Usage

You can use `-h` or `--help` to see the different options:
```
poetry run pytorch-nmf -h
```
