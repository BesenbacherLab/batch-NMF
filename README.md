# batch-NMF
pytorch implementation of NMF extentions that incorporate batch effects including Batch NMF(bNMF) and Sigmoid Filter NMF (sfNMF).

## Requirements

batch-nmf requires Python 3.9 or above.

## Installation

kmerPaPa can be installed using pip:
```
pip install batch-nmf
```

or using pipx:
```
pipx install batch-nmf
```

## Usage

You can use `-h` or `--help` to see the different options:
```
batch-nmf -h
```

The program has two main commands "fit" and "predict". "fit" is used to learn both a set of signatures and their weights from an input matrix. "predict" uses a known set of signatures and infers their weight from the input matrix.

The following command will fit a bNMF model to the input matrix in the test_data directory:

```
batch-nmf fit --input_matrix test_data/length_data.tsv --batch_file test_data/batch_file.tsv --exposure_file test_data/estimated_exposures_bNMF_2sigs.tsv --signature test_data/estimated_signatures_bNMF_2sigs.tsv --n_signatures 2 --filter_file test_data/estimated_filters_bNMF_2sigs.tsv
```


## Development

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

poetry creates a virtual environment so that the current project is isolated from the rest of the system. To run a command in the poetry virtual environment you type `poetry run {command}`. So to run the batch_nmf tool you can type:
```
poetry run batch_nmf
```
This only works if you are in the current directory. If you want to run the code from a different directory you can first make sure that the current version is installed in the poetry environment by running `poetry install` and then using the `which` command to get the full path of the installed version:
```
poetry install
poetry run which batch-nmf
```
That will return a full path to an executable that can be run from any directory.

### Using poetry to add dependencies

If we want to use numpy in our program we can add it as a dependency using the command:
```
poetry add numpy
```
poetry will then find a version of numpy that fits with the requirements of other packages and install it.


