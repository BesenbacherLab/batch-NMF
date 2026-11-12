# batch-NMF
pytorch implementation of NMF extentions that incorporate batch effects including Batch NMF(bNMF) and Sigmoid Filter NMF (sfNMF).

## Requirements

batch-nmf requires Python 3.9 or above.

## Installation

To get batch-nmf and test_data to test run it on, you can first clone the repo:
```
git clone https://github.com/BesenbacherLab/batch-NMF.git
```
then go to the directory:
```
cd batch-NMF
```
The command line tool can then be installed using pip:
```
pip install dist/batch_nmf-0.1.0-py3-none-any.whl
```

or using pipx:
```
pipx install dist/batch_nmf-0.1.0-py3-none-any.whl
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
Add `--seed` with any integer (e.g. `--seed 1`) to make a run repeatable; without it, every run starts from different random values.

Running this will take 1-2 minutes on a normal labtop. If you need to fit a bNMF or sfNMF model to large datasets with hundreds of samples we recommend running it on a GPU to limit the running time.

## Plotting output
The estimated signatures and filters can be plotted using the script `R_script/plot_lengths.R`. 

Running this script requires that R is installed as well as the R package "tidyverse". If R is not installed on your system it can be installed together with the necessary libraries using conda:
```
conda create -n R
conda activate R
conda install -c conda-forge r-tidyverse
```

The script can be used like this to produce a plot of the signatures:
```
./R_scripts/plot_lengths.R test_data/estimated_signatures_bNMF_2sigs.tsv signatures_bNMF_2sigs.png
```
And a plot of the estimated filters:
```
./R_scripts/plot_lengths.R test_data/estimated_filters_bNMF_2sigs.tsv filters_bNMF_2sigs.png
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


