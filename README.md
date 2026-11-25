# batch-NMF
PyTorch implementation of NMF extensions that model batch effects: Batch NMF (bNMF) and Sigmoid Filter NMF (sfNMF).

## Requirements

batch-nmf requires Python 3.10 or above, and git (pip uses it to download the code).

## Installation

Install the command line tool `batch-nmf` directly from GitHub with pip, preferably in a virtual environment:
```
python -m venv batch-nmf-env
source batch-nmf-env/bin/activate
pip install git+https://github.com/BesenbacherLab/batch-NMF.git
```

To install a specific version, add its tag (or a commit hash) at the end:
```
pip install git+https://github.com/BesenbacherLab/batch-NMF.git@1.0.0
```

[uv](https://docs.astral.sh/uv/) and [pipx](https://pipx.pypa.io) can install it the same way, each in an environment of its own:
```
uv tool install git+https://github.com/BesenbacherLab/batch-NMF.git
pipx install git+https://github.com/BesenbacherLab/batch-NMF.git
```

Check that it works with:
```
batch-nmf --version
```

batch-nmf depends on PyTorch. On Linux, pip installs the GPU (CUDA) version of PyTorch by default, which is a download of several GB. If you will only run batch-nmf on CPUs, install the CPU version of PyTorch first:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/BesenbacherLab/batch-NMF.git
```

## Usage

You can use `-h` or `--help` to see the different options:
```
batch-nmf -h
```

The program has two main commands "fit" and "predict". "fit" is used to learn both a set of signatures and their weights from an input matrix. "predict" uses a known set of signatures and infers their weight from the input matrix.

`--method` chooses the model:

- `bNMF` (the default): batch NMF.
- `sfNMF`: sigmoid filter NMF.
- `adam`: standard NMF without batch effects, fitted with the Adam optimizer like bNMF and sfNMF.
- `scipy`: standard NMF without batch effects, fitted with scikit-learn's multiplicative updates.

bNMF and sfNMF need to know which batch each sample comes from: give `--batch_file`, or `--all_separate_batch` to treat every sample as its own batch. `adam` and `scipy` use neither.

The example data are in the `test_data` directory of the repository. To try them, clone the repository and go to its directory:
```
git clone https://github.com/BesenbacherLab/batch-NMF.git
cd batch-NMF
```

The following command will fit a bNMF model to the input matrix in the test_data directory:

```
batch-nmf fit --input_matrix test_data/length_data.tsv --batch_file test_data/batch_file.tsv --exposure_file test_data/estimated_exposures_bNMF_2sigs.tsv --signature_file test_data/estimated_signatures_bNMF_2sigs.tsv --n_signatures 2 --filter_file test_data/estimated_filters_bNMF_2sigs.tsv
```
Add `--seed` with any integer (e.g. `--seed 1`) to make a run repeatable; without it, every run starts from different random values.

Running this takes a minute or two on a laptop. Larger datasets take longer: with several hundred samples and many signatures, a fit can take an hour or more. batch-nmf uses a GPU automatically if PyTorch finds one. On CPUs, set `OMP_NUM_THREADS` to the number of cores it may use (for example `export OMP_NUM_THREADS=2` in a cluster job with 2 cores).

## Plotting output
The estimated signatures and filters can be plotted using the script `R_scripts/plot_lengths.R` in the repository.

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

For development we use [uv](https://docs.astral.sh/uv/) to handle dependencies. Clone the repository and install the project with all its dependencies in a virtual environment (`.venv`):
```
git clone https://github.com/BesenbacherLab/batch-NMF.git
cd batch-NMF
uv sync
```

The code is installed in editable mode, so changes take effect without reinstalling. To run the tool in this environment:
```
uv run batch-nmf -h
```
or, from any directory, with the full path `/path/to/batch-NMF/.venv/bin/batch-nmf`.

Run the tests with:
```
make test
```

Add a dependency (here numpy) with:
```
uv add numpy
```

The version number is set in `pyproject.toml`. To make a new version, bump it with `uv version` (which also updates `uv.lock`), commit, and tag the commit so that it can be installed with `@<version>`:
```
uv version --bump minor    # or --bump patch / --bump major
git commit -am "chore: version $(uv version --short)"
git tag $(uv version --short)
git push && git push --tags
```
