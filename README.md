# The Context-Unified Encoding (CUE) model

A spiking neural network model extending the ordinal serial encoding (OSE) model
and temporal context model (TCM) to integrate short- and long-term memory.


## Installation

```bash
git clone https://github.com/ctn-archive/cue-model.git
cd cue-model
pip install .
```

## Requirements

* Python 3.5 or later  (earlier versions might work, but are untested)

Further dependencies should be installed automatically during the installation:

* [Nengo 2.5 or later](https://www.nengo.ai/)
* [nengo_spa 0.3.x](https://github.com/nengo/nengo_spa)
* [nengo_extras @d63e12aa787419fcafed32027105583d614e9e6d](https://github.com/nengo/nengo_extras)
* matplotlib
* NumPy
* SciPy
* Pandas
* [PyTry](https://github.com/tcstewar/pytry)
* [Psyrun](https://github.com/jgosmann/psyrun)
* SciPy
* Seaborn
* statsmodels


## Running the model

Single trials can be run with the `pytry` command and one of the files in
`cue/trials` as argument. For example:

```bash
pytry cue/trials/default.py --data_format npz --seed 42
```

Use the `--help` argument to get a list of command line options and parameters:

```bash
pytry cue/trials/default.py --help
```

To run larger sets of simulations with different seeds use the `psy run`
command. Without arguments it will run multiple simulations for all
experimental conditions. To run only selected experimental conditions, pass
them as arguments. For example:

```bash
psy run immediate delayed
```

To list available experimental conditions use `psy list`.


## Data evaluation and plots

For data evaluation and plots see the [Jupyter](http://jupyter.org/) notebooks
in the `notebooks` directory.
