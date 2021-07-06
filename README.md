# havvarsel-data-driven-pred

Playpen for experimentation on and prototyping of data-driven predictions in the Havvarsel project.

The repository provides code to fetch data from different sources (see `DataImporter.py` for details) and provides a testbed for the data analysis (see `DataAnalyser.py` for details).

## Requirements and set-up 

This sandbox requires the installation of python3 and some related packages:

1. For an existing version of python3 this can be ensured with
```
pip3 install -r requirements.txt
```

2. Otherwise using Miniconda, a suitable environment is created by
```
conda env create -f conda_environment.yml
conda activate datadriven
```

## How to use

Each file contains further information and a small example call in its header. To get familiar with the code, we recommend to take a look at those. 

An example on how to construct a workable dataset can be executed by `run_example.sh` (read the header therein for the technicalities) - WARNING: Long run time!
