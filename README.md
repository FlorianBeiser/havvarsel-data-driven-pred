# havvarsel-data-driven-pred

Playpen for experimentation on and prototyping of data-driven predictions in the Havvarsel project.

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

An example on how to construct a workable dataset with data from Havvarsel-Frost and Frost (not yet THREDDS) can be executed by `run_example.sh` (read the header therein for the technicalities).

The technical functionalities are hidden in `FrostImporter.py` (remark the introduction and the NOTEs there for further technical information).
