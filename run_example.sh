#!/bin/bash

# This shell script exectutes an example as reference.
# How to use:
# > "conda env create -f conda_environment.yml"
# > "conda activate datadriven"
# > "chmod +x run_example.sh"
# > "./run_example.sh"

## Execute example run
python DataImporter.py -id 1 -S 2017-01-01T00:00 -E 2020-12-31T23:59
