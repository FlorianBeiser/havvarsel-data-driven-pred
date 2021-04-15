#!/bin/bash

# This shell script exectutes an example as reference.
# How to use:
# > "conda env create -f conda_environment.yml"
# > "conda activate datadriven"
# > "chmod +x run_example.sh"
# > "./run_example.sh"

## Execute example run
param1="air_temperature"
n1="10"

param2="wind_speed"
n2=10

param3="cloud_area_fraction"
n3=5

param4="mean(surface_downwelling_shortwave_flux_in_air PT1H)"
n4=3

param5="sum(duration_of_sunshine PT1H)"
n5=3

python FrostImporter.py --fab all -id 1 -param "$param1" -n $n1 -param "$param2" -n $n2 -param "$param3" -n $n3 -param "$param4" -n $n4 -param "$param5" -n $n5 -S 2016-01-01T00:00 -E 2020-12-31T23:59
