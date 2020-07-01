#!/bin/bash

source venv3/bin/activate

rm -fr L2L-OMNIGLOT/run_results L2L-OMNIGLOT/trajectories/
./clear_experiment_output.sh

python fexplorer.py
