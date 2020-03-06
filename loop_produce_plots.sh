#!/bin/bash

source ./venv3/bin/activate 

for i in {1..1000}
do
    python per_gen_analysis.py
#     scp *pdf unix.sussex.ac.uk:~
    sleep 2h
done
