#!/usr/bin/env bash

#SBATCH --account=hhd34
#SBATCH --partition=gpus
#SBATCH --nodes=8
#SBATCH --ntasks=32
### SBATCH --ntasks-per-node=4
### SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=5
### SBATCH --ntasks-per-core=1
### SBATCH --threads-per-core=1

#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --mail-user=g.pineda-garcia@sussex.ac.uk
#SBATCH --mail-type=ALL

####  # --exclusive

source venv3/bin/activate

python3 fexplorer.py
