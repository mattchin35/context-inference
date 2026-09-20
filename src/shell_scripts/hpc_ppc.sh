#!/bin/bash

# Which partition to use
#SBATCH -p unlimited

# Name to identify job
#SBATCH --job-name=ppc_cluster

# Number of threads
#SBATCH -n 16

# Number of tasks total (1 per node)
#SBATCH --ntasks=1

# Number of tasks per node
#--tasks-per-node=1

# Memory per node.
#SBATCH --mem=128gb

# Time limit to run script
#SBATCH -t 72:00:00

#Where to save the output (log)
#SBATCH -o /gs/gsfs0/users/mchin1/logs/ppc_cluster_%j.log

#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

#SBATCH --mail-user=matthew.chin@einsteinmed.edu # Where to send mail

# Job submission for slurm scheduler

# prepare module/environment needs
echo "Sourcing .bashrc..."
source ~/.bashrc
echo "Activating conda environment..."
conda activate spikesort  # replace spikesort with your conda environment name
echo PYTHONPATH="$HOME/context_inference" python src/<code_name>.py
PYTHONPATH="$HOME/context_inference" python src/<code_name>.py
echo "Done!"
