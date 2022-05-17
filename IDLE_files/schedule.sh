#!/bin/bash

#SBATCH --job-name='testjob'
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=testjob-%j-stdout.log
#SBATCH --error=testjob-%j-stderr.log
#SBATCH --time=01:00:00

source ~/.lenstronomyenv/bin/activate
echo "Submitting Slurm job"
python Doing_Fits.py