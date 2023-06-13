#!/bin/bash
#SBATCH --job-name="sam+clip"
#SBATCH -p dgx2q # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 1  # number of cores
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

module load slurm

# Prefered method using srun
source /home/hanna/sam-with-clip-video/venv/bin/activate
srun python main.py "$1"
deactivate
