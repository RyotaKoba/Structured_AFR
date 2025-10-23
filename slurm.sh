#!/bin/bash
#SBATCH --job-name=afr_pruning
#SBATCH --partition=a6000
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=result.txt
#SBATCH --error=error.txt

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error


set -e

singularity exec --nv rkoba_pruning.sif sh start.sh
# singularity exec --nv pro6000.sif sh start.sh
# singularity exec --nv pruning.sif sh start.sh
# singularity exec --nv analyzer.sif sh start.sh
