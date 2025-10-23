#!/bin/bash
#SBATCH --job-name=afr_pruning
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=result_analysis.txt
#SBATCH --error=error_analysis.txt

set -e


singularity exec --nv analyzer.sif python analyzer.py
