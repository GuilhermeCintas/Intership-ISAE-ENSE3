#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=joao-guilherme.cintas@isae-supaero.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=minipadsCINTAS
#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err #STDERR
#SBATCH --time=7-00:00:00
#SBATCH --partition=long

module purge
module load python/3.7
source activate amb
python train_pads3d2_25.py
