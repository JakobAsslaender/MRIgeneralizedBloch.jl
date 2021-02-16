#!/bin/bash
#SBATCH -p cpu_short,cpu_medium,cpu_long
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-100
#SBATCH -t 0-06:00:00

## module load julia
echo $HOSTNAME
echo $SLURM_ARRAY_TASK_ID
~/julia-1.5.2/bin/julia --project=~/Documents/Julia/MT_generalizedBloch ~/Documents/Julia/MT_generalizedBloch/Fit/Fit_gBloch_MatApprox_Slice_20210205.jl

wait

