#!/bin/bash

#SBATCH -n 4
#SBATCH -p sjayasurgpu1
#SBATCH -q sjayasur
#SBATCH --gres=gpu:1
#SBATCH --constraint=RTX6000

#SBATCH -t 2-0:0 
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vtorresm@asu.edu

# Always purge modules to ensure consistent environments
module purge

module load anaconda/py3

source activate lensless

python train_full.py -e 10 -n e2vid_mselpips_f -cl MSE  -el LPIPS -lr 1e-4 -i ../../../../data/sjayasur/victor_lensless2/data/sequences/
python train_full.py -e 10 -n e2vid_lpips_f    -cl None -el LPIPS -lr 1e-4 -i ../../../../data/sjayasur/victor_lensless2/data/sequences/

