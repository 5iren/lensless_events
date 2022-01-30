#!/bin/bash

#SBATCH -n 4
#SBATCH -p sjayasurgpu1
#SBATCH -q sjayasur
#SBATCH --gres=gpu:1
#SBATCH -t 5-0:0 
#SBATCH --constraint=RTX6000
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge

module load anaconda/py3

source activate lensless

python train.py -e 200 -lr 1e-5 