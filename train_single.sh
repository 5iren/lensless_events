#!/bin/bash

#SBATCH -n 8
#SBATCH -p sjayasurgpu1
#SBATCH -q sjayasur
#SBATCH --gres=gpu:1
#SBATCH -t 1-0:0 
#SBATCH --constraint=RTX6000
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge

module load anaconda/py3

source activate lensless

python train_singleUN.py -e 202 -lr 1e-4 -b 32 -l MSE -i data/single_gaussian/ -a 0.5 -w 2
python train_singleUN.py -e 205 -lr 1e-4 -b 32 -l MSE -i data/single_gaussian/ -a 0.5 -w 5
python train_singleUN.py -e 210 -lr 1e-4 -b 32 -l MSE -i data/single_gaussian/ -a 0.5 -w 10
