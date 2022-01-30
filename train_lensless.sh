#!/bin/bash

#SBATCH -n 8
#SBATCH -p sjayasurgpu1
#SBATCH -q sjayasur
#SBATCH --gres=gpu:1
#SBATCH -t 0-1:0 
#SBATCH --constraint=RTX6000
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge

module load anaconda/py3

source activate lensless

python train.py -i data/lensless_videos_dataset/ -e 100 -lr 1e-3 
echo "1 done"
python train.py -i data/lensless_videos_dataset/ -e 100 -lr 1e-4 
echo "2 done"
python train.py -i data/lensless_videos_dataset/ -e 100 -lr 1e-5
echo "3 done"
python train.py -i data/lensless_videos_dataset/ -e 100 -lr 1e-6
echo "4 done"