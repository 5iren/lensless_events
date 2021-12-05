#Train script

python train.py -e 50 -te 10 -b 8 -lr 0.0000050
python train.py -e 50 -te 10 -b 8 -lr 0.0000010
python train.py -e 50 -te 10 -b 8 -lr 0.0000005

python train.py -e 50 -te 10 -b 8 -lr 0.0000050 --conv_transpose
python train.py -e 50 -te 10 -b 8 -lr 0.0000010 --conv_transpose
python train.py -e 50 -te 10 -b 8 -lr 0.0000005 --conv_transpose
