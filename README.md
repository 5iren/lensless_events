# Lensless Events

## Requirements

### Libraries:
Pytorch:

    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

OpenCV:

    pip install opencv-python

Matplotlib: 

    pip install matplotlib

Pandas:

    conda install pandas

pysummary:

    pip install torchsummary

## Scripts:

`train.py` trains UNet model and saves trained model in `results/model/{name}.pth`

`reconstruct.py` performs inference and reconstruction from infered voxels using [rpg_e2vid](https://github.com/uzh-rpg/rpg_e2vid) functions

`plots.ipynb` plots losses