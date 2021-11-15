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

    pip install pysummary

## Scripts:

`simPSF.py` simulates grayscale video captured with lensless camera and a diffraction layer (PSF convolution) 

`createTimeWindows.py` creates numpy array files per event window from an event .txt file 

`train.py` trains model defined in `model/` and saves trained model in `model/${epoch}_state_dict.pth`

`test.py` performs inference from model and state dictionary

`reconstruct.py` performs reconstruction from infered voxels using [rpg_e2vid](https://github.com/uzh-rpg/rpg_e2vid) functions