import os
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from model.unet import UNet
from torch.utils.data import DataLoader
from src.nn_utils import lenslessEventsVoxel, lenslessEvents
import matplotlib.pyplot as plt
from PIL import Image
import cv2


#Set paths
dataset_dir = 'data/lensless_videos_dataset/'
test_lensless_path = dataset_dir + 'test/lensless_events'
test_gt_path = dataset_dir + 'test/gt_events'
model_path = 'model/state_dict.pth'
save_path = 'results/inference_results/'

#Load CUDA
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print("[INFO] Using device: ", device)

#Create datasets
print("[INFO] Loading dataset...")

#Transforms
transform = None
# mean = (0, 0, 0)
# std = (0.2, 0.2, 0.2)
#transform = transforms.Compose([transforms.Normalize(mean, std)])
#unorm = UnNormalize(mean, std)

#Create datasets
print("[INFO] Loading dataset and dataloader...")
test_data = lenslessEventsVoxel(test_lensless_path, test_gt_path, transform = transform)

#Create dataloaders
testloader = DataLoader(test_data, batch_size=1, shuffle=True)

#Load trained Model
net = UNet(3,3)
net.load_state_dict(torch.load(model_path))
net.eval()

#Loss function and optimizer
criterion = torch.nn.MSELoss()

#Infer loop
print("[INFO] Performing inference on {} examples".format(len(testloader)))
test_loss = []
test_running_loss = 0

#Test
result_num = 0
with torch.no_grad():

    for data in testloader:
        result_num +=1
        lensless, gt = data
        output = net(lensless)

        #Transpose to display
        lensless = np.transpose(lensless[0], (1,2,0))
        gt = np.transpose(gt[0], (1,2,0))
        output = np.transpose(output[0], (1,2,0))

        lensless_new = ( lensless - lensless.min() ) / ( lensless.max() - lensless.min() )
        gt_new = ( gt - gt.min() ) / ( gt.max() - gt.min() )
        output_new = ( output - output.min() ) / ( output.max() - output.min() )



        #Show in Plot
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        fig.tight_layout()
        ax[0].imshow(lensless_new)
        ax[0].set_title("Lensless")
        ax[1].imshow(gt_new)
        ax[1].set_title("Ground Truth")
        ax[2].imshow(output_new)
        ax[2].set_title("Output")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[2].set_xticks([])
        ax[2].set_yticks([])

        plt.show()



