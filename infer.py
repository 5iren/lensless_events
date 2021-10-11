import os
import torch
import lpips
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from unet import UNet
from torch.utils.data import DataLoader
from dogNoisy import dogNoisy, UnNormalize
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from psnr import PSNR

def saturateImage(image):
    """ Saturates image to avoid over or underflow (0-255)
        image: CxWxH tensor (float)
        returns: saturated image (uint8)
        """
    image *= 255
    image = np.array(image)
    image = np.transpose(image, (1,2,0))

    b, g, r  = cv2.split(image)
    b[b > 255] = 255
    b[b < 0] = 0
    g[g > 255] = 255
    g[g < 0] = 0
    r[r > 255] = 255
    r[r < 0] = 0
    image = cv2.merge((b, g, r))

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image.astype(np.uint8)


#Set paths
test_dataset_path = 'data/dog_dataset_noisy/test/'
test_gt_path = 'data/dog_dataset_noisy/test_gt/'
model_path = 'state_dict.pth'

#Load CUDA
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("[INFO] Using device: ", device)

#Create datasets
print("[INFO] Loading dataset...")

#Transforms
mean = (0.5, 0.5, 0.5)
std = (0.2, 0.2, 0.2)
transform = transforms.Compose([transforms.ConvertImageDtype(torch.float),
                                transforms.Normalize(mean, std)])
unorm = UnNormalize(mean, std)

#Load datasets
test_data = dogNoisy(test_dataset_path, test_gt_path, transform=transform)

#Create dataloaders
testloader = DataLoader(test_data, batch_size=1, shuffle=True)

#Load trained Model
net = UNet(3,3)
net.load_state_dict(torch.load(model_path))
net.eval()

#Loss function and optimizer
criterion = torch.nn.MSELoss()


#Train loop
print("[INFO] Inference...")
test_loss = []
test_psnr = []
test_running_loss = 0
test_running_psnr = 0

#Test
with torch.no_grad():

    for data in testloader:

        noisy, gt = data

        output = net(noisy)

        image = saturateImage(unorm(noisy[0]))
        output = saturateImage(unorm(output[0]))
        gt_image = saturateImage(unorm(gt[0]))

        noisy_psnr = PSNR(gt_image, image)
        output_psnr = PSNR(gt_image, output)

        fig, ax = plt.subplots(1,3)
        ax[0].imshow(image)
        ax[0].set_title("Noisy - PSNR: {:.2f} dB".format(noisy_psnr))
        ax[1].imshow(output)
        ax[1].set_title("Output - PSNR: {:.2f} dB".format(output_psnr))
        ax[2].imshow(gt_image)
        ax[2].set_title("Ground Truth")

        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].axes.get_yaxis().set_visible(False)
        ax[1].axes.get_xaxis().set_visible(False)
        ax[1].axes.get_yaxis().set_visible(False)
        ax[2].axes.get_xaxis().set_visible(False)
        ax[2].axes.get_yaxis().set_visible(False)

        plt.show()


