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
dataset_dir = 'data/lensless_videos_dataset/'
test_lensless_path = dataset_dir + 'train/lensless_events'
test_gt_path = dataset_dir + 'train/gt_events'
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


        # #Show in Plot
        # fig, ax = plt.subplots(1,3, figsize=(12,4))
        # fig.tight_layout()
        # ax[0].imshow(lensless)
        # ax[0].set_title("Lensless")
        # ax[1].imshow(gt)
        # ax[1].set_title("Ground Truth")
        # ax[2].imshow(output)
        # ax[2].set_title("Output")
        # ax[0].set_xticks([])
        # ax[0].set_yticks([])
        # ax[1].set_xticks([])
        # ax[1].set_yticks([])
        # ax[2].set_xticks([])
        # ax[2].set_yticks([])

        # plt.show()

        #Parameter statistics
        print("Statistics")
        print("Lensless max: ", gt.max())
        print("Lensless min: ", gt.min())
        print("Lensless mean: ", gt.mean())
        print("Lensless std: ", gt.std())
        
        # lensless_normalized = lensless - lensless.mean() 
        # lensless_normalized /= lensless.std()
        lensless_normalized = gt * -1

        print("Lensless_normalized max: ", lensless_normalized.max())
        print("Lensless_normalized min: ", lensless_normalized.min())
        print("Lensless_normalized mean: ", lensless_normalized.mean())
        print("Lensless_normalized std: ", lensless_normalized.std())



        #Show in Plot
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        fig.tight_layout()
        ax[0].imshow(gt)
        ax[0].set_title("Groundtruth")
        ax[1].imshow(lensless_normalized)
        ax[1].set_title("Normalized")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.show()
        brea

        

        # #Save
        # gt_path = save_path + str(result_num).zfill(3) + "_gt.png"
        # result_path = save_path + str(result_num).zfill(3) + "_output.png"
        # cv2.imwrite(gt_path, gt.numpy()*255)
        # cv2.imwrite(result_path, output.numpy()*255)


