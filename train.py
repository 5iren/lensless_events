import os
import torch
import lpips
from torch._C import get_default_dtype
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.io import write_jpeg
import numpy as np
from model.unet import UNet
from torch.utils.data import DataLoader
from dogNoisy import dogNoisy, UnNormalize
import matplotlib.pyplot as plt
from PIL import Image
import cv2


#Set paths
train_dataset_path = 'data/dog_dataset_noisy/train/'
test_dataset_path = 'data/dog_dataset_noisy/test/'
train_gt_path = 'data/dog_dataset_noisy/train_gt/'
test_gt_path = 'data/dog_dataset_noisy/test_gt/'

#Load CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("[INFO] Using device: ", device)

#Create datasets
print("[INFO] Loading dataset...")

#Transforms
mean = (0.5, 0.5, 0.5)
std = (0.2, 0.2, 0.2)
transform = transforms.Compose([transforms.ConvertImageDtype(torch.float),
                                transforms.Normalize(mean, std)])
unorm = UnNormalize(mean, std)

train_data = dogNoisy(train_dataset_path, train_gt_path, transform=transform)
test_data = dogNoisy(test_dataset_path, test_gt_path, transform=transform)

# lensless_path = 'data/lensless_videos_dataset/lensless_events'
# gt_path = 'data/lensless_videos_dataset/gt_events'
# #lensless_data, gt_data = lenslessEvents(lensless_path, gt_path)
# event_dataset = lenslessEvents(lensless_path, gt_path)

# lensless, gt = event_dataset[0]
# print(lensless.mean())
# print(gt.mean())

#Create dataloaders
trainloader = DataLoader(train_data, batch_size=1, shuffle=True)
testloader = DataLoader(test_data, batch_size=1, shuffle=False)

#Load Model
net = UNet(3,3)
net.to(device)

#Training parameters
learning_rate = 0.00001
epochs = 125

#Loss function and optimizer
#criterion = lpips.LPIPS(net='vgg').to(device)
criterion = torch.nn.MSELoss()
#criterion = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(net.parameters(), lr = learning_rate, betas = (.9, .999), eps=.00000001)

#Train loop
print("[INFO] Training...")
train_loss = []
test_loss = []
train_psnr = []
test_psnr = []
for epoch in range(epochs):
    train_running_loss = 0
    test_running_loss = 0
    train_running_psnr = 0
    test_running_psnr = 0

    #Train
    for data in trainloader:
        #Get image and ground truth
        noisy, gt = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        #Forward + backward + optimize
        output = net(noisy)
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()

        #Compute PSNR
        image = unorm(output.detach().cpu()[0])
        gt_image = unorm(gt.detach().cpu()[0])
        image = saturateImage(image)
        gt_image = saturateImage(gt_image)
        train_running_psnr += PSNR(gt_image, image)

    #Test
    with torch.no_grad():
        for data in testloader:
            #Get image and ground truth
            noisy, gt = data[0].to(device), data[1].to(device)

            #Forward 
            output = net(noisy)
            loss = criterion(output, gt)
            test_running_loss += loss.item()

            #Compute PSNR
            image = unorm(output.detach().cpu()[0])
            gt_image = unorm(gt.detach().cpu()[0])
            image = saturateImage(image)
            gt_image = saturateImage(gt_image)
            test_running_psnr += PSNR(gt_image, image)

    #cv2.imwrite('results/' + str(epoch+1).zfill(3) + '.jpeg', image)

    #Print Statistics
    train_loss.append(train_running_loss / len(trainloader))
    test_loss.append(test_running_loss / len(testloader))

    train_psnr.append(train_running_psnr / len(trainloader))
    test_psnr.append(test_running_psnr / len(testloader))

    print("[%3d / %3d] Train loss: %.4f | Train PSNR: %.4f | Test loss: %.4f | Test PSNR: %.4f" % (epoch + 1, epochs, 
                                                                                 train_running_loss / len(trainloader), 
                                                                                 train_running_psnr / len(trainloader), 
                                                                                 test_running_loss / len(testloader),
                                                                                 test_running_psnr / len(testloader)))

#Save trained model
torch.save(net.state_dict(), 'results/state_dict.pth')

#Save Loss graphic
fig1, ax1 = plt.subplots()
ax1.plot(train_loss, color = 'blue', label="Train")
ax1.plot(test_loss, color = 'red', label="Test")
ax1.legend()
ax1.set_title("Train and test loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_ylim(top = 1, bottom= 0)
fig1.savefig('results/losses.png')

#Save PSNR graphic
fig2, ax2 = plt.subplots()
ax2.plot(train_psnr, color = 'green', label="Train")
ax2.plot(test_psnr, color = 'purple', label="Test")
ax2.legend()
ax2.set_title("Train and test PSNR")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("PSNR (dB)")
ax2.set_ylim(top = 31, bottom= 27)
fig2.savefig('results/psnr.png')