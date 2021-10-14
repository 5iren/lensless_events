#import os
import torch
from torchvision import transforms
#import torchvision.transforms.functional as F
import numpy as np
from model.unet import UNet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#from PIL import Image
#import cv2
from utils import lenslessEventsVoxel, lenslessEvents


#Set paths
lensless_path = 'data/lensless_videos_dataset/lensless_events'
gt_path = 'data/lensless_videos_dataset/gt_events'

#Load CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("[INFO] Using device: ", device)

#Transforms
transform = None
# mean = (0, 0, 0)
# std = (0.2, 0.2, 0.2)
#transform = transforms.Compose([transforms.Normalize(mean, std)])
#unorm = UnNormalize(mean, std)

#Create datasets
print("[INFO] Loading dataset and dataloader...")
train_data = lenslessEventsVoxel(lensless_path, gt_path, transform = transform)
#train_data = lenslessEvents(lensless_path, gt_path)

#Create dataloaders
trainloader = DataLoader(train_data, batch_size=1, shuffle=True)
#testloader = DataLoader(test_data, batch_size=1, shuffle=False)

#Load Model
print("[INFO] Loading model...")
net = UNet(3,3)
net.to(device)

#Training parameters
learning_rate = 0.001
epochs = 10

#Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

#Train loop
print("[INFO] Training...")
train_loss = []
#test_loss = []

for epoch in range(epochs):
    train_running_loss = 0
    #test_running_loss = 0

    #Train
    for data in trainloader:
        #Get image and ground truth
        lensless, gt = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        #Forward + backward + optimize
        output = net(lensless)
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()

        

    #Test
    # with torch.no_grad():
    #     for data in testloader:
    #         #Get image and ground truth
    #         lensless, gt = data[0].to(device), data[1].to(device)

    #         #Forward 
    #         output = net(lensless)
    #         loss = criterion(output, gt)
    #         test_running_loss += loss.item()


    #Print Statistics
    train_loss.append(train_running_loss / len(trainloader))


    print("[%3d / %3d] Train loss: %.6f" % (epoch + 1, epochs, 
                                            train_running_loss / len(trainloader)))

    # lensless = np.transpose(lensless[0][0].cpu().detach().numpy(), (1,2,0))
    # gt = np.transpose(gt[0][0].cpu().detach().numpy(), (1,2,0))
    # output = np.transpose(output[0][0].cpu().detach().numpy(), (1,2,0))

    lensless = lensless[0][0].cpu().detach().numpy()
    gt = gt[0][0].cpu().detach().numpy()
    output = output[0][0].cpu().detach().numpy()

    
    fig0, ax0 = plt.subplots(1,3)
    ax0[0].imshow(lensless)
    ax0[0].set_title('Lensless')
    ax0[1].imshow(gt)
    ax0[1].set_title('Groundtruth')
    ax0[2].imshow(output)
    ax0[2].set_title('Output')
    fig0.savefig('results/'+str(epoch+1).zfill(3) + '_comparisons.png')

                                                            

#Save trained model
#torch.save(net.state_dict(), 'model/state_dict.pth')

#Save Loss graphic
fig1, ax1 = plt.subplots()
ax1.plot(train_loss, color = 'blue', label="Train")
#ax1.plot(test_loss, color = 'red', label="Test")
ax1.legend()
ax1.set_title("Train loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_ylim(top = max(train_loss)*1.1 , bottom= 0)
fig1.savefig('plots/losses.png')