#import os
import argparse
from tabnanny import verbose
import torch
from torch.nn.modules import conv
from torchvision import transforms
from torchsummary import summary
import numpy as np
from model.unet import UNet
#from model.unet2 import UNet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.nn_utils import lenslessEventsVoxelUN
import time
from tqdm import tqdm


def train(epochs, test_epochs, learning_rate, dataset_dir, batch_size, num_bins, loss_fn, optim, conv_transpose):

    #Set paths
    train_lensless_path = dataset_dir + "lensless"
    train_gt_path = dataset_dir + "gt"

    #File name
    arch = 'Unet'
    fileName = 'arch('+arch+')-e('+str(epochs)+')-l('+str(loss_fn)+')-o('+str(optim)+')-lr('+str(learning_rate)+')'

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
    whole_dataset = lenslessEventsVoxelUN(train_lensless_path, train_gt_path, num_bins, transform = transform)

    #Train/Test split
    train_data, test_data = torch.utils.data.random_split(whole_dataset, [int(len(whole_dataset)*0.7), len(whole_dataset)-int(len(whole_dataset)*0.7)])

    #Create dataloaders
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    print("       Train dataset length: ", len(train_data))
    print("       Test dataset length:  ", len(test_data))
    print("       Train minibatches:    ", len(trainloader))
    print("       Test  minibatches:    ", len(testloader))

    #Load Model
    print("[INFO] Loading model...")
    net = UNet(num_bins,num_bins, bilinear= (not conv_transpose)) #Bilinear True for upsample, False for ConvTranspose2D
    net.to(device)
    #summary(net, (num_bins, 260, 348)) #prints summary

    #Loss function and optimizer
    if loss_fn == 'MSE':
        criterion = torch.nn.MSELoss()
    elif loss_fn == 'L1':
        criterion = torch.nn.L1Loss()
    else:
        print("Loss Function not recognized (MSE, L1)")

    if optim == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr = learning_rate)
    else:
        print("Optimizer not recognized (Adam, AdamW)")

    #Learning rate decay
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)    

    #Train loop
    print("[INFO] Training...")
    train_loss = []
    test_loss = []
    train_loss_e = []
    test_loss_e = []

    for epoch in range(1,epochs+1):
        train_running_loss = 0
        test_running_loss = 0

        #Train
        net.train()
        #for data in tqdm(trainloader):
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
            train_loss.append(float(loss.item()))
            train_running_loss += float(loss.item())
        
        #scheduler.step()

        #Test
        net.eval()
        with torch.no_grad():
            #for data in tqdm(testloader):
            for data in testloader:
                #Get image and ground truth
                test_lensless,  test_gt = data[0].to(device), data[1].to(device)

                #Forward 
                test_output = net(test_lensless)
                loss = criterion(test_output,  test_gt)
                test_loss.append(float(loss.item()))
                test_running_loss += float(loss.item())


        #Print Statistics
        train_loss_e.append(train_running_loss / len(trainloader))
        test_loss_e.append(test_running_loss / len(testloader))
        print("[%3d / %3d] Train loss: %.6f | Test loss: %.6f " % (epoch, epochs, 
                                                train_running_loss / len(trainloader),
                                                test_running_loss / len(testloader))
                                                )
        
                                                  
    #Save trained model
    torch.save(net.state_dict(), 'results/model/'+fileName+'.pth')

    #Save Loss in array
    np.save('results/plots/'+fileName+'_train', train_loss)
    np.save('results/plots/'+fileName+'_test', test_loss)
    np.save('results/plots/'+fileName+'_train_e', train_loss_e)
    np.save('results/plots/'+fileName+'_test_e', test_loss_e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",  "--dataset_dir",     help="directory",                   default="data/single_gaussian/")
    parser.add_argument("-e",  "--epochs",          help="total number of epochs",      type=int,   default=20)
    parser.add_argument("-te", "--test_epochs",     help="epochs to produce result",    type=int,   default=5)
    parser.add_argument("-lr", "--learning_rate",   help="for adam optimizer",          type=float, default=1e-5)
    parser.add_argument("-b",  "--batch_size",      help="batch size for training",     type=int,   default=4)
    parser.add_argument("-c",  "--num_bins",        help="number of bins or channels",  type=int,   default=5)
    parser.add_argument("-l",  "--loss_fn",         help="Loss function",               type=str,   default='MSE')
    parser.add_argument("-o",  "--optim",           help="Optimizer",                   type=str,   default='Adam')
    parser.add_argument("--conv_transpose",         help="use conv_transpose",          action='store_true')

    #Get arguments
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    epochs = args.epochs
    test_epochs = args.test_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_bins = args.num_bins
    conv_transpose = args.conv_transpose
    loss_fn = args.loss_fn
    optim = args.optim

    #Print info in console
    print("-------Training Parameters--------")
    print("----------------------------------")
    print("Dataset directory:       ", dataset_dir)
    print("Epochs:                  ", epochs)
    print("Bins:                    ", num_bins)
    print("Batch size:              ", batch_size)
    print("Loss function:           ", loss_fn)
    print("Optimizer:               ", optim)
    print("Learning rate:           ", learning_rate)
    print("Using conv transpose:    ", conv_transpose)
    print("----------------------------------")

    #Train 
    train(epochs, test_epochs, learning_rate, dataset_dir, batch_size, num_bins, loss_fn, optim, conv_transpose)

    print("Done!")