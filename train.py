#import os
import argparse
import torch
from torch.nn.modules import conv
from torchvision import transforms
from torchsummary import summary
import numpy as np
from model.unet import UNet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.nn_utils import lenslessEventsVoxel, lenslessEvents


def train(epochs, test_epochs, learning_rate, dataset_dir, batch_size, num_bins, conv_transpose):

    #Set paths
    train_lensless_path = dataset_dir + 'train/lensless_events'
    train_gt_path = dataset_dir + 'train/gt_events'
    test_lensless_path = dataset_dir + 'test/lensless_events'
    test_gt_path = dataset_dir + 'test/gt_events'

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
    train_data = lenslessEventsVoxel(train_lensless_path, train_gt_path, num_bins, transform = transform)
    test_data = lenslessEventsVoxel(test_lensless_path, test_gt_path, num_bins, transform = transform)

    #Create dataloaders
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    print("       Train dataset length: ", len(train_data))
    print("       Test dataset length: ", len(test_data))

    #Load Model
    print("[INFO] Loading model...")
    net = UNet(num_bins,num_bins, bilinear= (not conv_transpose)) #Bilinear True for upsample, False for ConvTranspose2D
    net.to(device)
    summary(net, (num_bins, 260, 348)) #prints summary

    #Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

    #Train loop
    print("[INFO] Training...")
    train_loss = []
    test_loss = []

    for epoch in range(1,epochs+1):
        train_running_loss = 0
        test_running_loss = 0

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
            train_running_loss += float(loss.item())

        #Test
        with torch.no_grad():
            for data in testloader:
                #Get image and ground truth
                test_lensless,  test_gt = data[0].to(device), data[1].to(device)

                #Forward 
                test_output = net(test_lensless)
                loss = criterion(test_output,  test_gt)
                test_running_loss += float(loss.item())


        #Print Statistics
        train_loss.append(train_running_loss / len(trainloader))
        test_loss.append(test_running_loss / len(testloader))


        print("[%3d / %3d] Train loss: %.6f | Test loss: %.6f " % (epoch, epochs, 
                                                train_running_loss / len(trainloader),
                                                test_running_loss / len(testloader))
                                                )

        if epoch % test_epochs == 0:
            # #Get train example from CUDA to display
            # lensless = np.transpose(lensless[0].cpu().detach().numpy(), (1,2,0))
            # gt = np.transpose(gt[0].cpu().detach().numpy(), (1,2,0))
            # output = np.transpose(output[0].cpu().detach().numpy(), (1,2,0))

            # #Get test example from CUDA to display
            # test_lensless = np.transpose(test_lensless[0].cpu().detach().numpy(), (1,2,0))
            # test_gt = np.transpose(test_gt[0].cpu().detach().numpy(), (1,2,0))
            # test_output = np.transpose(test_output[0].cpu().detach().numpy(), (1,2,0))

            # #Limit channels to display
            # lensless = lensless[:,:,1:4]
            # gt = gt[:,:,1:4]
            # output = output[:,:,1:4]

            # test_lensless = test_lensless[:,:,1:4]
            # test_gt = test_gt[:,:,1:4]
            # test_output = test_output[:,:,1:4]

            # #Normalize before displaying
            # lensless = ( lensless - lensless.min() ) / ( lensless.max() - lensless.min() )
            # gt = ( gt - gt.min() ) / ( gt.max() - gt.min() )
            # output = ( output - output.min() ) / ( output.max() - output.min() )

            # test_lensless = ( test_lensless - test_lensless.min() ) / ( test_lensless.max() - test_lensless.min() )
            # test_gt = ( test_gt - test_gt.min() ) / ( test_gt.max() - test_gt.min() )
            # test_output = ( test_output - test_output.min() ) / ( test_output.max() - test_output.min() )


            # #Draw figure with input, groundtruth and output from training data
            # fig, ax = plt.subplots(1,3, figsize=(12,4))
            # fig.tight_layout()
            # ax[0].imshow(lensless)
            # ax[0].set_title('Lensless')
            # ax[0].set_xticks([])
            # ax[0].set_yticks([])
            # ax[1].imshow(gt)
            # ax[1].set_title('Groundtruth')
            # ax[1].set_xticks([])
            # ax[1].set_yticks([])
            # ax[2].imshow(output)
            # ax[2].set_title('Output')
            # ax[2].set_xticks([])
            # ax[2].set_yticks([])
            # fig.savefig('results/train/'+str(epoch).zfill(3) + '_comparisons.png')

            # #Draw figure with input, groundtruth and output from test data
            # fig0, ax0 = plt.subplots(1,3, figsize=(12,4))
            # fig0.tight_layout()
            # ax0[0].imshow(test_lensless)
            # ax0[0].set_title('Lensless')
            # ax0[0].set_xticks([])
            # ax0[0].set_yticks([])
            # ax0[1].imshow(test_gt)
            # ax0[1].set_title('Groundtruth')
            # ax0[1].set_xticks([])
            # ax0[1].set_yticks([])
            # ax0[2].imshow(test_output)
            # ax0[2].set_title('Output')
            # ax0[2].set_xticks([])
            # ax0[2].set_yticks([])
            # fig0.savefig('results/test/'+str(epoch).zfill(3) + '_comparisons.png')
                                                  
            #Save trained model
            torch.save(net.state_dict(), 'model/relu_up'+str(not conv_transpose)+'_e'+str(epoch)+'_lr'+str(learning_rate)+'_state_dict.pth')

    #Save Loss graphic
    fig1, ax1 = plt.subplots()
    ax1.plot(train_loss, color = 'blue', label="Train")
    ax1.plot(test_loss, color = 'red', label="Test")
    ax1.legend()
    ax1.set_title("Losses")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_ylim(top = 1.1*max(max(train_loss), max(test_loss)) , bottom = 0.9*min(min(train_loss), min(test_loss)))
    fig1.savefig('results/plots/relu_up'+str(not conv_transpose)+'_e'+str(epochs)+'_lr'+str(learning_rate)+'_losses.png')
    np.save('results/plots/relu_up'+str(not conv_transpose)+'_e'+str(epochs)+'_lr'+str(learning_rate)+'_train', train_loss)
    np.save('results/plots/relu_up'+str(not conv_transpose)+'_e'+str(epochs)+'_lr'+str(learning_rate)+'_test', test_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",  "--dataset_dir",     help="directory",                   default="data/lensless_videos_dataset/")
    parser.add_argument("-e",  "--epochs",          help="total number of epochs",      type=int,   default=300)
    parser.add_argument("-te", "--test_epochs",     help="epochs to produce result",    type=int,   default=5)
    parser.add_argument("-lr", "--learning_rate",   help="for adam optimizer",          type=float, default=.001)
    parser.add_argument("-b",  "--batch_size",      help="batch size for training",     type=int,   default=4)
    parser.add_argument("-c",  "--num_bins",        help="number of bins or channels",  type=int,   default=5)
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

    #Print info in console
    print("-------Training Parameters--------")
    print("----------------------------------")
    print("Epochs:                  ", epochs)
    print("Test epochs:             ", test_epochs)
    print("Learning rate:           ", learning_rate)
    print("Batch size:              ", batch_size)
    print("Using conv trasnpose:    ", conv_transpose)
    print("----------------------------------")

    #Train 
    train(epochs, test_epochs, learning_rate, dataset_dir, batch_size, num_bins, conv_transpose)

    print("Done!")