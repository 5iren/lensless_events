import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import ReflectionPad2d
from torch.utils.data import DataLoader
from src.nn_utils import lenslessEventsVoxelSeq
from src.rec_utils import load_model
from model.unet import UNet
import matplotlib.pyplot as plt
import lpips

def display_postprocessing(lensless_v, gt_v, reconstruction_v):
    #Detach from GPU
    lensless_v = lensless_v.detach().cpu().numpy()
    gt_v = gt_v.detach().cpu().numpy()
    reconstruction_v = reconstruction_v.detach().cpu().numpy()

    #Transpose
    lensless_v = np.transpose(lensless_v[0], (1,2,0))
    gt_v = np.transpose(gt_v[0], (1,2,0))
    reconstruction_v = np.transpose(reconstruction_v[0], (1,2,0))

    #Limit to 3 ch
    lensless_v = lensless_v[:,:,1:4]
    gt_v = gt_v[:,:,1:4]
    reconstruction_v = reconstruction_v[:,:,1:4]

    #Normalize to display
    lensless_v = ( lensless_v - lensless_v.min() ) / ( lensless_v.max() - lensless_v.min() )
    gt_v = ( gt_v - gt_v.min() ) / ( gt_v.max() - gt_v.min() )
    reconstruction_v = ( reconstruction_v - reconstruction_v.min() ) / ( reconstruction_v.max() - reconstruction_v.min() )

    return lensless_v, gt_v, reconstruction_v

def train(epochs, learning_rate, dataset_dir, batch_size, num_bins, arch_name, frozen_e2vid, cnn_loss, e2vid_loss, cnn_loss_weight, e2vid_loss_weight):

    ########## Set paths ##########
    train_lensless_path = dataset_dir + 'train/lensless'
    train_gt_path = dataset_dir + 'train/gt'
    test_lensless_path = dataset_dir + 'test/lensless'
    test_gt_path = dataset_dir + 'test/gt'
    fileName = arch_name

    ########## Load CUDA ##########
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[INFO] Using device: ", device)

    ########## Load datasets ##########
    #Create datasets
    print("[INFO] Loading dataset and dataloader...")
    train_data = lenslessEventsVoxelSeq(train_lensless_path, train_gt_path, num_bins, transform = None)
    test_data = lenslessEventsVoxelSeq(test_lensless_path, test_gt_path, num_bins, transform = None)

    #Create dataloaders
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print("       Train dataset length: ", len(train_data))
    print("       Test dataset length:  ", len(test_data))
    print("       Train minibatches:    ", len(trainloader))
    print("       Test  minibatches:    ", len(testloader))


    ########## Models ##########
    #Load Custom Model
    print("[INFO] Loading models...")
    rec_cnn = UNet(num_bins, num_bins) 
    rec_cnn.to(device)
    #summary(net, (num_bins, 260, 348)) #prints summary

    #Load pre-trained E2VID model
    model_path = 'model/E2VID_lightweight.pth.tar'
    gt_e2vid = load_model(model_path, device)
    gt_e2vid.eval()
    #Freeze
    for param in gt_e2vid.parameters():
        param.requires_grad = False
    gt_e2vid.to(device)

    #Load pre-trained E2VID model to finetune
    lensless_e2vid = load_model(model_path, device)
    #Freeze if argument passed
    if frozen_e2vid:
        print("[INFO] E2VID Frozen")
        for param in lensless_e2vid.parameters():
            param.requires_grad = False
    else:
        print("[INFO] Finetuning E2VID")
    lensless_e2vid.to(device)
    
    ########## Loss and optimizer ##########
    #CNN Loss 
    if cnn_loss == "MSE" or cnn_loss == "mse":
        print("[INFO] CNN Loss: MSE")
        compute_cnn_loss = True
        cnn_loss = torch.nn.MSELoss()
    else:
        print("[INFO] CNN Loss: NONE")
        compute_cnn_loss = False

    #E2VID Loss
    if e2vid_loss == "LPIPS" or cnn_loss == "lpips":
        print("[INFO] E2VID Loss: LPIPS")
        compute_e2vid_loss = True
        e2vid_loss = lpips.LPIPS(net='vgg').to(device)
    else:
        print("[INFO] E2VID Loss: NONE")
        compute_e2vid_loss = False
    
    optimizer = torch.optim.Adam(list(rec_cnn.parameters()) + list(lensless_e2vid.parameters()), lr = learning_rate)

    ########## Train loop ##########
    print("[INFO] Training...")
    train_loss_s = []
    test_loss_s = []
    train_loss_e = []
    test_loss_e = []
    min_test_loss = 10

    for epoch in range(1,epochs+1):
        train_running_loss = 0
        test_running_loss = 0
       
        #Train
        rec_cnn.train()
        lensless_e2vid.train()
        #for data in trainloader:
        for data in tqdm(trainloader):
            lensless_prev_state = None
            gt_prev_state = None
            seq_running_loss = 0

            #optimizer.zero_grad()

            #Get image and ground truth
            lenslessf_v, gtf_v = data[0].to(device), data[1].to(device)
            seq_len = lenslessf_v.shape[1]

            #for ii in tqdm(range(seq_len)):
            for ii in range(seq_len):

                # zero the parameter gradients
                optimizer.zero_grad()
                
                lensless_v = lenslessf_v[:,ii]
                gt_v = gtf_v[:,ii]

                #Forward 
                reconstruction_v = rec_cnn(lensless_v)

                #CNN Loss
                if compute_cnn_loss:
                    cnn_rec_loss = cnn_loss(reconstruction_v, gt_v)
                else:
                    cnn_rec_loss = 0
                
                ######## E2VID ##########
                #Pad (required by e2vid)
                m = ReflectionPad2d((3,3,2,2))
                reconstruction_v = m(reconstruction_v)
                gt_v = m(gt_v)

                #Forward pass (if prev_states = None, no recurrency)
                lensless_rec, lensless_state = lensless_e2vid(reconstruction_v, prev_states = lensless_prev_state)
                gt_rec, gt_prev_state = gt_e2vid(gt_v, prev_states = gt_prev_state)

                #Detach to avoid double gradient descent
                lensless_prev_state = []
                lensless_prev_state.append((lensless_state[0][0].detach(),lensless_state[0][1].detach()))
                lensless_prev_state.append((lensless_state[1][0].detach(),lensless_state[1][1].detach()))
                lensless_prev_state.append((lensless_state[2][0].detach(),lensless_state[2][1].detach()))

                #Crop (DAVIS specific numbers) and detach
                lensless_rec = lensless_rec[:, 0, 2:262,3:349]#.detach().numpy()
                gt_rec = gt_rec[:, 0, 2:262,3:349]#.detach().numpy()
                #######################

                #Convert to three channels for lpips
                lensless_rec = torch.unsqueeze(lensless_rec, 1)
                lensless_rec = lensless_rec.expand(-1,3,-1,-1)
                gt_rec = torch.unsqueeze(gt_rec, 1)
                gt_rec = gt_rec.expand(-1,3,-1,-1)

                #E2VID Loss
                if compute_e2vid_loss:
                    e2vid_rec_loss = e2vid_loss(lensless_rec, gt_rec).mean()
                else:
                    e2vid_rec_loss = 0
                
                #Loss
                loss = cnn_loss_weight*cnn_rec_loss + e2vid_loss_weight*e2vid_rec_loss

                #Backward 
                loss.backward()

                #Optimize
                optimizer.step()

                #Accumulate frame loss to compute sequence loss
                seq_running_loss += float(loss.item())

            #Save loss per sequence on numpy array 
            train_loss_s.append(seq_running_loss / seq_len)

            #Accumulate per sequence loss to compute epoch loss
            train_running_loss += (seq_running_loss / seq_len)
            
        #Save loss per epoch 
        train_loss_curr = train_running_loss / len(trainloader)
        train_loss_e.append(train_loss_curr)

        #Postprocessing for display
        lensless_v, gt_v, reconstruction_v = display_postprocessing(lensless_v, gt_v, reconstruction_v)

        #Save train figure
        fig, ax = plt.subplots(2,3, figsize = (18, 10))
        ax[0][1].imshow(gt_rec[0][0].detach().cpu().numpy(), cmap = 'gray')
        ax[0][1].set_title("Groundtruth reconstruction")
        ax[0][1].axis('off')
        ax[0][2].imshow(lensless_rec[0][0].detach().cpu().numpy(), cmap = 'gray')
        ax[0][2].set_title("Deblurred reconstruction")
        ax[0][2].axis('off')
        ax[0][0].axis('off')
        ax[1][0].imshow(lensless_v, cmap = 'gray')
        ax[1][0].set_title("Blurred Voxel")
        ax[1][0].axis('off')
        ax[1][1].imshow(gt_v, cmap = 'gray')
        ax[1][1].set_title("GT Voxel")
        ax[1][1].axis('off')
        ax[1][2].imshow(reconstruction_v, cmap = 'gray')
        ax[1][2].set_title("Deblurred Voxel")
        ax[1][2].axis('off')
        fig.suptitle(f"Train Loss: {loss.item()}")
        plt.savefig(f"results/samples/{arch_name}/train_{epoch:03}.png")
        
        #Test
        rec_cnn.eval()
        lensless_e2vid.eval()
        with torch.no_grad():
            for data in testloader:
            #for data in tqdm(testloader):
                lensless_prev_state = None
                gt_prev_state = None
                seq_running_loss = 0

                #Get image and ground truth
                lenslessf_v, gtf_v = data[0].to(device), data[1].to(device)
                seq_len = lenslessf_v.shape[1]

                #for ii in tqdm(range(seq_len)):
                for ii in range(seq_len):
                    
                    lensless_v = lenslessf_v[:,ii]
                    gt_v = gtf_v[:,ii]

                    #Forward 
                    reconstruction_v = rec_cnn(lensless_v)

                    #CNN Loss
                    if compute_cnn_loss:
                        cnn_rec_loss = cnn_loss(reconstruction_v, gt_v)
                    else:
                        cnn_rec_loss = 0
                    
                    ######## E2VID ##########
                    #Pad (required by e2vid)
                    m = ReflectionPad2d((3,3,2,2))
                    reconstruction_v = m(reconstruction_v)
                    gt_v = m(gt_v)

                    #Forward pass (if prev_states = None, no recurrency)
                    lensless_rec, lensless_state = lensless_e2vid(reconstruction_v, prev_states = lensless_prev_state)
                    gt_rec, gt_prev_state = gt_e2vid(gt_v, prev_states = gt_prev_state)

                    #Crop (DAVIS specific numbers) and detach
                    lensless_rec = lensless_rec[:, 0, 2:262,3:349]#.detach().numpy()
                    gt_rec = gt_rec[:, 0, 2:262,3:349]#.detach().numpy()
                    #######################

                    #Convert to three channels for lpips
                    lensless_rec = torch.unsqueeze(lensless_rec, 1)
                    lensless_rec = lensless_rec.expand(-1,3,-1,-1)
                    gt_rec = torch.unsqueeze(gt_rec, 1)
                    gt_rec = gt_rec.expand(-1,3,-1,-1)

                    #E2VID Loss
                    if compute_e2vid_loss:
                        e2vid_rec_loss = e2vid_loss(lensless_rec, gt_rec).mean()
                    else:
                        e2vid_rec_loss = 0

                    #Loss
                    loss = cnn_loss_weight*cnn_rec_loss + e2vid_loss_weight*e2vid_rec_loss

                    #Accumulate frame loss to compute sequence loss
                    seq_running_loss += float(loss.item())

                #Save loss per sequence on numpy array 
                test_loss_s.append(seq_running_loss / seq_len)

                #Accumulate per sequence loss to compute epoch loss
                test_running_loss += (seq_running_loss / seq_len)

            #Save loss per epoch 
            test_loss_curr = test_running_loss / len(testloader)
            test_loss_e.append(test_loss_curr)

        #Postprocessing for display
        lensless_v, gt_v, reconstruction_v = display_postprocessing(lensless_v, gt_v, reconstruction_v)

        #Save test figure
        fig, ax = plt.subplots(2,3, figsize = (18, 10))
        ax[0][1].imshow(gt_rec[0][0].detach().cpu().numpy(), cmap = 'gray')
        ax[0][1].set_title("Groundtruth reconstruction")
        ax[0][1].axis('off')
        ax[0][2].imshow(lensless_rec[0][0].detach().cpu().numpy(), cmap = 'gray')
        ax[0][2].set_title("Deblurred reconstruction")
        ax[0][2].axis('off')
        ax[0][0].axis('off')
        ax[1][0].imshow(lensless_v, cmap = 'gray')
        ax[1][0].set_title("Blurred Voxel")
        ax[1][0].axis('off')
        ax[1][1].imshow(gt_v, cmap = 'gray')
        ax[1][1].set_title("GT Voxel")
        ax[1][1].axis('off')
        ax[1][2].imshow(reconstruction_v, cmap = 'gray')
        ax[1][2].set_title("Deblurred Voxel")
        ax[1][2].axis('off')
        fig.suptitle(f"Test Loss: {loss.item()}")
        plt.savefig(f"results/samples/{arch_name}/test_{epoch:03}.png")
        

        #Print Statistics
        print(f"[{epoch:3d} / {epochs:3d}] Train loss: {train_loss_curr:.6f} | Test loss: {test_loss_curr:.6f}")
        
        if test_loss_curr < min_test_loss:
            min_test_loss = test_loss_curr
            print("[INFO] Saving best test model...")
            torch.save(rec_cnn.state_dict(), f"results/model/cnn_{fileName}_{epoch}.pth")
            torch.save(lensless_e2vid.state_dict(), f"results/model/rnn_{fileName}_{epoch}.pth")

    #Save trained model
    torch.save(rec_cnn.state_dict(), f"results/model/cnn_{fileName}.pth")
    torch.save(lensless_e2vid.state_dict(), f"results/model/rnn_{fileName}.pth")

    #Save Losses in array
    np.save(f"results/plots/{fileName}/train_s", train_loss_s)
    np.save(f"results/plots/{fileName}/train_e", train_loss_e)
    np.save(f"results/plots/{fileName}/test_s", test_loss_s)
    np.save(f"results/plots/{fileName}/test_e", test_loss_e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",   "--dataset_dir",      help="directory",                   default="data/gaussian_sequences_small/")
    parser.add_argument("-e",   "--epochs",           help="total number of epochs",      type=int,   default=1)
    parser.add_argument("-lr",  "--learning_rate",    help="for adam optimizer",          type=float, default=1e-4)
    parser.add_argument("-b",   "--batch_size",       help="batch size for training",     type=int,   default=1)
    parser.add_argument("-c",   "--num_bins",         help="number of bins or channels",  type=int,   default=5)
    parser.add_argument("-n",   "--arch_name",        help="Identifier for model",        type=str,   default='NOID')
    parser.add_argument("-fe",  "--frozen_e2vid",     help="Default is false",            action="store_true")
    parser.add_argument("-cl",  "--cnn_loss",         help="Loss function for CNN",       type=str,   default="MSE")
    parser.add_argument("-el",  "--e2vid_loss",       help="Loss function for e2vid",     type=str,   default="LPIPS")
    parser.add_argument("-cl_w","--cnn_loss_weight",  help="Loss function for CNN",       type=float, default=1.0)
    parser.add_argument("-el_w","--e2vid_loss_weight",help="Loss function for e2vid",     type=float, default=1.0)

    #Get arguments
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_bins = args.num_bins
    arch_name = args.arch_name
    frozen_e2vid = args.frozen_e2vid
    cnn_loss = args.cnn_loss
    e2vid_loss = args.e2vid_loss
    cnn_loss_weight = args.cnn_loss_weight
    e2vid_loss_weight = args.e2vid_loss_weight

    #Print info in console
    print("-------Training Parameters--------")
    print("----------------------------------")
    print("Dataset directory:       ", dataset_dir)
    print("Epochs:                  ", epochs)
    #print("Bins:                    ", num_bins)
    #print("Batch size:              ", batch_size)
    print("Learning rate:           ", learning_rate)
    print("Frozen E2VID:            ", frozen_e2vid)
    print("CNN Loss:                ", cnn_loss)
    print("CNN Loss Weight:         ", cnn_loss_weight)
    print("E2VID Loss:              ", e2vid_loss)
    print("E2VID Loss Weight:       ", e2vid_loss_weight)
    print("----------------------------------")

    #Create dirs
    #Save plot comparing gt and lensless reconstruction
    path = f'results/samples/{arch_name}'
    isExist = os.path.exists(path)
    if not isExist:
        os.mkdir(f"results/samples/{arch_name}")

    path = f'results/plots/{arch_name}'
    isExist = os.path.exists(path)
    if not isExist:
        os.mkdir(f"results/plots/{arch_name}")

    #Train 
    train(epochs, learning_rate, dataset_dir, batch_size, num_bins, arch_name, frozen_e2vid, cnn_loss, e2vid_loss, cnn_loss_weight, e2vid_loss_weight)

    print("Done!")