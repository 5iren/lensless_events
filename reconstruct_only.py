from unittest import result
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.nn import ReflectionPad2d
from torch.utils.data import DataLoader
import cv2
from model.unet import UNet
from src.nn_utils import lenslessEventsVoxelSeq
from src.rec_utils import load_model, E2VIDRecurrent



def reconstruct(dataset_dir, trained_e2vid_model_path, save_path):

    #Set paths
    test_lensless_path = dataset_dir + '/lensless'
    test_gt_path = dataset_dir + '/gt'

    print(test_lensless_path)
    print(test_gt_path)

    #Load CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[INFO] Using device: ", device)

    #Create datasets
    print("[INFO] Loading dataset...")

    #Transforms
    transform = None

    #Create datasets
    print("[INFO] Loading dataset and dataloader...")
    test_data = lenslessEventsVoxelSeq(test_lensless_path, test_gt_path, transform = transform)

    #Create dataloaders
    testloader = DataLoader(test_data, batch_size=1, shuffle=False)
    print("       Test dataset length:  ", len(test_data))
    print("       Test  minibatches:    ", len(testloader))


    #Load frozen E2VID model for GT
    frozen_e2vid_model_path = 'model/E2VID_lightweight.pth.tar'
    frozen_e2vid = load_model(frozen_e2vid_model_path, device)
    frozen_e2vid.eval()
    frozen_e2vid.to(device)

    #Load trained E2VID model for deblurring
    config = {'num_bins': 5, 'skip_type': 'sum', 'recurrent_block_type': 'convlstm', 'num_encoders': 3, 'base_num_channels': 32, 'num_residual_blocks': 2, 'use_upsample_conv': False, 'norm': 'BN'}
    trained_e2vid = E2VIDRecurrent(config)
    trained_e2vid.load_state_dict(torch.load(trained_e2vid_model_path, map_location=device))
    #trained_e2vid = load_model(trained_e2vid_model_path, device)
    trained_e2vid.eval()
    trained_e2vid.to(device)

    #Loss
    #criterion = torch.nn.MSELoss()
    
    
    #Infer loop
    print("[INFO] Performing inference on {} examples".format(len(test_data)))
    result_num = 0
    with torch.no_grad():

        for data in testloader:
            
            #Initialize previous states
            gt_prev_state = None
            lensless_prev_state = None

            lenslessf_v, gtf_v = data[0].to(device), data[1].to(device)
            seq_len = lenslessf_v.shape[1]
            # print("------------------")
            # print("Voxel Statistics: ")
            # print(f"Lensless Min: {lensless_min}")
            # print(f"Lensless Max: {lensless_max}")
            # print(f"GT Min: {gt_min}")
            # print(f"GT Max: {gt_max}")

            for ii in tqdm(range(seq_len)):
                result_num +=1

                #Forward pass 
                lensless_v = lenslessf_v[:,ii]
                gt_v = gtf_v[:,ii]
                

                # print("------------------")
                # print("Voxel Statistics: ")
                # print(f"Lensless_v Min: {lensless_v.min()}")
                # print(f"Lensless_v Max: {lensless_v.max()}")
                # print(f"Lensless_v Mean: {lensless_v.mean()}")
                # print(f"Lensless_v Std: {lensless_v.std()}")
                # print(f"GT_v Min: {gt_v.min()}")
                # print(f"GT_v Max: {gt_v.max()}")
                # print(f"GT_v Mean: {gt_v.mean()}")
                # print(f"GT_v Std: {gt_v.std()}")
                # print(f"Output Min: {reconstruction_v.min()}")
                # print(f"Output Max: {reconstruction_v.max()}")
                # print(f"Output Mean: {reconstruction_v.mean()}")
                # print(f"Output Std: {reconstruction_v.std()}")
                #print(f"Loss: {loss}")

                
                #E2VID
                #Pad (required by e2vid)
                m = ReflectionPad2d((3,3,2,2))
                reconstruction_v = m(lensless_v)
                gt_v = m(gt_v)


                #Forward pass (if prev_states = None, no recurrency)
                lensless_rec, lensless_prev_state = trained_e2vid(reconstruction_v, prev_states = lensless_prev_state)
                gt_rec, gt_prev_state = frozen_e2vid(gt_v, prev_states = gt_prev_state)


                # Unsharp mask
                # TODO

                # Intensity rescaler
                # TODO

                # Crop (DAVIS specific numbers) and detach
                lensless_rec = lensless_rec[:, 0, 2:262,3:349]
                gt_rec = gt_rec[:, 0, 2:262,3:349]


                # Image filter
                # TODO

                #loss = criterion(lensless_rec, gt_rec)

                lensless_rec = lensless_rec[0].detach().cpu().numpy()
                gt_rec = gt_rec[0].detach().cpu().numpy()

            
        
                if result_num == 50 or result_num == 100 or result_num == 150:
                    #Detach form GPU
                    lensless_v = lensless_v.detach().cpu().numpy()
                    gt_v = gt_v.detach().cpu().numpy()
                    reconstruction_v = reconstruction_v.detach().cpu().numpy()

                    #Transpose voxel to display
                    lensless_v = np.transpose(lensless_v[0], (1,2,0))
                    gt_v = np.transpose(gt_v[0], (1,2,0))
                    reconstruction_v = np.transpose(reconstruction_v[0], (1,2,0))


                    #Limit to 3 channels to display
                    lensless_v = lensless_v[:,:,1:4]
                    gt_v = gt_v[:,:,1:4]
                    reconstruction_v = reconstruction_v[:,:,1:4]

                    #Normalize to display
                    lensless_v = ( lensless_v - lensless_v.min() ) / ( lensless_v.max() - lensless_v.min() )
                    gt_v = ( gt_v - gt_v.min() ) / ( gt_v.max() - gt_v.min() )
                    reconstruction_v = ( reconstruction_v - reconstruction_v.min() ) / ( reconstruction_v.max() - reconstruction_v.min() )

                    #Show in Plot
                    fig, ax = plt.subplots(1,5, figsize=(18,4))
                    fig.tight_layout()
                    ax[0].imshow(lensless_v)
                    ax[0].set_title("Lensless events")
                    ax[1].imshow(gt_v)
                    ax[1].set_title("Ground Truth events")
                    ax[2].imshow(reconstruction_v)
                    ax[2].set_title("CNN Output")
                    ax[3].imshow(gt_rec, cmap = 'gray')
                    ax[3].set_title("Ground Truth Reconstruction")
                    ax[4].imshow(lensless_rec, cmap = 'gray')
                    ax[4].set_title("CNN Output Reconstruction")
                    ax[0].axis('off')
                    ax[1].axis('off')
                    ax[2].axis('off')
                    ax[3].axis('off')
                    ax[4].axis('off')
                    #fig.suptitle(f"Loss: {loss}")
                    plt.savefig(f"{save_path}/plot/plt_{str(result_num).zfill(3)}.png", bbox_inches='tight')
                    #plt.show()
            
                # if result_num == 5:
                #   break

                #Save images
                cv2.imwrite(f"{save_path}/rec/img{str(result_num).zfill(3)}.png", lensless_rec*255)
                cv2.imwrite(f"{save_path}/gt/img{str(result_num).zfill(3)}.png", gt_rec*255)
        #cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",  "--dataset_dir",     help="directory",                   default="data/gaussian_sequences_single/train/")
    parser.add_argument("-em", "--e2vid_model_path",help="frozen e2vid model directory",default="model/E2VID_lightweight.pth.tar")
    parser.add_argument("-o",  "--save_path",       help="results directory",           default="results/inference/lpips/")


    #Get arguments
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    save_path = args.save_path
    e2vid_model_path = args.e2vid_model_path

    #Print info in console
    print("-------Training Parameters--------")
    print("----------------------------------")
    print("Dataset directory:       ", dataset_dir)
    print("E2VID model path:        ", e2vid_model_path)
    print("Save path:               ", save_path)
    print("----------------------------------")

    #Train 
    reconstruct(dataset_dir, e2vid_model_path, save_path)

    print("Done!")