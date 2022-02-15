import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.nn import ReflectionPad2d
from torch.utils.data import DataLoader
import cv2
from model.unet import UNet
from src.nn_utils import lenslessEventsVoxelUN, lenslessEventsVoxel, lenslessEventsVoxel1
from src.rec_utils import load_model



def reconstruct(dataset_dir, model_path, save_path, dataset_type):

    #Set paths
    test_lensless_path = dataset_dir + '/lensless'
    test_gt_path = dataset_dir + '/gt'

    #Load CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[INFO] Using device: ", device)

    #Create datasets
    print("[INFO] Loading dataset...")

    #Transforms
    transform = None

    
    #Create datasets
    print("[INFO] Loading dataset and dataloader...")
    if dataset_type == "norm":
        test_data = lenslessEventsVoxel(test_lensless_path, test_gt_path, transform = transform)
    elif dataset_type == "norm1":
        test_data = lenslessEventsVoxel1(test_lensless_path, test_gt_path, transform = transform)
    elif dataset_type == "unnorm":
        test_data = lenslessEventsVoxelUN(test_lensless_path, test_gt_path, transform = transform)
    else:
        print("Dataset type not found")

    #Create dataloaders
    testloader = DataLoader(test_data, batch_size=1, shuffle=False)

    #Load Custom Model
    print("[INFO] Loading model...")
    net = UNet(5,5)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    #Load E2VID model
    model_path = 'model/E2VID_lightweight.pth.tar'
    model = load_model(model_path, device)
    #model.to(device)
    model.eval()

    #Initialize previous states
    gt_prev_state = None
    output_prev_state = None

    #Infer loop
    print("[INFO] Performing inference on {} examples".format(len(test_data)))
    test_loss = []
    test_running_loss = 0

    #Loss
    criterion = torch.nn.MSELoss()

    #Test
    result_num = 0
    with torch.no_grad():

        for data in tqdm(testloader):
            result_num +=1
            if dataset_type == "norm" or dataset_type == "norm1":
                lensless_v, gt_v, lensless_min, lensless_max, gt_min, gt_max = data
            elif dataset_type == "unnorm":
                lensless_v, gt_v = data
            
            # print("------------------")
            # print("Voxel Statistics: ")
            # print(f"Lensless Min: {lensless_min}")
            # print(f"Lensless Max: {lensless_max}")
            # print(f"GT Min: {gt_min}")
            # print(f"GT Max: {gt_max}")

            #Forward pass 
            output_v = net(lensless_v)
            loss = criterion(output_v, gt_v)

            # print("------------------")
            # print("Voxel Statistics: ")
            # print(f"Lensless_v Min: {lensless_v.min()}")
            # print(f"Lensless_v Max: {lensless_v.max()}")
            # print(f"Lensless_v Mean: {lensless_v.mean()}")
            # print(f"GT_v Min: {gt_v.min()}")
            # print(f"GT_v Max: {gt_v.max()}")
            # print(f"GT_v Mean: {gt_v.mean()}")
            # print(f"Output Min: {output_v.min()}")
            # print(f"Output Max: {output_v.max()}")
            # print(f"Output Mean: {output_v.mean()}")
            # print(f"Loss: {loss}")

            
            #Unnormalize
            if dataset_type == "norm":
                #output_v = (output_v * (lensless_max - lensless_min) + lensless_min).float()
                output_v = (output_v * (gt_max - gt_min) + gt_min).float()
                gt_v = (gt_v * (gt_max - gt_min) + gt_min).float()
            if dataset_type == "norm1":
                #output_v = ((output_v + 1) * (lensless_max - lensless_min) / 2 + lensless_min).float()
                output_v = ((output_v + 1) * (gt_max - gt_min) / 2 + gt_min).float()
                gt_v =  ((gt_v + 1) * (gt_max - gt_min) / 2 + gt_min).float()


            #Pad (required by e2vid)
            m = ReflectionPad2d((3,3,2,2))
            output = m(output_v)
            gt = m(gt_v)

            #Forward pass (if prev_states = None, no recurrency)
            output_rec, output_states = model(output, prev_states = output_prev_state)
            gt_rec, gt_states = model(gt, prev_states = gt_prev_state)

            #Save states for next 
            output_prev_state = output_states
            gt_prev_state = gt_states

            # Unsharp mask
            # TODO

            # Intensity rescaler
            # TODO

            # Crop (DAVIS specific numbers) and detach
            output_rec = output_rec[0, 0, 2:262,3:349].numpy()
            gt_rec = gt_rec[0, 0, 2:262,3:349].numpy()

            # Image filter
            # TODO
        

            # #Transpose voxel to display
            # lensless_v = np.transpose(lensless_v[0], (1,2,0))
            # gt_v = np.transpose(gt_v[0], (1,2,0))
            # output_v = np.transpose(output_v[0], (1,2,0))

            # #Limit to 3 channels to display
            # lensless_v = lensless_v[:,:,1:4]
            # gt_v = gt_v[:,:,1:4]
            # output_v = output_v[:,:,1:4]

            # #Normalize to display
            # lensless_new = ( lensless_v - lensless_v.min() ) / ( lensless_v.max() - lensless_v.min() )
            # gt_new = ( gt_v - gt_v.min() ) / ( gt_v.max() - gt_v.min() )
            # output_new = ( output_v - output_v.min() ) / ( output_v.max() - output_v.min() )
            # # lensless_new = lensless_v
            # # gt_new = gt_v
            # # output_new = output_v

            # #Show in Plot
            # fig, ax = plt.subplots(1,5, figsize=(18,3))
            # fig.tight_layout()
            # ax[0].imshow(lensless_new)
            # ax[0].set_title("Lensless events")
            # ax[1].imshow(gt_new)
            # ax[1].set_title("Ground Truth events")
            # ax[2].imshow(output_new)
            # ax[2].set_title("CNN Output")
            # ax[3].imshow(gt_rec, cmap = 'gray')
            # ax[3].set_title("Ground Truth Reconstruction")
            # ax[4].imshow(output_rec, cmap = 'gray')
            # ax[4].set_title("CNN Output Reconstruction")
            # ax[0].axis('off')
            # ax[1].axis('off')
            # ax[2].axis('off')
            # ax[3].axis('off')
            # ax[4].axis('off')
            # plt.show()
            
            # if result_num == 10:
            #   break

            #Show image
            #cv2.imshow('Reconstruction', output_rec)
            #cv2.waitKey(1)

            #Save images
            cv2.imwrite(f"{save_path}/rec/img{str(result_num).zfill(3)}.png", output_rec*255)
            cv2.imwrite(f"{save_path}/gt/img{str(result_num).zfill(3)}.png", gt_rec*255)
        #cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",  "--dataset_dir",     help="directory",                   default="data/single_gaussian_small/")
    parser.add_argument("-m",  "--model_path",      help="trained model directory",     default="model/single_gaussian/arch(NORM)-e(200)-l(L1)-o(Adam)-lr(0.0001).pth")
    parser.add_argument("-o",  "--save_path",       help="results directory",           default="results/inference/l1/")
    parser.add_argument("-d",  "--dataset_type",    help="dataset type ",               default="norm")

    #Get arguments
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    model_path = args.model_path
    save_path = args.save_path + args.dataset_type
    dataset_type = args.dataset_type

    #Print info in console
    print("-------Training Parameters--------")
    print("----------------------------------")
    print("Dataset directory:       ", dataset_dir)
    print("Model path:              ", model_path)
    print("Save path:               ", save_path)
    print("Dataset:                 ", dataset_type)
    print("----------------------------------")

    #Train 
    reconstruct(dataset_dir, model_path, save_path, dataset_type)

    print("Done!")