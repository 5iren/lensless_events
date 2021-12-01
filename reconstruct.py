from src.rec_utils import load_model
import torch 
from torch.nn import ReflectionPad2d
import matplotlib.pyplot as plt
import os
import cv2

#Load voxels
gt_path = 'reconstruction/voxels/airplane2/gt'
rec_path = 'reconstruction/voxels/airplane2/rec'
_, _, gt_files = next(os.walk(gt_path))
_, _, rec_files = next(os.walk(rec_path))
gt_files.sort()
rec_files.sort()

#Load CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("[INFO] Using device: ", device)

#Load model
model_path = 'reconstruction/E2VID_lightweight.pth.tar'
model = load_model(model_path)
model.to(device)
model.eval()

#Initialize previous states
gt_prev_state = None
output_prev_state = None
with torch.no_grad():
    for i in range(len(gt_files)):
        gt1_path = os.path.join(gt_path, gt_files[i])
        rec1_path = os.path.join(rec_path, rec_files[i])

        #Load voxel grid
        output = torch.load(rec1_path).to(device)
        gt = torch.load(gt1_path).to(device)

        #Pad (required by e2vid)
        m = ReflectionPad2d((3,3,2,2))
        output = m(output)
        gt = m(gt)
        # print("Pad shapes: ")
        # print(output.shape)
        # print(gt.shape)

        #Forward pass (if prev_states = None, no recurrency)
        output_rec, output_states = model(output, prev_states = output_prev_state)
        gt_rec, gt_states = model(gt, prev_states = gt_prev_state)

        #Save states for next 
        output_prev_state = output_states
        gt_prev_state = gt_states
        #output_prev_state = None
        #gt_prev_state = None


        # Unsharp mask
        # TODO

        # Intensity rescaler
        # TODO

        # Crop (DAVIS specific numbers) and detach
        output_rec = output_rec[0, 0, 2:262,3:349].cpu().detach().numpy()
        gt_rec = gt_rec[0, 0, 2:262,3:349].cpu().detach().numpy()
        # print(output_rec.shape)
        # print(gt_rec.shape)

        # Image filter
        # TODO

        # # Show image
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(gt_rec, cmap='gray')
        # ax[0].set_title('GT reconstruction')
        # ax[1].imshow(output_rec, cmap='gray')
        # ax[1].set_title('Lensless reconstruction')
        # ax[0].set_xticks([])
        # ax[0].set_yticks([])
        # ax[1].set_xticks([])
        # ax[1].set_yticks([])
        # plt.show()

        #Save image
        plt.imsave("reconstruction/results/airplane2/rec/"+str(i).zfill(3)+".png", output_rec, cmap='gray')
        plt.imsave("reconstruction/results/airplane2/gt/"+str(i).zfill(3)+".png", gt_rec, cmap='gray')

