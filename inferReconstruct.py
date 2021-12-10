import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.nn import ReflectionPad2d
from torch.utils.data import DataLoader
import cv2
from model.unet import UNet
from src.nn_utils import lenslessEventsVoxel
from src.rec_utils import load_model


#Set paths
folder = 'running1'
dataset_dir = 'data/lensless_videos_dataset/' + folder
test_lensless_path = dataset_dir + '/lensless_events'
test_gt_path = dataset_dir + '/gt_events'
#model_path = 'model/arch(Unet)-e(300)-l(L1)-o(Adam)-lr(1e-05).pth'
model_path = 'model/arch(Unet)-e(300)-l(MSE)-o(Adam)-lr(1e-05).pth'
#model_path = 'model/500_state_dict.pth'
save_path = 'results/inference/'

#Load CUDA
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("[INFO] Using device: ", device)

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
testloader = DataLoader(test_data, batch_size=1, shuffle=False)

#Load Custom Model
net = UNet(5,5)
net.load_state_dict(torch.load(model_path))
net.eval()

#Load E2VID model
model_path = 'reconstruction/E2VID_lightweight.pth.tar'
model = load_model(model_path)
#model.to(device)
model.eval()

#Initialize previous states
gt_prev_state = None
output_prev_state = None

#Infer loop
print("[INFO] Performing inference on {} examples".format(len(test_data)))
test_loss = []
test_running_loss = 0

#Test
result_num = 0
with torch.no_grad():

    for data in tqdm(testloader):
        result_num +=1
        lensless_v, gt_v = data

        #Forward pass 
        output_v = net(lensless_v)


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
    

        #Transpose voxel to display
        lensless_v = np.transpose(lensless_v[0], (1,2,0))
        gt_v = np.transpose(gt_v[0], (1,2,0))
        output_v = np.transpose(output_v[0], (1,2,0))

        #Limit to 3 channels to display
        lensless_v = lensless_v[:,:,1:4]
        gt_v = gt_v[:,:,1:4]
        output_v = output_v[:,:,1:4]

        #Normalize 
        lensless_new = ( lensless_v - lensless_v.min() ) / ( lensless_v.max() - lensless_v.min() )
        gt_new = ( gt_v - gt_v.min() ) / ( gt_v.max() - gt_v.min() )
        output_new = ( output_v - output_v.min() ) / ( output_v.max() - output_v.min() )

        #Show in Plot
        fig, ax = plt.subplots(1,5, figsize=(18,3))
        fig.tight_layout()
        ax[0].imshow(lensless_new)
        ax[0].set_title("Lensless events")
        ax[1].imshow(gt_new)
        ax[1].set_title("Ground Truth events")
        ax[2].imshow(output_new)
        ax[2].set_title("CNN Output")
        ax[3].imshow(gt_rec, cmap = 'gray')
        ax[3].set_title("Ground Truth Reconstruction")
        ax[4].imshow(output_rec, cmap = 'gray')
        ax[4].set_title("CNN Output Reconstruction")
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        ax[3].axis('off')
        ax[4].axis('off')
        plt.show()
        
        if result_num == 10:
            break

        #Show image
        #cv2.imshow('Reconstruction', output_rec)
        #cv2.waitKey(1)

        #Save images
        #cv2.imwrite('results/inference/rec/img'+str(result_num).zfill(3)+'.png', output_rec*255)
        #cv2.imwrite('results/inference/gt/img'+str(result_num).zfill(3)+'.png', gt_rec*255)
    #cv2.destroyAllWindows()


