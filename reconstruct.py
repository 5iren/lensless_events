from src.rec_utils import load_model
import torch 
from torch.nn import ReflectionPad2d
import matplotlib.pyplot as plt


#Load model
model_path = 'reconstruction/E2VID_lightweight.pth.tar'
model = load_model(model_path)
model.eval()

#Load voxel grid
output = torch.load('reconstruction/output1.pt')
gt = torch.load('reconstruction/gt1.pt')
# print("Load shapes: ")
# print(output.shape)
# print(gt.shape)

#Pad (required by e2vid)
m = ReflectionPad2d((3,3,2,2))
output = m(output)
gt = m(gt)
# print("Pad shapes: ")
# print(output.shape)
# print(gt.shape)

#Forward pass (if prev_states = None, no recurrency)
output_rec, output_states = model(output, prev_states = None)
gt_rec, gt_states = model(gt, prev_states = None)
# print(output_rec.shape)
# print(gt_rec.shape)

# Unsharp mask
# TODO

# Intensity rescaler
# TODO

# Crop (DAVIS specific numbers) and detach
output_rec = output_rec[0, 0, 2:262,3:349].detach().numpy()
gt_rec = gt_rec[0, 0, 2:262,3:349].detach().numpy()
# print(output_rec.shape)
# print(gt_rec.shape)

# Image filter
# TODO


fig, ax = plt.subplots(1,2)
ax[0].imshow(output_rec, cmap='gray')
ax[1].imshow(gt_rec, cmap='gray')
plt.show()
