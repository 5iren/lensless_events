import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from representation import event_reader, event_transforms

#Create dataset class: creates event volume with voxel grid method (B, H, W)
class lenslessEventsVoxel(Dataset):
    def __init__(self, lensless_events_dir, gt_events_dir, num_bins = 5, transform = None):
        self.lensless_events_dir = lensless_events_dir
        self.gt_events_dir = gt_events_dir
        self.num_bins = num_bins
        self.width = 346
        self.height = 260
        self.transform = transform
        #From computing mean and std on whole dataset
        #self.lensless_mean = 0.0013
        #self.lensless_std = 0.1483
        #self.gt_mean = 0.0011
        #self.gt_std = 0.2928

        #Get list of event windows
        _, _, self.lensless_event_files = next(os.walk(self.lensless_events_dir))
        self.lensless_event_files.sort()
        
        _, _, self.gt_event_files = next(os.walk(self.gt_events_dir))
        self.gt_event_files.sort()

        print("\tDataset: Normalized")

    def __len__(self):
        return len(self.lensless_event_files)

    def __getitem__(self, idx):
        ##### Get Lensless voxel grid #####
        #Create voxel grid from events
        lensless_data = np.load(os.path.join(self.lensless_events_dir, self.lensless_event_files[idx]))
        t = lensless_data['t']
        x = lensless_data['x']
        y = lensless_data['y']
        p = lensless_data['p']

        lensless_event_data = event_reader.EventData(t, x, y, p, self.width, self.height)
        lensless_voxel = event_transforms.ToVoxelGrid(self.num_bins)(lensless_event_data)

        #Normalize voxel
        lensless_voxel_min = lensless_voxel.min()
        lensless_voxel_max = lensless_voxel.max() 
        lensless_voxel -= lensless_voxel_min
        lensless_voxel /= (lensless_voxel_max - lensless_voxel_min)

        #Convert to tensor
        lensless_voxel = torch.as_tensor(lensless_voxel, dtype=torch.float32)
      
        ##### Get GT voxel grid #####
        #Create voxel grid from events
        gt_data = np.load(os.path.join(self.gt_events_dir, self.gt_event_files[idx]))
        t = gt_data['t']
        x = gt_data['x']
        y = gt_data['y']
        p = gt_data['p']

        gt_event_data = event_reader.EventData(t, x, y, p, self.width, self.height)
        gt_voxel = event_transforms.ToVoxelGrid(self.num_bins)(gt_event_data)

        #Normalize voxel 
        gt_voxel_min = gt_voxel.min()
        gt_voxel_max = gt_voxel.max()
        gt_voxel -= gt_voxel_min 
        gt_voxel /= (gt_voxel_max - gt_voxel_min )

        #Convert to tensor
        gt_voxel = torch.as_tensor(gt_voxel, dtype=torch.float32)

        #Apply transforms
        if self.transform:
            lensless_voxel = self.transform(lensless_voxel)
            gt_voxel = self.transform(gt_voxel)

        return lensless_voxel, gt_voxel, lensless_voxel_min, lensless_voxel_max, gt_voxel_min, gt_voxel_max

#Unnormalized version
class lenslessEventsVoxelUN(Dataset):
    def __init__(self, lensless_events_dir, gt_events_dir, num_bins = 5, transform = None):
        self.lensless_events_dir = lensless_events_dir
        self.gt_events_dir = gt_events_dir
        self.num_bins = num_bins
        self.width = 346
        self.height = 260
        self.transform = transform

        #Get list of event windows
        _, _, self.lensless_event_files = next(os.walk(self.lensless_events_dir))
        self.lensless_event_files.sort()
        
        _, _, self.gt_event_files = next(os.walk(self.gt_events_dir))
        self.gt_event_files.sort()

        print("\tDataset: Unnormalized")

    def __len__(self):
        return len(self.lensless_event_files)

    def __getitem__(self, idx):
        ##### Get Lensless voxel grid #####
        #Create voxel grid from events
        lensless_data = np.load(os.path.join(self.lensless_events_dir, self.lensless_event_files[idx]))
        t = lensless_data['t']
        x = lensless_data['x']
        y = lensless_data['y']
        p = lensless_data['p']

        lensless_event_data = event_reader.EventData(t, x, y, p, self.width, self.height)
        lensless_voxel = event_transforms.ToVoxelGrid(self.num_bins)(lensless_event_data)

        #Normalize voxel 
        #lensless_voxel -= lensless_voxel.min()
        #lensless_voxel /= (lensless_voxel.max() - lensless_voxel.min())

        #Convert to tensor
        lensless_voxel = torch.as_tensor(lensless_voxel, dtype=torch.float32)
      
        ##### Get GT voxel grid #####
        #Create voxel grid from events
        gt_data = np.load(os.path.join(self.gt_events_dir, self.gt_event_files[idx]))
        t = gt_data['t']
        x = gt_data['x']
        y = gt_data['y']
        p = gt_data['p']

        gt_event_data = event_reader.EventData(t, x, y, p, self.width, self.height)
        gt_voxel = event_transforms.ToVoxelGrid(self.num_bins)(gt_event_data)

        #Normalize voxel 
        #gt_voxel -= gt_voxel.min()
        #gt_voxel /= (gt_voxel.max() - gt_voxel.min())

        #Convert to tensor
        gt_voxel = torch.as_tensor(gt_voxel, dtype=torch.float32)

        #Apply transforms
        if self.transform:
            lensless_voxel = self.transform(lensless_voxel)
            gt_voxel = self.transform(gt_voxel)

        return lensless_voxel, gt_voxel

#Create dataset class: creates one event volume with voxel grid method for each polarity and concatenates along time dimension (Bx2, H, W)
class lenslessEventsVoxel2(Dataset):
    def __init__(self, lensless_events_dir, gt_events_dir, num_bins = 5, transform = None):
        self.lensless_events_dir = lensless_events_dir
        self.gt_events_dir = gt_events_dir
        self.num_bins = num_bins
        self.width = 346
        self.height = 260
        self.transform = transform
        #From computing mean and std on whole dataset
        # self.lensless_mean = 0.0013
        # self.lensless_std = 0.1483
        # self.gt_mean = 0.0011
        # self.gt_std = 0.2928

        #Get list of event windows
        _, _, self.lensless_event_files = next(os.walk(self.lensless_events_dir))
        self.lensless_event_files.sort()
        
        _, _, self.gt_event_files = next(os.walk(self.gt_events_dir))
        self.gt_event_files.sort()

        print("\tDataset: Double voxel grid")

    def __len__(self):
        return len(self.lensless_event_files)

    def __getitem__(self, idx):
        ##### Get Lensless voxel grid #####
        #Create voxel grid from events
        lensless_data = np.load(os.path.join(self.lensless_events_dir, self.lensless_event_files[idx]))
        t = lensless_data['t']
        x = lensless_data['x']
        y = lensless_data['y']
        p = lensless_data['p']

        #Partition between positive and negative
        positive_events = p.nonzero()
        negative_events = np.where(p == 0)
        t_pos = t[positive_events]
        t_neg = t[negative_events]
        x_pos = x[positive_events]
        x_neg = x[negative_events]
        y_pos = y[positive_events]
        y_neg = y[negative_events]
        p_pos = p[positive_events]
        p_neg = p[negative_events]

        #Transform into voxel grids
        pos_events = event_reader.EventData(t_pos, x_pos, y_pos, p_pos, self.width, self.height)
        neg_events = event_reader.EventData(t_neg, x_neg, y_neg, p_neg, self.width, self.height)
        
        #Check length is not 0
        if len(pos_events) > 0:
            lensless_pos_voxel = event_transforms.ToVoxelGrid2(self.num_bins)(pos_events)
        else:
            lensless_pos_voxel = np.zeros((self.num_bins, self.height, self.width))

        if len(neg_events) > 0:
            lensless_neg_voxel = event_transforms.ToVoxelGrid2(self.num_bins)(neg_events)
        else:
            lensless_neg_voxel = np.zeros((self.num_bins, self.height, self.width))

        #Concatenate along time dimension
        lensless_voxel = np.concatenate((lensless_neg_voxel, lensless_pos_voxel))

        #Normalize voxel 
        #lensless_voxel -= self.lensless_mean
        #lensless_voxel /= self.lensless_std

        #Convert to tensor
        lensless_voxel = torch.as_tensor(lensless_voxel, dtype=torch.float32)
      
      
        ##### Get GT voxel grid #####
        #Create voxel grid from events
        gt_data = np.load(os.path.join(self.gt_events_dir, self.gt_event_files[idx]))
        t = gt_data['t']
        x = gt_data['x']
        y = gt_data['y']
        p = gt_data['p']

        #Partition between positive and negative
        positive_events = p.nonzero()
        negative_events = np.where(p == 0)
        t_pos = t[positive_events]
        t_neg = t[negative_events]
        x_pos = x[positive_events]
        x_neg = x[negative_events]
        y_pos = y[positive_events]
        y_neg = y[negative_events]
        p_pos = p[positive_events]
        p_neg = p[negative_events]

        #Transform into voxel grids
        pos_events = event_reader.EventData(t_pos, x_pos, y_pos, p_pos, self.width, self.height)
        neg_events = event_reader.EventData(t_neg, x_neg, y_neg, p_neg, self.width, self.height)
        
        #Check length is not 0
        if len(pos_events) > 0:
            gt_pos_voxel = event_transforms.ToVoxelGrid2(self.num_bins)(pos_events)
        else:
            gt_pos_voxel = np.zeros((self.num_bins, self.height, self.width))

        if len(neg_events) > 0:
            gt_neg_voxel = event_transforms.ToVoxelGrid2(self.num_bins)(neg_events)
        else:
            gt_neg_voxel = np.zeros((self.num_bins, self.height, self.width))

        #Concatenate along time dimension
        gt_voxel = np.concatenate((gt_neg_voxel, gt_pos_voxel))

        #Normalize voxel 
        #gt_voxel -= self.gt_mean
        #gt_voxel /= self.gt_std

        #Convert to tensor
        gt_voxel = torch.as_tensor(gt_voxel, dtype=torch.float32)

        #Apply transforms
        # if self.transform:
        #     lensless_voxel = self.transform(lensless_voxel)
        #     gt_voxel = self.transform(gt_voxel)

        #Statistics
        # print("Lensless statistics: ")
        # print(f"Lensless mean: {lensless_voxel.mean()}")
        # print(f"Lensless max: {lensless_voxel.max()}")
        # print(f"Lensless min: {lensless_voxel.min()}")
        # print(f"Lensless std: {lensless_voxel.std()}")
        # print(f"Non zero values: {len(lensless_voxel.nonzero())}")

        # print("GT statistics: ")
        # print(f"GT mean: {gt_voxel.mean()}")
        # print(f"GT max: {gt_voxel.max()}")
        # print(f"GT min: {gt_voxel.min()}")
        # print(f"GT std: {gt_voxel.std()}")
        # print(f"Non zero values: {len(gt_voxel.nonzero())}")

        return lensless_voxel, gt_voxel

#Create dataset class: creates event volume with accumulation method (B, H, W)
class lenslessEvents(Dataset):
    def __init__(self, lensless_events_dir, gt_events_dir, num_bins = 3, transform = None):
        self.lensless_events_dir = lensless_events_dir
        self.gt_events_dir = gt_events_dir
        self.num_bins = num_bins
        self.width = 346
        self.height = 260
        self.transform = transform

        #Get list of event windows
        _, _, self.lensless_event_files = next(os.walk(self.lensless_events_dir))
        self.lensless_event_files.sort()
        
        _, _, self.gt_event_files = next(os.walk(self.gt_events_dir))
        self.gt_event_files.sort()

        print("\tDataset: Event Accumulation")


    def __len__(self):
        return len(self.lensless_event_files)

    def __getitem__(self, idx):
        ##### Create Lensless #####
        #Load window
        lensless_data = np.load(os.path.join(self.lensless_events_dir, self.lensless_event_files[idx]))
        t = lensless_data['t']
        x = lensless_data['x']
        y = lensless_data['y']
        p = lensless_data['p']

        lensless_events = np.zeros((1, self.height, self.width))
        for i in range(t.shape[0]):
            lensless_events[0][y[i]][x[i]] = p[i]

        #Convert to tensor
        lensless_events = torch.as_tensor(lensless_events, dtype=torch.float32)


        ##### Get GT voxel grid #####
        #Create voxel grid from events
        gt_data = np.load(os.path.join(self.gt_events_dir, self.gt_event_files[idx]))
        t = gt_data['t']
        x = gt_data['x']
        y = gt_data['y']
        p = gt_data['p']


        gt_events = np.zeros((1, self.height, self.width))
        for i in range(t.shape[0]):
            gt_events[0][y[i]][x[i]] = p[i] 

        

        #Convert to tensor
        gt_events = torch.as_tensor(gt_events, dtype=torch.float32)

        

        #Apply transforms
        if self.transform:
            lensless_events = self.transform(lensless_events)
            gt_events = self.transform(gt_events)

        return lensless_events, gt_events

#Add clipping normalization -1 to 1
class lenslessEventsVoxelClip(Dataset):
    def __init__(self, lensless_events_dir, gt_events_dir, num_bins = 5, transform = None):
        self.lensless_events_dir = lensless_events_dir
        self.gt_events_dir = gt_events_dir
        self.num_bins = num_bins
        self.width = 346
        self.height = 260
        self.transform = transform
        #From computing mean and std on whole dataset
        #self.lensless_mean = 0.0013
        #self.lensless_std = 0.1483
        #self.gt_mean = 0.0011
        #self.gt_std = 0.2928

        #Get list of event windows
        _, _, self.lensless_event_files = next(os.walk(self.lensless_events_dir))
        self.lensless_event_files.sort()
        
        _, _, self.gt_event_files = next(os.walk(self.gt_events_dir))
        self.gt_event_files.sort()

        print("\tDataset: [-1, 1] Clipped & Normalized")


    def __len__(self):
        return len(self.lensless_event_files)

    def __getitem__(self, idx):
        ##### Get Lensless voxel grid #####
        #Create voxel grid from events
        lensless_data = np.load(os.path.join(self.lensless_events_dir, self.lensless_event_files[idx]))
        t = lensless_data['t']
        x = lensless_data['x']
        y = lensless_data['y']
        p = lensless_data['p']

        lensless_event_data = event_reader.EventData(t, x, y, p, self.width, self.height)
        lensless_voxel = event_transforms.ToVoxelGrid(self.num_bins)(lensless_event_data)
        lensless_voxel = torch.as_tensor(lensless_voxel, dtype=torch.float32)

        #Perform clipping operation
        event_volume_flat = lensless_voxel.view(-1)
        nonzero = torch.nonzero(event_volume_flat)
        nonzero_values = event_volume_flat[nonzero]
        if nonzero_values.shape[0]:
            lower = torch.kthvalue(nonzero_values,
                                   max(int(0.02 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            upper = torch.kthvalue(nonzero_values,
                                   max(int(0.98 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            max_val = max(abs(lower), upper)
            lensless_voxel = torch.clamp(lensless_voxel, -max_val, max_val)
            
            #Normalize voxel
            lensless_voxel /= max_val
            # lensless_voxel_min = lensless_voxel.min()
            # lensless_voxel_max = lensless_voxel.max() 
            # lensless_voxel -= lensless_voxel_min
            # lensless_voxel /= (lensless_voxel_max - lensless_voxel_min)


      
        ##### Get GT voxel grid #####
        #Create voxel grid from events
        gt_data = np.load(os.path.join(self.gt_events_dir, self.gt_event_files[idx]))
        t = gt_data['t']
        x = gt_data['x']
        y = gt_data['y']
        p = gt_data['p']

        gt_event_data = event_reader.EventData(t, x, y, p, self.width, self.height)
        gt_voxel = event_transforms.ToVoxelGrid(self.num_bins)(gt_event_data)
        gt_voxel = torch.as_tensor(gt_voxel, dtype=torch.float32)


        #Perform clipping operation
        event_volume_flat = gt_voxel.view(-1)
        nonzero = torch.nonzero(event_volume_flat)
        nonzero_values = event_volume_flat[nonzero]
        if nonzero_values.shape[0]:
            lower = torch.kthvalue(nonzero_values,
                                   max(int(0.02 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            upper = torch.kthvalue(nonzero_values,
                                   max(int(0.98 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            max_val = max(abs(lower), upper)
            gt_voxel = torch.clamp(gt_voxel, -max_val, max_val)

            #Normalize voxel
            gt_voxel /= max_val 
            # gt_voxel_min = gt_voxel.min()
            # gt_voxel_max = gt_voxel.max()
            # gt_voxel -= gt_voxel_min 
            # gt_voxel /= (gt_voxel_max - gt_voxel_min )


        #Apply transforms
        if self.transform:
            lensless_voxel = self.transform(lensless_voxel)
            gt_voxel = self.transform(gt_voxel)

        return lensless_voxel, gt_voxel, max_val