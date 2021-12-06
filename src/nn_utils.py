import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from representation import event_reader, event_transforms

#Unormalize
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

#Create dataset class
class lenslessEventsVoxel(Dataset):
    def __init__(self, lensless_events_dir, gt_events_dir, num_bins = 5, transform = None):
        self.lensless_events_dir = lensless_events_dir
        self.gt_events_dir = gt_events_dir
        self.num_bins = num_bins
        self.width = 346
        self.height = 260
        self.transform = transform
        #From computing mean and std on whole dataset
        self.lensless_mean = 0.0013
        self.lensless_std = 0.1483
        self.gt_mean = 0.0011
        self.gt_std = 0.2928

        #Get list of event windows
        _, _, self.lensless_event_files = next(os.walk(self.lensless_events_dir))
        self.lensless_event_files.sort()
        
        _, _, self.gt_event_files = next(os.walk(self.gt_events_dir))
        self.gt_event_files.sort()

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

        gt_event_data = event_reader.EventData(t, x, y, p, self.width, self.height)
        gt_voxel = event_transforms.ToVoxelGrid(self.num_bins)(gt_event_data)

        #Normalize voxel 
        #gt_voxel -= self.gt_mean
        #gt_voxel /= self.gt_std

        #Convert to tensor
        gt_voxel = torch.as_tensor(gt_voxel, dtype=torch.float32)

        #Apply transforms
        if self.transform:
            lensless_voxel = self.transform(lensless_voxel)
            gt_voxel = self.transform(gt_voxel)

        return lensless_voxel, gt_voxel

#Create dataset class
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