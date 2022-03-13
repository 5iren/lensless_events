import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from representation import event_reader, event_transforms

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

        #Convert to tensor
        gt_voxel = torch.as_tensor(gt_voxel, dtype=torch.float32)

        #Apply transforms
        if self.transform:
            lensless_voxel = self.transform(lensless_voxel)
            gt_voxel = self.transform(gt_voxel)

        return lensless_voxel, gt_voxel

#Unnormalized sequence version
class lenslessEventsVoxelSeq(Dataset):
    def __init__(self, lensless_events_dir, gt_events_dir, num_bins = 5, transform = None):
        self.lensless_events_dir = lensless_events_dir
        self.gt_events_dir = gt_events_dir
        self.num_bins = num_bins
        self.width = 346
        self.height = 260
        self.transform = transform

        #Get list of sequences
        _, self.lensless_sequences, _ = next(os.walk(self.lensless_events_dir))
        self.lensless_sequences.sort()
        
        _, self.gt_sequences, _ = next(os.walk(self.gt_events_dir))
        self.gt_sequences.sort()

        print("\tDataset: Sequence")

    def __len__(self):
        return len(self.lensless_sequences)

    def __getitem__(self, idx):
        ##### Get Lensless voxel grid #####
        #Create voxel grids from events
        sequence_dir = os.path.join(self.lensless_events_dir, self.lensless_sequences[idx])
        _, _, event_windows = next(os.walk(sequence_dir))
        event_windows.sort()
        lensless_voxels = []
        for i, event_window_name in enumerate(event_windows):
            event_window = np.load(os.path.join(sequence_dir, event_window_name))
            t = event_window['t']
            x = event_window['x']
            y = event_window['y']
            p = event_window['p']

            lensless_event_data = event_reader.EventData(t, x, y, p, self.width, self.height)
            lensless_voxels.append(event_transforms.ToVoxelGrid(self.num_bins)(lensless_event_data))
        #Convert to tensor
        lensless_voxels = torch.as_tensor(np.array(lensless_voxels), dtype=torch.float32)
      
        ##### Get GT voxel grid #####
        #Create voxel grids from events
        sequence_dir = os.path.join(self.gt_events_dir, self.gt_sequences[idx])
        _, _, event_windows = next(os.walk(sequence_dir))
        event_windows.sort()
        gt_voxels = []
        for i, event_window_name in enumerate(event_windows):
            event_window = np.load(os.path.join(sequence_dir, event_window_name))
            t = event_window['t']
            x = event_window['x']
            y = event_window['y']
            p = event_window['p']

            gt_event_data = event_reader.EventData(t, x, y, p, self.width, self.height)
            gt_voxels.append(event_transforms.ToVoxelGrid(self.num_bins)(gt_event_data))
        #Convert to tensor
        gt_voxels = torch.as_tensor(np.array(gt_voxels), dtype=torch.float32)

        return lensless_voxels, gt_voxels