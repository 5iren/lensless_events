import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from representation import event_reader, event_transforms

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    full_gaussian = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    gaussian = full_gaussian / full_gaussian.sum()
    return  gaussian

def framePreprocess(frame, davis_height, davis_width, davis_ratio):
    #Convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Get shape
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    frame_ratio = frame_height / frame_width

    #If frame_ratio > davis_ratio: resize width
    if frame_ratio > davis_ratio:
        #Get new frame and width from ratio
        new_height = int(davis_width * frame_ratio)
        new_width = davis_width

        #Resize
        frame = cv2.resize(frame, (new_width, new_height))

        #Center crop
        height_difference = new_height - davis_height
        cropped_frame = frame[int(height_difference/2):int(height_difference/2+davis_height),:]

    #If frame_ratio < davis_ratio: resize height
    elif frame_ratio < davis_ratio:
        #Get new frame and width from ratio
        new_height = davis_height
        new_width = int(davis_height / frame_ratio)
        
        #Resize
        frame = cv2.resize(frame, (new_width, new_height))

        #Center crop
        width_difference = new_width - davis_width
        cropped_frame = frame[:, int(width_difference/2):int(width_difference/2+davis_width)]

    assert cropped_frame.shape[0] == davis_height, 'Height is not '+str(davis_height)+". Frame height: " +str(frame.shape[0])
    assert cropped_frame.shape[1] == davis_width, 'Width is not '+str(davis_width)+". Frame width: " +str(frame.shape[1])

    return cropped_frame

def fftConvolve(frame, psf):
    """ Convolve frame and PSF to simulate lensless image

    frame and psf must be of the same square shape
    """
    #Get fourier transform of PSF
    psf_fft = np.fft.fft2(psf)

    #Get fourier trasnform of frame
    frame_fft = np.fft.fft2(frame)

    #Product
    frame_fft = frame_fft * psf_fft

    #Inverse and shift
    frame_fft = np.fft.ifft2(frame_fft)
    frame_fft = np.fft.ifftshift(frame_fft)

    return frame_fft

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
        # lensless_voxel -= lensless_voxel.mean()
        # lensless_voxel /= lensless_voxel.std()

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
        # gt_voxel -= gt_voxel.mean()
        # gt_voxel /= gt_voxel.std()

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