import math
import numpy as np
import torch
import os
import sys

sys.path.append('/home/colmanglagovich/dev/AstroboticEventCameras/EventPipeline/representation')
# import voxel_grid as vg

# import cppimport
# vox_cpp = cppimport.imp_from_filepath('/home/colmanglagovich/dev/AstroboticEventCameras/EventPipeline/representation/vox_code.cpp')


def voxel_histogram(t, x, y, p, width, height, num_bins):
    '''
    This is a simple voxel transformation that does not do interpolation between time voxels
    and instead just counts events that fall within a voxel.
    '''

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = t[-1]
    first_stamp = t[0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    # ts = (num_bins - 1) * (t - first_stamp) / deltaT
    ts = (num_bins) * (t - first_stamp) / deltaT
    xs = x.astype(np.int32)
    ys = y.astype(np.int32)
    pols = p.astype(np.int32)
    pols[pols == 0] = -1  # polarity should be +1 / -1
    tis = ts.astype(np.int32)
    valid_idxs = tis < num_bins

    # Cache some important arrays
    ys_w = ys * width
    tis_wh = tis * width * height

    idxs = xs + ys_w + tis_wh

    voxel_grid = np.bincount(idxs[valid_idxs], weights=pols[valid_idxs], minlength=(num_bins * height * width))

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


class ToVoxelGridPytorch:
    '''
    Transforms EventData into Voxel Grid.
    Input: AER in form of EventData object
    '''
    def __init__(self, num_bins, device):
        assert(num_bins > 0)
        self.num_bins = num_bins
        self.device = device

    def __call__(self, events):
        height = events.height
        width = events.width

        '''
        Takes in EventData object, returns 3-dim array
        '''
        assert(events.total_events() > 0)
        assert(width > 0)
        assert(height > 0)

        # voxel_grid = np.zeros((self.num_bins * height * width,), np.float32)

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events.t[-1]
        first_stamp = events.t[0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        ts = (self.num_bins - 1) * (events.t - first_stamp) / deltaT

        device = self.device
        ts = torch.as_tensor(ts, dtype=torch.float32, device=device)
        xs = torch.as_tensor(events.x.astype(np.int64), dtype=torch.int64, device=device)
        ys = torch.as_tensor(events.y.astype(np.int64), dtype=torch.int64, device=device)
        pols = torch.as_tensor(events.p.astype(np.int64), dtype=torch.int64, device=device)
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = ts.int()
        dts = (ts - tis).float()
        vals_left = pols * (1.0 - dts)
        vals_right = pols * dts

        # Cache some important arrays
        ys_w = ys * width
        tis_wh = tis * width * height
        wh = width * height

        idxs = xs + ys_w + tis_wh

        valid_indices = tis < self.num_bins
        left_indices = idxs[valid_indices]
        left_weights = vals_left[valid_indices]

        valid_indices = (tis + 1) < self.num_bins
        right_indices = (idxs + wh)[valid_indices]
        right_weights = vals_right[valid_indices]

        indices = torch.cat((left_indices, right_indices))
        weights = torch.cat((left_weights, right_weights))

        voxel_grid = torch.zeros(self.num_bins*width*height, dtype=weights.dtype, device=device).scatter_add_(0, indices, weights)



        voxel_grid = voxel_grid.reshape((self.num_bins, height, width))

        return voxel_grid


class ToVoxelGridCPP:
    '''
    Transforms EventData into Voxel Grid.
    Input: AER in form of EventData object
    '''
    def __init__(self, num_bins):
        assert(num_bins > 0)
        self.num_bins = num_bins

    def __call__(self, events):
        height = events.height
        width = events.width

        '''
        Takes in EventData object, returns 3-dim array
        '''
        assert(events.total_events() > 0)
        assert(width > 0)
        assert(height > 0)

        # voxel_grid = np.zeros((self.num_bins * height * width,), np.float32)

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events.t[-1]
        first_stamp = events.t[0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        ts = (self.num_bins - 1) * (events.t - first_stamp) / deltaT
        xs = events.x.astype(np.int32)
        ys = events.y.astype(np.int32)
        pols = events.p
        num_elements = self.num_bins*height*width

        voxel_grid = vox_cpp.voxel_transform(ts, xs, ys, pols, width, height, self.num_bins)

        voxel_grid = np.reshape(voxel_grid, (self.num_bins, height, width))

        return voxel_grid

class ToVoxelGridCython:
    '''
    Transforms EventData into Voxel Grid.
    Input: AER in form of EventData object
    '''
    def __init__(self, num_bins):
        assert(num_bins > 0)
        self.num_bins = num_bins

    def __call__(self, events):
        height = events.height
        width = events.width

        '''
        Takes in EventData object, returns 3-dim array
        '''
        assert(events.total_events() > 0)
        assert(width > 0)
        assert(height > 0)

        # voxel_grid = np.zeros((self.num_bins * height * width,), np.float32)

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events.t[-1]
        first_stamp = events.t[0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        ts = (self.num_bins - 1) * (events.t - first_stamp) / deltaT
        xs = events.x.astype(np.int32)
        ys = events.y.astype(np.int32)
        pols = events.p.astype(np.int32)
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = ts.astype(np.int32)
        dts = (ts - tis).astype(np.float32)
        vals_left = pols * (1.0 - dts)
        vals_right = pols * dts

        # Cache some important arrays
        ys_w = ys * width
        tis_wh = tis * width * height
        wh = width * height

        idxs = xs + ys_w + tis_wh

        valid_indices = tis < self.num_bins
        left_indices = idxs[valid_indices]
        left_weights = vals_left[valid_indices]

        valid_indices = (tis + 1) < self.num_bins
        right_indices = (idxs + wh)[valid_indices]
        right_weights = vals_right[valid_indices]

        indices = np.concatenate((left_indices, right_indices))
        weights = np.concatenate((left_weights, right_weights)).astype(np.float32)

        # print(indices.dtype)
        # print(weights.dtype)
        
        # print(np.max(indices))

        # voxel_grid = np.bincount(indices, weights=weights, minlength=(self.num_bins * height * width))
        voxel_grid = vg.cython_bincount(indices, weights=weights, minlength=(self.num_bins*height*width))
        # print(voxel_grid.dtype)
        # exit()
        # print(vg)

        voxel_grid = np.reshape(voxel_grid, (self.num_bins, height, width))

        return voxel_grid

'''
Transforms on windows of the AER stream.
 - Event image (histogram)
 - Voxel grid

Voxel grid code adapted from https://github.com/uzh-rpg/rpg_e2vid/blob/master/utils/inference_utils.py
'''

class ToVoxelGrid:
    '''
    Transforms EventData into Voxel Grid.
    Input: AER in form of EventData object
    '''
    def __init__(self, num_bins):
        assert(num_bins > 0)
        self.num_bins = num_bins

    def __call__(self, events):
        height = events.height
        width = events.width

        '''
        Takes in EventData object, returns 3-dim array
        '''
        assert(events.total_events() > 0)
        assert(width > 0)
        assert(height > 0)

        # voxel_grid = np.zeros((self.num_bins * height * width,), np.float32)

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events.t[-1]
        first_stamp = events.t[0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        ts = (self.num_bins - 1) * (events.t - first_stamp) / deltaT
        xs = events.x.astype(np.int32)
        ys = events.y.astype(np.int32)
        pols = events.p.astype(np.int32)
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = ts.astype(np.int32)
        dts = (ts - tis).astype(np.float32)
        vals_left = pols * (1.0 - dts)
        vals_right = pols * dts

        # Cache some important arrays
        ys_w = ys * width
        tis_wh = tis * width * height
        wh = width * height

        idxs = xs + ys_w + tis_wh

        valid_indices = tis < self.num_bins
        left_indices = idxs[valid_indices]
        left_weights = vals_left[valid_indices]

        valid_indices = (tis + 1) < self.num_bins
        right_indices = (idxs + wh)[valid_indices]
        right_weights = vals_right[valid_indices]

        indices = np.concatenate((left_indices, right_indices))
        weights = np.concatenate((left_weights, right_weights))

        voxel_grid = np.bincount(indices, weights=weights, minlength=(self.num_bins * height * width))
        # voxel_grid = np.zeros(self.num_bins * height * width, dtype=np.float32)
        # np.add.at(voxel_grid, indices, weights)
        
        voxel_grid = np.reshape(voxel_grid, (self.num_bins, height, width))

        return voxel_grid


class Normalize:
    '''
    Normalize Whatever input comes in to between 0-1
    Input: AER in form of EventData object
    # TODO: rename to Intensity Rescaler
    '''
    def __init__(self, pmax=98, pmin=2):
        self.pmax = pmax
        self.pmin = pmin

    def __call__(self, sample):
        s_max, s_min = np.percentile(sample.ravel(), (self.pmax, self.pmin))

        diff = s_max - s_min
        if diff == 0:
            diff = 1

        s = (sample - s_min)/(diff)

        s = np.clip(s, 0.0, 1.0)

        return s


class VoxelNormalize:
    '''
    Normalize voxel grid to have unit mean and unit std.
    TODO: What should input type be? numpy or torch.Tensor?
    '''
    def __init__(self):
        pass

    def __call__(self, sample):
        # sample = torch.as_tensor(sample, dtype=torch.float32)
        nonzero = (sample != 0)
        num_nonzero = nonzero.sum()

        if num_nonzero > 0:
            nonzero_points = sample[nonzero]
            sample[nonzero] = (nonzero_points - nonzero_points.mean()) / nonzero_points.std()
        
        sample = torch.as_tensor(sample, dtype=torch.float32)
        return sample


class ToEventImage:
    '''
    Transforms into 2-channel event image
    Input: AER in form of EventData object
    '''
    def __call__(self, sample):
        '''
        sample: EventData input
        '''
        image = np.zeros((2, sample.height, sample.width), np.float32)
        x = sample.x.astype(int)
        y = sample.y.astype(int)
        p = sample.p.astype(int)
        pos = p == 1
        posimg = np.zeros((sample.height, sample.width),np.float32).ravel()
        np.add.at(posimg, x[pos] + y[pos]*sample.width, 1)

        posimg = posimg.reshape((sample.height, sample.width))

        negimg = np.zeros((sample.height, sample.width),np.float32).ravel()
        np.add.at(negimg, x[~pos] + y[~pos]*sample.width, 1)

        negimg = negimg.reshape((sample.height, sample.width))

        image[0,...] = posimg
        image[1,...] = negimg
        return image


class HighPassFilter:
    '''
    Stateful transform to generate HPF image from and event dataset iterator.
    Note: Takes in events with polarity: {0,1}. It handles turning 0 polarity into -1.
    Input: AER in form of EventData object
    '''
    def __init__(self, cutoff_frequency=5):
        self.cutoff = cutoff_frequency
        self.uninitialized = True

    def __call__(self, events):
        height, width = events.height, events.width

        if self.uninitialized:
            self.time_surface = np.zeros((height, width), dtype=np.float32)
            self.image_state = np.zeros((height, width), dtype=np.float32)
            self.uninitialized = False

        # TODO: figure out vectorized method to compute HPF
        # TODO: zip t,x,y,p for possible performance gains
        e = events
        pols = e.p
        pols[pols==0] = -1
        for et, ex, ey, ep in zip(e.t, e.x, e.y, pols): 
            beta = np.exp(-self.cutoff * (et - self.time_surface[ey, ex]))
            self.image_state[ey, ex] = beta * self.image_state[ey, ex] + ep
            self.time_surface[ey, ex] = et

        beta = np.exp(-self.cutoff * (e.t[-1] - self.time_surface))
        self.image_state *= beta
        # TODO: Don't fill with e.t, instead remember last time surface?
        self.time_surface.fill(e.t[-1])

        return self.image_state


class LeakyIntegrator:
    '''
    Stateful transform to generate leaky integrator image.
    Note: Takes in events with polarity: {0,1}. It handles turning 0 polarity into -1.
    Input: AER in form of EventData object
    '''
    def __init__(self, beta=1.0):
        self.beta = beta
        self.uninitialized = True
    
    def __call__(self, events):
        height, width = events.height, events.width

        if self.uninitialized:
            self.image_state = np.zeros((height, width), dtype=np.float32)
            self.uninitialized = False
            
        e = events
        for i in range(e.total_events()):
            pol = -1 if e.p[i] == 0 else 1
            self.image_state[e.y[i], e.x[i]] = self.beta * self.image_state[e.y[i], e.x[i]] + pol

        return self.image_state


# TODO: Time surface
