import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os

'''
Contains readers that yield chunks of events, either by number of events or duration in a window.
'''

### come up with better names for these

class EventData:
    __slots__ = 't', 'x', 'y', 'p', 'width', 'height'
    def __init__(self, t, x, y, p, width, height):
        self.x = x
        self.y = y
        self.t = t
        self.p = p
        self.width = width
        self.height = height

    def total_events(self):
        return len(self.t)

    def __repr__(self):
        return f'{self.width}x{self.height} AER w/ {self.total_events()} events'

    def __len__(self):
        return(len(self.t))

class NumEventsIter:
    '''
    Iterator that yields fixed-number-of-event windows from dataset source.
    Can use causal reconstruction transformation methods (time surface, leaky integrator) with
    the appropriate event transforms since these methods require historical data (can't processed on a single window).
    '''
    def __init__(self, data_path, n_events=30_000, width=240, height=180, transforms=None, total_events=None):
        self.n_events = n_events
        self.width = width
        self.height = height

        self.transforms = transforms

        # Found that Pandas is faster than Numpy at reading csv
        self.iterator = pd.read_csv(data_path, delim_whitespace=True, header=None,
                        names=['t', 'x', 'y', 'p'],
                        dtype={'t': np.float32, 'x': np.uint16, 'y': np.uint16, 'p': np.int8},
                        engine='c', nrows=total_events, chunksize=n_events, memory_map=True)

    def __iter__(self):
        return self

    def __next__(self):
        window = self.iterator.__next__().to_numpy()
        t = window[:,0].astype(np.float32)
        x = window[:,1].astype(np.uint16)
        y = window[:,2].astype(np.uint16)
        p = window[:,3].astype(np.int8)

        events = EventData(t, x, y, p, self.width, self.height)

        if self.transforms:
            for transform in self.transforms:
                events = transform(events)

        return events

    def get_images_list(self):
        images_list = []
        for sample in self:
            images_list.append(sample)

        return images_list


# TODO: How do we handle the end of the dataset where there aren't enough datapoints? RN I just chop off the end.
class NumEventsDataset(Dataset):
    '''
    NumEventsDataset that allows indexed access into the windows.
    Can be used with stateless transforms.
    '''
    def __init__(self, data_path, n_events=30_000, width=240, height=180, skiprows=0, transforms=None, total_events=None):
        self.n_events = n_events
        self.width = width
        self.height = height

        self.transforms = transforms

        data = pd.read_csv(data_path, delim_whitespace=True, header=None,
                names=['t', 'x', 'y', 'p'],
                dtype={'t': np.float32, 'x': np.uint16, 'y': np.uint16, 'p': np.int8},
                engine='c', nrows=total_events, skiprows=skiprows)

        self.t = data['t'].to_numpy()
        self.x = data['x'].to_numpy()
        self.y = data['y'].to_numpy()
        self.p = data['p'].to_numpy()

    def __len__(self):
        return int(np.floor(len(self.t)/self.n_events))

    def __getitem__(self, idx):
        idx_start = idx * self.n_events
        t = self.t[idx_start:idx_start+self.n_events]
        x = self.x[idx_start:idx_start+self.n_events]
        y = self.y[idx_start:idx_start+self.n_events]
        p = self.p[idx_start:idx_start+self.n_events]


        events = EventData(t, x, y, p, self.width, self.height)

        if self.transforms:
            for transform in self.transforms:
                events = transform(events)
        
        return events

    def get_images_list(self):
        '''
        This method is here to accumulate all the images at once to feed into an algorithm.
        '''
        images_list = []
        for i in range(len(self)):
            images_list.append(self[i])
        return images_list


# TODO: How do we handle the end of the dataset where there aren't enough datapoints? RN I just chop off the end.
class NumEventsDatasetLazy(Dataset):
    '''
    NumEventsDataset that allows indexed access into the windows.
    Can be used with stateless transforms.
    '''
    def __init__(self, data_path, n_events=30_000, width=240, height=180, skiprows=0, transforms=None, total_events=None):
        self.data_path = data_path
        self.n_events = n_events
        self.width = width
        self.height = height
        self.skiprows = skiprows

        self.transforms = transforms

        if total_events is None:
            # Count total number of lines to determine dataset length
            with open(data_path) as fp:
                self.total_events = sum(1 for line in fp) - skiprows
        else:
            self.total_events = total_events




    def __len__(self):
        return int(self.total_events / self.n_events)

    def __getitem__(self, idx):
        skiprows = self.n_events*idx + self.skiprows
        data = pd.read_csv(self.data_path, delim_whitespace=True, header=None,
                names=['t', 'x', 'y', 'p'],
                dtype={'t': np.float32, 'x': np.uint16, 'y': np.uint16, 'p': np.int8},
                engine='c', nrows=self.n_events, skiprows=skiprows)
        
        t = data['t'].to_numpy()
        x = data['x'].to_numpy()
        y = data['y'].to_numpy()
        p = data['p'].to_numpy()

        events = EventData(t, x, y, p, self.width, self.height)

        if self.transforms:
            for transform in self.transforms:
                events = transform(events)
        
        return events

    def get_images_list(self):
        '''
        This method is here to accumulate all the images at once to feed into an algorithm.
        '''
        images_list = []
        for i in range(len(self)):
            images_list.append(self[i])
        return images_list

class TimeWindowDataset(Dataset):
    '''
    TimeWindowDataset that allows indexed access into the windows.
    Can be used with stateless transforms.

    skiprows: Number of rows to skip in file before AER data starts.
    '''
    def __init__(self, data_path, duration_ms=50, width=240, height=180, skiprows=0, transforms=None, total_events=None, all_full=False):
        self.duration_s = duration_ms * 1e-3
        self.width = width
        self.height = height

        self.transforms = transforms

        data = pd.read_csv(data_path, delim_whitespace=True, header=None,
                names=['t', 'x', 'y', 'p'],
                dtype={'t': np.float32, 'x': np.uint16, 'y': np.uint16, 'p': np.int8},
                engine='c', nrows=total_events, skiprows=skiprows)

        self.t = data['t'].to_numpy()
        self.x = data['x'].to_numpy()
        self.y = data['y'].to_numpy()
        self.p = data['p'].to_numpy()
        # Determine if we allow last window to go beyond last timestamp in stream.
        if all_full:
            self.n_windows = int(np.floor(self.t[-1] / (self.duration_s)))
        else:
            self.n_windows = int(np.ceil(self.t[-1] / (self.duration_s)))

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        t_start, t_end = idx*self.duration_s, (idx+1)*self.duration_s
        window_idxs = (self.t >= t_start) & (self.t < t_end)
        t = self.t[window_idxs]
        y = self.y[window_idxs]
        p = self.p[window_idxs]
        x = self.x[window_idxs]


        events = EventData(t, x, y, p, self.width, self.height)

        if self.transforms:
            for transform in self.transforms:
                events = transform(events)
        
        return events

    def get_images_list(self):
        '''
        This method is here to accumulate all the images at once to feed into an algorithm.
        '''
        images_list = []
        for i in range(len(self)):
            images_list.append(self[i])
        return images_list


class TimeWindowDatasetCached(Dataset):
    '''
    TimeWindowDataset that allows indexed access into the windows.
    Can be used with stateless transforms.

    Splits AER into time windows and caches numpy arrays in folder next to txt file
    for fast retrieval.

    skiprows: Number of rows to skip in file before AER data starts.
    '''
    def __init__(self, data_path, duration_ms=50, width=240, height=180, skiprows=0, transforms=None, total_events=None, all_full=False):
        self.duration_s = duration_ms * 1e-3
        self.width = width
        self.height = height

        self.transforms = transforms

        data = pd.read_csv(data_path, delim_whitespace=True, header=None,
                names=['t', 'x', 'y', 'p'],
                dtype={'t': np.float32, 'x': np.uint16, 'y': np.uint16, 'p': np.int8},
                engine='c', nrows=total_events, skiprows=skiprows)
        t = data['t'].to_numpy()
        x = data['x'].to_numpy()
        y = data['y'].to_numpy()
        p = data['p'].to_numpy()
        # Determine if we allow last window to go beyond last timestamp in stream.
        if all_full:
            self.n_windows = int(np.floor(t[-1] / (self.duration_s)))
        else:
            self.n_windows = int(np.ceil(t[-1] / (self.duration_s)))
        # Check for existence of usable cache
        data_folder = Path(data_path).parent
        cache_folder = data_folder / f'windows_{duration_ms:.3f}ms/'
        if not cache_folder.exists():
            cache_folder.mkdir(exist_ok=False)
            # Build the cache
            for idx in range(self.__len__()):
                win_file = cache_folder / self.make_filename(idx)

                t_start, t_end = idx*self.duration_s, (idx+1)*self.duration_s
                window_idxs = (t >= t_start) & (t < t_end)

                np.savez(win_file, t=t[window_idxs], x=x[window_idxs], y=y[window_idxs], p=p[window_idxs])

        self.cache_folder = cache_folder
        

    def __len__(self):
        return self.n_windows

    def make_filename(self, idx):
        return f'window_{idx:04}.npz'

    def __getitem__(self, idx):
        # TODO: time this implementation over saving each array to its own file
        win_file = self.cache_folder / self.make_filename(idx)

        npfile = np.load(win_file)
        t = npfile['t']
        x = npfile['x']
        y = npfile['y']
        p = npfile['p']

        events = EventData(t, x, y, p, self.width, self.height)

        if self.transforms:
            for transform in self.transforms:
                events = transform(events)
        
        return events

    def get_images_list(self):
        '''
        This method is here to accumulate all the images at once to feed into an algorithm.
        '''
        images_list = []
        for i in range(len(self)):
            images_list.append(self[i])
        return images_list

class TimeWindowDatasetCachedMVSEC(Dataset):
    '''
    ** Doesn't align events with flow exactly 
    TimeWindowDataset that allows indexed access into the windows.
    Can be used with stateless transforms.

    Splits AER into time windows and caches numpy arrays in folder next to txt file
    for fast retrieval.

    skiprows: Number of rows to skip in file before AER data starts.
    
    converts ros time to miliseconds for MVSEC data
    '''
    def __init__(self, data_path, duration_ms=50, width=240, height=180, skiprows=0, transforms=None, total_events=None, all_full=False):
        self.duration_s = duration_ms * 1e-3
        self.width = width
        self.height = height

        self.transforms = transforms

        data = pd.read_csv(data_path, delim_whitespace=True, header=None,
                names=['t', 'x', 'y', 'p'],
                dtype={'t': np.float64, 'x': np.uint16, 'y': np.uint16, 'p': np.int8},
                engine='c', nrows=total_events, skiprows=skiprows)
        t = (data['t'].to_numpy())-(data['t'][0])
        x = data['x'].to_numpy()
        y = data['y'].to_numpy()
        p = data['p'].to_numpy()
        #print(t[-1])
        # Determine if we allow last window to go beyond last timestamp in stream.
        if all_full:
            self.n_windows = int(np.floor(t[-1] / (self.duration_s)))
        else:
            self.n_windows = int(np.ceil(t[-1] / (self.duration_s)))
        # Check for existence of usable cache
        data_folder = Path(data_path).parent
        cache_folder = data_folder / f'windows_{duration_ms:.3f}ms/'
        if not cache_folder.exists():
            cache_folder.mkdir(exist_ok=False)
            # Build the cache
            for idx in range(self.__len__()):
                win_file = cache_folder / self.make_filename(idx)

                t_start, t_end = idx*self.duration_s, (idx+1)*self.duration_s
                window_idxs = (t >= t_start) & (t < t_end)

                np.savez(win_file, t=t[window_idxs], x=x[window_idxs], y=y[window_idxs], p=p[window_idxs])

        self.cache_folder = cache_folder
        #print(total_events/self.n_windows)

    def __len__(self):
        return self.n_windows

    def make_filename(self, idx):
        return f'window_{idx:04}.npz'

    def __getitem__(self, idx):
        # TODO: time this implementation over saving each array to its own file
        win_file = self.cache_folder / self.make_filename(idx)

        npfile = np.load(win_file)
        t = npfile['t']
        x = npfile['x']
        y = npfile['y']
        p = npfile['p']

        events = EventData(t, x, y, p, self.width, self.height)

        if self.transforms:
            for transform in self.transforms:
                events = transform(events)
        
        return events

    def get_images_list(self):
        '''
        This method is here to accumulate all the images at once to feed into an algorithm.
        '''
        images_list = []
        for i in range(len(self)):
            images_list.append(self[i])
        return images_list
        
        
class DatasetCachedMVSEC(Dataset):
    '''
    Dataset for use with MVSEC previously cached events (create_mvsec_dataset.py)

    Retrieves events from cached files

    '''
    def __init__(self, data_path, width=240, height=180, transforms=None, total_events=None, all_full=False):
        self.width = width
        self.height = height
        self.transforms = transforms
        
        self.cache_folder = data_path 


    def __len__(self):
        return len(os.listdir(self.cache_folder))

    def __getitem__(self, idx):
        win_file = str(self.cache_folder) + '/window_'+str(idx)+'.npz'
        npfile = np.load(win_file)
        t = npfile['t']
        x = npfile['x']
        y = npfile['y']
        p = npfile['p']

        events = EventData(t, x, y, p, self.width, self.height)

        if self.transforms:
            for transform in self.transforms:
                events = transform(events)
        
        return events

    def get_images_list(self):
        '''
        This method is here to accumulate all the images at once to feed into an algorithm.
        '''
        images_list = []
        for i in range(len(self)):
            images_list.append(self[i])
        return images_list


class TimeWindowDatasetLazy(Dataset):
    '''
    TimeWindowDataset that allows indexed access into the windows.
    Can be used with stateless transforms.

    Lazily loads windows of stream so worker's don't consume all memory.

    skiprows: Number of rows to skip in file before AER data starts.
    '''
    def __init__(self, data_path, duration_ms=50, width=240, height=180, skiprows=0, transforms=None, total_events=None, all_full=False):
        self.data_path = data_path
        self.duration_s = duration_ms * 1e-3
        self.width = width
        self.height = height
        self.skiprows = skiprows
        self.transforms = transforms

        self.index_win_starts = []
        self.index_win_lengths = []

        chunksize = 500_000

        # Build our index of row start numbers and window lengths for efficient csv reading
        chunks = pd.read_csv(data_path, delim_whitespace=True, header=None,
                names=['t', 'x', 'y', 'p'],
                dtype={'t': np.float32, 'x': np.uint16, 'y': np.uint16, 'p': np.int8},
                engine='c', nrows=total_events, skiprows=skiprows, chunksize=chunksize)

        end_time = self.duration_s
        start_idx = 0
        end_idx = None
        chunk_number = 0
        total_length = None

        # Chunk dataset for this preloading step to reduce memory usage
        for chunk in chunks:
            # For each chunk, find end index of the next time window
            while True:
                t = chunk['t'].to_numpy()
                total_length = len(t) + chunk_number*chunksize

                next_window = t >= end_time
                # Read next chunk if this chunk doesn't contain the last event in window
                if np.count_nonzero(next_window) == 0:
                    break
                end_idx = np.nonzero(next_window)[0][0]
                # Add chunk offset to global index
                end_idx += chunk_number * chunksize
                self.index_win_starts.append(start_idx)
                self.index_win_lengths.append(end_idx - start_idx)
                start_idx = end_idx
            
                end_time += self.duration_s
            
            chunk_number += 1

        # Determine if we allow last window to go beyond last timestamp in stream.
        if not all_full:
            if total_length > sum(self.index_win_lengths):
                self.index_win_starts.append(self.index_win_starts[-1] + self.index_win_lengths[-1])
                self.index_win_lengths.append(total_length - self.index_win_starts[-1])

        # Offset all start indices by skiprows
        self.index_win_starts = [skiprows + idx for idx in self.index_win_starts]

    def __len__(self):
        return len(self.index_win_lengths)

    def __getitem__(self, idx):
        skiprows = self.index_win_starts[idx]
        nrows = self.index_win_lengths[idx]

        window = pd.read_csv(self.data_path, delim_whitespace=True, header=None,
                names=['t', 'x', 'y', 'p'],
                dtype={'t': np.float32, 'x': np.uint16, 'y': np.uint16, 'p': np.int8},
                engine='c', nrows=nrows, skiprows=skiprows)

        t = window['t'].to_numpy()
        x = window['x'].to_numpy()
        y = window['y'].to_numpy()
        p = window['p'].to_numpy()
        
        events = EventData(t, x, y, p, self.width, self.height)

        if self.transforms:
            for transform in self.transforms:
                events = transform(events)
        
        return events

    def get_images_list(self):
        '''
        This method is here to accumulate all the images at once to feed into an algorithm.
        '''
        images_list = []
        for i in range(len(self)):
            images_list.append(self[i])
        return images_list