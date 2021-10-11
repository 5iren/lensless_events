import numpy as np
import os
import pandas as pd

#Set directories
lensless_txt_dir = 'data/lensless_videos_dataset/lensless_txt'
gt_txt_dir = 'data/lensless_videos_dataset/gt_txt'
lensless_events_dir = 'data/lensless_videos_dataset/lensless_events'
gt_events_dir = 'data/lensless_videos_dataset/gt_events'

#Timewindow properties
timewindow_ms = 10
max_time_ms = 3000

#Load all txt files in directories
_, _, gt_txt_files = next(os.walk(gt_txt_dir))
_, _, lensless_txt_files = next(os.walk(lensless_txt_dir))

#For naming
def make_filename(idx, video_name):
        return f'{video_name}_window_{idx:04}'

#Loop through gt txt files
for txt_file_name in gt_txt_files:
    idx = 1
    video_name = txt_file_name.split('.')[0]
    txt_file_path = os.path.join(gt_txt_dir, txt_file_name)

    #Read contents of txt file and create event data frame
    data = pd.read_csv(txt_file_path, delim_whitespace=True, 
                names=['t', 'x', 'y', 'p'],
                dtype={'t': np.float32, 'x': np.uint16, 'y': np.uint16, 'p': np.int8},
                engine='c')

    t = data['t'].to_numpy()
    x = data['x'].to_numpy()
    y = data['y'].to_numpy()
    p = data['p'].to_numpy()

    #Loop through time windows
    for window in range(0, max_time_ms, timewindow_ms):
        t_start = window * 1e-3
        t_end = t_start + timewindow_ms * 1e-3

        #If events withing timewindow: save numpy array
        window_idxs = (t >= t_start) & (t < t_end)
        win_file = os.path.join( gt_events_dir, make_filename(idx, video_name))
        np.savez(win_file, t=t[window_idxs], x=x[window_idxs], y=y[window_idxs], p=p[window_idxs])

        idx += 1

#Loop through lensles txt files
for txt_file_name in lensless_txt_files:
    idx = 1
    video_name = txt_file_name.split('.')[0]
    txt_file_path = os.path.join(lensless_txt_dir, txt_file_name)

    #Read contents of txt file and create event data frame
    data = pd.read_csv(txt_file_path, delim_whitespace=True, 
                names=['t', 'x', 'y', 'p'],
                dtype={'t': np.float32, 'x': np.uint16, 'y': np.uint16, 'p': np.int8},
                engine='c')

    t = data['t'].to_numpy()
    x = data['x'].to_numpy()
    y = data['y'].to_numpy()
    p = data['p'].to_numpy()

    #Loop through time windows
    for window in range(0, max_time_ms, timewindow_ms):
        t_start = window * 1e-3
        t_end = t_start + timewindow_ms * 1e-3

        #If events withing timewindow: save numpy array
        window_idxs = (t >= t_start) & (t < t_end)
        win_file = os.path.join(lensless_events_dir, make_filename(idx, video_name))
        np.savez(win_file, t=t[window_idxs], x=x[window_idxs], y=y[window_idxs], p=p[window_idxs])

        idx += 1