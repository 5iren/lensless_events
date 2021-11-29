import numpy as np
import os
import pandas as pd
#TODO: Handle non-zero on both timewindows 


#Set directories
lensless_txt_dir = 'data/lensless_videos_dataset/lensless_txt'
gt_txt_dir =       'data/lensless_videos_dataset/gt_txt'
lensless_events_dir = 'data/lensless_videos_dataset/lensless_events'
gt_events_dir =       'data/lensless_videos_dataset/gt_events'

#Timewindow properties
timewindow_ms = 30
max_time_ms = 3000 - timewindow_ms
min_events = 3000

#Load all txt files in directories
_, _, gt_txt_files = next(os.walk(gt_txt_dir))
_, _, lensless_txt_files = next(os.walk(lensless_txt_dir))

#Check if files sizes are the same
assert len(gt_txt_files) == len(lensless_txt_files) ,  "Quantity of files is not the same"

#For naming
def make_filename(idx, video_name):
        return f'{video_name}_window_{idx:04}'

#For storing valid indices (non-zero)

#Loop through txt files
for i in range(len(gt_txt_files)):
    idx = 1
    #Get name
    gt_video_name = gt_txt_files[i].split('.')[0]
    lensless_video_name = lensless_txt_files[i].split('.')[0]
    
    #Get full path 
    gt_file_path = os.path.join(gt_txt_dir, gt_txt_files[i])
    lensless_file_path = os.path.join(lensless_txt_dir, lensless_txt_files[i])

    #Read contents fo txt file and create event data frame
    gt_data = pd.read_csv(gt_file_path, delim_whitespace=True, 
                          names=['t', 'x', 'y', 'p'],
                          dtype={'t': np.float32, 'x': np.uint16, 'y': np.uint16, 'p': np.int8},
                          engine='c')

    gt_t = gt_data['t'].to_numpy()
    gt_x = gt_data['x'].to_numpy()
    gt_y = gt_data['y'].to_numpy()
    gt_p = gt_data['p'].to_numpy()

    lensless_data = pd.read_csv(lensless_file_path, delim_whitespace=True, 
                                names=['t', 'x', 'y', 'p'],
                                dtype={'t': np.float32, 'x': np.uint16, 'y': np.uint16, 'p': np.int8},
                                engine='c')

    lensless_t = lensless_data['t'].to_numpy()
    lensless_x = lensless_data['x'].to_numpy()
    lensless_y = lensless_data['y'].to_numpy()
    lensless_p = lensless_data['p'].to_numpy()

    #Loop through time windows
    for window in range(0, max_time_ms, timewindow_ms):
        t_start = window * 1e-3
        t_end = t_start + timewindow_ms * 1e-3

        #If events within timewindow: save numpy array
        gt_window_idxs = (gt_t >= t_start) & (gt_t < t_end)
        lensless_window_idxs = (lensless_t >= t_start) & (lensless_t < t_end)
        gt_win_file = os.path.join(gt_events_dir, make_filename(idx, gt_video_name))
        lensless_win_file = os.path.join(lensless_events_dir, make_filename(idx, lensless_video_name))

        #Save window if it contains more than the minimum events
        if (gt_t[gt_window_idxs].shape[0] > min_events) and  (lensless_t[lensless_window_idxs].shape[0] > min_events):
            np.savez(gt_win_file, t=gt_t[gt_window_idxs], x=gt_x[gt_window_idxs], y=gt_y[gt_window_idxs], p=gt_p[gt_window_idxs])
            np.savez(lensless_win_file, t=lensless_t[lensless_window_idxs], x=lensless_x[lensless_window_idxs], y=lensless_y[lensless_window_idxs], p=lensless_p[lensless_window_idxs])
        idx += 1