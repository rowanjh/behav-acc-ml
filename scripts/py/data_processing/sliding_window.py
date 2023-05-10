##################################################
# All functions related to applying sliding window on a dataset
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

import numpy as np

def nunique(a, axis):
    """
    Returns the number of unique values along an axis of a numpy array. 
    Used to determine if a window has transitions (more than 1 unique behaviour).
    Code from https://stackoverflow.com/questions/48473056/number-of-unique-elements-per-row-in-a-numpy-array
    """
    return (np.diff(np.sort(a,axis=axis),axis=axis)!=0).sum(axis=axis)+1


def sliding_window_seconds(data, length_in_seconds=1, sampling_rate=50, overlap_ratio=None):
    """
    Return a sliding window measured in seconds over a data array.

    :param data: dataframe
        Input array, can be numpy or pandas dataframe
    :param length_in_seconds: int, default: 1
        Window length as seconds
    :param sampling_rate: int, default: 50
        Sampling rate in hertz as integer value
    :param overlap_ratio: int, default: None
        Overlap is meant as percentage and should be an integer value
    :return: tuple of windows and indices
    """
    windows = []
    indices = []
    curr = 0
    overlapping_elements = 0
    win_len = int(length_in_seconds * sampling_rate)
    if overlap_ratio is not None:
        overlapping_elements = int((overlap_ratio / 100) * win_len)
        if overlapping_elements >= win_len:
            print('Number of overlapping elements exceeds window size.')
            return
    while curr < len(data) - win_len:
        windows.append(data[curr:curr + win_len])
        indices.append([curr, curr + win_len])
        curr = curr + win_len - overlapping_elements
    return np.array(windows), np.array(indices)


def sliding_window_samples(data, samples_per_window, overlap_ratio):
    """
    Return a sliding window measured in number of samples over a data array.

    :param data: dataframe
        Input array, can be numpy or pandas dataframe
    :param samples_per_window: int
        Window length as number of samples per window
    :param overlap_ratio: int
        Overlap is meant as percentage and should be an integer value
    :return: dataframe, list
        Tuple of windows and indices
    """
    windows = []
    indices = []
    curr = 0
    win_len = int(samples_per_window)
    if overlap_ratio is not None:
        overlapping_elements = int((overlap_ratio / 100) * (win_len))
        if overlapping_elements >= win_len:
            print('Number of overlapping elements exceeds window size.')
            return
    while curr < len(data) - win_len:
        windows.append(data[curr:curr + win_len])
        indices.append([curr, curr + win_len])
        curr = curr + win_len - overlapping_elements
    try:
        result_windows = np.array(windows)
        result_indices = np.array(indices)
    except:
        result_windows = np.empty(shape=(len(windows), win_len, data.shape[1]), dtype=object)
        result_indices = np.array(indices)
        for i in range(0, len(windows)):
            result_windows[i] = windows[i]
            result_indices[i] = indices[i]
    return result_windows, result_indices


def apply_sliding_window(full_data, parameters):
    """
    Function which transforms a dataset into windows of a specific size and overlap.

    :param data_x: numpy float array
        Array containing the features (can be 2D)
    :param data_y: numpy float array
        Array containing the corresponding labels to the dataset (is 1D)
    :param sliding_window_size: integer or float
        Size of each window (either in seconds or units)
    :param unit: string, ['units', 'seconds']
        Unit in which the sliding window is measured
    :param sampling_rate: integer
        Number of hertz in which the dataset is sampled
    :param sliding_window_overlap: integer
        Amount of overlap between the sliding windows (measured in percentage, e.g. 20 is 20%)
    :return:
    """
    sliding_window_size = parameters.sw_length
    sampling_rate = parameters.sampling_rate
    sliding_window_overlap = parameters.sw_overlap
        
    output_x = None
    output_y = None
    full_data = full_data.to_numpy()
    for i, fold in enumerate(np.unique(full_data[:, 0])):
        fold_data = full_data[full_data[:, 0] == fold]
        fold_x, fold_y = fold_data[:, :-1], fold_data[:, -1]
        if parameters.sw_unit == 'units':
            tmp_x, _ = sliding_window_samples(fold_x, sliding_window_size, sliding_window_overlap)
            tmp_y, _ = sliding_window_samples(fold_y, sliding_window_size, sliding_window_overlap)
        elif parameters.sw_unit == 'seconds':
            tmp_x, _ = sliding_window_seconds(fold_x, sliding_window_size, sampling_rate, sliding_window_overlap)
            tmp_y, _ = sliding_window_seconds(fold_y, sliding_window_size, sampling_rate, sliding_window_overlap)

            # Find windows that are non-contiguous (e.g. the next data point belongs to a different segment, 
            # which happens because all segments data were stacked into one big data frame)
            n_windows = tmp_x.shape[0]
            non_contiguous_windows = []
            for i in range(n_windows):
                this_window = tmp_x[i,:,:]
                segment_ids = np.unique(this_window[:,1])
                if len(segment_ids) > 1:
                    non_contiguous_windows.append(i)
            # Purge non-contiguous windows
            tmp_x = np.delete(tmp_x, non_contiguous_windows, 0)
            tmp_y = np.delete(tmp_y, non_contiguous_windows, 0)
            
        if output_x is None:
            output_x = tmp_x
            output_y = tmp_y
        else:
            output_x = np.concatenate((output_x, tmp_x), axis=0)
            output_y = np.concatenate((output_y, tmp_y), axis=0)
    
    # Split off ID columns into separate variables
    output_fold = output_x[:,0,0].astype('int')
    output_obs = output_x[:,0,1].astype('int')
    output_time = output_x[:,0,2]
    output_x = output_x[:, :, 3:6].astype(np.float32)



    # Add a label if the window is a transition
    beh_count = nunique(output_y, 1)
    transition = beh_count > 1
    
    # Take the final sample in each window to be the selected label (note: transitions are not excluded)
    output_y = output_y[:,-1].astype(np.int8)
    data = {"win_X":output_x, "win_fold":output_fold, "win_segment_id":output_obs, 
            "win_start": output_time,"win_y": output_y, "transition": transition}

    return data
