import numpy as np
import os
import scipy
import matplotlib.pyplot as plt

#avg_across_bins (t)
def average_frames(data, avg_block=5):
    # Ensure the first dimension (time) is divisible by avg_block
    if data.shape[0] % avg_block != 0:
        raise ValueError(f"Data length {data.shape[0]} is not divisible by avg_block {avg_block}")

    # Reshape the data to (new_time, avg_block, pixels)
    grouped = data.reshape(-1, avg_block, data.shape[1])
    
    # Average along the second dimension (axis=1) to reduce time dimension
    avg_data = np.mean(grouped, axis=1)
    
    return avg_data

def minmax_scale(data, new_min=0, new_max=1):
    # Find the global min and max values across the entire dataset
    min_value = np.min(data)
    max_value = np.max(data)
    
    # Scale the data based on the global min and max
    tiff_scaled = (data - min_value) / (max_value - min_value) * (new_max - new_min) + new_min
    return tiff_scaled

def threshold(data):
    # Compute mean and std across the time dimension (axis=0)
    mean_across_time = np.mean(data, axis=0)
    std_across_time = np.std(data, axis=0)

    # Define the threshold
    thresh = mean_across_time + 1 * std_across_time
    print(f"Threshold array length: {len(thresh)}")

    # Apply threshold: set values below the threshold to 0
    tiff_thresholded = np.where(data >= thresh, data, 0) #changed threshold to thershold (bcs disconintuity), changed back to 0 is ok
    return tiff_thresholded

def detrend(tiff, save_dir, top_n_traces=10, breakpoint=20):
    """
    Detrend tiff and plot the top N traces.
    - tiff:  2D array where rows are time and columns are pixs or rois.
    - top_n_traces (int): The number of traces to plot. Defaults to 10.
    - breakpoint (int or list): Breakpoints for detrending. Defaults to 20.
    """
    
    detrended_tiff = scipy.signal.detrend(tiff, axis=0, type='linear', bp=breakpoint, overwrite_data=False)

    # Plot the top N detrended traces
    fig, axs = plt.subplots(top_n_traces, 1, figsize=(30, 2 * top_n_traces), sharex=True, sharey=True)

    for i in range(top_n_traces):
        axs[i].plot(detrended_tiff[:, i])
        axs[i].set_ylabel(f'pix {i + 1}', fontsize=15)
    
    fig.suptitle(f'Top {top_n_traces} Detrended Traces', size=30)
    fig.text(0.5, 0.04, 'time (sec)', ha='center', va='center', size=25)
    fig.text(0.04, 0.5, 'fluorescence', ha='center', va='center', rotation='vertical', size=25)
    
    plt.xticks(fontsize=15)
    
    # Save the plot
    save_path_top_traces = os.path.join(save_dir, f'top_{top_n_traces}_detr_traces.png')
    plt.savefig(save_path_top_traces, bbox_inches='tight')
    plt.show()

    # Plot the 11th trace (or the next one after top_n_traces)
    if tiff.shape[1] > top_n_traces:
        fig = plt.figure(figsize=(30, 10))
        plt.plot(detrended_tiff[:, top_n_traces])
        plt.title(f'Trace {top_n_traces + 1} Detrended', size=20)

        save_path_single_trace = os.path.join(save_dir_preprocessing, 'detrended_trace.png')
        plt.savefig(save_path_single_trace, bbox_inches='tight')
        plt.show()

    return detrended_tiff

def prepare_data(data_in, iscell):
    
    iscell_bool = iscell[:,0].astype(bool) # getting rid of 'non-cells'
    
    data = data_in[iscell_bool,:] # choose which data to analyse (spks or fluo)
    data = zscore(data, axis=1)
    
    return data
    
def pad(var):

    var = np.concatenate(([var[0]], var))
    print(f' Congrats! New shape is {len(var)}')
    return var 


