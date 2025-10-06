#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
# going to root directory (if not there yet)
current_dir = os.getcwd().split('/')[-1]
if current_dir != 'pixelNMF': 
    os.chdir('..')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tifffile
import cv2
from skimage import filters
from scipy.ndimage import label
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from pathlib import Path
# Optional dependency: Hartigan's Dip Test
try:
    from diptest import diptest
except Exception:
    diptest = None
# Make sure local package 'patchnmf' (under mousecraft/) is importable
try:
    from patchnmf.data_io import *
    from patchnmf.analyse.videography_compute import *
    from patchnmf.analyse.videography_plot import *
except ModuleNotFoundError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    mousecraft_dir = os.path.join(project_root, 'mousecraft')
    if mousecraft_dir not in sys.path:
        sys.path.insert(0, mousecraft_dir)
    # Retry imports after adjusting sys.path
    from patchnmf.data_io import *
    from patchnmf.analyse.videography_compute import *
    from patchnmf.analyse.videography_plot import *

# Plot control: disable interactive windows if --no-plots
NO_PLOTS = ('--no-plots' in sys.argv)
if NO_PLOTS:
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        plt.ioff()
        plt.show = lambda *args, **kwargs: None  # no blocking
    except Exception:
        pass

# setting common plot params 
from matplotlib import rcParams

rcParams['figure.figsize'] = [20, 4]
rcParams['font.size'] = 15
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.autolayout'] = True
rcParams['agg.path.chunksize'] = 10000  # reduce path complexity to avoid OOM in draw_path

def mkdir(path):
    if not os.path.exists(path): os.makedirs(path)
    return path

data_path = rf'C:\Users\zaggila\Documents\pixelNMF\data_proc\cells'
# sessions = sorted([f for f in os.listdir(data_path) if f.endswith('_cell_control')])
# print(f'All sessions: {sessions}')

ds = 'sz105\\2025_05_30_a'

subject_path = os.path.join(data_path, ds)  # default subject path
movie_path = os.path.join(subject_path, 'cam_crop.tif')

# Optional: allow external motion energy path via CLI argument
external_me_path = None
video_input_path = None  # allow passing a video path (tif/tiff/avi) via CLI
# Optional method overrides via CLI (positional):
# argv[1] = motion_energy.npy OR video path (optional)
# argv[2] = binary method (one of: otsu, li, mean_sd)
# argv[3] = twitch method (one of: li, mad, percentile_95, mean_3sd, otsu)
# Optional flags:
#   --no-plots
#   --fps <raw_fps_number>
binary_method = 'otsu'
twitch_method = 'li'
# Optional frame range selection (start inclusive, end exclusive in original frame index space)
start_frame = None
end_frame = None
if len(sys.argv) > 1 and isinstance(sys.argv[1], str):
    candidate = sys.argv[1]
    if os.path.exists(candidate):
        lc = candidate.lower()
        if lc.endswith('.npy'):
            external_me_path = os.path.abspath(candidate)
            subject_path = os.path.dirname(external_me_path)
            movie_path = None
        elif lc.endswith(('.tif', '.tiff', '.avi')):
            video_input_path = os.path.abspath(candidate)
            subject_path = os.path.dirname(video_input_path)
            movie_path = video_input_path
        else:
            pass
    elif sys.argv[1].lower() in {'otsu','li','mean_sd'}:
        binary_method = sys.argv[1].lower()
if len(sys.argv) > 2:
    arg2 = sys.argv[2].lower()
    if arg2 in {'otsu','li','mean_sd'}:
        binary_method = arg2
    elif arg2 in {'mad','percentile_95','mean_3sd','li','otsu'}:
        twitch_method = arg2
if len(sys.argv) > 3:
    arg3 = sys.argv[3].lower()
    if arg3 in {'mad','percentile_95','mean_3sd','li','otsu'}:
        twitch_method = arg3

# Parse optional flags
raw_fps = None
if '--fps' in sys.argv:
    try:
        idx = sys.argv.index('--fps')
        raw_fps = float(sys.argv[idx + 1])
    except Exception:
        raw_fps = None
if '--start' in sys.argv:
    try:
        idx = sys.argv.index('--start')
        start_frame = int(sys.argv[idx + 1])
    except Exception:
        start_frame = None
if '--end' in sys.argv:
    try:
        idx = sys.argv.index('--end')
        end_frame = int(sys.argv[idx + 1])
    except Exception:
        end_frame = None

# set parameters 
gaussian_sigma = 5

# Thresholds for binarisation
threshold_binary = 'otsu'  # threshold for binary: a/r state detcetion -
threshold_twitch = 'otsu' #threshold for twitch detection (more permissive -> detect more/lose less)
# thresh_factor = 1.0  # used for "mean+std" option

# Averaging configuration
avg_block = 5  # frames are averaged by 5 later

# Twitch detection thresholds – framerate in averaged domain
if raw_fps is not None and raw_fps > 0:
    framerate = float(raw_fps) / avg_block  # effective Hz after averaging
else:
    framerate = 3  # default effective Hz if unknown
scaling_factor = avg_block  # for matching back to original time

active_motion_duration_min = framerate * 1 # 1 sec 
long_active_motion_duration_min = framerate * 3 # at least 3 seconds to be considered a long awake motion eg run 
twitch_duration_max = active_motion_duration_min
twitch_min_distance_from_active = framerate * 3  # at least 3 sec away from awake motion to be detected as twitch 
print(f' max_twitch_allowed: {twitch_duration_max/framerate:.2f} sec (fps={framerate:.2f} Hz, raw_fps={raw_fps if raw_fps is not None else "default"})') 

is_sleap = False

save_dir = subject_path
save_dir_videography = mkdir(f'{save_dir}/mousecraft_automatic_classifications')
save_dir_videography = Path(save_dir_videography)

# # Compute/Load motion energy

def compute_motion_energy(movie_path=None, xrange=None, yrange=None, save_path=None, start_frame=None, end_frame=None):
    """
    Compute motion energy from a multi-frame TIFF movie of mouse movement.

    Parameters:
    - movie_path: Path to the multi-frame TIFF file (None, unless ititial run)
    - xrange: Range of x-values to crop the image (optional)
    - yrange: Range of y-values to crop the image (optional)
    - save_path: Path to save the motion energy result (optional)
    """
    
    #check if motion energy has already been computed, if that's the case load it

    motion_energy_path = os.path.join(save_path,"motion_energy.npy")
    if os.path.exists(motion_energy_path):

        print(f'Motion energy already computed. Loading from {motion_energy_path}')
        motion_energy = np.load(motion_energy_path)
        return motion_energy

    if not movie_path:

        raise ValueError('Please provide the tiff for the initial run') 
    
    # Handle TIFF stacks and AVI videos
    ext = os.path.splitext(movie_path)[1].lower()
    if ext in ('.tif', '.tiff'):
        # Stream frames to avoid loading entire stack into RAM
        with tifffile.TiffFile(movie_path) as tif:
            pages = tif.pages
            num_frames = len(pages)
            first = pages[0].asarray()
            height, width = first.shape
            print(f'Loaded TIFF (streaming) with {num_frames} frames, height={height}, width={width}')

            # Determine iteration bounds (compute diff for i in [iter_start, iter_end])
            iter_start = 1 if not start_frame or start_frame < 1 else min(start_frame, num_frames - 1)
            iter_end = (num_frames - 1) if not end_frame else min(end_frame - 1, num_frames - 1)
            if iter_end < iter_start:
                iter_end = iter_start

            me_values = []
            img_prev = pages[iter_start - 1].asarray().astype(np.float32)
            for i in range(iter_start, iter_end + 1):
                img = pages[i].asarray().astype(np.float32)
                diff = img - img_prev
                me_values.append(float(np.sum(diff * diff)))
                img_prev = img
                if i % 1000 == 0:
                    print(f'Done computing for {i}/{num_frames} frames')
            motion_energy = np.array(me_values, dtype=np.float64)
    elif ext == '.avi':
        cap = cv2.VideoCapture(movie_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {movie_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Determine iteration bounds
        iter_start = 1 if not start_frame or start_frame < 1 else min(start_frame, total - 1)
        iter_end = (total - 1) if not end_frame else min(end_frame - 1, total - 1)
        if iter_end < iter_start:
            iter_end = iter_start

        # Seek to iter_start - 1
        target_prev = iter_start - 1
        if target_prev > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_prev)
        ret, frame_prev = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Failed to read frame for initialization")
        frame_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
        me_values = []
        i = iter_start
        while i <= iter_end:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff = gray - frame_prev
            me_values.append(float(np.sum(diff * diff)))
            frame_prev = gray
            if i % 1000 == 0:
                print(f'Done computing for {i}/{total} frames')
            i += 1
        cap.release()
        motion_energy = np.array(me_values, dtype=np.float64)
        print(f'Loaded AVI with {len(motion_energy)} ME frames in selected range')
    else:
        raise ValueError(f"Unsupported video format: {ext}")

    # Normalize motion energy
    if motion_energy.size > 0:
        maxv = np.max(motion_energy)
        if maxv > 0:
            motion_energy = motion_energy / maxv

    # Save to disk for reuse
    if save_path is not None:
        try:
            os.makedirs(save_path, exist_ok=True)
            # Avoid overwriting full ME if a subset was computed
            if start_frame is None and end_frame is None:
                out_name = 'motion_energy.npy'
            else:
                s = 0 if start_frame is None else start_frame
                e = 'end' if end_frame is None else end_frame
                out_name = f'motion_energy_subset_{s}_{e}.npy'
            np.save(os.path.join(save_path, out_name), motion_energy)
            print(f"Saved {out_name} to {save_path}")
        except Exception as e:
            print(f"Warning: could not save motion energy: {e}")

    return motion_energy

# Load motion energy: prefer external path if provided, else compute from video
if external_me_path is not None:
    print(f"Loading external motion energy: {external_me_path}")
    motion_energy_orig = np.load(external_me_path)
    # Apply optional slicing on external ME
    if start_frame is not None or end_frame is not None:
        s = 0 if start_frame is None else max(0, start_frame)
        e = None if end_frame is None else max(s, end_frame)
        motion_energy_orig = motion_energy_orig[s:e]
        print(f"Using ME slice [{s}:{e}] -> length {len(motion_energy_orig)}")
else:
    # If a video path was passed via CLI, prefer that; otherwise use default movie_path
    chosen_movie = video_input_path if video_input_path is not None else movie_path
    motion_energy_orig = compute_motion_energy(
        movie_path=chosen_movie,
        xrange=None,
        yrange=None,
        save_path=os.path.dirname(chosen_movie) if chosen_movie else subject_path,
        start_frame=start_frame,
        end_frame=end_frame
    )

def _pad_to_multiple(data, multiple):
    data = np.asarray(data)
    remainder = len(data) % multiple
    if remainder == 0:
        return data
    pad_len = multiple - remainder
    return np.pad(data, (0, pad_len), mode='edge')

# Preview original ME
plt.plot(motion_energy_orig)
plt.show()

# # Load SLEAP output (if computed)

#load sleap output

if is_sleap is True:
    files = os.listdir(subject_path)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    # Check if there is exactly one CSV file
    if len(csv_files) == 1:
        # Load the CSV file into a DataFrame
        sleap_output_path = os.path.join(subject_path, csv_files[0])  # Get full path to the CSV file
        sleap_output = pd.read_csv(sleap_output_path)  # Pass the file path
        print(f"Loaded {csv_files[0]}")
    else:
        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV files found")
        else:
            raise FileExistsError("Multiple CSV files found. Please ensure there is only one CSV file in the subject directory.")

# # Average motion energy (from 15Hz to 3Hz - denoise)

def average_frames(data, avg_block=None):
    # Ensure data is a NumPy array
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    # Check divisibility
    if len(data) % avg_block != 0:
        raise ValueError(f"Data length {len(data)} is not divisible by avg_block {avg_block}")

    # Reshape for 1D data
    if len(data.shape) == 1:  
        grouped = data.reshape(-1, avg_block)
    else:  # Reshape for 2D data
        grouped = data.reshape(-1, avg_block, data.shape[1])

    # Average across the `avg_block` dimension
    avg_data = np.mean(grouped, axis=1)
    print(f'Congrats again! Data is now {len(avg_data)}')
    return avg_data

# Ensure length is divisible by avg_block before averaging
motion_energy_ready = _pad_to_multiple(motion_energy_orig, avg_block)
motion_energy = average_frames(motion_energy_ready, avg_block = 5)

length_acts = len(motion_energy)
# for plotting xticks to seconds
frame_ticks = range(0, length_acts+ 100, 300)  # 300 frames is 100 sec
second_ticks = [int(tick/framerate) for tick in frame_ticks]   # Convert to seconds

# # Smooth motion energy

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

# # Compute SNR (signal to noise ratio) of motion energy & adjust smoothing 

snr = signaltonoise(motion_energy)
print(snr)

min_window_secs = 3  # minimal smoothing in seconds
max_window_secs = 15  # maximal smoothing in seconds here 

# Map SNR to window size: higher SNR -> shorter window
snr = np.clip(snr, 0.1, 10)  # avoid divide-by-zero or extremes
snr_norm = (10 - snr) / 9.9  # normalize to [0, 1]
adaptive_window_secs = min_window_secs + snr_norm * (max_window_secs - min_window_secs)
adaptive_window_length = int(adaptive_window_secs * framerate)

# Ensure odd and >= polyorder+2
polyorder = 3
if adaptive_window_length % 2 == 0:
    adaptive_window_length += 1
adaptive_window_length = max(adaptive_window_length, polyorder + 3)

print(f"Adaptive window length: {adaptive_window_length}")

# smooth 
smoothed_motion_energy = savgol_filter(motion_energy, window_length=adaptive_window_length,
                                       polyorder=polyorder, mode='interp')
# plot 
plt.figure(figsize=(30, 5), dpi=300)
plt.plot(motion_energy, label='raw motion energy', linewidth=2)
plt.plot(smoothed_motion_energy, label='smoothed motion energy', linewidth=3)
plt.legend()
plt.title(f"adaptive Smoothing | SNR = {snr:.2f}")
plt.tight_layout()
plt.show()

# # Bimodality test
# 
# Check whether there are the motion contains two states:
# 
# If yes: proceed with active-rest binarisation
# If not: Then assume we only have rest/sleep state and do directly 

def is_bimodal_with_two_peaks(data, plot=False, prominence=0.1):
    """
    Check if data is bimodal by KDE peak detection.
    
    Parameters:
        data (array-like): 1D data (e.g. smoothed motion energy)
        plot (bool): Whether to plot KDE with detected peaks
        prominence (float): Minimum prominence to count a peak
        
    Returns:
        bimodal (bool): True if exactly 2 peaks detected
        num_peaks (int): Number of peaks detected
    """
    data = np.asarray(data)
    x_vals = np.linspace(data.min(), data.max(), 1000)
    kde = gaussian_kde(data)
    density = kde(x_vals)

    peaks, properties = find_peaks(density, prominence=prominence)
    num_peaks = len(peaks)
    bimodal = (num_peaks == 2)

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(x_vals, density, label='KDE')
        plt.plot(x_vals[peaks], density[peaks], 'ro', label='Peaks')
        plt.title(f"KDE: {num_peaks} peak(s) detected")
        plt.xlabel("Motion Energy")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return bimodal, num_peaks

if diptest is not None:
    dip_stat, dip_p_value = diptest(smoothed_motion_energy)
    print(f"Hartigan's Dip Test: Dip = {dip_stat:.2f}, p-value = {dip_p_value:.3f}")
else:
    dip_stat, dip_p_value = None, 1.0
    print("Hartigan's Dip Test not available (diptest not installed). Skipping.")

bimodal, num_peaks = is_bimodal_with_two_peaks(smoothed_motion_energy, plot=True)

if bimodal:
    print("OK Detected bimodal distribution (2 peaks)")
else:
    print(f"WARN Detected {num_peaks} peaks - not strictly bimodal")

if dip_p_value < 0.05 or bimodal==True:
    bimodality = True 
else:
    bimodality = False   


def bimodality_test(data, kde_prominence=0.1, plot=False):
    """
    Detect bimodality using KDE peak counting and Hartigan's Dip Test.

    Parameters:
        data (array-like): The data to test (e.g., smoothed motion energy)
        kde_prominence (float): Minimum prominence for KDE peaks
        plot (bool): Whether to plot the KDE with detected peaks

    Returns:
        bimodal (bool): True if either method confirms bimodality
        details (dict): Info on Dip Test and KDE peak count
    """
    data = np.asarray(data)

    # KDE + peak detection
    x_vals = np.linspace(data.min(), data.max(), 1000)
    kde = gaussian_kde(data)
    density = kde(x_vals)
    peaks, props = find_peaks(density, prominence=kde_prominence)
    num_kde_peaks = len(peaks)

    # Dip test (if available)
    if diptest is not None:
        dip_stat, dip_p = diptest(data)  # dip test for bimodality : https://skeptric.com/dip-statistic/
    else:
        dip_stat, dip_p = None, 1.0

    bimodal = (num_kde_peaks == 2) or (dip_p < 0.05)

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(x_vals, density, label="KDE")
        plt.plot(x_vals[peaks], density[peaks], "ro", label=f"Peaks ({num_kde_peaks})")
        plt.title(f"KDE Bimodality: {num_kde_peaks} peak(s), Dip p={dip_p:.3f}")
        plt.xlabel("Motion Energy")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

    bimodality_metrics = {
        "bimodal": bimodal,
        "num_kde_peaks": num_kde_peaks,
        "dip_stat": dip_stat,
        "dip_p": dip_p
    }

    return bimodal, bimodality_metrics

bimodal, dbimodality_metrics = bimodality_test(smoothed_motion_energy, kde_prominence=0.1, plot=True)

print(bimodal)

# # Choose statistical threshold for binarisation
# 
# Otsu should theoretically work if there is active-wake transition

# if bimodality 

def compute_thresholds_for_bin_state_detection(motion_signal, title='', save_dir=None, plot=True):
    '''
    Compute statistical thresholds to use for threshold-based state detection (active/awake - rest)
    Here we use: Otsu (prefer if binary distribution), Li (mutliple peak distribution), or mean+sd (gaussian distribution)
    '''
    
    # mean+sd threshold 
    motion_mean = np.mean(motion_signal)
    motion_sd = np.std(motion_signal)

    threshold_motion_mean_sd = motion_mean + motion_sd

    threshold_motion_li = filters.threshold_li(motion_signal)

    threshold_motion_otsu = filters.threshold_otsu(motion_signal)

    if plot:
        plt.figure(figsize=(5, 5), dpi=300)
        plt.title(f'{title}')
        plt.hist(motion_signal, bins=70, alpha=0.9)

        # mark threshold lines
        plt.axvline(x=threshold_motion_mean_sd, color='red', label='mean + sd', linestyle='--')
        plt.axvline(x=threshold_motion_otsu, color='salmon', label='Otsu', linestyle='--')
        plt.axvline(x=threshold_motion_li, color='darkred', label='Li', linestyle='--')

        plt.legend()
        plt.savefig(save_dir / f'{title}.png')
        plt.show()

    # Return 3 thresholds for binary state detection
    return threshold_motion_mean_sd, threshold_motion_li, threshold_motion_otsu

def compute_thresholds_for_twitch(motion_signal, save_dir=None, plot=True):
    """
    Compute thresholds tailored for twitch detection on rest-only distribution.
    Returns five values: mean+3*sd, Li, Otsu, MAD-based, 95th percentile.
    """
    motion_signal = np.asarray(motion_signal)
    motion_mean = np.mean(motion_signal)
    motion_sd = np.std(motion_signal)

    threshold_motion_mean_sd = motion_mean + 3*motion_sd  # stricter for twitches
    threshold_motion_li = filters.threshold_li(motion_signal)
    threshold_motion_otsu = filters.threshold_otsu(motion_signal)
    threshold_95 = np.percentile(motion_signal, 95)

    # Local import to avoid hard dependency at module import time
    try:
        from statsmodels.robust import mad
        threshold_mad = np.median(motion_signal) + 3 * mad(motion_signal)
    except Exception:
        threshold_mad = None

    if plot:
        plt.figure(figsize=(5, 5), dpi=300)
        plt.title('stat thresholds on rest (twitch)')
        plt.hist(motion_signal, bins=70, alpha=0.9)
        # mark threshold lines
        plt.axvline(x=threshold_motion_mean_sd, color='red', label='mean + 3*sd', linestyle='--')
        plt.axvline(x=threshold_motion_otsu, color='salmon', label='Otsu', linestyle='--')
        plt.axvline(x=threshold_motion_li, color='darkred', label='Li', linestyle='--')
        if threshold_mad is not None:
            plt.axvline(x=threshold_mad, color='pink', label='MAD', linestyle='--')
        plt.axvline(x=threshold_95, color='purple', label='95th percentile', linestyle='--')
        plt.legend()
        if save_dir is not None:
            plt.savefig(Path(save_dir) / 'binary_state_thresholds_twitch.png')
        plt.show()

    return threshold_motion_mean_sd, threshold_motion_li, threshold_motion_otsu, threshold_mad, threshold_95

# --- Binary state detection (active vs rest) ---
# Compute candidate thresholds on smoothed motion energy
threshold_motion_mean_sd, threshold_motion_li, threshold_motion_otsu = \
    compute_thresholds_for_bin_state_detection(smoothed_motion_energy, 
                                               title='stat_thresholds_for_bin_state_detection', 
                                               save_dir=save_dir_videography, plot=True)

# Select binary threshold based on user choice
if binary_method == 'otsu':
    binary_threshold = threshold_motion_otsu
elif binary_method == 'li':
    binary_threshold = threshold_motion_li
elif binary_method == 'mean_sd':
    binary_threshold = threshold_motion_mean_sd
else:
    binary_threshold = threshold_motion_otsu

def binarise_motion(motion_signal, binary_threshold, min_duration, min_inactive_gap=9, bimodality=False):
    """Binarise motion signal into active (1) vs rest (0) using threshold and min segment duration."""
    bin_motion_signal = np.zeros(len(motion_signal), dtype=int)

    above_thresh = motion_signal > binary_threshold
    labeled_array, n_features = label(above_thresh)

    has_active_motion = False
    for i in range(1, n_features + 1):
        segment = np.where(labeled_array == i)[0]
        if len(segment) > min_duration:
            has_active_motion = True

    if has_active_motion or bimodality:
        for i in range(1, n_features + 1):
            segment = np.where(labeled_array == i)[0]
            if len(segment) > min_duration:
                bin_motion_signal[segment] = 1

    inds_active_state = np.where(bin_motion_signal == 1)[0]
    inds_rest_state = np.where(bin_motion_signal == 0)[0]

    return has_active_motion, bin_motion_signal, inds_active_state, inds_rest_state

# Run binarisation on smoothed motion energy
has_active_motion, bin_motion_energy, inds_active_state, inds_rest_state = \
    binarise_motion(smoothed_motion_energy, binary_threshold=binary_threshold, 
                    min_duration=active_motion_duration_min, min_inactive_gap=9, 
                    bimodality=bimodality)

# Onsets/offsets for active motion
active_motion_onsets = get_onsets(bin_motion_energy)
active_motion_offsets = get_offsets(bin_motion_energy)

# --- Twitch candidate selection on rest periods ---
rest_inds = np.where(bin_motion_energy == 0)[0]
rest_period = motion_energy[rest_inds] if len(rest_inds) > 0 else np.array([])

# Compute twitch thresholds on rest-only distribution
if len(rest_period) > 0:
    twitch_threshold_motion_mean_sd, twitch_threshold_motion_li, twitch_threshold_motion_otsu, threshold_mad, threshold_95 = \
        compute_thresholds_for_twitch(rest_period, save_dir=save_dir_videography, plot=True)
else:
    twitch_threshold_motion_mean_sd = twitch_threshold_motion_li = twitch_threshold_motion_otsu = None
    threshold_mad = threshold_95 = None

# Choose twitch threshold from user method or heuristic default
if twitch_method == 'li' and twitch_threshold_motion_li is not None:
    twitch_threshold = twitch_threshold_motion_li
elif twitch_method == 'mad' and threshold_mad is not None:
    twitch_threshold = threshold_mad
elif twitch_method == 'percentile_95' and threshold_95 is not None:
    twitch_threshold = threshold_95
elif twitch_method == 'mean_3sd' and twitch_threshold_motion_mean_sd is not None:
    twitch_threshold = twitch_threshold_motion_mean_sd
elif twitch_method == 'otsu' and twitch_threshold_motion_otsu is not None:
    twitch_threshold = twitch_threshold_motion_otsu
else:
    # Fallback heuristic
    if len(rest_period) > 0 and (bimodality is True):
        twitch_threshold = twitch_threshold_motion_li if twitch_threshold_motion_li is not None else (
            twitch_threshold_motion_otsu if twitch_threshold_motion_otsu is not None else threshold_mad)
    else:
        twitch_threshold = threshold_mad if threshold_mad is not None else twitch_threshold_motion_otsu

# Indices of unfiltered twitch candidates (rest frames above threshold)
if len(rest_period) > 0 and twitch_threshold is not None:
    inds_mask = rest_period > twitch_threshold
    inds_twitches_unfiltered = rest_inds[inds_mask]
else:
    inds_twitches_unfiltered = np.array([], dtype=int)

def filter_twitches_by_awake_proximity(inds_twitches, inds_active_state, min_distance=None):

    inds_active_array = inds_active_state[0] if isinstance(inds_active_state, tuple) else inds_active_state

    filtered_twitches = []

    if len(inds_active_array) == 0:
        return np.array(inds_twitches)
        
    for idx_twitch in inds_twitches:
        distance = np.abs(inds_active_array - idx_twitch)
        if np.min(distance) > min_distance:
            filtered_twitches.append(idx_twitch)

    filtered_twitches = np.array(filtered_twitches)

    return filtered_twitches

inds_twitches_filtered_step1 = filter_twitches_by_awake_proximity(inds_twitches_unfiltered, inds_active_state, min_distance=twitch_min_distance_from_active) 

inds_twitches_segments = find_sequential_groups(inds_twitches_filtered_step1)

if len(inds_active_state) > 1: 
    active_motion_segments = find_sequential_groups(inds_active_state)
else:
    active_motion_segments = []  
    
resting_motion_segments = find_sequential_groups(inds_rest_state)

def filter_segments_by_duration(segments, duration_threshold):
    return [twitches for twitches in segments if len(twitches) <= duration_threshold]

inds_twitches_filtered_step2 = filter_segments_by_duration(inds_twitches_segments, twitch_duration_max)

def binarise_twitch(motion_energy, twitch_segments):

    bin_twitch = np.zeros(len(motion_energy))

    flat_inds = [idx for segment in twitch_segments for idx in segment] 
    # flatten the nested list
    flat_inds = [idx for segment in twitch_segments for idx in segment] # filtered corrected twitches 

    print("Flattened Indices:", flat_inds)

    # Assign 1 to twitch inds
    for idx in flat_inds:
        if idx < len(motion_energy):  # Ensure index is within bounds
            bin_twitch[idx] = 1
            print(f"Assigning 1 at column {idx}")
        else:
            print(f"Skipping out-of-bounds index: {idx}")
    return bin_twitch

bin_twitch_unfilt = binarise_twitch(motion_energy, inds_twitches_filtered_step2)
# Keep an alias for downstream classification that expects the unfiltered twitch binary
bin_putative_twitch = bin_twitch_unfilt

# 
# # Define onset and offset of twitch

active_onsets = get_onsets(bin_motion_energy)
active_offsets = get_offsets(bin_motion_energy)
twitch_onsets = get_onsets(bin_twitch_unfilt)
twitch_offsets = get_offsets(bin_twitch_unfilt)

# # Plot interval between twitches (if burst - remove ?)

# iti.min() #/framerate # 2*1/framerate = 0.66*2= 0.13 (133 ms)

iti = np.diff(twitch_onsets/framerate)

plt.figure(figsize=(5,5))
plt.hist(iti.flatten(), bins=200)
plt.xlabel('sec')
plt.ylabel('twitches')
plt.show()

# most twithces occur in burst fashion 

# # Filter twitches based on inter-twitch-interval
# 
# (aka discard bursts, keep sparse 'trusted' twitches)

inds_twitches_filtered_final = filter_bursty_twitches(inds_twitches_filtered_step2, min_frame_gap=1) # 1 frame is ~330 ms

print(inds_twitches_filtered_final)

# # Final binary twitch signal -after all filters-

bin_twitch = binarise_twitch(motion_energy, inds_twitches_filtered_final) #expects class segments

# # Get final onsets (post sparsness filtering)

twitch_onsets = get_onsets(bin_twitch)
twitch_offsets = get_offsets(bin_twitch)

# # Plot twitches detected before and after sparse filtering 

fig, axs = plt.subplots(2,1, figsize=(20,7), dpi=300)
axs[0].plot(bin_twitch_unfilt, linewidth=0.8, c='red')
axs[0].plot(bin_twitch, alpha=0.8, c='green')
axs[0].plot(motion_energy, alpha=0.7)

axs[1].plot(bin_twitch_unfilt[:1000],c='red', alpha=0.8, label='before sparse filter')
axs[1].plot(bin_twitch[:1000], alpha=0.8, c='green', label = 'after sparse filter')
axs[1].plot(motion_energy[:1000], alpha=0.7)
plt.legend(loc = 'upper right')
plt.show()

 

# # Plot filtered twitches detected on motion energy

def plot_detected_twitches(
    motion_energy,
    smoothed_motion_energy,
    bin_motion_energy,
    threshold_twitches,
    threshold_motion_energy,
    inds_twitches,
    frame_ticks,
    second_ticks,
    save_dir=None
):
    """
    Plot the motion energy and smoothed motion energy with twitch detection.

    Parameters:
    - motion_energy: Raw motion energy signal.
    - smoothed motion energy: Smoothed motion energy signal.
    - threshold_twitches: Threshold for twitch detection.
    - threshold_motion_energy: Threshold for binary motion energy.
    - inds_twitches: Indices where twitch segments start (onsets).
    - frame_ticks: Frame indices for x-axis.
    - second_ticks: Time ticks in seconds for x-axis.
    - save_dir_videography: Optional directory to save the figure.
    """
    
    fig, axs = plt.subplots(2, 1, figsize=(15, 7), dpi=150)

    # Plot raw and smoothed motion energy
    axs[0].plot(motion_energy, color='orange', linewidth=2, label='mot_en')
    axs[0].plot(smoothed_motion_energy, color='blue', linewidth=1.5, label='mot_en smoothed')
    axs[0].axhline(y=threshold_twitches, color='red', linestyle='--', label='Twitch threshold')
    axs[0].set_xticks(ticks=frame_ticks)
    axs[0].set_xticklabels(second_ticks, fontsize=12)
    axs[0].legend(loc='upper right', fontsize=10)
    axs[0].set_ylabel('mot_en', fontsize=15)
    axs[0].set_title('Twitch detection', fontsize=18)

    trio_motion_energy = bin_motion_energy.copy()
    trio_motion_energy[inds_twitches] = -1  # mark twitch segments with -1

    # Plot raw motion energy and binary motion energy (including twitch detection)
    axs[1].plot(motion_energy, color='blue', linewidth=1, label='mot_en')
    axs[1].plot(trio_motion_energy, color='darkorange', linewidth=1, label='3 states on mot_en')
