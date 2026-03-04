import cv2
import tifffile as tifffile
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import label, binary_closing
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.filters import threshold_li, threshold_otsu
from skimage import filters
from scipy.ndimage import label, binary_closing


def smooth_with_gaussian(var, sigma=None):
    smoothed_var = gaussian_filter1d(var, sigma=sigma)

    return smoothed_var

def compute_motion_energy(session_path, movie_path=None, xrange=None, yrange=None, save_path=None):
    """
    Compute motion energy from a multi-frame TIFF movie of mouse movement.

    Parameters:
    - movie_path: Path to the multi-frame TIFF file (None, unless ititial run)
    - xrange: Range of x-values to crop the image (optional)
    - yrange: Range of y-values to crop the image (optional)
    - save_path: Path to save the motion energy result (optional)
    """
    
    #check if motion energy has already been computed, if that's the case load it

    motion_energy_path = Path(session_path) / "camera_processed" / "motion_energy.npy"
    if os.path.exists(motion_energy_path):

        print(f'Motion energy already computed. Loading from {motion_energy_path}')
        motion_energy = np.load(motion_energy_path)
        return motion_energy

    if not movie_path:

        raise ValueError('Please provide the tiff for the initial run') 
    
    # Load the TIFF movie (multi-frame TIFF)
    movie = tifffile.imread(movie_path)
    num_frames, height, width = movie.shape

    print(f'Loaded movie with {num_frames} frames, height={height}, width={width}')

    # Initialize motion energy array
    motion_energy = np.zeros(num_frames)
    img_prev = movie[0]

    # Iterate over the frames and compute motion energy
    for i in range(1, num_frames):
        img = movie[i]

        # Compute motion energy as squared differences between consecutive frames
        diff = img - img_prev
        squared_diff = diff ** 2
        motion_energy[i] = np.sum(squared_diff)

        # Update img_prev for the next iteration
        img_prev = img

        # Print progress every 1000 frames
        if i % 1000 == 0:
            print(f'Done computing for {i}/{num_frames} frames')
    
    # Normalize motion energy
    motion_energy = motion_energy[1:]  # Skip the first frame (no previous frame to compare)
    motion_energy /= np.max(motion_energy)
    
    return motion_energy

def compute_thresholds_for_bin_state_detection(motion_signal, save_dir=None, plot=True):
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
        plt.title('stat thresholds on mot_en')
        plt.hist(motion_signal, bins=70, alpha=0.9)

        # mark threshold lines
        plt.axvline(x=threshold_motion_mean_sd, color='red', label='mean + sd', linestyle='--')
        plt.axvline(x=threshold_motion_otsu, color='salmon', label='Otsu', linestyle='--')
        plt.axvline(x=threshold_motion_li, color='darkred', label='Li', linestyle='--')

        plt.legend()
        plt.savefig(save_dir + 'binary_state_thresholds.png')
        plt.show()

    return threshold_motion_mean_sd, threshold_motion_li, threshold_motion_otsu
    
def binarise_motion(motion_signal, binary_threshold, min_duration):
    '''
    Binarises motion signal (eg mot_en) into 0s (rest) and 1s (active)
    input params:
    motion_signal: motion energy or other
    binary_threshold: sts threshold (li, otsu or mean_sd for state detection)
    min_di=uration: min_duration threshold to be detected as awake/active motion
    
    returns:
    bin_motion_signal: bin array of 0s and 1s 
    inds_active_state: indices of frames whens state is active
    inds_rest_state: indices of frames when state is rest/inactive 

    '''
    
    bin_motion_signal = np.zeros(len(motion_signal), dtype=int)

    # get boolean array 
    
    all_active_motions_detected = motion_signal > binary_threshold 

    # labels continuous segments that pass the threshold 

    labeled_array, n_features = label(all_active_motions_detected) #how many active motions where found 
    
    for i in range(1, n_features+1):
        segment = np.where(labeled_array == i)[0]
        if len(segment) > min_duration:
            bin_motion_signal[segment] = 1

    inds_active_state = np.where(bin_motion_signal ==1)
    inds_rest_state = np.where(bin_motion_signal ==0)
    
    return bin_motion_signal, inds_active_state, inds_rest_state

def classify_active_motion_segments(bin_motion_signal, motion_signal, short_threshold, long_threshold):
    """
    Classify active/awake motion segments into short (1–3 sec), long (>3 sec), and too short (<1 sec, those are exluded in awake motion detection and set to 0).

    Parameters:
    - bin_motion_signal: 1D array of HMM or thresholded motion states (1 for active, 0 for inactive)
    - motion_signal: 1D array of motion energy values
    - short_threshold: in frames, minimum duration for short active motions (e.g., 3 or 1s at 3Hz)
    - long_threshold: in frames, minimum duration for long active motions (e.g., 9 or 3s at 3Hz)

    Returns:
    - bin_short_active_motion: binary array marking short motions (1-3s)
    - bin_long_active_motion: binary array marking long motions (>3s)
    - bin_too_short_active_motion: binary array marking short blips (<1s)
    - and correspondings inds of long, short or too short (excluded) motions
    """
    labeled_array, num_features = label(bin_motion_signal == 1)

    bin_short_active_motion = np.zeros(len(motion_signal), dtype=int)
    bin_long_active_motion = np.zeros(len(motion_signal), dtype=int)
    bin_too_short_active_motion = np.zeros(len(motion_signal), dtype=int)

    for i in range(1, num_features + 1):
        segment = np.where(labeled_array == i)[0]

        if len(segment) > long_threshold:
            bin_long_active_motion[segment] = 1
        elif len(segment) > short_threshold:
            bin_short_active_motion[segment] = 1
        elif len(segment) > 0: # but shorter than short threshold 
            bin_too_short_active_motion[segment] = 0 # mark as inactive 

        long_active_motion_inds = np.where(bin_long_active_motion==1)[0]
        short_active_motion_inds = np.where(bin_short_active_motion==1)[0]
        too_short_active_motion_inds = np.where(bin_too_short_active_motion==1)[0]

            
    return bin_short_active_motion, short_active_motion_inds, bin_long_active_motion, long_active_motion_inds, bin_too_short_active_motion, too_short_active_motion_inds


'''
Define hyperparameters for twitch detection. Added a minimum distance (in frames) required 
for the twitch to be detected (this is to avoid inter-active-sleep periods artifacts, if the twitch happens too close 
too an active motion (precedes or follows) exclude it, becs we cannot be sure the mouse was asleep already or in between.
We only keep best, robust twitches 
'''
def filter_twitches_by_awake_proximity(inds_twitches, inds_active_state, min_distance=None):
    filtered_twitches = []

    for idx_twitch in inds_twitches:
        distance = np.abs(inds_active_state - idx_twitch)
        if np.min(distance) > min_distance:
            filtered_twitches.append(idx_twitch)

    filtered_twitches = np.array(filtered_twitches)

    return filtered_twitches

def filter_twitches_only_post_active_mot(inds_twitches, inds_active_state, min_distance=None):
    filtered_twitches = []

    for idx_twitch in inds_twitches:
        distances = inds_active_state - idx_twitch  # Compute relative distances
        closest_active_motion = np.min(distances[distances >= 0], initial=np.inf)  # Only consider future active motions

        # Exclude only if the twitch is after and within min_distance
        if closest_active_motion > min_distance:
            filtered_twitches.append(idx_twitch)

    return np.array(filtered_twitches)

def remove_twitch_bursts(twitch_binary, framerate, min_interval=1):
    """
    Removes all twitches that occur within `min_interval` seconds of another twitch.
    Both twitches in a burst/close proximimity are excluded.
    
    Parameters:
        twitch_binary (np.ndarray): Binary vector of twitch detections (1 = twitch, 0 = no twitch)
        fs (float): framerate (Hz)
        min_interval (float): minimum interval required between twitches for those to be detected (in sec)
    
    Returns:
        np.ndarray: filtered binary twitch array 
    """
    twitch_indices = np.where(twitch_binary == 1)[0]
    to_remove = set()
    
    # Compare each twitch to the next one
    for i in range(len(twitch_indices) - 1):
        curr_idx = twitch_indices[i]
        next_idx = twitch_indices[i + 1]
        interval = (next_idx - curr_idx) / framerate # in sec 
        
        if interval < min_interval:
            to_remove.add(curr_idx)
            to_remove.add(next_idx)
    
    filtered_twitches = np.copy(twitch_binary)
    filtered_twitches[list(to_remove)] = 0
    return filtered_twitches 

def filter_bursty_twitches(inds_twitches, min_frame_gap=3):
    """
    Keeps only the first twitch in bursts occurring within `min_frame_gap` frames.
    
    Parameters:
        inds_twitches (list of list of int): Each inner list contains frame indices for a twitch.
        min_frame_gap (int): Minimum allowed gap between two twitch events (in frames).
        
    Returns:
        list of list of int: Filtered list of twitch events (same structure).
    """
    if not inds_twitches:
        return []

    filtered = [inds_twitches[0]]
    last_kept = inds_twitches[0][0]  # take the first index of the first twitch

    for twitch in inds_twitches[1:]:
        current_onset = twitch[0]  # first index of current twitch
        if current_onset - last_kept >= min_frame_gap:
            filtered.append(twitch)
            last_kept = current_onset

    return filtered

def binarise_twitch(motion_energy, twitch_segments):

    # Initialize bin_twitch   
    bin_twitch = np.zeros(len(motion_energy))

    # flatten the nested list
    flat_inds = [idx for segment in twitch_segments for idx in segment] # filtered corrected twitch segments 

    print("Flattened Indices:", flat_inds)

    # Assign 1 to twitch inds
    for idx in flat_inds:
        if idx < len(motion_energy):  # Ensure index is within bounds
            bin_twitch[idx] = 1
            print(f"Assigning 1 at column {idx}")
        else:
            print(f"Skipping out-of-bounds index: {idx}")
    return bin_twitch


# Define a helper function to get discrete active state segments
''' modified to handle empty arrays
'''
def get_active_segments(active_indices):
    segments = []
    if active_indices.size == 0:
        return segments  # Return an empty list if there are no active indices
    start = active_indices[0]
    for i in range(1, len(active_indices)):
        # Check if the current index is not consecutive with the previous
        if active_indices[i] != active_indices[i - 1] + 1:
            end = active_indices[i - 1]
            segments.append((start, end))
            start = active_indices[i]
    segments.append((start, active_indices[-1]))
    return segments

def find_sequential_groups(arr):
    groups = []
    current_group = [arr[0]]
    
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1] + 1:
            current_group.append(arr[i])
        else:
            groups.append(current_group)
            current_group = [arr[i]]
    groups.append(current_group)
    
    return groups

def get_onsets(bin_motion):
    onsets = np.where((bin_motion[1:] == 1) & (bin_motion[:-1] == 0))[0] + 1 #[0] specific to stcuture of array
    return onsets

def get_offsets(bin_motion):
    offsets = np.where((bin_motion[1:] == 0) & (bin_motion[:-1] == 1))[0]  #[0] here 0->1 we want the 0 as offset so not +1

# filter twitches based on duration 
def filter_segments_by_duration(segments, duration_threshold):
# Filter out groups based on the duration threshold
    return [twitches for twitches in segments if len(twitches) <= duration_threshold]


'''
Input a motion or batches of motiona nd output the average length of the motion in seconds 

'''

def get_length_of_motion(concatenated_motion, frame_rate): #can even be a xhole motion or concatenated motion segments
    if len(concatenated_motion) == 1: 
        mean_length_motion_sec = len(concatenated_motion[0]) / frame_rate
    
    mean_length_motion = np.mean([len(segment) for segment in concatenated_motion])

    mean_length_motion_sec = mean_length_motion / frame_rate

    return mean_length_motion_sec

# Define a helper function to get discrete active state segments
def get_active_segments(active_indices):
    segments = []
    start = active_indices[0]
    for i in range(1, len(active_indices)):
        # Check if the current index is not consecutive with the previous
        if active_indices[i] != active_indices[i - 1] + 1:
            segments.append((start, active_indices[i - 1]))
            start = active_indices[i]
    segments.append((start, active_indices[-1]))  # Add the last segment
    return segments

# compute correlation (coupling) between behaviour and pcs (smoothing a bit before computing corr)

def compute_corrs(behaviour, pcs):
    beh_pcs_coupling = []
    for i in range(pcs.shape[0]): #no. of pcs computed, time 
        corr = np.corrcoef(gaussian_filter1d(pcs[i,:], sigma=3), gaussian_filter1d(behaviour, sigma=3))[0,1]
        beh_pcs_coupling.append(corr)
    return beh_pcs_coupling

def get_onsets(bin_motion):
    onsets = np.where((bin_motion[1:] == 1) & (bin_motion[:-1] == 0))[0] + 1 #[0] specific to stcuture of array
    if bin_motion[0] == 1:  
        onsets = np.insert(onsets, 0, 0) # solves issue with twitch detected at first frames  
    return onsets
    
def get_offsets(bin_motion):
    offsets = np.where((bin_motion[1:] == 0) & (bin_motion[:-1] == 1))[0] + 1 #[0] specific to stcuture of array
    return offsets  

# add me 
def load_s2p(session_path):
    iscell = np.load(os.path.join(session_path, "suite2p/plane0/iscell.npy"))[:,0].astype(bool) #iscell.shape is (2203,)
    fluo = np.load(os.path.join(session_path, "suite2p/plane0/F.npy")) #fluo.shape is (1925, 36000)
    
    # handling zeros rows in fluo (also need to modify iscell)
    nan_rows = np.unique(np.where(fluo == 0)[0]) # indices of rows (ROIs) that include zeros
    print(f'Found {len(nan_rows)} (/{len(iscell)}) ROIs including zeros')
    iscell[nan_rows] = False # new iscell with removed NaN rows
    
    fluo = fluo[iscell] # reindexing with removed NaN rows
    fluo.min()
    spks = np.load(os.path.join(session_path, "suite2p/plane0/spks.npy"))[iscell] #taking the spks of only the chosen iscell
    stat = np.load(os.path.join(session_path, "suite2p/plane0/stat.npy"), allow_pickle=True)[iscell] #same for stats, len(stat) = iscell good rois 
    ops = np.load(os.path.join(session_path, "suite2p/plane0/ops.npy"), allow_pickle=True).item()
    xpix = [stat[i]["xpix"] for i in range(stat.shape[0])] #contains iscell lists that contain the pixels that contour the ROI (to check eg the number of pixels of ROI 100 you can: len(xpix[100]))
    ypix = [stat[i]["ypix"] for i in range(stat.shape[0])]
    nframes = fluo.shape[1]
    avg_im = ops["meanImg"]
    max_proj = ops['max_proj']
    
    framerate= ops['fs']
    
    tstamps = np.arange(0, nframes/framerate, 1/framerate)
    n_neurons = spks.shape[0] #add to dictionary
       
    # colors = ['y', 'g', 'r', 'c', 'm', 'royalblue', 'orange']
    
    print(f'Found {sum(iscell)} (/{len(iscell)}) iscell ROIs')
    print(f'Shape: {fluo.shape}')

    return fluo, spks, stat, xpix, ypix, nframes, avg_im, max_proj, framerate, tstamps, n_neurons, max_proj 


# for jm adapatation 
import pandas as pd

def read_params(cell_nmf_params, ds):
    """
    Reads imaging and session metadata parameters from Excel for a given dataset.

    Parameters
    ----------
    cell_nmf_params : str
        Path to Excel file containing parameters for multiple datasets.
    ds : str
        Dataset name (e.g., 'sz85_2024-04-25_b' or 'jm042/2024-08-27_a').

    Returns
    -------
    params_table : pd.DataFrame
        Table of session parameters.
    age, day_imgd, FOV, framerate, duration, resolution, n_cells, n_components, time_per_frame
    """

    # --- Load Excel file ---
    df_params = pd.read_excel(cell_nmf_params)
    pd.options.display.float_format = '{:.3f}'.format

    # --- Normalize dataset name ---
    df_params = df_params.dropna(how='all')

    # Optionally, drop rows where 'ds' is NaN
    df_params = df_params.dropna(subset=['ds']).reset_index(drop=True)

    ds_norm = ds.replace('/', '_')  # replace / with _
    ds_variants = [ds, ds.replace('-', '_'), ds.replace('_', '-'), ds_norm]

    # --- Find matching dataset in Excel ---
    match = df_params[df_params['ds'].isin(ds_variants)]
    if match.empty:
        raise ValueError(
            f"Dataset '{ds}' not found in {cell_nmf_params}. "
            f"Available datasets: {df_params['ds'].unique()}"
        )

    params = match.iloc[0]

    # --- Extract parameters ---
    genotype = params.get('genotype')
    days_imaged = params.get('days_imaged')
    tracked = params.get('tracked')
    blur_std = params.get('blur_std')
    downs_fact = params.get('downs_fact', None)
    x_axis = params.get('x_axis')
    y_axis = params.get('y_axis')
    original_FOV = params.get('original_FOV')
    framerate = params.get('framerate')
    duration = params.get('duration')
    resolution = params.get('resolution')
    n_components_elbow = params.get('n_components_elbow')
    n_components_min = params.get('n_components_min')
    age = params.get('age')
    weight_g = params.get('weight_g')
    s2p = params.get('s2p')
    cells = params.get('cells', None)
    Behaviour = params.get('Behaviour', None)
    mousecraft = params.get('mousecraft', None)
    SLEAP = params.get('SLEAP', None)
    
    # --- Compute dependent value ---
    if 'time_per_frame' in params and not pd.isna(params['time_per_frame']):
        time_per_frame = params['time_per_frame']
    elif framerate:
        time_per_frame = 1 / framerate
    else:
        time_per_frame = None

    # --- Build clean DataFrame for display ---
    params_table = pd.DataFrame({
        'Parameter': [
            'genotype', 'days_imaged', 'tracked', 'blur_std', 'downs_fact',
            'x_axis', 'y_axis', 'original_FOV', 'framerate', 'duration',
            'resolution', 'n_components_elbow', 'n_components_min', 'age',
            'weight_g', 's2p', 'cells', 'Behaviour', 'mousecraft', 'SLEAP'            
        ],
        'Value': [
            genotype, days_imaged, tracked, blur_std, downs_fact,
            x_axis, y_axis, original_FOV, framerate, duration,
            resolution, n_components_elbow, n_components_min, age,
            weight_g, s2p, cells, Behaviour, mousecraft, SLEAP
        ]
    })

    # --- Display table nicely ---
    if 'style_table' in globals():
        styled_params_table = params_table.style.pipe(style_table).set_properties(**{'text-align': 'center'})
        display(styled_params_table)
    else:
        display(params_table)

    # --- Return both table and individual params ---
    return (
        params_table, genotype, days_imaged, tracked, blur_std, downs_fact,
        x_axis, y_axis, original_FOV, framerate, duration, resolution,
        n_components_elbow, n_components_min, age, weight_g, s2p, cells,
        Behaviour, mousecraft, SLEAP
    )

## add me

def downsample_dict(annotations_dict, original_framerate, target_framerate=3):
    """
    Downsample frame-based motion/twitch annotations by aligning raw frames to the downsampled framerate.
    Creates '_avg' keys for onsets/offsets in downsampled frames.
    
    Args:
        annotations_dict (dict): Original annotations dictionary (with pandas Series or arrays).
        original_framerate (float): Original framerate in Hz.
        target_framerate (float): Target framerate in Hz. Default=3.
        
    Returns:
        dict: Updated dictionary with '_avg' keys containing downsampled frame indices.
    """
    ds_factor = original_framerate / target_framerate
    annotations_ds = annotations_dict.copy()
    
    onset_offset_keys = ['active_mot_onsets', 'active_mot_offsets', 'twitch_onsets', 'twitch_offsets']
    
    for key in onset_offset_keys:
        if key in annotations_dict:
            frames = annotations_dict[key].values if hasattr(annotations_dict[key], 'values') else np.array(annotations_dict[key])
            # Map to downsampled frames
            frames_ds = np.floor(frames / ds_factor).astype(int)
            # Create '_avg' key
            annotations_ds[key.replace('active_', 'active_') + '_avg'] = frames_ds
    
    return annotations_ds

def load_mousecraft_validations(params_path, ds=None, data_root=None, mode="cells"): # or pixels depeds on ds
    '''
    either loads for one ds if specified or all
    
    '''
    
    # Load metadata
    params_df = pd.read_excel(params_path)

    if ds is not None:
    # Filter to just the given dataset
        ds_metadata = ds.replace("/", "_").strip() 
        mc_df = params_df[params_df['ds'] == ds_metadata]
    
        if mc_df.empty:
            raise ValueError(f"Dataset {ds_metadata} not found in params file.")
    else:
        # Batch mode: all Mousecraft=1 datasets
        mc_df = params_df[params_df['mousecraft'] == 1]

    results = {}

    for _, row in mc_df.iterrows():
        ds = row['ds']  # e.g. 'sz089_2024-06-06_a_cell_control'
        framerate = row['framerate']
        duration = row['duration']  # in seconds

        # Parse mouse_id and session
        parts = ds.split('_')
        mouse_id = parts[0]  # 'sz089'
        session = parts[1] + "_" + parts[2]  # '2024-06-06_a'

        # Build path: data_proc/cells/sz089/2024-06-06_a/mousecraft/validation_MF_final.xls
        validation_path = Path(data_root) / mode / mouse_id / session / 'mousecraft_output' / 'validation_HF_final.xlsx'

        if not validation_path.exists():
            print(f"⚠️ Missing Mousecraft file for {ds} at {validation_path}")
            continue

        # Load Mousecraft validation
        val_df = pd.read_excel(validation_path)

        # Filter only accepted / edited / manually added
        valid_events = val_df[val_df['status'].isin(['accepted', 'edited', 'manually added'])]
        TP_events = val_df[val_df['status'].isin(['accepted', 'edited'])]*100
        FP_events = val_df[val_df['status'].isin(['rejected'])]*100
        mousecraft_idx = TP_events - FP_events / TP_events + FP_events # +-1 
        
        # Separate twitches and active motions
        twitches = valid_events[valid_events['event_type'] == 'twitch']
        active = valid_events[valid_events['event_type'] == 'active']

        # Count motions 
        twitch_count = len(twitches)
        active_motion_count = len(active)

        # compute duration of twitches (how much time from Rest is attributed to twitching ?)
        twitch_onsets = twitches['onset'] #val onsets 
        twitch_offsets = twitches['offset'] #val offets
        twitch_durations = ((twitches['offset'] - twitches['onset'])/framerate).tolist()
        mean_twitch_duration = np.mean(twitch_durations)
        
        # compute ITI (inter-twitch interval)
        inter_twitch_intrvl = np.diff(twitches['onset'])/framerate
        mean_inter_twitch_intrvl = np.mean(inter_twitch_intrvl)

        # compute duration of active motions 

        active_mot_onsets = active['onset'] #val onsets 
        active_mot_offsets = active['offset'] #val offets
        
        if len(active) == 0: # in case there are no active motions 
            active_motion_durations = 0
            mean_active_motion_duration = np.nan
            inter_active_motion_intrvl = []
            mean_inter_active_motion_intrvl = np.nan
        
        else:
            active_motion_durations = ((active['offset'] - active['onset'])/framerate).tolist()
            mean_active_motion_duration = np.mean(active_motion_durations)
            # active ITI 
            inter_active_motion_intrvl = np.diff(active['onset'])/framerate
            mean_inter_active_motion_intrvl = np.mean(inter_active_motion_intrvl)

        # Compute time in active motion (awake) 
        active_motion_time_sec = 0
        for _, e in active.iterrows():
            onset = e['onset']
            offset = e['offset']
            active_motion_time_sec += (offset - onset) / framerate
        
        total_time_sec = duration
        
        #compute time in rest (sleep)
        rest_time_sec = total_time_sec - active_motion_time_sec
        active_time_percentage = (active_motion_time_sec/total_time_sec)*100
        rest_time_percentage = (rest_time_sec/total_time_sec)*100
        twitch_frequency = twitch_count / rest_time_sec # in Hz 

        # Save all metrices we are interested in  
        results[ds] = {
            "twitch_count": twitch_count,
            'twitch_frequency': twitch_frequency,
            'twitch_durations': twitch_durations, 
            'mean_twitch_duration': mean_twitch_duration,
            'twitch_intervals': inter_twitch_intrvl,
            'mean_twitch_interval': mean_inter_twitch_intrvl,
            'twitch_onsets': twitch_onsets,
            'twitch_offsets': twitch_offsets,
            
            'active_motion_count': active_motion_count,
            'active_motion_durations': active_motion_durations,
            'mean_active_motion_duration': mean_active_motion_duration,
            'active_motion_intervals': inter_active_motion_intrvl,
            'mean_active_motion_interval': mean_inter_active_motion_intrvl,
            'active_mot_onsets': active_mot_onsets,
            'active_mot_offsets': active_mot_offsets, 

            # % in Awake - Rest (sleep)
            "active_motion_time_sec": active_motion_time_sec,
            "rest_time_sec": rest_time_sec,
            "active_time_percentage": active_time_percentage,
            "rest_time_percentage": rest_time_percentage, 
            "total_time_sec": total_time_sec,
            
            # mousecraft performance 
            "TP_events": TP_events,
            "FP_events": FP_events,
            "mousecraft_idx": mousecraft_idx
        }

    return pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': 'ds'}) # dataframe to merge with metadata xls 


def load_auto_annotations(session_path, framerate, duration, mode=None): # mode 'rescaled" back to orig time or else: downsampled (on avg time)
    motion_annotation_path = session_path / "motion_annotation_average"
    if mode == 'rescaled':
        df = pd.read_excel(motion_annotation_path / 'automatic_annotations_rescaled.xlsx')
    else: 
        df = pd.read_excel(motion_annotation_path / 'automatic_annotations.xlsx') # downsampled (as detected on avg motion trace)

    def clean_col(series):
        # Convert column to numeric and drop NaNs
        return pd.to_numeric(series, errors='coerce').dropna().values

    active_onsets = clean_col(df['active_onsets'])
    active_offsets = clean_col(df['active_offsets'])
    
    # in case an onset has no corresponding offset (eg a motion started at the end of recording)
    min_len = min(len(active_onsets), len(active_offsets))
    
    active_onsets = active_onsets[:min_len]
    active_offsets = active_offsets[:min_len]
       
    twitch_onsets = clean_col(df['twitch_onsets'])
    twitch_offsets = clean_col(df['twitch_offsets'])
    complex_onsets = clean_col(df['complex_onsets'])
    complex_offsets = clean_col(df['complex_offsets'])

    active_motion_count = len(active_onsets)   
    active_motion_durations = []
    mean_active_motion_duration = 0
    inter_active_motion_intrvl = []
    mean_inter_active_motion_intrvl = []
    
    if len(active_onsets)>0: 
        active_motion_durations = ((active_offsets - active_onsets) / framerate).tolist() if active_motion_count > 0 else []
        mean_active_motion_duration = np.mean(active_motion_durations) if active_motion_count > 0 else np.nan
        inter_active_motion_intrvl = np.diff(active_onsets) / framerate if active_motion_count > 1 else np.array([])
        mean_inter_active_motion_intrvl = np.mean(inter_active_motion_intrvl) if len(inter_active_motion_intrvl) > 0 else np.nan
        active_motion_time_sec = np.sum((active_offsets - active_onsets) / framerate) if active_motion_count > 0 else 0

    # Twitches
    twitch_count = len(twitch_onsets)
    twitch_durations = ((twitch_offsets - twitch_onsets) / framerate).tolist() if twitch_count > 0 else []
    mean_twitch_duration = np.mean(twitch_durations) if twitch_count > 0 else np.nan
    inter_twitch_intrvl = np.diff(twitch_onsets) / framerate if twitch_count > 1 else np.array([])
    mean_inter_twitch_intrvl = np.mean(inter_twitch_intrvl) if len(inter_twitch_intrvl) > 0 else np.nan

    rest_time_sec = duration - active_motion_time_sec
    active_time_percentage = (active_motion_time_sec/total_time_sec)*100
    rest_time_percentage = (rest_time_sec/total_time_sec)*100
    twitch_frequency = twitch_count / rest_time_sec if rest_time_sec > 0 else np.nan

    return {
        "active_motion_count": active_motion_count,
        "active_motion_durations": active_motion_durations,
        "mean_active_motion_duration": mean_active_motion_duration,
        "active_motion_intervals": inter_active_motion_intrvl.tolist(),
        "mean_active_motion_interval": mean_inter_active_motion_intrvl,
        "active_mot_onsets": active_onsets.tolist(),
        "active_mot_offsets": active_offsets.tolist(),

        "twitch_count": twitch_count,
        "twitch_durations": twitch_durations,
        "mean_twitch_duration": mean_twitch_duration,
        "twitch_intervals": inter_twitch_intrvl.tolist(),
        "mean_twitch_interval": mean_inter_twitch_intrvl,
        "twitch_onsets": twitch_onsets.tolist(),
        "twitch_offsets": twitch_offsets.tolist(),

        "complex_onsets": complex_onsets.tolist(),
        "complex_offsets": complex_offsets.tolist(),

        "active_motion_time_sec": active_motion_time_sec,
        "rest_time_sec": rest_time_sec,
        'active_time_percentage': active_time_percentage,
        'rest_time_percentage': rest_time_percentage,
        "total_time_sec": duration,
        "twitch_frequency": twitch_frequency
    }


def compute_motion_energy(session_path, movie_path=None, xrange=None, yrange=None, save_path=None):
    """
    Compute motion energy from a multi-frame TIFF movie of mouse movement.

    Parameters:
    - movie_path: Path to the multi-frame TIFF file (None, unless ititial run)
    - xrange: Range of x-values to crop the image (optional)
    - yrange: Range of y-values to crop the image (optional)
    - save_path: Path to save the motion energy result (optional)
    """
    
    #check if motion energy has already been computed, if that's the case load it

    motion_energy_path = Path(session_path) / "camera_processed" / "motion_energy.npy"
    if os.path.exists(motion_energy_path):

        print(f'Motion energy already computed. Loading from {motion_energy_path}')
        motion_energy = np.load(motion_energy_path)
        return motion_energy

    if not movie_path:

        raise ValueError('Please provide the tiff for the initial run') 
    
    # Load the TIFF movie (multi-frame TIFF)
    movie = tifffile.imread(movie_path)
    num_frames, height, width = movie.shape

    print(f'Loaded movie with {num_frames} frames, height={height}, width={width}')

    # Initialize motion energy array
    motion_energy = np.zeros(num_frames)
    img_prev = movie[0]

    # Iterate over the frames and compute motion energy
    for i in range(1, num_frames):
        img = movie[i]

        # Compute motion energy as squared differences between consecutive frames
        diff = img - img_prev
        squared_diff = diff ** 2
        motion_energy[i] = np.sum(squared_diff)

        # Update img_prev for the next iteration
        img_prev = img

        # Print progress every 1000 frames
        if i % 1000 == 0:
            print(f'Done computing for {i}/{num_frames} frames')
    
    # Normalize motion energy
    motion_energy = motion_energy[1:]  # Skip the first frame (no previous frame to compare)
    motion_energy /= np.max(motion_energy)
    
    return motion_energy

# locate, read and load HF Mousecraft outputs and save into variables 

def load_mousecraft_validations_all(params_path, data_root, ds=None, mode="cells"):
    """
    Scan all datasets under data_root/mode and load Mousecraft validation files if they exist.

    Returns a DataFrame with the results.
    """
    params_df = pd.read_excel(params_path)  # to read duration 
    
    # Handle different folder structures for cells vs pixels
    if mode == "cells":
        data_root = Path(data_root) / "cells"
    elif mode == "pixels":
        data_root = Path(data_root) / "pixels"
    else:
        data_root = Path(data_root) / mode

    results = {}
    
    # Get Mousecraft-enabled datasets from params_df
    if 'mousecraft' in params_df.columns:
        mousecraft_datasets = set(params_df[params_df['mousecraft'] == 1]['ds'])
        print(f"📊 Mousecraft-enabled datasets: {len(mousecraft_datasets)}")
    else:
        mousecraft_datasets = None
        print("⚠️ No 'mousecraft' column in params_df - processing all datasets")

    # Iterate over all mouse folders
    for mouse_folder in data_root.iterdir():
        if not mouse_folder.is_dir():
            continue

        # Iterate over all session folders
        for session_folder in mouse_folder.iterdir():
            if not session_folder.is_dir():
                continue

            # Build dataset name
            ds_name = f"{mouse_folder.name}_{session_folder.name}"
            
            # Skip if this dataset is not marked for Mousecraft processing
            if mousecraft_datasets and ds_name not in mousecraft_datasets:
                continue

            # Mousecraft validation path - handle different possible locations
            possible_validation_folders = [
                session_folder / "mousecraft_output",  # Standard location
                session_folder,  # Sometimes files are directly in session folder
                session_folder.parent / "mousecraft_output" / session_folder.name,  # Alternative structure
            ]
            
            valid_files = ["validation_HF_final.xlsx", "mousecraft_validated_labels_HF_final.xlsx"]
            
            validation_path = None
            for valid_folder in possible_validation_folders:
                for valid_file in valid_files:
                    candidate_path = valid_folder / valid_file
                    if candidate_path.exists():
                        validation_path = candidate_path
                        print(f'Found {valid_file} in {valid_folder}')  # MOVED BEFORE break
                        break  # Now this break happens after printing
                if validation_path:
                    break
                    
            if validation_path is None:
                print(f"⚠️ Missing Mousecraft file for {ds_name} at {session_folder}. Skipping.")
                continue

            print(f"✅ Found Mousecraft file for {ds_name}: {validation_path}")

            try:
                # Load Mousecraft validation
                val_df = pd.read_excel(validation_path)

                # Filter only accepted / edited / manually added
                valid_events = val_df[val_df['status'].isin(['accepted', 'edited', 'manually added'])]
                TP_events = val_df[val_df['status'].isin(['accepted', 'edited'])]
                FP_events = val_df[val_df['status'].isin(['rejected'])]
                TP_events_percentage =  len(TP_events)/len(TP_events) + len(FP_events)
                FP_events_percentage =  len(FP_events)/len(TP_events) + len(FP_events)
                performance_idx = (len(TP_events) - len(FP_events)) / (len(TP_events) + len(FP_events)) if (len(TP_events) + len(FP_events)) > 0 else np.nan
                
                # Twitches vs Active motions
                twitches = valid_events[valid_events['event_type'] == 'twitch']
                active = valid_events[valid_events['event_type'] == 'active']

                # ✅ Read total_time_sec (duration) from params_df
                total_time_sec = np.nan
                framerate = np.nan 

                if 'ds' in params_df.columns:
                    match = params_df.loc[params_df['ds'] == ds_name]
                    if not match.empty:
                        total_time_sec = match['duration'].iloc[0] if 'duration' in match.columns else np.nan
                        framerate = match['framerate'].iloc[0] if 'framerate' in match.columns else np.nan
                elif {'mouse', 'session'}.issubset(params_df.columns):
                    match = params_df.loc[
                        (params_df['mouse'] == mouse_folder.name) &
                        (params_df['session'] == session_folder.name)
                    ]
                    if not match.empty:
                        total_time_sec = match['duration'].iloc[0] if 'duration' in match.columns else np.nan
                        framerate = match['framerate'].iloc[0] if 'framerate' in match.columns else np.nan

                # Handle case where framerate might be missing
                if np.isnan(framerate):
                    print(f"⚠️ No framerate found for {ds_name}, using default 30 Hz")
                    framerate = 30.0

                
                twitch_count = len(twitches)
                active_motion_count = len(active)
                
                twitch_onsets = twitches['onset']
                twitch_offsets = twitches['offset']
                twitch_durations = ((twitch_offsets - twitch_onsets)).tolist()
                mean_twitch_duration = np.mean(twitch_durations)/framerate if twitch_count > 0 else 0
                
                inter_twitch_intrvl = np.diff(twitch_onsets).tolist() if twitch_count > 1 else []
                mean_inter_twitch_intrvl = np.mean(inter_twitch_intrvl) if len(inter_twitch_intrvl) > 0 else 0

                active_motion_onsets = active['onset']
                active_motion_offsets = active['offset']
                
                active_motion_durations = ((active_motion_offsets - active_motion_onsets)).tolist() if active_motion_count > 0 else []
                mean_active_motion_duration = np.mean(active_motion_durations)/framerate if active_motion_count > 0 else 0 #in sec 
                
                inter_active_motion_intrvl = np.diff(active_motion_onsets).tolist() if active_motion_count > 1 else []
                mean_inter_active_motion_intrvl = np.mean(inter_active_motion_intrvl) if len(inter_active_motion_intrvl) > 0 else 0

                # Compute derived quantities
                active_motion_time_sec = np.sum(active_motion_durations)/framerate if active_motion_count > 0 else np.nan # sec 
                rest_time_sec = total_time_sec - active_motion_time_sec if not np.isnan(total_time_sec) else np.nan
                active_time_percentage = (active_motion_time_sec/total_time_sec) * 100
                rest_time_percentage = (rest_time_sec/total_time_sec) * 100
                twitch_frequency = twitch_count / rest_time_sec if rest_time_sec and rest_time_sec > 0 else np.nan

                # Store results
                results[ds_name] = {
                    "active_motion_count": active_motion_count,
                    "active_motion_durations": active_motion_durations,
                    "mean_active_motion_duration": mean_active_motion_duration,
                    "active_motion_intervals": inter_active_motion_intrvl,
                    "mean_active_motion_interval": mean_inter_active_motion_intrvl,
                    "active_mot_onsets": active_motion_onsets.tolist(),
                    "active_mot_offsets": active_motion_offsets.tolist(),

                    "twitch_count": twitch_count,
                    "twitch_durations": twitch_durations,
                    "mean_twitch_duration": mean_twitch_duration,
                    "twitch_intervals": inter_twitch_intrvl,
                    "mean_twitch_interval": mean_inter_twitch_intrvl,
                    "twitch_onsets": twitch_onsets.tolist(),
                    "twitch_offsets": twitch_offsets.tolist(),

                    "active_motion_time_sec": active_motion_time_sec,
                    "rest_time_sec": rest_time_sec,
                    "total_time_sec": total_time_sec,
                    'active_time_percentage': active_time_percentage, 
                    'rest_time_percentage':rest_time_percentage,
                    "twitch_frequency": twitch_frequency,
                    "framerate": framerate, 
                    "TP_events_percentage": TP_events_percentage,
                    "FP_events_percentage": FP_events_percentage,
                    "performance_idx": performance_idx
                }
                
            except Exception as e:
                print(f"❌ Error processing {ds_name}: {e}")
                continue

    if not results:
        print("❌ No Mousecraft validation data found!")
        return pd.DataFrame()

    result_df = (
        pd.DataFrame.from_dict(results, orient='index')
        .reset_index()
        .rename(columns={'index': 'ds'})
    )
    
    print(f"🎯 Successfully processed {len(result_df)} datasets")
    return result_df


## add me

def load_auto_annotations(session_path, framerate, duration, mode=None): # mode 'rescaled" back to orig time or else: downsampled (on avg time)

    session_path = Path(session_path)  # <-- convert string to Path
    motion_annotation_path = session_path / "motion_annotation_average"
    
    if mode == 'rescaled':
        df = pd.read_excel(motion_annotation_path / 'automatic_annotations_rescaled.xlsx')
    else:
        df = pd.read_excel(motion_annotation_path / 'automatic_annotations_avg.xlsx')

    def clean_col(series):
        # Convert column to numeric and drop NaNs
        return pd.to_numeric(series, errors='coerce').dropna().values

    active_onsets = clean_col(df['active_onsets'])
    active_offsets = clean_col(df['active_offsets'])
    
    # in case an onset has no corresponding offset (eg a motion started at the end of recording)
    min_len = min(len(active_onsets), len(active_offsets))
    
    active_onsets = active_onsets[:min_len]
    active_offsets = active_offsets[:min_len]
       
    twitch_onsets = clean_col(df['twitch_onsets'])
    twitch_offsets = clean_col(df['twitch_offsets'])
    complex_onsets = clean_col(df['complex_onsets'])
    complex_offsets = clean_col(df['complex_offsets'])
    active_motion_count = len(active_onsets)   
    active_motion_durations = []
    mean_active_motion_duration = 0
    inter_active_motion_intrvl = []
    mean_inter_active_motion_intrvl = []
    
    if len(active_onsets)>0: 
        active_motion_durations = ((active_offsets - active_onsets) / framerate).tolist() if active_motion_count > 0 else []
        mean_active_motion_duration = np.mean(active_motion_durations) if active_motion_count > 0 else np.nan
        inter_active_motion_intrvl = np.diff(active_onsets) / framerate if active_motion_count > 1 else np.array([])
        mean_inter_active_motion_intrvl = np.mean(inter_active_motion_intrvl) if len(inter_active_motion_intrvl) > 0 else np.nan
        active_motion_time_sec = np.sum((active_offsets - active_onsets) / framerate) if active_motion_count > 0 else 0

    # Twitches
    twitch_count = len(twitch_onsets)
    # Make sure onsets and offsets are the same length
    min_len = min(len(twitch_onsets), len(twitch_offsets))
    twitch_onsets = twitch_onsets[:min_len]
    twitch_offsets = twitch_offsets[:min_len]

    # Inter-twitch intervals (only if more than 1 twitch exists)
    inter_twitch_intrvl = np.diff(twitch_onsets) / framerate if len(twitch_onsets) > 1 else np.array([])
    twitch_durations = ((twitch_offsets - twitch_onsets) / framerate).tolist() if twitch_count > 0 else []
    mean_twitch_duration = np.mean(twitch_durations) if twitch_count > 0 else np.nan
    inter_twitch_intrvl = np.diff(twitch_onsets) / framerate if twitch_count > 1 else np.array([])
    mean_inter_twitch_intrvl = np.mean(inter_twitch_intrvl) if len(inter_twitch_intrvl) > 0 else np.nan

    rest_time_sec = duration - active_motion_time_sec
    active_time_percentage = (active_motion_time_sec/total_time_sec) * 100
    rest_time_percentage = (rest_time_sec/total_time_sec) * 100
    twitch_frequency = twitch_count / rest_time_sec if rest_time_sec > 0 else np.nan

    return {
        "active_motion_count": active_motion_count,
        "active_motion_durations": active_motion_durations,
        "mean_active_motion_duration": mean_active_motion_duration,
        "active_motion_intervals": inter_active_motion_intrvl.tolist(),
        "mean_active_motion_interval": mean_inter_active_motion_intrvl,
        "active_mot_onsets": active_onsets.tolist(),
        "active_mot_offsets": active_offsets.tolist(),

        "twitch_count": twitch_count,
        "twitch_durations": twitch_durations,
        "mean_twitch_duration": mean_twitch_duration,
        "twitch_intervals": inter_twitch_intrvl.tolist(),
        "mean_twitch_interval": mean_inter_twitch_intrvl,
        "twitch_onsets": twitch_onsets.tolist(),
        "twitch_offsets": twitch_offsets.tolist(),

        "complex_onsets": complex_onsets.tolist(),
        "complex_offsets": complex_offsets.tolist(),

        "active_motion_time_sec": active_motion_time_sec,
        "rest_time_sec": rest_time_sec,
        "active_time_percentage": active_time_percentage,
        "rest_time_percentage": rest_time_percentage, 
        "total_time_sec": duration,
        "twitch_frequency": twitch_frequency
    }


def df_to_dict(df, index_col="ds"):
    """
    Convert (the validated) DataFrame into a flat dictionary where each column
    becomes a single flattened list of values (even if the original column contains lists).
    """
    if df is None or df.empty:
        return {}

    if index_col not in df.columns:
        raise ValueError(f"Expected '{index_col}' column in DataFrame")

    result_dict = {}

    for col in df.columns:
        if col == index_col:
            continue

        column_vals = df[col].tolist()

        # Flatten list-of-lists into a single list
        flat_vals = []
        for v in column_vals:
            if isinstance(v, (list, np.ndarray)):
                flat_vals.extend(list(v))   # EXTEND, not append
            elif pd.isna(v):
                continue
            else:
                flat_vals.append(v)

        result_dict[col] = flat_vals

    return result_dict

# downsample dictionary from a certain framerate to another (translate frames to doansampled times) eg. 15 or 30 Hz to 3 Hz and append new keys _avg 
def downsample_dict(annotations_dict, original_framerate, target_framerate):
    """
    Downsample frame-based annotations and add '_avg' keys.
    Automatically flattens lists of lists.
    """
    if annotations_dict is None:
        return None

    if original_framerate % target_framerate != 0:
        raise ValueError("Framerate ratio must be an integer.")

    ds_factor = original_framerate / target_framerate
    result = annotations_dict.copy()

    onset_offset_keys = [
        'active_mot_onsets', 'active_mot_offsets',
        'twitch_onsets', 'twitch_offsets'
    ]

    for key in onset_offset_keys:
        if key not in annotations_dict:
            continue

        frames = annotations_dict[key]

        if frames is None:
            result[f"{key}_avg"] = None
            continue

        # Flatten in case it's a list of lists
        if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], list):
            frames = frames[0]

        frames = np.array(frames)
        frames_ds = np.floor(frames / ds_factor).astype(int)

        result[f"{key}_avg"] = frames_ds.tolist()  # Save as flat list

    return result


# extract onset-offset segments and return dictionary with keys the class and a list with tuples (onset,offset), (onset2, offset2) ... (onsetn, offsetn)

# solves different nomenclature issue in naming in pixelnmf ex: ds = 'sz105_2025-05-30_a' or ds = 'sz105_2025_05_29_a', offset
def extract_motion_segments(beh_dict, onsets_a, offsets_a, onsets_b, offsets_b, total_frames=None):
    """
    Extract motion state segments (active, twitch, rest, rest+twitch) from behavior dictionary.

    Parameters
    ----------
    beh_dict : dict
        Must contain:
        - 'onsets_a', 'offsets_a'
        - 'onsets_b', 'offsets_b', where _a and _b are states of interest eg active, twitch 
    total_frames : int, optional
        Total number of frames in the recording. 
        If not provided, tries to infer from beh_dict['total_time_sec'] * beh_dict['framerate'].

    Returns
    -------
    segments : dict
        {
          "active": [(onset, offset), ...],
          "twitch": [(onset, offset), ...],
          "rest":   [(onset, offset), ...],
          "rest+twitch": [(onset, offset), ...]
        }
    """

    # Make sure total_frames is defined
    if total_frames is None:
        total_frames = int(beh_dict['total_time_sec'] * beh_dict['framerate'])

    # Collect active + twitch
    active = list(zip(beh_dict[onsets_a], beh_dict[offsets_a]))
    twitch = list(zip(beh_dict[onsets_b], beh_dict[offsets_b]))

    # Mark all frames that are part of active or twitch
    used = [False] * total_frames
    for onset, offset in active + twitch:
        onset=int(onset)
        offset=int(offset)
        used[onset:offset] = [True] * (offset - onset)

    # Extract rest segments as gaps (frames not in active or twitch)
    rest = []
    in_rest = False
    rest_start = None
    for i, u in enumerate(used):
        if not u and not in_rest:
            rest_start = i
            in_rest = True
        elif u and in_rest:
            rest.append((rest_start, i))  # rest ends at i (exclusive)
            in_rest = False
    if in_rest:
        rest.append((rest_start, total_frames))

    # Extract rest+twitch segments = all frames NOT in active
    used_active = [False] * total_frames
    for onset, offset in active:
        onset=int(onset)
        offset=int(offset)
        used_active[onset:offset] = [True] * (offset - onset)

    rest_twitch = []
    in_segment = False
    segment_start = None
    for i, u in enumerate(used_active):
        if not u and not in_segment:
            segment_start = i
            in_segment = True
        elif u and in_segment:
            rest_twitch.append((segment_start, i))  # segment ends at i (exclusive)
            in_segment = False
    if in_segment:
        rest_twitch.append((segment_start, total_frames))

    return {
        "active": active,
        "twitch": twitch,
        "rest": rest,
        "rest+twitch": rest_twitch
    }

def compute_downsample_factor(initial_fps, target_fps):
    """
    Compute averaging factor to convert from initial_fps to target_fps.

    Parameters
    ----------
    initial_fps : float
        Original sampling frequency (e.g., 30)
    target_fps : float
        Desired sampling frequency (e.g., 3)

    Returns
    -------
    int
        Averaging factor
    """
    if target_fps >= initial_fps:
        raise ValueError("Target FPS must be lower than initial FPS.")

    factor = initial_fps / target_fps

    if not factor.is_integer():
        raise ValueError(
            f"Downsampling factor must be integer. Got {factor}. "
            "Choose a compatible target_fps."
        )

    return int(factor)

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

from scipy import stats 

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from diptest import diptest

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

    # Dip test
    dip_stat, dip_p = diptest(data) # dip test for bimodality : https://skeptric.com/dip-statistic/

    bimodal = (num_kde_peaks == 2) or (dip_p < 0.05)

    if plot:
        plt.figure(figsize=(4, 4), dpi=300)
        plt.plot(x_vals, density, label="KDE", linewidth=7)
        plt.plot(x_vals[peaks], density[peaks], "ro", label=f"Peaks ({num_kde_peaks})")
        plt.title(f"Bimodality check") #: KDE & Hartigan's test={"*" if bimodal else "ns"}") #  KDE & Hartigan's test={"*" if bimodal else "ns"}"
        plt.xlabel("motion energy")
        plt.ylabel("density")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    if bimodal:
        print("✅ Detected bimodal distribution (2 peaks)")
    else:
        print(f"⚠️ Detected {num_peaks} peaks — not strictly bimodal")
        
    bimodality_metrics = {
        "bimodal": bimodal,
        "num_kde_peaks": num_kde_peaks,
        "dip_stat": dip_stat,
        "dip_p": dip_p
    }

    return bimodal, bimodality_metrics

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

# compute thresholds

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
        plt.figure(figsize=(4, 3), dpi=300)
        plt.title(f'{title}', y=1.2)
        plt.hist(motion_signal, bins=50, alpha=0.9)

        # mark threshold lines
        plt.axvline(x=threshold_motion_mean_sd, color='red', label='mean + sd', linestyle='--')
        plt.axvline(x=threshold_motion_otsu, color='salmon', label='Otsu', linestyle='--', linewidth=2)
        plt.axvline(x=threshold_motion_li, color='darkred', label='Li', linestyle='--')

        plt.legend(loc='upper right', frameon=False, fontsize=7)
        plt.savefig(save_dir / f'{title}.png')
        plt.show()

    return threshold_motion_mean_sd, threshold_motion_li, threshold_motion_otsu

from scipy.ndimage import label

def binarise_motion(motion_signal, binary_threshold, min_duration, min_inactive_gap=9, bimodality=False):
    '''
    Binarises motion signal into 0s (rest) and 1s (active).
    
    Parameters:
    - motion_signal: 1D array-like, motion energy or similar signal.
    - binary_threshold: threshold to classify activity.
    - min_duration: minimum number of consecutive frames to qualify as active motion.
    - min_inactive_gap: optional, minimum gap to consider between active segments (not yet implemented here).
    - bimodality: bool, whether the motion signal distribution is considered bimodal.

    Returns:
    - bin_motion_signal: binary array of 0s (rest) and 1s (active)
    - inds_active_state: indices where state is active (1)
    - inds_rest_state: indices where state is rest (0)
    '''
    
    bin_motion_signal = np.zeros(len(motion_signal), dtype=int)

    # Step 1: Check if any motion segment passes the threshold and min_duration
    above_thresh = motion_signal > binary_threshold
    labeled_array, n_features = label(above_thresh)
    
    has_active_motion = False
    for i in range(1, n_features + 1):
        segment = np.where(labeled_array == i)[0]
        if len(segment) > min_duration:
            has_active_motion = True
            break
    
    # Step 2: Apply binarisation if active motion exists or bimodality is True
    if has_active_motion or bimodality:
        for i in range(1, n_features + 1):
            segment = np.where(labeled_array == i)[0]
            if len(segment) > min_duration:
                bin_motion_signal[segment] = 1
        # Otherwise, remains 0 (rest)
    
    # Step 3: Get indices
    inds_active_state = np.where(bin_motion_signal == 1)[0]
    inds_rest_state = np.where(bin_motion_signal == 0)[0]

    return has_active_motion, bin_motion_signal, inds_active_state, inds_rest_state

from statsmodels.robust import mad

def compute_thresholds_for_twitch_state_detection(motion_signal, save_dir=None, plot=True):
    '''
    Compute statistical thresholds to use for threshold-based state detection (active/awake - rest)
    Here we use: Otsu (prefer if binary distribution), Li (mutliple peak distribution), or mean+sd (gaussian distribution)
    '''    
    # mean+sd threshold 
    motion_mean = np.mean(motion_signal)
    motion_sd = np.std(motion_signal)

    threshold_motion_mean_sd = motion_mean + 3*motion_sd

    threshold_motion_li = filters.threshold_li(motion_signal)

    threshold_motion_otsu = filters.threshold_otsu(motion_signal)

    threshold_95 = np.percentile(motion_signal, 95) 

    # log_signal = np.log1p(motion_signal)  # log1p avoids issues with zeroes
    # threshold_log= np.percentile(motion_signal, 95)

    med = np.median(motion_signal)
    mad_val = mad(motion_signal)
    threshold_mad = med + 3 * mad_val  # 3x MAD is analogous to ~3σ in normal data

    if plot:
        plt.figure(figsize=(3, 3), dpi=300)
        plt.title('stat thresholds on mot_en', y=1.2)
        plt.hist(motion_signal, bins=50, alpha=0.9)

        # mark threshold lines
        plt.axvline(x=threshold_motion_mean_sd, color='red', label='mean + sd', linestyle='--')
        plt.axvline(x=threshold_motion_otsu, color='salmon', label='Otsu', linestyle='--')
        plt.axvline(x=threshold_motion_li, color='darkred', label='Li', linestyle='--')
        # plt.axvline(x=threshold_log, color='pink', label='95 threshold on log_motion')
        plt.axvline(x=threshold_mad, color='pink', label='threshold mad', linestyle='--')
        plt.axvline(x=threshold_95, color='purple', label='threshold 95%', linestyle='--')


        plt.legend(loc='upper right', frameon=False, fontsize=5)
        plt.savefig(save_dir / 'binary_state_thresholds.png')
        plt.show()

    return threshold_motion_mean_sd, threshold_motion_li, threshold_motion_otsu, threshold_mad, threshold_95

def binarise_twitch(motion_energy, twitch_segments):

    # Initialize bin_twitch   
    bin_twitch = np.zeros(len(motion_energy))

    # flatten the nested list
    flat_inds = [idx for segment in twitch_segments for idx in segment] # filtered corrected twitch segments 

    print("Flattened Indices:", flat_inds)

    # Assign 1 to twitch inds
    for idx in flat_inds:
        if idx < len(motion_energy):  # Ensure index is within bounds
            bin_twitch[idx] = 1
            print(f"Assigning 1 at column {idx}")
        else:
            print(f"Skipping out-of-bounds index: {idx}")
    return bin_twitch


def filter_twitches_by_awake_proximity(inds_twitches, inds_active_state, min_distance=None):

    inds_active_array = inds_active_state[0] if isinstance(inds_active_state, tuple) else inds_active_state

    filtered_twitches = []

    if len(inds_active_array) == 0:
        # if unimodal distribution so no active motions: return all twitches as valid
        return np.array(inds_twitches)
        

    for idx_twitch in inds_twitches:
        distance = np.abs(inds_active_array - idx_twitch)
        if np.min(distance) > min_distance:
            filtered_twitches.append(idx_twitch)

    filtered_twitches = np.array(filtered_twitches)

    return filtered_twitches

# filter twitches based on duration 
def filter_segments_by_duration(segments, duration_threshold):
# Filter out groups based on the duration threshold
    return [twitches for twitches in segments if len(twitches) <= duration_threshold]

def binarise_twitch(motion_energy, twitch_segments):

    # Initialize bin_twitch   
    bin_twitch = np.zeros(len(motion_energy))

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


def get_behavior_classification(bin_putative_twitch, bin_twitch, bin_motion_energy, active_onsets, twitch_onsets):
    """
    Classify behavior states and key motion onsets.

    Parameters:
        bin_putative_twitch (np.ndarray): raw unfiltered twitch binary array
        bin_twitch (np.ndarray): filtered twitch binary array
        bin_motion_energy (np.ndarray): binarized motion energy (1 = active, 0 = rest)

    Returns:
        dict: classified behavior array and key motion index groups
    """
    classified_behavior = bin_motion_energy.copy()
    classified_behavior[twitch_onsets] = -1
    classified_behavior[active_onsets] = 1

    # 2. Complex motion onsets (filtered twitches)
    complex_motion = (bin_putative_twitch == 1) & (bin_twitch == 0) # not-twitches 

    complex_onsets = np.where((complex_motion[1:]== 1) & (complex_motion[:-1]== 0))[0] +1
    complex_offsets = np.where((complex_motion[1:] ==0) & (complex_motion[:-1] ==1))[0] +1

    complex_motion_segments = []
    for onset,offset in zip(complex_onsets, complex_offsets):
        complex_motion_segments.append((onset,offset))
        classified_behavior[onset:offset] =2 

    # Index groups
    active_and_complex_motions = np.where((classified_behavior == 1) | (classified_behavior == 2))[0]
    only_active_motions = np.where(classified_behavior == 1)[0]
    twitch_onsets = np.where(classified_behavior == -1)[0]

    return classified_behavior, active_and_complex_motions, only_active_motions, complex_motion_segments ,complex_onsets, complex_offsets