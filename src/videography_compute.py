import cv2
import tifffile as tifffile
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

def smooth_with_gaussian(var, sigma=None):
    smoothed_var = gaussian_filter1d(var, sigma=sigma)

    return smoothed_var

def compute_motion_energy(movie_path=None, xrange=None, yrange=None, save_path=None):
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
    motion_energy /= np.max(motion_energy) # scale between 0-1
    
    return motion_energy
'''
Define hyperparameters for twitch detection. Added a minimum distance (in frames) required 
for the twitch to be detected (this is to avoid inter-active-sleep periods artifacts, if the twitch happens too close 
too an active motion (precedes or follows) exclude it, becs we cannot be sure the mouse was asleep already or in between.
We only keep best, robust twitches 
'''
def filter_twitches(inds_twitches, inds_active_state, min_distance=None):
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