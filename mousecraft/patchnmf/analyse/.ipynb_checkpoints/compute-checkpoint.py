import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, label, find_objects

# image processing
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_li
from skimage.measure import find_contours
from scipy.ndimage import center_of_mass
from sklearn.decomposition import PCA

def downsample_tiff_avg(tiff, n=4):

    tiff_ds = []
    for i in range(tiff.shape[0]):
        kernel = np.ones((n, n))
        convolved = convolve2d(tiff[i,:,:], kernel, mode='valid')
        this_tiff_ds = convolved[::n, ::n] / n
        #tiff_ds = np.concatenate((tiff_ds, this_tiff_ds))
        tiff_ds.append(this_tiff_ds)
        
        if i % 100 == 0:
            print(f'Done with {i} frames') #to check progress

    tiff_ds = np.array(tiff_ds)

    plt.imshow(np.mean(tiff, 0), cmap='gray')
    plt.title('original frame example')
    plt.show()
    plt.imshow(np.mean(tiff_ds, 0), cmap='gray')
    plt.title('downsampled frame example')
    plt.show()

    return tiff_ds

def compute_nmfpx_blur_thr_std(nmf_px, tiff_shape, blur_std=None, min_size=None):
    _, x, y = tiff_shape
    n_components = nmf_px.n_components

    loading_imgs = []
    loading_imgs_filt = []
    rois_auto = []
    std_thresholds = [] 
    
    for i in range(n_components):

        loading = nmf_px.components_[i,:] # attr of nmf object
        loading_img = loading.reshape(x, y) #reshape ith nmf

        # normalize motifs between 0-1
        min_val = np.min(loading_img)
        max_val = np.max(loading_img)
        loading_img = (loading_img - min_val) / (max_val - min_val)
        
        loading_img_filt, filtered_mask, std_thresh = get_thr_img_auto_std(loading_img, blur_std, min_size) #blur and thresh #was i instead of min_size
        
        # append to lists
        loading_imgs.append(loading_img)
        loading_imgs_filt.append(loading_img_filt)
        rois_auto.append(filtered_mask) #add it to the list, fisrt iteration list is empty, after second iteration 1 nmf gets into teh lsit as threshloded matrix pf t and f
        std_thresholds.append(std_thresh) # store the binarisation threshold 
    
    return loading_imgs, loading_imgs_filt, rois_auto, std_thresholds 

def get_thr_img_auto_std(loading_img, blur_std=None, min_size=None):
    # Apply Gaussian blur to smooth the image
    loading_img_filt = gaussian_filter(loading_img, blur_std)
    
    # Calculate threshold based on standard deviation
    std_thresh = 2 * np.std(loading_img_filt)
    
    # Create binary mask by thresholding
    roi_auto = loading_img_filt > std_thresh

    # Label connected components in the binary mask
    labeled_outputs, number_of_objects = label(roi_auto)
    
    # Find the slices for each labeled object
    objects = find_objects(labeled_outputs)

    # Create a new mask to store only large enough objects
    filtered_mask = np.zeros_like(roi_auto, dtype=bool)
    
    for i, obj_slice in enumerate(objects):
        if obj_slice is not None:
            # Calculate the size of the object in pixels
            obj_size = np.sum(labeled_outputs[obj_slice] == (i + 1))
            
            # Keep only objects larger than the specified min_size
            if obj_size >= min_size:
                filtered_mask[obj_slice] = (labeled_outputs[obj_slice] == (i + 1))

    # Return the filtered image, the mask for significant objects, and the threshold
    return loading_img_filt, filtered_mask, std_thresh


def sort_by_pc1(data, component): #time, rois 
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(data.T)  # rois, time 
    sorted_indices = np.argsort(pc1[:,component])
    sorted_data = data[:, sorted_indices]
    return sorted_data

def reconstruct_data(W, H, data):

    reconstructed_data = np.dot(W, H)
    return reconstructed_data

# def compute_centroids(rois_auto, tiff_shape, save_dir=None):
#     rois_auto_centroids = []
#     for roi in rois_auto:
#         centroid = np.array(center_of_mass(roi))
#         rois_auto_centroids.append(centroid)

#     return rois_auto_centroids 

#auto li threhsold as jure proposed 

# only compute W correlation ( space corr mat) for filtered ('good') rois




'''
Get all roi_areas excluding the patterns that are bigger than a threhsold 
(set to 50% of the total FOV here, but you may want to adjust it) 
Too big patterns are usually noise (as seen in simulation. 
However check also the temproal trace and if its too frequent then for sure it is noise.
'''

def get_all_filtered_roi_areas(rois_auto,loading_imgs, loading_imgs_filt, x_axis, y_axis, total_area, min_size, max_size, res = None): 
    
    all_roi_areas = []
    filtered_rois_auto = []
    filtered_rois_idxs = []
    filtered_loading_imgs = []
    filtered_loading_imgs_filt = []

    for i, roi in enumerate(rois_auto):
           
        roi_area = np.sum(roi)*(res**2) #this should handle multiple object patterns, computes the size of both 
            
        if min_size <= roi_area <= max_size:
            all_roi_areas.append(roi_area)
            filtered_rois_auto.append(roi)
            filtered_rois_idxs.append(i) #store the idx of 'good' rois to include them in later comps
            # Append the corresponding images for filtered ROIs
            filtered_loading_imgs.append(loading_imgs[i])
            filtered_loading_imgs_filt.append(loading_imgs_filt[i])

    return all_roi_areas, filtered_rois_auto, filtered_rois_idxs, filtered_loading_imgs, filtered_loading_imgs_filt
    

def filter_w_mat(W, filtered_rois_idxs):
    filtered_w = W[filtered_rois_idxs, :]

    return filtered_w

def filter_h_mat(H, filtered_rois_idxs):
    filtered_h = H.T[filtered_rois_idxs, :] #n_components, time 
    return filtered_h

def compute_iou(rois_auto): # of filtered rois_auto (important, i do filtered_rois_auto =  rois_auto)
    ious = []
    for i in range(len(rois_auto)):
        roi1 = rois_auto[i]
        
        for j in range(i+1, len(rois_auto)):
            roi2 = rois_auto[j]
           
            intersection = np.logical_and(roi1, roi2)
            union = np.logical_or(roi1, roi2) 
            iou = np.sum(intersection)/np.sum(union)
            ious.append(iou)
    mean_iou = np.mean(ious)

    return ious, mean_iou  

import warnings

def get_dist_binned_iou(distances, ious, dist_binning=10):
    distances_um = distances *10

    max_dist = np.ceil(np.max(distances_um)) #round up the max distance 
    bins_iou = np.arange(0, max_dist + dist_binning, dist_binning) 
    means = []
    stds = []

    for i in bins_iou[:-1]: #loop thoruhg distances
        lbo=i
        print(lbo)
        ubo= i + dist_binning
        # ious_in_bound = np.array(ious)[np.logical_and(distances <= lbo, distances < ubo)]
        ious_in_bound = np.array(ious)[(distances_um >= lbo) & (distances_um < ubo)]
        with warnings.catch_warnings():  # Ignore warnings for empty bins
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means.append(np.mean(ious_in_bound) if ious_in_bound.size > 0 else np.nan)  # Handle empty bin
            stds.append(np.std(ious_in_bound) if ious_in_bound.size > 0 else np.nan)  # Handle empty bin
    
    return np.array(means), np.array(stds), np.array(bins_iou)

# def get_dist_binned_corr(distances, h_correlations, dist_binning=10):
#     max_dist = np.ceil(np.max(distances)) #round up the max distance 
#     bins = np.arange(0, max_dist + dist_binning, dist_binning) 
#     means = []
#     stds = []

#     for i in bins[:-1]: #loop thoruhg distances
#         lbo=i
#         ubo= i + dist_binning
#         # ious_in_bound = np.array(ious)[np.logical_and(distances <= lbo, distances < ubo)]
#         h_corrs_in_bound = np.array(h_correlations)[(distances >= lbo) & (distances < ubo)]
#         with warnings.catch_warnings():  # Ignore warnings for empty bins
#             warnings.simplefilter("ignore", category=RuntimeWarning)
#             means.append(np.mean(h_corrs_in_bound) if h_corrs_in_bound.size > 0 else np.nan)  # Handle empty bin
#             stds.append(np.std(h_corrs_in_bound) if h_corrs_in_bound.size > 0 else np.nan)  # Handle empty bin
    
#     return np.array(means), np.array(stds), bins_iou 

# Currently replacing function under (todo: homogenise with upper function style)
def compute_dist_bin_corr(distances, correlations, bin_size=10):
    distances_um = distances *10
    bins = np.arange(0, np.round(np.max(distances_um)) + bin_size, bin_size)
    print(f'bins{bins}')

    bin_means_corr = []
    bin_stds_corr = []
    bin_cents_corr = []

    for i in range(len(bins) - 1):
        # Get the indices for the distances that fall within the current bin
        bin_indices = np.where((distances_um >= bins[i]) & (distances_um < bins[i+1]))[0]
    
        # Calculate the mean IoU for the current bin if there are values in the bin
        if len(bin_indices) > 0:
            mean_corr = np.mean(correlations[bin_indices])  # Ensure corrs is np array
            std_corr = np.std(correlations[bin_indices])
            
            bin_means_corr.append(mean_corr)
            bin_stds_corr.append( std_corr)
            # Store the center of the bin for plotting
            bin_cents_corr.append((bins[i] + bins[i+1]) / 2)

    return np.array(bins), np.array(bin_means_corr), np.array(bin_stds_corr), np.array(bin_cents_corr)

    
def get_roi_conts(rois_auto):
    conts = []
    n_conts = []

    for roi in rois_auto:
        roi_cont = find_contours(roi)
        conts.append(roi_cont)
        n_conts.append(len(roi_cont))

    
    return conts, n_conts
    
#this maybe not the best to use, it does not necessarily corresponf to time. Better use H for that (adapted) 

def get_loading_times(nmf_t):
    loading_times = []
    n_components = nmf_t.n_components

    for i in range(n_components):
        loading_time = nmf_t.components_[i, :]
        loading_times.append(loading_time)

    return loading_times

#this function identifies which pixel is the corresponding to the centroid
def get_pixel_indices_for_centroids(rois_auto_centroids, x_axis_FOV):
    centroid_pixel_indices = []

    for centroid in rois_auto_centroids:
        centroid_x = int(centroid[0])
        centroid_y = int(centroid[1])
        #print(centroid_y) #checking 
        
        centroid_pixel_index = centroid_x * x_axis_FOV  + centroid_y
        centroid_pixel_indices.append(centroid_pixel_index)
    
    return centroid_pixel_indices

def get_norm_comp_number(n_components, original_FOV): #    normalize the number of components by the size of the field of view (FOV).
    n_comps_norm = n_components / original_FOV**2
    n_comps_per_1000_sq_μm = n_comps_norm*1000**2
    print(f'Number of components normalized is {n_comps_per_1000_sq_μm:.3f} sth that needs to be confirmed')
    return n_comps_norm

def is_border_touching(roi , y_axis_FOV, x_axis_FOV):
    height, width = y_axis_FOV, x_axis_FOV 
    return np.any(roi[0, :]) or np.any(roi[:, 0]) or np.any(roi[height - 1, :]) or np.any(roi[:, width - 1])

def compute_iou(rois_auto):
    ious = []
    for i in range(len(rois_auto)):
        roi1 = rois_auto[i]
        
        for j in range(i+1, len(rois_auto)):
            roi2 = rois_auto[j]
           
            intersection = np.logical_and(roi1, roi2)
            union = np.logical_or(roi1, roi2) 
            iou = np.sum(intersection)/np.sum(union)
            ious.append(iou)
    mean_iou = np.mean(ious)

    return ious, mean_iou      

def compute_iou_matrix(rois_auto):
    ious_mat = np.zeros((len(rois_auto), len(rois_auto)))
    for i in range(len(rois_auto)):
        roi1 = rois_auto[i]
        
        for j in range(len(rois_auto)):
            roi2 = rois_auto[j]
           
            intersection = np.logical_and(roi1, roi2)
            union = np.logical_or(roi1, roi2) 
            ious_mat[i, j] = np.sum(intersection)/np.sum(union)
            
    return ious_mat  

def compute_correlations_for_roi(roi_mask, data, n_pixels=10):
    """
    Computes correlations within a given ROI and between inside and outside the ROI.

    Parameters:
    - roi_mask: Binary mask for the ROI, where 1 indicates pixels inside the ROI.
    - data: 2D array of shape (n_timepoints, n_pixels) containing pixel time series data.
    - n_pixels: Number of pixels to randomly select from inside and outside the ROI.

    Returns:
    - corr_within: Correlation matrix for pixels within the ROI.
    - corr_inside_outside_vals: Correlation matrix between inside and outside pixels.
    - inside_idx: Indices of selected inside pixels.
    - outside_idx: Indices of selected outside pixels.
    """
    
    # Flatten the roi_mask to match the shape of the tiff_data
    roi_mask_flat = roi_mask.flatten()
    
    # Get indices of the pixels inside and outside the ROI
    inside_pixels_idx = np.argwhere(roi_mask_flat == 1).flatten()  # Inside ROI
    outside_pixels_idx = np.argwhere(roi_mask_flat == 0).flatten()  # Outside ROI

    # Set n_pixels equal to the number of inside pixels
    n_pixels = len(inside_pixels_idx)

    # Check if there are enough outside pixels for sampling
    if len(outside_pixels_idx) < n_pixels:
        print("Skipping ROI as there are insufficient outside pixels.")
        return None, None, None, None  # Skip this ROI if not enough outside pixels

    # Randomly select n_pixels inside the ROI
    #random_inside_idx = np.random.choice(inside_pixels_idx, n_pixels, replace=False)
    random_inside_idx = inside_pixels_idx  # All pixels inside the ROI

def compute_within_without_pattern_corr(correlations_in , correlations_in_out):
    upper_tri_within = correlations_in[np.triu_indices_from(correlations_in, k=1)]
    upper_tri_in_out = correlations_in_out[np.triu_indices_from(correlations_in_out, k=1)]

    mean_corr_within = np.mean(upper_tri_within)
    mean_corr_in_out = np.mean(upper_tri_in_out)

    return mean_corr_within, mean_corr_in_out
