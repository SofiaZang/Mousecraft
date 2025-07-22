import numpy as np
import matplotlib.pyplot as plt
import random
import os

from scipy.stats import zscore
from scipy.ndimage import gaussian_filter, convolve1d

def poiss_train(tau, t_max=10):
    # tau - parameter of exponential distribution
    # t_max - duration of spike train
    
    st = []
    st.append(np.random.exponential(tau)) # first spike time
    count = 0
    
    while st[count] < t_max:
        st.append(st[count] + np.random.exponential(tau)) # subsequent spike times with ISIs from exponential distribution
        count += 1
    
    # removing final spike (outside [0, 10])
    st = st[0:-1]
    
    return np.array(st), count

def make_sim_path(sim_id):
    sim_path = f'../data/{sim_id}'
    if not os.path.exists(sim_path):
        os.mkdir(sim_path)
        
    return sim_path

# running simulation
def run_simulation(params):
    # t = np.arange(0,params.t_max)
    movie_blank = np.zeros((params.xy_px, params.xy_px, params.t_max))
    movie = np.copy(movie_blank)

    truth_ts = []
    truth_pxs = []

    act_kernel = np.exp(-np.arange(0,params.kernel_range)/params.act_tau)

    for i in range(params.n_patches):
        # randomly generating patch in pixel space
        patch_d = random.randrange(params.patch_size_min, params.patch_size_max)
        patch_x = random.randrange(patch_d, params.xy_px-patch_d)
        patch_y = random.randrange(patch_d, params.xy_px-patch_d)

        # randomly generating the activation of patch in time
        act_t,_ = poiss_train(params.poiss_tau, t_max=params.t_max)

        # computing contribution of patch to movie and adding to final movie
        patch_movie = np.copy(movie_blank)
        patch_movie[patch_x-patch_d:patch_x+patch_d, patch_y-patch_d:patch_y+patch_d, np.round(act_t).astype(int)] = params.act_scaling
        patch_movie = convolve1d(patch_movie, act_kernel) # adding 'GCamp decay'
        patch_movie = gaussian_filter(patch_movie, params.smoothing_sigma) # smoothing square in image
        movie += patch_movie # accumulate contribution of all patches in 'movie'

        # getting and saving ground truth
        truth_px = patch_movie[:,:,np.round(act_t).astype(int)[0]]
        truth_t = np.mean(patch_movie,(0,1))
        truth_ts.append(truth_t)
        truth_pxs.append(truth_px)

    # adding noise
    noise_mask = np.random.poisson(params.im_noise, movie.shape)
    movie += noise_mask
    
    movie = movie.transpose((2, 0, 1))
    
    return truth_pxs, truth_ts, movie

def reshape_list_im_to_mat(list_im):
    list_im_np = np.array(list_im)

    if list_im_np.ndim > 2: # if list contains images (pxNMF), in case of tNMF this step is skipped
        (x, y, z) = list_im_np.shape
        list_im_np = list_im_np.reshape(x, y*z)

    return list_im_np

def plot_covmat(covmat):
    plt.imshow(covmat)
    plt.ylabel('true px patch')
    plt.xlabel('empirical px patch')

def covariance_sort(list1, list2):
    # first making them from lists to matrices for computation
    mat1 = reshape_list_im_to_mat(list1) # ground truth
    mat2 = reshape_list_im_to_mat(list2) # nmf

    covmat = zscore(mat1, 1) @ zscore(mat2, 1).T
    plot_covmat(covmat)
    sort_ind = np.argmax(covmat, 0) # indices of best match between ground truth and simulation
    print(f'Sorting by indices (maxima along columns): {sort_ind}\n')
        
    return sort_ind

