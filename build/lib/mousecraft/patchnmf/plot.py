import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from matplotlib.colors import Normalize
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import zscore


def plot_traces(tstamps, grid_fluo, ind_neurons, tmin=0, tmax=None, title='', save_dir=None):

    ind_tmax = grid_fluo.shape[1] if tmax==None else int(tmax/(tstamps[1]-tstamps[0]))
    ind_tmin = int(tmin/(tstamps[1]-tstamps[0]))
    
    fluo_plot = grid_fluo[ind_neurons, ind_tmin:ind_tmax]
    # spks_plot = spks[ind_neurons, ind_tmin:ind_tmax]
    tstamps_plot = tstamps[ind_tmin:ind_tmax]
    
    fig, axs = plt.subplots(10, 1, figsize=(5,5), dpi=200)
    axs[0].set_title(title)
                   
    for i in range(len(ind_neurons)):
        
        axs[i].plot(tstamps_plot, fluo_plot[i,:], linewidth=0.5)
        # axs[i].plot(tstamps_plot, spks_plot[i,:], linewidth=0.5, c='grey')

        axs[i].set_yticks([]) # y label is arbitrary units
        axs[i].set_ylabel("")
        axs[i].spines["top"].set_linewidth(0)
        axs[i].spines["right"].set_linewidth(0)
        axs[i].spines["left"].set_linewidth(0)  

        if i != len(ind_neurons)-1:
            axs[i].spines["bottom"].set_linewidth(0)
            axs[i].set_xticks([])
            axs[i].set_xlabel("")
        else:
            axs[i].set_xlabel('Time (s)')
        
        if type(save_dir) == str:
            save_filename = os.path.join(save_dir, title)
            plt.savefig(save_filename)


def generate_random_hex_colors(num_colors):
    random_colors = []
    for _ in range(num_colors):
        # Generate random values for hue, saturation, and value
        hue = random.random()  # Random hue value between 0 and 1
        saturation = random.uniform(0.5, 1.0)  # Random saturation value between 0.5 and 1
        value = random.uniform(0.5, 1.0)  # Random value (brightness) between 0.5 and 1
        
        # Convert HSV to RGB
        rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert RGB to hexadecimal format
        hex_color = "#{:02X}{:02X}{:02X}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))
        
        # Add the hexadecimal color to the list
        random_colors.append(hex_color)
    
    return random_colors

def plot_acts(data, title='', save_dir=None, save=True):
    _, axs = plt.subplots(2, 1, figsize=(10,4), dpi = 400, gridspec_kw={'height_ratios': [3, 1]})
    
    axs[0].imshow(data.T, cmap='gray_r', aspect='auto', vmin=np.percentile(data,1), vmax=np.percentile(data,95),alpha=0.8) # z scored for 0.001 data.mean()+ 2*data.std() 
    axs[0].xaxis.set_visible(False)
    axs[0].set_ylabel('roi id')
    axs[0].set_title(f'{title}', fontsize=20)
 

    axs[1].plot(np.mean(data, axis=1), label='avg') # mean across px 
    axs[1].set_xlim(0, data.shape[0])
    axs[1].set_ylabel('avg.')
    axs[1].set_xlabel('time (frames)')
#    axs[1].set_ylim([(np.min(data) - 0.5), np.max(data)]) # +0.5 for better visuals

    if save:
        plt.savefig(save_dir + f'/acts_{title}.png')


        
# if i want to plot pc1 of acts as well under or motion for example
# def plot_acts_sorted(data, embedding, add_plot=[], component=0, title='', save_dir=None, save=True):
    
#     temp = np.argsort(embedding[:,component])
#     plot_acts(data[:, temp].T, save=False, title=f'sorted_{title}')
#     if len(add_plot)>0:
#         plt.plot(add_plot[:,component]*10, label=f'PC{component}')
    
#     plot_corr(data.T[temp,:], title=f'acts_corr_sorted_{title}', save_dir=save_dir,save=save)

#     if save:
#         plt.savefig(save_dir + f'/acts_sorted_{title}.png')

def plot_acts_sorted(data, embedding, component=0, title='', save_dir=None, save=True):
    # Sort based on the embedding's specified component
    temp = np.argsort(embedding[:, component]) # expects rois, embedding 
    
    # Plot the sorted data (activities) based on the embedding's sorted component
    plot_acts(data[:, temp], save=False, title=f'sorted_{title}') #T in plot_acts for plotting     
    if save:
        plt.savefig(save_dir + f'/acts_sorted_{title}.png')
    
    plt.show()

#plots pc colorcoded cells 

def plot_pcs(pcs, behaviour, behaviour_name, framerate, save_dir, n=10, zspacing=15):
  plt.style.use('default')
  plt.figure(figsize=(18,13), dpi=200)
  for i in range(n):
    cmap = plt.get_cmap('viridis', n)

    plt.plot(np.arange(behaviour.shape[0]) / framerate,  # Time axis
        zscore(behaviour) - (i * zspacing),  # Normalized motion energy trace
        label=f'{behaviour_name}',
        color='gray',
        linewidth=2,
        alpha=0.6
    )
    plt.plot(np.arange(pcs.shape[1])/framerate, zscore(pcs.T[:,i]) - i*zspacing, color=cmap(i), alpha=0.7)
    # plt.plot(np.arange(len(motion_energy)/framerate, zscore(motion_energy) - (i * zspacing)), color='orange')

  plt.yticks((np.arange(n))*zspacing*-1, [f'PC{i+1}' for i in range(n)], fontsize=25)
  plt.title(f'PCs-{behaviour_name}', fontsize=30)
  # plt.yticks(['mot_en'])
  ax = plt.gca()
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.savefig(str(save_dir) + f'PCs_plotted_with_{behaviour_name}.png') #change to without str and /
            
def plot_top_bottom_rois(fluo, xpix, ypix, image, pc1_values, title='rois_colored', save_path=None):
    # Number of cells to include in top and bottom 30%
    num_cells = len(pc1_values)
    num_top_bottom = int(0.3 * num_cells)
    num_middle = num_cells - 2 * num_top_bottom  # Middle 40%

    
    # Sort indices based on pc1 values
    sorted_indices = np.argsort(pc1_values)
    bottom_30_indices = sorted_indices[:num_top_bottom]  # Bottom 30% (least contributing)
    top_30_indices = sorted_indices[-num_top_bottom:]    # Top 30% (most contributing)
    middle_40_indices = sorted_indices[num_top_bottom:num_top_bottom + num_middle]  # Middle 40%


    # Initialize lists to hold data for plotting
    x = np.array([])
    y = np.array([])
    colors = []

    # Collect coordinates and colors for the top 30% and bottom 30% cells
    for idx in bottom_30_indices:
        x = np.concatenate((x, xpix[idx]))
        y = np.concatenate((y, ypix[idx]))
        colors.extend(['blue'] * len(xpix[idx]))  # Color bottom 30% in blue

    for idx in top_30_indices:
        x = np.concatenate((x, xpix[idx]))
        y = np.concatenate((y, ypix[idx]))
        colors.extend(['r'] * len(xpix[idx]))  # Color bottom 30% in blue

    for idx in middle_40_indices:
        x = np.concatenate((x, xpix[idx]))
        y = np.concatenate((y, ypix[idx]))
        colors.extend(['white'] * len(xpix[idx]))  # Color top 30% in red

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    ax.imshow(image, cmap='gray', interpolation='nearest')
    scatter = ax.scatter(x, y, c=colors, s=0.05, alpha=0.2)

    ax.set_title(title, fontsize=20)
    ax.axis('off')


        # Define a discrete colormap for the colorbar
    cmap = mcolors.ListedColormap(['blue', 'lightblue', 'red'])
    bounds = [0, 1, 2, 3]  # Define bounds for the three categories
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create colorbar with the three categories
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_ticks([0.5, 1.5, 2.5])
    cbar.set_ticklabels(['Bottom 30%', 'Middle 40%', 'Top 30%'])
    cbar.ax.tick_params(labelsize=15)
    
    # Save the plot
    plt.savefig(save_dir + f'{title}.png')

def create_blue_white_red_colormap():
    """
    Creates a custom blue-white-red colormap.
    Blue for low values, white for middle, and red for high values.
    """
    colors = [
        (0, "blue"),      # Start with blue for the lowest values
        (0.5, "white"),   # Middle point as white
        (1, "r")        # End with red for the highest values
    ]
    custom_cmap = LinearSegmentedColormap.from_list("blue_white_red", colors)
    return custom_cmap


def normalize_img(img):
    normalized_img = img - np.min(img) /np.max(img) - np.min(img)

    return normalized_img
        
def plot_nmfpx_blur_thr(loading_imgs, loading_imgs_filt, rois_auto, title='', save_dir=None):
    n_components = len(loading_imgs)

    # normalize motifs between 0-1
    loading_imgs = [normalize_img(img) for img in loading_imgs]

    all_im_list = [loading_imgs, loading_imgs_filt, rois_auto]

    fig, axs = plt.subplots(n_components, 3, figsize=(9, 3*n_components))
    
    plt.gca().set_facecolor('black') 

    for (i, im_list) in enumerate(all_im_list): # why does this need to be indented????
        for (j, im) in enumerate(im_list):
            axs[j,i].imshow(im, cmap='jet')
            plt.gca().set_facecolor('black') 

            axs[j,i].xaxis.set_ticklabels('') 
            axs[j,i].yaxis.set_ticklabels('') 
            axs[j,i].xaxis.set_ticks([]) 
            axs[j,i].yaxis.set_ticks([]) 

            if i == 0:
                axs[j,i].set_ylabel(f'NMF {j}', color='white')

            if j == 0:
                axs[j,0].set_title(f'Raw', color='white')
                axs[j,1].set_title(f'Smoothed', color='white')
                axs[j,2].set_title(f'Sm. and thr.', color='white')   

    # Set the title and axis labels to white
    for ax in axs.flatten():
        ax.set_title(ax.get_title(), color='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        # Set x and y tick labels to white
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_color('white')  
    
    if save_dir != None:
        print('Saving nmfpx_blur_thr.png ...')
        plt.savefig(f'{save_dir}/nmfpx_blur_thr_{title}.png', transparent=True)

def plot_nmf_thresholds(std_thresholds, save_dir=None):
    
    plt.plot(std_thresholds)
    plt.title('thresholds for rois_auto')
    plt.xlabel('NMF')
    plt.ylabel('2*std threshold value')
    plt.savefig(save_dir + '/std_thresholds.png', bbox_inches="tight")
    plt.show()        

def plot_nmf_t(nmf_t, gt_acts=None, plot_gt=False): #this is on pixels and plots the component on the fov

    fig, axs = plt.subplots(nmf_t.components_.shape[0], 1, figsize=(10, nmf_t.components_.shape[0]))

    for i in range(nmf_t.components_.shape[0]):
        axs[i].plot(nmf_t.components_[i, :]/np.max(nmf_t.components_[i, :]), c='C1')
        if plot_gt:
            axs[i].plot(gt_acts[i]/np.max(gt_acts[i]), c='C0')
            
        axs[i].axis('off')

    plt.show()
    
def plot_all_roi_areas(rois_auto, bins=None, alpha=0.5, save_dir=None):
    
    plt.figure()
    plt.hist(all_roi_areas, alpha=alpha)
    plt.title('Patch area (including borderline patches)')
    plt.xlabel('Area (um^2)')
    plt.ylabel('Count')
    plt.savefig(save_dir + 'all_areas.png')

    plt.show()

def plot_rois_colorcoded(iscell, xpix, ypix, image, c_cells, encircle=None, cmap='Reds', title='rois_colored', save_path=None):
    # Reformatting data to implement as a single scatter
    x = np.array([])
    y = np.array([])
    c = np.array([])

    for i in range(len(iscell)):
        x = np.concatenate((x, xpix[i]))
        y = np.concatenate((y, ypix[i]))

        c_cell = c_cells[i]
        n_px = len(ypix[i])
        c = np.concatenate((c, np.ones(n_px) * c_cell))

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)
    ax.imshow(image, cmap='gray', interpolation='nearest')        
    
    # Create the scatter plot
    scatter = ax.scatter(x, y, c=c, s=0.07, cmap=cmap, alpha=0.4)
    ax.set_title(title, fontsize=45)
    ax.axis('off')

    # Set up color normalization and color map
    norm = Normalize(vmin=np.min(c), vmax=np.max(c))
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # Create the colorbar associated with the axis
    cbar = fig.colorbar(scalar_map, ax=ax, cax=None, shrink=0.805)
    cbar.set_ticks([np.min(c), np.max(c)])
    cbar.set_ticklabels([f"{np.min(c):.2f}", f"{np.max(c):.2f}"])
    cbar.ax.tick_params(labelsize=20)

    if save_path:
        plt.savefig(save_path + f'{title}.png', bbox_inches='tight')
    plt.show()

# def plot_rois_colorcoded(iscell, xpix, ypix, image, c_cells, encircle=None, cmap='jet', title='rois_colored', save_path=None):
#     # reformatting data to implement as a single scatter
    
#     x = np.array([])
#     y = np.array([])
#     c = np.array([])
    
#     for i in range(len(iscell)):
#         x = np.concatenate((x, xpix[i]))
#         y = np.concatenate((y, ypix[i]))

#         c_cell = c_cells[i]
#         n_px = len(ypix[i])
#         c = np.concatenate((c, np.ones(n_px)*c_cell))

#     plt.figure(figsize=(10,10),dpi=200)
#     plt.imshow(image, cmap='Greys_r',  interpolation='nearest')        
#     plt.scatter(x, y, c=c,  s=0.01, cmap=cmap, alpha=0.8)
#     plt.title(title, fontsize=45)
#     plt.axis('off')
    
    
#     cmap = plt.get_cmap(cmap)
#     norm=Normalize(vmin=np.min(c),vmax=np.max(c))
#     scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    
#     # cbar = plt.colorbar()
#     cbar = plt.colorbar(scalar_map, shrink=0.805)
#     cbar.set_ticks([np.min(c), np.max(c)])
#     cbar.set_ticklabels([0, len(iscell)])
#     cbar.ax.tick_params(labelsize=20)

#     plt.savefig(save_path + f'{title}.png')
    
    
    # if type(save_path) == str:
    #     save_filename = os.path.join(save_path, title)
    #     plt.savefig(save_filename)

# updated version of above function -> plotting ROIs as contours instead of scatters

def plot_rois_contours(rois_auto, tiff_shape, save_dir=None):
    plt.figure(figsize=(10, 10))
    
    for i, roi in enumerate(reversed(rois_auto)):  
        # Create contour plot (without invalid 'fill' argument)
        contours = plt.contour(roi[::-1, :], colors=f'C{i}', linewidths=1.2)
        
        # Ensure contours exist before iterating
        if hasattr(contours, "collections"):
            for contour in contours.collections:  
                contour.set_edgecolor(f'C{i}')
                contour.set_facecolor("none")  # Prevents unwanted fills

    plt.title("ROI Contours")
    plt.savefig(save_dir + 'contours_patterns.png')
    plt.show()
    

def plot_rois_cont_colorcoded(xpix, ypix, image_array, c_cells=None, title='', cmap=None, save_path=None, multicomp_plot=False, normalize_colormap=True, sat_fact=2/3):

    ## start of function
    if c_cells is None:
        colors = ['C0'] * len(xpix) # set color to default if not specifiedS
    elif multicomp_plot:
        colors = [f'C{c}' for c in c_cells]
    else:
        colors = map_vec_to_colors(c_cells, normalize_colormap=True)

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(3,3),dpi=300)

    # Display the normalized image
    ax.imshow(image_array, cmap='gray', vmax=sat_fact*np.max(image_array)) # saturating a bit (Sofia: this to be changed)

    count = 0
    # Plot each ROI contour with a different color
    for roi_coords in zip(ypix, xpix):
        roi_patch = np.zeros_like(image_array, dtype=np.float32)
        roi_patch[roi_coords] = 1
        contours = ax.contour(roi_patch, levels=[0.5], colors=[colors[count]], alpha=0.5, linewidths=0.3)
        count+=1

    plt.axis('off')
    plt.title(title)

    if not c_cells is None:
        cmap = plt.get_cmap(cmap)
        norm = Normalize(vmin=0, vmax=len(xpix))
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(scalar_map, ax=ax, shrink=0.805)
        cbar.set_ticks([0, len(xpix)])
        cbar.set_ticklabels([0, len(xpix)])
        cbar.ax.tick_params(labelsize=20)

    plt.savefig(save_dir + f'{title}.png')

def map_vec_to_colors(vec, cmap=None, normalize_colormap=True):
    
    if normalize_colormap:    
        # Normalize the float values between 0 and 1
        normalized_values = np.array(vec)
        normalized_values = (normalized_values - np.min(normalized_values)) / (np.max(normalized_values) - np.min(normalized_values))
    else:
        normalized_values = np.array(vec)
    
    
    # Create the jet colormap
    cmap = plt.get_cmap(cmap)

    # Map the normalized values to RGB colors
    colors = cmap(normalized_values)
    return colors


def plot_rois_scatters(rois_auto, tiff_shape, save_dir=None):

    n, x, y = tiff_shape
    plt.figure(figsize=(10,10))

    plt.style.use('dark_background')

    #plt.ylim((-y, 0))
    #plt.xlim((0,x))
    plt.axis('off')

    for (i, roi) in enumerate(reversed(rois_auto)): # reversed to plot more obvious components first, idont know why jure named it t
        roi_scat = np.nonzero(roi)
        plt.scatter(roi_scat[1], -roi_scat[0]+roi.shape[0]-1, marker='s', s=65, alpha=0.8) # - is because of image processing convention
        plt.savefig(save_dir + 'scatter_patterns.png')


def plot_rois_overlay_except_rois(rois_auto, tiff_shape, save_dir=None):

    n, x, y = tiff_shape
    plt.figure(figsize=(10,10))

    plt.figure(figsize=(10,10))
    plt.gca().set_facecolor('black')

    for (i, roi) in enumerate(reversed(rois_auto)): # reversed to plot more obvious components first
        if i >= 4:
            roi_scat = np.nonzero(roi)
            plt.scatter(roi_scat[1], -roi_scat[0], marker='s', s=40, alpha=1) # - is because of image processing convention
            plt.ylim((-y, 0))
            plt.xlim((0,x))
            plt.axis('off')
    
    if save_dir != None:
        print('Saving rois_overlay.png ...')
        plt.savefig(f'{save_dir}/rois_overlay.png', transparent=True)
        plt.show()
        

def plot_roi_conts_largest(conts, tiff_shape, save_dir=None):
    
    # plots the largest contour in each ROI
    
    n, x, y = tiff_shape

    plt.figure(figsize=(10,10))
    plt.gca().set_facecolor('black') 

    #mean_image = np.mean(tiff, axis=0)
    # Plot the average image as the background
    #plt.imshow(mean_image, cmap='viridis', extent=(0, x, 0, y), origin='upper', alpha=0.5)
    
    if save_dir != None:
        print('Saving rois_conts_largest.png ...')
        plt.savefig(f'{save_dir}/rois_conts_largest.png')

    for (i, roi_cont) in enumerate(conts):

        plt.plot(roi_cont[0][:,1], roi_cont[0][:,0], linewidth=8, alpha=0.7)
        plt.gca().set_facecolor('black') 

        plt.ylim((y, 0))
        plt.xlim((0,x))
        plt.axis('off')
     
        plt.savefig(save_dir + 'rois_conts_largest.png', transparent=True)

    plt.show()   

def plot_roi_conts_largest_mean_img(conts, tiff_shape, mean_image, save_dir=None):
    
    # plots the largest contour in each ROI
    
    n, x, y = tiff_shape

    plt.figure(figsize=(10,10))
    plt.gca().set_facecolor('black') 

    #mean_image = np.mean(tiff, axis=0)
    # Plot the average image as the background
    plt.imshow(mean_image, cmap='viridis', extent=(0, x, 0, y), origin='upper', alpha=0.7)
    
    if save_dir != None:
        print('Saving rois_conts_largest.png ...')
        plt.savefig(f'{save_dir}/rois_conts_largest.png')

    for (i, roi_cont) in enumerate(conts):

        plt.plot(roi_cont[0][:,1], roi_cont[0][:,0], linewidth=8, alpha=0.9)
        plt.gca().set_facecolor('black') 

        plt.ylim((y, 0))
        plt.xlim((0,x))
        plt.axis('off')
     
    if save_dir is not None:
        print('Saving rois_conts_largest.png ...')
        plt.savefig(f'{save_dir}/rois_conts_largest_mean_img.png', transparent=True)

    plt.show()

def plot_roi_conts_largest_mean_img_except_rois(conts, tiff_shape, mean_image, save_dir=None):
    
    # plots the largest contour in each ROI
    
    n, x, y = tiff_shape

    plt.figure(figsize=(10,10))
    plt.gca().set_facecolor('black') 

    #mean_image = np.mean(tiff, axis=0)
    # Plot the average image as the background
    plt.imshow(mean_image, cmap='viridis', extent=(0, x, 0, y), origin='upper', alpha=0.7)
    
    if save_dir != None:
        print('Saving rois_conts_largest.png ...')
        plt.savefig(f'{save_dir}/rois_conts_largest.png')

    for (i, roi_cont) in enumerate(conts):
          if i >= 4:

              plt.plot(roi_cont[0][:,1], roi_cont[0][:,0], linewidth=8, alpha=0.9)
              plt.gca().set_facecolor('black') 

              plt.ylim((y, 0))
              plt.xlim((0,x))
              plt.axis('off')
     
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        print('Saving rois_conts_largest_except_rois.png ...')
        plt.savefig(f'{save_dir}/rois_conts_largest_mean_img_except_rois.png', transparent=True)

    plt.show()      
        
def plot_roi_area_hist(rois_auto, n_bins=10, resolution=1.2, save_dir=None):    #why jure put resolution=1.2? I change to fit all resolutions
    roi_areas = [np.sum(roi)*(resolution**2) for roi in rois_auto]
    plt.hist(roi_areas, n_bins);
    plt.title('patch area distribution')
    plt.xlabel('area (um^2)')
    
    if save_dir != None:
        print('Saving roi_area_hist.png ...')
        plt.savefig(f'{save_dir}/roi_area_hist.png')

def plot_nmf_corr_mat(nmf_matrix, filename, save_dir=None):
    plt.figure(figsize=(3,3), dpi=200)
    plt.gca().set_facecolor('black') 
    
    nmf_corr_mat = np.corrcoef(nmf_matrix) # correlation matrix of this, model.components_
    plt.imshow(nmf_corr_mat, vmin=-1, vmax=1, cmap='bwr')
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.tick_params(axis='y', colors='white')
    plt.title(f'{filename}')
    plt.xlabel('NMF component', color='white')
    plt.ylabel('NMF component', color='white')
    plt.xticks([], color='white')
    plt.yticks([], color='white')
    
    if save_dir != None:
        print(f'Saving {filename} ...')
        plt.savefig(f'{save_dir}/{filename}.png', transparent=True)

    plt.show()
    return nmf_corr_mat


def plot_nmf_temp_corr(H, save_dir=None):
    plt.figure(figsize=(3,3), dpi=200)
    plt.gca().set_facecolor('black')

    nmf_temp_corr = np.corrcoef(H)
    plt.imshow(nmf_temp_coeff, vmin=-1, vmax=1, cmap='bwr')
    cbar = plt.colorbar()
    plt.xlabel('NMF components')
    plt.ylabel('NMF components')
    plt.title('Components: Time correlations')

    if save_dir !=None:
        print('Saving nmf_temp_corr.png...:)')
        plt.savefig(f'{save_dir}/H_nmf_corr.png', transparent=True)

    return nmf_temp_corr


def plot_px_nmf_corr_except_rois(nmf_px, exclude_last=4, save_dir=None):
    # Select the components to include in the correlation matrix
    included_components = nmf_px.components_[:-exclude_last]

    plt.figure(figsize=(3, 3), dpi=200)
    plt.gca().set_facecolor('black')

    nmf_px_corrmat = np.corrcoef(included_components)  # correlation matrix with last components excluded
    plt.imshow(nmf_px_corrmat, vmin=0, vmax=1, cmap='bwr')
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.tick_params(axis='y', colors='white')
    plt.xlabel('NMF component', color='white')
    plt.ylabel('NMF component', color='white')
    plt.xticks([], color='white')
    plt.yticks([], color='white')

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        print('Saving px_nmf_corr_except.png ...')
        plt.savefig(f'{save_dir}/rx_nmf_corr_except.png', transparent=True)        

# def plot_roi_loading_time(rois_auto, loading_times, title='NOTE: the L and R do not necc. correspond'):

#     plt.style.use('dark_background')
    
#     n_components = len(rois_auto)

#     fig, axs = plt.subplots(n_components, 2, figsize=(5, n_components), width_ratios=[1, 5], dpi=200)
#     plt.suptitle(title, fontsize=10)

#     for (i, loading_time) in enumerate(loading_times):
#         axs[i,0].imshow(rois_auto[i], cmap='jet') #was gray 
#         axs[i,0].xaxis.set_ticklabels('') 
#         axs[i,0].yaxis.set_ticklabels('') 
#         axs[i,0].xaxis.set_ticks([]) 
#         axs[i,0].yaxis.set_ticks([]) 

#         axs[i,1].plot(loading_time, c='cyan') #was gray
#         axs[i,1].axis('off')

#         if i == 0:
#             axs[i,0].set_title('PX component', fontsize=7)
#             axs[i,1].set_title('T component (activation of PX component over time)', fontsize=7)
    
#     plt.show()

def plot_roi_loading_time(rois_auto, H, title='space & time comps', save_dir=None):
    
    n_components = len(rois_auto) 

    fig, axs = plt.subplots(len(rois_auto), 2, figsize=(5, len(rois_auto)), width_ratios=[1, 4], dpi=200)
    plt.suptitle(title, fontsize=10)

    for (i, H) in enumerate(H):
        axs[i,0].imshow(rois_auto[i], cmap='plasma') #was gray 
        axs[i,0].xaxis.set_ticklabels('') 
        axs[i,0].yaxis.set_ticklabels('') 
        axs[i,0].xaxis.set_ticks([]) 
        axs[i,0].yaxis.set_ticks([]) 

        axs[i,1].plot(H.T[:2000], c='b') #was gray
        axs[i,1].axis('off')

        if i == 0:
            axs[i,0].set_title('spatial component', fontsize=7)
            axs[i,1].set_title('temporal component', fontsize=7)
    plt.savefig(save_dir + ('space_and_time_comps.png'))
    plt.show()

#plot space corr mat (of already filtered ( see filter_w_mat in compute.py) rois
def plot_nmf_space_corr(nmf_space_corr, save_dir=None):
    # index the nmf_space_corr matrix to include only filtered ROIs
    
    plt.imshow(nmf_space_corr, vmin=0, vmax=1, cmap='bwr')
    plt.colorbar()
    plt.xlabel('NMF components')
    plt.ylabel('NMF components')
    plt.title('Components: Space correlations')
    
    # Save the figure to the specified directory
    plt.savefig(save_dir + '/corr_mat_space_filtered.png', bbox_inches="tight")
    
    plt.show()

def plot_all_roi_areas(rois_auto, bins=None, alpha=0.5, save_dir=None):
    
    plt.figure()
    plt.hist(all_roi_areas, alpha=alpha)
    plt.title('Patch area (including borderline patches)')
    plt.xlabel('Area (um^2)')
    plt.ylabel('Count')

    plt.show()

def plot_compare_px_truth_nmf(sort_ind_px, truth_pxs, loading_imgs):
    n_components = len(loading_imgs)
    _, axs = plt.subplots(n_components, 2, dpi=100, figsize=(4*2, 4*n_components))

    for i in range(n_components):
        axs[i,0].imshow(loading_imgs[i], cmap='gray')
        axs[i,1].imshow(truth_pxs[sort_ind_px[i]], cmap='gray')
        axs[i,0].axis('off')
        axs[i,1].axis('off')
        
    axs[0,0].set_title('pxNMF component')
    axs[0,1].set_title('Closest ground truth')
    plt.show()

def plot_cv_opt(params, train_err, test_err):
    train_err_np = np.array([train_err[i][1] for i in range(len(train_err))])
    test_err_np = np.array([test_err[i][1] for i in range(len(test_err))])

    n_nmf_opt = np.argmin(test_err_np)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5), dpi=200)
    ax.plot(np.arange(1,len(train_err_np[1:])+1), train_err_np[1:], 'o-', label='Train Data') # without 'average' component
    ax.plot(np.arange(1,len(test_err_np[1:])+1), test_err_np[1:], 'o-', label='Test Data') # without 'average' component
    ax.set_ylabel('MSE')
    ax.set_xlabel('r')
    ax.axvline(params.n_patches, color='k', dashes=[1,1.5], label='ground truth r') # +1 because of mean component
    ax.axvline(n_nmf_opt, color='grey', dashes=[1,3], label='cvNMF estimate r')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    fig.tight_layout()

def plot_rois_auto_centroids(rois_auto, rois_auto_centroids, save_dir):
    for i,roi in enumerate(rois_auto):
        for centroid in rois_auto_centroids:
            plt.imshow(roi, cmap='jet')
            plt.scatter(rois_auto_centroids[i][1], rois_auto_centroids[i][0])
        plt.show()    

def plot_compare_t_truth_nmf(sort_ind_t, truth_ts, loading_times):
    
    n_components=len(loading_times)
    _, axs = plt.subplots(n_components, dpi=300, figsize=(8, 1 * n_components))

    for i in range(n_components):
        axs[i].plot(zscore(loading_times[i]), color='grey', label='NMF')
        axs[i].plot(zscore(truth_ts[sort_ind_t[i]]), label='closest true')
        axs[i].axis('off')

    axs[0].legend(fontsize=5)
    plt.show()

def animate_movie(movie, sim_path):
    fig = plt.figure()
    im = plt.imshow(-movie[0,:,:], vmin=-np.max(movie), vmax=0, cmap='Greys')
    plt.axis('off')

    def ani_fun(i):
        im.set_array(-movie[i,:,:])
        return([im])

    anim = animation.FuncAnimation(fig, ani_fun,
                                   frames=movie.shape[0], interval=20, blit=True)

    anim.save(f'{sim_path}/anim.gif', fps=30)

def plot_dist_bin_ious(bins_iou, means_iou, stds_iou, save_dir=None):

    bin_cents_iou = (bins_iou[:-1] + bins_iou[1:]) / 2  # Midpoints of bins
    
    # Create the plot
    plt.figure(figsize=(5, 5), dpi=200)
    plt.errorbar(bin_cents_iou, means_iou, yerr=stds_iou, fmt='o-', color='blue', 
                 label='Mean IoU with ±1 STD', markersize=5, capsize=5)
    
    plt.title("Mean IoU - centroid istance", fontsize=20)
    plt.xlabel("distance between pattern centroids", fontsize=15)
    plt.ylabel("Mean IoU", fontsize=15)
    plt.xticks(bins_iou)  # Optional: Adjust x-ticks for better visibility
    plt.ylim(0, 1)  # Assuming IoU values range from 0 to 1
    plt.legend()
    ax= plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_dir + 'pattern_iou_centroid_dist.png', bbox_inches="tight")

    
    plt.show()

    #CYRRENTLY REPLACED BY FUNCTION UNDER (homogenise with upper, todo)
# def plot_dist_bin_h_corrs(bins_h_corr, means_h_corr, stds_h_corr, save_dir=None):

#     bin_cents_h_corr = (bins_h_corr[:-1] + bins_h_corr[1:]) / 2  # Midpoints of bins
    
#     # Create the plot
#     plt.figure(figsize=(10, 6))
#     plt.errorbar(bin_cents_h_corr, means_h_corr, yerr=stds_h_corr, fmt='o-', color='blue', 
#                  label='Mean corr with ±1 STD', markersize=5, capsize=5)
    
#     plt.title("Pattern correlation - centroid istance", fontsize=16)
#     plt.xlabel("distance between pattern centroids", fontsize=14)
#     plt.ylabel("Mean pattern correlation", fontsize=14)
#     plt.xticks(bins_h_corr)  # Optional: Adjust x-ticks for better visibility
#     plt.ylim(0, 1)  # Assuming IoU values range from 0 to 1
#     plt.legend()
#     ax= plt.gca()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.savefig(save_dir + 'pattern_correl_centroid_dist.png', bbox_inches="tight")

    
#     plt.show()

def plot_bin_dist_corr(bin_means_corr, bin_stds_corr, bin_cents_corr, save_dir=None):
    
    plt.figure(figsize=(5, 5), dpi=200)
    plt.plot(bin_cents_corr, bin_means_corr, marker='o', color='blue', label='mean pattern corr')
    # Plot settings
    plt.title("Mean pattern correlation-centroid distance", fontsize=20)
    plt.xlabel("distance (um)", fontsize=15)
    plt.ylabel("mean corr_ceoff", fontsize=15)
    # Fill between the error margins
    plt.fill_between(bin_cents_corr, 
                     bin_means_corr - bin_stds_corr, 
                     bin_means_corr + bin_stds_corr, 
                     color='blue', alpha=0.3, label='±1 std')
        
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Add the legend
    plt.legend()
    # Save the figure
    plt.savefig(save_dir + 'pattern_hcorr_centroid_dist_avg.png', bbox_inches="tight")
    # Display the plot
    plt.show()

#EXPLORATORY ANALYSIS

def plot_acts(data, title='', cmap=None, style='light', save_dir=None, save=True):
    if style == 'dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
        
    _, axs = plt.subplots(2, 1, figsize=(10,4), dpi = 400, gridspec_kw={'height_ratios': [3, 1]})

    data_min = np.min(data)
    data_max = np.max(data)

    #enhance contrast by minmax scaling if the range of values is too small
    if data_max - data_min < 1e-5:
        data_rescaled = (data - data_min) / (data_max - data_min)
    
    else:
        data_rescaled = data  # If constant, no rescaling

    vmin = np.percentile(data_rescaled, 1)  # Dynamic percentiles
    vmax = np.percentile(data_rescaled, 99)
    
    # # Use a logarithmic scale to emphasize smaller differences (optional)
    # data_log_scaled = np.log1p(data_clipped - vmin)  # Use log1p for log scaling

    print(f"vmin: {vmin}, vmax: {vmax}, min value: {data_min}, max value: {data_max}")  # Debug info
    
    axs[0].imshow(data_rescaled.T, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax,alpha=0.8) # z scored for 0.001 data.mean()+ 2*data.std() 
    axs[0].xaxis.set_visible(False)
    axs[0].set_ylabel('roi id')
 

    axs[1].plot(np.mean(data, axis=1), label='avg') # mean across px 
    axs[1].set_xlim(0, data.shape[0])
    axs[1].set_ylabel('avg.')
    axs[1].set_xlabel('time (frames)')
#    axs[1].set_ylim([(np.min(data) - 0.5), np.max(data)]) # +0.5 for better visuals

    if save:
        plt.savefig(save_dir + f'/acts_{title}.png')
        
    plt.tight_layout()
    plt.show()

def plot_acts_sorted(tstamps, data, embedding, bottom_plot='Mean', add_plot=None, component=0, tmin=0, tmax=None, title='embedding', save_dir=None, vmax=1.96):
        
    # getting sorting indices
    temp = np.argsort(embedding[:,component])
    
    plot_acts(tstamps, data[temp,:], bottom_plot=bottom_plot, add_plot=add_plot, tmin=tmin, tmax=tmax, title=title, save_dir=save_dir, vmax=vmax)
    
    
    if type(save_path) == str:
        save_filename = os.path.join(save_path, title)
        plt.savefig(save_filename, transparent=True)

def compute_correlations_for_roi_10pxs(roi_mask, data, n_pixels=10):
    """
    Computes correlations within a given ROI and between inside and outside the ROI.

    Parameters:
    - roi_mask: Binary mask for the ROI, where 1 indicates pixels inside the ROI.
    - data: 2D array of shape (n_timepoints, n_pixels) containing pixel time series data.
    - n_pixels: Number of pixels to randomly select from inside and outside the ROI (default: 10) (this is used for visualisation).

    Returns:
    - corr_within: Correlation matrix for selected pixels within the ROI.
    - corr_inside_outside_vals: Correlation matrix between selected inside and outside pixels.
    - inside_idx: Indices of selected inside pixels.
    - outside_idx: Indices of selected outside pixels.
    """
    
    # Flatten the roi_mask to match the shape of the data
    roi_mask_flat = roi_mask.flatten()
    
    # Get indices of the pixels inside and outside the ROI
    inside_pixels_idx = np.argwhere(roi_mask_flat == 1).flatten()  # Inside ROI
    outside_pixels_idx = np.argwhere(roi_mask_flat == 0).flatten()  # Outside ROI

    # Check if there are enough outside pixels for sampling
    if len(outside_pixels_idx) < n_pixels or len(inside_pixels_idx) < n_pixels:
        print("Skipping ROI as there are insufficient pixels.")
        return None, None, None, None  # Skip this ROI if not enough pixels

    # Randomly select n_pixels inside and outside the ROI
    inside_idx = np.random.choice(inside_pixels_idx, n_pixels, replace=False)
    outside_idx = np.random.choice(outside_pixels_idx, n_pixels, replace=False)
    
    # Extract the time series of the selected pixels from the data
    inside_pixel_traces = data[:, inside_idx]  # Shape: (n_timepoints, n_pixels)
    outside_pixel_traces = data[:, outside_idx]  # Shape: (n_timepoints, n_pixels)
    
    # Compute pairwise correlation for selected inside pixels (within-ROI correlation)
    corr_within = np.corrcoef(inside_pixel_traces.T)  # Shape: (n_pixels, n_pixels)
    
    # Combine traces for inside and outside pixels
    combined_traces = np.hstack((inside_pixel_traces, outside_pixel_traces))  # Shape: (n_timepoints, 2*n_pixels)
    
    # Compute pairwise correlation between inside and outside pixels
    corr_inside_outside = np.corrcoef(combined_traces.T)  # Shape: (2*n_pixels, 2*n_pixels)
    
    # Extract only the correlation values between inside and outside pixels
    corr_inside_outside_vals = corr_inside_outside[:n_pixels, n_pixels:]  # Shape: (n_pixels, n_pixels)
    
    return corr_within, corr_inside_outside_vals, inside_idx, outside_idx

def compute_correlations_for_roi(roi_mask, data, n_pixels=None):
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
    inside_idx = inside_pixels_idx  # All pixels inside the ROI
    
    # Randomly select n_pixels outside the ROI
    outside_idx = np.random.choice(outside_pixels_idx, n_pixels, replace=False)
    
    # Extract the time series of the selected pixels from the data
    inside_pixel_traces = data[:, inside_idx]  # Shape: (n_timepoints, n_pixels)
    outside_pixel_traces = data[:, outside_idx]  # Shape: (n_timepoints, n_pixels)
    
    # Compute pairwise correlation for selected inside pixels (within-ROI correlation)
    corr_within = np.corrcoef(inside_pixel_traces.T)  # Shape: (n_pixels, n_pixels)
    
    # Combine traces for inside and outside pixels
    combined_traces = np.hstack((inside_pixel_traces, outside_pixel_traces))  # Shape: (n_timepoints, 2*n_pixels)
    
    # Compute pairwise correlation between inside and outside pixels
    corr_inside_outside = np.corrcoef(combined_traces.T)  # Shape: (2*n_pixels, 2*n_pixels)
    
    # Extract only the correlation values between inside and outside pixels
    corr_inside_outside_vals = corr_inside_outside[:n_pixels, n_pixels:]  # Shape: (n_pixels, n_pixels)
    
    return corr_within, corr_inside_outside_vals, inside_idx, outside_idx
