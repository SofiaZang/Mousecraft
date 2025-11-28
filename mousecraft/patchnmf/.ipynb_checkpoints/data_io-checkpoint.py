import os 
import numpy as np
from skimage import io
from skimage.util import img_as_uint
import pandas as pd
from IPython.display import display


def get_tiff(ds):
    tiff_all = []
    for (i, ti) in enumerate(os.listdir(f'data/{ds}/downsampled_tiff')):
        print(ti)
        
        if i == 0:
            tiff = io.imread(f'data/{ds}/downsampled_tiff/{ti}',  plugin='pil') # initialise tiff
            tiff = img_as_uint(tiff) 
        else:
            tiff_i = io.imread(f'data/{ds}/downsampled_tiff/{ti}',  plugin='pil')
            tiff_i = img_as_uint(tiff_i)
            tiff = np.concatenate((tiff, tiff_i))
            print(tiff_i.shape)

    # making sure smallest value of tiff is zero - just a linear transform, shouldn't affect NMF ? 
    tiff -= np.min(tiff)
    print(f'Shape of video: {tiff.shape}') 

    return tiff
    
# load as time, rois
def get_numpy(ds):
    
    npy_path = os.path.join(f'data/{ds}', 'preprocessed_tiff.npy')
    data = np.load(npy_path)
    print(f' Data shape : {data.shape}')
    return data 

# load as time, rois
def get_numpy_exploratory(ds):
    
    npy_path = os.path.join(f'data/{ds}', 'preprocessed_tiff_no_thresholding.npy')
    data = np.load(npy_path)
    print(f' Data shape : {data.shape}')
    return data 
    
def get_save_path(ds):
    save_path = os.getcwd() + '/data/' + ds + '/patch_sz/'

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    print('SavePath: ', save_path)
    return save_path

def percentage(percentage, whole):
    return (percentage * whole)/100

def average_frames(data, avg_block=None):
    # Check if the time dimension is divisible by the avg_block
    if data.shape[0] % avg_block != 0:
        raise ValueError(f"Data length {data.shape[0]} is not divisible by avg_block {avg_block}")
    
    # If data is 1D (e.g., X or Y movement values separately)
    if len(data.shape) == 1:  
        # Reshape the data to (new_time, avg_block)
        grouped = data.reshape(-1, avg_block)
    else:  
        # For 2D data (e.g., X and Y movement values together), reshape it to (new_time, avg_block, 2)
        grouped = data.reshape(-1, avg_block, data.shape[1])

    # Average along the second dimension (axis=1) to reduce the time dimension
    avg_data = np.mean(grouped, axis=1)
    print(f' Congrats! New shape is {len(avg_data)}')
    
    return avg_data

def pad(var):
    var = np.concatenate(([var[0]], var))
    print(f' Congrats! New shape is {len(var)}')
    return var 

def smooth_with_gaussian(var, sigma=None):
    smoothed_var = gaussian_filter1d(var, sigma=sigma)

    return smoothed_var

def export_conts_fiji(conts, save_path):
    os.makedirs(save_path + 'roi_to_fiji', exist_ok=True)
    # writing to text file (for FIJI export)
    for (i, roi_cont) in enumerate(conts):
        
        with open(save_path + f'roi_to_fiji/nmf{i+1}_roi.txt', 'w') as f:
            for j in range(len(roi_cont[0])):
                f.write(f'{roi_cont[0][j,1]}    {roi_cont[0][j,0]}\n')
                #does not output, to be fixed (sofia)
#styling the parameter table
def style_table(styler):
    styler.set_table_styles([
        {'selector': 'thead th', 
         'props': [('font-weight', 'bold'), ('background-color', 'green'), ('color', 'white'), ('text-align', 'center')]},  # Green header with white text
        {'selector': 'tbody tr:nth-child(odd) td', 
         'props': [('background-color', 'black'), ('color', 'white'), ('text-align', 'center')]},  # Black background with white text for odd rows
        {'selector': 'tbody tr:nth-child(even) td', 
         'props': [('background-color', 'white'), ('color', 'black'), ('text-align', 'center')]}  # White background with black text for even rows
    ])
    return styler

# styling the parameter table
def hide_index(styler):
    styler.set_table_styles({
        '': [
            {'selector': '.row0', 'props': [('display', 'none')]}
        ]
    })
    return styler

def read_params(pixelnmf_params, ds):

    # Load Excel file containing parameters
    df_params = pd.read_excel(pixelnmf_params)
    
    # define dataset 
    pd.options.display.float_format = '{:.3f}'.format
    
    params = df_params[df_params['ds'] == ds].iloc[0]
    
    # Assign parameters to variables
    blur_std = params['blur_std']
    downs_fact = params['downs_fact']
    res_orig = params['res_orig']
    x_axis = params['x_axis']
    y_axis = params['y_axis']
    original_FOV = params['original_FOV']
    framerate = params['framerate'] # (Hz)
    duration = params['duration'] #(s)
    n_components_elbow = params['n_components_elbow'] #cv_nmf output
    n_components_min = params['n_components_min'] #cv_nmf output

    # Calculate dependent parameters
    resolution = original_FOV / x_axis
    time_per_frame = 1 / framerate
        
    # Print table of parameters
    params_table = pd.DataFrame({
        'Parameter': ['blur_std', 'downs_fact', 'res_orig', 'x_axis', 'y_axis', 'original_FOV', 'framerate', 'duration', 'n_components_elbow', 'n_components_min', 'resolution', 'time_per_frame'],
        'Value': [blur_std, downs_fact, res_orig, x_axis, y_axis, original_FOV, framerate, duration, n_components_elbow, n_components_min, resolution, time_per_frame]
    })
    
    styled_params_table = params_table.style.pipe(style_table).set_properties(**{'text-align': 'center'})

    # Display the styled table
    display(styled_params_table)

    return params_table, blur_std, downs_fact, res_orig, x_axis, y_axis, original_FOV, framerate, duration, n_components_elbow, n_components_min, resolution, time_per_frame
  
