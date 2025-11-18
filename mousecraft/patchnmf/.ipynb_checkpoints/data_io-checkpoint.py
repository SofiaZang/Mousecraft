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

# read and load all parameters for all data from data_params.xlsx 

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
