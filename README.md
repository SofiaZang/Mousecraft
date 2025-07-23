# MouseCraft

A very fun GUI & splendid GUI used for validating behavior annotations.

## Features

- Load video and motion enegry or classification labels
- Validate, edit, and add motion events
- Export results in multiple formats (MF, HF) (.npy, .csv) & plot some performance statistics 

## Installation

### Using Conda (Recommended)

For local installation (tried in Windows, to be tried in other os's)

1. Install an Anaconda distribution of Python or miniconda (miniforge, if anaconda wont work try via miniforge same commands) & GitBash 

2.  Open an anaconda prompt / command prompt with conda for python 3 in the path.
Navigate where you want the mousecraft to live.

3. In the command prompt git clone https://github.com/SofiaZang/mouse_motion_analysis.git mousecraft repository. You can also get the cloning link under green code button here https://github.com/SofiaZang/mouse_motion_analysis .
If this does not work, do the same using, git clone in GitBash.

4. Do cd mousecraft to go into the root folder of mousecraft gui. Then create a new environment with conda env create -f environment.yml . Friendly tip: Make sure to have at least 1-2 GB free on your main disk. This creates the mousecraft envinroment, which you can then activate running: conda activate mousecraft

Once env is activated do: pip install -e . 
This installs mousecraft to your local repository!

Now run python -m mousecraft and you're all set. 
You can also try just: mousecraft but in case this won't work use the abobe command.

### Using pip only

Another way to install mousecraft via pip is:

pip install git+https://github.com/yourusername/mousecraft.git in the same directory where you want the mousecraft repository to live in.

Everytime you want to use mousecraft, you have to first run: conda activate mousecraft and then: mousecraft or python -m mousecraft 

![alt text](image-1.png)
 
Friendly tip #2: Do not attempt to exit, it won't work :) 

![alt text](image.png)

## Dependencies

dependencies:
  - python=3.9
  - pyqt
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - scipy
  - tqdm
  - opencv-python
  - pyqt
  - r-diptest 
  - pip
  - pip:
      - colorama==0.4.6
      - dill==0.3.6
      - et_xmlfile==2.0.0
      - isort==5.12.0
      - lazy-object-proxy==1.9.0
      - mccabe==0.7.0
      - pylint==2.16.1
      - PySimpleGUI==5.0.8.3
      - python-dateutil==2.9.0.post0
      - pytz==2025.2
      - six==1.17.0
      - tomlkit==0.11.6
      - tzdata==2025.2
      - wrapt==1.14.1
      - jupyter
      - tifffile
      - scikit-image
      - seaborn
      - statsmodels


### Inputs
Input formats: 

Mousecraft accepts any .npy file of the motion signal you have eg. motion_energy.npy or if you have run the automatic classification notebook then the annotation labels saves as 
gui_lables.csv or .xls under the folder 'mouse_motion_average' under the dataset you analysed. This file will also contain the motion signal value for each frame of the recording so will automatically load and display both motion signal and proposed annotations.

You also need to load an .avi maybe .mp4 or other formats (TO DO: all video formats)

Directory structure

Mousecraft looks for all inputs in the folder you define. 

See examples in this notebook https://colab.research.google.com/drive/1Sfts_onqzadvvcDXfVnDBEA_10ic7_YI?usp=sharing

Tips: For the notebook to run you need the motion_energy.npy (or other motion_signal.npy) or the .tiff file of the behavior you want to annotate if you have not computed the motion energy and it will be automatically computed in the notebook run.
 
## License

[Your License Here]
