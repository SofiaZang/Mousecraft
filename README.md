# Mousecraft                                                     <img width="360" height="328" alt="image" src="https://github.com/user-attachments/assets/85e1e2fc-9308-42e5-8c03-8616f15092d6" />
                                                                                                                                                                                                                                       
A human & mouse-friendly GUI used for validating or creating behavior annotations.

## Features

- Load video(s) and motion signal or annotation outputs
- Validate, edit, and add motion events
- Export results in multiple formats (.npy, .csv) & plot some performance statistics 

## Installation

### Using Conda (Recommended)

For local installation (tried in Windows, to be tried in other os's) 

1. Install Anaconda (https://www.anaconda.com/download) or Miniconda &  install also: GitBash (https://git-scm.com/downloads).
Tip: If Anaconda/Miniconda gives issues, try installing Miniforge and do the procedure there.
   
2.  Open an git bash and navigate to the folder where you want Mousecraft to be installed.
  
3. Clone the repository to your computer pasting: ```git clone https://github.com/SofiaZang/Mousecraft.git``` mousecraft repository. You can also get the cloning link under green code button here https://github.com/SofiaZang/Mousecraft.git
   
4. Open an anaconda prompt and navigate to the cloned root folder of the MouseCraft GUI ``cd Mousecraft``. 

5. Create the MouseCraft envirnoment with ``conda env create -f environment.yml``
   
If this is slow, try: ``conda env create -f environment.yml -c conda-forge --strict-channel-priority``
  
6. Activate MouseCraft environment with ``conda activate mousecraft`` . 
   
7. Then do: ``pip install -e .`` 
This installs mousecraft package to your local repository!

8. Open the GUI: Run ``python -m mousecraft`` and you're all set! 

### Using pip only

Another way to install the mousecraft package via pip is:

1. ``pip install git+https://github.com/SofiaZang/Mousecraft.git`` after navigateing to the directory where you want the MouseCraft to be installed in.
   
### After Installation:
Everytime you want to use Mousecraft, you have to first run ``conda activate mousecraft`` and then: ``mousecraft`` or ``python -m mousecraft`` 

<img width="1357" height="987" alt="mousecraft_starter" src="https://github.com/user-attachments/assets/456d843b-7a18-4dda-8eae-03704c6cb3cc" />

Friendly tip #2: Do not attempt to exit, it won't work :) 

<img width="1912" height="993" alt="image" src="https://github.com/user-attachments/assets/d0daebf8-7641-4566-a66d-78084d17992c" />
<img width="1916" height="1016" alt="Mousecraft_new_v" src="https://github.com/user-attachments/assets/cce28603-d2d5-4f77-bab2-28f0fc4f013c" />


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
  - opencv
  - pyqt
  - pip
  - pip:
      - colorama==0.4.6
      - dill==0.3.6
      - et_xmlfile==2.0.0
      - isort==5.12.0
      - lazy-object-proxy==1.9.0
      - mccabe==0.7.0
      - pylint==2.16.1
      - PySimpleGUI
      - python-dateutil
      - pytz
      - six==1.17.0
      - tomlkit==0.11.6
      - tzdata
      - wrapt==1.14.1
      - jupyter
      - tifffile
      - scikit-image
      - seaborn
      - statsmodels


## Inputs

Mousecraft accepts any .npy file (or .json) of the motion signal you have eg. motion_energy.npy (or motion_energy.json) or if you have run the automatic classification notebook then the annotation labels saved as gui_lables.csv or .xlsx under the folder 'mouse_motion_average' under the dataset folder you analysed. This file will also contain the motion signal value for each frame of the recording so will automatically load and display both motion signal and proposed annotations. If you have used another method to classify your signal, you can still load your proposed annotations as long as they follow the same matrxi structure with columns: 'frame_idx, signal, event_type 1, event_type 2, event_type 3, .. event_type_n'

<img width="730" height="327" alt="image" src="https://github.com/user-attachments/assets/8581b1f9-45bb-478c-b755-9ef1b168ac83" />

<img width="279" height="118" alt="image" src="https://github.com/user-attachments/assets/ccd656b2-f914-4919-b33a-12522b038588" />

You also need to load a movie in one of the following .avi, .mp4, .mkv, .mov, .tiff formats (tip: downsample the movie or compress before loading, else may be too heavy and crash)

See examples of the classification algorithm mousecraft uses in this notebook https://colab.research.google.com/drive/1Sfts_onqzadvvcDXfVnDBEA_10ic7_YI?usp=sharing

Tips: For the notebook to run you need the motion_energy.npy (or other motion_signal.npy) or the .tiff file of the behavior you want to annotate if you have not computed the motion energy and it will be automatically computed in the notebook run.

## Using the GUI

<img width="1130" height="379" alt="image" src="https://github.com/user-attachments/assets/2e8a90bc-dff4-409f-a51f-37182ad4e677" />

The interface is divided into two main panels:

Left Panel: Video playback and control
Right Panel: Motion energy timeline, event navigation, and annotation tools

### Video Display & Controls (Left Panel)

#### Video Display
Main video window: Displays the loaded video of the mouse behavior. 
Shows frame-by-frame playback and updates when navigating events.

#### Load Video

Button: Load Video and Add a Camera
Opens a file dialog to select a video file (.avi, .mp4, .mov, .tiff, etc.).
You can load a second video and adjust both size. 

#### Playback Controls

Play ▶: Starts playing the video from the current frame at the defined FPS.
Pause ⏸: Pauses playback but keeps the current frame in view.
Stop ⏹: Stops playback and resets to frame 0.

#### FPS Control
Textbox (FPS): Defines frames per second for playback speed (e.g., enter 5 for 5 fps). By default the movie will play at 1 fps.

#### Frame Slider
Allows manual scrubbing through frames of the loaded video. You can also just write the frame you want to check and you will be teleported there.

### Onset Status and Performance Metrics

#### Onset Status
Displays whether the current frame corresponds to an annotated onset (e.g., Active/Twitch).

#### Performance Score
Text box displaying annotation statistics. Accepted = 1, Rejected = -1, Edited = 0.5, Pending = 0, Manually-added = 0. 
This can be adjusted (in the main gui code).

### Save & Export

#### Export Folder
Field + Button (...): Choose where to save annotations and outputs.

Mousecraft also saves automatically every 20 min if you have performed at least one action within these 20 min. First autosave asks you for the output path and then  its set for the following saves. If you close mousecraft before having finished the validation, you will see the curernt progress saved as _pending files. Once you complete and no pending events remain, the output files will overwrite any pending ones and be saved as _final.

Tip: In case you have made some mistake and an event overlaps with another, an error message appears before save.

### Motion Energy Timeline (Right Panel)

#### Timeline Plot
Displays the motion energy trace over frames.

#### Load An Input
Loads a precomputed motion energy file (.csv, .xlsx, .npy, .json) or a classification (Active/Twitch events) for review and editing.
Annotated events appear as colored spans: Yellow: Active events, Purple: Twitch events, Green/Red/Orange dots: Validation status (Accepted, Rejected, Edited).

#### Add A Second Input

Do the same as Load An Input. The second input appears under the first one. 

#### Zoom Controls
🔍+ / 🔍-: Zoom in and out of the timeline.
Reset Zoom: Resets timeline view to full length.

### Event Navigation & Validation

#### Dropdown filters:
Event Type: Filter events by type (All, Active, Twitch). This allows you to navigate around only the chosen events.

Event Status: Filter events by validation status (All, Accepted, Rejected, Edited, Manually Added, Pending). Same as above, navigate only in the status of interest events.

#### Navigation Buttons
← Prev Onset / Next Onset →: Move between annotated events based on current filters.

#### Validation Controls

✓ Accept: Marks event as accepted.

✎ Edit: Allows manual editing of event onset/offset frames. If you only change the offset, a message will appear asking you if you accept the given defined onset or want to change it also.

✗ Reject: Marks event as rejected. This event then disappears.

Change Type: Changes an event’s type (e.g., Twitch → Active).

↩ Undo: Reverts the last validation action.

### Manual Event Addition

#### Event Type:
Dropdown to choose event type (Twitch, Active, Complex) for manual annotation.

#### Add a Type : 
Open a window. You can name the type and choose a color for this type. This tape will appears in every dropdown menu (event type of manual edition and onset navigation). 

#### Onset / Offset Frame
Spinboxes to manually enter frame numbers for onset and offset of new events.

#### Set Current Frame
Buttons to set the current video frame as onset or offset.

#### Add Event
Adds the new event to the annotation timeline. If the added or edited event overlaps fully with another event, the automatic event will be rejected and this new addition kept.

#### Edit Threshold (frames)
This matters for accuracy tracking. Sets the frame tolerance used when classifying edited events (e.g., 5 means ±5 frames from original is considered “minor edit” the status is edited but the dot is Green and the score is +1). This can be adjusted per user. (If for example the signal is averaged 5 times, we keep the tolerance at 5 frames).

## Mousecraft outputs

### Main outputs:

<img width="730" height="327" alt="image" src="https://github.com/user-attachments/assets/eb89236a-7453-4f5e-a3dd-8c4bf1db6041" />

Mousecraft currently outputs 2 main outputs in .npy and .csv format:

#### validation_HF (Human Friendly) 

<img width="400" height="223" alt="image" src="https://github.com/user-attachments/assets/b12a4357-8e9a-4809-9985-815b4dfe9f56" />

#### validation_MF (Machine Friendly)

<img width="613" height="509" alt="image" src="https://github.com/user-attachments/assets/dac8431f-fa92-4e61-855c-a574f600e2db" />

Same information but each line is a frame (in same format as input .csv) and this is the input when you continue validation from _pending.

#### Other outputs:

and also 2 .pngs and .json that showcase the overall performance of mousecraft validation

<img width="1359" height="1011" alt="validation_status_pie_final" src="https://github.com/user-attachments/assets/c936c207-6bcf-4b6c-a2ae-f25b42b73d14" />
<img width="4472" height="1676" alt="validation_comparison_plot_final" src="https://github.com/user-attachments/assets/6ead7ed5-d7c9-4074-a92d-6bbec33706d5" />
<img width="506" height="300" alt="image" src="https://github.com/user-attachments/assets/3e00e065-4ced-4398-ae6f-fbb022ae08c9" />

## License

MIT 
