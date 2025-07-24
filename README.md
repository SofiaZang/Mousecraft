# MouseCraft                                                     <img width="360" height="328" alt="image" src="https://github.com/user-attachments/assets/85e1e2fc-9308-42e5-8c03-8616f15092d6" />
                                                                                                                                                                                                                                         
A fun & splendid GUI used for validating behavior annotations.

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

3. In the command prompt ```git clone https://github.com/SofiaZang/Mousecraft.git``` mousecraft repository. You can also get the cloning link under green code button here https://github.com/SofiaZang/Mousecraft.git.
If this does not work, do the git clone in GitBash.

4. Do ``cd Mousecraft`` to go into the root folder of mousecraft gui.
5. Then create a new environment with ``conda env create -f environment.yml`` .
  
6. This creates the mousecraft envinroment, which you can then activate running ``conda activate mousecraft``
   
7.Once env is activated do: ``pip install -e .`` 
This installs mousecraft to your local repository!

8. Now run ``python -m mousecraft`` and you're all set. 
You can also try just ``mousecraft`` but in case this won't work use the abobe command.

### Using pip only

Another way to install the mousecraft package via pip is:

``pip install git+https://github.com/SofiaZang/Mousecraft.git`` in the same directory where you want the mousecraft repository to live in.

Everytime you want to use Mousecraft, you have to first run ``conda activate mousecraft`` and then: ``mousecraft`` or ``python -m mousecraft`` 

<img width="1357" height="987" alt="mousecraft_starter" src="https://github.com/user-attachments/assets/456d843b-7a18-4dda-8eae-03704c6cb3cc" />

Friendly tip #2: Do not attempt to exit, it won't work :) 

<img width="1905" height="1023" alt="mousecraft_gui" src="https://github.com/user-attachments/assets/7986e9d6-0633-4c6e-9bbf-e4147a85b8b1" />

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


## Inputs
Input formats: 

Mousecraft accepts any .npy file of the motion signal you have eg. motion_energy.npy or if you have run the automatic classification notebook then the annotation labels saves as 
gui_lables.csv or .xls under the folder 'mouse_motion_average' under the dataset you analysed. This file will also contain the motion signal value for each frame of the recording so will automatically load and display both motion signal and proposed annotations.

You also need to load an .avi maybe .mp4 or other formats (TO DO: all video formats)

Directory structure

Mousecraft looks for all inputs in the folder you define. 

See examples in this notebook https://colab.research.google.com/drive/1Sfts_onqzadvvcDXfVnDBEA_10ic7_YI?usp=sharing

Tips: For the notebook to run you need the motion_energy.npy (or other motion_signal.npy) or the .tiff file of the behavior you want to annotate if you have not computed the motion energy and it will be automatically computed in the notebook run.

## Using the GUI
The interface is divided into two main panels:

Left Panel: Video playback and control
Right Panel: Motion energy timeline, event navigation, and annotation tools

### Video Display & Controls (Left Panel)

#### Video Display
Main video window: Displays the loaded video of the mouse behavior. 
Shows frame-by-frame playback and updates when navigating events.

#### Load Video

Button: Load Video
Opens a file dialog to select a video file (.avi, .mp4, .mov, .tiff, etc.).

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

#### Load Motion Energy
Loads a precomputed motion energy file (.csv, .xlsx, .npy).

#### Load Classifications

Loads precomputed event classifications (Active/Twitch events) for review and editing.

Annotated events appear as colored spans: Yellow: Active events, Purple: Twitch events, Green/Red/Orange dots: Validation status (Accepted, Rejected, Edited).

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

TODO 
