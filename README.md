# Mousecraft                                                     <img width="360" height="328" alt="image" src="https://github.com/user-attachments/assets/85e1e2fc-9308-42e5-8c03-8616f15092d6" />
                                                                                                                                                                                                                                         
A fun GUI used for classifying and validating behavior annotations.

## Features

- Load video and motion signal (eg. motion energy, eye motion) or classification labels (gui labels)
- Move from onset to onset and accept/reject, edit, or add motion events
- Export results in multiple formats (.npy, .csv) & plot some performance statistics 

## Installation

### Using Conda (Recommended)

For local installation (tried in Windows, to be tried in other OS)

1. Install an Anaconda distribution of Python or miniconda (miniforge, if anaconda wont work try via miniforge) & GitBash 

2.  Open an anaconda prompt / command prompt with conda for python 3 in the path.
Navigate where you want the MouseCraft files to live. For example: cd Documents

3. Once under code folder do  ```git clone https://github.com/SofiaZang/Mousecraft.git``` mousecraft repository. You can also get the cloning link under green code button here https://github.com/SofiaZang/Mousecraft.git.
If this does not work, do the git clone (step 2: navigate to desired folder and step : clone the repository) in GitBash. 

4. Once cloned in the anaconda prompt, do ``cd Mousecraft`` to go into the root folder of MouseCraft GUI.
   
5. Then create a new environment with ``conda env create -f environment.yml`` . In case creating this environment takes too long, try ``conda env create -f environment.yml -c conda-forge --strict-channel-priority``. Else, try creating the envirnonment via miniconda.
  
6. Activate the MouseCraft environment ``conda activate mousecraft`` . 
   
7. Once activated do: ``pip install -e .`` 
This installs mousecraft codes to your local repository!

8. Now run ``python -m mousecraft`` and you're all set. The GUI should open! You have two options Play or Exit buttons - tip: Choose Play to open the GUI! 

### Using pip only (alternative installation)

Another way to install the mousecraft package via pip is:

``pip install git+https://github.com/SofiaZang/Mousecraft.git`` in the same directory where you want the mousecraft repository to live in.

### General

Everytime you want to use Mousecraft, you have to first run ``conda activate mousecraft``, navigate to the mousecraft folder and inside Mousecraft folder and then: ``python -m mousecraft`` 

<img width="657" height="200" alt="image" src="https://github.com/user-attachments/assets/0ed43675-a008-4779-87cf-d181f9af7360" />

Then the GUI opens! 

<img width="1357" height="987" alt="mousecraft_starter" src="https://github.com/user-attachments/assets/456d843b-7a18-4dda-8eae-03704c6cb3cc" />

Friendly tip #2: Do not attempt to exit, it won't work :) 

<img width="1912" height="993" alt="image" src="https://github.com/user-attachments/assets/d0daebf8-7641-4566-a66d-78084d17992c" />

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


### How to use MouseCraft:

## Inputs

**Annotate and Validate in the GUI:

Mousecraft accepts any '.npy"' file of the motion signal you have computed for example 'motion_energy.npy' or 'eye_motion.npy'. Once you load your signal you can click on 'Compute classification" button to launch the annotation.

Last, load your behavioral video recording in .avi or .mp4. Now you can inspect video and annotations in synch and validate accordingly.

**Annotate in the notebook and load in the GUI for validation-only:

Else, you can independently run the 'motion_classifier.ipynb' notebook and load the 'gui_labels.xlsx' file that the notebook saves under the folder'mouse_motion_average'. This file will loads and displays both motion signal and proposed annotations so you don't need to 'Compute classification' in the GUI.

The notebook will look for the following configuration:

-dataset_id
  - camera_processed
    - 'motion_energy.npy'

If you did not compute motion_energy yet, you can do so within the notebook by setting the 'movie_path' variable to your behavioral video (.tiff). 
-dataset_id
  - camera_processed
  - mouse_video.tif 

## Using the GUI

The interface is divided into two main panels:

Left Panel: Video playback and controls
Right Panel: motion signal annotation timeseries, event navigation, and validation tools

### Video Display & Controls (Left Panel)

#### Load Video

Button: Load Video and Add a Camera
Opens a file dialog to select a video file (.avi, .mp4, .mov, .tiff, etc.).
You can load a second video and adjust both size. 

#### Video Display

Main video window: Displays the loaded video of the mouse behavior. Shows frame-by-frame playback and updates in real-time while navigating events.

#### Playback Controls

Play ▶: Starts playing the video from the current frame at the defined FPS.
Pause ⏸: Pauses playback but keeps the current frame in view.
Stop ⏹: Stops playback and resets to frame 0.

#### FPS Control
Textbox (FPS): Defines frames per second for playback speed (e.g., enter 5 for 5 fps). By default the movie will play at 1 fps.

#### Frame Slider

Allows manual scrubbing through frames of the loaded video. You can also just enter the index of the frame you want to inspect and you will teleport there.

### Onset Status and Performance Metrics

#### Onset Status

Informs whether the current frame corresponds to an annotated onset (e.g., Active or Twitch onset).

#### Performance Score

Text box displaying annotation statistics. Accepted = 1, Rejected = -1, Edited = 0.5, Pending = 0, Manually-added = 0. The more you reject, you penalise the score and the more you accept you boost it. This is just to keep track of correct guesses of the annotation and potentially adapt it.

### Save & Export

#### Export Folder

Field + Button (...): Choose where to save annotations and GUI outputs.

Mousecraft auto-saves your progress every 20 min if you have performed at least one action within these 20 min. First autosave asks you for the output path and then its set for the following saves. If you close mousecraft before having finished the validation, you will not lose current progress as it is saved with prefix '_pending'. Once you complete and no pending events remain, the output files will overwrite any pending ones and be saved with prefix '_final'.

### Motion Energy Timeline (Right Panel)

#### Timeline Plot

Displays the motion trace over time.

#### Load An Input

Loads a precomputed motion energy file (.csv, .xlsx, .npy) or a classification ('gui_labels.xlsx') for review and editing.
Annotated events appear as colored spans: Yellow: Active events, Purple: Twitch events, Green/Red/Orange dots: Validation status (Accepted, Rejected, Edited).

#### Add A Second Input

Do the same as Load An Input. The second input appears under the first one. For example, you may want to load whole body movie and only eye close up movie as a second input and run the classification independently on both.

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

Dropdown to choose event type (by default Twitch, Active, Complex) for manual annotation.

Tip: In case you have made some mistake and an event overlaps with another, an error message appears before save.

#### Add a Type: 
If the type of motion you want to annotate is different than the default options, you can add the movement type of interest.
You can name the type and choose a color for it. This tape will now appear in every dropdown menu (event type of manual edition and onset navigation). 

#### Onset / Offset Frame

Spinboxes to manually enter frame numbers for onset and offset of new events.

#### Set Current Frame

Buttons to set the current video frame as onset or offset.

#### Add Event

Adds the new event to the annotation timeline. If the added or edited event overlaps fully with another event, the automatic event will be rejected and this new addition will be saved.

#### Edit Threshold (frames)
This matters for accuracy tracking. Sets the frame tolerance used when classifying edited events (e.g., 5 means ±5 frames from original is considered “minor edit” the event status in the output file is edited but the dot is Green and the score is +1). This can be adjusted per user if you are interested to evaluate the automatic annotation's performance. (If for example the signal is averaged 5 times, we keep the tolerance at 5 frames).

## Mousecraft outputs

### Main outputs:

Mousecraft currently outputs 2 main outputs in .npy and .csv format:

#### validation_HF (Human Friendly) 

<img width="400" height="223" alt="image" src="https://github.com/user-attachments/assets/b12a4357-8e9a-4809-9985-815b4dfe9f56" />

#### validation_MF (Machine Friendly)

<img width="613" height="509" alt="image" src="https://github.com/user-attachments/assets/dac8431f-fa92-4e61-855c-a574f600e2db" />

Same information but each line is a frame (in same format as input .csv) and this is the input you can reload when you continue validating from _pending.

#### Other outputs:

and also 2 .pngs and .json that showcase the overall performance of mousecraft validation

<img width="1359" height="1011" alt="validation_status_pie_final" src="https://github.com/user-attachments/assets/c936c207-6bcf-4b6c-a2ae-f25b42b73d14" />
<img width="4472" height="1676" alt="validation_comparison_plot_final" src="https://github.com/user-attachments/assets/6ead7ed5-d7c9-4074-a92d-6bbec33706d5" />
<img width="506" height="300" alt="image" src="https://github.com/user-attachments/assets/3e00e065-4ced-4398-ae6f-fbb022ae08c9" />

## License

MIT 
