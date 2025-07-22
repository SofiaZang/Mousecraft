# MouseCraft

A very fun GUI & splendid GUI used for validating behavior annotations.

## Features

- Load video and motion enegry or classification labels
- Validate, edit, and add motion events
- Export results in multiple formats (MF, HF) (.npy, .csv) & plot some performance statistics 

## Installation

### Using Conda (Recommended)

git clone https://github.com/SofiaZang/mouse_motion_analysis/tree/main
cd mousecraft
conda env create -f environment.yml
conda activate mousecraft
pip install .
```

### Using pip only
```sh
pip install git+https://github.com/yourusername/mousecraft.git
```

## Running MouseCraft

```sh
mousecraft
```

or

```sh
python -m mousecraft
```

## Dependencies

- PyQt5
- numpy
- pandas
- matplotlib
- opencv-python

## License

[Your License Here]
