# MouseCraft

A GUI tool for annotating mouse motion events in video.

## Features

- Load video and motion energy files
- Annotate, edit, and validate motion events
- Export results in multiple formats

## Installation

### Using Conda (Recommended)
```sh
git clone https://github.com/yourusername/mousecraft.git
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
