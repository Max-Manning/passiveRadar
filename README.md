![](./title_image.png)

## What Is This???

This repository is a fork from [Max Manning - Passive Radar](https://github.com/Max-Manning/passiveRadar) and contains processing code for a software defined radio based passive radar. Passive radars don't transmit any signals of their own - instead, they locate targets by detecting the echoes of ambient radio signals that bounce off of them. Check out [this page](https://dopplerfish.com/passive-radar/) for more information about how passive radar works.

## Usage

First clone the repository and make a conda environment with the required packages (or install them manually).

```
git clone https://github.com/Max-Manning/passiveRadar
cd passiveRadar
conda env create -f environment.yaml
conda activate radar-env
```

To see how the data is, it is useful to run the following command

```
python signal_preview.py --config PRconfig.yaml
```

Put the data file in the same directory as `main.py`, and run the following command to process the data and save it to a file:

```
python main.py --config prconfig.yaml
```

To create multiple images of Bistatic Range x Doppler Shift, you can run

```
python range_doppler_plot.py --config PRconfig.yaml --mode frames
```

or, if you have [ffmpeg](https://ffmpeg.org/download.html) installed on your computer, you can generate an animation.

```
python range_doppler_plot.py --config PRconfig.yaml --mode video
```
