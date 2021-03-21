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

The run the whole data processing pipeline, i.e. read binary files then run tracking logic, you can run

```
python the_one_script.py --config prconfig.yaml
```

But if the main goal is to work on the AI processing or tracker, there's one script which uses some test data already processed and makes things much faster to develop. You can use the script below

```
python script_with_test_data.py.py --config prconfig.yaml
```

If there's the need to focus on the processing section, or you need to create new test data, run

```
python main.py --config PRconfig.yaml
```

To see how the data is, it is useful to run the following command

```
python signal_preview.py --config PRconfig.yaml
```
