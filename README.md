A passive radar locates targets by detecting the echoes of broadcast signals that bounce off of them. This repository contains signal processing scripts for a FM radio based passive radar. Check out [this page](https://dopplerfish.com/passive-radar/) for more information about how passive radar works.

## File Contents

`PRconfig.yaml`: Configuration file for defining passive radar parameters. (Note that not all parameters defined in `PRconfig.yaml` are currently used, but will be added in a later version.)

`PRrun.py`: Main entry point to the program. Computes a sequence of range-doppler maps from the raw input signals. Broadly the steps are:

1. Load the passive radar data, tune to the center frequency of the channel that is being used and decimate to the channel bandwidth
2. Correct a constant time offset between the two input channels
3. Apply a least squares adaptive filter for clutter removal
4. Compute range-doppler maps for each signal block and save them to a file. 

`passiveRadar\PRalgo.py` : Contains most of the passive radar signal processing functionality (least squares filtering and range-doppler processing.)

`passiveRadar\signal_utils`: various utility functions such as `deinterleave_IQ` and `frequency_shift`.

`passiveRadar\range_doppler_plot.py`: Once the range-doppler surfaces for each time step have been computed, `range_doppler_plot.py` can be used to generate passive radar video frames. Each frame is stored as an image in a specified directory. Use your favorite video software to render them into a video. 

## Obtaining Passive Radar Data

The following GNUradio block diagram can be used to receive passive radar data from a pair of RTL-SDR dongles [modified to share a clock](http://kaira.sgo.fi/2013/09/16-dual-channel-coherent-digital.html).

![](./GNUradio_blockDiagram.jpg)

*Recording baseband data like this results in pretty large files so unfortunately I can't post any example files in this repository. Feel free to contact me if you would like and I will find a way to send the example data to you.*

## Input File Format

The program expects the input to be two continuous streams of interleaved IQ data (the reference and observation channels). The current implementation assumes both channels are contained in a single hdf5 file, but it would be straightforward to use a different format. For large input files it is important that the data can be lazily read into memory as it is needed instead of loaded all at the same time.

An easy way to convert binary data files to hdf5 is the `h5import` command-line tool which is included in [the latest hdf5 release](https://www.hdfgroup.org/downloads/hdf5/).

## Algorithm Details

### Least Squares Filtering:

Three least squares filter implementations are provided in `passiveRadar\PRalgo.py`:

1. `LS_Filter` : direct least squares implementation based on matrix inversion

2. `LS_Filter_SVD` : least squares filter implementation based on the singular value decomposition. Slower than the direct matrix inversion method but guaranteed stability.

3. `LS_Filter_Toeplitz` : least squares filter implementation which makes strong assumptions about the statistical stationarity of the input signals.  Much faster than the previous two implementations but can be inaccurate if the assumptions are violated (in my experience this rarely happens.  

### Range-Doppler Processing

Range-doppler processing is achieved by computing the cross-ambiguity function. Two cross-ambiguity function  implementations are provided in `passiveRadar\PRalgo.py`:

1. `direct_xambg`: Computes the cross-ambiguity function directly.

2. `fast_xambg`: Fast cross-ambiguity algorithm that uses decimation followed by a FFT. About twice as fast as the direct implementation. 

## Contributing

I am actively looking for contributors to help make this project more usable. Please submit a pull request if you have any ideas about how the software can be improved.

   

   

