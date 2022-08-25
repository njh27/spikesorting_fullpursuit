## Spike Sorting Full Binary Pursuit

This package provides a neural spike sorting algorithm implementation as described
in our publication Hall et al. The internal algorithm clusters spike waveforms in a manner derived from
the isocut algorithm of [Chung et al. 2017](https://www.sciencedirect.com/science/article/pii/S0896627317307456) to discover neuron spike waveforms. These spike waveform templates are then used to derive
all the spike event times associated with each neuron using a slightly modified version
of the binary pursuit algorithm proposed by [Pillow et al. 2013](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0062123). The current code-base significantly extends this work by adding alignment, sharpening, and noise estimation among other things. Because all spike events are ultimately discovered by the binary
pursuit algorithm (not just the missed events as suggested originally by Pillow et al.) we have
called the algorithm Full Binary Pursuit (FBP).

A primary focus in fully using the binary pursuit approach to detect spike times is to enable
detection of spikes that overlap in space and time in the voltage trace. The goal
is to accurately detect spikes in the voltage while minimizing false discoveries in order
to allow analyses of single, isolated neurons and cross correlogram analyses between
simultaneously recorded neurons even in the face of high firing rates.

### Installation

#### Requirements
This package depends on the numpy and scipy python packages. The easiest way
to ensure the majority of package requirements are met is to install via the ANACONDA
source and API. Additional requirements not part of the standard Anaconda install
are pyopencl, and the multiprocessing library, all of which are freely available
and installable via the Anaconda API. The cython package was used to generate
C code extensions included in the c_cython sub-package and would be required for
modifications to these files.

A GPU is required to run the binary pursuit algorithm via pyopencl. This step
can be tricky and depends on specific hardware/firmware/software configurations.
We give brief guidance here, but can off little help otherwise. You may need
to install an OpenCL driver for your specific GPU in order for it to work with
OpenCL. Depending on the age of your machine and/or GPU, it may be the case that
you need to choose to install an older version of pyopencl. The oldest case we
tested was pyopencl version 2019.1.2.

Memory usage is very high especially for large files and floating point values.
The opencl GPU code generally only handles up to float32 and so it makes the
most sense to input numpy voltage arrays as datatype "np.float32". Operations
such as filtering and especially Wiener filter increase this memory use and
again mitigates toward float32 being used. The arrays of spike clips are often
copied and can become similarly large. Total processing for 16 channel recording
running ~14 processes with tens of thousands of spikes can easily consume on
the order of 200 GB of RAM.

#### UPDATED numpy memmap to reduce memory consumption
The total memory usage will always depend on the segment duration, number
of threshold crossings identified in a segment, and the number of simultaneous
processes that are run. Minimally, with 'use_memmap'=True, all voltage segments
are held in memory for the processes that are currently clustering them. In the
worst case, this is [ (num processes) x (bytes in 1 voltage segment) ]. Each
process holds its own set of clips which can be memory mapped as well. Fitting
of clips is performed in memory for a subsampling of clips (default 1e5) to
reduce memory load. These also require copying in memory for PCA compuation.
All clips are output in memory in PCA space, requiring a [ (num clips) x
(num principal components selected) ] data matrix in memory for each process.
Wiener filtering is done 1 segment at a time, but requires creating 5
additional arrays equal in size to 1 segment of voltage and all the spike clips
for a single channel discovered in that segment. This step is done
in memory for improved processing speed because it is unlikely to consume
more memory than the clustering step across multiple processes.

The most recent version of pyopencl can be installed with conda using:  
```
conda install -c conda-forge pyopencl
```

Older versions can be installed by specifying the version. e.g.:  
```
conda install -c conda-forge pyopencl=2019.1.2
```

A simple test to see whether pyopencl can detect your graphics card is to run:  
```
import pyopencl as cl
platforms = cl.get_platforms()
for platform in platforms:
	devices = platform.get_devices(cl.device_type.GPU)
	print(devices)
```
If successful this should print the name of your GPU(s). If multiple GPUs are
detected, the current code searches for the one with greatest memory for use.
This can be checked or modified in binary_pursuit_parallel.py ~lines 221-231.

#### Install package
Copy the remote git repository locally, for example with:
```
git clone https://github.com/njh27/spikesorting_fullpursuit.git
```
Navigate to the directory containing the package spikesorting_fullpursuit (the
	directory where the clone command above was executed). Type:  
```
pip install -e spikesorting_fullpursuit
```
If successful, you should now be able to import the package in python using:
```
import spikesorting_fullpursuit
```

### Testing with demos
Basic usage is shown in the scripts and Jupyter notebook provided in "demos". Successfully running
these demos on your own machine should also provide evidence that the software is correctly
installed and able to use a GPU.  
> **NOTE:**  
If running this script causes your computer to hang or crash, you might try testing
first with 'do_overlap_recheck' set to "False". The overlap recheck can be time
consuming and may cause your GPU to either crash or timeout. It is likely that
the watchdog timer for your operating system or graphics card will need to be
increased in order to successfully run. Alternatively, you could run using a
smaller 'max_gpu_memory' or run with smaller segment durations, both of which
will sort less data in each GPU call and therefore might run faster without
causing a timeout.
Be sure that the program is discovering and using the desired GPU.

When using the Jupyter notebook under git, it will likely be useful to configure
how git handles notebook changes and output. Specifically, you may consider using
the nbstripout package, which can be installed either through pip or anaconda
as explained here: [Mateos Blog](http://mateos.io/blog/jupyter-notebook-in-git/)

To run the demo scripts from the command line where the git repository was installed,
scripts should be run in this order:  
First make the voltage data and save ground truth:
```
python ./spikesorting_fullpursuit/demos/make_and_save_voltage.py test_voltage.npy test_ground_truth.pickle
```
Then run the sorting algorithm on the generated synthetic data:
```
python ./spikesorting_fullpursuit/demos/test_demo_run_sort.py test_voltage.npy sorted_demo.pickle
```
Run automated postprocessing to place sorter output into a dictionary sorted neurons:
```
python ./spikesorting_fullpursuit/demos/test_demo_postprocessing.py sorted_demo.pickle sorted_neurons.pickle
```
Finally run a quick script that compares sorted results to the ground truth generated
data and make a couple figures:
```
python ./spikesorting_fullpursuit/demos/test_demo_results.py sorted_neurons.pickle test_ground_truth.pickle
```


### Usage
Once FBP is installed and correctly detecting and using the host machine GPU,
users must accomplish two objectives to sort their own data.

1) Voltage timeseries of data must be in an N channels by M samples Numpy array.

2) The user must create an appropriate electrode object that inherits from the
**AbstractProbe** class defined in the **electrode.py** module. The AbstractProbe class
requires specification of the basic details of the recorded voltage data such as sampling rate
and the number of channels. The primary purpose of the user defined subclass object is
to specify the "get_neighbors" function. This function takes a single input, a channel
number, and returns its corresponding "neighborhood" of channels. Users should ensure
to keep the error check shown in the provided objects to ensure only channels within
range are requested. The returned neighborhood is a numpy array of channel numbers. The
returned numbers will be used to define a neighborhood for clustering of detected spike
events. Returned neighborhoods for channel c, **MUST** include c.


### Settings
The function `spike_sort` takes a number of optional arguments that can adjust the behavior of the sorter.
In general, the default settings to `spike_sort` should yield a fairly good sort for reasonably isolated
neurons. A brief description of the optional arguments and their default values follows. The inputs are given
to the call to spikesorting_parallel via a settings dictionary argument, e.g.  
```
		'sigma': 4.0, # Threshold based on noise level for discovering spikes
		'clip_width': [-8e-4, 8e-4], # Width of clip in seconds, used for clustering. Made symmetric with largest value for binary pursuit!
		'p_value_cut_thresh': 0.01, # Statistical criterion for splitting clusters during iso-cut
		segment_duration': 300, # Seconds (None/Inf uses the entire recording) Can be increased but not decreased by sorter to be same size
		'segment_overlap': 150, # Seconds of overlap between adjacent segments
		'do_branch_PCA': True, # Use branch PCA method to split clusters
		'do_branch_PCA_by_chan': True, # Repeat branch PCA on each single channel separately
		'do_overlap_recheck': True, # Explicitly check if each spike is better accounted for as a sum of 2 spikes (templates)
		'filter_band': (300, 6000), # Sorting DOES NOT FILTER THE DATA! This is information for the sorter to use. Filter voltage as desired
				BEFORE sorting
		'do_ZCA_transform': True, # Whether to perform ZCA whitening on voltage before sorting. Should be used for multichannel recordings
		'check_components': 20, # Number of PCs to check for clustering. None means all
		'max_components': 5, # Max number of PCs to use to form the clustering space, out of those checked
		'min_firing_rate': 0.1, # Neurons with fewer threshold crossings than satisfy this rate are removed
		'use_rand_init': True, # If true, initial clustering uses at least some randomly chosen centers
		'add_peak_valley': False, # Use peak valley in addition to PCs for clustering space
		'max_gpu_memory': None, # Maximum bytes to try to store on GPU during sorting. None means use as much memory as possible
		'save_1_cpu': True, # If true, leaves one CPU not in use during parallel clustering
		'sort_peak_clips_only': True, # If True, each sort only uses clips with peak on the main channel. Improves speed and accuracy but can
				miss clusters for very low firing rate units on multiple channels in rare occasions
		'n_cov_samples': 20000, # Number of random clips to use to estimate noise covariance matrix. Empirically and qualitatively, 100,000
				tends to produce nearly identical results across attempts, 10,000 has some small variance. This step can be slow for large values.
		# e.g. sigma_bp_noise = 95%: 1.645, 97.5%: 1.96, 99%: 2.326; NOTE: these are one sided
		'sigma_bp_noise': 2.326, # Number of noise standard deviations an expected template match must exceed the decision boundary by.
				Otherwise it is a candidate for deletion or increased threshold.
		'sigma_bp_CI': None, # Number of noise standard deviations a template match must fall under for a spike to be added. np.inf or None
				ignores this parameter. Can help with large noise event rejection, but is relatively fickle and can miss real spikes as well.
		'absolute_refractory_period': 10e-4, # Absolute refractory period expected between spikes of a single neuron. This is used in
				postprocesing.
		'get_adjusted_clips': False, # Returns spike clips after the waveforms of any potentially overlapping spikes have been removed
		'max_binary_pursuit_clip_width_factor': 1.0, # The factor by which binary pursuit template matching can be increased relative to clip
				width for clustering. The best values for clustering and template matching are not always the same.
				# Factor of 1.0 means use the same clip width. Less than 1 is invalid and will use the clip width.
		'verbose': False, # Set to true for more things to be printed while the sorter runs
		'test_flag': False, # Indicates a test run of parallel code that does NOT spawn multiple processes so errors can be seen
		'log_dir': None, # Directory where output logs will be saved as text files for each parallel process during clustering. Processes can
				not usually print to the main screen so output can be seen here.
		'output_separability_metrics': False, # Setting True will output the separability metrics dictionary for each segment. This contains
				a lot of information not currently used after sorting, such as noise covariance matrices and templates used by binary pursuit.
		'wiener_filter': True, # Use wiener filter on data before binary pursuit. MUST use sort_peak_clips_only!
    'wiener_filter_smoothing': 150 # Hz or None for no smoothing. Smooths the signal and noise voltage spectra in the frequency domain so that the filter does not become overly specific to the frequency of discovered templates. A roughly ideal number will be the max
		frequency of all events on a channel (combined over all neurons). Probably in the range ~100-200 is good with similar results.
		'same_wiener': True, # If true, compute Wiener filter over all channels at once, using the same filter for every channel. Otherwise compute separately for each channel
    'use_memmap': True, # Will keep clips and voltages stored in numpy memmap files (voltage is preloaded as needed into ram for faster processing)
    'memmap_dir': None, # Location to memmap numpy arrays. None uses os.getcwd(). Should all be deleted after successfully running
		'memmap_fID': None # Optional identifier for naming memmap files for this specific file sort. Useful to prevent multiple simultaneous sorts from repeating file names and overwritting each other's data or causing an error
		'save_clips': True, # Saves all discovered clips in output file. These can get VERY large, so it's optional. Can be recomputed from voltage for postprocessing.
		'serial_zca': True, # Do ZCA serially instead of parallel. Parallel can load a LOT of voltage arrays/copies into memory but is faster.
```

### Output
The immediate output from the sorter should be entered into the postprocessing
WorkItemSummary object as shown in the examples. The final output from this
process is a Python list of dictionaries, where each dictionary (list element)
represents the data for a single sorted unit. The dictionary contains lots of
information about the sorting procedure performed, but importantly contains
NumPy arrays of the spike indices for each detected spike event for that unit.

> **NOTE:**
The current version outputs all of the spike clip waveforms. This is very
memory intensive and it might be desirable to remove these from the final output.
However, the clips are used for stitching segments and are needed for automated
post-processing using the WorkItemSummary object.
