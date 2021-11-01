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
smaller 'max_gpu_memory' or with shorter segment durations, which will sort less
data in each GPU call and therefore might run faster without causing a timeout.
Be sure that the program is discovering and using the desired GPU.

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
settings['sigma'] = 4.0 # Threshold based on noise level
	settings['clip_width'] = [-8e-4, 8e-4]# Width of clip in seconds
	settings['p_value_cut_thresh'] = 0.01
	settings['segment_duration'] = 300 # Seconds (None/Inf uses the entire recording) Can be increased but not decreased by sorter to be same size
	settings['segment_overlap'] = 150 # Seconds of overlap between adjacent segments
	settings['do_branch_PCA'] = True # Use branch pca method to split clusters
	settings['do_branch_PCA_by_chan'] = True
	settings['do_overlap_recheck'] = True
	settings['filter_band'] = (300, 6000) # This is information for the sorter to use. Sorting DOES NOT FILTER THE DATA!
	settings['do_ZCA_transform'] = True
	settings['check_components'] = 20 # Number of PCs to check. None means all
	settings['max_components'] = 5 # Max number to use, of those checked
	settings['min_firing_rate'] = 0.1 # Neurons with fewer threshold crossings than satisfy this rate are removed
	settings['use_rand_init'] = True # Initial clustering uses at least some randomly chosen centers
	settings['add_peak_valley'] = False # Use peak valley in addition to PCs for sorting
	settings['max_gpu_memory'] = None # None means use as much memory as possible
	settings['save_1_cpu'] = True
	settings['sort_peak_clips_only'] = True # If True, each sort only uses clips with peak on the main channel. Improves speed and accuracy but can miss clusters for low firing rate units on multiple channels
	settings['n_cov_samples'] = 20000 # Number of random clips to use to estimate noise covariance matrix. Empirically and qualitatively, 100,000 tends to produce nearly identical results across attempts, 10,000 has some small variance.
	# sigma_bp_noise = 95%: 1.645, 97.5%: 1.96, 99%: 2.326; NOTE: these are one sided
	settings['sigma_bp_noise'] = 2.326 # Number of noise standard deviations an expected template match must exceed the decision boundary by. Otherwise it is a candidate for deletion or increased threshold.
	settings['sigma_bp_CI'] = 12.0 # Number of noise standard deviations a template match must exceed for a spike to be added. np.inf or None ignores this parameter.
	settings['absolute_refractory_period'] = 10e-4
	settings['get_adjusted_clips'] = False
	settings['max_binary_pursuit_clip_width_factor'] = 1.0 # Factor of 1.0 means use the same clip width. Less than 1 is invalid and will use the clip width.
	settings['verbose'] = False
	settings['test_flag'] = False # Indicates a test run of parallel code that does NOT spawn multiple processes
	settings['log_dir'] = None # Directory where output logs will be saved as text files
	```

### Output
The immediate output from the sorter should be entered into the postprocessing
WorkItemSummary object as shown in the examples. The final output from this
process is a Python list of dictionaries, where each dictionary (list element)
represents the data for a single sorted unit. The dictionary contains lots of
information about the sorting procedure performed, but importantly contains
NumPy arrays of the spike indices for each detected spike event for that unit.
The current version also outputs all of the spike clip waveforms. This is very
memory intensive and it might be desirable to remove these from the final output.
