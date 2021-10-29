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






## Settings
The function `spike_sort` takes a number of optional arguments that can adjust the behavior of the sorter.
In general, the default settings to `spike_sort` should yield a fairly good sort for reasonably isolated
neurons. A description of the optional arguments and their default values follows.

Optional settings:
 - `sigma`: The threshold for identifying spikes during the initial sort. The value is based on the magnitude
of the spike relative to noise using an algorithm proposed by *Quiroga et al. (2004)*. The default value is
`4.0`, and should be a fairly good compromise between finding low threshold units and avoiding clustering of noise.
Higher values of sigma will also work well, provided that binary pursuit is enabled (as units with some spikes
lower than threshold will be detected in binary pursuit). Lower values are recommended (`3.0` to `3.5`) if
binary pursuit is disabled.
 - `verbose`: A boolean variable that defines whether the sorting algorithm should output information about
the progress of the sort. By default this is `false`.
 - `verbose_merge`: A boolean variable that defines whether the underlying iso-cut algorithm should print
information about merges/splits performed during the sort. This is a debugging option and should not be used
in normal environments (due to the huge quantity of text that will be output to the screen). The value defaults
to `false`.
 - `timeseries_type`: A type of neurophysiology timeseries that the primary sort should be performed on. By default
this is the `SpikeTimeseries`, which sorts based on the band-pass filtered spike timeseries. Technically, sorts
can also be performed on `WideBandTimeseries` or `LFPTimeseries`, but these instances would be rare.
 - `threshold_type`: A String that defines the type of threshold to use to identify spikes during the intial sort. That
is, this settings defines in which direction a spike is required to exceed the threshold defined by `sigma`. By default,
this is `"absolute"`, meaning that a spike can cross +/- the threshold. Other allowed values are `"negative"` or "`positive"`.
 - `sharpen`: A boolean variable which enables sharpening after clustering across a channels neighborhood. This is enabled
by default. Sharpening prevents the emergence of new clusters due to the overlapping occurence of a spike on the primary
channel and a synchronous spike on a neighboring channel.
 - `clip_size`: A vector specifying the duration (in seconds) of individual clips used for sorting. This defaults to
`[0.3e-3, 0.7e-3]`, indicating that a clip uses 300ms prior to the onset of the threshold crossing and 700ms after the
threshold crossings.
 - `compute_noise`: A boolean variable that ask whether we should estimate the amount of noise for each putative neuron. This
defaults to `false`.
 - `remove_false_positive`: A boolean variable that removes spikes that cluster with noise following sorting. This flag
requires that `compute_nosie` is also `true`. This is `true` by default.
 - `branch_pca`: A boolean variable that specifies whether we should "branch" on the sort following initial sorting. This
flag is `true` by default. Branch PCA potentially splits individual units found after initial sorting into many more units
using data obtained across the neighborhood.
 - `preprocess`: A boolean variable that specifies whether preprocessing (filtering and ZCA) is performed on the recording.
This is `true` by default. However, if you are sorting multiple times, preprocessing only needs to be performed once. Subsequent
calls to `spike_sort` can benefit from `preprocess=false`.
 - `spike_filter`: The bandwidth to use during preprocessing to convert a `WideBandTimeseries` into a `SpikeTimeseries`. The value
is specified in Hz. This defaults to `[300, 8000]`.
 - `compute_lfp`: A boolean variable that specifies whether we compute the `LFPTimeseries` from the `WideBandTimeseries` during
preprocessing. This defaults to `false`. However, if you do plan on using the `LFPTimeseries` either during the sort or afterwards
(which is not typical), set this value equal to `true`.
 - `lfp_filter`: The bandwidth of the Bandpass filter that converts a `WideBandTimeseries` into an `LFPTimeseries` during preprocessing.
See the specification of `spike_filter` for more information. This defaults to `[25, 500]`. This is only useful if `compute_lfp = true`.
 - `zca`: A boolean variable specifying whether to perform the ZCA algorithm across channels during preprocessing. This value defaults
to `false` but can potentially improve the sort across mutiple channels if the channels within the electrode share common-mode noise.
 - `compute_optimal_number_of_pca_components`: A boolean variable that specifies whether the compute the number of PCA components using
cross validation for the initial sort. This value defaults to `true`.
 - `max_pca_components`: An integer representing the maximum number of PCA components to use to sort. The default is `Inf`. If
`compute_optimal_number_of_pca_components` is `true` then the value of PCA components that ends up being used in the sort is
less than or equal to this number. A value of `Inf` implies that the code should use as many principle components as possible.
 - `skip_width`: A floating point number that represents the time between detecting a threshold crossing during the initial
sort and finding a new threshold crossings. By default this is equal to the duration of `clip_size`.
 - `remove_isi_violations`: A boolean variable that asks whether we should remove instances of ISI violations within a neuron.
This is `true` by default and removes instances where the neuron fires two spikes with a duration of less than 1ms between them.
 - `low_firing_neuron_threshold`: A floating point number that defaults to `0.0` that specifies the minimum mean
firing rate of neuron. Neurons that are below this mean firing rate are removed from the output structure. This
setting is disabled by default (`0.0`).  
 - `iso_cut_p_threshold`: The statistical threshold used to determine if the iso-cut algorithm should split two clusters
that it compares (the p-value of the KS statistic). When the p-value for a comparison of two clusters is larger than
`iso_cut_p_threshold`, the two clusters are merged, otherwise they are split at the optimal split point. Smaller values
of `iso_cut_p_threshold` (e.g., `0.01`) require more evidence to split clusters, which will result in fewer neurons after
the sort.
 - `binary_pursuit`: A boolean variable which species whether we before the `binary_pursuit` algorithm following spike
sorting. The binary pursuit algorithm finds "hidden" (secret) spikes that are not detected during the intial sort
due to the presence of a different neuron's spikes (e.g., overlapping spikes) or spikes that are slightly under
the specified threshold. By default this value is `true`.
 - `force_cpu`: A boolean variable which specifies whether GPU algorithms should be forced to run on the CPU. This
is `false` by default, but can be a useful debugging flag if there are issues with computations on your system's
graphics card.
- `gpu_memory_bytes`: The maximum amount of memory to use for computation on the GPU. By default this is `nothing` (which reserves
1GB of memory for the host operating system and then uses the remaining available memory for computation).
 - `segment_duration`: The duration of individual segments to use to sort the neurons (in seconds). By default, this value is
`nothing`, indicating that the entire recording should be sorted en-masse. This setting is useful if there is instability in the
recording. When `segment_duration` is a real number, sorting is performed on each `segment_duration` individually, and then neurons
are combined across segments using a stitching algorithm.
 - `stitch_segments`: A boolean variable that specifies whether we should join
neurons across individual segments. By default this variable is `true`. When
`false` spike sorting will return the neurons found in each segment as unique
neurons in the output, allowing the user to join neurons manually.
 - `remove_redundant_neurons`: A boolean variable that specifies whether to remove neurons (across contacts) that are likely
the same unit. This defaults to `false`. In general, the user should inspect the returned data for duplicate neurons rather than
relying on this algorithm.
 - `combine_overlapping_templates`: A boolean variable that default to `true` that specifies whether to test if a given neuron's
template is the sum of two other detected neurons (that is, represents an overlap). When enabled, the spikes for this "overlap" unit
are assigned to the unit with the larger variance. Additional spikes for the second unit can then be detected via the binary pursuit
algorithm.
 - `random_seed`: An integer that specifies that random seed to use to ensure that sorting results are reproduceable across
multiple runs. This is disabled by default, by setting `random_seed = nothing`.
 - `save_clips`: A boolean flag that specifies whether we should save the found clips for every spike in the output neuron
 structure. By default this is true (this makes post-processing easier because the clips are saved after appropriate filtering).
 However, it also makes the output size much larger.
