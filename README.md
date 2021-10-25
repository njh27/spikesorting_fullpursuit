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
Navigate to the directory containing the package spikesorting_fullpursuit. Type:  
```
pip install -e spikesorting_fullpursuit
```

### Usage
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


Once FBP is
installed and correctly detecting and using the host machine GPU, users must accomplish
two objectives to sort their own data.

1) Voltage timeseries of data must be in an N channels by M samples Numpy array.

2) The user must create an appropriate electrode object that inherits from  the
"AbstractProbe" class defined in the "electrode.py" module. The AbstractProbe class
requires specification of the basic details of the recorded voltage such as sampling rate
and the number of channels. The primary purpose of the user defined subclass object is
to specify the "get_neighbors" function.




Calls to `spike_sort` using an `AbstractRecording` will be wrapped in a generic electrode object, where `neighbors` only returns the current electrode (e.g., no neighbors).

To sort a file:
```julia
using NeurophysToolbox, SpikeSorting

# Open a PL2 recording in the current directory
recording = PL2Recording("filename.pl2")

# Sort the file. Each electrode is sorted
# independently
neurons = spike_sort(recording, verbose=true)
```

Note that the above example assumes that each contact on the electrode is independent (that is,
there is no case where a neuron can appear on multiple channels simultaneously). This is because,
as noted above, the `AbstractRecording` is wrapped in a generic electrode object where each neighbor
is a singleton. To define a more complex geometry, we need to define a new class which inherits from
`AbstractProbe`. For instance, consider a linear probe with 16 channels spaced 50 microns apart. We can
assume that a single neuron will only ever appear on contacts within 200 microns of the primary channel.
In this case, we can define our probe and sort our file as:
```julia
using NeurophysToolbox, SpikeSorting

# Define a linear array
mutable struct LinearArray <: AbstractProbe
	recording::AbstractNeurophysiologyRecording
end
NeurophysToolbox.channels(x::LinearArray) = channels(x.recording)
NeurophysToolbox.attributes(x::LinearArray) = attributes(x.recording)

# Define our neighbors function for linear arrays with 50 um
# contact spacing
function neighbors(x::LinearArray, index::Integer)
	contact_locations = range(0, step=50, length=16) # contact locations in microns
	current_contact_location = contact_locations[index]
	# Find all contacts within 200 um of the current contact
	return findall(abs.(contact_locations .- current_contact_location) .< 200)
end

# Load our recording
recording = PL2Recording("filename.pl2")

# Sort using the linear array geometry
neurons = spike_sort(LinearArray(recording), verbose=true)
```

While we have defined a LinearArray object above, this was merely an example. The SpikeSorting package
defines a number of generic probe objects that will probably fit most use-cases. A listing of available
probes can be found in `electrode.jl`. Briefly, the package provides
 - `GenericProbe`: A probe where each contact is independent (the `neighbors` function returns just the
index of the current contact. If you have a multiple channel recording and do not specify an `AbstractProbe`,
the spike sorting package will assume you are using this probe.
 - `AllNeighborsProbe`: All contacts are assumed to be neighbors with all other contacts. This is appropriate
when the contacts are spaced very close together (e.g., a tetrode or a heptode).
 - `XYProbe`: A more complicated probe structure that takes a vector of contact "x" and "y" positions (in any units)
as well a distance threshold (in the same units as "x" and "y"). Contacts that are within the distance threshold
are considered neighbors of a given contact. Use as follows: `XYProbe(x_positions, y_positions, threshold, recording)`.
 - `XYZProbe`: An even more complex probe geometry that takes a vector of contact "x" and "y" positions in any unit (just like
the `XYProbe`. However, this class also takes a vector of probe id's (a vector of integers that same length as x and y).
Contacts that are within the position threshold are considered neighbors only if they exist on the same probe number. Use as
follows: `XYZProbe(x_positions, y_positions, contact_probe_ids, threshold, recording)`.

### Sorting complex probe geometries
The following example shows how one might want to spike sort a Plexon S- or V-probe with 32 contacts in two,
non-staggered, columns. In this example, the two columns of contacts are 50 microns apart with 100
micron spacing between adjacent contacts in the same column. We define a neighbor to be any contact that is within
200 microns of the current contact.

```julia
using NeurophysToolbox, SpikeSorting

# Load our recording
recording = PL2Recording("filename.pl2")

# Note that we assume each of the X and Y locations
# passed into XYProbe corresponds 1:1 to the channels
# in the recording. In this example the probe geometry
# looks like:contact_positions_z
#             |    .       .    |
#             |    .       .    | 32 channels
#             |    5       6    | 50 um between columns
#             |    3       4    | 100 um between contacts in
#             |    1       2    |    a column (i.e., distance
#              \               /     between #1 and #3)
#               \             /  * Not to scale...
#                \           /

# 0, 50, 0, 50, 0, ....
x_positions = repeat([0, 50], 16)
# 0, 0, 100, 100, 200, 200...
y_positions = [Integer(floor(i / 2)) * 100 for i in range(0, length=32)]
probe = XYProbe(x_positions, y_positions, 200, recording)

# Sort using the S-/V-probe geometry array geometry
neurons = spike_sort(probe, verbose=true)
```

### Sorting multi-probe recordings
The generic `XYZProbe` can be used to describe experimental sessions where multiple independent
electrodes are recorded from different brain areas simultaneously. This probe configuration assumes
that a neuron cannot exist across two different physical probes. That is, contacts from different
probes are never neighbors. The inputs to `XYZProbe` are the x- and y-positions of the contacts across
all probes followed by a vector or "probe-ids". These probe-ids must be integers. The lengths of
these three vectors should be the same (equal to the total number of channels in the recording).

As an example, consider the Plexon S- or V-probe geometry considered above. We might use
two of these 32 channels probes to record from two brain areas simultaneously. Therefore,
our ultimate recording would have 64 total channels. If "probe 1" corresponds to channels 1 - 32
in the recording whereas "probe 2" corresponds to channels 33 - 64 in the recording, we could describe
this complicated multi-probe configuration as:
```julia
using NeurophysToolbox, SpikeSorting

# Load our recording
recording = PL2Recording("filename.pl2")

x_positions = repeat(repeat([0, 50], 16), 2)
y_positions = repeat([Integer(floor(i / 2)) * 100 for i in range(0, length=32)], 2)
probe_ids = vcat(zeros(Int64, 32), ones(Int64, 32)) # These must be integers
probe = XYZProbe(x_positions, y_positions, probe_ids, 200, recording)

# Sort using the multi-electrode geometry
neurons = spike_sort(probe, verbose=true)
```

### Sorting multiple recordings from the same session
In some cases, we want to treat recording files from the same experimental session
as if we had recorded them continuously. Imagine that our recording session is
split over three files (e.g., one for tuning, one for a baseline block, and one for an experimental manipulation).
Ideally, we would want to treat these three files like they had all come from the same
combined recording such that a neuron identified in the tuning block continues to be identified
in the subsequent two blocks.

To perform sorting or multiple files as if they were recorded continuously, we can pass
a vector of `AbstractRecording`s to the spike sorting algorithm. Note that all of the recordings
must have the same number/type of channels and the same sampling rates within a given
channel. The output of the sorting algorithm with then be a vector of neuron vectors. Each
element of the outer vector corresponds to a given input file (with spike indices from the
beginning of the respective file). For example:
```julia
using NeurophysToolbox, SpikeSorting

baseline_recording = PL2Recording("baseline_recording.pl2")
manipulation_recording = PL2Recording("manipulation_recording.pl2")

# Sort the two recordings as if they were recording continuously where all
# contacts are neighbors
baseline_neurons, manipulation_neurons = spike_sort([baseline_recording, manipulation_recording], x -> AllNeighborsProbe(x), verbose=true)

# Importantly, baseline_neurons[1] is the same neuron as manipulation_neurons[1]
# However, baseline_neurons[1] has spike_indices that are relative to the start of
# baseline_recording and manipulation_neurons[1] has spike_indices relative to the
# start of manipulation_recording
```
You may notice a strange second argument to the spike_sort function: `x -> AllNeighborsProbe(x)`. This is
called an anonymous function that takes a single input argument, in this case `x`. This function will
be called for the combined recording to convert it into an AbstractProbe. In this example, the combined
recording (across the baseline and manipulation epochs) are converted into an `AllNeighborsProbe`. Importantly,
this function can only take a single argument, the recording to be converted. If we wanted to use
more complex geometry (such as the Plexon S-/V-probe described above), we can do that as well:
```julia
using NeurophysToolbox, SpikeSorting
baseline_recording = PL2Recording("baseline_recording.pl2")
manipulation_recording = PL2Recording("manipulation_recording.pl2")
# Sort the two recordings as if they were recording continuously using an XYProbe
x_positions = repeat(repeat([0, 50], 16), 2)
y_positions = repeat([Integer(floor(i / 2)) * 100 for i in range(0, length=32)], 2)
baseline_neurons, manipulation_neurons = spike_sort([baseline_recording, manipulation_recording], x -> XYProbe(x_positions, y_positions, 200, x), verbose=true)
```

## Running work items in parallel
By default, the spike sorting algorithm runs serially on a single process/thread. However, it is possible
to run sorting across multiple workers using the `Distributed` package. You first need to start
up workers either by starting Julia with the process switch (e.g., `julia -p 4` starts up
Julia with 4 workers) or by adding workers after starting julia through the `addprocs()` function
of the `Distributed` package. Note that adding processes only needs to be performed *once*
after starting up Julia. Second, the recording that is to be sorted needs to be distributed in
shared memory across these workers. This ensures that the large voltage traces are not
copied between processes (all processes access the same underlying memory region on the system).
This can be done using the `SharedMemoryRecording` structure that is part of the `NeurophysToolbox`.
An example of how to do this is show below:
```julia
using Distributed
using NeurophysToolbox, SpikeSorting

# Add as many workers as there are CPU threads on this system
# You can also specify the number of workers by passing an integer into
# addprocs()
addprocs() # Only needs to be performed once after starting julia

# Open the PL2Recording
recording = PL2Recording("filename.pl2")

# Move the recording into shared memory and share across all workers
# NOTE: This must be done after any calls to addprocs()
recording = SharedMemoryRecording(recording)

# Call the spike sorting function with the appropriate probe wrapper
# Here we use the AllNeighborsProbe, but any probe could be used
neurons = spike_sort(AllNeighborsProbe(recording), verbose=true)
```
Note that "work items" are scheduled in parallel. When sorting a recording, a work item
is created for every segment for every channel (the total number of work items is the number
of segments by the number of channels). If you are sorting a single channel with a single
segment (the default), parallelization will only slow down the sorting process. Therefore,
it is recommended to only use parallel processing when there is more than one channel
or when `segment_duration` is less than the duration of the recording.


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
