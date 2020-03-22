import numpy as np
from spikesorting_python.src import segment
from spikesorting_python.src import preprocessing
from spikesorting_python.src import sort
from spikesorting_python.src import overlap
from spikesorting_python.src import binary_pursuit
from spikesorting_python.src import consolidate
import warnings



"""
    Since alignment is biased toward down neurons, up can be shifted. """
def check_upward_neurons(clips, event_indices, neuron_labels, curr_chan_inds,
                            clip_width, sampling_rate):
    templates, labels = segment.calculate_templates(clips[:, curr_chan_inds], neuron_labels)
    window, clip_width = segment.time_window_to_samples(clip_width, sampling_rate)
    center_index = -1 * min(int(round(clip_width[0] * sampling_rate)), 0)
    units_shifted = []
    for t_ind, temp in enumerate(templates):
        if np.amax(temp) > np.abs(np.amin(temp)):
            # Template peak is greater than absolute valley so realign on max
            label_ind = neuron_labels == labels[t_ind]
            event_indices[label_ind] += np.argmax(clips[label_ind, :][:, curr_chan_inds], axis=1) - int(center_index)
            units_shifted.append(labels[t_ind])

    return event_indices, units_shifted


def branch_pca_2_0(neuron_labels, clips, curr_chan_inds, p_value_cut_thresh=0.01,
                    add_peak_valley=False, check_components=None,
                    max_components=None, use_rand_init=True, method='pca'):
    """
    """
    neuron_labels_copy = np.copy(neuron_labels)
    clusters_to_check = [ol for ol in np.unique(neuron_labels_copy)]
    next_label = int(np.amax(clusters_to_check) + 1)
    while len(clusters_to_check) > 0:
        curr_clust = clusters_to_check.pop()
        curr_clust_bool = neuron_labels_copy == curr_clust
        clust_clips = clips[curr_clust_bool, :]
        if clust_clips.ndim == 1:
            clust_clips = np.expand_dims(clust_clips, 0)
        if clust_clips.shape[0] == 1:
            # Only one spike so don't try to sort
            continue
        median_cluster_size = min(100, int(np.around(clust_clips.shape[0] / 1000)))

        # Re-cluster and sort using only clips from current cluster
        if method.lower() == 'pca':
            scores = preprocessing.compute_pca(clust_clips, check_components, max_components,
                        add_peak_valley=add_peak_valley, curr_chan_inds=curr_chan_inds)
        elif method.lower() == 'chan_pca':
            scores = preprocessing.compute_pca_by_channel(clust_clips, curr_chan_inds,
                        check_components, max_components, add_peak_valley=add_peak_valley)
        else:
            raise ValueError("Sharpen method must be either 'pca', or 'chan_pca'.")
        n_random = max(100, np.around(clust_clips.shape[0] / 100)) if use_rand_init else 0
        clust_labels = sort.initial_cluster_farthest(scores, median_cluster_size, n_random=n_random)
        clust_labels = sort.merge_clusters(scores, clust_labels,
                        p_value_cut_thresh=p_value_cut_thresh)
        new_labels = np.unique(clust_labels)
        if new_labels.size > 1:
            # Found at least one new cluster within original so reassign labels
            for nl in new_labels:
                temp_labels = neuron_labels_copy[curr_clust_bool]
                temp_labels[clust_labels == nl] = next_label
                neuron_labels_copy[curr_clust_bool] = temp_labels
                clusters_to_check.append(next_label)
                next_label += 1

    return neuron_labels_copy


def spike_sort(Probe, sigma=4.5, clip_width=[-6e-4, 10e-4],
               p_value_cut_thresh=0.01, check_components=None,
               max_components=10, min_firing_rate=1,
               do_binary_pursuit=True, add_peak_valley=False,
               do_ZCA_transform=True, do_branch_PCA=True,
               use_GPU=True, max_gpu_memory=None, use_rand_init=True,
               cleanup_neurons=False, verbose=False):

    """ Example:
    import electrode
    import SpikeSorting
    class TestProbe(electrode.AbstractProbe):
        def __init__(self, sampling_rate, voltage_array, num_channels=None):
            if num_channels is None:
                num_channels = voltage_array.shape[1]
            electrode.AbstractProbe.__init__(self, sampling_rate, num_channels, voltage_array=voltage_array)

        def get_neighbors(self, channel):
            # This is a tetrode that uses all other contacts as its neighbors
            start = 0
            stop = 4
            return np.arange(start, stop)

    spike_sort_kwargs = {'sigma': 3., 'clip_width': [-6e-4, 8e-4],
                         'p_value_cut_thresh': 0.01, 'max_components': None,
                         'min_firing_rate': 1, do_binary_pursuit=True,
                         'cleanup_neurons': False, 'verbose': True}
    Probe = TestProbe(samples_per_second, voltage_array, num_channels=4)
    neurons = SpikeSorting.spike_sort(Probe, **spike_sort_kwargs)
    """

    skip = np.amax(np.abs(clip_width)) / 2
    align_window = [skip, skip]
    if verbose: print("Finding thresholds")
    thresholds = segment.median_threshold(Probe, sigma=sigma)
    if do_ZCA_transform:
        if verbose: print("Doing ZCA transform")
        zca_cushion = (2 * np.ceil(np.amax(np.abs(clip_width)) * Probe.sampling_rate)).astype(np.int64)
        zca_matrix = preprocessing.get_noise_sampled_zca_matrix(Probe.voltage, thresholds, sigma, zca_cushion, n_samples=1e7)
        zca_voltage = zca_matrix @ Probe.voltage
        Probe.voltage = zca_voltage
        if verbose: print("Finding ZCA'ed thresholds")
        thresholds = segment.median_threshold(Probe, sigma=sigma)

    if verbose: print("Identifying threshold crossings")
    crossings = segment.identify_threshold_crossings(Probe, thresholds, skip=skip, align_window=align_window)
    min_cluster_size = (np.floor(min_firing_rate * Probe.n_samples / Probe.sampling_rate)).astype(np.int64)
    if min_cluster_size < 1:
        min_cluster_size = 1
    if verbose: print("Using minimum cluster size of", min_cluster_size)

    labels = [[] for x in range(0, Probe.num_electrodes)]
    waveforms  = [[] for x in range(0, Probe.num_electrodes)]
    new_waveforms = [[] for x in range(0, Probe.num_electrodes)]
    for chan in range(0, Probe.num_electrodes):
        if verbose: print("Working on electrode ", chan)
        median_cluster_size = min(100, int(np.around(crossings[chan].size / 1000)))
        if verbose: print("Getting clips")
        clips, valid_event_indices = segment.get_multichannel_clips(Probe, Probe.get_neighbors(chan), crossings[chan], clip_width=clip_width, thresholds=thresholds)
        crossings[chan] = segment.keep_valid_inds([crossings[chan]], valid_event_indices)
        _, _, clip_samples, _, curr_chan_inds = segment.get_windows_and_indices(clip_width, Probe.sampling_rate, chan, Probe.get_neighbors(chan))

        if verbose: print("Start initial clustering and merge")
        # Do initial single channel sort
        scores = preprocessing.compute_pca(clips[:, curr_chan_inds],
                    check_components, max_components, add_peak_valley=add_peak_valley,
                    curr_chan_inds=np.arange(0, curr_chan_inds.size))
        n_random = max(100, np.around(crossings[chan].size / 100)) if use_rand_init else 0
        neuron_labels = sort.initial_cluster_farthest(scores, median_cluster_size, n_random=n_random)
        neuron_labels = sort.merge_clusters(scores, neuron_labels,
                            split_only = False,
                            p_value_cut_thresh=p_value_cut_thresh)
        curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
        if verbose: print("Currently", curr_num_clusters.size, "different clusters")

        # Single channel branch
        if curr_num_clusters.size > 1 and do_branch_PCA:
            neuron_labels = branch_pca_2_0(neuron_labels, clips[:, curr_chan_inds],
                                np.arange(0, curr_chan_inds.size),
                                p_value_cut_thresh=p_value_cut_thresh,
                                add_peak_valley=add_peak_valley,
                                check_components=check_components,
                                max_components=max_components,
                                method='pca')
            curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
            if verbose: print("After SINGLE BRANCH", curr_num_clusters.size, "different clusters")

        # Multi channel branch
        if Probe.num_electrodes > 1 and do_branch_PCA:
            neuron_labels = branch_pca_2_0(neuron_labels, clips, curr_chan_inds,
                                p_value_cut_thresh=p_value_cut_thresh,
                                add_peak_valley=add_peak_valley,
                                check_components=check_components,
                                max_components=max_components,
                                method='pca')
            curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
            if verbose: print("After MULTI BRANCH", curr_num_clusters.size, "different clusters")

        # Multi channel branch by channel
        if Probe.num_electrodes > 1 and do_branch_PCA:
            neuron_labels = branch_pca_2_0(neuron_labels, clips, curr_chan_inds,
                                p_value_cut_thresh=p_value_cut_thresh,
                                add_peak_valley=add_peak_valley,
                                check_components=check_components,
                                max_components=max_components,
                                method='chan_pca')
            curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
            if verbose: print("After MULTI BY CHAN BRANCH", curr_num_clusters.size, "different clusters")

        # Delete any clusters under min_cluster_size before binary pursuit
        if verbose: print("Current smallest cluster has", np.amin(n_per_cluster), "spikes")
        if np.any(n_per_cluster < min_cluster_size):
            for l_ind in range(0, curr_num_clusters.size):
                if n_per_cluster[l_ind] < min_cluster_size:
                    keep_inds = ~(neuron_labels == curr_num_clusters[l_ind])
                    crossings[chan] = crossings[chan][keep_inds]
                    neuron_labels = neuron_labels[keep_inds]
                    clips = clips[keep_inds, :]
                    print("Deleted cluster", curr_num_clusters[l_ind], "with", n_per_cluster[l_ind], "spikes")

        if neuron_labels.size == 0:
            if verbose: print("No clusters over min_firing_rate")
            labels[chan] = neuron_labels
            waveforms[chan] = clips
            new_waveforms[chan] = np.array([])
            if verbose: print("Done.")
            continue

        # Realign any units that have a template with peak > valley
        crossings[chan], units_shifted = check_upward_neurons(clips,
                                            crossings[chan], neuron_labels,
                                            curr_chan_inds, clip_width,
                                            Probe.sampling_rate)
        if verbose: print("Found", len(units_shifted), "upward neurons that were realigned", flush=True)
        if len(units_shifted) > 0:
            clips, valid_event_indices = segment.get_multichannel_clips(Probe,
                                            Probe.get_neighbors(chan),
                                            crossings[chan],
                                            clip_width=clip_width,
                                            thresholds=thresholds)
            crossings[chan] = segment.keep_valid_inds(
                                [crossings[chan]], valid_event_indices)

        # Realign spikes based on correlation with current cluster templates before doing binary pursuit
        crossings[chan], neuron_labels, _ = segment.align_events_with_template(Probe, chan, neuron_labels, crossings[chan], clip_width=clip_width)
        if do_binary_pursuit:
            if verbose: print("currently", np.unique(neuron_labels).size, "different clusters")
            if verbose: print("Doing binary pursuit")
            if not use_GPU:
                use_GPU_message = ("Using CPU binary pursuit to find " \
                                    "secret spikes. This can be MUCH MUCH " \
                                    "slower and uses more " \
                                    "memory than the GPU version.")
                warnings.warn(use_GPU_message, RuntimeWarning, stacklevel=2)
                crossings[chan], neuron_labels, new_inds = overlap.binary_pursuit_secret_spikes(
                                        Probe, chan, neuron_labels, crossings[chan],
                                        thresholds[chan], clip_width)
                clips, valid_event_indices = segment.get_multichannel_clips(Probe, Probe.get_neighbors(chan), crossings[chan], clip_width=clip_width, thresholds=thresholds)
                crossings[chan], neuron_labels = segment.keep_valid_inds([crossings[chan], neuron_labels], valid_event_indices)
            else:
                crossings[chan], neuron_labels, new_inds, clips = binary_pursuit.binary_pursuit(
                    Probe, chan, crossings[chan], neuron_labels, clip_width,
                    thresholds=thresholds, kernels_path=None,
                    max_gpu_memory=max_gpu_memory)
        else:
            # Need to get newly aligned clips and new_inds = False
            clips, valid_event_indices = segment.get_multichannel_clips(Probe, Probe.get_neighbors(chan), crossings[chan], clip_width=clip_width, thresholds=thresholds)
            crossings[chan], neuron_labels = segment.keep_valid_inds([crossings[chan], neuron_labels], valid_event_indices)
            new_inds = np.zeros(crossings[chan].size, dtype=np.bool)

        if verbose: print("currently", np.unique(neuron_labels).size, "different clusters")
        # Map labels starting at zero and put labels in order
        sort.reorder_labels(neuron_labels)
        labels[chan] = neuron_labels
        waveforms[chan] = clips
        new_waveforms[chan] = new_inds
        if verbose: print("Done.")

    # Now that everything has been sorted, condense our representation of the neurons
    if verbose: print("Summarizing neurons")
    neurons = consolidate.summarize_neurons(Probe, crossings, labels, waveforms, thresholds, clip_width=clip_width, new_waveforms=new_waveforms, max_components=max_components)

    if verbose: print("Ordering neurons and finding peak valleys")
    neurons = consolidate.recompute_template_wave_properties(neurons)
    neurons = consolidate.reorder_neurons_by_raw_peak_valley(neurons)

    # Consolidate neurons across channels
    if cleanup_neurons:
        if verbose: print("Removing noise units")
        neurons = consolidate.remove_noise_neurons(neurons)
        if verbose: print("Combining same neurons in neighborhood")
        neurons = consolidate.combine_stolen_spikes(neurons, max_offset_samples=20, p_value_combine=.05, p_value_cut_thresh=p_value_cut_thresh, max_components=max_components)
        neurons = consolidate.combine_neighborhood_neurons(neurons, overlap_time=5e-4, overlap_threshold=5, max_offset_samples=10, p_value_cut_thresh=p_value_cut_thresh, max_components=max_components)
        neurons = consolidate.recompute_template_wave_properties(neurons)
        neurons = consolidate.reorder_neurons_by_raw_peak_valley(neurons)
        neurons = consolidate.remove_across_channel_duplicate_neurons(neurons, overlap_time=5e-4, overlap_threshold=5)

    if verbose: print("Done.")
    print("YOU WANT THIS TO SAVE THINGS LIKE THE SIGMA VALUE USED, AND THE MEDIAN DEVIATION (FOR COMPUTING SNR) AND REALLY ALL PARAMETERS")
    print("You want to use here, and in consolidate the SNR based on the threshold value found divided by SIGMA instead of Matt method.")
    print("David did this by just passing the input kwgs into a settings dictionary for the output")
    print("Also I need to make sure that all ints are np.int64 so this is compatible with windows compiled sort_cython, including in the segment files etc...")
    print("Also need to delete the load file functionality in electrode.py since it only assumes .npy file")
    return neurons
