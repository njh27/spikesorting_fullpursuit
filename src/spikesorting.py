import numpy as np
from spikesorting_python.src import segment
from spikesorting_python.src import preprocessing
from spikesorting_python.src import sort
from spikesorting_python.src import overlap
from spikesorting_python.src import binary_pursuit
from spikesorting_python.src import consolidate



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


def sharpen_clusters(clips, neuron_labels, curr_chan_inds, p_value_cut_thresh,
                     merge_only=False, add_peak_valley=False,
                     check_components=None, max_components=None,
                     max_iters=np.inf, method='pca'):
    """
    """
    n_labels = np.unique(neuron_labels).size
    print("Entering sharpen with", n_labels, "different clusters", flush=True)
    n_iters = 0
    while (n_labels > 1) and (n_iters < max_iters):
        if method.lower() == 'projection':
            scores = preprocessing.compute_template_projection(clips, neuron_labels,
                        curr_chan_inds, add_peak_valley=add_peak_valley,
                        max_templates=max_components)
        elif method.lower() == 'pca':
            scores = preprocessing.compute_template_pca(clips, neuron_labels,
                        curr_chan_inds, check_components, max_components,
                        add_peak_valley=add_peak_valley)
        elif method.lower() == 'chan_pca':
            scores = preprocessing.compute_template_pca_by_channel(clips, neuron_labels,
                        curr_chan_inds, check_components, max_components,
                        add_peak_valley=add_peak_valley)
        else:
            raise ValueError("Sharpen method must be either 'projection', 'pca', or 'chan_pca'.")

        neuron_labels = sort.merge_clusters(scores, neuron_labels,
                            merge_only=merge_only, split_only=False,
                            p_value_cut_thresh=p_value_cut_thresh)
        n_labels_sharpened = np.unique(neuron_labels).size
        print("Went from", n_labels, "to", n_labels_sharpened, flush=True)
        if n_labels == n_labels_sharpened:
            break
        n_labels = n_labels_sharpened
        n_iters += 1

    return neuron_labels


def branch_pca_2_0(neuron_labels, clips, curr_chan_inds, p_value_cut_thresh=0.01,
                    add_peak_valley=False, check_components=None,
                    max_components=None, method='pca'):
    """
    """

    neuron_labels_copy = np.copy(neuron_labels)
    clusters_to_check = [ol for ol in np.unique(neuron_labels_copy)]
    next_label = int(np.amax(clusters_to_check) + 1)
    # p_value_cut_thresh = p_value_cut_thresh / len(clusters_to_check)
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
        clust_labels = sort.initial_cluster_farthest(scores, median_cluster_size)
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
               cleanup_neurons=False, verbose=False,
               remove_false_positives=False):

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
                         'cleanup_neurons': False, 'verbose': True,
                         'remove_false_positives': False}
    Probe = TestProbe(samples_per_second, voltage_array, num_channels=4)
    neurons = SpikeSorting.spike_sort(Probe, **spike_sort_kwargs)
    """

    skip = np.amax(np.abs(clip_width)) / 2
    align_window = [skip, skip]
    if verbose: print("Finding thresholds")
    thresholds = segment.median_threshold(Probe, sigma=sigma)
    if do_ZCA_transform and Probe.num_electrodes > 1:
        if verbose: print("Doing ZCA transform")
        zca_cushion = int(2 * np.ceil(np.amax(np.abs(clip_width)) * Probe.sampling_rate))
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
    false_positives = [[] for x in range(0, Probe.num_electrodes)]
    false_negatives = [[] for x in range(0, Probe.num_electrodes)]
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
        neuron_labels = sort.initial_cluster_farthest(scores, median_cluster_size)
        neuron_labels = sort.merge_clusters(scores, neuron_labels,
                            split_only = False,
                            p_value_cut_thresh=p_value_cut_thresh)
        curr_num_clusters = np.unique(neuron_labels).size
        if verbose: print("Currently", curr_num_clusters, "different clusters")

        # # Do multi channel sort, all channels considered together SPLIT ONLY
        # if Probe.num_electrodes > 1:
        #     scores = preprocessing.compute_pca(clips, check_components, max_components,
        #                 add_peak_valley=add_peak_valley, curr_chan_inds=curr_chan_inds)
        #     # neuron_labels = sort.initial_cluster_farthest(scores, median_cluster_size)
        #     neuron_labels = sort.merge_clusters(scores, neuron_labels,
        #                         split_only = True,
        #                         p_value_cut_thresh=p_value_cut_thresh)
        #     curr_num_clusters = np.unique(neuron_labels).size
        #     if verbose: print("Currently", curr_num_clusters, "different clusters")
        #
        # # Do multi channel sort, each channel considered separately SPLIT ONLY
        # if Probe.num_electrodes > 1:
        #     scores = preprocessing.compute_pca_by_channel(clips, curr_chan_inds,
        #                 check_components, max_components, add_peak_valley=add_peak_valley)
        #     neuron_labels = sort.merge_clusters(scores, neuron_labels,
        #                         split_only = True,
        #                         p_value_cut_thresh=p_value_cut_thresh)
        #     curr_num_clusters = np.unique(neuron_labels).size
        #     if verbose: print("After PCA by channel", curr_num_clusters, "different clusters")

        # Single channel branch
        if curr_num_clusters > 1 and do_branch_PCA:
            neuron_labels = branch_pca_2_0(neuron_labels, clips[:, curr_chan_inds],
                                np.arange(0, curr_chan_inds.size),
                                p_value_cut_thresh=p_value_cut_thresh,
                                add_peak_valley=add_peak_valley,
                                check_components=check_components,
                                max_components=max_components,
                                method='pca')
            curr_num_clusters = np.unique(neuron_labels).size
            if verbose: print("After SINGLE BRANCH", curr_num_clusters, "different clusters")

        # Multi channel branch
        if Probe.num_electrodes > 1 and do_branch_PCA:
            neuron_labels = branch_pca_2_0(neuron_labels, clips, curr_chan_inds,
                                p_value_cut_thresh=p_value_cut_thresh,
                                add_peak_valley=add_peak_valley,
                                check_components=check_components,
                                max_components=max_components,
                                method='pca')
            curr_num_clusters = np.unique(neuron_labels).size
            if verbose: print("After MULTI BRANCH", curr_num_clusters, "different clusters")

        # Multi channel branch by channel
        if Probe.num_electrodes > 1 and do_branch_PCA:
            neuron_labels = branch_pca_2_0(neuron_labels, clips, curr_chan_inds,
                                p_value_cut_thresh=p_value_cut_thresh,
                                add_peak_valley=add_peak_valley,
                                check_components=check_components,
                                max_components=max_components,
                                method='chan_pca')
            curr_num_clusters = np.unique(neuron_labels).size
            if verbose: print("After MULTI BY CHAN BRANCH", curr_num_clusters, "different clusters")

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

        # if verbose: print("Sharpening clips onto templates")
        # neuron_labels = sharpen_clusters(clips, neuron_labels, curr_chan_inds,
        #                     p_value_cut_thresh, merge_only=True,
        #                     add_peak_valley=add_peak_valley,
        #                     check_components=check_components,
        #                     max_components=max_components, max_iters=1,
        #                     method='pca')

        # # Delete any clusters under min_cluster_size before binary pursuit
        # unique_vals, n_unique = np.unique(neuron_labels, return_counts=True)
        # if verbose: print("Current smallest cluster has", np.amin(n_unique), "spikes")
        # n_small_clusters = 0
        # if np.any(n_unique < min_cluster_size):
        #     for l_ind in range(0, unique_vals.size):
        #         if n_unique[l_ind] < min_cluster_size:
        #             keep_inds = ~(neuron_labels == unique_vals[l_ind])
        #             crossings[chan] = crossings[chan][keep_inds]
        #             neuron_labels = neuron_labels[keep_inds]
        #             clips = clips[keep_inds, :]
        #             n_small_clusters += 1
        # if verbose: print("Deleted", n_small_clusters, "small clusters under min_cluster_size")
        #
        # if neuron_labels.size == 0:
        #     if verbose: print("No clusters over min_firing_rate")
        #     false_positives[chan] = np.zeros(unique_labels.size)
        #     false_negatives[chan] = np.zeros(unique_labels.size)
        #     labels[chan] = neuron_labels
        #     waveforms[chan] = clips
        #     new_waveforms[chan] = np.array([])
        #     if verbose: print("Done.")
        #     continue
        #
        # if verbose: print("Sharpening clips onto templates")
        # neuron_labels = sharpen_clusters(clips, neuron_labels, curr_chan_inds,
        #                     p_value_cut_thresh, merge_only=False,
        #                     add_peak_valley=add_peak_valley,
        #                     check_components=check_components,
        #                     max_components=max_components, max_iters=1,
        #                     method='projection')

        # Realign spikes based on correlation with current cluster templates before doing binary pursuit
        crossings[chan], neuron_labels, _ = segment.align_events_with_template(Probe, chan, neuron_labels, crossings[chan], clip_width=clip_width)
        clips, valid_event_indices = segment.get_multichannel_clips(Probe, Probe.get_neighbors(chan), crossings[chan], clip_width=clip_width, thresholds=thresholds)
        crossings[chan], neuron_labels = segment.keep_valid_inds([crossings[chan], neuron_labels], valid_event_indices)

        if verbose: print("Finished first cluster and merge")

        if do_binary_pursuit:
            if verbose: print("currently", np.unique(neuron_labels).size, "different clusters")
            if verbose: print("Doing binary pursuit")
            # crossings[chan], neuron_labels, new_inds = overlap.binary_pursuit_secret_spikes(
            #                         Probe, chan, neuron_labels, crossings[chan],
            #                         thresholds[chan], clip_width,
            #                         return_adjusted_clips=False)
            # clips, valid_event_indices = segment.get_multichannel_clips(Probe, Probe.get_neighbors(chan), crossings[chan], clip_width=clip_width, thresholds=thresholds)
            # crossings[chan], neuron_labels = segment.keep_valid_inds([crossings[chan], neuron_labels], valid_event_indices)

            crossings[chan], neuron_labels, new_inds, clips = binary_pursuit.binary_pursuit(
                Probe, chan, crossings[chan], neuron_labels, clip_width,
                thresholds[chan], kernels_path=None, max_gpu_memory=None)

            if verbose: print("currently", np.unique(neuron_labels).size, "different clusters")
        else:
            new_inds = np.zeros(crossings[chan].size, dtype=np.bool)

        # if verbose: print("Sharpening clips onto templates")
        # neuron_labels = sharpen_clusters(clips, neuron_labels, curr_chan_inds,
        #                     p_value_cut_thresh, merge_only=True,
        #                     add_peak_valley=add_peak_valley,
        #                     check_components=check_components,
        #                     max_components=max_components, max_iters=np.inf,
        #                     method='pca')

        # if verbose: print("Sharpening single channel clips onto templates")
        # neuron_labels = sharpen_clusters(clips[:, curr_chan_inds], neuron_labels,
        #                     np.arange(0, curr_chan_inds.size),
        #                     p_value_cut_thresh, merge_only=True,
        #                     add_peak_valley=add_peak_valley,
        #                     check_components=check_components,
        #                     max_components=max_components, max_iters=np.inf,
        #                     method='pca')

        if verbose: print("currently", np.unique(neuron_labels).size, "different clusters")
        if verbose: print("Done sorting")
        # Map labels starting at zero and put labels in order
        sort.reorder_labels(neuron_labels)
        unique_labels, unique_label_counts = np.unique(neuron_labels, return_counts=True)

        # Collect noise clips from all times
        # if verbose: print("Identifying noise segments")
        # noise_crossings = segment.make_noise_crossings(Probe, chan, crossings[chan], crossings[chan].size, clip_width=clip_width, skip=skip, align_window=align_window)
        # noise_clips, valid_event_indices = segment.get_multichannel_clips(Probe, Probe.get_neighbors(chan), noise_crossings, clip_width=clip_width, thresholds=thresholds)
        # noise_crossings = segment.keep_valid_inds([noise_crossings], valid_event_indices)
        # noise_crossings = np.hstack((noise_crossings, noise_crossings))
        # noise_clips = np.vstack((noise_clips, -1* noise_clips))

        # noise_labels = np.hstack((np.zeros(noise_crossings.size, dtype='int'), np.ones(noise_crossings.size, dtype='int')))
        # noise_labels = sort.merge_clusters(noise_clips, noise_labels, sharpen=False, p_value_cut_thresh=p_value_cut_thresh, max_components=use_components)


        # Put noise labels in order from 0 to number of unique noise labels
        # sort.reorder_labels(noise_labels)
        # unique_noise_labels, noise_label_counts = np.unique(noise_labels, return_counts=True)

        # Now that we have our noise labels, we are going to try a merge with
        # each signal cluster individually, calculating the false positive
        # and false negative ratio for each
        false_positives[chan] = np.zeros(unique_labels.size)
        false_negatives[chan] = np.zeros(unique_labels.size)
        # is_false_positive = np.zeros(neuron_labels.size, dtype='bool')
        # for j, unique_label in enumerate(unique_labels):
        #     # Create a new noise cluster that is exactly equal to the number
        #     # of clips in the current signal cluster.  Assign them arbitrary
        #     # labels as 0 or 1 and combine everything for sorting.
        #     current_signal_inds = np.nonzero(neuron_labels == unique_label)[0]
        #     current_noise_inds = np.random.choice(np.arange(0, noise_clips.shape[0]), unique_label_counts[j], replace=True)
        #
        #     noise_label = 0
        #     signal_label = 1
        #     current_labels = np.full(2 * unique_label_counts[j], noise_label, dtype='int')
        #     current_labels[unique_label_counts[j]:] = signal_label
        #     current_clips = np.zeros((2 * unique_label_counts[j], clips.shape[1]))
        #
        #     current_clips[0:unique_label_counts[j], :] = noise_clips[current_noise_inds, :]
        #     current_clips[unique_label_counts[j]:, :] = clips[current_signal_inds, :]
        #     is_noise = np.zeros(current_labels.size, dtype='bool')
        #     is_noise[0:unique_label_counts[j]] = True
        #     print("!!! NEED TO CHOOSE NOISE COMPONENTS ???")
        #     scores = preprocessing.compute_pca_by_channel(clips, samples_per_chan, max_components, add_peak_valley=add_peak_valley)
        #     current_labels = sort.merge_clusters(current_clips, current_labels, p_value_cut_thresh=1.1, max_components=use_components)
        #     # current_labels = branch_pca_2_0(current_labels, current_clips, p_value_cut_thresh=p_value_cut_thresh, max_components=use_components)
        #
        #     if np.unique(current_labels).size < 2: # Catch the condition where all clips move to the signal label
        #         current_labels[:] = noise_label
        #
        #     # Find cluster with most signal clips and call it signal cluster
        #     n_signal = 0
        #     for c_label in np.unique(current_labels):
        #         n_current = np.count_nonzero(np.logical_and(~is_noise, current_labels == c_label))
        #         if n_current > n_signal:
        #             n_signal = n_current
        #             signal_label = c_label
        #
        #     # Compute our false positives and false negatives
        #     # False positives are the "signal" clips that clusters with noise
        #     # False negatives are noise clips that cluster with the "signal"
        #     false_positives[chan][j] = np.count_nonzero(np.logical_and(~is_noise, current_labels != signal_label)) / unique_label_counts[j]
        #     false_negatives[chan][j] = np.count_nonzero(np.logical_and(is_noise, current_labels == signal_label)) / unique_label_counts[j]
        #     is_false_positive[current_signal_inds[current_labels[unique_label_counts[j]:] != signal_label]] = True
        #
        # if remove_false_positives:
        #     neuron_labels = neuron_labels[~is_false_positive]
        #     crossings[chan] = crossings[chan][~is_false_positive]
        #     clips = clips[~is_false_positive, :]
        #     new_inds = new_inds[~is_false_positive]
        #     sort.reorder_labels(neuron_labels)
        #     unique_labels = np.unique(neuron_labels)

        if verbose: print("Done.")
        labels[chan] = neuron_labels
        waveforms[chan] = clips
        new_waveforms[chan] = new_inds

    # Now that everything has been sorted, condense our representation of the neurons
    if verbose: print("Summarizing neurons")
    neurons = consolidate.summarize_neurons(Probe, crossings, labels, waveforms, thresholds, false_positives, false_negatives, clip_width=clip_width, new_waveforms=new_waveforms, max_components=max_components)

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
