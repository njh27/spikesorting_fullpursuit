import numpy as np
from spikesorting_fullpursuit import segment, preprocessing, sort, binary_pursuit
import warnings
import copy



def spike_sorting_settings(**kwargs):
    settings = {}

    settings['sigma'] = 4.0 # Threshold based on  noise level
    settings['verbose'] = False
    settings['clip_width'] = [-6e-4, 10e-4]# Width of clip in seconds
    settings['do_branch_PCA'] = True # Use branch pca method to split clusters
    settings['do_branch_PCA_by_chan'] = True
    settings['filter_band'] = (300, 6000)
    settings['do_ZCA_transform'] = True
    settings['use_rand_init'] = True # Initial clustering uses at least some randomly chosen centers
    settings['add_peak_valley'] = False # Use peak valley in addition to PCs for sorting
    settings['check_components'] = None # Number of PCs to check. None means all, else integer
    settings['max_components'] = 10 # Max number to use, of those checked, integer
    settings['min_firing_rate'] = 1. # Neurons with fewer threshold crossings than this are removed
    settings['p_value_cut_thresh'] = 0.05
    settings['do_binary_pursuit'] = True
    settings['use_GPU'] = True # Force algorithms to run on the CPU rather than the GPU
    settings['max_gpu_memory'] = None # Use as much memory as possible
    settings['segment_duration'] = None # Seconds (nothing/Inf uses the entire recording)
    settings['segment_overlap'] = None # Seconds of overlap between adjacent segments
    settings['binary_pursuit_only'] = True # If true, all spikes are found and classified by binary pursuit
    settings['cleanup_neurons'] = False # Remove garbage at the end

    for k in kwargs.keys():
        if k not in settings:
            raise TypeError("Unknown parameter key {0}.".format(k))
        settings[k] = kwargs[k]

    return settings


"""
    Wavelet alignment can bounce back and forth based on noise blips if
    the spike clip is nearly symmetric in peak/valley. """
def check_spike_alignment(clips, event_indices, neuron_labels, curr_chan_inds,
                         settings):
    templates, labels = segment.calculate_templates(clips[:, curr_chan_inds], neuron_labels)
    any_merged = False
    unit_inds_to_check = [x for x in range(0, len(templates))]
    previously_aligned_dict = {}
    while len(unit_inds_to_check) > 1:
        # Find nearest cross corr template matched pair
        best_corr = -np.inf
        best_shift = 0
        for i in range(0, len(unit_inds_to_check)):
            for j in range(i + 1, len(unit_inds_to_check)):
                t_ind_1 = unit_inds_to_check[i]
                t_ind_2 = unit_inds_to_check[j]
                cross_corr = np.correlate(templates[t_ind_1],
                                          templates[t_ind_2], mode='full')
                max_corr_ind = np.argmax(cross_corr)
                if cross_corr[max_corr_ind] > best_corr:
                    best_corr = cross_corr[max_corr_ind]
                    best_shift = max_corr_ind - cross_corr.shape[0]//2
                    best_pair_inds = [t_ind_1, t_ind_2]

        # Get clips for best pair and optimally align them with each other
        select_n_1 = neuron_labels == labels[best_pair_inds[0]]
        select_n_2 = neuron_labels == labels[best_pair_inds[1]]
        clips_1 = clips[select_n_1, :][:, curr_chan_inds]
        clips_2 = clips[select_n_2, :][:, curr_chan_inds]

        # Align and truncate clips for best match pair
        if best_shift > 0:
            clips_1 = clips_1[:, best_shift:]
            clips_2 = clips_2[:, :-1*best_shift]
        elif best_shift < 0:
            clips_1 = clips_1[:, :best_shift]
            clips_2 = clips_2[:, -1*best_shift:]
        else:
            # No need to shift, or even check these further
            if clips_1.shape[0] >= clips_2.shape[0]:
                unit_inds_to_check.remove(best_pair_inds[1])
            else:
                unit_inds_to_check.remove(best_pair_inds[0])
            continue
        # Check if the main merges with its best aligned leftover
        combined_clips = np.vstack((clips_1, clips_2))
        pseudo_labels = np.ones(combined_clips.shape[0], dtype=np.int64)
        pseudo_labels[clips_1.shape[0]:] = 2
        scores = preprocessing.compute_pca(combined_clips,
                    settings['check_components'], settings['max_components'],
                    add_peak_valley=settings['add_peak_valley'],
                    curr_chan_inds=np.arange(0, combined_clips.shape[1]))
        pseudo_labels = sort.merge_clusters(scores, pseudo_labels,
                            split_only = False, merge_only=True,
                            p_value_cut_thresh=settings['p_value_cut_thresh'])
        if np.all(pseudo_labels == 1) or np.all(pseudo_labels == 2):
            any_merged = True
            if clips_1.shape[0] >= clips_2.shape[0]:
                # Align all neuron 2 clips with neuron 1 template
                event_indices[select_n_2] += -1*best_shift
                unit_inds_to_check.remove(best_pair_inds[1])
                if best_pair_inds[1] in previously_aligned_dict:
                    for unit in previously_aligned_dict[best_pair_inds[1]]:
                        select_unit = neuron_labels == unit
                        event_indices[select_unit] += -1*best_shift
                if best_pair_inds[0] not in previously_aligned_dict:
                    previously_aligned_dict[best_pair_inds[0]] = []
                previously_aligned_dict[best_pair_inds[0]].append(best_pair_inds[1])
            else:
                # Align all neuron 1 clips with neuron 2 template
                event_indices[select_n_1] += best_shift
                unit_inds_to_check.remove(best_pair_inds[0])
                # Check if any previous units are tied to this one and should
                # also shift
                if best_pair_inds[0] in previously_aligned_dict:
                    for unit in previously_aligned_dict[best_pair_inds[0]]:
                        select_unit = neuron_labels == unit
                        event_indices[select_unit] += best_shift
                # Make this unit follow neuron 1 in the event neuron 1 changes
                # in a future iteration
                if best_pair_inds[1] not in previously_aligned_dict:
                    previously_aligned_dict[best_pair_inds[1]] = []
                previously_aligned_dict[best_pair_inds[1]].append(best_pair_inds[0])
        else:
            unit_inds_to_check.remove(best_pair_inds[0])
            unit_inds_to_check.remove(best_pair_inds[1])

    return event_indices, any_merged


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
        if clust_clips.shape[0] <= 1:
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
            raise ValueError("Branch method must be either 'pca', or 'chan_pca'.")
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


def spike_sort_item(Probe, work_item, settings):
    """
    """
    chan = work_item['channel']

    skip = np.amax(np.abs(settings['clip_width'])) / 2
    align_window = [skip, skip]
    if settings['verbose']: print("Identifying threshold crossings")
    crossings = segment.identify_threshold_crossings(Probe, chan, work_item['thresholds'][chan], skip=skip, align_window=align_window)
    if crossings.size == 0:
        if settings['verbose']: print("No crossings over threshold.")
        if settings['verbose']: print("Done.")
        return [], [], [], []
    min_cluster_size = (np.floor(settings['min_firing_rate'] * Probe.n_samples / Probe.sampling_rate)).astype(np.int64)
    if min_cluster_size < 1:
        min_cluster_size = 1
    if settings['verbose']: print("Using minimum cluster size of", min_cluster_size)
    _, _, clip_samples, _, curr_chan_inds = segment.get_windows_and_indices(settings['clip_width'], Probe.sampling_rate, chan, work_item['neighbors'])

    # Realign spikes based on a common wavelet
    crossings = segment.wavelet_align_events(Probe, chan, crossings,
                                        settings['clip_width'],
                                        settings['filter_band'])

    median_cluster_size = min(100, int(np.around(crossings.size / 1000)))
    if settings['verbose']: print("Getting clips")
    clips, valid_event_indices = segment.get_multichannel_clips(Probe, work_item['neighbors'], crossings, clip_width=settings['clip_width'])
    crossings = segment.keep_valid_inds([crossings], valid_event_indices)

    if settings['verbose']: print("Start initial clustering and merge")
    # Do initial single channel sort. Start with single channel only because
    # later branching can split things out using multichannel info, but it
    # can't put things back together again
    if crossings.size > 1:
        scores = preprocessing.compute_pca(clips[:, curr_chan_inds],
                    settings['check_components'], settings['max_components'], add_peak_valley=settings['add_peak_valley'],
                    curr_chan_inds=np.arange(0, curr_chan_inds.size))
        n_random = max(100, np.around(crossings.size / 100)) if settings['use_rand_init'] else 0
        neuron_labels = sort.initial_cluster_farthest(scores, median_cluster_size, n_random=n_random)
        neuron_labels = sort.merge_clusters(scores, neuron_labels,
                            split_only = False,
                            p_value_cut_thresh=settings['p_value_cut_thresh'])

        curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
        if settings['verbose']: print("After first sort", curr_num_clusters.size, "different clusters", flush=True)

        crossings, neuron_labels, _ = segment.align_templates(Probe, chan, neuron_labels,
                            crossings, clip_width=settings['clip_width'])
        clips, valid_event_indices = segment.get_multichannel_clips(Probe,
                                        work_item['neighbors'],
                                        crossings,
                                        clip_width=settings['clip_width'])
        crossings, neuron_labels = segment.keep_valid_inds(
                [crossings, neuron_labels], valid_event_indices)

        scores = preprocessing.compute_pca(clips[:, curr_chan_inds],
                    settings['check_components'], settings['max_components'], add_peak_valley=settings['add_peak_valley'],
                    curr_chan_inds=np.arange(0, curr_chan_inds.size))
        n_random = max(100, np.around(crossings.size / 100)) if settings['use_rand_init'] else 0
        neuron_labels = sort.initial_cluster_farthest(scores, median_cluster_size, n_random=n_random)
        neuron_labels = sort.merge_clusters(scores, neuron_labels,
                            split_only = False,
                            p_value_cut_thresh=settings['p_value_cut_thresh'])

        crossings, neuron_labels, _ = segment.align_events_with_template(Probe,
                        chan, neuron_labels, crossings,
                        clip_width=settings['clip_width'])
        clips, valid_event_indices = segment.get_multichannel_clips(Probe,
                                        work_item['neighbors'],
                                        crossings,
                                        clip_width=settings['clip_width'])
        crossings, neuron_labels = segment.keep_valid_inds(
                [crossings, neuron_labels], valid_event_indices)

        curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
        if settings['verbose']: print("After re-sort", curr_num_clusters.size, "different clusters", flush=True)

        crossings, any_merged = check_spike_alignment(clips,
                        crossings, neuron_labels, curr_chan_inds, settings)
        if any_merged:
            # Resort based on new clip alignment
            if settings['verbose']: print("Re-sorting after check spike alignment")
            clips, valid_event_indices = segment.get_multichannel_clips(Probe, work_item['neighbors'], crossings, clip_width=settings['clip_width'])
            crossings = segment.keep_valid_inds([crossings], valid_event_indices)
            scores = preprocessing.compute_pca(clips[:, curr_chan_inds],
                        settings['check_components'], settings['max_components'], add_peak_valley=settings['add_peak_valley'],
                        curr_chan_inds=np.arange(0, curr_chan_inds.size))
            n_random = max(100, np.around(crossings.size / 100)) if settings['use_rand_init'] else 0
            neuron_labels = sort.initial_cluster_farthest(scores, median_cluster_size, n_random=n_random)
            neuron_labels = sort.merge_clusters(scores, neuron_labels,
                                split_only = False,
                                p_value_cut_thresh=settings['p_value_cut_thresh'])

        curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
    else:
        neuron_labels = np.zeros(1, dtype=np.int64)
        curr_num_clusters = np.zeros(1, dtype=np.int64)
    if settings['verbose']: print("Currently", curr_num_clusters.size, "different clusters")

    crossings, neuron_labels, _ = segment.align_events_with_template(Probe,
                    chan, neuron_labels, crossings,
                    clip_width=settings['clip_width'])
    clips, valid_event_indices = segment.get_multichannel_clips(Probe,
                                    work_item['neighbors'],
                                    crossings,
                                    clip_width=settings['clip_width'])
    crossings, neuron_labels = segment.keep_valid_inds(
            [crossings, neuron_labels], valid_event_indices)

    if settings['do_binary_pursuit']:
        # Remove deviant clips before doing branch PCA to avoid getting clusters
        # of overlaps or garbage
        keep_clips = preprocessing.cleanup_clusters(clips[:, curr_chan_inds], neuron_labels)
        crossings, neuron_labels = segment.keep_valid_inds(
                [crossings, neuron_labels], keep_clips)
        clips = clips[keep_clips, :]

    # Single channel branch
    if curr_num_clusters.size > 1 and settings['do_branch_PCA']:
        neuron_labels = branch_pca_2_0(neuron_labels, clips[:, curr_chan_inds],
                            np.arange(0, curr_chan_inds.size),
                            p_value_cut_thresh=settings['p_value_cut_thresh'],
                            add_peak_valley=settings['add_peak_valley'],
                            check_components=settings['check_components'],
                            max_components=settings['max_components'],
                            use_rand_init=settings['use_rand_init'],
                            method='pca')
        curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
        if settings['verbose']: print("After SINGLE BRANCH", curr_num_clusters.size, "different clusters")

    if settings['do_branch_PCA'] and settings['do_binary_pursuit']:
        # Remove deviant clips before doing branch PCA to avoid getting clusters
        # over overlaps or garbage, this time on full neighborhood
        keep_clips = preprocessing.cleanup_clusters(clips, neuron_labels)
        crossings, neuron_labels = segment.keep_valid_inds(
                [crossings, neuron_labels], keep_clips)
        clips = clips[keep_clips, :]

    # Multi channel branch
    if Probe.num_channels > 1 and settings['do_branch_PCA']:
        neuron_labels = branch_pca_2_0(neuron_labels, clips, curr_chan_inds,
                            p_value_cut_thresh=settings['p_value_cut_thresh'],
                            add_peak_valley=settings['add_peak_valley'],
                            check_components=settings['check_components'],
                            max_components=settings['max_components'],
                            use_rand_init=settings['use_rand_init'],
                            method='pca')
        curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
        if settings['verbose']: print("After MULTI BRANCH", curr_num_clusters.size, "different clusters")

    # Multi channel branch by channel
    if Probe.num_channels > 1 and settings['do_branch_PCA_by_chan'] and settings['do_branch_PCA']:
        neuron_labels = branch_pca_2_0(neuron_labels, clips, curr_chan_inds,
                            p_value_cut_thresh=settings['p_value_cut_thresh'],
                            add_peak_valley=settings['add_peak_valley'],
                            check_components=settings['check_components'],
                            max_components=settings['max_components'],
                            use_rand_init=settings['use_rand_init'],
                            method='chan_pca')
        curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
        if settings['verbose']: print("After MULTI BY CHAN BRANCH", curr_num_clusters.size, "different clusters")

    # Delete any clusters under min_cluster_size before binary pursuit
    if settings['verbose']: print("Current smallest cluster has", np.amin(n_per_cluster), "spikes")
    if np.any(n_per_cluster < min_cluster_size):
        for l_ind in range(0, curr_num_clusters.size):
            if n_per_cluster[l_ind] < min_cluster_size:
                keep_inds = ~(neuron_labels == curr_num_clusters[l_ind])
                crossings = crossings[keep_inds]
                neuron_labels = neuron_labels[keep_inds]
                clips = clips[keep_inds, :]
                print("Deleted cluster", curr_num_clusters[l_ind], "with", n_per_cluster[l_ind], "spikes")

    if neuron_labels.size == 0:
        if settings['verbose']: print("No clusters over min_firing_rate")
        if settings['verbose']: print("Done.")
        return [], [], [], []

    # Realign spikes based on correlation with current cluster templates before doing binary pursuit
    crossings, neuron_labels, _ = segment.align_events_with_template(Probe, chan, neuron_labels, crossings, clip_width=settings['clip_width'])
    if settings['do_binary_pursuit']:

        # keep_clips = preprocessing.keep_cluster_centroid(clips, neuron_labels, n_keep=settings['binary_pursuit_only'])
        # crossings, neuron_labels = segment.keep_valid_inds(
        #         [crossings, neuron_labels], keep_clips)

        if settings['verbose']: print("currently", np.unique(neuron_labels).size, "different clusters")
        if settings['verbose']: print("Doing binary pursuit")
        crossings, neuron_labels, bp_bool, clips = binary_pursuit.binary_pursuit(
            Probe, chan, crossings, neuron_labels, settings['clip_width'],
            thresh_sigma=1.645, # Normal dists are - 95: 1.645;  97.5: 1.96; 99: 2.326
            find_all=settings['binary_pursuit_only'],
            kernels_path=None, max_gpu_memory=settings['max_gpu_memory'])
    else:
        # Need to get newly aligned clips and bp_bool = False
        clips, valid_event_indices = segment.get_multichannel_clips(Probe, work_item['neighbors'], crossings, clip_width=settings['clip_width'])
        crossings, neuron_labels = segment.keep_valid_inds([crossings, neuron_labels], valid_event_indices)
        bp_bool = np.zeros(crossings.size, dtype=np.bool)

    if len(neuron_labels) == 0:
        # Nothing found in binary pursuit, probably with binary_pursuit_only == True
        if settings['verbose']: print("No clusters over min_firing_rate")
        if settings['verbose']: print("Done.")
        return [], [], [], []

    if settings['verbose']: print("currently", np.unique(neuron_labels).size, "different clusters")
    # Adjust crossings for segment start time
    crossings += work_item['index_window'][0]
    # Map labels starting at zero
    sort.reorder_labels(neuron_labels)
    if settings['verbose']: print("Done.")

    return crossings, neuron_labels, clips, bp_bool


def spike_sort(Probe, **kwargs):
    """

    Note: Clips and voltages will be output in the data type Probe.v_dtype.
    However, most of the arithmetic is computed in np.float64. Clips are cast
    as np.float64 for determining PCs and cast back when done. All of binary
    pursuit is conducted as np.float32 for memory and GPU compatibility.

    See 'spike_sorting_settings' above for a list of allowable kwargs.
    Example:
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
    neurons = fbp.spike_sort(Probe, **spike_sort_kwargs)
    """
    # Get our settings
    settings = spike_sorting_settings(**kwargs)
    if settings['binary_pursuit_only'] and not settings['use_GPU']:
        raise ValueError("Running binary pursuit only without using GPU is not implemented because it would take forever. Must set use_GPU to True if binary_pursuit_only is True.")
    if settings['binary_pursuit_only'] and not settings['do_binary_pursuit']:
        raise ValueError("Running binary pursuit only implies do_binary_pursuit is True, but do_binary_pursuit was input as 'False'.")
    # Check that Probe neighborhood function is appropriate. Otherwise it can
    # generate seemingly mysterious errors
    try:
        check_neighbors = Probe.get_neighbors(0)
    except:
        raise ValueError("Input Probe object must have a valid get_neighbors() method.")
    if type(check_neighbors) != np.ndarray:
        raise ValueError("Probe get_neighbors() method must return a numpy ndarray of dtype np.int64.")
    elif check_neighbors.dtype != np.int64:
        raise ValueError("Probe get_neighbors() method must return a numpy ndarray of dtype np.int64.")
    # For convenience, necessary to define clip width as negative for first entry
    if settings['clip_width'][0] > 0:
        settings['clip_width'] *= -1
    if not settings['use_GPU'] and settings['do_binary_pursuit']:
        use_GPU_message = ("Using CPU binary pursuit to find " \
                            "secret spikes. This can be MUCH MUCH " \
                            "slower and uses more " \
                            "memory than the GPU version. Returned " \
                            "clips will NOT be adjusted.")
        warnings.warn(use_GPU_message, RuntimeWarning, stacklevel=2)

    # Convert segment duration and overlaps to indices from their values input
    # in seconds and adjust as needed
    if (settings['segment_duration'] is None) or (settings['segment_duration'] == np.inf) \
        or (settings['segment_duration'] * Probe.sampling_rate >= Probe.n_samples):
        settings['segment_overlap'] = 0
        settings['segment_duration'] = Probe.n_samples
    else:
        if settings['segment_overlap'] is None:
            # If segment is specified with no overlap, use minimal overlap that
            # will not miss spikes on the edges
            clip_samples = segment.time_window_to_samples(settings['clip_width'], Probe.sampling_rate)[0]
            settings['segment_overlap'] = int(3 * (clip_samples[1] - clip_samples[0]))
        else:
            settings['segment_overlap'] = int(np.ceil(settings['segment_overlap'] * Probe.sampling_rate))
        input_duration_seconds = settings['segment_duration']
        settings['segment_duration'] = int(np.floor(settings['segment_duration'] * Probe.sampling_rate))
        if settings['segment_overlap'] >= settings['segment_duration']:
            raise ValueError("Segment overlap must be <= segment duration.")
        # Minimum number of segments at current segment duration and overlap
        # needed to cover all samples. Using floor will round to find the next
        # multiple that is >= the input segment duration.
        n_segs = np.floor((Probe.n_samples - settings['segment_duration'])
                          / (settings['segment_duration'] - settings['segment_overlap']))
        # Modify segment duration to next larger multiple of recording duration
        # given fixed, unaltered input overlap duration
        settings['segment_duration'] = int(np.ceil((Probe.n_samples
                        + n_segs * settings['segment_overlap']) / (n_segs + 1)))
        print("Input segment duration was rounded from", input_duration_seconds, "up to", settings['segment_duration']/Probe.sampling_rate, "seconds to make segments equal length.")

    segment_onsets = []
    segment_offsets = []
    curr_onset = 0
    while (curr_onset < Probe.n_samples):
        segment_onsets.append(curr_onset)
        segment_offsets.append(min(curr_onset + settings['segment_duration'], Probe.n_samples))
        if segment_offsets[-1] >= Probe.n_samples:
            break
        curr_onset += settings['segment_duration'] - settings['segment_overlap']
    print("Using ", len(segment_onsets), "segments per channel for sorting.")

    if settings['do_ZCA_transform']:
        zca_cushion = (2 * np.ceil(np.amax(np.abs(settings['clip_width'])) \
                         * Probe.sampling_rate)).astype(np.int64)
        # thresholds = segment.median_threshold(Probe.voltage, settings['sigma'])
        # zca_matrix = preprocessing.get_noise_sampled_zca_matrix(Probe.voltage,
        #                 thresholds, settings['sigma'],
        #                 zca_cushion, n_samples=1e7)

    # Build the sorting work items
    segment_voltages = []
    work_items = []
    chan_neighbors = []
    chan_neighbor_inds = []
    for x in range(0, len(segment_onsets)):
        if settings['verbose']: print("Finding voltage and thresholds for segment", x+1, "of", len(segment_onsets))
        # Need to copy or else ZCA transforms will duplicate in overlapping
        # time segments. Copy happens during matrix multiplication
        # Slice over num_channels should keep same shape
        seg_voltage = Probe.voltage[0:Probe.num_channels,
                                   segment_onsets[x]:segment_offsets[x]]
        if settings['do_ZCA_transform']:
            if settings['verbose']: print("Doing ZCA transform")
            thresholds = segment.median_threshold(seg_voltage, settings['sigma'])
            zca_matrix = preprocessing.get_noise_sampled_zca_matrix(seg_voltage,
                            thresholds, settings['sigma'],
                            zca_cushion, n_samples=1e6)
            # @ makes new copy
            seg_voltage = (zca_matrix @ seg_voltage).astype(Probe.v_dtype)
        thresholds = segment.median_threshold(seg_voltage, settings['sigma'])
        segment_voltages.append(seg_voltage)
        for chan in range(0, Probe.num_channels):
            # Ensure we just get neighbors once in case its complicated
            if x == 0:
                chan_neighbors.append(Probe.get_neighbors(chan))
                cn_ind = next((idx[0] for idx, val in np.ndenumerate(chan_neighbors[chan]) if val == chan), None)
                if cn_ind is None:
                    raise ValueError("Probe get_neighbors(chan) function must return a neighborhood that includes the channel 'chan'.")
                chan_neighbor_inds.append(cn_ind)
            work_items.append({'channel': chan,
                               'neighbors': chan_neighbors[chan],
                               'chan_neighbor_ind': chan_neighbor_inds[chan],
                               'n_samples': segment_offsets[x] - segment_onsets[x],
                               'seg_number': x,
                               'index_window': [segment_onsets[x], segment_offsets[x]],
                               'overlap': settings['segment_overlap'],
                               'thresholds': thresholds,
                               })

    sort_data = []
    # Put the work items through the sorter
    for wi_ind, w_item in enumerate(work_items):
        if settings['verbose']: print("Working on item {0}/{1} on channel {2} segment {3}".format(wi_ind+1, len(work_items), w_item['channel'], w_item['seg_number']))
        # Create a probe copy specific to this segment
        segProbe = copy.copy(Probe) # Only shallow copy then reassign stuff that changes
        segProbe.n_samples = w_item['n_samples']
        segProbe.voltage = segment_voltages[w_item['seg_number']]
        w_item['ID'] = wi_ind # Assign ID number in order of deployment
        crossings, labels, clips, bp_bool = spike_sort_item(segProbe, w_item, settings)
        sort_data.append([crossings, labels, clips, bp_bool, w_item['ID']])

    sort_info = settings
    curr_chan_win, _ = segment.time_window_to_samples(
                                    settings['clip_width'], Probe.sampling_rate)
    sort_info.update({'n_samples': Probe.n_samples,
                        'n_channels': Probe.num_channels,
                        'n_samples_per_chan': curr_chan_win[1] - curr_chan_win[0],
                        'sampling_rate': Probe.sampling_rate,
                        'n_segments': len(segment_onsets)})

    if settings['verbose']: print("Done.")

    print("Also need to delete the load file functionality in electrode.py since it only assumes .npy file")
    return sort_data, work_items, sort_info
