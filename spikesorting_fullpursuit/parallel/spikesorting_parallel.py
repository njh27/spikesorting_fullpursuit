import pickle
import os
import sys
sys.path.append(os.getcwd())
from shutil import rmtree
import mkl
import numpy as np
import multiprocessing as mp
import psutil
import time
from traceback import print_tb
from spikesorting_fullpursuit.parallel import segment_parallel
from spikesorting_fullpursuit import sort, preprocessing, full_binary_pursuit
from spikesorting_fullpursuit.parallel import binary_pursuit_parallel



def spike_sorting_settings_parallel(**kwargs):
    settings = {
        'sigma': 4.0, # Threshold based on noise level
        'clip_width': [-8e-4, 8e-4], # Width of clip in seconds
        'p_value_cut_thresh': 0.01, # Statistical criterion for splitting clusters during iso-cut
        segment_duration': 300, # Seconds (None/Inf uses the entire recording) Can be increased but not decreased by sorter to be same size
        'segment_overlap': 150, # Seconds of overlap between adjacent segments
        'do_branch_PCA': True, # Use branch PCA method to split clusters
        'do_branch_PCA_by_chan': True, # Repeat branch PCA on each single channel separately
        'do_overlap_recheck': True, # Explicitly check if each spike is better accounted for as a sum of 2 spikes (templates)
        'filter_band': (300, 6000), # Sorting DOES NOT FILTER THE DATA! This is information for the sorter to use. Filter voltage as desired BEFORE sorting
        'do_ZCA_transform': True, # Whether to perform ZCA whitening on voltage before sorting
        'check_components': 20, # Number of PCs to check for clustering. None means all
        'max_components': 5, # Max number of PCs to use to form the clustering space, out of those checked
        'min_firing_rate': 0.1, # Neurons with fewer threshold crossings than satisfy this rate are removed
        'use_rand_init': True, # If true, initial clustering uses at least some randomly chosen centers
        'add_peak_valley': False, # Use peak valley in addition to PCs for clustering space
        'max_gpu_memory': None, # Maximum bytes to tryto store on GPU during sorting. None means use as much memory as possible
        'save_1_cpu': True, # If true, leaves one CPU not in use during parallel clustering
        'sort_peak_clips_only': True, # If True, each sort only uses clips with peak on the main channel. Improves speed and accuracy but can miss clusters for low firing rate units on multiple channels
        'n_cov_samples': 20000, # Number of random clips to use to estimate noise covariance matrix. Empirically and qualitatively, 100,000 tends to produce nearly identical results across attempts, 10,000 has some small variance.
        # sigma_bp_noise = 95%: 1.645, 97.5%: 1.96, 99%: 2.326; NOTE: these are one sided
        'sigma_bp_noise': 2.326, # Number of noise standard deviations an expected template match must exceed the decision boundary by. Otherwise it is a candidate for deletion or increased threshold.
        'sigma_bp_CI': 12.0, # Number of noise standard deviations a template match must exceed for a spike to be added. np.inf or None ignores this parameter.
        'absolute_refractory_period': 10e-4, # Absolute refractory period expected between spikes of a single neuron. This is used in postprocesing.
        'get_adjusted_clips': False, # Returns spike clips after the waveforms of any potentially overlapping spikes have been removed
        'max_binary_pursuit_clip_width_factor': 1.0, # The factor by which binary pursuit template matching can be increased relative to clip width for clustering. The best values for clustering and template matching are not always the same.
                                                     # Factor of 1.0 means use the same clip width. Less than 1 is invalid and will use the clip width.
        'verbose': False, # Set to true for more things to be printed while the sorter runs
        'test_flag': False, # Indicates a test run of parallel code that does NOT spawn multiple processes
        'log_dir': None, # Directory where output logs will be saved as text files for each parallel process during clustering. Processes can not usually print to the main screen.
        }

    for k in kwargs.keys():
        if k not in settings:
            raise TypeError("Unknown parameter key {0}.".format(k))
        settings[k] = kwargs[k]

    # Check validity of settings
    if settings['clip_width'][0] > 0.0:
        print("First element of clip width: ", settings['clip_width'][0], " is positive. Using negative value of: ", -1*settings['clip_width'][0])
        settings['clip_width'][0] *= -1
    if settings['filter_band'][0] < 0 or settings['filter_band'][1] < 0:
        raise ValueError("Input setting 'filter_band' must be a positve numbers")

    # Check validity of most other settings
    for key in settings.keys():
        if key in ['do_branch_PCA', 'do_branch_PCA_by_chan', 'do_ZCA_transform',
                    'use_rand_init', 'add_peak_valley', 'save_1_cpu',
                    'sort_peak_clips_only', 'get_adjusted_clips']:
            if type(settings[key]) != bool:
                if settings[key] != 'False' and settings[key] != 0:
                    settings[key] = True
                else:
                    settings[key] = False
                print("Input setting '{0}' was converted to boolean value: ".format(key), settings[key])
        if key in ['segment_duration', 'segment_overlap']:
            # Note actual relative values for overlap are checked in main function
            if settings[key] <= 0:
                raise ValueError("Input setting '{0}' must be a postive number".format(key))
        if key in ['check_components', 'max_components']:
            if settings[key] <= 0  or type(settings[key]) != int:
                raise ValueError("Input setting '{0}' must be a postive integer".format(key))
        if key in ['min_firing_rate', 'sigma_bp_noise',
                    'max_binary_pursuit_clip_width_factor']:
            if settings[key] < 0:
                print("Input setting '{0}' was invalid and converted to zero".format(key))
        if key in ['sigma_bp_CI']:
            if settings[key] is None:
                settings[key] = np.inf

    return settings


def single_thresholds_and_samples(voltage, sigma):
    if voltage.ndim == 1:
        voltage = np.expand_dims(voltage, 0)
    num_channels = voltage.shape[0]
    thresholds = np.empty((num_channels, ))
    samples_over_thresh = []
    for chan in range(0, num_channels):
        abs_voltage = np.abs(voltage[chan, :])
        thresholds[chan] = sigma * np.nanmedian(abs_voltage) / 0.6745
        samples_over_thresh.append(np.count_nonzero(abs_voltage > thresholds[chan]))

    return thresholds, samples_over_thresh


def init_zca_voltage(seg_voltages_to_share, shapes_to_share, v_dtype):
    # Make voltages global so they don't need to be pickled to share with processes
    global zca_pool_dict
    zca_pool_dict = {}
    zca_pool_dict['share_voltage'] = []
    zca_pool_dict['voltage_dtype'] = v_dtype

    for seg in range(0, len(seg_voltages_to_share)):
        zca_pool_dict['share_voltage'].append([seg_voltages_to_share[seg], shapes_to_share[seg]])

    return


def parallel_zca_and_threshold(seg_num, sigma, zca_cushion, n_samples):
    """ Multiprocessing wrapper for single_thresholds_and_samples and
    preprocessing.get_noise_sampled_zca_matrix to get the ZCA voltage for each
    segment in parallel. """
    mkl.set_num_threads(1) # Only 1 thread per process
    # Get shared raw array voltage as numpy view
    seg_voltage = np.frombuffer(zca_pool_dict['share_voltage'][seg_num][0],
                    dtype=zca_pool_dict['voltage_dtype']).reshape(zca_pool_dict['share_voltage'][seg_num][1])

    # Plug numpy view into required functions
    thresholds, _ = single_thresholds_and_samples(seg_voltage, sigma)
    zca_matrix = preprocessing.get_noise_sampled_zca_matrix(seg_voltage,
                    thresholds, sigma, zca_cushion, n_samples)

    # @ makes new copy
    zca_seg_voltage = (zca_matrix @ seg_voltage).astype(zca_pool_dict['voltage_dtype'])
    # Get thresholds for newly ZCA'ed voltage
    thresholds, seg_over_thresh = single_thresholds_and_samples(zca_seg_voltage, sigma)

    # Copy ZCA'ed segment voltage to the raw array buffer so we can re-use it for sorting
    # Doesn't need to be returned since its written to shared dictionary buffer
    np.copyto(seg_voltage, zca_seg_voltage)

    return thresholds, seg_over_thresh


def threshold_and_zca_voltage_parallel(seg_voltages, sigma, zca_cushion, n_samples=1e6):
    """ Use parallel processing to get thresholds and ZCA voltage for each segment
    """
    n_threads = mkl.get_max_threads() # Incoming number of threads
    n_processes = psutil.cpu_count(logical=True) # Use maximum processors

    # Copy segment voltages to shared multiprocessing array
    seg_voltages_to_share = []
    shapes_to_share = []
    for seg in range(0, len(seg_voltages)):
        share_voltage = mp.RawArray(np.ctypeslib.as_ctypes_type(seg_voltages[seg].dtype), seg_voltages[seg].size)
        share_voltage_np = np.frombuffer(share_voltage, dtype=seg_voltages[seg].dtype).reshape(seg_voltages[seg].shape)
        np.copyto(share_voltage_np, seg_voltages[seg])
        seg_voltages_to_share.append(share_voltage)
        shapes_to_share.append(seg_voltages[seg].shape)

    # Run in main process so available in main
    init_zca_voltage(seg_voltages_to_share, shapes_to_share, seg_voltages[0].dtype)

    order_results = []
    with mp.Pool(processes=n_processes, initializer=init_zca_voltage, initargs=(seg_voltages_to_share, shapes_to_share, seg_voltages[0].dtype)) as pool:
        try:
            for seg in range(0, len(seg_voltages)):
                order_results.append(pool.apply_async(parallel_zca_and_threshold, args=(seg, sigma, zca_cushion, n_samples)))
        finally:
            pool.close()
            pool.join()
    # Everything should be in segment order since pool async results were put in
    # list according to seg order. Read out results and return
    results_tuple = [x.get() for x in order_results]
    thresholds_list = [x[0] for x in results_tuple]
    seg_over_thresh_list = [x[1] for x in results_tuple]

    mkl.set_num_threads(n_threads) # Reset threads back

    return thresholds_list, seg_over_thresh_list


def allocate_cpus_by_chan(samples_over_thresh):
    """ Assign CPUs/threads according to number of threshold crossings,
    THIS IS EXTREMELY APPROXIMATE since it counts time points
    above threshold, and each spike will have more than one of these.
    Sort time also depends on number of units and other factors but this
    is a decent starting point without any other information available. """
    cpu_alloc = []
    median_crossings = np.median(samples_over_thresh)
    for magnitude in samples_over_thresh:
        if magnitude > 5*median_crossings:
            cpu_alloc.append(2)
        else:
            cpu_alloc.append(1)

    return cpu_alloc


class NoSpikesError(Exception):
    # Dummy exception class soley for exiting the 'try' loop if there are no spikes
    pass


def print_process_info(title):
    print(title, flush=True)
    print('module name:', __name__, flush=True)
    print('parent process:', os.getppid(), flush=True)
    print('process id:', os.getpid(), flush=True)


"""
    Wavelet alignment can bounce back and forth based on noise blips if
    the spike waveform is nearly symmetric in peak/valley. """
def check_spike_alignment(clips, event_indices, neuron_labels, curr_chan_inds,
                         settings):
    templates, labels = segment_parallel.calculate_templates(clips[:, curr_chan_inds], neuron_labels)
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


def spike_sort_item_parallel(data_dict, use_cpus, work_item, settings):
    """
    do_ZCA_transform, filter_band is not used here but prevents errors from passing kwargs.
    """
    # Initialize variables in case this exits on error
    crossings, neuron_labels = [], []
    exit_type = None
    def wrap_up():
        data_dict['results_dict'][work_item['ID']] = [crossings, neuron_labels]
        data_dict['completed_items'].append(work_item['ID'])
        data_dict['exits_dict'][work_item['ID']] = exit_type
        data_dict['completed_items_queue'].put(work_item['ID'])
        for cpu in use_cpus:
            data_dict['cpu_queue'].put(cpu)
        return
    try:
        # Print this process' errors and output to a file
        if not settings['test_flag'] and settings['log_dir'] is not None:
            # Move stdout to the log_dir file
            if sys.platform == 'win32':
                sys.stdout = open(settings['log_dir'] + "\\SpikeSortItem" + str(work_item['ID']) + ".out", "w")
                sys.stderr = open(settings['log_dir'] + "\\SpikeSortItem" + str(work_item['ID']) + "_errors.out", "w")
            else:
                sys.stdout = open(settings['log_dir'] + "/SpikeSortItem" + str(work_item['ID']) + ".out", "w")
                sys.stderr = open(settings['log_dir'] + "/SpikeSortItem" + str(work_item['ID']) + "_errors.out", "w")
            print_process_info("spike_sort_item_parallel item {0}, channel {1}, segment {2}.".format(work_item['ID'], work_item['channel'], work_item['seg_number']))

        # Setup threads and affinity based on use_cpus if not on mac OS
        if 'win32' == sys.platform:
            proc = psutil.Process()  # get self pid
            # proc.cpu_affinity(use_cpus)
        if settings['test_flag']:
            mkl.set_num_threads(8)
        else:
            mkl.set_num_threads(len(use_cpus))

        # Get the all the needed info for this work item
        # Functions that get this dictionary only ever use these items since
        # we separately extract the voltage and the neighbors
        item_dict = {'sampling_rate': data_dict['sampling_rate'],
                     'n_samples': work_item['n_samples'],
                     'thresholds': work_item['thresholds'],
                     'v_dtype': data_dict['v_dtype']}
        chan = work_item['channel']
        seg_volts_buffer = data_dict['segment_voltages'][work_item['seg_number']][0]
        seg_volts_shape = data_dict['segment_voltages'][work_item['seg_number']][1]
        voltage = np.frombuffer(seg_volts_buffer, dtype=item_dict['v_dtype']).reshape(seg_volts_shape)
        neighbors = work_item['neighbors']

        skip = np.amax(np.abs(settings['clip_width'])) / 2
        align_window = [skip, skip]
        if settings['verbose']: print("Identifying threshold crossings", flush=True)
        # Note that n_crossings is NOT just len(crossings)! It is the raw number
        # of threshold crossings. Values of crossings obey skip and align window.
        crossings, n_crossings = segment_parallel.identify_threshold_crossings(voltage[chan, :], item_dict, item_dict['thresholds'][chan], skip=skip, align_window=align_window)
        if crossings.size == 0:
            exit_type = "No crossings over threshold."
            # Raise error to force exit and wrap_up()
            crossings, neuron_labels = [], []
            raise NoSpikesError
        # Save this number for later
        settings['n_threshold_crossings'][chan] = n_crossings
        min_cluster_size = (np.floor(settings['min_firing_rate'] * item_dict['n_samples'] / item_dict['sampling_rate'])).astype(np.int64)
        if min_cluster_size < 1:
            min_cluster_size = 1
        if settings['verbose']: print("Using minimum cluster size of", min_cluster_size, flush=True)
        _, _, clip_samples, _, curr_chan_inds = segment_parallel.get_windows_and_indices(settings['clip_width'], item_dict['sampling_rate'], chan, neighbors)

        exit_type = "Found crossings"

        # Realign spikes based on a common wavelet
        crossings = segment_parallel.wavelet_align_events(
                            item_dict, voltage[chan, :], crossings,
                            settings['clip_width'],
                            settings['filter_band'])

        clips, valid_event_indices = segment_parallel.get_multichannel_clips(item_dict, voltage[neighbors, :], crossings, clip_width=settings['clip_width'])
        crossings = segment_parallel.keep_valid_inds([crossings], valid_event_indices)

        if settings['sort_peak_clips_only']:
            keep_clips = preprocessing.keep_max_on_main(clips, curr_chan_inds)
            clips = clips[keep_clips, :]
            crossings = crossings[keep_clips]
            curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
            if settings['verbose']: print("After keep max on main removed", np.count_nonzero(~keep_clips), "clips", flush=True)
        if crossings.size == 0:
            exit_type = "No crossings over threshold."
            # Raise error to force exit and wrap_up()
            crossings, neuron_labels = [], []
            raise NoSpikesError

        exit_type = "Found first clips"

        if settings['verbose']: print("Start initial clustering and merge", flush=True)
        # Do initial single channel sort. Start with single channel only because
        # later branching can split things out using multichannel info, but it
        # can't put things back together again
        median_cluster_size = min(100, int(np.around(crossings.size / 1000)))
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

            if settings['sort_peak_clips_only']:
                crossings, neuron_labels, _ = segment_parallel.align_events_with_template(
                                item_dict, voltage[chan, :], neuron_labels, crossings,
                                clip_width=settings['clip_width'])

                crossings, neuron_labels, _ = segment_parallel.align_templates(
                                item_dict, voltage[chan, :], neuron_labels, crossings,
                                clip_width=settings['clip_width'])

                clips, valid_event_indices = segment_parallel.get_multichannel_clips(
                                                item_dict, voltage[neighbors, :],
                                                crossings, clip_width=settings['clip_width'])
                crossings, neuron_labels = segment_parallel.keep_valid_inds(
                        [crossings, neuron_labels], valid_event_indices)

                scores = preprocessing.compute_pca(clips[:, curr_chan_inds],
                            settings['check_components'], settings['max_components'], add_peak_valley=settings['add_peak_valley'],
                            curr_chan_inds=np.arange(0, curr_chan_inds.size))
                n_random = max(100, np.around(crossings.size / 100)) if settings['use_rand_init'] else 0
                neuron_labels = sort.initial_cluster_farthest(scores, median_cluster_size, n_random=n_random)
                neuron_labels = sort.merge_clusters(scores, neuron_labels,
                                    split_only = False,
                                    p_value_cut_thresh=settings['p_value_cut_thresh'])

                curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
                if settings['verbose']: print("After re-sort", curr_num_clusters.size, "different clusters", flush=True)

            crossings, any_merged = check_spike_alignment(clips,
                            crossings, neuron_labels, curr_chan_inds, settings)
            if any_merged:
                # Resort based on new clip alignment
                if settings['verbose']: print("Re-sorting after check spike alignment")
                clips, valid_event_indices = segment_parallel.get_multichannel_clips(
                                                item_dict, voltage[neighbors, :],
                                                crossings, clip_width=settings['clip_width'])
                crossings = segment_parallel.keep_valid_inds([crossings], valid_event_indices)
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
        if settings['verbose']: print("Currently", curr_num_clusters.size, "different clusters", flush=True)

        crossings, neuron_labels, _ = segment_parallel.align_events_with_template(
                        item_dict, voltage[chan, :], neuron_labels, crossings,
                        clip_width=settings['clip_width'])
        clips, valid_event_indices = segment_parallel.get_multichannel_clips(
                                        item_dict, voltage[neighbors, :],
                                        crossings, clip_width=settings['clip_width'])
        crossings, neuron_labels = segment_parallel.keep_valid_inds(
                [crossings, neuron_labels], valid_event_indices)

        # Remove deviant clips before doing branch PCA to avoid getting clusters
        # of overlaps or garbage
        keep_clips = preprocessing.cleanup_clusters(clips[:, curr_chan_inds], neuron_labels)
        crossings, neuron_labels = segment_parallel.keep_valid_inds(
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
            if settings['verbose']: print("After SINGLE BRANCH", curr_num_clusters.size, "different clusters", flush=True)

        if settings['do_branch_PCA']:
            # Remove deviant clips before doing branch PCA to avoid getting clusters
            # of overlaps or garbage, this time on full neighborhood
            keep_clips = preprocessing.cleanup_clusters(clips, neuron_labels)
            crossings, neuron_labels = segment_parallel.keep_valid_inds(
                    [crossings, neuron_labels], keep_clips)
            clips = clips[keep_clips, :]

        # Multi channel branch
        if data_dict['num_channels'] > 1 and settings['do_branch_PCA']:
            neuron_labels = branch_pca_2_0(neuron_labels, clips, curr_chan_inds,
                                p_value_cut_thresh=settings['p_value_cut_thresh'],
                                add_peak_valley=settings['add_peak_valley'],
                                check_components=settings['check_components'],
                                max_components=settings['max_components'],
                                use_rand_init=settings['use_rand_init'],
                                method='pca')
            curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
            if settings['verbose']: print("After MULTI BRANCH", curr_num_clusters.size, "different clusters", flush=True)

        # Multi channel branch by channel
        if data_dict['num_channels'] > 1 and settings['do_branch_PCA_by_chan'] and settings['do_branch_PCA']:
            neuron_labels = branch_pca_2_0(neuron_labels, clips, curr_chan_inds,
                                p_value_cut_thresh=settings['p_value_cut_thresh'],
                                add_peak_valley=settings['add_peak_valley'],
                                check_components=settings['check_components'],
                                max_components=settings['max_components'],
                                use_rand_init=settings['use_rand_init'],
                                method='chan_pca')
            curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
            if settings['verbose']: print("After MULTI BRANCH by channel", curr_num_clusters.size, "different clusters", flush=True)

        # Delete any clusters under min_cluster_size before binary pursuit
        if settings['verbose']: print("Current smallest cluster has", np.amin(n_per_cluster), "spikes", flush=True)
        if np.any(n_per_cluster < min_cluster_size):
            for l_ind in range(0, curr_num_clusters.size):
                if n_per_cluster[l_ind] < min_cluster_size:
                    keep_inds = ~(neuron_labels == curr_num_clusters[l_ind])
                    crossings = crossings[keep_inds]
                    neuron_labels = neuron_labels[keep_inds]
                    clips = clips[keep_inds, :]
                    print("Deleted cluster", curr_num_clusters[l_ind], "with", n_per_cluster[l_ind], "spikes", flush=True)

        if neuron_labels.size == 0:
            exit_type = "No clusters over min_firing_rate."
            # Raise error to force exit and wrap_up()
            crossings, neuron_labels = [], []
            raise NoSpikesError

        exit_type = "Finished sorting clusters"

        # Realign spikes based on correlation with current cluster templates before doing binary pursuit
        crossings, neuron_labels, valid_inds = segment_parallel.align_events_with_template(item_dict, voltage[chan, :], neuron_labels, crossings, clip_width=settings['clip_width'])
        clips = clips[valid_inds, :]

        if settings['verbose']: print("currently", np.unique(neuron_labels).size, "different clusters", flush=True)
        # Map labels starting at zero and put labels in order
        sort.reorder_labels(neuron_labels)
        if settings['verbose']: print("Successfully completed item ", str(work_item['ID']), flush=True)
        exit_type = "Success"
    except NoSpikesError:
        if settings['verbose']: print("No spikes to sort.")
        if settings['verbose']: print("Successfully completed item ", str(work_item['ID']), flush=True)
        exit_type = "Success"
    except Exception as err:
        exit_type = err
        print_tb(err.__traceback__)
        if settings['test_flag']:
            raise # Reraise any exceptions in test mode only
    finally:
        wrap_up()


def init_data_dict(init_dict=None):
    global data_dict
    data_dict = {}
    data_dict['segment_voltages'] = init_dict['segment_voltages']
    if init_dict is not None:
        for k in init_dict.keys():
            data_dict[k] = init_dict[k]
    return


def spike_sort_parallel(Probe, **kwargs):
    """ Perform spike sorting algorithm using python multiprocessing module.
    See 'spike_sorting_settings_parallel' above for a list of allowable kwargs.

    Note: The temporary directory to store spike clips is created manually, not
    using the python tempfile module. Multiprocessing and tempfile seem to have
    some problems across platforms. For certain errors or keyboard interrupts
    the file may not be appropriately deleted. Before using the directory, the
    temp directory is deleted if it exists, so subsequent successful runs of
    sorting using the same directory will remove the temp directory.

    Note: Clips and voltages will be output in the data type Probe.v_dtype.
    However, most of the arithmetic is computed in np.float64. Clips are cast
    as np.float64 for determining PCs and cast back when done. All of binary
    pursuit is conducted as np.float32 for memory and GPU compatibility.
    See also:
    '
    """
    n_threads = mkl.get_max_threads() # Incoming number of threads
    # Get our settings
    settings = spike_sorting_settings_parallel(**kwargs)
    # Check that fitler is appropriate
    if settings['filter_band'][0] > Probe.sampling_rate/2 or settings['filter_band'][1] > Probe.sampling_rate/2:
        raise ValueError("Input setting 'filter_band' exceeds Nyquist limit for sampling rate of", Probe.sampling_rate)
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
    elif np.any(np.diff(check_neighbors) <= 0):
        raise ValueError("Probe get_neighbors() method must return neighbor channels IN ORDER without duplicates.")
    # For convenience, necessary to define clip width as negative for first entry
    if settings['clip_width'][0] > 0:
        settings['clip_width'] *= -1
    manager = mp.Manager()
    init_dict = {'num_channels': Probe.num_channels, 'sampling_rate': Probe.sampling_rate,
                 'results_dict': manager.dict(), 'v_dtype': Probe.v_dtype,
                 'completed_items': manager.list(), 'exits_dict': manager.dict(),
                 'gpu_lock': manager.Lock(), 'filter_band': settings['filter_band']}
    if settings['log_dir'] is not None:
        if os.path.exists(settings['log_dir']):
            rmtree(settings['log_dir'])
            time.sleep(.5) # NEED SLEEP SO CAN DELETE BEFORE RECREATING!!!
        os.makedirs(settings['log_dir'])
    # if settings['tmp_clips_dir'] is None:
    #     settings['tmp_clips_dir'] = os.getcwd() + '/tmp_clips'
    # # Clear out room for temp clips in current directory
    # if os.path.exists(settings['tmp_clips_dir']):
    #     rmtree(settings['tmp_clips_dir'])
    # os.mkdir(settings['tmp_clips_dir'])

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
            clip_samples = segment_parallel.time_window_to_samples(settings['clip_width'], Probe.sampling_rate)[0]
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
    settings['n_threshold_crossings'] = np.zeros(Probe.num_channels)

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

    # Build the sorting work items
    seg_voltages = []
    for x in range(0, len(segment_onsets)):
        # Slice over num_channels should keep same shape
        # Build list in segment order
        seg_voltages.append(Probe.voltage[0:Probe.num_channels,
                                          segment_onsets[x]:segment_offsets[x]])
    samples_over_thresh = []
    if not settings['test_flag'] and settings['do_ZCA_transform']:
        # Use parallel processing to get zca voltage and thresholds
        if settings['verbose']: print("Doing parallel ZCA transform and thresholding for", len(segment_onsets), "segments")
        thresholds_list, seg_over_thresh_list = threshold_and_zca_voltage_parallel(
                    seg_voltages, settings['sigma'], zca_cushion, n_samples=1e6)
        for x in seg_over_thresh_list:
            samples_over_thresh.extend(x)
        init_dict['segment_voltages'] = zca_pool_dict['share_voltage']
    else:
        init_dict['segment_voltages'] = []
        thresholds_list = []
        seg_over_thresh_list = []
        for x in range(0, len(segment_onsets)):
            # Need to copy or else ZCA transforms will duplicate in overlapping
            # time segments. Copy happens during matrix multiplication
            seg_voltage = seg_voltages[x]
            if settings['do_ZCA_transform']:
                if settings['verbose']: print("Finding voltage and thresholds for segment", x+1, "of", len(segment_onsets))
                if settings['verbose']: print("Doing ZCA transform")
                thresholds, _ = single_thresholds_and_samples(seg_voltage, settings['sigma'])
                zca_matrix = preprocessing.get_noise_sampled_zca_matrix(seg_voltage,
                                thresholds, settings['sigma'],
                                zca_cushion, n_samples=1e6)
                # @ makes new copy
                seg_voltage = (zca_matrix @ seg_voltage).astype(Probe.v_dtype)
            thresholds, seg_over_thresh = single_thresholds_and_samples(seg_voltage, settings['sigma'])
            thresholds_list.append(thresholds)
            samples_over_thresh.extend(seg_over_thresh)
            # Allocate shared voltage buffer. List is appended in SEGMENT ORDER
            init_dict['segment_voltages'].append([mp.RawArray(np.ctypeslib.as_ctypes_type(Probe.v_dtype), seg_voltage.size), seg_voltage.shape])
            np_view = np.frombuffer(init_dict['segment_voltages'][x][0], dtype=Probe.v_dtype).reshape(seg_voltage.shape) # Create numpy view
            np.copyto(np_view, seg_voltage) # Copy segment voltage to voltage buffer

    work_items = []
    chan_neighbors = []
    chan_neighbor_inds = []
    for x in range(0, len(segment_onsets)):
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
                               'thresholds': thresholds_list[x],
                               })




    if not settings['test_flag']:
        if settings['log_dir'] is None:
            print("No log dir specified. Won't be able to see output from processes")
        # Sort  work_items and samples_over_thresh by descending order of
        # samples over threshold. If testing we do not do this to keep
        # random numbers consistent with single channel sorter
        # Zip only returns tuple, so map it to a list
        samples_over_thresh, work_items = map(list, zip(*[[x, y] for x, y in reversed(
                                            sorted(zip(samples_over_thresh,
                                            work_items), key=lambda pair: pair[0]))]))
    n_cpus = psutil.cpu_count(logical=True)
    if settings['save_1_cpu']:
        n_cpus -= 1
    cpu_queue = manager.Queue(n_cpus)
    for cpu in range(n_cpus):
        cpu_queue.put(cpu)
    # cpu_alloc returned in order of samples_over_thresh/work_items
    cpu_alloc = allocate_cpus_by_chan(samples_over_thresh)
    # Make sure none exceed number available
    for x in range(0, len(cpu_alloc)):
        if cpu_alloc[x] > n_cpus:
            cpu_alloc[x] = n_cpus
    init_dict['cpu_queue'] = cpu_queue
    completed_items_queue = manager.Queue(len(work_items))
    init_dict['completed_items_queue'] = completed_items_queue
    for x in range(0, len(work_items)):
        # Initializing keys for each result seems to prevent broken pipe errors
        init_dict['results_dict'][x] = None
        init_dict['exits_dict'][x] = None

    # Call init function to ensure data_dict is globally available before passing
    # it into each process
    init_data_dict(init_dict)
    processes = []
    proc_item_index = []
    completed_items_index = 0
    process_errors_list = []
    print("Starting sorting pool")
    # Put the work items through the sorter
    for wi_ind, w_item in enumerate(work_items):
        w_item['ID'] = wi_ind # Assign ID number in order of deployment
        # With timeout=None, this will block until sufficient cpus are available
        # as requested by cpu_alloc
        # NOTE: this currently goes in order, so if 1 CPU is available but next
        # work item wants 2, it will wait until 2 are available rather than
        # starting a different item that only wants 1 CPU...
        use_cpus = [cpu_queue.get(timeout=None) for x in range(cpu_alloc[wi_ind])]
        n_complete = len(data_dict['completed_items']) # Do once to avoid race
        if n_complete > completed_items_index:
            for ci in range(completed_items_index, n_complete):
                print("Completed item", work_items[data_dict['completed_items'][ci]]['ID'], "from chan", work_items[data_dict['completed_items'][ci]]['channel'], "segment", work_items[data_dict['completed_items'][ci]]['seg_number'])
                print("Exited with status: ", data_dict['exits_dict'][data_dict['completed_items'][ci]])
                completed_items_index += 1
                if not settings['test_flag']:
                    done_index = proc_item_index.index(data_dict['completed_items'][ci])
                    del proc_item_index[done_index]
                    processes[done_index].join()
                    processes[done_index].close()
                    del processes[done_index]
                    # Store error type if any
                    if data_dict['exits_dict'][data_dict['completed_items'][ci]] != "Success":
                        process_errors_list.append([work_items[data_dict['completed_items'][ci]]['ID'], data_dict['exits_dict'][data_dict['completed_items'][ci]]])

        if not settings['test_flag']:
            print("Starting item {0}/{1} on CPUs {2} for channel {3} segment {4}".format(wi_ind+1, len(work_items), use_cpus, w_item['channel'], w_item['seg_number']))
            time.sleep(.5) # NEED SLEEP SO PROCESSES AREN'T MADE TOO FAST AND FAIL!!!
            proc = mp.Process(target=spike_sort_item_parallel,
                              args=(data_dict, use_cpus, w_item, settings))
            proc.start()
            processes.append(proc)
            proc_item_index.append(wi_ind)
        else:
            print("Starting item {0}/{1} on CPUs {2} for channel {3} segment {4}".format(wi_ind+1, len(work_items), use_cpus, w_item['channel'], w_item['seg_number']))
            spike_sort_item_parallel(data_dict, use_cpus, w_item, settings)
            print("finished sort one item")

    if not settings['test_flag']:
        # Wait here a bit to print out items as they complete and to ensure
        # no process are left behind, as can apparently happen if you attempt to
        # join() too soon without being sure everything is finished (especially using queues)
        while completed_items_index < len(work_items) and not settings['test_flag']:
            finished_item = completed_items_queue.get()
            try:
                done_index = proc_item_index.index(finished_item)
            except ValueError:
                # This item was already finished above so just clearing out
                # completed_items_queue
                continue
            print("Completed item", finished_item+1, "from chan", work_items[finished_item]['channel'], "segment", work_items[finished_item]['seg_number'])
            print("Exited with status: ", data_dict['exits_dict'][finished_item])
            completed_items_index += 1

            del proc_item_index[done_index]
            processes[done_index].join()
            processes[done_index].close()
            del processes[done_index]
            # Store error type if any
            if data_dict['exits_dict'][finished_item] != "Success":
                process_errors_list.append([finished_item, data_dict['exits_dict'][finished_item]])

    # Make sure all the processes finish up and close even though they should
    # have finished above
    while len(processes) > 0:
        p = processes.pop()
        p.join()
        p.close()
        del p

    # Set threads/processes back to normal now that we are done
    mkl.set_num_threads(n_threads)

    sort_data = []
    sort_info = settings
    curr_chan_win, _ = segment_parallel.time_window_to_samples(
                                    settings['clip_width'], Probe.sampling_rate)
    sort_info.update({'n_samples': Probe.n_samples,
                      'n_channels': Probe.num_channels,
                      'n_samples_per_chan': curr_chan_win[1] - curr_chan_win[0],
                      'sampling_rate': Probe.sampling_rate,
                      'n_segments': len(segment_onsets)})
    sort_info['separability_metrics'] = [[] for x in range(0, sort_info['n_segments'])]

    for seg_number in range(0, len(segment_onsets)):
        seg_data = full_binary_pursuit.full_binary_pursuit(work_items,
                    data_dict, seg_number, sort_info, Probe.v_dtype,
                    overlap_ratio_threshold=2,
                    absolute_refractory_period=settings['absolute_refractory_period'],
                    kernels_path=None,
                    max_gpu_memory=settings['max_gpu_memory'])
        sort_data.extend(seg_data)

    # Re-print any errors so more visible at the end of sorting
    for pe in process_errors_list:
        print("Item number", pe[0], "had the following error:")
        print("            ", pe[1])

    if settings['verbose']: print("Done.")
    return sort_data, work_items, sort_info
