import pickle
import os
import sys
sys.path.append(os.getcwd())
from shutil import rmtree
import mkl
import numpy as np
import multiprocessing as mp
import psutil
from spikesorting_python.src.parallel import segment_parallel
from spikesorting_python.src import sort
from spikesorting_python.src.parallel import overlap_parallel
from spikesorting_python.src.parallel import binary_pursuit_parallel
from spikesorting_python.src import consolidate
from spikesorting_python.src import electrode
from spikesorting_python.src import preprocessing
from spikesorting_python.src import consolidate
from scipy import signal
import time
from traceback import print_tb
import warnings
from copy import copy



def spike_sorting_settings_parallel(**kwargs):
    settings = {}

    settings['sigma'] = 4.0 # Threshold based on  noise level
    settings['verbose'] = False
    settings['test_flag'] = False # Indicates a test run of parallel code that does NOT spawn multiple processes
    settings['log_dir'] = None # Directory where output logs will be saved as text files
    # settings['verbose_merge'] = False
    # settings['threshold_type'] = "absolute"
    # settings['sharpen'] = True
    settings['clip_width'] = [-6e-4, 10e-4]# Width of clip in seconds
    # settings['compute_noise'] = False # Estimate noise per cluster
    # settings['remove_false_positives'] = True # Remove false positives? Requires compute_noise = true
    settings['do_branch_PCA'] = True # Use branch pca method to split clusters
    # settings['preprocess'] = True
    settings['filter_band'] = (300, 6000)
    # settings['compute_lfp'] = False
    # settings['lfp_filter'] = (25, 500)
    settings['do_ZCA_transform'] = True
    settings['use_rand_init'] = True # Initial clustering uses at least some randomly chosen centers
    settings['add_peak_valley'] = False # Use peak valley in addition to PCs for sorting
    settings['check_components'] = None # Number of PCs to check. None means all
    settings['max_components'] = 10 # Max number to use, of those checked
    settings['min_firing_rate'] = 1. # Neurons with fewer threshold crossings than this are removed
    settings['p_value_cut_thresh'] = 0.05
    settings['do_binary_pursuit'] = True
    settings['use_GPU'] = True # Force algorithms to run on the CPU rather than the GPU
    settings['max_gpu_memory'] = None # Use as much memory as possible
    settings['save_1_cpu'] = True
    settings['segment_duration'] = None # Seconds (nothing/Inf uses the entire recording)
    settings['segment_overlap'] = None # Seconds of overlap between adjacent segments
    settings['cleanup_neurons'] = False # Remove garbage at the end
    # settings['random_seed'] = None # The random seed to use (or nothing, if unspecified)

    for k in kwargs.keys():
        if k not in settings:
            raise TypeError("Unknown parameter key {0}.".format(k))
        settings[k] = kwargs[k]

    return settings


def init_pool_dict(volt_array, volt_shape, init_dict=None):
    global pool_dict
    pool_dict = {}
    pool_dict['share_voltage'] = volt_array
    pool_dict['share_voltage_shape'] = volt_shape
    if init_dict is not None:
        for k in init_dict.keys():
            pool_dict[k] = init_dict[k]
    return


def init_load_voltage():
    mkl.set_num_threads(1)


def load_voltage_parallel(PL2Reader, read_source):
    """
    """

    include_chans = []
    for cc in range(0, len(PL2Reader.info['analog_channels'])):
        if PL2Reader.info['analog_channels'][cc]['source_name'].upper() == read_source and bool(PL2Reader.info['analog_channels'][cc]['enabled']):
            include_chans.append(cc)

    order_results = []
    with mp.Pool(processes=psutil.cpu_count(logical=True), initializer=init_load_voltage, initargs=()) as pool:
        try:
            for chan in range(0, len(include_chans)):
                order_results.append(pool.apply_async(PL2Reader.load_analog_channel, args=(include_chans[chan], False, None)))
        finally:
            pool.close()
            pool.join()
    voltage = np.vstack([x.get() for x in order_results])

    return voltage


def filter_one_chan(chan, b_filt, a_filt):

    mkl.set_num_threads(2)
    voltage = np.frombuffer(pool_dict['share_voltage']).reshape(pool_dict['share_voltage_shape'])
    filt_voltage = signal.filtfilt(b_filt, a_filt, voltage[chan, :], padlen=None)
    return filt_voltage


def filter_parallel(Probe, low_cutoff=300, high_cutoff=6000):

    print("Allocating filter array and copying voltage")
    filt_X = mp.RawArray('d', Probe.voltage.size)
    filt_X_np = np.frombuffer(filt_X).reshape(Probe.voltage.shape)
    np.copyto(filt_X_np, Probe.voltage)
    low_cutoff = low_cutoff / (Probe.sampling_rate / 2)
    high_cutoff = high_cutoff / (Probe.sampling_rate / 2)
    b_filt, a_filt = signal.butter(1, [low_cutoff, high_cutoff], btype='band')
    print("Performing voltage filtering")
    filt_results = []
    # Number of processes was giving me out of memory error on Windows 10 until
    # I dropped it to half number of cores.
    with mp.Pool(processes=psutil.cpu_count(logical=False)//2, initializer=init_pool_dict, initargs=(filt_X, Probe.voltage.shape)) as pool:
        try:
            for chan in range(0, Probe.num_electrodes):
                filt_results.append(pool.apply_async(filter_one_chan, args=(chan, b_filt, a_filt)))
        finally:
            pool.close()
            pool.join()
    filt_voltage = np.vstack([x.get() for x in filt_results])

    return filt_voltage


def zca_one_chan(chan):

    mkl.set_num_threads(1)
    voltage = np.frombuffer(pool_dict['share_voltage']).reshape(pool_dict['share_voltage_shape'])
    zca_data = np.matmul(pool_dict['zca_matrix'][chan, :], voltage)
    result_voltage = np.frombuffer(pool_dict['result_voltage']).reshape(pool_dict['share_voltage_shape'])
    np.copyto(result_voltage[chan, :], zca_data)
    return None


def zca_parallel(shared_voltage, result_voltage, Probe, zca_matrix):
    init_dict = {'zca_matrix': zca_matrix, 'result_voltage': result_voltage}
    zca_results = []
    with mp.Pool(processes=psutil.cpu_count(logical=True), initializer=init_pool_dict, initargs=(shared_voltage, Probe.voltage.shape, init_dict)) as pool:
        try:
            for chan in range(0, Probe.num_electrodes):
                zca_results.append(pool.apply_async(zca_one_chan, args=(chan,)))
        finally:
            pool.close()
            pool.join()
    # zca_voltage = np.vstack([x.get() for x in zca_results])
    X_np = np.frombuffer(shared_voltage).reshape(Probe.voltage.shape)
    result_voltage = np.frombuffer(result_voltage).reshape(Probe.voltage.shape)
    print("Copying ZCA results to main voltage")
    np.copyto(X_np, result_voltage)


def thresh_and_size_one_chan(chan, sigma):

    mkl.set_num_threads(1)
    voltage = np.frombuffer(pool_dict['share_voltage']).reshape(pool_dict['share_voltage_shape'])
    abs_voltage = np.abs(voltage[chan, :])
    threshold = np.nanmedian(abs_voltage) / 0.6745
    threshold *= sigma
    n_crossings = np.count_nonzero(abs_voltage > threshold)

    return n_crossings, threshold, chan


def get_thresholds_and_work_order(shared_voltage, Probe, sigma):
    order_results = []
    with mp.Pool(processes=psutil.cpu_count(logical=True), initializer=init_pool_dict, initargs=(shared_voltage, Probe.voltage.shape)) as pool:
        try:
            for chan in range(0, Probe.num_electrodes):
                order_results.append(pool.apply_async(thresh_and_size_one_chan, args=(chan, sigma)))
        finally:
            pool.close()
            pool.join()

    crossings_per_s = np.empty(Probe.num_electrodes, dtype=np.int64)
    thresholds = np.empty(Probe.num_electrodes)
    for or_ind in range(0, len(order_results)):
        data = order_results[or_ind].get()
        crossings_per_s[or_ind] = data[0] / (Probe.n_samples / Probe.sampling_rate)
        thresholds[or_ind] = data[1]

    return thresholds, crossings_per_s


def single_thresholds_and_samples(voltage, sigma):
    if voltage.ndim == 1:
        voltage = np.expand_dims(voltage, 0)
    num_electrodes = voltage.shape[0]
    thresholds = np.empty((num_electrodes, ))
    samples_over_thresh = []
    for chan in range(0, num_electrodes):
        abs_voltage = np.abs(voltage[chan, :])
        thresholds[chan] = np.nanmedian(abs_voltage) / 0.6745
        samples_over_thresh.append(np.count_nonzero(abs_voltage > thresholds[chan]))
    thresholds *= sigma

    return thresholds, samples_over_thresh


def allocate_cpus_by_chan(samples_over_thresh):
    """ Assign CPUs/threads according to number of threshold crossings,
    THIS IS EXTREMELY APPROXIMATE since it counts time points
    above threshold, and each spike will have more than one of these.
    Sort time also depends on number of units and other factors but this
    is a decent starting point without any other information available. """
    cpu_alloc = []
    median_crossings = np.median(samples_over_thresh)
    for magnitude in samples_over_thresh:
        if magnitude > 10*median_crossings:
            cpu_alloc.append(6)
        elif magnitude > 5*median_crossings:
            cpu_alloc.append(4)
        elif magnitude > median_crossings:
            cpu_alloc.append(2)
        else:
            cpu_alloc.append(1)

    return cpu_alloc


class NoSpikesError(Exception):
    pass


def print_process_info(title):
    print(title, flush=True)
    print('module name:', __name__, flush=True)
    print('parent process:', os.getppid(), flush=True)
    print('process id:', os.getpid(), flush=True)


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
    crossings, neuron_labels, clips, new_inds = [], [], [], []
    exit_type = None
    def wrap_up():
        data_dict['results_dict'][work_item['ID']] = [crossings, neuron_labels, new_inds]
        with open('temp_clips' + str(work_item['ID']) + '.pickle', 'wb') as fp:
            pickle.dump(clips, fp, protocol=-1)
        data_dict['completed_items'].append(work_item['ID'])
        data_dict['exits_dict'][work_item['ID']] = exit_type
        data_dict['completed_items_queue'].put(work_item['ID'])
        for cpu in use_cpus:
            data_dict['cpu_queue'].put(cpu)
        return
    try:
        # Print this process' errors and output to a file
        if not settings['test_flag']:
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
            proc.cpu_affinity(use_cpus)
        if settings['test_flag']:
            mkl.set_num_threads(8)
        else:
            mkl.set_num_threads(len(use_cpus))

        # Get the all the needed info for this work item
        # Functions that get this dictionary only ever use these items since
        # we separately extract the voltage and the neighbors
        item_dict = {'sampling_rate': data_dict['sampling_rate'],
                     'n_samples': work_item['n_samples'],
                     'thresholds': work_item['thresholds']}
        chan = work_item['channel']
        seg_volts_buffer = data_dict['segment_voltages'][work_item['seg_number']][0]
        seg_volts_shape = data_dict['segment_voltages'][work_item['seg_number']][1]
        voltage = np.frombuffer(seg_volts_buffer).reshape(seg_volts_shape)
        neighbors = work_item['neighbors']

        skip = np.amax(np.abs(settings['clip_width'])) / 2
        align_window = [skip, skip]
        if settings['verbose']: print("Identifying threshold crossings", flush=True)
        crossings = segment_parallel.identify_threshold_crossings(voltage[chan, :], item_dict, item_dict['thresholds'][chan], skip=skip, align_window=align_window)
        if crossings.size == 0:
            exit_type = "No crossings over threshold."
            # Raise error to force exit and wrap_up()
            crossings, neuron_labels, clips, new_inds = [], [], [], []
            raise NoSpikesError
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

        median_cluster_size = min(100, int(np.around(crossings.size / 1000)))
        clips, valid_event_indices = segment_parallel.get_multichannel_clips(item_dict, voltage[neighbors, :], crossings, clip_width=settings['clip_width'])
        crossings = segment_parallel.keep_valid_inds([crossings], valid_event_indices)

        exit_type = "Found first clips"

        if settings['verbose']: print("Start initial clustering and merge", flush=True)
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

        if settings['do_binary_pursuit']:
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

        if settings['do_branch_PCA'] and settings['do_binary_pursuit']:
            # Remove deviant clips before doing branch PCA to avoid getting clusters
            # of overlaps or garbage, this time on full neighborhood
            keep_clips = preprocessing.cleanup_clusters(clips, neuron_labels)
            crossings, neuron_labels = segment_parallel.keep_valid_inds(
                    [crossings, neuron_labels], keep_clips)
            clips = clips[keep_clips, :]

        # Multi channel branch
        if data_dict['num_electrodes'] > 1 and settings['do_branch_PCA']:
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
        if data_dict['num_electrodes'] > 1 and settings['do_branch_PCA']:
            neuron_labels = branch_pca_2_0(neuron_labels, clips, curr_chan_inds,
                                p_value_cut_thresh=settings['p_value_cut_thresh'],
                                add_peak_valley=settings['add_peak_valley'],
                                check_components=settings['check_components'],
                                max_components=settings['max_components'],
                                use_rand_init=settings['use_rand_init'],
                                method='chan_pca')
            curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
            if settings['verbose']: print("After MULTI BY CHAN BRANCH", curr_num_clusters.size, "different clusters", flush=True)

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
            crossings, neuron_labels, clips, new_inds = [], [], [], []
            raise NoSpikesError

        exit_type = "Finished sorting clusters"

        # Realign spikes based on correlation with current cluster templates before doing binary pursuit
        crossings, neuron_labels, _ = segment_parallel.align_events_with_template(item_dict, voltage[chan, :], neuron_labels, crossings, clip_width=settings['clip_width'])
        if settings['do_binary_pursuit']:
            if settings['verbose']: print("currently", np.unique(neuron_labels).size, "different clusters", flush=True)
            if settings['verbose']: print("Doing binary pursuit", flush=True)
            if not settings['use_GPU']:
                crossings, neuron_labels, new_inds = overlap_parallel.binary_pursuit_secret_spikes(
                                item_dict, chan, neighbors, voltage[neighbors, :],
                                neuron_labels, crossings,
                                settings['clip_width'])
                clips, valid_event_indices = segment_parallel.get_multichannel_clips(item_dict, voltage[neighbors, :], crossings, clip_width=settings['clip_width'])
                crossings, neuron_labels = segment_parallel.keep_valid_inds([crossings, neuron_labels], valid_event_indices)
            else:
                with data_dict['gpu_lock']:
                    crossings, neuron_labels, new_inds, clips = binary_pursuit_parallel.binary_pursuit(
                                item_dict, chan, neighbors, voltage[neighbors, :],
                                crossings, neuron_labels, settings['clip_width'],
                                kernels_path=None, max_gpu_memory=settings['max_gpu_memory'])
                    time.sleep(.5) # Sleep half a second to let openCL objects close before releasing lock
            exit_type = "Finished binary pursuit"
        else:
            clips, valid_event_indices = segment_parallel.get_multichannel_clips(item_dict, voltage[neighbors, :], crossings, clip_width=settings['clip_width'])
            crossings, neuron_labels = segment_parallel.keep_valid_inds([crossings, neuron_labels], valid_event_indices)
            new_inds = np.zeros(crossings.size, dtype='bool')

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
        exit_type = err #exit_type + "--Then FAILED TO COMPLETE!"
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
    """
    See 'spike_sorting_settings_parallel' above for a list of allowable kwargs.

    See also:
    Documentation in spikesorting 'spike_sort()'
    """
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
    # Get our settings
    settings = spike_sorting_settings_parallel(**kwargs)
    manager = mp.Manager()
    if not settings['use_GPU'] and settings['do_binary_pursuit']:
        use_GPU_message = ("Using CPU binary pursuit to find " \
                            "secret spikes. This can be MUCH MUCH " \
                            "slower and uses more " \
                            "memory than the GPU version. Returned " \
                            "clips will NOT be adjusted.")
        warnings.warn(use_GPU_message, RuntimeWarning, stacklevel=2)
    init_dict = {'num_electrodes': Probe.num_electrodes, 'sampling_rate': Probe.sampling_rate,
                 'results_dict': manager.dict(),
                 'completed_items': manager.list(), 'exits_dict': manager.dict(),
                 'gpu_lock': manager.Lock(), 'filter_band': settings['filter_band']}
    if settings['log_dir'] is not None:
        if os.path.exists(settings['log_dir']):
            rmtree(settings['log_dir'])
            time.sleep(.5) # NEED SLEEP SO CAN DELETE BEFORE RECREATING!!!
        os.makedirs(settings['log_dir'])

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
        # thresholds, _ = single_thresholds_and_samples(Probe.voltage, settings['sigma'])
        # zca_matrix = preprocessing.get_noise_sampled_zca_matrix(Probe.voltage,
        #                 thresholds, settings['sigma'],
        #                 zca_cushion, n_samples=1e7)

    # Build the sorting work items
    init_dict['segment_voltages'] = []
    samples_over_thresh = []
    work_items = []
    chan_neighbors = []
    chan_neighbor_inds = []
    for x in range(0, len(segment_onsets)):
        if settings['verbose']: print("Finding voltage and thresholds for segment", x+1, "of", len(segment_onsets))
        # Need to copy or else ZCA transforms will duplicate in overlapping
        # time segments. Copy happens during matrix multiplication
        # Slice over num_electrodes should keep same shape
        seg_voltage = Probe.voltage[0:Probe.num_electrodes,
                                   segment_onsets[x]:segment_offsets[x]]
        if settings['do_ZCA_transform']:
            if settings['verbose']: print("Doing ZCA transform")
            thresholds, _ = single_thresholds_and_samples(seg_voltage, settings['sigma'])
            zca_matrix = preprocessing.get_noise_sampled_zca_matrix(seg_voltage,
                            thresholds, settings['sigma'],
                            zca_cushion, n_samples=1e6)
            seg_voltage = zca_matrix @ seg_voltage # @ makes new copy
        thresholds, seg_over_thresh = single_thresholds_and_samples(seg_voltage, settings['sigma'])
        samples_over_thresh.extend(seg_over_thresh)
        # Allocate shared voltage buffer. List is appended in SEGMENT ORDER
        init_dict['segment_voltages'].append([mp.RawArray('d', seg_voltage.size), seg_voltage.shape])
        np_view = np.frombuffer(init_dict['segment_voltages'][x][0]).reshape(seg_voltage.shape) # Create numpy view
        np.copyto(np_view, seg_voltage) # Copy segment voltage to voltage buffer
        for chan in range(0, Probe.num_electrodes):
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
                print("Completed item", work_items[data_dict['completed_items'][ci]]['ID']+1, "from chan", work_items[data_dict['completed_items'][ci]]['channel'], "segment", work_items[data_dict['completed_items'][ci]]['seg_number'])
                print("Exited with status: ", data_dict['exits_dict'][data_dict['completed_items'][ci]])
                completed_items_index += 1
                if not settings['test_flag']:
                    done_index = proc_item_index.index(data_dict['completed_items'][ci])
                    del proc_item_index[done_index]
                    processes[done_index].join()
                    processes[done_index].close()
                    del processes[done_index]

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

    sort_data = []
    for wi_ind, w_item in enumerate(work_items):
        if w_item['ID'] in data_dict['results_dict'].keys():
            with open('temp_clips' + str(w_item['ID']) + '.pickle', 'rb') as fp:
                waveforms = pickle.load(fp)
            os.remove('temp_clips' + str(w_item['ID']) + '.pickle')
            # Append list of crossings, labels, waveforms, new_waveforms
            sort_data.append([data_dict['results_dict'][w_item['ID']][0],
                              data_dict['results_dict'][w_item['ID']][1],
                              waveforms,
                              data_dict['results_dict'][w_item['ID']][2],
                              w_item['ID']])
            # I am not sure why, but this has to be added here. It does not work
            # when done above directly on the global data_dict elements
            if type(sort_data[-1][0]) == np.ndarray:
                if sort_data[-1][0].size > 0:
                    # Adjust crossings for segment start time
                    sort_data[-1][0] += w_item['index_window'][0]
        else:
            # This work item found nothing (or raised an exception)
            sort_data.append([[], [], [], [], w_item['ID']])
    sort_info = settings
    curr_chan_win, _ = segment_parallel.time_window_to_samples(
                                    settings['clip_width'], Probe.sampling_rate)
    sort_info.update({'n_samples': Probe.n_samples,
                      'n_channels': Probe.num_electrodes,
                      'n_samples_per_chan': curr_chan_win[1] - curr_chan_win[0],
                      'sampling_rate': Probe.sampling_rate,
                      'n_segments': len(segment_onsets)})

    if settings['verbose']: print("Done.")
    return sort_data, work_items, sort_info


if __name__ == '__main__':
    import os
    import sys
    from shutil import rmtree

    # python.exe spikesorting_python\src\parallel\spikesorting_parallel.py C:\Nate\LearnDirTunePurk_Yoda_46.pl2 C:\Nate\sorted_yoda_46.pickle

    proc = psutil.Process()  # get self pid
    proc.cpu_affinity(cpus=list(range(psutil.cpu_count())))
    mkl.set_num_threads(8)
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    fname_PL2 = sys.argv[1]
    if len(sys.argv) < 2:
        last_slash = [pos for pos, char in enumerate(fname_PL2) if char == '\\'][-1]+1
        first_ = [pos for pos, char in enumerate(fname_PL2) if char == '_'][0]+1
        save_fname = fname_PL2[0:last_slash] + 'sorted_' + fname_PL2[first_:-4]
    else:
        save_fname = sys.argv[2]
    if not '.pickle' in save_fname[-7:]:
        save_fname = save_fname + '.pickle'
    log_dir = save_fname[0:-7] + '_std_logs'
    # if os.path.exists(log_dir):
    #     rmtree(log_dir)
    #     time.sleep(.5) # NEED SLEEP SO CAN DELETE BEFORE RECREATING!!!
    # os.makedirs(log_dir)
    print("Sorting data from PL2 file: ", fname_PL2)
    print("Output will be saved as: ", save_fname)

    """ Using a filter band less than about 300 Hz for high pass and/or less than
    about 6000 Hz for low pass can really influence the ZCA and ruin everything.
    A good clip width usually has little discontinuity but not excessive.
    sigma is usually better either high or low - about 2.75-3.25 or 4.0-4.5+.
    Basically if you get an up and down noise cluster out, then sorter can
    distinguish the noise from spikes. Otherwise the noise is in one of the
    clusters. With high thresholds you will want binary pursuit enable to find
    the remaining missed spikes.
    """
    # Run 49 at sigma 4, width 6-8 fitler 1000-8000
    spike_sort_args = {'sigma': 4.,
                       'clip_width': [-6e-4, 8e-4], 'filter_band': (1000, 8000),
                       'p_value_cut_thresh': 0.05, 'check_components': None,
                       'max_components': 10,
                       'min_firing_rate': 2., 'do_binary_pursuit': True,
                       'segment_duration': 300,
                       'segment_overlap': 150,
                       'add_peak_valley': False, 'do_branch_PCA': True,
                       'use_GPU': True, 'max_gpu_memory': None,
                       'save_1_cpu': False, 'use_rand_init': True,
                       'cleanup_neurons': False,
                       'verbose': True, 'test_flag': False, 'log_dir': log_dir,
                       'do_ZCA_transform': True}

    # use_voltage_file = 'C:\\Users\\plexon\\Documents\\Python Scripts\\voltage_49_1min'
    use_voltage_file = None

    if use_voltage_file is None:
        spike_sort_args['do_ZCA_transform'] = True
    else:
        spike_sort_args['do_ZCA_transform'] = False

    if spike_sort_args['cleanup_neurons']:
        print("CLEANUP IS ON WON'T BE ABLE TO TEST !")
    sys.path.append('c:\\users\\plexon\\documents\\python scripts')
    import PL2_read
    pl2_reader = PL2_read.PL2Reader(fname_PL2)

    if use_voltage_file is None or not use_voltage_file:
        if not spike_sort_args['do_ZCA_transform']:
            print("!!! WARNING !!!: ZCA transform is OFF but data is being loaded from PL2 file.")
        print("Reading voltage from file")
        raw_voltage = load_voltage_parallel(pl2_reader, 'SPKC')
        # t_t_start = int(40000 * 60 * 10)
        # t_t_stop =  int(40000 * 60 * 15)
        SProbe = electrode.SProbe16by2(pl2_reader.info['timestamp_frequency'], voltage_array=raw_voltage)#[:, t_t_start:t_t_stop])
    else:
        with open(use_voltage_file, 'rb') as fp:
            voltage_array = pickle.load(fp)
        SProbe = electrode.SProbe16by2(pl2_reader.info['timestamp_frequency'], voltage_array=voltage_array)

    filt_voltage = filter_parallel(SProbe, low_cutoff=spike_sort_args['filter_band'][0], high_cutoff=spike_sort_args['filter_band'][1])
    SProbe = electrode.SProbe16by2(pl2_reader.info['timestamp_frequency'], voltage_array=filt_voltage)
    # SProbe = electrode.SingleTetrode(pl2_reader.info['timestamp_frequency'], voltage_array=filt_voltage)
    SProbe.filter_band = spike_sort_args['filter_band']
    print("Start sorting")
    sort_data, work_items, sorter_info = spike_sort_parallel(SProbe, **spike_sort_args)

    print("Saving neurons file as", save_fname)
    with open(save_fname, 'wb') as fp:
        pickle.dump((sort_data, work_items, sorter_info), fp, protocol=-1)
