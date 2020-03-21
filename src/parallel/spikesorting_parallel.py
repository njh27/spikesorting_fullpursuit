import pickle
import os
import sys
import signal as sg
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
import PL2_read
import time
from traceback import print_tb



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
    voltage = np.frombuffer(data_dict['share_voltage']).reshape(data_dict['share_voltage_shape'])
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
    with mp.Pool(processes=psutil.cpu_count(logical=False), initializer=init_data_dict, initargs=(filt_X, Probe.voltage.shape)) as pool:
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
    voltage = np.frombuffer(data_dict['share_voltage']).reshape(data_dict['share_voltage_shape'])
    zca_data = np.matmul(data_dict['zca_matrix'][chan, :], voltage)
    result_voltage = np.frombuffer(data_dict['result_voltage']).reshape(data_dict['share_voltage_shape'])
    np.copyto(result_voltage[chan, :], zca_data)
    return None


def zca_parallel(shared_voltage, result_voltage, Probe, zca_matrix):
    init_dict = {'zca_matrix': zca_matrix, 'result_voltage': result_voltage}
    zca_results = []
    with mp.Pool(processes=psutil.cpu_count(logical=True), initializer=init_data_dict, initargs=(shared_voltage, Probe.voltage.shape, init_dict)) as pool:
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
    voltage = np.frombuffer(data_dict['share_voltage']).reshape(data_dict['share_voltage_shape'])
    abs_voltage = np.abs(voltage[chan, :])
    threshold = np.nanmedian(abs_voltage) / 0.6745
    threshold *= sigma
    n_crossings = np.count_nonzero(abs_voltage > threshold)

    return n_crossings, threshold, chan


def get_thresholds_and_work_order(shared_voltage, Probe, sigma):
    order_results = []
    with mp.Pool(processes=psutil.cpu_count(logical=True), initializer=init_data_dict, initargs=(shared_voltage, Probe.voltage.shape)) as pool:
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
    work_order = np.argsort(crossings_per_s)[-1::-1]

    return thresholds, work_order, crossings_per_s


def single_thresholds_and_order(Probe, sigma):

    thresholds = np.empty((Probe.num_electrodes))
    crossings_per_s = np.empty((Probe.num_electrodes))
    for chan in range(0, Probe.num_electrodes):
        abs_voltage = np.abs(Probe.get_voltage(chan))
        thresholds[chan] = np.nanmedian(abs_voltage) / 0.6745
        crossings_per_s[chan] = np.count_nonzero(abs_voltage > thresholds[chan])
    thresholds *= sigma
    work_order = np.argsort(crossings_per_s)[-1::-1]

    return thresholds, work_order, crossings_per_s


def allocate_cpus_by_chan(crossings_per_s):
    """ Assign CPUs/threads according to number of threshold crossings,
    THIS IS EXTREMELY APPROXIMATE since it counts time points
    above threshold, and each spike will have more than one of these.
    Sort time also depends on number of units and other factors but this
    is a decent starting point without any other information available. """

    n_cpus = psutil.cpu_count(logical=True)
    cpu_queue = mp.Manager().Queue(n_cpus)
    for cpu in range(n_cpus):
        cpu_queue.put(cpu)
    cpu_alloc = []
    median_crossings = np.median(crossings_per_s)
    for magnitude in crossings_per_s:
        # cpu_alloc.append(6)
        # print("GIVING EVERYONE 6 CPUS", flush=True)
        # continue
        if magnitude > 10*median_crossings:
            cpu_alloc.append(6)
        elif magnitude > 5*median_crossings:
            cpu_alloc.append(4)
        elif magnitude > median_crossings:
            cpu_alloc.append(2)
        else:
            cpu_alloc.append(1)

    return cpu_queue, cpu_alloc


def print_process_info(title):
    print(title, flush=True)
    print('module name:', __name__, flush=True)
    print('parent process:', os.getppid(), flush=True)
    print('process id:', os.getpid(), flush=True)


"""
    Since alignment is biased toward down neurons, up can be shifted. """
def check_upward_neurons(clips, event_indices, neuron_labels, curr_chan_inds,
                            clip_width, sampling_rate):
    templates, labels = segment_parallel.calculate_templates(clips[:, curr_chan_inds], neuron_labels)
    window, clip_width = segment_parallel.time_window_to_samples(clip_width, sampling_rate)
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


def spike_sort_one_chan(data_dict, use_cpus, chan, neighbors, sigma=4.5,
                        clip_width=[-6e-4, 10e-4], p_value_cut_thresh=0.01,
                        check_components=None, max_components=10,
                        min_firing_rate=1,
                        do_binary_pursuit=True, add_peak_valley=False,
                        do_branch_PCA=True, max_gpu_memory=None,
                        use_rand_init=True,
                        cleanup_neurons=False, verbose=False, test_flag=False,
                        log_dir=None):
    """
    """
    # Initialize variables in case this exits on error
    crossings, neuron_labels, clips, new_inds  = [], [], [], []
    exit_type = None
    def wrap_up():
        data_dict['results_dict'][chan] = (crossings, neuron_labels, new_inds)
        with open('temp_clips' + str(chan) + '.pickle', 'wb') as fp:
            pickle.dump(clips, fp, protocol=-1)
        data_dict['completed_chans'].append(chan)
        data_dict['exits_dict'][chan] = exit_type
        data_dict['completed_chans_queue'].put(chan)
        for cpu in use_cpus:
            data_dict['cpu_queue'].put(cpu)
        return
    try:
        # Print this process' errors and output to a file
        if not test_flag:
            if sys.platform == 'win32':
                sys.stdout = open(log_dir + "\\SpikeSortChan" + str(chan) + ".out", "w")
                sys.stderr = open(log_dir + "\\SpikeSortChan" + str(chan) + "_errors.out", "w")
                print_process_info('spike_sort_one_chan on channel' + str(chan))
            else:
                sys.stdout = open(log_dir + "/SpikeSortChan" + str(chan) + ".out", "w")
                sys.stderr = open(log_dir + "/SpikeSortChan" + str(chan) + "_errors.out", "w")
                print_process_info('spike_sort_one_chan on channel' + str(chan))

        # Setup threads and affinity based on use_cpus if not on mac OS
        if 'win32' == sys.platform:
            proc = psutil.Process()  # get self pid
            proc.cpu_affinity(use_cpus)
        if test_flag:
            mkl.set_num_threads(8)
        else:
            mkl.set_num_threads(len(use_cpus))
        voltage = np.frombuffer(data_dict['share_voltage']).reshape(data_dict['share_voltage_shape'])

        skip = np.amax(np.abs(clip_width)) / 2
        align_window = [skip, skip]
        if verbose: print("Identifying threshold crossings", flush=True)
        crossings = segment_parallel.identify_threshold_crossings(voltage[chan, :], data_dict, data_dict['thresholds'][chan], skip=skip, align_window=align_window)
        min_cluster_size = (np.floor(min_firing_rate * data_dict['n_samples'] / data_dict['sampling_rate'])).astype(np.int64)
        if min_cluster_size < 1:
            min_cluster_size = 1
        if verbose: print("Using minimum cluster size of", min_cluster_size, flush=True)

        exit_type = "Found crossings"

        median_cluster_size = min(100, int(np.around(crossings.size / 1000)))
        clips, valid_event_indices = segment_parallel.get_multichannel_clips(data_dict, voltage[neighbors, :], crossings, clip_width=clip_width, neighbor_thresholds=data_dict['thresholds'][neighbors])
        crossings = segment_parallel.keep_valid_inds([crossings], valid_event_indices)
        _, _, clip_samples, _, curr_chan_inds = segment_parallel.get_windows_and_indices(clip_width, data_dict['sampling_rate'], chan, neighbors)

        exit_type = "Found first clips"

        if verbose: print("Start initial clustering and merge", flush=True)
        # Do initial single channel sort
        scores = preprocessing.compute_pca(clips[:, curr_chan_inds],
                    check_components, max_components, add_peak_valley=add_peak_valley,
                    curr_chan_inds=np.arange(0, curr_chan_inds.size))
        n_random = max(100, np.around(crossings.size / 100)) if use_rand_init else 0
        neuron_labels = sort.initial_cluster_farthest(scores, median_cluster_size, n_random=n_random)
        neuron_labels = sort.merge_clusters(scores, neuron_labels,
                            split_only = False,
                            p_value_cut_thresh=p_value_cut_thresh)
        curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
        if verbose: print("Currently", curr_num_clusters.size, "different clusters", flush=True)

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
            if verbose: print("After SINGLE BRANCH", curr_num_clusters.size, "different clusters", flush=True)

        # Multi channel branch
        if data_dict['num_electrodes'] > 1 and do_branch_PCA:
            neuron_labels = branch_pca_2_0(neuron_labels, clips, curr_chan_inds,
                                p_value_cut_thresh=p_value_cut_thresh,
                                add_peak_valley=add_peak_valley,
                                check_components=check_components,
                                max_components=max_components,
                                method='pca')
            curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
            if verbose: print("After MULTI BRANCH", curr_num_clusters.size, "different clusters", flush=True)

        # Multi channel branch by channel
        if data_dict['num_electrodes'] > 1 and do_branch_PCA:
            neuron_labels = branch_pca_2_0(neuron_labels, clips, curr_chan_inds,
                                p_value_cut_thresh=p_value_cut_thresh,
                                add_peak_valley=add_peak_valley,
                                check_components=check_components,
                                max_components=max_components,
                                method='chan_pca')
            curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
            if verbose: print("After MULTI BY CHAN BRANCH", curr_num_clusters.size, "different clusters", flush=True)

        # Delete any clusters under min_cluster_size before binary pursuit
        if verbose: print("Current smallest cluster has", np.amin(n_per_cluster), "spikes", flush=True)
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
            if verbose: print("Done.", flush=True)
            # Raise error to force exit and wrap_up()
            raise RuntimeError("No clusters over min_firing_rate.")

        # Realign any units that have a template with peak > valley
        crossings, units_shifted = check_upward_neurons(clips, crossings,
                                    neuron_labels, curr_chan_inds, clip_width,
                                    data_dict['sampling_rate'])
        if verbose: print("Found", len(units_shifted), "upward neurons that were realigned", flush=True)
        exit_type = "Found upward spikes"
        if len(units_shifted) > 0:
            clips, valid_event_indices = segment_parallel.get_multichannel_clips(
                                            data_dict, voltage[neighbors, :],
                                            crossings, clip_width=clip_width,
                                            neighbor_thresholds=data_dict['thresholds'][neighbors])
            crossings, neuron_labels = segment_parallel.keep_valid_inds(
                                        [crossings, neuron_labels],
                                        valid_event_indices)

        exit_type = "Finished sorting clusters"
        # Realign spikes based on correlation with current cluster templates before doing binary pursuit

        crossings, neuron_labels, _ = segment_parallel.align_events_with_template(data_dict, voltage[chan, :], neuron_labels, crossings, clip_width=clip_width)
        if do_binary_pursuit:
            if verbose: print("currently", np.unique(neuron_labels).size, "different clusters", flush=True)
            if verbose: print("Doing binary pursuit", flush=True)
            # crossings, neuron_labels, new_inds = overlap_parallel.binary_pursuit_secret_spikes(
            #                 data_dict, chan, neighbors, voltage[neighbors, :],
            #                 neuron_labels, crossings, data_dict['thresholds'][chan],
            #                 clip_width, return_adjusted_clips=False)
            # clips, valid_event_indices = segment_parallel.get_multichannel_clips(data_dict, voltage[neighbors, :], crossings, clip_width=clip_width, neighbor_thresholds=data_dict['thresholds'][neighbors])
            # crossings, neuron_labels = segment_parallel.keep_valid_inds([crossings, neuron_labels], valid_event_indices)

            with data_dict['gpu_lock']:
                crossings, neuron_labels, new_inds, clips = binary_pursuit_parallel.binary_pursuit(
                            data_dict, chan, neighbors, voltage[neighbors, :],
                            crossings, neuron_labels, clip_width,
                            thresholds=data_dict['thresholds'][neighbors],
                            kernels_path=None, max_gpu_memory=max_gpu_memory)
            exit_type = "Finished binary pursuit"
        else:
            clips, valid_event_indices = segment_parallel.get_multichannel_clips(data_dict, voltage[neighbors, :], crossings, clip_width=clip_width, neighbor_thresholds=data_dict['thresholds'][neighbors])
            crossings, neuron_labels = segment_parallel.keep_valid_inds([crossings, neuron_labels], valid_event_indices)
            new_inds = np.zeros(crossings.size, dtype='bool')

        if verbose: print("currently", np.unique(neuron_labels).size, "different clusters", flush=True)
        if verbose: print("Done sorting", flush=True)
        # Map labels starting at zero and put labels in order
        sort.reorder_labels(neuron_labels)
        if verbose: print("Successfully completed channel ", str(chan), flush=True)
        exit_type = "Success"
    except Exception as err:
        exit_type = err #exit_type + "--Then FAILED TO COMPLETE!"
        print_tb(err.__traceback__)
        if test_flag:
            raise # Reraise any exceptions in test mode only
    finally:
        wrap_up()


def init_data_dict(X, X_shape, init_dict=None):
    global data_dict
    data_dict = {}
    data_dict['share_voltage'] = X
    data_dict['share_voltage_shape'] = X_shape
    if init_dict is not None:
        for k in init_dict.keys():
            data_dict[k] = init_dict[k]
    return


def spike_sort_parallel(Probe, sigma=4.5, clip_width=[-6e-4, 10e-4],
                        filter_band=(300, 6000), p_value_cut_thresh=0.01,
                        check_components=None, max_components=10,
                        min_firing_rate=1, do_binary_pursuit=True,
                        add_peak_valley=False, do_branch_PCA=True,
                        max_gpu_memory=None, use_rand_init=True, cleanup_neurons=False,
                        verbose=False, test_flag=False, log_dir=None,
                        do_ZCA_transform=True):

    kwargs = {'sigma': sigma, 'clip_width': clip_width,
              'p_value_cut_thresh': p_value_cut_thresh,
              'check_components': check_components,
              'max_components': max_components,
              'min_firing_rate': min_firing_rate,
              'do_binary_pursuit': do_binary_pursuit,
              'add_peak_valley': add_peak_valley,
              'do_branch_PCA': do_branch_PCA,
              'max_gpu_memory': max_gpu_memory,
              'use_rand_init': use_rand_init,
              'cleanup_neurons': cleanup_neurons,
              'verbose': verbose, 'test_flag': test_flag, 'log_dir': log_dir}
    init_dict = {'num_electrodes': Probe.num_electrodes, 'sampling_rate': Probe.sampling_rate,
                 'n_samples': Probe.n_samples, 'results_dict': mp.Manager().dict(),
                 'completed_chans': mp.Manager().list(), 'exits_dict': mp.Manager().dict(),
                 'gpu_lock': mp.Lock(), 'filter_band': filter_band}

    print("Allocating and copying voltage")
    X = mp.RawArray('d', Probe.voltage.size)
    X_np = np.frombuffer(X).reshape(Probe.voltage.shape)
    np.copyto(X_np, Probe.voltage)
    if do_ZCA_transform:
        print("Doing ZCA transform")
        thresholds, _, _ = single_thresholds_and_order(Probe, sigma)
        zca_cushion = (2 * np.ceil(np.amax(np.abs(clip_width)) * Probe.sampling_rate)).astype(np.int64)
        zca_matrix = preprocessing.get_noise_sampled_zca_matrix(Probe.voltage, thresholds, sigma, zca_cushion, n_samples=1e7)
        Probe.voltage = zca_matrix @ Probe.voltage
        # with open('voltage_49_10min', 'wb') as fp:
        #     pickle.dump(Probe.voltage, fp, protocol=-1)

    X_np = np.frombuffer(X).reshape(Probe.voltage.shape)
    np.copyto(X_np, Probe.voltage)

    print("Determining work order and thresholds")
    # thresholds, work_order, crossings_per_s = get_thresholds_and_work_order(X, Probe, sigma)
    thresholds, work_order, crossings_per_s = single_thresholds_and_order(Probe, sigma)
    init_dict['thresholds'] = thresholds
    cpu_queue, cpu_alloc = allocate_cpus_by_chan(crossings_per_s)
    init_dict['cpu_queue'] = cpu_queue
    completed_chans_queue = mp.Manager().Queue(work_order.size)
    init_dict['completed_chans_queue'] = completed_chans_queue

    # Call init function to ensure data_dict is globally available before passing
    # it into each process
    init_data_dict(X, Probe.voltage.shape, init_dict)
    processes = []
    proc_chan_index = []
    completed_chans_index = 0
    print("Starting sorting pool")
    if test_flag:
        # Do everything in order to keep random numbers consistent with single channel sorter
        work_order.sort()
    for chan in work_order:
        # With timeout=None, this will block until sufficient cpus are available
        # as requested by cpu_alloc
        use_cpus = [cpu_queue.get(timeout=None) for x in range(cpu_alloc[chan])]
        n_complete = len(data_dict['completed_chans']) # Do once to avoid race
        if n_complete > completed_chans_index:
            for ci in range(completed_chans_index, n_complete):
                print("Completed chan ", data_dict['completed_chans'][ci])
                print("Exited with status: ", data_dict['exits_dict'][data_dict['completed_chans'][ci]])
                completed_chans_index += 1

                if not test_flag:
                    done_index = proc_chan_index.index(data_dict['completed_chans'][ci])
                    del proc_chan_index[done_index]
                    processes[done_index].join()
                    processes[done_index].close()
                    del processes[done_index]

        if not test_flag:
            print("Starting chan {0} on CPUs {1}".format(chan, use_cpus))
            time.sleep(.5) # NEED SLEEP SO PROCESSES AREN'T MADE TOO FAST AND FAIL!!!
            proc = mp.Process(target=spike_sort_one_chan, args=(data_dict, use_cpus, chan, Probe.get_neighbors(chan)), kwargs=kwargs)
            proc.start()
            processes.append(proc)
            proc_chan_index.append(chan)
        else:
            print("Doing one process on channel", chan)
            spike_sort_one_chan(data_dict, use_cpus, chan, Probe.get_neighbors(chan), **kwargs)
            print("finished sort one chan")

    if not test_flag:
        # Wait here a bit to print out chans as they complete and to ensure
        # no process are left behind, as can apparently happen if you attempt to
        # join() too soon without being sure everything is finished (especially using Queue's)
        while completed_chans_index < len(work_order) and not test_flag:
            finished_chan = completed_chans_queue.get()
            try:
                done_index = proc_chan_index.index(finished_chan)
            except ValueError:
                # This channel was already finished above so just clearing out
                # completed_chans_queue
                continue
            print("Completed chan ", finished_chan)
            print("Exited with status: ", data_dict['exits_dict'][finished_chan])
            completed_chans_index += 1

            del proc_chan_index[done_index]
            processes[done_index].join()
            processes[done_index].close()
            del processes[done_index]

    crossings = [[] for x in range(0, Probe.num_electrodes)]
    labels = [[] for x in range(0, Probe.num_electrodes)]
    waveforms = [[] for x in range(0, Probe.num_electrodes)]
    new_waveforms = [[] for x in range(0, Probe.num_electrodes)]
    for chan in range(0, Probe.num_electrodes):
        if chan in data_dict['results_dict'].keys():
            crossings[chan] = data_dict['results_dict'][chan][0]
            labels[chan] = data_dict['results_dict'][chan][1]
            with open('temp_clips' + str(chan) + '.pickle', 'rb') as fp:
                waveforms[chan] = pickle.load(fp)
            os.remove('temp_clips' + str(chan) + '.pickle')
            new_waveforms[chan] = data_dict['results_dict'][chan][2]

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

    return neurons


if __name__ == '__main__':
    import os
    import sys
    from shutil import rmtree

    # python.exe SpikeSortingParallel.py C:\Nate\LearnDirTunePurk_Yoda_49.pl2 C:\Nate\neurons_yoda_49.pickle

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
        save_fname = fname_PL2[0:last_slash] + 'neurons_' + fname_PL2[first_:-4]
    else:
        save_fname = sys.argv[2]
    if not '.pickle' in save_fname[-7:]:
        save_fname = save_fname + '.pickle'
    log_dir = save_fname[0:-7] + '_std_logs'
    if os.path.exists(log_dir):
        rmtree(log_dir)
        time.sleep(.5) # NEED SLEEP SO CAN DELETE BEFORE RECREATING!!!
    os.makedirs(log_dir)
    print("Sorting data from PL2 file: ", fname_PL2)
    print("Output neurons will be saved as: ", save_fname)

    """ Using a filter band less than about 300 Hz for high pass and/or less than
    about 6000 Hz for low pass can really influence the ZCA and ruin everything.
    A good clip width usually has little discontinuity but not excessive.
    sigma is usually better either high or low - about 2.75-3.25 or 4.0-4.5+.
    Basically if you get an up and down noise cluster out, then sorter can
    distinguish the noise from spikes. Otherwise the noise is in one of the
    clusters. With high thresholds you will want binary pursuit enable to find
    the remaining missed spikes.
    """

    spike_sort_args = {'sigma': 4.5,
                       'clip_width': [-6e-4, 8e-4], 'filter_band': (300, 6000),
                       'p_value_cut_thresh': 0.001, 'check_components': None,
                       'max_components': 10,
                       'min_firing_rate': 2, 'do_binary_pursuit': True,
                       'add_peak_valley': False, 'do_branch_PCA': True,
                       'max_gpu_memory': None, 'use_rand_init': True,
                       'cleanup_neurons': False,
                       'verbose': True, 'test_flag': False, 'log_dir': log_dir,
                       'do_ZCA_transform': True}

    use_voltage_file = 'C:\\Users\\plexon\\Documents\\Python Scripts\\voltage_49_1min'
    # use_voltage_file = None

    if use_voltage_file is None:
        spike_sort_args['do_ZCA_transform'] = True
    else:
        spike_sort_args['do_ZCA_transform'] = False

    if spike_sort_args['cleanup_neurons']:
        print("CLEANUP IS ON WON'T BE ABLE TO TEST !")

    pl2_reader = PL2_read.PL2Reader(fname_PL2)

    if use_voltage_file is None or not use_voltage_file:
        if not spike_sort_args['do_ZCA_transform']:
            print("!!! WARNING !!!: ZCA transform is OFF but data is being loaded from PL2 file.")
        print("Reading voltage from file")
        raw_voltage = load_voltage_parallel(pl2_reader, 'SPKC')
        # t_t_start = int(40000 * 60 * 25)
        # t_t_stop =  int(40000 * 60 * 35)
        SProbe = electrode.SProbe16by2(pl2_reader.info['timestamp_frequency'], voltage_array=raw_voltage) #[:, t_t_start:t_t_stop])
    else:
        with open(use_voltage_file, 'rb') as fp:
            voltage_array = pickle.load(fp)
        SProbe = electrode.SProbe16by2(pl2_reader.info['timestamp_frequency'], voltage_array=voltage_array)

    filt_voltage = filter_parallel(SProbe, low_cutoff=spike_sort_args['filter_band'][0], high_cutoff=spike_sort_args['filter_band'][1])
    SProbe = electrode.SProbe16by2(pl2_reader.info['timestamp_frequency'], voltage_array=filt_voltage)
    # SProbe = electrode.SingleTetrode(pl2_reader.info['timestamp_frequency'], voltage_array=filt_voltage)
    SProbe.filter_band = spike_sort_args['filter_band']
    print("Start sorting")
    neurons = spike_sort_parallel(SProbe, **spike_sort_args)

    print("Saving neurons file")
    with open(save_fname, 'wb') as fp:
        pickle.dump(neurons, fp, protocol=-1)
