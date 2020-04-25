import numpy as np
from spikesorting_python.src import segment
from spikesorting_python.src.sort import merge_clusters
from spikesorting_python.src import preprocessing
from spikesorting_python.src.c_cython import sort_cython
from scipy import stats
from scipy.optimize import nnls, lsq_linear
import copy
import warnings


import matplotlib.pyplot as plt





def remove_across_channel_duplicate_neurons(neuron_summary, overlap_time=2.5e-4, ccg_peak_threshold=2):
    """ Assumes input neurons list is already sorted from most desireable to least
        desireable units, because neurons toward the end of neuron summary will be
        deleted if they violate with neurons to their right.  If consolidation
        is desired, that should be run first because this will delete neurons that
        potentially should have been combined. Duplicates are only removed if they
        are within each other's neighborhoods. """

    max_samples = int(round(overlap_time * neuron_summary[0]['sampling_rate']))
    remove_neuron = [False for x in range(0, len(neuron_summary))]
    for (i, neuron1) in enumerate(neuron_summary):
        if remove_neuron[i]:
            continue
        for j in range(i+1, len(neuron_summary)):
            neuron2 = neuron_summary[j]
            if remove_neuron[j]:
                continue
            if np.intersect1d(neuron1['neighbors'], neuron2['neighbors']).size == 0:
                continue
            # Check if these two have overlapping spikes, i.e. an unusual peak in their
            # CCG at extremely fine timescales
            actual_overlaps = zero_symmetric_ccg(neuron1['spike_indices'], neuron2['spike_indices'], max_samples, max_samples)[0]
            if (actual_overlaps[1] > ccg_peak_threshold * (actual_overlaps[0] + actual_overlaps[2])):
                remove_neuron[j] = True

    # Remove any offending neurons
    for n, should_remove in reversed(list(enumerate(remove_neuron))):
        if should_remove:
            print("Deleting neuron", n, "for spike violations")
            del neuron_summary[n]

    return neuron_summary


def find_overlapping_spike_bool(spikes1, spikes2, max_samples=20, except_equal=False):
    """ Finds an index into spikes2 that indicates whether a spike index of spike2
        occurs within +/- max_samples of a spike index in spikes1.  Spikes1 and 2
        are numpy arrays of spike indices in units of samples. Input spikes1 and
        spikes2 will be SORTED IN PLACE because this function won't work if they
        are not ordered. If spikes1 == spikes2 and except_equal is True, the first
        spike in the pair of overlapping spikes is flagged as True in the output
        overlapping_spike_bool. """
    max_samples = np.ceil(max_samples).astype(np.int64)
    overlapping_spike_bool = np.zeros(spikes2.size, dtype=np.bool)
    spike_1_index = 0
    n_spike_1 = 0
    for spike_2_index in range(0, spikes2.size):
        for spike_1_index in range(n_spike_1, spikes1.size):
            if except_equal and spikes1[spike_1_index] == spikes2[spike_2_index]:
                continue
            if ((spikes1[spike_1_index] >= spikes2[spike_2_index] - max_samples) and
               (spikes1[spike_1_index] <= spikes2[spike_2_index] + max_samples)):
                overlapping_spike_bool[spike_2_index] = True
                break
            elif spikes1[spike_1_index] > spikes2[spike_2_index] + max_samples:
                break
        n_spike_1 = spike_1_index

    return overlapping_spike_bool


def remove_binary_pursuit_duplicates(event_indices, new_spike_bool, tol_inds=1):
    """ Preferentially KEEPS spikes found in binary pursuit.
    """
    keep_bool = np.ones(event_indices.size, dtype=np.bool)
    curr_index = 0
    next_index = 1
    while next_index < event_indices.size:
        if event_indices[next_index] - event_indices[curr_index] <= tol_inds:
            if new_spike_bool[curr_index] and ~new_spike_bool[next_index]:
                keep_bool[next_index] = False
                # keep_bool[curr_index] = False
                curr_index = next_index
            elif ~new_spike_bool[curr_index] and new_spike_bool[next_index]:
                keep_bool[curr_index] = False
                # keep_bool[next_index] = False
            elif new_spike_bool[curr_index] and new_spike_bool[next_index]:
                # Should only be possible for first index?
                keep_bool[next_index] = False
            else:
                # Two spikes with same index, neither from binary pursuit.
                #  Should be chosen based on templates or some other means.
                pass
        else:
            curr_index = next_index
        next_index += 1

    return keep_bool


def remove_spike_event_duplicates(event_indices, clips, unit_template, tol_inds=1):
    """
    """
    keep_bool = np.ones(event_indices.size, dtype=np.bool)
    template_norm = unit_template / np.linalg.norm(unit_template)
    curr_index = 0
    next_index = 1
    while next_index < event_indices.size:
        if event_indices[next_index] - event_indices[curr_index] <= tol_inds:
            projections = np.vstack((clips[curr_index, :], clips[next_index, :])) @ template_norm
            if projections[0] >= projections[1]:
                # current spike is better or equal
                keep_bool[next_index] = False
                next_index += 1
            else:
                # next spike is better
                keep_bool[curr_index] = False
                curr_index = next_index
                next_index += 1
        else:
            curr_index = next_index
            next_index += 1

    return keep_bool


def remove_spike_event_duplicates_across_chans(event_indices, clips, thresholds,
                                               tol_inds=1):
    """
    """
    keep_bool = np.ones(event_indices.size, dtype=np.bool)
    curr_index = 0
    next_index = 1
    while next_index < event_indices.size:
        if event_indices[next_index] - event_indices[curr_index] <= tol_inds:
            curr_thresh_ratio = (np.amax(clips[curr_index, :]) - np.amin(clips[curr_index, :])) \
                                / thresholds[curr_index]
            next_thresh_ratio = (np.amax(clips[next_index, :]) - np.amin(clips[next_index, :])) \
                                / thresholds[next_index]
            # Choose the clip with higher signal to threshold ratio
            if curr_thresh_ratio >= next_thresh_ratio:
                # current spike is better or equal
                keep_bool[next_index] = False
                next_index += 1
            else:
                # next spike is better
                keep_bool[curr_index] = False
                curr_index = next_index
                next_index += 1
        else:
            curr_index = next_index
            next_index += 1

    return keep_bool


def compute_spike_trains(spike_indices_list, bin_width_samples, min_max_samples):

    if type(spike_indices_list) is not list:
        spike_indices_list = [spike_indices_list]
    if type(min_max_samples) is not list:
        min_max_samples = [0, min_max_samples]
    bin_width_samples = int(bin_width_samples)
    train_len = int(np.ceil((min_max_samples[1] - min_max_samples[0] + bin_width_samples/2) / bin_width_samples) + 1) # Add one because it's a time/samples slice
    spike_trains_list = [[] for x in range(0, len(spike_indices_list))]

    for ind, unit in enumerate(spike_indices_list):
        unit_select = np.logical_and(unit >= min_max_samples[0], unit <= min_max_samples[1])
        # Convert the spike indices to units of bin_width_samples
        spikes = (np.floor((unit[unit_select] + bin_width_samples/2 - min_max_samples[0]) / bin_width_samples)).astype(np.int64)
        spike_trains_list[ind] = np.zeros(train_len, dtype=np.bool)
        spike_trains_list[ind][spikes] = True

    return spike_trains_list if len(spike_trains_list) > 1 else spike_trains_list[0]


def zero_symmetric_ccg(spikes_1, spikes_2, samples_window=40, d_samples=40, return_trains=False):
    """ Returns the cross correlogram of
        spikes_2 relative to spikes_1 at each lag from -samples_window to
        +samples_window in steps of d_samples.  Bins of the CCG are centered
        at these steps, and so will include a bin centered on lag zero if d_samples
        divides evently into samples_window.

        This is a more basic version of other CCG functions in that it is made
        to quickly compute a simple, zero-centered CCG with integer window
        value and integer step size where the window and step are in UNITS OF
        SAMPLES!  Remember that CCG is computed over a BINARY spike train,
        so using excessively large bins will actually cause the overlaps
        to decrease. """

    samples_window = int(samples_window)
    d_samples = int(d_samples)
    samples_axis = np.arange(-1 * samples_window, samples_window+d_samples, d_samples).astype(np.int64)
    counts = np.zeros(samples_axis.size, dtype=np.int64)

    # Convert the spike indices to units of d_samples
    spikes_1 = (np.floor((spikes_1 + d_samples/2) / d_samples)).astype(np.int64)
    spikes_2 = (np.floor((spikes_2 + d_samples/2) / d_samples)).astype(np.int64)
    samples_axis = (np.floor((samples_axis + d_samples/2) / d_samples)).astype(np.int64)

    # Convert to spike trains
    train_len = int(max(spikes_1[-1], spikes_2[-1])) + 1 # This is samples, so need to add 1

    spike_trains_1 = np.zeros(train_len, dtype=np.bool)
    spike_trains_1[spikes_1] = True
    spike_trains_2 = np.zeros(train_len, dtype=np.bool)
    spike_trains_2[spikes_2] = True

    for lag_ind, lag in enumerate(samples_axis):
        # Construct a timeseries for neuron 2 based on the current spike index as the
        # zero point
        if lag > 0:
            counts[lag_ind] = np.count_nonzero(np.logical_and(spike_trains_1[0:-1*lag],
                                            spike_trains_2[lag:]))
        elif lag < 0:
            counts[lag_ind] = np.count_nonzero(np.logical_and(spike_trains_2[0:lag],
                                            spike_trains_1[-1*lag:]))
        else:
            counts[lag_ind] = np.count_nonzero(np.logical_and(spike_trains_1, spike_trains_2))

    if return_trains:
        return counts, samples_axis, spike_trains_1, spike_trains_2
    else:
        return counts, samples_axis


def calculate_expected_overlap(n1, n2, overlap_time, sampling_rate):
    """ Returns the expected number of overlapping spikes between neuron 1 and
    neuron 2 within a time window 'overlap_time' assuming independent spiking.
    As usual, spike_indices within each neuron must be sorted. """
    first_index = max(n1['spike_indices'][0], n2['spike_indices'][0])
    last_index = min(n1['spike_indices'][-1], n2['spike_indices'][-1])
    num_ms = int(np.ceil((last_index - first_index) / (sampling_rate / 1000)))
    if num_ms <= 0:
        # Neurons never fire at the same time, so expected overlap is 0
        return 0.

    # Find spike indices from each neuron that fall within the same time window
    # Should be impossible to return None because of num_ms check above
    n1_start = next((idx[0] for idx, val in np.ndenumerate(n1['spike_indices']) if val >= first_index), None)
    n1_stop = next((idx[0] for idx, val in np.ndenumerate(n1['spike_indices'][::-1]) if val <= last_index), None)
    n1_stop = n1['spike_indices'].shape[0] - n1_stop - 1 # Not used as slice so -1
    n2_start = next((idx[0] for idx, val in np.ndenumerate(n2['spike_indices']) if val >= first_index), None)
    n2_stop = next((idx[0] for idx, val in np.ndenumerate(n2['spike_indices'][::-1]) if val <= last_index), None)
    n2_stop = n2['spike_indices'].shape[0] - n2_stop - 1 # Not used as slice so -1

    # Infer number of spikes in overlapping window based on first and last index
    n1_count = n1_stop - n1_start
    n2_count = n2_stop - n2_start
    expected_overlap = (overlap_time * 1000 * n1_count * n2_count) / num_ms

    return expected_overlap


def calc_spike_width(clips, clip_width, sampling_rate):
    center_ind = int(round(np.abs(clip_width[0])*sampling_rate)) + 1
    template = np.mean(clips, axis=0)
    if template[center_ind] > 0:
        template *= -1
    first_peak = np.argmax(template[0:center_ind])
    second_peak = np.argmax(template[center_ind+1:]) + center_ind
    spike_width = second_peak - first_peak

    return spike_width


def calc_duplicate_tol_inds(spike_indices, sampling_rate,
                            absolute_refractory_period, clip_width):
    all_isis = np.diff(spike_indices)
    refractory_inds = int(round(absolute_refractory_period * sampling_rate))
    clip_inds = int(round(np.amax(np.abs(clip_width)) * sampling_rate))
    bin_edges = np.arange(0, 2*refractory_inds + 1, 1)
    counts, xval = np.histogram(all_isis, bin_edges)
    if refractory_inds - clip_inds < 2:
        raise ValueError("REFRACTORY AND CLIP INDICES ARE BASICALLY EQUAL! I should probably fix this...")
    mean_obs = np.mean(counts[clip_inds:refractory_inds+1])
    mean_std = np.std(counts[clip_inds:refractory_inds+1])
    for c_ind in range(clip_inds, -1, -1):
        # NOTE: this stops at 0!
        if (counts[c_ind] < mean_obs - 3*mean_std) \
            or (counts[c_ind] > mean_obs + 3*mean_std):
            break
    duplicate_tol_inds = c_ind+1
    adjusted_num_isi_violations = np.sum(counts[duplicate_tol_inds:refractory_inds+1])
    # print("Values were", duplicate_tol_inds, mean_obs, mean_std, clip_inds, refractory_inds, adjusted_num_isi_violations)
    # plt.bar(xval[0:-1], counts, width=xval[1]-xval[0], align='edge')
    # plt.xlim([0, 70])
    # plt.axvline(duplicate_tol_inds)
    # plt.show()

    return duplicate_tol_inds, adjusted_num_isi_violations


def calc_fraction_mua_to_peak(spike_indices, sampling_rate, duplicate_tol_inds,
                         absolute_refractory_period, check_window=0.5):

    # duplicate_tol_inds, adjusted_num_isi_violations = calc_duplicate_tol_inds(
    #                         spike_indices, sampling_rate,
    #                         absolute_refractory_period, clip_width)
    # duplicate_tol_inds = calc_spike_width(clips, clip_width, sampling_rate) + 1
    all_isis = np.diff(spike_indices)
    refractory_inds = int(round(absolute_refractory_period * sampling_rate))
    bin_width = refractory_inds - duplicate_tol_inds
    if bin_width <= 0:
        print("duplicate_tol_inds encompasses absolute_refractory_period. duplicate tolerence enforced at 1.")
        duplicate_tol_inds = 1
        bin_width = refractory_inds - duplicate_tol_inds
    check_inds = int(round(check_window * sampling_rate))
    bin_edges = np.arange(duplicate_tol_inds+1, check_inds + bin_width, bin_width)
    counts, xval = np.histogram(all_isis, bin_edges)
    isi_peak = np.amax(counts)
    num_isi_violations = counts[0]
    # num_isi_violations = np.count_nonzero(all_isis < refractory_inds)
    # n_duplicates = np.count_nonzero(all_isis <= duplicate_tol_inds)
    # num_isi_violations -= n_duplicates
    # num_isi_violations = adjusted_num_isi_violations
    if num_isi_violations < 0:
        num_isi_violations = 0
    if isi_peak == 0:
        isi_peak = max(num_isi_violations, 1.)
    fraction_mua_to_peak = num_isi_violations / isi_peak

    # print("Values were", num_isi_violations, isi_peak, n_duplicates, bin_width, refractory_inds)
    # plt.bar(xval[0:-1], counts, width=xval[1]-xval[0], align='edge')
    # plt.xlim([0, 70])
    # plt.axvline(duplicate_tol_inds)
    # plt.show()
    return fraction_mua_to_peak


def calc_isi_violation_rate(spike_indices, sampling_rate,
        absolute_refractory_period, duplicate_tol_inds):
    """ Compute the spiking activity that occurs during the ISI violation
    period, absolute_refractory_period, relative to the total number of
    spikes within the given segment, 'seg' and unit label 'label'.
    spike_indices must be in sorted order for this to work. """

    index_isi = np.diff(spike_indices)
    refractory_adjustment = duplicate_tol_inds / sampling_rate
    if absolute_refractory_period - refractory_adjustment <= 0:
        print("duplicate_tol_inds encompasses absolute_refractory_period. duplicate tolerence enforced at", 1)
        duplicate_tol_inds = 1
        refractory_adjustment = 0
    num_isi_violations = np.count_nonzero(index_isi / sampling_rate
                                          < absolute_refractory_period)
    n_duplicates = np.count_nonzero(index_isi <= duplicate_tol_inds)
    # Remove duplicate spikes from this computation and adjust the number
    # of spikes and time window accordingly
    num_isi_violations -= n_duplicates
    isi_violation_rate = num_isi_violations \
                         * (1.0 / (absolute_refractory_period - refractory_adjustment))\
                         / (spike_indices.size - n_duplicates)
    return isi_violation_rate

def mean_firing_rate(spike_indices, sampling_rate):
    """ Compute mean firing rate for input spike indices. Spike indices must be
    in sorted order for this to work. """
    if spike_indices[0] == spike_indices[-1]:
        return 0. # Only one spike
    mean_rate = sampling_rate * spike_indices.size / (spike_indices[-1] - spike_indices[0])
    return mean_rate

def fraction_mua(spike_indices, sampling_rate, duplicate_tol_inds,
                 absolute_refractory_period):
    """ Estimate the fraction of noise/multi-unit activity (MUA) using analysis
    of ISI violations. We do this by looking at the spiking activity that
    occurs during the ISI violation period. """
    isi_violation_rate = calc_isi_violation_rate(spike_indices, sampling_rate,
                                absolute_refractory_period, duplicate_tol_inds)
    mean_rate = mean_firing_rate(spike_indices, sampling_rate)
    if mean_rate == 0.:
        return 0.
    else:
        return isi_violation_rate / mean_rate


class WorkItemSummary(object):
    """
    """
    def __init__(self, sort_data, work_items, sort_info,
                 duplicate_tol_inds=1, absolute_refractory_period=10e-4,
                 max_mua_ratio=0.05, n_max_merge_test_clips=None,
                 merge_test_overlap_indices=None, verbose=False):

        self.check_input_data(sort_data, work_items)
        self.sort_info = sort_info
        self.n_chans = self.sort_info['n_channels']
        self.n_segments = self.sort_info['n_segments']
        self.duplicate_tol_inds = duplicate_tol_inds
        self.absolute_refractory_period = absolute_refractory_period
        self.max_mua_ratio = max_mua_ratio
        self.n_max_merge_test_clips = n_max_merge_test_clips
        if n_max_merge_test_clips is None:
            self.n_max_merge_test_clips = np.inf
        if merge_test_overlap_indices is None:
            merge_test_overlap_indices = work_items[0]['overlap']
        self.merge_test_overlap_indices = merge_test_overlap_indices

        self.is_stitched = False # Repeated stitching can change results so track
        self.last_overlap_ratio_threshold = np.inf
        self.verbose = verbose
        # Organize sort_data to be arranged by channel and segment.
        self.organize_sort_data()
        # Put all segment data in temporal order
        self.temporal_order_sort_data()

    def check_input_data(self, sort_data, work_items):
        """ Quick check to see if everything looks right with sort_data and
        work_items before proceeding. """
        if len(sort_data) != len(work_items):
            raise ValueError("Sort data and work items must be lists of the same length")
        # Make sure sort_data and work items are ordered one to one
        # by ordering their IDs
        sort_data.sort(key=lambda x: x[4])
        work_items.sort(key=lambda x: x['ID'])
        for s_data, w_item in zip(sort_data, work_items):
            if len(s_data) != 5:
                raise ValueError("sort_data for item", s_data[4], "is not the right size")
            if s_data[4] != w_item['ID']:
                raise ValueError("Could not match work order of sort data and work items. First failure on sort_data", s_data[4], "work item ID", w_item['ID'])
            empty_count = 0
            n_spikes = len(s_data[0])
            for di in range(0, 4):
                if len(s_data[di]) == 0:
                    empty_count += 1
                else:
                    pass
                    # This is caught below by empty_count !=4 ...
                    # if s_data[di].shape[0] != n_spikes:
                    #     raise ValueError("Every data element of sort_data list must indicate the same number of spikes. Element", w_item['ID'], "does not.")
            if empty_count > 0:
                # Require that if any data element is empty, all are empty (this
                # is how they should be coming out of sorter)
                for di in range(0, 4):
                    s_data[di] = []
                if empty_count != 4:
                    sort_data_message = ("One element of sort data, but not " \
                                        "all, indicated no spikes for work " \
                                        "item {0} on channel {1}, segment " \
                                        "{2}. All data for this item are " \
                                        "being removed.".format(w_item['ID'], w_item['channel'], w_item['seg_number']))
                    warnings.warn(sort_data_message, RuntimeWarning, stacklevel=2)
        self.sort_data = sort_data
        self.work_items = work_items

    def organize_sort_data(self):
        """Takes the output of the spike sorting algorithms and organizes is by
        channel in segment order.
        Parameters
        ----------
        sort_data: list
            Each element contains a list of the 4 outputs of spike_sort_segment,
            and the ID of the work item that created it, split up by
            individual channels into 'work_items' (or spike_sort_parallel output)
            "crossings, labels, waveforms, new_waveforms", in this order, for a
            single segment and a single channel of sorted data.
            len(sort_data) = number segments * channels == len(work_items)
        work_items: list
            Each list element contains the dictionary of information pertaining to
            the corresponding element in sort_data.
        n_chans: int
            Total number of channels sorted. If None this number is inferred as
            the maximum channel found in work_items['channel'] + 1.
        Returns
        -------
        Returns None but reassigns class attributes as follows:
        self.sort_data : sort_data list of dictionaries
            A list of the original elements of sort_data and organzied such that
            each list index corresponds to a channel number, and within each channel
            sublist data are in ordered by segment number as indicated in work_items.
        self.work_items : work_items list of dictionaries
            Same as for organized data but for the elements of work_items.
        """
        organized_data = [[] for x in range(0, self.n_chans)]
        organized_items = [[] for x in range(0, self.n_chans)]
        for chan in range(0, self.n_chans):
            chan_items, chan_data = zip(*[[x, y] for x, y in sorted(
                                        zip(self.work_items, self.sort_data), key=lambda pair:
                                        pair[0]['seg_number'])
                                        if x['channel'] == chan])
            organized_data[chan].extend(chan_data)
            organized_items[chan].extend(chan_items)
        self.sort_data = organized_data
        self.work_items = organized_items

    def temporal_order_sort_data(self):
        """ Places all data within each segment in temporal order according to
        the spike event times. Use 'stable' sort for output to be repeatable
        because overlapping segments and binary pursuit can return identical
        dupliate spikes that become sorted in different orders. """
        self.neuron_summary_seg_inds = []
        for chan in range(0, self.n_chans):
            for seg in range(0, self.n_segments):
                if chan == 0:
                    # This is the same for every channel so only do once
                    self.neuron_summary_seg_inds.append(self.work_items[0][seg]['index_window'])
                    if self.neuron_summary_seg_inds[-1][1] - self.neuron_summary_seg_inds[-1][0] <= self.merge_test_overlap_indices:
                        raise ValueError("Number of merge test overlap indices must be less than the total segment size!")
                if len(self.sort_data[chan][seg][0]) == 0:
                    continue # No spikes in this segment
                spike_order = np.argsort(self.sort_data[chan][seg][0], kind='stable')
                self.sort_data[chan][seg][0] = self.sort_data[chan][seg][0][spike_order]
                self.sort_data[chan][seg][1] = self.sort_data[chan][seg][1][spike_order]
                self.sort_data[chan][seg][2] = self.sort_data[chan][seg][2][spike_order, :]
                self.sort_data[chan][seg][3] = self.sort_data[chan][seg][3][spike_order]

    def get_snr(self, chan, seg, full_template):
        # Get SNR on the main channel
        background_noise_std = self.work_items[chan][seg]['thresholds'][self.work_items[chan][seg]['chan_neighbor_ind']] / self.sort_info['sigma']
        main_win = [self.sort_info['n_samples_per_chan'] * self.work_items[chan][seg]['chan_neighbor_ind'],
                    self.sort_info['n_samples_per_chan'] * (self.work_items[chan][seg]['chan_neighbor_ind'] + 1)]
        main_template = full_template[main_win[0]:main_win[1]]
        range = np.amax(main_template) - np.amin(main_template)
        return range / (3 * background_noise_std)

    def delete_label(self, chan, seg, label):
        """ Remove this unit corresponding to label from current segment. """
        keep_indices = self.sort_data[chan][seg][1] != label
        self.sort_data[chan][seg][0] = self.sort_data[chan][seg][0][keep_indices]
        self.sort_data[chan][seg][1] = self.sort_data[chan][seg][1][keep_indices]
        self.sort_data[chan][seg][2] = self.sort_data[chan][seg][2][keep_indices, :]
        self.sort_data[chan][seg][3] = self.sort_data[chan][seg][3][keep_indices]
        return keep_indices

    def get_isi_violation_rate(self, chan, seg, label):
        """ Compute the spiking activity that occurs during the ISI violation
        period, absolute_refractory_period, relative to the total number of
        spikes within the given segment, 'seg' and unit label 'label'. """
        select_unit = self.sort_data[chan][seg][1] == label
        if ~np.any(select_unit):
            # There are no spikes that match this label
            raise ValueError("There are no spikes for label", label, ".")
        unit_spikes = self.sort_data[chan][seg][0][select_unit]
        main_win = [self.sort_info['n_samples_per_chan'] * self.work_items[chan][seg]['chan_neighbor_ind'],
                    self.sort_info['n_samples_per_chan'] * (self.work_items[chan][seg]['chan_neighbor_ind'] + 1)]
        duplicate_tol_inds = calc_spike_width(
                                self.sort_data[chan][seg][2][select_unit][:, main_win[0]:main_win[1]],
                                self.sort_info['clip_width'], self.sort_info['sampling_rate'])
        duplicate_tol_inds += self.duplicate_tol_inds
        refractory_adjustment = duplicate_tol_inds / self.sort_info['sampling_rate']
        if self.absolute_refractory_period - refractory_adjustment <= 0:
            print("duplicate_tol_inds encompasses absolute_refractory_period. duplicate tolerence enforced at", self.duplicate_tol_inds)
            duplicate_tol_inds = self.duplicate_tol_inds
            refractory_adjustment = 0
        index_isi = np.diff(unit_spikes)
        num_isi_violations = np.count_nonzero(
            index_isi / self.sort_info['sampling_rate']
            < self.absolute_refractory_period)
        n_duplicates = np.count_nonzero(index_isi <= duplicate_tol_inds)
        # Remove duplicate spikes from this computation and adjust the number
        # of spikes and time window accordingly
        num_isi_violations -= n_duplicates
        isi_violation_rate = num_isi_violations \
                             * (1.0 / (self.absolute_refractory_period - refractory_adjustment))\
                             / (np.count_nonzero(select_unit) - n_duplicates)
        return isi_violation_rate

    def get_mean_firing_rate(self, chan, seg, label):
        """ Compute mean firing rate for the input channel, segment, and label. """
        select_unit = self.sort_data[chan][seg][1] == label
        first_index = next((idx[0] for idx, val in np.ndenumerate(select_unit) if val), [None])
        if first_index is None:
            # There are no spikes that match this label
            raise ValueError("There are no spikes for label", label, ".")
        last_index = next((select_unit.shape[0] - idx[0] - 1 for idx, val in np.ndenumerate(select_unit[::-1]) if val), None)
        if self.sort_data[chan][seg][0][first_index] == self.sort_data[chan][seg][0][last_index]:
            return 0. # Only one spike
        mean_rate = self.sort_info['sampling_rate'] * np.count_nonzero(select_unit)\
                    / (self.sort_data[chan][seg][0][last_index] - self.sort_data[chan][seg][0][first_index])
        return mean_rate

    def get_fraction_mua(self, chan, seg, label):
        """ Estimate the fraction of noise/multi-unit activity (MUA) using analysis
        of ISI violations. We do this by looking at the spiking activity that
        occurs during the ISI violation period. """
        isi_violation_rate = self.get_isi_violation_rate(chan, seg, label)
        mean_rate = self.get_mean_firing_rate(chan, seg, label)
        if mean_rate == 0.:
            return 0.
        else:
            return isi_violation_rate / mean_rate

    def get_fraction_mua_to_peak(self, chan, seg, label, check_window=0.5):
        """
        """
        select_unit = self.sort_data[chan][seg][1] == label
        if ~np.any(select_unit):
            # There are no spikes that match this label
            raise ValueError("There are no spikes for label", label, ".")
        unit_spikes = self.sort_data[chan][seg][0][select_unit]
        # duplicate_tol_inds, adjusted_num_isi_violations = calc_duplicate_tol_inds(unit_spikes,
        #                         self.sort_info['sampling_rate'],
        #                         self.absolute_refractory_period,
        #                         self.sort_info['clip_width'])
        main_win = [self.sort_info['n_samples_per_chan'] * self.work_items[chan][seg]['chan_neighbor_ind'],
                    self.sort_info['n_samples_per_chan'] * (self.work_items[chan][seg]['chan_neighbor_ind'] + 1)]
        duplicate_tol_inds = calc_spike_width(
                                self.sort_data[chan][seg][2][select_unit][:, main_win[0]:main_win[1]],
                                self.sort_info['clip_width'], self.sort_info['sampling_rate'])
        duplicate_tol_inds += self.duplicate_tol_inds
        all_isis = np.diff(unit_spikes)
        refractory_inds = int(round(self.absolute_refractory_period * self.sort_info['sampling_rate']))
        bin_width = refractory_inds - duplicate_tol_inds
        if bin_width <= 0:
            print("duplicate_tol_inds encompasses absolute_refractory_period. duplicate tolerence enforced at", self.duplicate_tol_inds)
            duplicate_tol_inds = self.duplicate_tol_inds
            bin_width = refractory_inds - duplicate_tol_inds
        check_inds = int(round(check_window * self.sort_info['sampling_rate']))
        bin_edges = np.arange(duplicate_tol_inds+1, check_inds + bin_width, bin_width)
        counts, xval = np.histogram(all_isis, bin_edges)
        isi_peak = np.amax(counts)
        num_isi_violations = counts[0]
        # num_isi_violations = np.count_nonzero(all_isis < refractory_inds)
        # n_duplicates = np.count_nonzero(all_isis <= self.sort_data[chan][seg][4])
        # num_isi_violations -= n_duplicates
        # num_isi_violations = adjusted_num_isi_violations
        if num_isi_violations < 0:
            num_isi_violations = 0
        if isi_peak == 0:
            isi_peak = max(num_isi_violations, 1.)
        fraction_mua_to_peak = num_isi_violations / isi_peak

        return fraction_mua_to_peak

    def delete_mua_units(self):
        for chan in range(0, self.n_chans):
            for seg in range(0, self.n_segments):
                for l in np.unique(self.sort_data[chan][seg][1]):
                    mua_ratio = self.get_fraction_mua_to_peak(chan, seg, l)
                    if mua_ratio > self.max_mua_ratio:
                        self.delete_label(chan, seg, l)

    def merge_test_two_units(self, clips_1, clips_2, p_cut, method='template_pca',
                             split_only=False, merge_only=False, curr_chan_inds=None):
        if self.sort_info['add_peak_valley'] and curr_chan_inds is None:
            raise ValueError("Must give curr_chan_inds if using peak valley.")
        clips = np.vstack((clips_1, clips_2))
        neuron_labels = np.ones(clips.shape[0], dtype=np.int64)
        neuron_labels[clips_1.shape[0]:] = 2
        if method.lower() == 'pca':
            scores = preprocessing.compute_pca(clips,
                        self.sort_info['check_components'],
                        self.sort_info['max_components'],
                        add_peak_valley=self.sort_info['add_peak_valley'],
                        curr_chan_inds=curr_chan_inds)
        elif method.lower() == 'template_pca':
            scores = preprocessing.compute_template_pca(clips, neuron_labels,
                        curr_chan_inds, self.sort_info['check_components'],
                        self.sort_info['max_components'],
                        add_peak_valley=self.sort_info['add_peak_valley'])
        elif method.lower() == 'projection':
            # Projection onto templates, weighted by number of spikes
            t1 = np.mean(clips_1, axis=0) * (clips_1.shape[0] / clips.shape[0])
            t2 = np.mean(clips_2, axis=0) * (clips_2.shape[0] / clips.shape[0])
            scores = clips @ np.vstack((t1, t2)).T
        else:
            raise ValueError("Unknown method", method, "for scores. Must use 'pca' or 'projection'.")
        neuron_labels = merge_clusters(scores, neuron_labels,
                            split_only=split_only, merge_only=merge_only,
                            p_value_cut_thresh=p_cut, flip_labels=False)

        label_is_1 = neuron_labels == 1
        label_is_2 = neuron_labels == 2
        if np.all(label_is_1) or np.all(label_is_2):
            clips_merged = True
        else:
            clips_merged = False
        neuron_labels_1 = neuron_labels[0:clips_1.shape[0]]
        neuron_labels_2 = neuron_labels[clips_1.shape[0]:]
        return clips_merged, neuron_labels_1, neuron_labels_2

    def check_missed_alignment_merge(self, chan, main_seg, leftover_seg,
                main_labels, leftover_labels, leftover_workspace, curr_chan_inds):
        """ Alternative to sort_cython.identify_clusters_to_compare that simply
        chooses the most similar template after shifting to optimal alignment.
        Intended as helper function so that neurons do not fail to stitch in the
        event their alignment changes between segments. """
        # Must distinguish between what we want to loop over again (or not) and
        # the actual values we want to output
        loop_main = [x for x in main_labels]
        loop_leftover = [x for x in leftover_labels]
        # Keep trying as long as there are leftovers to try
        while (len(loop_main) > 0) and (len(loop_leftover) > 0):
            # Need to find the best match between main and leftovers
            # NOTE: this is not normalized and so will be biased toward matching
            # larger spikes. I think this is a good thing since SNR should also
            # be highest for these units, do them first.
            best_corr = -np.inf
            best_shift = 0
            main_remove = set()
            leftover_remove = set()
            for ml in loop_main:
                # Choose main seg template
                ml_select = self.sort_data[chan][main_seg][1] == ml
                clips_1 = self.sort_data[chan][main_seg][2][ml_select, :]
                if main_seg != leftover_seg:
                    start_edge = self.neuron_summary_seg_inds[main_seg][1] - self.merge_test_overlap_indices
                    c1_start = next((idx[0] for idx, val in np.ndenumerate(
                                self.sort_data[chan][main_seg][0][ml_select])
                                if val >= start_edge), None)
                    if c1_start is None:
                        main_remove.add(ml)
                        continue
                    if c1_start == clips_1.shape[0]:
                        main_remove.add(ml)
                        continue
                    clips_1 = clips_1[c1_start:, :]
                    clips_1 = clips_1[max(clips_1.shape[0]-self.n_max_merge_test_clips, 0):, :]
                main_template = np.mean(clips_1, axis=0)
                for ll in loop_leftover:
                    # Find leftover template
                    if ll == ml:
                        continue
                    ll_select = leftover_workspace == ll
                    clips_2 = self.sort_data[chan][leftover_seg][2][ll_select, :]
                    if main_seg != leftover_seg:
                        stop_edge = self.neuron_summary_seg_inds[main_seg][1]
                        c2_start = next((idx[0] for idx, val in np.ndenumerate(
                                    self.sort_data[chan][leftover_seg][0][ll_select])
                                    if val >= start_edge), None)
                        c2_stop = next((idx[0] for idx, val in np.ndenumerate(
                                    self.sort_data[chan][leftover_seg][0][ll_select])
                                    if val > stop_edge), None)
                        if c2_start is None:
                            leftover_remove.add(ll)
                            continue
                        if c2_start == c2_stop:
                            leftover_remove.add(ll)
                            continue
                        clips_2 = clips_2[c2_start:c2_stop, :]
                        clips_2 = clips_2[:min(self.n_max_merge_test_clips, clips_2.shape[0]), :]
                    leftover_template = np.mean(clips_2, axis=0)
                    cross_corr = np.correlate(main_template[curr_chan_inds],
                                        leftover_template[curr_chan_inds],
                                        mode='full')
                    max_corr_ind = np.argmax(cross_corr)
                    if cross_corr[max_corr_ind] > best_corr:
                        best_corr = cross_corr[max_corr_ind]
                        best_shift = max_corr_ind - cross_corr.shape[0]//2
                        # if main_seg != leftover_seg:
                        #     best_ml_clips = clips_1[:, curr_chan_inds]
                        #     best_ll_clips = clips_2[:, curr_chan_inds]
                        # else:
                        #     best_ml_clips = clips_1
                        #     best_ll_clips = clips_2
                        best_ml_clips = clips_1
                        best_ll_clips = clips_2
                        best_ml_select = ml_select
                        best_ll_select = ll_select
                        chosen_ml = ml
                        chosen_ll = ll
            for mr in main_remove:
                loop_main.remove(mr)
            for lr in leftover_remove:
                loop_leftover.remove(lr)
            if np.isinf(best_corr):
                # Never found a match to reset the best ml/ll so we are done
                break
            # Align and truncate clips for best match pair
            if best_shift > 0:
                best_ml_clips = best_ml_clips[:, best_shift:]
                best_ll_clips = best_ll_clips[:, :-1*best_shift]
            elif best_shift < 0:
                best_ml_clips = best_ml_clips[:, :best_shift]
                best_ll_clips = best_ll_clips[:, -1*best_shift:]
            else:
                # No need to shift
                # I think this can still happen if perhaps there were duplicates
                # and so this nearest pair wasn't checked before?
                pass
            # Check if the main merges with its best aligned leftover
            is_merged, _, _ = self.merge_test_two_units(
                    best_ml_clips, best_ll_clips,
                    self.sort_info['p_value_cut_thresh'],
                    method='template_pca', merge_only=True,
                    curr_chan_inds=curr_chan_inds)

            if self.verbose: print("In 'check_missed_alignment_merge' Item", self.work_items[chan][main_seg]['ID'], "on chan", chan, "seg", main_seg, "merged", is_merged, "for labels", chosen_ml, chosen_ll)
            # print("Should start plotting best chosen_ml and chosen_ll, which merged", is_merged)
            # plt.plot(np.mean(best_ml_clips, axis=0))
            # plt.plot(np.mean(best_ll_clips, axis=0))
            # plt.show()
            if is_merged:
                # Update actual next segment label data with same labels
                # used in main_seg
                if chosen_ml < chosen_ll:
                    self.sort_data[chan][leftover_seg][1][best_ll_select] = chosen_ml
                    # This leftover is used up
                    leftover_labels.remove(chosen_ll)
                    loop_leftover.remove(chosen_ll)
                else:
                    # I think this shouldn't happen since labels should have
                    # been ordered on input
                    print("!!! Relabelling main_labels based on leftovers !!!")
                    print("main labels were", loop_main)
                    print("leftovers were", leftover_labels)
                    self.sort_data[chan][main_seg][1][best_ml_select] = chosen_ll
                    # This leftover is used up
                    leftover_labels.remove(chosen_ml)
                    loop_leftover.remove(chosen_ml)
                if chosen_ll in main_labels:
                    main_labels.remove(chosen_ll)
                    loop_main.remove(chosen_ll)
            else:
                # This main label had its pick of litter and failed so its done
                loop_main.remove(chosen_ml)
                if chosen_ml in loop_leftover:
                    loop_leftover.remove(chosen_ml)

    def stitch_segments(self):
        """
        Returns
        -------
        Returns None but reassigns class attributes as follows:
        self.sort_data[1] : np.int64
            Reassigns the labels across segments within each channel so that
            they have the same meaning.
        """
        if self.is_stitched:
            print("Neurons are already stitched. Repeated stitching can change results.")
            print("Skipped stitching")
            return
        self.is_stitched = True
        # Stitch each channel separately
        for chan in range(0, self.n_chans):
            print("Start stitching channel", chan)
            jump_to_end = False
            if len(self.sort_data[chan]) <= 1:
                # Need at least 2 segments to stitch
                jump_to_end = True
            # Start with the current labeling scheme in first segment,
            # which is assumed to be ordered from 0-N (as output by sorter)
            # Find first segment with labels and start there
            start_seg = 0
            while start_seg < len(self.sort_data[chan]):
                if len(self.sort_data[chan][start_seg][1]) == 0:
                    start_seg += 1
                    continue
                real_labels = np.unique(self.sort_data[chan][start_seg][1]).tolist()
                next_real_label = max(real_labels) + 1
                if len(real_labels) > 0:
                    break
                start_seg += 1
            start_new_seg = False
            if start_seg >= len(self.sort_data[chan])-1:
                # Need at least 2 remaining segments to stitch
                jump_to_end = True
            else:
                main_win = [self.sort_info['n_samples_per_chan'] * self.work_items[chan][start_seg]['chan_neighbor_ind'],
                            self.sort_info['n_samples_per_chan'] * (self.work_items[chan][start_seg]['chan_neighbor_ind'] + 1)]
                curr_chan_inds = np.arange(main_win[0], main_win[1], dtype=np.int64)

                split_memory_dicts = [{} for x in range(0, self.n_segments)]
            # Go through each segment as the "current segment" and set the labels
            # in the next segment according to the scheme in current
            for curr_seg in range(start_seg, len(self.sort_data[chan])-1):
                if jump_to_end:
                    break
                next_seg = curr_seg + 1
                if len(self.sort_data[chan][curr_seg][1]) == 0:
                    # curr_seg is now failed next seg of last iteration
                    start_new_seg = True
                    if self.verbose: print("skipping seg", curr_seg, "with no spikes")
                    continue
                if start_new_seg:
                    # Map all units in this segment to new real labels
                    self.sort_data[chan][curr_seg][1] += next_real_label # Keep out of range
                    for nl in np.unique(self.sort_data[chan][curr_seg][1]):
                        real_labels.append(nl)
                        if self.verbose: print("In NEXT SEG NEW (531) added real label", nl, chan, curr_seg)
                        next_real_label += 1
                    # start_new_seg = False
                if len(self.sort_data[chan][next_seg][1]) == 0:
                    # No units sorted in NEXT segment so start fresh next segment
                    start_new_seg = True
                    if self.verbose: print("skipping_seg", curr_seg, "because NEXT seg has no spikes")
                    for curr_l in np.unique(self.sort_data[chan][curr_seg][1]):
                        mua_ratio = self.get_fraction_mua_to_peak(chan,
                                            curr_seg, curr_l)
                        if mua_ratio > self.max_mua_ratio:
                            if self.verbose: print("Checking seg before new MUA (543) deleting at MUA ratio", mua_ratio, chan, curr_seg)
                            self.delete_label(chan, curr_seg, curr_l)
                            if curr_seg == 0:
                                if self.verbose: print("removing label 679", "label", curr_l, chan, curr_seg)
                                real_labels.remove(curr_l)
                            else:
                                if curr_l not in self.sort_data[chan][curr_seg-1][1]:
                                    # Remove from real labels if its not in previous
                                    if self.verbose: print("removing label 684", "label", curr_l,  chan, curr_seg)
                                    real_labels.remove(curr_l)
                    continue

                # Make 'fake_labels' for next segment that do not overlap with
                # the current segment and make a work space for the next
                # segment labels so we can compare curr and next without
                # losing track of original labels
                fake_labels = np.copy(np.unique(self.sort_data[chan][next_seg][1]))
                fake_labels += next_real_label # Keep these out of range of the real labels
                fake_labels = fake_labels.tolist()
                next_label_workspace = np.copy(self.sort_data[chan][next_seg][1])
                next_label_workspace += next_real_label

                curr_spike_start = 0 #max(self.sort_data[chan][curr_seg][0].shape[0] - 100, 0)
                next_spike_stop = self.sort_data[chan][next_seg][0].shape[0] #min(self.sort_data[chan][next_seg][0].shape[0], 100)
                # Find templates for units in each segment
                curr_templates, curr_labels = segment.calculate_templates(
                                        self.sort_data[chan][curr_seg][2][curr_spike_start:, :],
                                        self.sort_data[chan][curr_seg][1][curr_spike_start:])
                next_templates, next_labels = segment.calculate_templates(
                                        self.sort_data[chan][next_seg][2][:next_spike_stop, :],
                                        next_label_workspace[:next_spike_stop])
                # Find all pairs of templates that are mutually closest
                minimum_distance_pairs = sort_cython.identify_clusters_to_compare(
                                np.vstack(curr_templates + next_templates),
                                np.hstack((curr_labels, next_labels)), [])

                # Merge test all mutually closest clusters and track any labels
                # in the next segment (fake_labels) that do not find a match.
                # These are assigned a new real label.
                leftover_labels = [x for x in fake_labels]
                main_labels = [x for x in curr_labels]
                for c1, c2 in minimum_distance_pairs:
                    if (c1 in real_labels) and (c2 in fake_labels):
                        r_l = c1
                        f_l = c2
                    elif (c2 in real_labels) and (c1 in fake_labels):
                        r_l = c2
                        f_l = c1
                    else:
                        # Require one from each segment to compare
                        # In this condition units that were split from each
                        # other within a segment are actually closer to each
                        # other than they are to anything in the other segment.
                        # This suggests we should not merge these as one of
                        # them is likely garbage.
                        continue
                    # Choose current seg clips based on real labels
                    real_select = self.sort_data[chan][curr_seg][1] == r_l
                    clips_1 = self.sort_data[chan][curr_seg][2][real_select, :]
                    start_edge = self.neuron_summary_seg_inds[curr_seg][1] - self.merge_test_overlap_indices
                    c1_start = next((idx[0] for idx, val in np.ndenumerate(
                                self.sort_data[chan][curr_seg][0][real_select])
                                if val >= start_edge), None)
                    if c1_start is None:
                        continue
                    if c1_start == clips_1.shape[0]:
                        continue
                    clips_1 = clips_1[c1_start:, :]
                    clips_1 = clips_1[max(clips_1.shape[0]-self.n_max_merge_test_clips, 0):, :]
                    # Choose next seg clips based on original fake label workspace
                    fake_select = next_label_workspace == f_l
                    clips_2 = self.sort_data[chan][next_seg][2][fake_select, :]
                    stop_edge = self.neuron_summary_seg_inds[curr_seg][1]
                    c2_start = next((idx[0] for idx, val in np.ndenumerate(
                                self.sort_data[chan][next_seg][0][fake_select])
                                if val >= start_edge), None)
                    c2_stop = next((idx[0] for idx, val in np.ndenumerate(
                                self.sort_data[chan][next_seg][0][fake_select])
                                if val > stop_edge), None)
                    if c2_start is None:
                        continue
                    if c2_start == c2_stop:
                        continue
                    clips_2 = clips_2[c2_start:c2_stop, :]
                    clips_2 = clips_2[:min(self.n_max_merge_test_clips, clips_2.shape[0]), :]
                    is_merged, _, _ = self.merge_test_two_units(
                            clips_1, clips_2, self.sort_info['p_value_cut_thresh'],
                            method='template_pca', merge_only=True,
                            curr_chan_inds=curr_chan_inds)

                    if self.verbose: print("Item", self.work_items[chan][curr_seg]['ID'], "on chan", chan, "seg", curr_seg, "merged", is_merged, "for labels", r_l, f_l)

                    if is_merged:
                        # Update actual next segment label data with same labels
                        # used in curr_seg
                        self.sort_data[chan][next_seg][1][fake_select] = r_l
                        leftover_labels.remove(f_l)
                        # main_labels.remove(r_l)

                if curr_seg == 0 or start_new_seg:
                    pseudo_leftovers = [x for x in main_labels]
                    self.check_missed_alignment_merge(chan, curr_seg, curr_seg,
                                main_labels, pseudo_leftovers,
                                self.sort_data[chan][curr_seg][1],
                                curr_chan_inds)
                # Make sure none of the main labels is terminating due to a misalignment
                if len(leftover_labels) > 0:
                    self.check_missed_alignment_merge(chan, curr_seg, next_seg,
                                main_labels, leftover_labels, next_label_workspace,
                                curr_chan_inds)
                # Assign units in next segment that do not match any in the
                # current segment a new real label
                for ll in leftover_labels:
                    ll_select = next_label_workspace == ll
                    self.sort_data[chan][next_seg][1][ll_select] = next_real_label
                    real_labels.append(next_real_label)
                    if self.verbose: print("In leftover labels (612) added real label", next_real_label, chan, curr_seg)
                    next_real_label += 1
                # Make sure newly stitched segment labels are separate from
                # their nearest template match. This should help in instances
                # where one of the stitched units is mixed in one segment, but
                # in the other segment the units are separable (due to, for
                # instance, drift). This can help maintain segment continuity if
                # a unit exists in both segments but only has enough spikes/
                # separation to be sorted in one of the segments.
                # Do this by considering curr and next seg combined
                # NOTE: This could also be done by considering ALL previous
                # segments combined or once at the end over all data combined
                joint_clips = np.vstack((self.sort_data[chan][curr_seg][2],
                                         self.sort_data[chan][next_seg][2]))
                joint_labels = np.hstack((self.sort_data[chan][curr_seg][1],
                                          self.sort_data[chan][next_seg][1]))
                joint_templates, temp_labels = segment.calculate_templates(
                                        joint_clips, joint_labels)

                # Find all pairs of templates that are mutually closest
                minimum_distance_pairs = sort_cython.identify_clusters_to_compare(
                                np.vstack(joint_templates), temp_labels, [])
                if self.verbose: print("Doing split on joint min pairs", minimum_distance_pairs)
                tmp_reassign = np.zeros_like(joint_labels)
                # Perform a split only between all minimum distance pairs
                for c1, c2 in minimum_distance_pairs:
                    c1_select = joint_labels == c1
                    clips_1 = joint_clips[c1_select, :]
                    c2_select = joint_labels == c2
                    clips_2 = joint_clips[c2_select, :]
                    ismerged, labels_1, labels_2 = self.merge_test_two_units(
                            clips_1, clips_2, self.sort_info['p_value_cut_thresh'],
                            method='template_pca', split_only=True,
                            curr_chan_inds=curr_chan_inds)
                    if ismerged: # This can happen if the split cutpoint forces
                        continue # a merge so check and skip

                    # Reassign spikes in c1 that split into c2
                    # The merge test was done on joint clips and labels, so
                    # we have to figure out where their indices all came from
                    tmp_reassign[:] = 0
                    tmp_reassign[c1_select] = labels_1
                    tmp_reassign[c2_select] = labels_2
                    curr_reassign_index_to_c1 = tmp_reassign[0:self.sort_data[chan][curr_seg][1].size] == 1
                    curr_original_index_to_c1 = self.sort_data[chan][curr_seg][1][curr_reassign_index_to_c1]
                    self.sort_data[chan][curr_seg][1][curr_reassign_index_to_c1] = c1
                    curr_reassign_index_to_c2 = tmp_reassign[0:self.sort_data[chan][curr_seg][1].size] == 2
                    curr_original_index_to_c2 = self.sort_data[chan][curr_seg][1][curr_reassign_index_to_c2]
                    self.sort_data[chan][curr_seg][1][curr_reassign_index_to_c2] = c2

                    # Repeat for assignments in next_seg
                    next_reassign_index_to_c1 = tmp_reassign[self.sort_data[chan][curr_seg][1].size:] == 1
                    next_original_index_to_c1 = self.sort_data[chan][next_seg][1][next_reassign_index_to_c1]
                    self.sort_data[chan][next_seg][1][next_reassign_index_to_c1] = c1
                    next_reassign_index_to_c2 = tmp_reassign[self.sort_data[chan][curr_seg][1].size:] == 2
                    next_original_index_to_c2 = self.sort_data[chan][next_seg][1][next_reassign_index_to_c2]
                    self.sort_data[chan][next_seg][1][next_reassign_index_to_c2] = c2

                    # Check if split was a good idea and undo it if not. Basically,
                    # if a mixture is left behind after the split, we probably
                    # do not want to reassign labels on the basis of the comparison
                    # with the MUA mixture, so undo the steps above. Otherwise,
                    # either there were no mixtures or the split removed them
                    # so we carry on.
                    undo_split = False
                    for curr_l in [c1, c2]:
                        mua_ratio = 0.
                        if curr_l in self.sort_data[chan][curr_seg][1]:
                            mua_ratio = self.get_fraction_mua_to_peak(chan, curr_seg, curr_l)
                        if mua_ratio > self.max_mua_ratio:
                            undo_split = True
                            break
                        if curr_l in self.sort_data[chan][next_seg][1]:
                            mua_ratio = self.get_fraction_mua_to_peak(chan, next_seg, curr_l)
                        if mua_ratio > self.max_mua_ratio:
                            undo_split = True
                            break
                    if undo_split:
                        if self.verbose: print("undoing split between", c1, c2)
                        if 2 in labels_1:
                            self.sort_data[chan][curr_seg][1][curr_reassign_index_to_c2] = curr_original_index_to_c2
                            self.sort_data[chan][next_seg][1][next_reassign_index_to_c2] = next_original_index_to_c2
                        if 1 in labels_2:
                            self.sort_data[chan][curr_seg][1][curr_reassign_index_to_c1] = curr_original_index_to_c1
                            self.sort_data[chan][next_seg][1][next_reassign_index_to_c1] = next_original_index_to_c1
                    else:
                        split_memory_dicts[curr_seg][c1] = [curr_reassign_index_to_c1, curr_original_index_to_c1]
                        split_memory_dicts[curr_seg][c2] = [curr_reassign_index_to_c2, curr_original_index_to_c2]
                # Finally, check if the above split was unable to salvage any
                # mixtures in the current segment. If it wasn't, delete that
                # unit from the current segment and relabel any units in the
                # next segment that matched with it
                for curr_l in np.unique(self.sort_data[chan][curr_seg][1]):
                    mua_ratio = self.get_fraction_mua_to_peak(chan, curr_seg, curr_l)
                    if mua_ratio > self.max_mua_ratio:
                        # Remove this unit from current segment
                        if self.verbose: print("Deleting (1056) label", curr_l, "at MUA ratio", mua_ratio, "for chan", chan, "seg", curr_seg)
                        keep_indices = self.delete_label(chan, curr_seg, curr_l)

                        # Also need to remove from memory dict
                        for key_label in split_memory_dicts[curr_seg].keys():
                            split_memory_dicts[curr_seg][key_label][0] = \
                                split_memory_dicts[curr_seg][key_label][0][keep_indices]

                        if curr_seg == 0:
                            if self.verbose: print("Deleting (1065) label", "label", curr_l, chan, curr_seg)
                            real_labels.remove(curr_l)
                        else:
                            if curr_l in split_memory_dicts[curr_seg - 1].keys():
                                # This is in previous, so undo any effects split
                                # could have had
                                self.sort_data[chan][curr_seg - 1][1][split_memory_dicts[curr_seg - 1][curr_l][0]] = \
                                        split_memory_dicts[curr_seg - 1][curr_l][1]
                            if curr_l not in self.sort_data[chan][curr_seg - 1][1]:
                                # Remove from real labels if its not in previous
                                if self.verbose: print("Deleting (1075) label", "label", curr_l, chan, curr_seg)
                                real_labels.remove(curr_l)
                            # NOTE: I think split_memory_dicts[curr_seg - 1] can
                            # be deleted at this point to save memory

                        # Assign any units in next segment that stitched to this
                        # bad one, if any, a new label.
                        select_next_curr_l = self.sort_data[chan][next_seg][1] == curr_l
                        if any(select_next_curr_l):
                            self.sort_data[chan][next_seg][1][select_next_curr_l] = next_real_label
                            real_labels.append(next_real_label)
                            if self.verbose: print("In leftover after deletion (1086) added real label", next_real_label, chan, curr_seg)
                            next_real_label += 1
                # If we made it here then we are not starting a new seg
                start_new_seg = False
                if self.verbose: print("!!!REAL LABELS ARE !!!", real_labels, np.unique(self.sort_data[chan][curr_seg][1]), np.unique(self.sort_data[chan][next_seg][1]))
                if curr_seg > 0:
                    if self.verbose: print("Previous seg...", np.unique(self.sort_data[chan][curr_seg-1][1]))

            # It is possible to leave loop without checking last seg in the
            # event it is a new seg
            if start_new_seg and len(self.sort_data[chan][-1][0]) > 0:
                # Map all units in this segment to new real labels
                # Seg labels start at zero, so just add next_real_label. This
                # is last segment for this channel so no need to increment
                self.sort_data[chan][-1][1] += next_real_label
                real_labels.extend((np.unique(self.sort_data[chan][-1][1]) + next_real_label).tolist())
                if self.verbose: print("Last seg is new (747) added real labels", np.unique(self.sort_data[chan][-1][1]) + next_real_label, chan, curr_seg)
            # Check for MUA in the last segment as we did for the others
            curr_seg = len(self.sort_data[chan]) - 1
            if curr_seg < 0:
                continue
            if len(self.sort_data[chan][curr_seg][0]) == 0:
                continue
            for curr_l in np.unique(self.sort_data[chan][curr_seg][1]):
                mua_ratio = self.get_fraction_mua_to_peak(chan, curr_seg, curr_l)
                if mua_ratio > self.max_mua_ratio:
                    if self.verbose: print("Checking last seg MUA (757) deleting at MUA ratio", mua_ratio, chan, curr_seg)
                    self.delete_label(chan, curr_seg, curr_l)
                    if curr_seg == 0:
                        real_labels.remove(curr_l)
                    else:
                        if not start_new_seg and curr_l not in self.sort_data[chan][curr_seg-1][1]:
                            # Remove from real labels if its not in previous
                            real_labels.remove(curr_l)
            if self.verbose: print("!!!REAL LABELS ARE !!!", real_labels)

    def summarize_neurons_by_seg(self):
        """
        """
        self.neuron_summary_by_seg = [[] for x in range(0, self.n_segments)]
        for seg in range(0, self.n_segments):
            for chan in range(0, self.n_chans):
                for ind, neuron_label in enumerate(np.unique(self.sort_data[chan][seg][1])):
                    neuron = {}
                    neuron['summary_type'] = 'single_segment'
                    neuron["channel"] = self.work_items[chan][seg]['channel']
                    neuron['neighbors'] = self.work_items[chan][seg]['neighbors']
                    neuron['chan_neighbor_ind'] = self.work_items[chan][seg]['chan_neighbor_ind']
                    neuron['segment'] = self.work_items[chan][seg]['seg_number']
                    assert neuron['segment'] == seg, "Somethings messed up here?"
                    assert neuron['channel'] == chan

                    select_label = self.sort_data[chan][seg][1] == neuron_label
                    neuron["spike_indices"] = self.sort_data[chan][seg][0][select_label]
                    neuron['waveforms'] = self.sort_data[chan][seg][2][select_label, :]
                    neuron["new_spike_bool"] = self.sort_data[chan][seg][3][select_label]

                    # NOTE: This still needs to be done even though segments
                    # were ordered because of overlap!
                    # Ensure spike times are ordered. Must use 'stable' sort for
                    # output to be repeatable because overlapping segments and
                    # binary pursuit can return slightly different dupliate spikes
                    spike_order = np.argsort(neuron["spike_indices"], kind='stable')
                    neuron["spike_indices"] = neuron["spike_indices"][spike_order]
                    neuron['waveforms'] = neuron['waveforms'][spike_order, :]
                    neuron["new_spike_bool"] = neuron["new_spike_bool"][spike_order]
                    # Remove duplicates found in binary pursuit
                    keep_bool = remove_binary_pursuit_duplicates(neuron["spike_indices"],
                                    neuron["new_spike_bool"],
                                    tol_inds=self.sort_data[chan][seg][4])
                    neuron["spike_indices"] = neuron["spike_indices"][keep_bool]
                    neuron["new_spike_bool"] = neuron["new_spike_bool"][keep_bool]
                    neuron['waveforms'] = neuron['waveforms'][keep_bool, :]

                    # Remove any identical index duplicates (either from error or
                    # from combining overlapping segments), preferentially keeping
                    # the waveform best aligned to the template
                    neuron["template"] = np.mean(neuron['waveforms'], axis=0)
                    keep_bool = remove_spike_event_duplicates(neuron["spike_indices"],
                                    neuron['waveforms'], neuron["template"],
                                    tol_inds=self.sort_data[chan][seg][4])
                    neuron["spike_indices"] = neuron["spike_indices"][keep_bool]
                    neuron["new_spike_bool"] = neuron["new_spike_bool"][keep_bool]
                    neuron['waveforms'] = neuron['waveforms'][keep_bool, :]

                    # Recompute template and store output
                    neuron["template"] = np.mean(neuron['waveforms'], axis=0)
                    neuron['snr'] = self.get_snr(chan, seg, neuron["template"])
                    neuron['fraction_mua'] = self.get_fraction_mua_to_peak(chan, seg, neuron_label)
                    neuron['threshold'] = self.work_items[chan][seg]['thresholds'][self.work_items[chan][seg]['channel']]
                    neuron['main_win'] = [self.sort_info['n_samples_per_chan'] * neuron['chan_neighbor_ind'],
                                          self.sort_info['n_samples_per_chan'] * (neuron['chan_neighbor_ind'] + 1)]
                    self.neuron_summary_by_seg[seg].append(neuron)

    def remove_redundant_neurons(self, neurons, overlap_time=2.5e-4, overlap_ratio_threshold=2):
        """
        """
        # NOTE: Can't do this here and in the function that calls it...
        # if overlap_ratio_threshold >= self.last_overlap_ratio_threshold:
        #     print("Redundant neurons already removed at threshold >=", overlap_ratio_threshold. "Further attempts will have no effect.")
        #     print("Skipping remove_redundant_neurons.")
        #     return
        # else:
        #     self.last_overlap_ratio_threshold = overlap_ratio_threshold
        max_samples = int(round(overlap_time * self.sort_info['sampling_rate']))
        n_total_samples = 0

        # Create list of sets of excessive neuron overlap between all pairwise units
        violation_partners = [set() for x in range(0, len(neurons))]
        for n1_ind, n1 in enumerate(neurons):
            violation_partners[n1_ind].add(n1_ind)
            if n1['spike_indices'][-1] > n_total_samples:
                # Find the maximum number of samples over all neurons while we are here for use later
                n_total_samples = n1['spike_indices'][-1]
            for n2_ind in range(n1_ind+1, len(neurons)):
                n2 = neurons[n2_ind]
                if np.intersect1d(n1['neighbors'], n2['neighbors']).size == 0:
                    # Only count violations in neighborhood
                    continue
                violation_partners[n1_ind].add(n2_ind)
                violation_partners[n2_ind].add(n1_ind)
                # actual_overlaps = zero_symmetric_ccg(n1['spike_indices'], n2['spike_indices'], max_samples, max_samples)[0]
                # if actual_overlaps[1] > (actual_overlaps[0] + actual_overlaps[2]):
                #     # Mark as a violation if center of CCG is greater than sum of bins on either side of it
                #     violation_partners[n1_ind].add(n2_ind)
                #     violation_partners[n2_ind].add(n1_ind)

        overlap_ratio = np.zeros((len(neurons), len(neurons)))
        expected_ratio = np.zeros((len(neurons), len(neurons)))
        for neuron1_ind, neuron1 in enumerate(neurons):
            neuron1_spike_train = compute_spike_trains(neuron1['spike_indices'], max_samples, [0, n_total_samples])
            # Loop through all violators with neuron1
            # We already know these neurons are in each other's neighborhood
            # because violation_partners only includes neighbors
            for neuron2_ind in violation_partners[neuron1_ind]:
                if neuron2_ind <= neuron1_ind:
                    continue # Since our costs are symmetric, we only need to check indices greater than neuron1_ind
                neuron2 = neurons[neuron2_ind]
                if neuron1['channel'] == neuron2['channel']:
                    continue # If they are on the same channel, do nothing
                neuron2_spike_train = compute_spike_trains(neuron2['spike_indices'], max_samples, [0, n_total_samples])
                num_hits = np.count_nonzero(np.logical_and(neuron1_spike_train, neuron2_spike_train))
                neuron2_misses = np.count_nonzero(np.logical_and(neuron2_spike_train, ~neuron1_spike_train))
                neuron1_misses = np.count_nonzero(np.logical_and(neuron1_spike_train, ~neuron2_spike_train))

                num_misses = min(neuron1_misses, neuron2_misses)
                overlap_ratio[neuron1_ind, neuron2_ind] = num_hits / (num_hits + num_misses)
                overlap_ratio[neuron2_ind, neuron1_ind] = overlap_ratio[neuron1_ind, neuron2_ind]

                expected_hits = calculate_expected_overlap(neuron1, neuron2,
                                    overlap_time, self.sort_info['sampling_rate'])
                # NOTE: Should this be expected hits minus remaining number of spikes?
                expected_misses = min(neuron1['spike_indices'].shape[0], neuron2['spike_indices'].shape[0]) - expected_hits
                expected_ratio[neuron1_ind, neuron2_ind] = expected_hits / (expected_hits + expected_misses)
                expected_ratio[neuron2_ind, neuron1_ind] = expected_ratio[neuron1_ind, neuron2_ind]

        neurons_remaining_indices = [x for x in range(0, len(neurons))]
        neurons_to_remove = []
        max_accepted = 0.
        max_expected = 0.
        while True:
            # Look for our next best pair
            best_ratio = -np.inf
            best_expected = -np.inf
            best_pair = []
            for i in range(0, len(neurons_remaining_indices)):
                for j in range(i+1, len(neurons_remaining_indices)):
                    neuron_1_index = neurons_remaining_indices[i]
                    neuron_2_index = neurons_remaining_indices[j]
                    if (overlap_ratio[neuron_1_index, neuron_2_index] <
                        overlap_ratio_threshold * expected_ratio[neuron_1_index, neuron_2_index]):
                        # Overlap not high enough to merit deletion of one
                        # But track our proximity to input threshold
                        if overlap_ratio[neuron_1_index, neuron_2_index] > max_accepted:
                            max_accepted = overlap_ratio[neuron_1_index, neuron_2_index]
                            max_expected = overlap_ratio_threshold * expected_ratio[neuron_1_index, neuron_2_index]
                        continue
                    if overlap_ratio[neuron_1_index, neuron_2_index] > best_ratio:
                        best_ratio = overlap_ratio[neuron_1_index, neuron_2_index]
                        best_pair = [neuron_1_index, neuron_2_index]
                        best_expected = expected_ratio[neuron_1_index, neuron_2_index]

            if len(best_pair) == 0 or best_ratio == 0:
                # No more pairs exceed ratio threshold
                print("Maximum accepted ratio was", max_accepted, "at expected threshold", max_expected)
                break
            # if best_ratio <= overlap_ratio_threshold * best_expected:
            #     print("Stopped overlaps at ratio", best_ratio, "versus threshold", overlap_ratio_threshold * best_expected)
            #     break

            # We now need to choose one of the pair to delete.
            neuron_1 = neurons[best_pair[0]]
            neuron_2 = neurons[best_pair[1]]
            delete_1 = False
            delete_2 = False
            """First doing the MUA and spike number checks because at this point
            the stitch segments function has deleted anything with MUA over the
            input max_mua_ratio. We can then fall back to SNR, since SNR does
            not always correspond to isolation quality, specifically in the case
            where other neurons are present on the same channel. Conversely,
            low MUA can indicate good isolation, or perhaps that the unit has a
            very small number of spikes. So we first consider MUA and spike
            count jointly before deferring to SNR. """
            if neuron_1['fraction_mua'] < 1e-4 and neuron_2['fraction_mua'] < 1e-4:
                # Both MUA negligible so choose most spikes
                print("Both MUA negligible so choosing most spikes")
                if neuron_1['spike_indices'].shape[0] > neuron_2['spike_indices'].shape[0]:
                    delete_2 = True
                else:
                    delete_1 = True
            elif ((1-neuron_1['fraction_mua']) * neuron_1['spike_indices'].shape[0]
                   > 1.1*(1-neuron_2['fraction_mua']) * neuron_2['spike_indices'].shape[0]):
                # Neuron 1 has higher MUA weighted spikes
                print('Neuron 1 has higher MUA weighted spikes')
                print("MUA", neuron_1['fraction_mua'], neuron_2['fraction_mua'], "spikes", neuron_1['spike_indices'].shape[0], neuron_2['spike_indices'].shape[0])
                delete_2 = True
            elif ((1-neuron_2['fraction_mua']) * neuron_2['spike_indices'].shape[0]
                   > 1.1*(1-neuron_1['fraction_mua']) * neuron_1['spike_indices'].shape[0]):
                # Neuron 2 has higher MUA weighted spikes
                print('Neuron 2 has higher MUA weighted spikes')
                print("MUA", neuron_1['fraction_mua'], neuron_2['fraction_mua'], "spikes", neuron_1['spike_indices'].shape[0], neuron_2['spike_indices'].shape[0])
                delete_1 = True

            # Defer to choosing max SNR
            elif (neuron_1['snr'] > neuron_2['snr']):
                print("neuron 1 has higher SNR", neuron_1['snr'] , neuron_2['snr'])
                delete_2 = True
            else:
                delete_1 = True

            if delete_1:
                neurons_to_remove.append(best_pair[0])
                neurons_remaining_indices.remove(best_pair[0])
            if delete_2:
                neurons_to_remove.append(best_pair[1])
                neurons_remaining_indices.remove(best_pair[1])

        for n_ind in reversed(range(0, len(neurons))):
            if n_ind in neurons_to_remove:
                del neurons[n_ind]
        return neurons

    def remove_redundant_neurons_by_seg(self, overlap_time=2.5e-4, overlap_ratio_threshold=2):
        """
        """
        if overlap_ratio_threshold >= self.last_overlap_ratio_threshold:
            print("Redundant neurons already removed at threshold >=", overlap_ratio_threshold, "Further attempts will have no effect.")
            print("Skipping remove_redundant_neurons_by_seg.")
            return
        else:
            self.last_overlap_ratio_threshold = overlap_ratio_threshold
        for seg in range(0, self.n_segments):
            self.neuron_summary_by_seg[seg] = self.remove_redundant_neurons(
                    self.neuron_summary_by_seg[seg], overlap_time,
                    overlap_ratio_threshold)

    def stitch_neurons_across_channels(self):
        start_seg = 0
        while start_seg < self.n_segments:
            if len(self.neuron_summary_by_seg[start_seg]) == 0:
                start_seg += 1
                continue
            neurons = [[x] for x in self.neuron_summary_by_seg[start_seg]]
            break
        if start_seg >= self.n_segments-1:
            # Need at least 2 remaining neurons to stitch.
            if len(self.neuron_summary_by_seg[start_seg]) == 0:
                # No neurons with data found
                return [{}]
            # With this being the only neuron, we are done
            neurons = self.neuron_summary_by_seg[start_seg]
            return [neurons]

        for next_seg in range(start_seg+1, self.n_segments):
            if len(self.neuron_summary_by_seg[next_seg]) == 0:
                # Nothing to link
                continue
            # For each currently known neuron, remove the neuron it connects to
            # (if any) in the next segment and add it to its own list
            # This is done by following the link of the last unit in the list
            next_seg_inds = [x for x in range(0, len(self.neuron_summary_by_seg[next_seg]))]
            for n_list in neurons:
                link_index = n_list[-1]['next_seg_link']
                if link_index is None:
                    continue
                n_list.append(self.neuron_summary_by_seg[next_seg][link_index])
                next_seg_inds.remove(link_index)
            for new_neurons in next_seg_inds:
                # Start a new list in neurons for anything that didn't link above
                neurons.append([self.neuron_summary_by_seg[next_seg][new_neurons]])
        return neurons

    def join_neuron_dicts(self, unit_dicts_list):
        """
        """
        if len(unit_dicts_list) == 0:
            return {}
        combined_neuron = {}
        combined_neuron['summary_type'] = 'across_channels'
        # Since a neuron can exist over multiple channels, we need to discover
        # all the channels that are present and track which channel data
        # correspond to
        combined_neuron['channel'] = []
        combined_neuron['neighbors'] = {}
        combined_neuron['chan_neighbor_ind'] = {}
        combined_neuron['main_windows'] = {}
        n_total_spikes = 0
        n_peak = 0
        for x in unit_dicts_list:
            n_total_spikes += x['spike_indices'].shape[0]
            if x['channel'] not in combined_neuron['channel']:
                combined_neuron["channel"].append(x['channel'])
                combined_neuron['neighbors'][x['channel']] = x['neighbors']
                combined_neuron['chan_neighbor_ind'][x['channel']] = x['chan_neighbor_ind']
                combined_neuron['main_windows'][x['channel']] = x['main_win']
            if np.amax(x['template'][x['main_win'][0]:x['main_win'][1]]) \
                > np.amin(x['template'][x['main_win'][0]:x['main_win'][1]]):
                n_peak += 1

        if n_peak / len(unit_dicts_list) > 0.5:
            # Majority of templates have larger peak than valley
            align_peak = True
        else:
            align_peak = False
        waveform_clip_center = int(round(np.abs(self.sort_info['clip_width'][0] * self.sort_info['sampling_rate']))) + 1

        for unit in unit_dicts_list:
            n_unit_events = unit['spike_indices'].size
            if n_unit_events == 0:
                continue

        chan_to_ind_map = {}
        for ind, chan in enumerate(combined_neuron['channel']):
            chan_to_ind_map[chan] = ind

        channel_selector = []
        indices_by_unit = []
        waveforms_by_unit = []
        new_spike_bool_by_unit = []
        threshold_by_unit = []
        segment_by_unit = []
        snr_by_unit = []
        for unit in unit_dicts_list:
            n_unit_events = unit['spike_indices'].size
            if n_unit_events == 0:
                continue
            # First adjust all spike indices to where they would have been if
            # aligned to the specified peak or valley so that spike times are
            # all on equal footing when computing redundant spikes, MUA etc.
            if align_peak:
                shift = np.argmax(unit['template'][unit['main_win'][0]:unit['main_win'][1]]) - waveform_clip_center
            else:
                shift = np.argmin(unit['template'][unit['main_win'][0]:unit['main_win'][1]]) - waveform_clip_center

            indices_by_unit.append(unit['spike_indices'] + shift)
            waveforms_by_unit.append(unit['waveforms'])
            new_spike_bool_by_unit.append(unit['new_spike_bool'])
            # Make and append a bunch of book keeping numpy arrays
            channel_selector.append(np.full(n_unit_events, unit['channel']))
            threshold_by_unit.append(np.full(n_unit_events, unit['threshold']))
            segment_by_unit.append(np.full(n_unit_events, unit['segment']))
            snr_by_unit.append(np.full(n_unit_events, unit['snr']))

        # Now combine everything into one
        channel_selector = np.hstack(channel_selector)
        threshold_by_unit = np.hstack(threshold_by_unit)
        segment_by_unit = np.hstack(segment_by_unit)
        snr_by_unit = np.hstack(snr_by_unit)
        combined_neuron["spike_indices"] = np.hstack(indices_by_unit)
        combined_neuron['waveforms'] = np.vstack(waveforms_by_unit)
        combined_neuron["new_spike_bool"] = np.hstack(new_spike_bool_by_unit)

        # NOTE: This still needs to be done even though segments
        # were ordered because of overlap!
        # Ensure everything is ordered. Must use 'stable' sort for
        # output to be repeatable because overlapping segments and
        # binary pursuit can return slightly different dupliate spikes
        spike_order = np.argsort(combined_neuron["spike_indices"], kind='stable')
        combined_neuron["spike_indices"] = combined_neuron["spike_indices"][spike_order]
        combined_neuron['waveforms'] = combined_neuron['waveforms'][spike_order, :]
        combined_neuron["new_spike_bool"] = combined_neuron["new_spike_bool"][spike_order]
        channel_selector = channel_selector[spike_order]
        threshold_by_unit = threshold_by_unit[spike_order]
        segment_by_unit = segment_by_unit[spike_order]
        snr_by_unit = snr_by_unit[spike_order]
        # combined_neuron['duplicate_tol_inds'], _ = calc_duplicate_tol_inds(
        #                         combined_neuron["spike_indices"],
        #                         self.sort_info['sampling_rate'],
        #                         self.absolute_refractory_period,
        #                         self.sort_info['clip_width'])
        max_dup = 0
        combined_neuron['duplicate_tol_inds'] = self.duplicate_tol_inds
        for chan in combined_neuron['channel']:
            chan_select = channel_selector == chan
            main_win = combined_neuron['main_windows'][chan]
            duplicate_tol_inds = calc_spike_width(
                combined_neuron['waveforms'][chan_select, main_win[0]:main_win[1]],
                self.sort_info['clip_width'], self.sort_info['sampling_rate'])
            duplicate_tol_inds += self.duplicate_tol_inds
            if duplicate_tol_inds > max_dup:
                combined_neuron['duplicate_tol_inds'] = duplicate_tol_inds

        # Remove duplicates found in binary pursuit
        keep_bool = remove_binary_pursuit_duplicates(combined_neuron["spike_indices"],
                        combined_neuron["new_spike_bool"],
                        tol_inds=combined_neuron['duplicate_tol_inds'])
        combined_neuron["spike_indices"] = combined_neuron["spike_indices"][keep_bool]
        combined_neuron["new_spike_bool"] = combined_neuron["new_spike_bool"][keep_bool]
        combined_neuron['waveforms'] = combined_neuron['waveforms'][keep_bool, :]
        channel_selector = channel_selector[keep_bool]
        threshold_by_unit = threshold_by_unit[keep_bool]
        segment_by_unit = segment_by_unit[keep_bool]
        snr_by_unit = snr_by_unit[keep_bool]

        # Get each spike's channel of origin and the waveforms on main channel
        main_waveforms = np.zeros((combined_neuron['waveforms'].shape[0], self.sort_info['n_samples_per_chan']))
        combined_neuron['channel_selector'] = {}
        for chan in combined_neuron['channel']:
            chan_select = channel_selector == chan
            combined_neuron['channel_selector'][chan] = chan_select
            main_win = combined_neuron['main_windows'][chan]
            main_waveforms[chan_select, :] = combined_neuron['waveforms'][chan_select, main_win[0]:main_win[1]]

        # Remove any identical index duplicates (either from error or
        # from combining overlapping segments), preferentially keeping
        # the waveform with largest peak-value to threshold ratio on its main
        # channel
        keep_bool = remove_spike_event_duplicates_across_chans(combined_neuron["spike_indices"],
                        main_waveforms, threshold_by_unit, tol_inds=combined_neuron['duplicate_tol_inds'])
        combined_neuron["spike_indices"] = combined_neuron["spike_indices"][keep_bool]
        combined_neuron["new_spike_bool"] = combined_neuron["new_spike_bool"][keep_bool]
        combined_neuron['waveforms'] = combined_neuron['waveforms'][keep_bool, :]
        for chan in combined_neuron['channel']:
            combined_neuron['channel_selector'][chan] = combined_neuron['channel_selector'][chan][keep_bool]
        channel_selector = channel_selector[keep_bool]
        threshold_by_unit = threshold_by_unit[keep_bool] # NOTE: not currently output...
        segment_by_unit = segment_by_unit[keep_bool]
        snr_by_unit = snr_by_unit[keep_bool]

        # Recompute things of interest like SNR and templates by channel as the
        # average over all data from that channel and store for output
        combined_neuron["template"] = {}
        combined_neuron["snr"] = {}
        chans_to_remove = []
        for chan in combined_neuron['channel']:
            if snr_by_unit[combined_neuron['channel_selector'][chan]].size == 0:
                # Spikes contributing from this channel have been removed so
                # remove all its data below
                chans_to_remove.append(chan)
            else:
                combined_neuron["template"][chan] = np.mean(
                    combined_neuron['waveforms'][combined_neuron['channel_selector'][chan], :], axis=0)
                combined_neuron['snr'][chan] = np.mean(snr_by_unit[combined_neuron['channel_selector'][chan]])
        for chan_ind in reversed(range(0, len(chans_to_remove))):
            chan_num = chans_to_remove[chan_ind]
            del combined_neuron['channel'][chan_ind] # A list so use index
            del combined_neuron['neighbors'][chan_num] # Rest of these are all
            del combined_neuron['chan_neighbor_ind'][chan_num] # dictionaries so
            del combined_neuron['channel_selector'][chan_num] #  use value
            del combined_neuron['main_windows'][chan_num]
        combined_neuron['fraction_mua'] = calc_fraction_mua_to_peak(
                        combined_neuron["spike_indices"],
                        self.sort_info['sampling_rate'],
                        combined_neuron['duplicate_tol_inds'],
                        self.absolute_refractory_period)

        return combined_neuron

    def summarize_neurons_across_channels(self, overlap_time=2.5e-4, min_overlap_ratio=0.5):
        """ Creates output neurons list by combining segment-wise neurons across
        segments and across channels based on identical spikes found during the
        overlapping portions of consecutive segments. Requires that there be
        overlap between segments, and enough overlap to be useful.
        The overlap time is the window used to consider spikes the same"""
        if min_overlap_ratio <= 0. or min_overlap_ratio >= 1.:
            raise ValueError("Min overlap ratio must be a value in (0, 1)")
        max_samples = int(round(overlap_time * self.sort_info['sampling_rate']))
        # Start with neurons as those in first segment
        # Doesn't matter if there are no neurons yet it will get appended later
        # Establishes a list of dictionaries. Each dictionary represents a single
        # neuron as usual. Here we add an extra key 'next_seg_link' that indicates
        # the index of the neuron dictionary in the next segment that
        # corresponds with it. If no neurons in the next seg correspond, the
        # value for this key is None.
        start_seg = 0
        while start_seg < self.n_segments:
            if len(self.neuron_summary_by_seg[start_seg]) == 0:
                start_seg += 1
                continue
            break
        if start_seg >= self.n_segments-1:
            # Need at least 2 remaining segments to stitch.
            if start_seg == self.n_segments:
                # No neurons with data found
                return [{}]
            if len(self.neuron_summary_by_seg[start_seg]) == 0:
                # No neurons with data found
                return [{}]
            # With this being the only segment with data, we are done
            neurons = self.neuron_summary_by_seg[start_seg]
            neuron_summary = []
            for n in neurons:
                n['next_seg_link'] = None
                neuron_summary.append(self.join_neuron_dicts([n]))
            return neuron_summary

        for seg in range(start_seg, self.n_segments-1):
            if len(self.neuron_summary_by_seg[seg]) == 0:
                continue
            next_seg = seg + 1
            if len(self.neuron_summary_by_seg[next_seg]) == 0:
                # Current seg neurons can't overlap with neurons in next seg
                for n in self.neuron_summary_by_seg[seg]:
                    n['next_seg_link'] = None
                continue

            # From next segment start to current segment end
            overlap_win = [self.neuron_summary_seg_inds[next_seg][0], self.neuron_summary_seg_inds[seg][1]]
            n_total_samples = overlap_win[1] - overlap_win[0]
            if n_total_samples < 10 * self.sort_info['sampling_rate']:
                summarize_message = ("Consecutive segment overlap is less than "\
                                     "10 seconds which may not provide enough "\
                                     "data to yield good across channel neuron summary")
                warnings.warn(sort_data_message, RuntimeWarning, stacklevel=2)
            # This is NOT necessarily a square matrix!
            overlap_ratio = np.zeros((len(self.neuron_summary_by_seg[seg]), len(self.neuron_summary_by_seg[next_seg])))
            for cn_ind in range(0, len(self.neuron_summary_by_seg[seg])):
                cn_start = next((idx[0] for idx, val in np.ndenumerate(
                            self.neuron_summary_by_seg[seg][cn_ind]['spike_indices'])
                            if val >= overlap_win[0]), None)
                if cn_start is None:
                    # current neuron has no spikes in the overlap
                    overlap_ratio[cn_ind, :] = 0.
                    continue
                cn_spike_train = compute_spike_trains(
                        self.neuron_summary_by_seg[seg][cn_ind]['spike_indices'][cn_start:],
                        max_samples, overlap_win)
                n_cn_spikes = np.count_nonzero(cn_spike_train)
                for nn_ind in range(0, len(self.neuron_summary_by_seg[next_seg])):
                    nn_stop = next((idx[0] for idx, val in np.ndenumerate(
                                self.neuron_summary_by_seg[next_seg][nn_ind]['spike_indices'])
                                if val >= overlap_win[1]), None)
                    if nn_stop is None or nn_stop == 0:
                        # next neuron has no spikes in the overlap
                        overlap_ratio[cn_ind, nn_ind] = 0.
                        continue
                    nn_spike_train = compute_spike_trains(
                            self.neuron_summary_by_seg[next_seg][nn_ind]['spike_indices'][:nn_stop],
                            max_samples, overlap_win)
                    n_nn_spikes = np.count_nonzero(nn_spike_train)
                    try:
                        num_hits = np.count_nonzero(np.logical_and(cn_spike_train, nn_spike_train))
                        nn_misses = np.count_nonzero(np.logical_and(nn_spike_train, ~cn_spike_train))
                        cn_misses = np.count_nonzero(np.logical_and(cn_spike_train, ~nn_spike_train))

                        # This is NOT symmetric matrix! Just a look up table.
                        overlap_ratio[cn_ind, nn_ind] = max(num_hits / (num_hits + nn_misses),
                                                            num_hits / (num_hits + cn_misses))
                    except:
                        print(num_hits, nn_misses, cn_misses)
                        print(cn_start, nn_stop, n_cn_spikes, n_nn_spikes)
                        print(self.neuron_summary_by_seg[next_seg][nn_ind]['spike_indices'].size)
                        print(self.neuron_summary_by_seg[next_seg][nn_ind]['spike_indices'][nn_stop], overlap_win[1])
                        print(self.neuron_summary_by_seg[next_seg][nn_ind]['spike_indices'][nn_stop+1])
                        raise

            # Assume there is no link and overwrite below if it passes threshold
            for n in self.neuron_summary_by_seg[seg]:
                n['next_seg_link'] = None
            max_cn = np.argmax(np.amax(overlap_ratio, axis=1))
            max_nn = np.argmax(overlap_ratio[max_cn, :])
            max_ratio = overlap_ratio[max_cn, max_nn]
            while max_ratio >= min_overlap_ratio:
                self.neuron_summary_by_seg[seg][max_cn]['next_seg_link'] = max_nn
                # Set to zero so these are not used again
                overlap_ratio[max_cn, :] = 0
                overlap_ratio[:, max_nn] = 0
                # Then recompute the next best value
                max_cn = np.argmax(np.amax(overlap_ratio, axis=1))
                max_nn = np.argmax(overlap_ratio[max_cn, :])
                max_ratio = overlap_ratio[max_cn, max_nn]
            print("Stopped with maximum rejected overlap ratio", max_ratio)

        neurons = self.stitch_neurons_across_channels()
        neuron_summary = []
        for n in neurons:
            neuron_summary.append(self.join_neuron_dicts(n))
        return neuron_summary

    def get_sort_data_by_chan(self):
        """ Returns all sorter data concatenated by channel across all segments,
        thus removing the concept of segment-wise sorting and giving all data
        by channel. """
        crossings, labels, waveforms, new_waveforms = [], [], [], []
        for chan in range(0, self.n_chans):
            seg_crossings, seg_labels, seg_waveforms, seg_new = [], [], [], []
            for seg in range(0, self.n_segments):
                seg_crossings.append(self.sort_data[chan][seg][0])
                seg_labels.append(self.sort_data[chan][seg][1])
                # Waveforms is 2D so can't stack with empty
                if len(self.sort_data[chan][seg][2]) > 0:
                    seg_waveforms.append(self.sort_data[chan][seg][2])
                seg_new.append(self.sort_data[chan][seg][3])
            # stacking with [] casts as float, so ensure maintained types
            crossings.append(np.hstack(seg_crossings).astype(np.int64))
            labels.append(np.hstack(seg_labels).astype(np.int64))
            if len(seg_waveforms) > 0:
                waveforms.append(np.vstack(seg_waveforms).astype(np.float64))
            else:
                waveforms.append([])
            new_waveforms.append(np.hstack(seg_new).astype(np.bool))
        return crossings, labels, waveforms, new_waveforms

    def summarize_neurons(self):
        """
            summarize_neurons(probe, threshold_crossings, labels)

        Return a summarized version of the threshold_crossings, labels, etc. for a given
        set of crossings, neuron labels, and waveforms organized by channel.
        This function returns a list of dictionarys (with symbol look-ups)
        with all essential information about the recording session and sorting.
        The dictionary contains:
        channel: The channel on which the neuron was recorded
        neighbors: The neighborhood of the channel on which the neuron was recorded
        clip_width: The clip width used for sorting the neuron
        sampling_rate: The sampling rate of the recording
        filter_band: The filter band used to filter the data before sorting
        spike_indices: The indices (sample number) in the voltage trace of the threshold crossings
         waveforms: The waveform clips across the entire neighborhood for the sorted neuron
         template: The template for the neuron as the mean of all waveforms.
         new_spike_bool: A logical index into waveforms/spike_indices of added secret spikes
            (only if the index 'new_waveforms' is input)
         mean_firing_rate: The mean firing rate of the neuron over the entire recording
        peak_valley: The peak valley of the template on the channel from which the neuron arises
        """
        # To the sort data segments into our conventional sorter items by
        # channel independent of segment/work item
        crossings, labels, waveforms, new_waveforms = self.get_sort_data_by_chan()
        neuron_summary = []
        for channel in range(0, len(crossings)):
            if len(crossings[channel]) == 0:
                print("Channel ", channel, " has no spikes and was skipped in summary!")
                continue
            for ind, neuron_label in enumerate(np.unique(labels[channel])):
                neuron = {}
                neuron['summary_type'] = 'single_channel'
                print("!!! I NEED TO MAKE THIS SHIFT/ALIGN ALL THE INDICES (IF NOT WAVEFORMS) SO DUPLICATES ETC ARE VALID!!!")
                try:
                    neuron['sort_info'] = self.sort_info
                    neuron['sort_quality'] = None
                    neuron["channel"] = channel
                    # Work items are sorted by channel so just grab neighborhood
                    # for the first segment
                    neuron['neighbors'] = self.work_items[channel][0]['neighbors']
                    neuron['chan_neighbor_ind'] = self.work_items[channel][0]['chan_neighbor_ind']
                    # These thresholds are not quite right since they differ
                    # for each segment...
                    neuron['thresholds'] = self.work_items[channel][0]['thresholds']

                    neuron["spike_indices"] = crossings[channel][labels[channel] == neuron_label]
                    neuron['waveforms'] = waveforms[channel][labels[channel] == neuron_label, :]
                    neuron["new_spike_bool"] = new_waveforms[channel][labels[channel] == neuron_label]

                    # NOTE: This still needs to be done even though segments
                    # were ordered because of overlap!
                    # Ensure spike times are ordered. Must use 'stable' sort for
                    # output to be repeatable because overlapping segments and
                    # binary pursuit can return slightly different dupliate spikes
                    spike_order = np.argsort(neuron["spike_indices"], kind='stable')
                    neuron["spike_indices"] = neuron["spike_indices"][spike_order]
                    neuron['waveforms'] = neuron['waveforms'][spike_order, :]
                    neuron["new_spike_bool"] = neuron["new_spike_bool"][spike_order]
                    # Remove duplicates found in binary pursuit
                    keep_bool = remove_binary_pursuit_duplicates(neuron["spike_indices"],
                                    neuron["new_spike_bool"],
                                    tol_inds=self.duplicate_tol_inds)
                    neuron["spike_indices"] = neuron["spike_indices"][keep_bool]
                    neuron["new_spike_bool"] = neuron["new_spike_bool"][keep_bool]
                    neuron['waveforms'] = neuron['waveforms'][keep_bool, :]

                    # Remove any identical index duplicates (either from error or
                    # from combining overlapping segments), preferentially keeping
                    # the waveform best aligned to the template
                    neuron["template"] = np.mean(neuron['waveforms'], axis=0)
                    keep_bool = remove_spike_event_duplicates(neuron["spike_indices"],
                                    neuron['waveforms'], neuron["template"],
                                    tol_inds=self.duplicate_tol_inds)
                    neuron["spike_indices"] = neuron["spike_indices"][keep_bool]
                    neuron["new_spike_bool"] = neuron["new_spike_bool"][keep_bool]
                    neuron['waveforms'] = neuron['waveforms'][keep_bool, :]

                    neuron["template"] = np.mean(neuron['waveforms'], axis=0)
                    # samples_per_chan = int(neuron['template'].size / neuron['neighbors'].size)
                    # main_start = np.where(neuron['neighbors'] == neuron['channel'])[0][0]
                    # main_template = neuron['template'][main_start*samples_per_chan:main_start*samples_per_chan + samples_per_chan]
                    # neuron["peak_valley"] = np.amax(main_template) - np.amin(main_template)
                except:
                    print("!!! NEURON {0} ON CHANNEL {1} HAD AN ERROR SUMMARIZING !!!".format(neuron_label, channel))
                    neuron["new_spike_bool"] = new_waveforms[channel]
                    neuron["channel"] = channel
                    neuron["all_spike_indices"] = crossings[channel]
                    neuron['all_labels'] = labels[channel]
                    neuron['all_waveforms'] = waveforms[channel]
                    raise
                neuron_summary.append(neuron)
        return neuron_summary






def get_aligned_shifted_template(template_1, template_2, max_samples_window):
    """ Finds the optimal cross correlation based alignment between the two
        input templates by shifting template_2 relative to template_1.  Outputs
        template_2 shifted by zero padding to be optimally aligned with template_1
        and the shift indices needed to get there (negative values are to the left).
        The maximum shift is +/- max_samples_window // 2. """

    temp_xcorr = np.correlate(template_1, template_2, mode='same')
    xcorr_center = np.floor(template_1.size/2).astype(np.int64)
    xcorr_window = max_samples_window // 2
    # Restrict max xcorr to be found within max_samples_window
    shift = xcorr_window - np.argmax(temp_xcorr[xcorr_center-xcorr_window:xcorr_center+xcorr_window])
    # Align clips by zero padding (if needed)
    if shift > 0:
        shifted_template_2 = np.hstack((template_2[shift:], np.zeros((shift))))
    elif shift < 0:
        shifted_template_2 = np.hstack((np.zeros((-1*shift)), template_2[0:shift]))
    else:
        shifted_template_2 = template_2

    return shifted_template_2, shift


def align_spikes_on_template(spikes, template, max_samples_window):

    aligned_spikes = np.empty_like(spikes)
    index_shifts = np.empty(spikes.shape[0], dtype=np.int64)
    xcorr_center = np.floor(template.size/2).astype(np.int64)
    xcorr_window = max_samples_window // 2
    for spk in range(0, spikes.shape[0]):
        temp_xcorr = np.correlate(template, spikes[spk, :], mode='same')
        # Restrict max xcorr to be found within max_samples_window
        shift = xcorr_window - np.argmax(temp_xcorr[xcorr_center-xcorr_window:xcorr_center+xcorr_window])
        # Align clips by zero padding (if needed)
        if shift > 0:
            aligned_spikes[spk, :] = np.hstack((spikes[spk, shift:], np.zeros(shift)))
        elif shift < 0:
            aligned_spikes[spk, :] = np.hstack((np.zeros(-1*shift), spikes[spk, 0:shift]))
        else:
            aligned_spikes[spk, :] = spikes[spk, :]
        index_shifts[spk] = shift

    return aligned_spikes, index_shifts


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

        neuron_labels = merge_clusters(scores, neuron_labels,
                            merge_only=merge_only, split_only=False,
                            p_value_cut_thresh=p_value_cut_thresh)
        n_labels_sharpened = np.unique(neuron_labels).size
        print("Went from", n_labels, "to", n_labels_sharpened, flush=True)
        if n_labels == n_labels_sharpened:
            break
        n_labels = n_labels_sharpened
        n_iters += 1

    return neuron_labels


def merge_test_spikes(spike_list, p_value_cut_thresh=0.05, max_components=20, min_merge_percent=0.99):
    """
        This will return TRUE for comparisons between empty spike lists, i.e., any group of spikes can
        merge with nothing. However, if one of the units in spike list has only spikes that are np.nan,
        this is regarded as a bad comparison and the merge is returned as FALSE. """

    # Remove empty elements of spike list
    empty_spikes_to_remove = []
    for ind in range(0, len(spike_list)):
        if spike_list[ind].shape[0] == 0:
            empty_spikes_to_remove.append(ind)
    empty_spikes_to_remove.sort()
    for es in reversed(empty_spikes_to_remove):
        del spike_list[es]
    if len(spike_list) <= 1:
        # There must be at least 2 units remaining to compare in spike_list or else we define the merge
        # between them to be True
        return True

    templates = np.empty((len(spike_list), spike_list[0].shape[1]))
    n_original = np.empty(len(spike_list))
    neuron_labels = np.empty(0, dtype=np.int64)
    for n in range(0, len(spike_list)):
        neuron_labels = np.hstack((neuron_labels, n * np.ones(spike_list[n].shape[0])))
        n_original[n] = spike_list[n].shape[0]
        templates[n, :] = np.nanmean(spike_list[n], axis=0)
        if np.count_nonzero(~np.any(np.isnan(spike_list[n]), axis=0)) == 0:
            # At least one neuron has no valid spikes
            return False
    neuron_labels = neuron_labels.astype(np.int64)
    clips = np.vstack(spike_list)
    clips = clips[:, ~np.any(np.isnan(clips), axis=0)] # No nans into merge_clusters

    scores = preprocessing.compute_pca_by_channel(clips, np.arange(0, clips.shape[1]), max_components, add_peak_valley=False)
    sharpen = True if n_original.size == 2 and (np.amax(n_original) > 100 * np.amin(n_original)) else False
    neuron_labels = merge_clusters(clips, neuron_labels, merge_only=False, p_value_cut_thresh=p_value_cut_thresh, min_cluster_size=1)

    n_after = np.empty(len(spike_list))
    for n in range(0, len(spike_list)):
        n_after[n] = np.count_nonzero(neuron_labels == n) / n_original[n]

    n_merged_neurons = np.count_nonzero(n_after < 1-min_merge_percent)
    if n_merged_neurons == len(spike_list) - 1:
        merged = True
    else:
        merged = False

    return merged


def combine_neurons_test(neuron1, neuron2, max_samples, max_offset_samples, p_value_cut_thresh, max_components, return_new_neuron=False):
    """
        """

    max_chans_per_spike = np.amax((neuron1['neighbors'].size, neuron2['neighbors'].size))
    window = [0, 0]
    window[0] = int(round(neuron1['clip_width'][0] * neuron1['sampling_rate']))
    window[1] = int(round(neuron1['clip_width'][1] * neuron1['sampling_rate'])) + 1 # Add one so that last element is included
    samples_per_chan = window[1] - window[0]
    spike_chan_inds = []
    for x in range(0, max_chans_per_spike):
        spike_chan_inds.append(np.arange(x*samples_per_chan, (x+1)*samples_per_chan))
    spike_chan_inds = np.vstack(spike_chan_inds)

    same_chan = True if neuron1['channel'] == neuron2['channel'] else False

    should_combine = False
    is_mixed = (False, 0) # Hard to tell which unit is mixed

    # Get indices for various overlapping channels between these neurons' neighborhoods
    overlap_chans, common1, common2 = np.intersect1d(neuron1['neighbors'], neuron2['neighbors'], return_indices=True)
    if neuron1['channel'] not in overlap_chans or neuron2['channel'] not in overlap_chans:
        print("Neurons do not have main channel in their intersection, returning False")
        return False, False, neuron2
    main_ind_n1_on_n1 = segment.get_windows_and_indices(neuron1['clip_width'], neuron1['sampling_rate'], neuron1['channel'], neuron1['neighbors'])[4]
    # Get indices of neuron2 on neuron2 main channel
    main_ind_n2_on_n2 = segment.get_windows_and_indices(neuron2['clip_width'], neuron2['sampling_rate'], neuron2['channel'], neuron2['neighbors'])[4]
    # Get indices of neuron1 on neuron2 main channel and neuron2 on neuron1 main channel
    main_ind_n1_on_n2 = segment.get_windows_and_indices(neuron1['clip_width'], neuron1['sampling_rate'], neuron2['channel'], neuron1['neighbors'])[4]
    main_ind_n2_on_n1 = segment.get_windows_and_indices(neuron2['clip_width'], neuron2['sampling_rate'], neuron1['channel'], neuron2['neighbors'])[4]

    # Find same and different spikes for each neuron
    different_times1 = ~find_overlapping_spike_bool(neuron2['spike_indices'], neuron1['spike_indices'], max_samples=max_samples)
    different_times2 = ~find_overlapping_spike_bool(neuron1['spike_indices'], neuron2['spike_indices'], max_samples=max_samples)

    # First test whether these are same unit by checking spikes from each
    # neuron in their full neighborhood overlap, separate for same and
    # different spikes
    different_spikes1 = neuron1['waveforms'][different_times1, :][:, spike_chan_inds[common1, :].flatten()]
    different_spikes2 = neuron2['waveforms'][different_times2, :][:, spike_chan_inds[common2, :].flatten()]
    if not same_chan:
        same_spikes1 = neuron1['waveforms'][~different_times1, :][:, spike_chan_inds[common1, :].flatten()]
        same_spikes2 = neuron2['waveforms'][~different_times2, :][:, spike_chan_inds[common2, :].flatten()]
    else:
        # Set same spikes to all spikes because same channel doesn't have have same spike times
        same_spikes1 = neuron1['waveforms'][:, spike_chan_inds[common1, :].flatten()]
        same_spikes2 = neuron2['waveforms'][:, spike_chan_inds[common2, :].flatten()]

    # Align neuron2 spikes on neuron1 and merge test them
    different_spikes2 = align_spikes_on_template(different_spikes2, neuron1['template'][spike_chan_inds[common1, :].flatten()], 2*max_offset_samples)[0]
    merged_different = merge_test_spikes([different_spikes1, different_spikes2], p_value_cut_thresh, max_components)
    same_spikes2 = align_spikes_on_template(same_spikes2, neuron1['template'][spike_chan_inds[common1, :].flatten()], 2*max_offset_samples)[0]
    if not same_chan:
        merged_same = merge_test_spikes([same_spikes1, same_spikes2], p_value_cut_thresh, max_components)
        merged_cross1 = merge_test_spikes([same_spikes1, different_spikes2], p_value_cut_thresh, max_components)
        merged_cross2 = merge_test_spikes([different_spikes1, same_spikes2], p_value_cut_thresh, max_components)
        merged_union = merge_test_spikes([same_spikes1, different_spikes1, different_spikes2], p_value_cut_thresh, max_components)
    else:
        # Merged same is computed over all spikes
        merged_same = merge_test_spikes([same_spikes1, same_spikes2], p_value_cut_thresh, max_components)
        merged_cross1 = True
        merged_cross2 = True
        merged_union = True

    if merged_same and (not merged_cross1 or not merged_cross2):
        # Suspicious of one of these neurons being a mixture but hard to say which...
        if merged_cross1 and not merged_cross2:
            mix_neuron_num = 1
        elif merged_cross2 and not merged_cross1:
            # neuron2 is a mixture?
            mix_neuron_num = 2
        else:
            # If both fail I really don't know what that means
            mix_neuron_num = 0
        is_mixed = (True, mix_neuron_num)
        return should_combine, is_mixed, neuron2
    elif not merged_same or not merged_different or not merged_cross1 or not merged_cross2 or not merged_union:
        return should_combine, is_mixed, neuron2

    # Now test using data from each neurons' primary channel data only
    different_spikes1 = np.hstack((neuron1['waveforms'][different_times1, :][:, main_ind_n1_on_n1], neuron1['waveforms'][different_times1, :][:, main_ind_n1_on_n2]))
    different_spikes2 = np.hstack((neuron2['waveforms'][different_times2, :][:, main_ind_n2_on_n1], neuron2['waveforms'][different_times2, :][:, main_ind_n2_on_n2]))
    if not same_chan:
        same_spikes1 = np.hstack((neuron1['waveforms'][~different_times1, :][:, main_ind_n1_on_n1], neuron1['waveforms'][~different_times1, :][:, main_ind_n1_on_n2]))
        same_spikes2 = np.hstack((neuron2['waveforms'][~different_times2, :][:, main_ind_n2_on_n1], neuron2['waveforms'][~different_times2, :][:, main_ind_n2_on_n2]))

    # Align neuron2 spikes on neuron1 and merge test them
    different_spikes2 = align_spikes_on_template(different_spikes2, np.hstack((neuron1['template'][main_ind_n1_on_n1], neuron1['template'][main_ind_n1_on_n2])), 2*max_offset_samples)[0]
    merged_different = merge_test_spikes([different_spikes1, different_spikes2], p_value_cut_thresh, max_components)
    if not same_chan:
        same_spikes2 = align_spikes_on_template(same_spikes2, np.hstack((neuron1['template'][main_ind_n1_on_n1], neuron1['template'][main_ind_n1_on_n2])), 2*max_offset_samples)[0]
        merged_same = merge_test_spikes([same_spikes1, same_spikes2], p_value_cut_thresh, max_components)
        merged_cross1 = merge_test_spikes([same_spikes1, different_spikes2], p_value_cut_thresh, max_components)
        merged_cross2 = merge_test_spikes([different_spikes1, same_spikes2], p_value_cut_thresh, max_components)
        merged_union = merge_test_spikes([same_spikes1, different_spikes1, different_spikes2], p_value_cut_thresh, max_components)
    else:
        # This skips same channel data to final single channel check below because their main channel is the same
        merged_same = True
        merged_cross1 = True
        merged_cross2 = True
        merged_union = True

    if merged_same and (not merged_cross1 or not merged_cross2):
        # Suspicious of one of these neurons being a mixture but hard to say which...
        if merged_cross1 and not merged_cross2:
            mix_neuron_num = 1
        elif merged_cross2 and not merged_cross1:
            # neuron2 is a mixture?
            mix_neuron_num = 2
        else:
            # If both fail I really don't know what that means
            mix_neuron_num = 0
        is_mixed = (True, mix_neuron_num)
        return should_combine, is_mixed, neuron2
    elif not merged_same or not merged_different or not merged_cross1 or not merged_cross2 or not merged_union:
        return should_combine, is_mixed, neuron2

    # Finally test using data only on neuron1's primary channel
    different_spikes1 = neuron1['waveforms'][different_times1, :][:, main_ind_n1_on_n1]
    different_spikes2 = neuron2['waveforms'][different_times2, :][:, main_ind_n2_on_n1]
    if not same_chan:
        same_spikes1 = neuron1['waveforms'][~different_times1, :][:, main_ind_n1_on_n1]
        same_spikes2 = neuron2['waveforms'][~different_times2, :][:, main_ind_n2_on_n1]
    else:
        # Set same spikes to all spikes because same channel doesn't have have same spike times
        same_spikes1 = neuron1['waveforms']
        same_spikes2 = neuron2['waveforms']

    # Align neuron2 spikes on neuron1 and merge test them
    different_spikes2 = align_spikes_on_template(different_spikes2, neuron1['template'][main_ind_n1_on_n1], 2*max_offset_samples)[0]
    merged_different = merge_test_spikes([different_spikes1, different_spikes2], p_value_cut_thresh, max_components)
    same_spikes2 = align_spikes_on_template(same_spikes2, neuron1['template'][main_ind_n1_on_n1], 2*max_offset_samples)[0]
    if not same_chan:
        merged_same = merge_test_spikes([same_spikes1, same_spikes2], p_value_cut_thresh, max_components)
        merged_cross1 = merge_test_spikes([same_spikes1, different_spikes2], p_value_cut_thresh, max_components)
        merged_cross2 = merge_test_spikes([different_spikes1, same_spikes2], p_value_cut_thresh, max_components)
        merged_union = merge_test_spikes([same_spikes1, different_spikes1, different_spikes2], p_value_cut_thresh, max_components)
    else:
        # Since these units have the same neighborhood, attempt to merge them on each of their channels separately
        merged_same = True
        for chan in range(0, neuron1['neighbors'].size):
            merged_same = merged_same and merge_test_spikes([same_spikes1[:, spike_chan_inds[chan, :]], same_spikes2[:, spike_chan_inds[chan, :]]], p_value_cut_thresh, max_components)
        merged_cross1 = True
        merged_cross2 = True
        merged_union = True

    if merged_same and (not merged_cross1 or not merged_cross2):
        # Suspicious of one of these neurons being a mixture but hard to say which...
        if merged_cross1 and not merged_cross2:
            mix_neuron_num = 1
        elif merged_cross2 and not merged_cross1:
            # neuron2 is a mixture?
            mix_neuron_num = 2
        else:
            # If both fail I really don't know what that means
            mix_neuron_num = 0
        is_mixed = (True, mix_neuron_num)
        return should_combine, is_mixed, neuron2
    elif not merged_same or not merged_different or not merged_cross1 or not merged_cross2 or not merged_union:
        return should_combine, is_mixed, neuron2

    # If we made it to here, these neurons are the same so combine them and return
    # the newly combined neuron if requested
    if return_new_neuron:
        # First copy neuron2 different waveforms and spike times then align them
        # to conform with neuron1
        new_neuron = copy.deepcopy(neuron2)
        different_spikes2 = new_neuron['waveforms'][different_times2, :][:, spike_chan_inds[common2, :].flatten()]
        different_spikes2, ind_shift = align_spikes_on_template(different_spikes2, neuron1['template'][spike_chan_inds[common1, :].flatten()], 2*max_offset_samples)
        # Then place in nan padded array that fits into neuron1 waveforms
        new_waves = np.full((different_spikes2.shape[0], max(neuron1['waveforms'].shape[1], different_spikes2.shape[1])), np.nan)
        for c2_ind, c1 in enumerate(common1): #c1, c2 in zip(common1, common2):
            new_waves[:, spike_chan_inds[c1]] = different_spikes2[:, spike_chan_inds[c2_ind]]
        # Reset new_neuron values to be the shifted different spikes before returning
        new_neuron['waveforms'] = new_waves
        new_neuron['spike_indices'] = new_neuron['spike_indices'][different_times2] + ind_shift
        new_neuron["new_spike_bool"] = new_neuron["new_spike_bool"][different_times2]
    else:
        new_neuron = neuron2

    # If we made it here these neurons have combined fully
    should_combine = True
    return should_combine, is_mixed, new_neuron





def count_ISI_violations(neuron, min_ISI):

    min_samples = np.ceil(min_ISI * neuron['sampling_rate']).astype(np.int64)
    # Find ISI violations, excluding equal values
    different_ISI_violations = find_overlapping_spike_bool(neuron['spike_indices'], neuron['spike_indices'], max_samples=min_samples, except_equal=True)
    # Shift so different_ISI_violations keeps the first violator
    different_ISI_violations = np.hstack((np.zeros(1, dtype=np.bool), different_ISI_violations[0:-1]))
    # Also find any duplicate values and keep only one of them
    repeats_bool = np.ones(neuron['spike_indices'].size, dtype=np.bool)
    repeats_bool[np.unique(neuron['spike_indices'], return_index=True)[1]] = False
    violation_index = np.logical_or(different_ISI_violations, repeats_bool)
    n_violations = np.count_nonzero(violation_index)

    return n_violations, violation_index


def find_ISI_spikes_to_keep(neuron, min_ISI):
    """ Finds spikes within a neuron that violate the input min_ISI.  When two
        spike times are found within an ISI less than min_ISI, the spike waveform
        with greater projection onto the neuron's template is kept.
    """

    min_samples = np.ceil(min_ISI * neuron['sampling_rate']).astype(np.int64)
    keep_bool = np.ones(neuron['spike_indices'].size, dtype=np.bool)
    template_norm = neuron['template'] / np.linalg.norm(neuron['template'])
    curr_index = 0
    next_index = 1
    while next_index < neuron['spike_indices'].size:
        if neuron['spike_indices'][next_index] - neuron['spike_indices'][curr_index] < min_samples:
            projections = neuron['waveforms'][curr_index:next_index+1, :] @ template_norm
            if projections[0] >= projections[1]:
                # current spike is better or equal
                keep_bool[next_index] = False
                next_index += 1
            else:
                # next spike is better
                keep_bool[curr_index] = False
                curr_index = next_index
                next_index += 1
        else:
            curr_index = next_index
            next_index += 1

    return keep_bool


def combine_two_neurons(neuron1, neuron2, min_ISI=6e-4):
    """
        Returns a modified neuron1 with neuron2's spikes, waveforms and all other
        data added to it. The user must beware whether the spike waveforms that
        are input are able to be sensibly concatenated vertically, else the
        neighborhood waveforms will be ruined. This also checks for and chooses
        between spikes that cause ISI violations.  This function DOES NOT
        alter or remove neuron2! """

    # Merge neuron2 spikes into neuron1
    neuron1["spike_indices"] = np.hstack((neuron1["spike_indices"], neuron2["spike_indices"]))
    neuron1['waveforms'] = np.vstack((neuron1["waveforms"], neuron2["waveforms"]))

    # Ensure spike times are ordered without ISI violations and stay matched with waveforms
    spike_order = np.argsort(neuron1["spike_indices"])
    neuron1["spike_indices"] = neuron1["spike_indices"][spike_order]
    neuron1['waveforms'] = neuron1['waveforms'][spike_order, :]
    neuron1["template"] = np.nanmean(neuron1['waveforms'], axis=0) # Need this for ISI removal
    wave_nans = np.isnan(neuron1['waveforms']) # Track nans
    neuron1['waveforms'][wave_nans] = 0 # Do not put nans into 'find_ISI_spikes_to_keep'
    keep_bool = find_ISI_spikes_to_keep(neuron1, min_ISI=min_ISI)
    neuron1['waveforms'][wave_nans] = np.nan

    neuron1["spike_indices"] = neuron1["spike_indices"][keep_bool]
    neuron1['waveforms'] = neuron1['waveforms'][keep_bool, :]
    if 'new_spike_bool' in neuron1.keys():
        neuron1["new_spike_bool"] = np.hstack((neuron1["new_spike_bool"], neuron2["new_spike_bool"]))
        neuron1["new_spike_bool"] = neuron1["new_spike_bool"][spike_order]
        neuron1["new_spike_bool"] = neuron1["new_spike_bool"][keep_bool]

    neuron1["template"] = np.nanmean(neuron1['waveforms'], axis=0) # Recompute AFTER ISI removal
    last_spike_t = neuron1["spike_indices"][-1] / neuron1['sampling_rate']
    first_spike_t = neuron1["spike_indices"][0] / neuron1['sampling_rate']
    neuron1["mean_firing_rate"] = neuron1["spike_indices"].shape[0] / (last_spike_t - first_spike_t)
    samples_per_chan = int(neuron1['template'].size / neuron1['neighbors'].size)
    main_start = np.where(neuron1['neighbors'] == neuron1['channel'])[0][0]
    main_template = neuron1['template'][main_start*samples_per_chan:main_start*samples_per_chan + samples_per_chan]
    neuron1["peak_valley"] = np.amax(main_template) - np.amin(main_template)

    return neuron1


def check_for_duplicate_spikes(clips, event_indices, neuron_labels, min_samples):
    """ DATA MUST BE SORTED IN ORDER OF EVENT INDICES FOR THIS TO WORK.  Duplicates
        are only removed within each neuron label.
    """

    keep_bool = np.ones(event_indices.size, dtype='bool')
    templates, labels = calculate_templates(clips, neuron_labels)
    template_norm = []
    for t in templates:
        template_norm.append(t / np.linalg.norm(t))
    for neuron in labels:
        curr_spikes = np.nonzero(neuron_labels == neuron)[0]
        curr_index = 0
        next_index = 1
        while next_index < curr_spikes.size:
            if event_indices[curr_spikes[next_index]] - event_indices[curr_spikes[curr_index]] < min_samples:
                projections = clips[(curr_spikes[curr_index], curr_spikes[next_index]), :] @ template_norm[np.nonzero(neuron == labels)[0][0]]
                if projections[0] > projections[1]:
                    # current spike is better
                    keep_bool[curr_spikes[next_index]] = False
                    next_index += 1
                else:
                    # next spike is better
                    keep_bool[curr_spikes[curr_index]] = False
                    curr_index = next_index
                    next_index += 1
            else:
                curr_index = next_index
                next_index += 1

    return keep_bool


def remove_ISI_violations(neurons, min_ISI=None):
    """ Loops over all neuron dictionaries in neurons list and checks for ISI
        violations under min_ISI.  Violations are removed by calling
        find_ISI_spikes_to_keep above. """

    for neuron in neurons:
        if min_ISI is None:
            min_ISI = np.amax(np.abs(neuron['clip_width']))

        keep_bool = find_ISI_spikes_to_keep(neuron, min_ISI=min_ISI)
        neuron["spike_indices"] = neuron["spike_indices"][keep_bool]
        neuron['waveforms'] = neuron['waveforms'][keep_bool, :]
        if 'new_spike_bool' in neuron.keys():
            neuron["new_spike_bool"] = neuron["new_spike_bool"][keep_bool]

    return neurons


def combine_neighborhood_neurons(neuron_summary, overlap_time=2.5e-4, max_offset_samples=10, p_value_cut_thresh=0.01, max_components=None):
    """ Neurons will be combined with the lower index neuron in neuron summary mergining into the
        higher indexed neuron and then being deleted from the returned neuron_summary.
        Will assume that all neurons in neuron_summary have the same sampling rate and clip width
        etc.  Any noise spikes should be removed before calling this function as merging
        them into good units will be tough to fix and this function essentially assumes
        everything is a decently isolated/sorted unit at this point.  Significant overlaps are
        required before checking whether spikes look the same for units that are not on the same
        channel.  Units that are on the same channel do not need to satisfy this requirement.
        Same channel units will be aligned as if they are in fact the same unit, then tested
        for whether their spikes appear the same given the same alignment.  This can merge
        a neuron that has been split into multiple units on the same channel due to small
        waveform variations that have shifted its alignment. This function calls itself recursively
        until no more merges are made.
        NOTE: ** I think this also requires that
        all neighborhoods for each neuron are numerically ordered !! I should probably check/fix this.

    PARAMETERS
        neuron_summary: list of neurons output by spike sorter
        overlap_time: time window in which two spikes occurring in the same neighborhood can be
            considered the same spike.  This should allow a little leeway to account for the
            possibility that spikes are aligned slighly differently on different channels even
            though they are in fact the same spike.
        max_offset_samples: the maximum number of samples that a neuron's spikes or template
            may be shifted in an attempt to align it with another neuron for comparison
    RETURNS
        neuron_summary: same as input but neurons now include any merged spikes and lesser
            spiking neurons that were merged have been removed. """


    neurons_to_combine = []
    neurons_to_delete = []
    already_tested = []
    already_merged = []
    max_samples = int(round(overlap_time * neuron_summary[0]['sampling_rate']))
    if max_components is None:
        max_components = 10

    # Create list of dictionaries of overlapping spikes and their percentage between all pairwise units
    neighbor_distances = [{x: 1.} for x in range(0, len(neuron_summary))]
    violation_partners = [set() for x in range(0, len(neuron_summary))]
    for n1_ind, n1 in enumerate(neuron_summary):
        for n2_ind in range(n1_ind+1, len(neuron_summary)):
            n2 = neuron_summary[n2_ind]
            if np.intersect1d(n1['neighbors'], n2['neighbors']).size == 0:
                continue
            actual_overlaps = zero_symmetric_ccg(n1['spike_indices'], n2['spike_indices'], max_samples, max_samples)[0]
            neighbor_distances[n2_ind][n1_ind] = actual_overlaps[1] / n1['spike_indices'].size
            neighbor_distances[n1_ind][n2_ind] = actual_overlaps[1] / n2['spike_indices'].size
            if actual_overlaps[1] > 2 * (actual_overlaps[0] + actual_overlaps[2]):
                violation_partners[n1_ind].add(n2_ind)
                violation_partners[n2_ind].add(n1_ind)

    # Start fresh with remaining units
    # Get some needed info about this recording and sorting
    n_chans = 0
    # max_chans_per_spike = 1
    for n in neuron_summary:
        if np.amax(n['neighbors']) > n_chans:
            n_chans = np.amax(n['neighbors'])
        # if n['neighbors'].size > max_chans_per_spike:
        #     max_chans_per_spike = n['neighbors'].size
    n_chans += 1 # Need to add 1 since chan nums start at 0
    inds_by_chan = [[] for x in range(0, n_chans)]
    for ind, n in enumerate(neuron_summary):
        inds_by_chan[n['channel']].append(ind)
    neurons_to_combine = []
    neurons_to_delete = []

    for neuron1_ind, neuron1 in enumerate(neuron_summary): # each neuron
        if neuron1_ind in neurons_to_delete or neuron1_ind in already_merged:
            continue
        for neighbor in neuron1['neighbors']: # each neighboring channel
            if neuron1_ind in already_merged:
                break
            for neuron2_ind in inds_by_chan[neighbor]: # each neuron on neighboring channel
                neuron2 = neuron_summary[neuron2_ind]
                if (neuron2_ind in neurons_to_delete) or (neuron1_ind in neurons_to_delete):
                    continue
                if (neuron2_ind == neuron1_ind) or (set([neuron1_ind, neuron2_ind]) in already_tested):
                    # Ensure a unit is only combined into another one time,
                    # skip same units, don't repeat comparisons
                    continue
                else:
                    already_tested.append(set([neuron1_ind, neuron2_ind]))
                overlap_chans, common1, common2 = np.intersect1d(neuron1['neighbors'], neuron2['neighbors'], return_indices=True)
                if neuron2['channel'] not in overlap_chans or neuron1['channel'] not in overlap_chans:
                    # Need both main channels in intersection
                    continue

                # Check if these neurons have unexpected overlap within time that they simultaneously
                # exist and thus may be same unit or a mixed unit
                if (neuron2_ind not in violation_partners[neuron1_ind]
                    and (neuron1['channel'] != neuron2['channel'])):
                    # Not enough overlap to suggest same neuron.  We have previously enforced no
                    # overlap on same channel so do not hold this against same chanel spikes
                    continue

                if neuron1_ind in already_merged or neuron2_ind in already_merged:
                    # Check already_merged here, AFTER the above mixed check because we don't want
                    # to overlook possible mixtures just because they contain a previously merged
                    # unit
                    continue

                # Make sure we merge into and thus align with the unit with the most spikes
                if neuron1['spike_indices'].size < neuron2['spike_indices'].size:
                    should_combine, is_mixed, neuron_summary[neuron1_ind] = combine_neurons_test(neuron2, neuron1, max_samples, max_offset_samples, p_value_cut_thresh, max_components, return_new_neuron=True)
                    if should_combine:
                        neurons_to_combine.append([neuron2_ind, neuron1_ind])
                        neurons_to_delete.append(neuron1_ind)
                        already_merged.append(neuron2_ind)
                    elif is_mixed[0]:
                        print("ONE OF NEURONS", neuron2_ind, neuron1_ind, "IS LIKELY A MIXTURE FROM CROSS MERGE TEST !!!")
                        if is_mixed[1] == 1:
                            print("Guessing it's neuron", neuron2_ind, "?")
                        elif is_mixed[1] == 2:
                            print("Guessing it's neuron", neuron1_ind, "?")
                        else:
                            print("Not sure which one because both failed cross?")
                else:
                    should_combine, is_mixed, neuron_summary[neuron2_ind] = combine_neurons_test(neuron1, neuron2, max_samples, max_offset_samples, p_value_cut_thresh, max_components, return_new_neuron=True)
                    if should_combine:
                        neurons_to_combine.append([neuron1_ind, neuron2_ind])
                        neurons_to_delete.append(neuron2_ind)
                        already_merged.append(neuron1_ind)
                    elif is_mixed:
                        print("ONE OF NEURONS", neuron1_ind, neuron2_ind, "IS LIKELY A MIXTURE FROM CROSS MERGE TEST !!!")
                        if is_mixed[1] == 1:
                            print("Guessing it's neuron", neuron1_ind, "?")
                        elif is_mixed[1] == 2:
                            print("Guessing it's neuron", neuron2_ind, "?")
                        else:
                            print("Not sure which one because both failed cross?")

    for ind1, ind2 in neurons_to_combine:
        print("Combining neurons", ind1, ind2)
        neuron_summary[ind1] = combine_two_neurons(neuron_summary[ind1], neuron_summary[ind2], min_ISI=overlap_time)

    # Remove lesser merged neurons
    neurons_to_delete.sort()
    for ind in reversed(neurons_to_delete):
        print("Deleting neuron", ind)
        del neuron_summary[ind]
    if len(neurons_to_delete) > 0:
        # Recurse until no more combinations are made
        print("RECURSING")
        neuron_summary = combine_neighborhood_neurons(neuron_summary, overlap_time=overlap_time, p_value_cut_thresh=p_value_cut_thresh)

    return neuron_summary
