import numpy as np
from spikesorting_python.src import segment
from spikesorting_python.src import sort
from spikesorting_python.src import preprocessing
from scipy import stats
from scipy.optimize import nnls, lsq_linear
import copy


"""
    compute_waves_and_template(spikes, labels, label)

Find all waveforms for a list of spikes given a label. The average
template is the mean template across all samples.
"""
def compute_waves_and_template(Probe, channel, indices, clip_width=[-2e-4, 4e-4]):
    spike_clips, valid_event_indices = segment.get_singlechannel_clips(Probe, channel, indices, clip_width=clip_width)
    if np.sum(valid_event_indices) != spike_clips.shape[0]:
        raise ValueError("Number of spike clips does not match number of valid event indices")

    return np.nanmean(spike_clips, axis=0), spike_clips


def recompute_template_wave_properties(neuron_summary):

    if not isinstance(neuron_summary, list):
        neuron_summary = [neuron_summary]

    for n in neuron_summary:
        n["template"] = np.nanmean(n["waveforms"], axis=0)
        if (n['template'].size / n['neighbors'].size) != int(n['template'].size / n['neighbors'].size):
            raise ValueError("Template size must be divisible by number of channels in template!")
        else:
            samples_per_chan = int(n['template'].size / n['neighbors'].size)
        main_start = np.where(n['neighbors'] == n['channel'])[0][0]
        main_template = n['template'][main_start*samples_per_chan:main_start*samples_per_chan + samples_per_chan]
        n["peak_valley"] = np.amax(main_template) - np.amin(main_template)

    return neuron_summary


def reorder_neurons_by_raw_peak_valley(neuron_summary):

    ordered_peak_valley = np.empty(len(neuron_summary))
    for n in range(0, len(neuron_summary)):
        ordered_peak_valley[n] = neuron_summary[n]['peak_valley']
    neuron_summary = [neuron_summary[x] for x in reversed(np.argsort(ordered_peak_valley))]

    return neuron_summary


def find_overlapping_spike_bool(spikes1, spikes2, max_samples=20, except_equal=False):
    """ Finds an index into spikes2 that indicates whether a spike index of spike2
        occurs within +/- max_samples of a spike index in spikes1.  Spikes1 and 2
        are numpy arrays of spike indices in units of samples. Input spikes1 and
        spikes2 will be SORTED IN PLACE because this function won't work if they
        are not ordered. If spikes1 == spikes2 and except_equal is True, the first
        spike in the pair of overlapping spikes is flagged as True in the output
        overlapping_spike_bool. """

    spikes1.sort()
    spikes2.sort()
    max_samples = np.ceil(max_samples).astype('int')
    overlapping_spike_bool = np.zeros(spikes2.size, dtype='bool')
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


def count_ISI_violations(neuron, min_ISI):

    min_samples = np.ceil(min_ISI * neuron['sampling_rate']).astype('int')
    # Find ISI violations, excluding equal values
    different_ISI_violations = find_overlapping_spike_bool(neuron['spike_indices'], neuron['spike_indices'], max_samples=min_samples, except_equal=True)
    # Shift so different_ISI_violations keeps the first violator
    different_ISI_violations = np.hstack((np.zeros(1, dtype='bool'), different_ISI_violations[0:-1]))
    # Also find any duplicate values and keep only one of them
    repeats_bool = np.ones(neuron['spike_indices'].size, dtype='bool')
    repeats_bool[np.unique(neuron['spike_indices'], return_index=True)[1]] = False
    violation_index = np.logical_or(different_ISI_violations, repeats_bool)
    n_violations = np.count_nonzero(violation_index)

    return n_violations, violation_index


def find_ISI_spikes_to_keep(neuron, min_ISI):
    """ Finds spikes within a neuron that violate the input min_ISI.  When two
        spike times are found within an ISI less than min_ISI, the spike waveform
        with greater projection onto the neuron's template is kept.
    """

    min_samples = np.ceil(min_ISI * neuron['sampling_rate']).astype('int')
    keep_bool = np.ones(neuron['spike_indices'].size, dtype='bool')
    template_norm = neuron['template'] / np.linalg.norm(neuron['template'])
    curr_index = 0
    next_index = 1
    while next_index < neuron['spike_indices'].size:
        if neuron['spike_indices'][next_index] - neuron['spike_indices'][curr_index] < min_samples:
            projections = neuron['waveforms'][curr_index:next_index+1, :] @ template_norm
            if projections[0] > projections[1]:
                # current spike is better
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


def compute_SNR(neurons):
    """ Compute SNR for each neuron following the method of Kelly et al. 2007:
        "Comparison of Recordings from Microelectrode Arrays and Single
         Electrodes in the Visual Cortex "
        """

    for neuron in neurons:
        curr_chan_inds = segment.get_windows_and_indices(neuron['clip_width'],
                neuron['sampling_rate'], neuron['channel'], neuron['neighbors'])[4]
        epsilon = neuron['waveforms'][:, curr_chan_inds] - neuron['template'][curr_chan_inds][None, :]
        std_epsilon = np.nanstd(epsilon)
        neuron['sort_quality']['SNR'] = ((np.nanmax(neuron['template'][curr_chan_inds])
                - np.nanmin(neuron['template'][curr_chan_inds])) / (3 * std_epsilon))

    return neurons


"""
    summarize_neurons(probe, threshold_crossings, labels)

Return a summarized version of the threshold_crossings, labels, etc. for a given
set of neurons. This function returns a list of dictionarys (with symbol look-ups)
with all essential information about the recording session and sorting.
The dictionary contains:
channel: The channel on which the neuron was recorded
neighbors: The neighborhood of the channel on which the neuron was recorded
clip_width: The clip width used for sorting the neuron
sampling_rate: The sampling rate of the recording
filter_band: The filter band used to filter the data before sorting
spike_indices: The indices (sample number) in the voltage trace of the threshold crossings
 waveforms: The waveform clips across the entire neighborhood for the sorted neuron
 template: The template for the neuron as the nanmean of all waveforms.
 new_spike_bool: A logical index into waveforms/spike_indices of added secret spikes
    (only if the index 'new_waveforms' is input)
 mean_firing_rate: The mean firing rate of the neuron over the entire recording
peak_valley: The peak valley of the template on the channel from which the neuron arises
"""
def summarize_neurons(Probe, threshold_crossings, labels, waveforms, thresholds,
        clip_width=[-2e-4, 4e-4], new_waveforms=None, max_components=None):

    neuron_summary = []
    for channel in range(0, len(threshold_crossings)):
        if len(threshold_crossings[channel]) == 0:
            print("Channel ", channel, " has no spikes and was skipped in summary!")
            continue
        if new_waveforms is not None:
            new_wave_bool = np.zeros(threshold_crossings[channel].size, dtype='bool')
            new_wave_bool[new_waveforms[channel]] = True
        for ind, neuron_label in enumerate(np.unique(labels[channel])):
            neuron = {}
            try:
                if max_components is not None:
                    neuron['max_components'] = max_components
                neuron['sort_quality'] = None
                neuron["channel"] = channel
                neuron['neighbors'] = Probe.get_neighbors(channel)
                neuron["clip_width"] = clip_width
                neuron['sampling_rate'] = Probe.sampling_rate
                neuron['filter_band'] = Probe.filter_band
                neuron["spike_indices"] = threshold_crossings[channel][labels[channel] == neuron_label]
                neuron['waveforms'] = waveforms[channel][labels[channel] == neuron_label, :]

                # Ensure spike times are ordered
                spike_order = np.argsort(neuron["spike_indices"])
                neuron["spike_indices"] = neuron["spike_indices"][spike_order]
                neuron['waveforms'] = neuron['waveforms'][spike_order, :]
                if new_waveforms is not None:
                    neuron["new_spike_bool"] = new_wave_bool[labels[channel] == neuron_label]
                    neuron["new_spike_bool"] = neuron["new_spike_bool"][spike_order]

                neuron["template"] = np.nanmean(neuron['waveforms'], axis=0) 
                last_spike_t = neuron["spike_indices"][-1] / Probe.sampling_rate
                first_spike_t = neuron["spike_indices"][0] / Probe.sampling_rate
                if first_spike_t == last_spike_t:
                    neuron["mean_firing_rate"] = 0
                else:
                    neuron["mean_firing_rate"] = neuron["spike_indices"].shape[0] / (last_spike_t - first_spike_t)
                samples_per_chan = int(neuron['template'].size / neuron['neighbors'].size)
                main_start = np.where(neuron['neighbors'] == neuron['channel'])[0][0]
                main_template = neuron['template'][main_start*samples_per_chan:main_start*samples_per_chan + samples_per_chan]
                neuron["peak_valley"] = np.amax(main_template) - np.amin(main_template)
            except:
                print("!!! NEURON {0} ON CHANNEL {1} HAD AN ERROR SUMMARIZING !!!".format(neuron_label, channel))
                if new_waveforms is not None:
                    neuron["new_spike_bool"] = new_wave_bool
                neuron["channel"] = channel
                neuron["clip_width"] = clip_width
                neuron["all_spike_indices"] = threshold_crossings[channel]
                neuron['all_labels'] = labels[channel]
                neuron['sampling_rate'] = Probe.sampling_rate
                neuron['filter_band'] = Probe.filter_band
                neuron['all_waveforms'] = waveforms[channel]
            neuron_summary.append(neuron)
            if len(thresholds) == len(threshold_crossings):
                neuron["threshold"] = thresholds[channel]
            else:
                neuron["threshold"] = None

    return neuron_summary


def remove_noise_neurons(neuron_summary, max_false_positive_percent=0.05, max_false_negative_percent=0.55):
    """ Deletes all neurons in neuron_summary list with sort quality metrics
        false_positives or false_negatives that exceed max_noise_overlap_percent.
    """

    for ind, neuron in reversed(list(enumerate(neuron_summary))):
        if neuron['sort_quality']['false_positives'] is None or neuron['sort_quality']['false_negatives'] is None:
            print("Neuron", ind, "on channel", neuron['channel'], "has no noise estimate!")
            continue
        if (neuron['sort_quality']['false_positives'] > max_false_positive_percent
            or neuron['sort_quality']['false_negatives'] > max_false_negative_percent):
            print("Deleting neuron", ind)
            del neuron_summary[ind]

    return neuron_summary


def calculate_expected_overlap(n1, n2, overlap_time):
    """ Returns the expected number of overlapping spikes between neuron 1 and
        neuron 2 within a time window 'overlap_time' assuming independent
        spiking. """

    first_index = max(n1['spike_indices'][0], n2['spike_indices'][0])
    last_index = min(n1['spike_indices'][-1], n2['spike_indices'][-1])
    n1_count = np.count_nonzero(np.logical_and(n1['spike_indices'] >= first_index, n1['spike_indices'] <= last_index))
    n2_count = np.count_nonzero(np.logical_and(n2['spike_indices'] >= first_index, n2['spike_indices'] <= last_index))
    num_ms = int(np.ceil((last_index - first_index) / (n1['sampling_rate'] / 1000)))
    if num_ms > 0:
        expected_overlap = (overlap_time * 1000 * n1_count * n2_count) / num_ms
    else:
        expected_overlap = 0

    return expected_overlap


def get_aligned_shifted_template(template_1, template_2, max_samples_window):
    """ Finds the optimal cross correlation based alignment between the two
        input templates by shifting template_2 relative to template_1.  Outputs
        template_2 shifted by zero padding to be optimally aligned with template_1
        and the shift indices needed to get there (negative values are to the left).
        The maximum shift is +/- max_samples_window // 2. """

    temp_xcorr = np.correlate(template_1, template_2, mode='same')
    xcorr_center = np.floor(template_1.size/2).astype('int')
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
    index_shifts = np.empty(spikes.shape[0], dtype='int')
    xcorr_center = np.floor(template.size/2).astype('int')
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
    neuron_labels = np.empty(0, dtype='int')
    for n in range(0, len(spike_list)):
        neuron_labels = np.hstack((neuron_labels, n * np.ones(spike_list[n].shape[0])))
        n_original[n] = spike_list[n].shape[0]
        templates[n, :] = np.nanmean(spike_list[n], axis=0)
        if np.count_nonzero(~np.any(np.isnan(spike_list[n]), axis=0)) == 0:
            # At least one neuron has no valid spikes
            return False
    neuron_labels = neuron_labels.astype('int')
    clips = np.vstack(spike_list)
    clips = clips[:, ~np.any(np.isnan(clips), axis=0)] # No nans into merge_clusters

    scores = preprocessing.compute_pca_by_channel(clips, np.arange(0, clips.shape[1]), max_components, add_peak_valley=False)
    sharpen = True if n_original.size == 2 and (np.amax(n_original) > 100 * np.amin(n_original)) else False
    neuron_labels = sort.merge_clusters(clips, neuron_labels, merge_only=False, p_value_cut_thresh=p_value_cut_thresh, min_cluster_size=1)

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


def compute_spike_trains(spike_indices_list, bin_width_samples, min_max_samples):

    if type(spike_indices_list) is not list:
        spike_indices_list = [spike_indices_list]
    if type(min_max_samples) is not list:
        min_max_samples = [0, min_max_samples]
    bin_width_samples = int(bin_width_samples)
    train_len = int(np.ceil((min_max_samples[1] - min_max_samples[0] + bin_width_samples/2) / bin_width_samples) + 1) # Add one because it's a time/samples slice
    spike_trains_list = [[] for x in range(0, len(spike_indices_list))]

    for ind, unit in enumerate(spike_indices_list):
        # Convert the spike indices to units of bin_width_samples
        spikes = (np.floor((unit + bin_width_samples/2) / bin_width_samples)).astype('int')
        spike_trains_list[ind] = np.zeros(train_len, dtype='bool')
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
    samples_axis = np.arange(-1 * samples_window, samples_window+d_samples, d_samples).astype('int')
    counts = np.zeros(samples_axis.size, dtype='int')

    # Convert the spike indices to units of d_samples
    spikes_1 = (np.floor((spikes_1 + d_samples/2) / d_samples)).astype('int')
    spikes_2 = (np.floor((spikes_2 + d_samples/2) / d_samples)).astype('int')
    samples_axis = (np.floor((samples_axis + d_samples/2) / d_samples)).astype('int')

    # Convert to spike trains
    train_len = int(max(spikes_1[-1], spikes_2[-1])) + 1 # This is samples, so need to add 1

    spike_trains_1 = np.zeros(train_len, dtype='bool')
    spike_trains_1[spikes_1] = True
    spike_trains_2 = np.zeros(train_len, dtype='bool')
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


def combine_and_delete_fragments(neuron_summary, overlap_time=5e-4, max_offset_samples=10, p_value_cut_thresh=0.01, max_components=None):
    """
        Do this first because fragments can mess up both SNR ordering and the
        mixture test below
        """

    max_samples = int(round(overlap_time * neuron_summary[0]['sampling_rate']))
    if max_components is None:
        max_components = 10

    # Create list of dictionaries of overlapping spikes and their percentage between all pairwise units
    # in each neighborhood
    neighbor_distances = [{x: 1.} for x in range(0, len(neuron_summary))]
    violation_partners = [set() for x in range(0, len(neuron_summary))]
    for n1_ind, n1 in enumerate(neuron_summary):
        for n2_ind in range(n1_ind+1, len(neuron_summary)):
            n2 = neuron_summary[n2_ind]
            if np.intersect1d(n1['neighbors'], n2['neighbors']).size == 0:
                continue
            # expected_overlap = calculate_expected_overlap(n1, n2, overlap_time)
            # actual_overlap = np.count_nonzero(find_overlapping_spike_bool(n1['spike_indices'], n2['spike_indices'], max_samples=max_samples))
            actual_overlaps = zero_symmetric_ccg(n1['spike_indices'], n2['spike_indices'], max_samples, max_samples)[0]
            neighbor_distances[n2_ind][n1_ind] = actual_overlaps[1] / n1['spike_indices'].size
            neighbor_distances[n1_ind][n2_ind] = actual_overlaps[1] / n2['spike_indices'].size
            if actual_overlaps[1] > (actual_overlaps[0] + actual_overlaps[2]):
                violation_partners[n1_ind].add(n2_ind)
                violation_partners[n2_ind].add(n1_ind)

    # return neighbor_distances, violation_partners

    # Check for fragments of good neurons and either combine or delete them
    fragments_to_check = []
    for n1 in range(0, len(neuron_summary)): # for each neuron
        for n2 in neighbor_distances[n1]: # go through its neighbor distances
            if n1 in fragments_to_check or n2 in fragments_to_check:
                continue
            max_dist = np.amax((neighbor_distances[n1][n2], neighbor_distances[n2][n1]))
            min_dist = np.amin((neighbor_distances[n1][n2], neighbor_distances[n2][n1]))
            if max_dist > 10 * min_dist:
                if neighbor_distances[n1][n2] < neighbor_distances[n2][n1]:
                    fragments_to_check.append(n1)
                else:
                    fragments_to_check.append(n2)

    neurons_to_delete = []
    neurons_to_combine = []
    for fc in fragments_to_check:
        fc_max = 0
        n_comp = None
        # Find best partner for this fragment and try to combine with it
        for n in neighbor_distances[fc]:
            fc_dist = neighbor_distances[n][fc] - neighbor_distances[fc][n]
            if fc_dist > fc_max:
                fc_max = fc_dist
                n_comp = n
        should_combine, _, neuron_summary[fc] = combine_neurons_test(neuron_summary[n_comp], neuron_summary[fc], max_samples, max_offset_samples, p_value_cut_thresh, max_components, return_new_neuron=True)
        if should_combine:
            neurons_to_combine.append([n_comp, fc])
        neurons_to_delete.append(fc) # Delete this fragment no matter what

    for ind1, ind2 in neurons_to_combine:
        print("Combining neurons", ind1, ind2)
        neuron_summary[ind1] = combine_two_neurons(neuron_summary[ind1], neuron_summary[ind2], min_ISI=overlap_time)
    # Remove lesser merged neurons
    neurons_to_delete.sort()
    for ind in reversed(neurons_to_delete):
        print("Deleting neuron", ind)
        del neuron_summary[ind]

    return neuron_summary


def remove_mixed_neurons(neuron_summary, overlap_time=2.5e-4):
    """
        The output is a new neuron summary with mixtures deleted AND reorded by each
        neuron's apparent importance for explaining the data, rather than relying
        strictly on an SNR ordering, which can cause problems. This output order can
        be used for deleting further overlaps, such that neurons to the right should
        be deleted over neurons to the left.
        This algorithm adds a neuron's spikes one at a time, with the goal being to
        create a set of maximum number of spikes.  The penalty for adding a neuron
        is its overlap with spikes already added.  So the final solution contains
        the maximum achievable number of spikes from the set of mutual violators
        that are disjoint from one another.  Each unit included in this solution
        is generally a good representative of a single unit contained in the
        mutual violators.
        """

    max_samples = int(round(overlap_time * neuron_summary[0]['sampling_rate']))
    n_total_samples = 0

    # Create list of sets of excessive neuron overlap between all pairwise units
    violation_partners = [set() for x in range(0, len(neuron_summary))]
    for n1_ind, n1 in enumerate(neuron_summary):
        violation_partners[n1_ind].add(n1_ind)
        if n1['spike_indices'][-1] > n_total_samples:
            # Find the maximum number of samples over all neurons while we are here for use later
            n_total_samples = n1['spike_indices'][-1]
        for n2_ind in range(n1_ind+1, len(neuron_summary)):
            n2 = neuron_summary[n2_ind]
            if np.intersect1d(n1['neighbors'], n2['neighbors']).size == 0:
                # Only count violations in neighborhood
                continue
            actual_overlaps = zero_symmetric_ccg(n1['spike_indices'], n2['spike_indices'], max_samples, max_samples)[0]
            if actual_overlaps[1] > (actual_overlaps[0] + actual_overlaps[2]):
                # Mark as a violation if center of CCG is greater than sum of bins on either side of it
                violation_partners[n1_ind].add(n2_ind)
                violation_partners[n2_ind].add(n1_ind)

    # First check that each neuron isn't adequately explained as a mixture of DISJOINT units
    # These are likely to be high SNR mixtures, which should be removed before doing the
    # analysis that follows as very high percentage mixtures can obstruct the correct solution
    component_units = [[] for x in range(0, len(neuron_summary))]
    best_sub_order = [[] for x in range(0, len(neuron_summary))]
    for n_ind, check_n in enumerate(neuron_summary):
        # For each neuron, consider its spikes as the full set to account for
        full_spike_train = compute_spike_trains(check_n['spike_indices'], max_samples, [0, n_total_samples])
        full_cost = np.count_nonzero(full_spike_train)
        best_total_cost = np.inf # Take any solution
        best_current_order = []

        # Make train for each other violation partner
        hit_trains = np.zeros((len(violation_partners[n_ind]) - 1, full_spike_train.size), dtype='bool')
        hit_train_index = np.zeros(len(violation_partners[n_ind]) - 1, dtype='int')
        for vp_ind, vp in enumerate(violation_partners[n_ind] - set([n_ind])):
            hit_trains[vp_ind, :] = compute_spike_trains(neuron_summary[vp]['spike_indices'], max_samples, [0, n_total_samples])
            hit_train_index[vp_ind] = vp # Need to track order since sets aren't ordered

        # Scan all of this neurons violations, with each neuron getting a turn going first
        for neuron_ind in (violation_partners[n_ind] - set([n_ind])):
            current_cost = full_cost
            neuron_spike_train = hit_trains[hit_train_index == neuron_ind, :]

            current_cost -= np.count_nonzero(np.logical_and(neuron_spike_train, full_spike_train)) # reward hits
            current_cost += np.count_nonzero(np.logical_xor(neuron_spike_train, full_spike_train)) # penalize misses AND false positives
            # Penalize spikes found in other violation partners to promote uniqueness
            current_cost += np.count_nonzero(np.logical_and(neuron_spike_train, np.any(hit_trains[hit_train_index != neuron_ind, :], axis=0)))

            remaining_train = np.logical_and(~neuron_spike_train, full_spike_train)
            subtracted_train = neuron_spike_train
            remaining_neurons = list(violation_partners[n_ind] - set([neuron_ind, n_ind]))
            neuron_order = [neuron_ind]
            while len(remaining_neurons) > 0:
                best_neuron = None
                best_cost = 0 # Take any 'improvement'
                neurons_to_check = [x for x in remaining_neurons]
                while len(neurons_to_check) > 0:
                    n2_ind = neurons_to_check.pop()
                    n2_spike_train = hit_trains[hit_train_index == n2_ind, :]
                    n2_cost = -1 * np.count_nonzero(np.logical_and(n2_spike_train, remaining_train)) # reward hits
                    n2_cost += np.count_nonzero(np.logical_and(n2_spike_train, ~remaining_train)) # penalize false positives
                     # Penalize for having spikes already subtracted by previous neurons to enforce disjointness
                    n2_cost += 1 * np.count_nonzero(np.logical_and(n2_spike_train, subtracted_train))
                    if n2_cost < best_cost:
                        best_cost = n2_cost
                        best_neuron = n2_ind
                        best_train = n2_spike_train
                if best_neuron is None:
                    break
                remaining_train = np.logical_and(~best_train, remaining_train)
                subtracted_train = np.logical_or(subtracted_train, best_train)
                neuron_order.append(best_neuron)
                remaining_neurons.remove(best_neuron)
                current_cost += best_cost
            if current_cost < best_total_cost:
                best_total_cost = current_cost
                best_current_order = neuron_order
                # Only delete if best solution can account for at least half of the spikes !
                if np.count_nonzero(subtracted_train) < .5 * neuron_summary[n_ind]['spike_indices'].size:
                    best_current_order = []
        best_sub_order[n_ind] = best_current_order

    # Now schedule any neurons that are described by more than one unit for deletion
    # at the end.  We still keep them in neuron_summary for the time being so as not
    # to mess up the indexing of neurons
    neurons_to_delete = []
    violations_to_remove = []
    for bs_ind, bs in enumerate(best_sub_order):
        if len(bs) > 1:
            violations_to_remove.append(bs_ind)
            print("Adding neuron", bs_ind, "for DISJOINT mixture deletion")
            # neurons_to_delete.append(bs_ind)
    # Remove these units from any violations so that they are not considered further
    for vp in violation_partners:
        for vr in violations_to_remove:
            if vr in vp:
                vp.remove(vr)

    # With large SNR mixtures removed, we move on to find lesser mixtures
    n2_counts = np.zeros(len(neuron_summary), dtype='int')
    already_checked = []
    output_order = []
    best_orders = []
    all_neuron_orders = []
    for neuron1_ind, neuron1 in enumerate(neuron_summary):
        if len(violation_partners[neuron1_ind]) == 1 and neuron1_ind not in neurons_to_delete:
            # If a neuron doesn't overlap with others, it is good for output
            output_order.append(neuron1_ind)
            best_orders.append([neuron1_ind])
            already_checked.append(set())
            # print("Skipping neuron", neuron1_ind, "with no violations")
            continue
        if neuron1_ind in violations_to_remove:
            # Continue ignoring these
            best_orders.append([])
            already_checked.append(set())
            continue

        # Group together all violators of neuron1 and their violators
        mutual_violations = set([neuron1_ind])
        for vp in violation_partners[neuron1_ind]:
            mutual_violations.update(violation_partners[vp])
        # Check if we have already done this group and don't repeat
        skip_current = False
        for ac_ind, ac_mv in enumerate(already_checked):
            if mutual_violations == ac_mv:
                best_orders.append(best_orders[ac_ind])
                already_checked.append(ac_mv)
                if neuron1_ind not in best_orders[ac_ind]:
                    n_overlaps = 0
                    for vp in violation_partners[neuron1_ind]:
                        if vp == neuron1_ind:
                            continue
                        if vp in best_orders[ac_ind]:
                            n_overlaps += 1
                        if n_overlaps > 1:
                            print("Adding neuron", neuron1_ind, "for mixture deletion")
                            neurons_to_delete.append(neuron1_ind)
                            break
                # # Below is a more aggressive way of removing potential mixtures
                # if neuron1_ind not in best_orders[ac_ind]:
                #     # Current neuron was not in the optimal solution so delete it
                #     print("Adding neuron", neuron1_ind, "for mixture deletion")
                #     neurons_to_delete.append(neuron1_ind)

                skip_current = True
                break
        if skip_current:
            continue

        # Make a spike train for every neuron in mutual_violations
        hit_trains = np.zeros((len(mutual_violations), full_spike_train.size), dtype='bool')
        hit_train_index = np.zeros(len(mutual_violations), dtype='int')
        spike_counts = np.zeros(len(mutual_violations), dtype='int')
        for vp_ind, vp in enumerate(mutual_violations):
            hit_trains[vp_ind, :] = compute_spike_trains(neuron_summary[vp]['spike_indices'], max_samples, [0, n_total_samples])
            hit_train_index[vp_ind] = vp # Need to track order since sets aren't ordered
            spike_counts[vp_ind] = neuron_summary[vp]['spike_indices'].size

        full_cost = np.count_nonzero(np.any(hit_trains, axis=0)) # This is maximum number of spikes
        best_total_cost = full_cost # Take any improvement
        best_neuron_order = None

        spike_order = np.argsort(spike_counts)[-1::-1]
        check_order = []
        for i_so in spike_order:
            check_order.append(hit_train_index[i_so])

        spike_intersection = np.all(hit_trains, axis=0)


        """ !!! NOTE !!!
            This needs to penalize for redundant spikes, but only so long as those spikes are not noise.
            Unique noise spikes should actually be penalized as well.  To accomplish this,
            I may need to guess what is noise, such as if a unit has a bunch of unique spikes but they are
            all small compared to the redundant spikes might imply the unit is a mixture of real neuron
            with noise, as opposed to a mixture of real neuron with other real neuron. """

        for neuron_ind in check_order: #mutual_violations: # Each neuron gets a turn going first
            current_cost = full_cost # Start cost anew each time
            neuron_spike_train = hit_trains[hit_train_index == neuron_ind, :]
            # There is no such thing as a "MISS" or a redundancy at this point
            # We also do not want to give the first neuron, which has no chance
            # of being penalized, credit at this point.  Subtracting it from the
            # cost gives a big advantage to possible mixtures, which can have
            # more spikes than big neurons without chance of penalty here!
            current_cost -= neuron_summary[neuron_ind]['spike_indices'].size * neuron_summary[neuron_ind]['sort_quality']['SNR']

            # if neuron_ind != 0:
            # unique_cost = (1 / neuron_summary[neuron_ind]['sort_quality']['SNR']) * np.count_nonzero(np.logical_and(~np.any(hit_trains[hit_train_index != neuron_ind, :], axis=0), neuron_spike_train))
            # current_cost += unique_cost # (1 / neuron_summary[neuron_ind]['sort_quality']['SNR']) * np.count_nonzero(np.logical_and(~np.any(hit_trains[hit_train_index != neuron_ind, :], axis=0), neuron_spike_train))
            # print("N unique spikes for neuron", neuron_ind, "is", unique_cost)
            # data = np.nansum(neuron_summary[neuron_ind]['waveforms'] ** 2, axis=1)
            # pval, cutpoint = sort.iso_cut(data, 1.1)

            # if neuron_ind == 2:
            #     unique_spike_index = np.logical_and(~np.any(hit_trains[hit_train_index != neuron_ind, :], axis=0), neuron_spike_train)

            remaining_neurons = list(mutual_violations - set([neuron_ind]))
            removed_train = neuron_spike_train
            removed_train = neuron_spike_train * neuron_summary[neuron_ind]['sort_quality']['SNR']


            neuron_order = [neuron_ind]
            while len(remaining_neurons) > 0:
                best_neuron = None
                best_cost = 0 # Take any improvement
                neurons_to_check = [x for x in remaining_neurons]
                while len(neurons_to_check) > 0:
                    n2_ind = neurons_to_check.pop()
                    n2_spike_train = hit_trains[hit_train_index == n2_ind, :]
                    # The best possible n2_cost is -n2 spikes.  The worst possible
                    # is +2 * n2 spikes.  50% new spikes and 50% redundant scores
                    # 0 and is excluded.  Anything better is included
                    n2_cost = 0
                    n2_cost -= neuron_summary[n2_ind]['spike_indices'].size * neuron_summary[n2_ind]['sort_quality']['SNR']
                    # n2_cost -= 1 * np.count_nonzero(np.logical_and(n2_spike_train, ~removed_train))
                    # NOTE: The factor used to penalize redundancy could be set as an input parameter
                    # Values <= 1 will include every neuron tested
                    # n2_cost += 2 * np.count_nonzero(np.logical_and(n2_spike_train, removed_train)) / neuron_summary[n2_ind]['sort_quality']['SNR'] # Penalize redundancy

                    n2_cost += 2 * np.sum(n2_spike_train * removed_train)

                    if n2_cost < best_cost:
                        best_cost = n2_cost
                        best_neuron = n2_ind
                        # best_train = n2_spike_train
                        best_train = n2_spike_train * neuron_summary[n2_ind]['sort_quality']['SNR']
                if best_neuron is None:
                    break

                n2_counts[best_neuron] += 1

                # removed_train = np.logical_or(removed_train, best_train)
                # print(removed_train.shape, best_train.shape)
                removed_train = np.amax(np.vstack((removed_train, best_train)), axis=0)
                neuron_order.append(best_neuron)
                remaining_neurons.remove(best_neuron)
                current_cost += best_cost
            # print("Current cost", current_cost, "best_total", best_total_cost)
            if current_cost < best_total_cost:
                best_total_cost = current_cost
                best_neuron_order = [x for x in neuron_order]
            all_neuron_orders.append(neuron_order)
            # print("Best order is", best_neuron_order, "all orders are", all_neuron_orders)

        if best_neuron_order is None:
            print("THIS HAS COMPLETELY FAILED???")
            best_orders.append([])
        else:
            best_orders.append(best_neuron_order)
            if neuron1_ind not in best_neuron_order:
                n_overlaps = 0
                for vp in violation_partners[neuron1_ind]:
                    if vp == neuron1_ind:
                        continue
                    if vp in best_neuron_order:
                        n_overlaps += 1
                    if n_overlaps > 1:
                        print("Adding neuron", neuron1_ind, "for mixture deletion")
                        neurons_to_delete.append(neuron1_ind)
                        break
            # # Below is a more aggressive way of removing potential mixtures
            # if neuron1_ind not in best_neuron_order:
            #     # Current neuron was not in the optimal solution so delete it
            #     print("Adding neuron", neuron1_ind, "for mixture deletion")
            #     neurons_to_delete.append(neuron1_ind)
        already_checked.append(mutual_violations)

    # print("Best orders", best_orders, "All orders", all_neuron_orders)

    # Rank neurons based on their order in the optimal solutions as this is likely
    # an indicator of their importance.
    rank_counts = np.zeros((len(neuron_summary), len(neuron_summary)), dtype='int')
    for bo in best_orders:
        for rank, value in enumerate(bo):
            rank_counts[value, rank] += 1
    # Make rank order by first adding neurons in order of first place finishes,
    # then all neurons in order of second place finishes etc.
    rank_orders = []
    for x in range(0, rank_counts.shape[1]):
        n_order = np.argsort(rank_counts[:, x])[-1::-1]
        rank_orders.extend(n_order[rank_counts[n_order, x] != 0])

    # As a sort of tiebreaker, repeat above procedure but on ALL solutions,
    # instead of just the single optimal one
    rank_counts = np.zeros((len(neuron_summary), len(neuron_summary)), dtype='int')
    for ao in all_neuron_orders:
        for rank, value in enumerate(ao):
            if rank == 0:
                continue
            rank_counts[value, rank] += 1
    for x in range(0, rank_counts.shape[1]):
        n_order = np.argsort(rank_counts[:, x])[-1::-1]
        rank_orders.extend(n_order[rank_counts[n_order, x] != 0])

    # Then add all remaining neurons (by just adding all neurons, they will be
    # ignored below if need be)
    rank_orders.extend([x for x in range(0, len(neuron_summary))])
    used = set()
    unique_rank_orders = []
    for x in rank_orders:
        if x not in used and x not in neurons_to_delete and x not in output_order:
            unique_rank_orders.append(x)
            used.add(x)

    output_order.extend(unique_rank_orders)

    print("outputting summary in order:", output_order)
    # print(best_orders)
    output_summary = []
    for ind in output_order:
        output_summary.append(neuron_summary[ind])

    return output_summary


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


def combine_stolen_spikes(neuron_summary, max_offset_samples=3, overlap_time=5e-4, p_value_cut_thresh=0.01, max_components=20):
    """ This routine attempts to identify neurons in neuron_summary that represent
        "stolen" spike waveforms and reassign them to their correct neuron.  Stolen
        spikes are those that were sorted into another cluster because a second
        neuron on a different channel tended to spike at nearly the same time
        causing the across-channel overlap of the two waveforms to produce spikes
        that were sorted into their own separate cluster.  Briefly, this is done
        by first calling the iso cut merge routine sort.merge_clusters using
        spike waveforms only within their primary channel (not using the entire
        neighborhood clips).  Any neurons that are merged in this case are
        suspect of having been separated due to spike stealing.  These units are
        investigated by asking whether a combination of the template they were
        sorted into and a template from a channel in its neighborhood are better
        able to account for their template residual error than the sharpened
        template alone using an F test with p-value threshold = p_value_combine.
        This function calls itself recursively until no more stolen spikes are
        identified.
    """

    # Get some needed info about this recording and sorting
    n_chans = 0
    max_chans_per_spike = 1
    for n in neuron_summary:
        if np.amax(n['neighbors']) > n_chans:
            n_chans = np.amax(n['neighbors'])
        if n['neighbors'].size > max_chans_per_spike:
            max_chans_per_spike = n['neighbors'].size
    n_chans += 1 # Need to add 1 since chan nums start at 0
    inds_by_chan = [[] for x in range(0, n_chans)]
    for ind, n in enumerate(neuron_summary):
        inds_by_chan[n['channel']].append(ind)

    window = [0, 0]
    window[0] = int(round(neuron_summary[0]['clip_width'][0] * neuron_summary[0]['sampling_rate']))
    window[1] = int(round(neuron_summary[0]['clip_width'][1] * neuron_summary[0]['sampling_rate'])) + 1 # Add one so that last element is included
    samples_per_chan = window[1] - window[0]
    spike_chan_inds = []
    for x in range(0, max_chans_per_spike):
        spike_chan_inds.append(np.arange(x*samples_per_chan, (x+1)*samples_per_chan))
    spike_chan_inds = np.vstack(spike_chan_inds)
    max_samples = int(round(overlap_time * neuron_summary[0]['sampling_rate']))

    neurons_to_combine = []
    neurons_to_delete = []
    neurons_used = set()
    for chan_inds in inds_by_chan:
        # Check sharpening one channel at a time
        if len(chan_inds) < 2:
            # Need more than one unit
            continue
        sharpened_away_neurons = []
        for neur_i in chan_inds:
            # For each neuron on this channel
            original_labels = neur_i*np.ones(neuron_summary[neur_i]['spike_indices'].size, dtype='int')
            clips = neuron_summary[neur_i]['waveforms']
            curr_chan_position = np.nonzero(neuron_summary[neur_i]['neighbors'] == neuron_summary[neur_i]['channel'])[0]
            curr_chan_inds = np.arange(curr_chan_position*samples_per_chan, (curr_chan_position+1)*samples_per_chan)
            for neur_j in chan_inds[neur_i+1:]:
                # Align and group each other neuron on this channel with neuron i
                original_labels = np.hstack((original_labels, neur_j*np.ones(neuron_summary[neur_j]['spike_indices'].size, dtype='int')))
                shifted_j = align_spikes_on_template(neuron_summary[neur_j]['waveforms'], neuron_summary[neur_i]['template'], 2*max_offset_samples)[0]

                # Attempt to sharpen these clusters into each other using only single channel waveforms
                clips = np.vstack((clips, shifted_j))
                sharp_labels = np.copy(original_labels)
                scores = preprocessing.compute_pca_by_channel(clips[:, curr_chan_inds], np.arange(0, curr_chan_inds.size), max_components, add_peak_valley=True)
                # Below merges labels into the larger cluster, so assumption here is that stolen spikes are less numerous
                sharp_labels = sort.merge_clusters(scores, sharp_labels, p_value_cut_thresh=p_value_cut_thresh, min_cluster_size=1)

                # Determine if a merge happened and to which neuron
                unique_originals = np.unique(original_labels)
                unique_sharp = np.unique(sharp_labels)
                for nl in unique_originals:
                    if nl not in unique_sharp:
                        # Find neuron that this unit was sharpened into, and add it to its corresponding list of sharpened_into_neurons
                        sharp_index = original_labels == nl
                        nl_index = [i for i in range(0, len(sharpened_away_neurons)) if sharpened_away_neurons[i][0] == nl]
                        if len(nl_index) == 0:
                            sharpened_away_neurons.append([nl, set(np.unique(sharp_labels[sharp_index]))])
                        else:
                            sharpened_away_neurons[nl_index[0]][1].update(np.unique(sharp_labels[sharp_index]))
        if len(sharpened_away_neurons) == 0:
            continue

        # Now go through every neuron that was sharpened away
        for sharpened_pair in sharpened_away_neurons:
            if sharpened_pair[0] in neurons_used:
                continue
            stolen_unit = neuron_summary[sharpened_pair[0]]
            best_p_value = 1.
            # Start with neuron(s) that stolen_unit was sharpened into and align them
            for neuron1_ind in sharpened_pair[1]:
                if neuron1_ind in neurons_used:
                    continue
                neuron1 = neuron_summary[neuron1_ind]
                shifted_n1 = get_aligned_shifted_template(stolen_unit['template'], neuron1['template'], 2*max_offset_samples)[0]
                adjusted_sharp_template = stolen_unit['template'] - shifted_n1

                if np.sum(adjusted_sharp_template ** 2) > 0.75* np.sum(stolen_unit['template'] ** 2):
                    # neuron1 template decreases stolen unit templat barely or not at all
                    continue

                # Now check neighboring neurons that are NOT on the same channel
                for neighbor in stolen_unit['neighbors']: # each neighboring channel
                    if neighbor == stolen_unit['channel']: # but NOT on SAME channel !
                        continue
                    for neuron2_ind in inds_by_chan[neighbor]: # each neuron on neighboring channel EXCEPT current
                        neuron2 = neuron_summary[neuron2_ind]
                        if neuron2_ind in neurons_used:
                            # Do not check combinations with already deleted/stolen units
                            continue
                        # Restrict comparisons to shared channels within neighborhood
                        overlap_chans, common1, common2 = np.intersect1d(neuron1['neighbors'], neuron2['neighbors'], return_indices=True)
                        if neuron1['channel'] not in overlap_chans or neuron2['channel'] not in overlap_chans:
                            # Do not look at overlapping channels that don't contain the neuron's primary channel
                            continue

                        # Compute indices into templates at overlapping channel points and align neuron2 with adjusted
                        samples_ind_n1 = spike_chan_inds[common1, :].flatten()
                        samples_ind_n2 = spike_chan_inds[common2, :].flatten()
                        shifted_n2 = get_aligned_shifted_template(adjusted_sharp_template[samples_ind_n1], neuron2['template'][samples_ind_n2], samples_per_chan)[0]

                        # Perfrom nested F test comparing neuron1 alone fit to stolen_unit with neuron1 + neuron2 fit
                        df_n = 1 # Numerator degress of freedom is one "parameter" template
                        df_d = samples_ind_n2.size - 2 -1 -1 # Denominator df is number of samples - 2 "parameter" templates - 1 "paramter" template - 1
                        SS_neuron_1_fit = np.sum((adjusted_sharp_template[samples_ind_n1]) ** 2)
                        SS_neuron_1_2_fit = np.sum((adjusted_sharp_template[samples_ind_n1] - (shifted_n2)) ** 2)
                        F = ((SS_neuron_1_fit - SS_neuron_1_2_fit) / df_n) / (SS_neuron_1_2_fit / (df_d))
                        p_value = 1 - stats.f.cdf(F, df_n, df_d)

                        if p_value < best_p_value:
                            # degrees of freedom can vary with number of overlapping points in neighborhood template so check p_values
                            # to track which combination accounts for the most
                            best_p_value = p_value
                            best_n_1 = neuron1_ind
                            best_n_2 = neuron2_ind
                            # Use n1 samples because it is on same channel as stolen_unit
                            best_n_1_samples_ind = samples_ind_n1
                            best_n_2_shift_template = shifted_n2

            if best_p_value < 0.05:
                # Subtract best neuron2 template match from sharpened neuron spikes then combine with neuron1
                print("Testing these", best_n_1, sharpened_pair[0])
                # Reset stolen_unit values after subtracting spike stealer template
                stolen_unit['waveforms'][:, best_n_1_samples_ind] -= best_n_2_shift_template
                # stolen_unit['waveforms'] = align_spikes_on_template(stolen_unit['waveforms'], neuron_summary[best_n_1]['template'], 2*max_offset_samples)[0]
                should_combine, _, neuron_summary[sharpened_pair[0]] = combine_neurons_test(neuron_summary[best_n_1], stolen_unit, max_samples,
                        max_offset_samples, p_value_cut_thresh, max_components, return_new_neuron=True)
                # Schedule for combining and deleting accordingly
                if should_combine:
                    print("Neuron", sharpened_pair[0], "was stolen from neuron", best_n_1, "by neuron", best_n_2, "!!!")
                    neurons_to_combine.append([best_n_1, sharpened_pair[0]])
                    neurons_to_delete.append(sharpened_pair[0])
                # neurons_used.update([best_n_1, sharpened_pair[0]])

    for ind1, ind2 in neurons_to_combine:
        print("Combining neurons", ind1, ind2)
        neuron_summary[ind1] = combine_two_neurons(neuron_summary[ind1], neuron_summary[ind2], min_ISI=np.amax(neuron_summary[ind1]['clip_width']))

    # Remove stolen neurons
    neurons_to_delete.sort()
    for ind in reversed(neurons_to_delete):
        print("Deleting neuron", ind)
        del neuron_summary[ind]

    if len(neurons_to_delete) > 0:
        # Recurse until no more changes are made
        neuron_summary = combine_stolen_spikes(neuron_summary, max_offset_samples=max_offset_samples, overlap_time=overlap_time, p_value_cut_thresh=p_value_cut_thresh, max_components=max_components)

    return neuron_summary


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
