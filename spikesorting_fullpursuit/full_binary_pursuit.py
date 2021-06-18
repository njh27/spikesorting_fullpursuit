import numpy as np
import pickle
import os
from copy import copy, deepcopy
from scipy.stats import norm
from spikesorting_fullpursuit import neuron_separability
from spikesorting_fullpursuit.consolidate import SegSummary
from spikesorting_fullpursuit.parallel.segment_parallel import get_multichannel_clips, time_window_to_samples
from spikesorting_fullpursuit.parallel import binary_pursuit_parallel
from spikesorting_fullpursuit.c_cython import sort_cython
from spikesorting_fullpursuit.utils.parallel_funs import noise_covariance_parallel



def get_binary_pursuit_clip_width(seg_w_items, clips_dict, voltage, data_dict, sort_info):
    """ Determines a clip width to use for binary pursuit by asking how much
    the current clip width must be increased so that it returns near a median
    voltage of 0 at both ends of the clip width. This is constrained by the
    sorting parameter max_binary_pursuit_clip_width_factor."""
    # Maximim factor by which the clip width can be increased
    if sort_info['max_binary_pursuit_clip_width_factor'] <= 1.0:
        # Do not use expanded clip widths, just return
        original_clip_starts = np.arange(0, sort_info['n_samples_per_chan']*(sort_info['n_channels']), sort_info['n_samples_per_chan'], dtype=np.int64)
        original_clip_stops = np.arange(sort_info['n_samples_per_chan'], (sort_info['n_samples_per_chan']+1)*sort_info['n_channels'], sort_info['n_samples_per_chan'], dtype=np.int64)
        return sort_info['clip_width'], original_clip_starts, original_clip_stops
    # Start by building the set of all clips for all units in segment
    all_events = []
    for w_item in seg_w_items:
        if w_item['ID'] in data_dict['results_dict'].keys():
            if len(data_dict['results_dict'][w_item['ID']][0]) == 0:
                # This work item found nothing (or raised an exception)
                continue
            all_events.append(data_dict['results_dict'][w_item['ID']][0])
    if len(all_events) == 0:
        # No events found so just return input clip width
        original_clip_starts = np.arange(0, sort_info['n_samples_per_chan']*(sort_info['n_channels']), sort_info['n_samples_per_chan'], dtype=np.int64)
        original_clip_stops = np.arange(sort_info['n_samples_per_chan'], (sort_info['n_samples_per_chan']+1)*sort_info['n_channels'], sort_info['n_samples_per_chan'], dtype=np.int64)
        return sort_info['clip_width'], original_clip_starts, original_clip_stops

    all_events = np.hstack(all_events)
    all_events.sort() # Must be sorted for get multichannel clips to work
    # Find the average clip for our max output clip width, double the original
    bp_clip_width = [sort_info['max_binary_pursuit_clip_width_factor']*v for v in sort_info['clip_width']]
    all_clips, valid_event_indices = get_multichannel_clips(clips_dict, voltage,
                                        all_events, clip_width=bp_clip_width)
    if np.count_nonzero(valid_event_indices) == 0:
        original_clip_starts = np.arange(0, sort_info['n_samples_per_chan']*(sort_info['n_channels']), sort_info['n_samples_per_chan'], dtype=np.int64)
        original_clip_stops = np.arange(sort_info['n_samples_per_chan'], (sort_info['n_samples_per_chan']+1)*sort_info['n_channels'], sort_info['n_samples_per_chan'], dtype=np.int64)
        return sort_info['clip_width'], original_clip_starts, original_clip_stops
    mean_clip = np.mean(all_clips, axis=0)
    bp_samples_per_chan = all_clips.shape[1] // sort_info['n_channels']
    first_indices = np.arange(0, bp_samples_per_chan*(sort_info['n_channels']-1)+1, bp_samples_per_chan, dtype=np.int64)
    last_indices = np.arange(bp_samples_per_chan-1, bp_samples_per_chan*sort_info['n_channels']+1, bp_samples_per_chan, dtype=np.int64)

    # Randomly choose 10 seconds worth of time points
    noise_sample_inds = np.random.choice(voltage.shape[1], 10*sort_info['sampling_rate'])
    median_noise = np.median(np.median(np.abs(voltage[:, noise_sample_inds]), axis=1))
    clip_end_tolerance = 0.05 * median_noise

    bp_chan_win_samples, _ = time_window_to_samples(bp_clip_width, sort_info['sampling_rate'])
    chan_win_samples, _ = time_window_to_samples(sort_info['clip_width'], sort_info['sampling_rate'])

    # Find the most we can increase the first indices to
    # chan_win_samples[0] is negative, we want positve here
    max_pre_samples = -1*bp_chan_win_samples[0] + chan_win_samples[0] # Don't shrink past original
    while np.all(np.abs(mean_clip[first_indices]) < clip_end_tolerance):
        if first_indices[0] >= max_pre_samples:
            break
        first_indices += 1

    # This is what's left of bp_chan_win_samples after we moved
    bp_clip_width[0] = -1 * (-1 * bp_chan_win_samples[0] - first_indices[0]) / sort_info['sampling_rate']

    # Most we can decrease the last indices to
    min_post_samples = (bp_samples_per_chan - bp_chan_win_samples[1]) + chan_win_samples[1] -1 # Don't shrink past original
    while np.all(np.abs(mean_clip[last_indices]) < clip_end_tolerance):
        if last_indices[0] <= min_post_samples:
            break
        last_indices -= 1
    bp_clip_width[1] = (bp_chan_win_samples[1] - (bp_samples_per_chan - last_indices[0])) / sort_info['sampling_rate']

    # Compute the indices required to slice the new bp_clip_width clips back to
    # their original input sort_info['clip_width'] size
    clip_start_ind = (-1*bp_chan_win_samples[0] + chan_win_samples[0]) - first_indices[0]
    clip_stop_ind = clip_start_ind + (chan_win_samples[1] - chan_win_samples[0])
    clip_n = last_indices[0] - first_indices[0] # New expanded clip width
    original_clip_starts = np.arange(clip_start_ind, clip_n*(sort_info['n_channels']), clip_n, dtype=np.int64)
    original_clip_stops = np.arange(clip_stop_ind, (clip_n+1)*sort_info['n_channels'], clip_n, dtype=np.int64)

    return bp_clip_width, original_clip_starts, original_clip_stops


def full_binary_pursuit(work_items, data_dict, seg_number,
                        sort_info, v_dtype, overlap_ratio_threshold,
                        absolute_refractory_period,
                        kernels_path=None, max_gpu_memory=None):
    """ This is the main function that runs binary pursuit. It first handles
    the unit and template consolidation and the removal of noise templates to
    create the final template set for binary pursuit derived from the input
    segment sorted data. Then the templates are input to binary pursuit. Output
    is finally formatted for final output. """
    # Get numpy view of voltage for clips and binary pursuit
    seg_volts_buffer = data_dict['segment_voltages'][seg_number][0]
    seg_volts_shape = data_dict['segment_voltages'][seg_number][1]
    voltage = np.frombuffer(seg_volts_buffer, dtype=v_dtype).reshape(seg_volts_shape)
    original_clip_width = [s for s in sort_info['clip_width']]
    original_n_samples_per_chan = copy(sort_info['n_samples_per_chan'])
    # Max shift indices to check for binary pursuit overlaps
    n_max_shift_inds = (original_n_samples_per_chan-1)

    # Determine the set of work items for this segment
    seg_w_items = [w for w in work_items if w['seg_number'] == seg_number]

    # Make a dictionary with all info needed for get_multichannel_clips
    clips_dict = {'sampling_rate': sort_info['sampling_rate'],
                  'n_samples': seg_w_items[0]['n_samples'],
                  'v_dtype': v_dtype}

    # Need to build this in format used for consolidate functions
    seg_data = []
    original_neighbors = []
    print("!!!!!!USING ACTUAL NEIGHBORS LINE 127 FULL BINARY PURSUIT")
    for w_item in seg_w_items:
        if w_item['ID'] in data_dict['results_dict'].keys():
            # Reset neighbors to all channels for full binary pursuit
            original_neighbors.append(w_item['neighbors'])
            # w_item['neighbors'] = np.arange(0, voltage.shape[0], dtype=np.int64)

            if len(data_dict['results_dict'][w_item['ID']][0]) == 0:
                # This work item found nothing (or raised an exception)
                seg_data.append([[], [], [], [], w_item['ID']])
                continue
            clips, _ = get_multichannel_clips(clips_dict, voltage,
                                    data_dict['results_dict'][w_item['ID']][0],
                                    clip_width=sort_info['clip_width'])

            # Insert list of crossings, labels, clips, binary pursuit spikes
            seg_data.append([data_dict['results_dict'][w_item['ID']][0],
                              data_dict['results_dict'][w_item['ID']][1],
                              clips,
                              np.zeros(len(data_dict['results_dict'][w_item['ID']][0]), dtype=np.bool),
                              w_item['ID']])
            if type(seg_data[-1][0][0]) == np.ndarray:
                if seg_data[-1][0][0].size > 0:
                    # Adjust crossings for segment start time
                    seg_data[-1][0][0] += w_item['index_window'][0]
        else:
            # This work item found nothing (or raised an exception)
            seg_data.append([[], [], [], [], w_item['ID']])

    # Pass a copy of current state of sort info to seg_summary. Actual sort_info
    # will be altered later but SegSummary must follow original data
    seg_summary = SegSummary(seg_data, seg_w_items, deepcopy(sort_info), v_dtype,
                        absolute_refractory_period=absolute_refractory_period,
                        verbose=False)
    if len(seg_summary.summaries) == 0:
        print("Found no neuron templates for binary pursuit")
        return [[[], [], [], [], None]]

    print("Entered with", len(seg_summary.summaries), "templates in segment", seg_number)

    # Need this chan_win before assigning binary pursuit clip width. Used for
    # find_overlap_templates
    chan_win, clip_width = time_window_to_samples(sort_info['clip_width'], sort_info['sampling_rate'])

    # Reassign binary pursuit clip width to clip width
    # (This is slightly confusing but keeps certain code compatability.
    # We will reset it to original value at the end.)
    sort_info['clip_width'], original_clip_starts, original_clip_stops = \
                get_binary_pursuit_clip_width(seg_w_items, clips_dict, voltage, data_dict, sort_info)
    # Store newly assigned binary pursuit clip width for final output
    if 'binary_pursuit_clip_width' not in sort_info:
        sort_info['binary_pursuit_clip_width'] = [0, 0]
    sort_info['binary_pursuit_clip_width'][0] = min(sort_info['clip_width'][0], sort_info['binary_pursuit_clip_width'][0])
    sort_info['binary_pursuit_clip_width'][1] = max(sort_info['clip_width'][1], sort_info['binary_pursuit_clip_width'][1])
    bp_chan_win, _ = time_window_to_samples(sort_info['clip_width'], sort_info['sampling_rate'])
    sort_info['n_samples_per_chan'] = bp_chan_win[1] - bp_chan_win[0]
    # This should be same as input samples per chan but could probably
    # be off by one due to rounding error of the clip width so
    # need to recompute
    bp_reduction_samples_per_chan = original_clip_stops[0] - original_clip_starts[0]
    if bp_reduction_samples_per_chan != original_n_samples_per_chan:
        # This should be coded so this never happens, but if it does it could be a difficult to notice disaster during consolidate
        raise RuntimeError("Template reduction from binary pursuit does not have the same number of samples as original!")
    print("Binary pursuit clip width is", sort_info['clip_width'], "from", original_clip_width)
    print("Binary pursuit samples per chan", sort_info['n_samples_per_chan'], "from", original_n_samples_per_chan)

    # Get the noise covariance over time within the binary pursuit clip width
    print("Computing clip noise covariance for each channel with", sort_info['n_cov_samples'], "clip samples")
    # Inputing rand_state as the current state should ensure that this function
    # stays on the same current random generator state such that starting
    # sorting at a given state will produce the same covariance matrix
    chan_covariance_mats = noise_covariance_parallel(voltage, bp_chan_win,
                                            sort_info['n_cov_samples'],
                                            rand_state=np.random.get_state())
    seg_summary.sharpen_across_chans()
    print("Sharpening reduced number of templates to", len(seg_summary.summaries))
    print("Checking", len(seg_summary.summaries), "neurons for potential sums")
    templates = []
    n_template_spikes = []
    print("COMPUTING TEMPLATE VARIANCES AT LINE 202 OF FULL BINARY PURSUIT")
    print("!!!! ZEROING OUT NON NEIGHBORHOOD CHANNELS OF CLIPS/TEMPLATES (line ~209 full binary pursuit) !!!!")
    template_covar = []
    for n in seg_summary.summaries:
        clips, _ = get_multichannel_clips(clips_dict, voltage,
                                n['spike_indices'],
                                clip_width=sort_info['clip_width'])
        for chan in range(0, sort_info['n_channels']):
            if chan not in n['neighbors']:
                c_win = [chan * sort_info['n_samples_per_chan'], (chan+1) * sort_info['n_samples_per_chan']]
                clips[:, c_win[0]:c_win[1]] = 0.0

        templates.append(np.mean(clips, axis=0))
        n_template_spikes.append(n['spike_indices'].shape[0])
        n_cov_samples = max(sort_info['n_cov_samples'], clips.shape[0])
        cov_sample_inds = np.random.randint(0, clips.shape[0], n_cov_samples)
        template_covar.append(np.cov(clips[cov_sample_inds, :], rowvar=False, ddof=0))

    templates = np.vstack(templates)
    n_template_spikes = np.array(n_template_spikes, dtype=np.int64)

    # The overlap check input here is hard coded to look at shifts +/- half
    # clip width
    templates_to_check = sort_cython.find_overlap_templates(templates,
                                sort_info['n_samples_per_chan'],
                                sort_info['n_channels'],
                                np.int64(np.abs(bp_chan_win[0])//1.5),
                                np.int64(np.abs(bp_chan_win[1])//1.5),
                                n_template_spikes)

    # Go through suspect templates in templates_to_check
    templates_to_delete = np.zeros(templates.shape[0], dtype=np.bool)
    # Use the sigma lower bound to decide the acceptable level of
    # misclassification between template sums
    confusion_threshold = norm.sf(sort_info['sigma_bp_noise'])
    for t_info in templates_to_check:
        # templates_to_check is not length of templates so need to find the
        # correct index of the template being checked
        t_ind = t_info[0]
        shift_temp = t_info[1]
        # sum_ind_1, sum_ind_2 = t_info[2]
        # p_confusion = neuron_separability.check_template_pair(
        #         templates[t_ind, :], shift_temp, chan_covariance_mats, sort_info)
        p_confusion = neuron_separability.check_template_pair_template(
                        templates[t_ind, :], shift_temp, template_covar[t_ind])
        print("P confusion", p_confusion, "Confusion threshold", confusion_threshold)
        if p_confusion > confusion_threshold:
            templates_to_delete[t_ind] = True

    # Remove these overlap templates from summary before sharpening
    for x in reversed(range(0, len(seg_summary.summaries))):
        if templates_to_delete[x]:
            del seg_summary.summaries[x]
            del template_covar[x]
    print("Removing sums reduced number of templates to", len(seg_summary.summaries))

    # Get updated neurons after removing overlap templates
    neurons = seg_summary.summaries

    # Return the original neighbors to the work items that were reset
    orig_neigh_ind = 0
    for w_item in seg_w_items:
        if w_item['ID'] in data_dict['results_dict'].keys():
            w_item['neighbors'] = original_neighbors[orig_neigh_ind]
            orig_neigh_ind += 1

    if len(neurons) == 0:
        # All data this segment found nothing (or raised an exception)
        seg_data = []
        for chan in range(0, sort_info['n_channels']):
            curr_item = None
            for w_item in seg_w_items:
                if w_item['channel'] == chan:
                    curr_item = w_item
                    break
            if curr_item is None:
                # This should never be possible, but just to be sure
                raise RuntimeError("Could not find a matching work item for unit")
            seg_data.append([[], [], [], [], curr_item['ID']])
        return seg_data

    # Get templates for remaining sharpened units across all channels in
    # voltage to input to separability metrics
    templates = []
    print("!!! ZEROING OUT NON NEIGHBOR CHANNELS FOR TEMPLATES LINE 284 full binary pursuit !!!")
    for n in neurons:
        if not n['deleted_as_redundant']:
            clips, _ = get_multichannel_clips(clips_dict, voltage,
                                    n['spike_indices'],
                                    clip_width=sort_info['clip_width'])

            for chan in range(0, sort_info['n_channels']):
                if chan not in n['neighbors']:
                    c_win = [chan * sort_info['n_samples_per_chan'], (chan+1) * sort_info['n_samples_per_chan']]
                    clips[:, c_win[0]:c_win[1]] = 0.0

            templates.append(np.mean(clips, axis=0))
    del seg_summary # No longer needed so clear memory

    separability_metrics = neuron_separability.compute_separability_metrics(
                                templates, chan_covariance_mats, sort_info, template_covar)
    # Identify templates similar to noise and decide what to do with them
    noisy_templates = neuron_separability.find_noisy_templates(
                                            separability_metrics, sort_info)
    separability_metrics = neuron_separability.set_bp_threshold(separability_metrics)
    separability_metrics, noisy_templates = neuron_separability.check_noise_templates(
                                    separability_metrics, sort_info, noisy_templates)
    separability_metrics = neuron_separability.delete_noise_units(
                                    separability_metrics, noisy_templates)


    if separability_metrics['templates'].shape[0] == 0:
        # All data this segment found nothing (or raised an exception)
        seg_data = []
        for chan in range(0, sort_info['n_channels']):
            curr_item = None
            for w_item in seg_w_items:
                if w_item['channel'] == chan:
                    curr_item = w_item
                    break
            if curr_item is None:
                # This should never be possible, but just to be sure
                raise RuntimeError("Could not find a matching work item for unit")
            seg_data.append([[], [], [], [], curr_item['ID']])
        return seg_data

    # print("!!!!!!! CHECKING FOR TOO MANY TEMPLATES !!!!!!!!")
    # if separability_metrics['templates'].shape[0] > 2:
    #     raise RuntimeError("TOO MANY TEMPLATES")
    print("Starting full binary pursuit search with", separability_metrics['templates'].shape[0], "templates in segment", seg_number)
    crossings, neuron_labels, bp_bool, clips = binary_pursuit_parallel.binary_pursuit(
                    voltage, v_dtype, sort_info,
                    separability_metrics,
                    n_max_shift_inds=n_max_shift_inds,
                    kernels_path=None, max_gpu_memory=max_gpu_memory)

    if not sort_info['get_adjusted_clips']:
        clips, _ = get_multichannel_clips(clips_dict, voltage,
                                crossings, clip_width=sort_info['clip_width'])

    # Save the separability metrics as used (and output) by binary_pursuit
    sort_info['separability_metrics'][seg_number] = separability_metrics

    chans_to_template_labels = {}
    for chan in range(0, sort_info['n_channels']):
        chans_to_template_labels[chan] = []
    for unit in np.unique(neuron_labels):
        # Find this unit's channel as the channel with max SNR of template
        curr_template = np.mean(clips[neuron_labels == unit, :], axis=0)
        unit_best_snr = -1.0
        unit_best_chan = None
        for chan in range(0, sort_info['n_channels']):
            background_noise_std = seg_w_items[0]['thresholds'][chan] / sort_info['sigma']
            chan_win = [sort_info['n_samples_per_chan'] * chan,
                          sort_info['n_samples_per_chan'] * (chan + 1)]
            chan_template = curr_template[chan_win[0]:chan_win[1]]
            temp_range = np.amax(chan_template) - np.amin(chan_template)
            chan_snr = temp_range / (3 * background_noise_std)
            if chan_snr > unit_best_snr:
                unit_best_snr = chan_snr
                unit_best_chan = chan
        chans_to_template_labels[unit_best_chan].append(unit)

    # Set these back to match input values
    sort_info['clip_width'] = original_clip_width
    sort_info['n_samples_per_chan'] = bp_reduction_samples_per_chan

    # Need to convert binary pursuit output to standard sorting output. This
    # requires data from every channel, even if it is just empty
    seg_data = []
    for chan in range(0, sort_info['n_channels']):
        curr_item = None
        for w_item in seg_w_items:
            if w_item['channel'] == chan:
                curr_item = w_item
                break
        if curr_item is None:
            # This should never be possible, but just to be sure
            raise RuntimeError("Could not find a matching work item for unit")
        if len(chans_to_template_labels[chan]) > 0:
            # Set data to empty defaults and append if they exist
            chan_events, chan_labels, chan_bp_bool, chan_clips = [], [], [], []
            for unit in chans_to_template_labels[chan]:
                select = neuron_labels == unit
                chan_events.append(crossings[select])
                chan_labels.append(neuron_labels[select])
                chan_bp_bool.append(bp_bool[select])


                # Get clips for this unit over all channels
                unit_clips = np.zeros((np.count_nonzero(select),
                                       curr_item['neighbors'].shape[0] * \
                                       sort_info['n_samples_per_chan']),
                                       dtype=v_dtype)
                # Map clips from all channels to current channel neighborhood
                for neigh in range(0, curr_item['neighbors'].shape[0]):
                    chan_ind = curr_item['neighbors'][neigh]
                    unit_clips[:, neigh*sort_info['n_samples_per_chan']:(neigh+1)*sort_info['n_samples_per_chan']] = \
                            clips[select, original_clip_starts[chan_ind]:original_clip_stops[chan_ind]]
                chan_clips.append(unit_clips)

            # Adjust crossings for seg start time
            chan_events = np.hstack(chan_events)
            chan_events += curr_item['index_window'][0]
            # Append list of crossings, labels, clips, binary pursuit spikes
            seg_data.append([chan_events,
                             np.hstack(chan_labels),
                             np.vstack(chan_clips),
                             np.hstack(chan_bp_bool),
                             curr_item['ID']])
        else:
            # This work item found nothing (or raised an exception)
            seg_data.append([[], [], [], [], curr_item['ID']])

    return seg_data
