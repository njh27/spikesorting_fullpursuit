import numpy as np
import pickle
import os
from copy import copy
from fbp.src.consolidate import SegSummary
from fbp.src.parallel.segment_parallel import get_multichannel_clips, time_window_to_samples
from fbp.src.parallel import binary_pursuit_parallel
from fbp.src.c_cython import sort_cython



import matplotlib.pyplot as plt



def remove_overlap_templates(templates, n_samples_per_chan, n_chans,
                            n_pre_inds, n_post_inds, n_template_spikes):
    """ Use this for testing. There is a corresponding cython verison that should
        be faster. """

    def get_shifted_template(template, shift):
        """ """
        shifted_template = np.zeros_like(template)
        for chan in range(0, n_chans):
            chan_temp = template[chan*n_samples_per_chan:(chan+1)*n_samples_per_chan]
            if shift > 0:
                shifted_template[chan*n_samples_per_chan+shift:(chan+1)*n_samples_per_chan] = \
                                chan_temp[:-shift]
            else:
                shifted_template[chan*n_samples_per_chan:(chan+1)*n_samples_per_chan+shift] = \
                                chan_temp[-shift:]
        return shifted_template


    templates_to_delete = np.zeros(templates.shape[0], dtype=np.bool)
    if templates.shape[0] < 3:
        # Need at least 3 templates for one to be sum of two others
        return templates_to_delete
    templates_SS = np.sum(templates ** 2, axis=1)
    sum_n1_n2_template = np.zeros(templates.shape[1])

    # Compute the possible template shifts up front so not iteratively repeating
    # the same computation over and over again
    all_template_shifts = []
    for t in range(0, templates.shape[0]):
        all_shifts = np.zeros((n_post_inds+1 + n_pre_inds, templates.shape[1]))
        as_ind = 0
        for s in range(-n_pre_inds, n_post_inds+1):
            all_shifts[as_ind, :] = get_shifted_template(templates[t, :], s)
            as_ind += 1
        all_template_shifts.append(all_shifts)

    for test_unit in range(0, templates.shape[0]):
        test_template = templates[test_unit, :]

        min_residual_SS = np.inf
        best_pair = None
        best_shifts = None
        best_inds = None

        for n1 in range(0, templates.shape[0]):
            if (n_template_spikes[n1] < 5*n_template_spikes[test_unit]) or (n1 == test_unit) or (templates_SS[n1] > templates_SS[test_unit]):
                continue
            template_1 = templates[n1, :]

            for n2 in range(n1+1, templates.shape[0]):
                if (n_template_spikes[n2] < 5*n_template_spikes[test_unit]) or (n2 == test_unit) or (templates_SS[n2] > templates_SS[test_unit]):
                    continue
                template_2 = templates[n2, :]

                s1_ind = 0
                for shift1 in range(-n_pre_inds, n_post_inds+1):
                    shifted_t1 = all_template_shifts[n1][s1_ind, :]
                    s2_ind = 0
                    for shift2 in range(-n_pre_inds, n_post_inds+1):
                        # Copy data from t1 shift into sum
                        sum_n1_n2_template[:] = shifted_t1[:]
                        shifted_t2 = all_template_shifts[n2][s2_ind, :]
                        sum_n1_n2_template += shifted_t2

                        residual_SS = np.sum((test_template - sum_n1_n2_template) ** 2)
                        if residual_SS < min_residual_SS:
                            min_residual_SS = residual_SS
                            best_pair = [n1, n2]
                            best_shifts = [shift1, shift2]
                            best_inds = [s1_ind, s2_ind]

                        s2_ind += 1
                    s1_ind += 1
        # print("MIN RESIDUAL", min_residual_SS)
        if 1 - (min_residual_SS / templates_SS[test_unit]) > 0.85:
            templates_to_delete[test_unit] = True
            # print("Deleting this template")
            # plt.plot(templates[test_unit])
            # plt.show()
            # print("As a sum of these tempalates at shifts", best_shifts)
            # plt.plot(all_template_shifts[best_pair[0]][best_inds[0], :])
            # plt.plot(all_template_shifts[best_pair[1]][best_inds[1], :])
            # plt.show()

    return templates_to_delete


def get_binary_pursuit_clip_width(seg_w_items, clips_dict, voltage, data_dict, sort_info):
    """"""
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
    print("clip tolerance is", clip_end_tolerance)

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
                        sort_info, v_dtype, overlap_ratio_threshold=2,
                        absolute_refractory_period=12e-4,
                        kernels_path=None, max_gpu_memory=None):

    # Get numpy view of voltage for clips and binary pursuit
    seg_volts_buffer = data_dict['segment_voltages'][seg_number][0]
    seg_volts_shape = data_dict['segment_voltages'][seg_number][1]
    voltage = np.frombuffer(seg_volts_buffer, dtype=v_dtype).reshape(seg_volts_shape)
    original_clip_width = [s for s in sort_info['clip_width']]
    original_n_samples_per_chan = copy(sort_info['n_samples_per_chan'])

    # Determine the set of work items for this segment
    seg_w_items = [w for w in work_items if w['seg_number'] == seg_number]

    # Make a dictionary with all info needed for get_multichannel_clips
    clips_dict = {'sampling_rate': sort_info['sampling_rate'],
                  'n_samples': seg_w_items[0]['n_samples'],
                  'v_dtype': v_dtype}

    # Need to build this in format used for consolidate functions
    seg_data = []
    original_neighbors = []
    for w_item in seg_w_items:
        if w_item['ID'] in data_dict['results_dict'].keys():
            # Reset neighbors to all channels for full binary pursuit
            original_neighbors.append(w_item['neighbors'])
            w_item['neighbors'] = np.arange(0, voltage.shape[0], dtype=np.int64)

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
            # I am not sure why, but this has to be added here. It does not work
            # when done above directly on the global data_dict elements
            if type(seg_data[-1][0][0]) == np.ndarray:
                if seg_data[-1][0][0].size > 0:
                    # Adjust crossings for segment start time
                    seg_data[-1][0][0] += w_item['index_window'][0]
        else:
            # This work item found nothing (or raised an exception)
            seg_data.append([[], [], [], [], w_item['ID']])

    seg_summary = SegSummary(seg_data, seg_w_items, sort_info, v_dtype,
                        absolute_refractory_period=absolute_refractory_period,
                        verbose=False)
    if len(seg_summary.summaries) == 0:
        print("Found no neuron templates for binary pursuit")
        return [[[], [], [], [], None]]

    print("Entered with", len(seg_summary.summaries), "templates in segment", seg_number)
    # for n in seg_summary.summaries:
    #     plt.plot(n['pursuit_template'])
    #     plt.show()

    # print("SKIPPING SUM TEMPLATES CHECK BECAUSE ITS BROKEN")
    print("Checking", len(seg_summary.summaries), "neurons for potential sums")
    templates = []
    n_template_spikes = []
    for n in seg_summary.summaries:
        templates.append(n['pursuit_template'])
        n_template_spikes.append(n['spike_indices'].shape[0])

    chan_win, clip_width = time_window_to_samples(sort_info['clip_width'], sort_info['sampling_rate'])
    templates = np.float32(np.vstack(templates))
    n_template_spikes = np.array(n_template_spikes, dtype=np.int64)
    templates_to_delete = sort_cython.remove_overlap_templates(templates, int(sort_info['n_samples_per_chan']),
                                int(sort_info['n_channels']),
                                np.int64(np.abs(chan_win[0])), np.int64(np.abs(chan_win[1])),
                                n_template_spikes)

    # Remove these redundant templates from summary before sharpening
    for x in reversed(range(0, len(seg_summary.summaries))):
        if templates_to_delete[x]:
            del seg_summary.summaries[x]
    # print("TEMPLATE REDUCTION IS OFF !!!!!")
    print("Removing sums reduced number of templates to", len(seg_summary.summaries))
    # plt.plot(templates[~templates_to_delete, :].T)
    # plt.show()

    # print("SHARPEN IS OFF!!!!")
    seg_summary.sharpen_across_chans()
    # seg_summary.remove_redundant_neurons(overlap_ratio_threshold=overlap_ratio_threshold)
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

    templates = []
    next_label = 0
    for n in neurons:
        if not n['deleted_as_redundant']:
            clips, _ = get_multichannel_clips(clips_dict, voltage,
                                    n['spike_indices'],
                                    clip_width=sort_info['clip_width'])
            # for chan in range(0, sort_info['n_channels']):
            #     if chan not in n['neighbors']:
            #         clips[:, chan*sort_info['n_samples_per_chan']:(chan+1)*sort_info['n_samples_per_chan']] = 0.0
            templates.append(np.median(clips, axis=0))
            next_label += 1
            # plt.plot(n['pursuit_template'])
            # plt.plot(templates[-1])
            # plt.show()

    del seg_summary

    templates = np.vstack(templates)
    print("Sharpening reduced number of templates to", templates.shape[0])
    print("Starting full binary pursuit search with", templates.shape[0], "templates in segment", seg_number)
    # plt.plot(templates.T)
    # plt.show()

    crossings, neuron_labels, bp_bool, clips = binary_pursuit_parallel.binary_pursuit(
                    templates, voltage, sort_info['sampling_rate'],
                    v_dtype, sort_info['clip_width'], sort_info['n_samples_per_chan'],
                    thresh_sigma=sort_info['sigma_noise_penalty'],
                    n_max_shift_inds=original_n_samples_per_chan-1,
                    get_adjusted_clips=sort_info['get_adjusted_clips'],
                    kernels_path=None,
                    max_gpu_memory=max_gpu_memory)

    if not sort_info['get_adjusted_clips']:
        clips, _ = get_multichannel_clips(clips_dict, voltage,
                                crossings, clip_width=sort_info['clip_width'])

    chans_to_template_labels = {}
    for chan in range(0, sort_info['n_channels']):
        chans_to_template_labels[chan] = []
    for unit in np.unique(neuron_labels):
        # Find this unit's channel as the channel with max SNR of template
        curr_template = np.median(clips[neuron_labels == unit, :], axis=0)
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
    # Make everything compatible with regular consolidate.WorkItemSummary
    sort_info['binary_pursuit_only'] = True

    return seg_data
