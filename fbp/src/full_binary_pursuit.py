import numpy as np
import pickle
import os
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
            if (n_template_spikes[n1] < 10*n_template_spikes[test_unit]) or (n1 == test_unit) or (templates_SS[n1] > templates_SS[test_unit]):
                continue
            template_1 = templates[n1, :]

            for n2 in range(n1+1, templates.shape[0]):
                if (n_template_spikes[n2] < 10*n_template_spikes[test_unit]) or (n2 == test_unit) or (templates_SS[n2] > templates_SS[test_unit]):
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
        return sort_info['clip_width']
    all_events = np.hstack(all_events)
    all_clips, _ = get_multichannel_clips(clips_dict, voltage,
                                    all_events, clip_width=sort_info['clip_width'])
    median_clip = np.median(all_clips, axis=0)
    first_indices = np.arange(0, sort_info['n_samples_per_chan']*(sort_info['n_channels']-1)+1, sort_info['n_samples_per_chan'], dtype=np.int64)
    last_indices = np.arange(sort_info['n_samples_per_chan']-1, sort_info['n_samples_per_chan']*sort_info['n_channels']+1, sort_info['n_samples_per_chan'], dtype=np.int64)
    median_abs_clip_value = np.median(np.median(np.abs(all_clips), axis=0))
    clip_end_tolerance = np.abs(0.1 * median_abs_clip_value)

    bp_clip_width = [v for v in sort_info['clip_width']]
    min_start = 2*bp_clip_width[0]
    step_size = bp_clip_width[0]/10
    while np.any(np.abs(median_clip[first_indices]) > clip_end_tolerance):
        bp_clip_width[0] += step_size
        if bp_clip_width[0] < min_start:
            print("Clip width start never converged. Using 2 times input start width.")
            bp_clip_width[0] = min_start
            break
        all_clips, _ = get_multichannel_clips(clips_dict, voltage,
                                        all_events, clip_width=bp_clip_width)
        median_clip = np.median(all_clips, axis=0)

    max_stop = 2*bp_clip_width[1]
    step_size = bp_clip_width[1]/10
    while np.any(np.abs(median_clip[last_indices]) > clip_end_tolerance):
        bp_clip_width[1] += step_size
        if bp_clip_width[1] > max_stop:
            print("Clip width stop never converged. Using 2 times input stop width.")
            bp_clip_width[1] = max_stop
            break
        all_clips, _ = get_multichannel_clips(clips_dict, voltage,
                                        all_events, clip_width=bp_clip_width)
        median_clip = np.median(all_clips, axis=0)

    return bp_clip_width



def full_binary_pursuit(work_items, data_dict, seg_number,
                        sort_info, v_dtype, overlap_ratio_threshold=2,
                        absolute_refractory_period=12e-4,
                        kernels_path=None, max_gpu_memory=None):

    # Get numpy view of voltage for clips and binary pursuit
    seg_volts_buffer = data_dict['segment_voltages'][seg_number][0]
    seg_volts_shape = data_dict['segment_voltages'][seg_number][1]
    voltage = np.frombuffer(seg_volts_buffer, dtype=v_dtype).reshape(seg_volts_shape)
    # Make a dictionary with all info needed for get_multichannel_clips
    clips_dict = {'sampling_rate': sort_info['sampling_rate'],
                  'n_samples': sort_info['n_samples'],
                  'v_dtype': v_dtype}

    # Determine the set of work items for this segment
    seg_w_items = [w for w in work_items if w['seg_number'] == seg_number]

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

    bp_clip_width = get_binary_pursuit_clip_width(seg_w_items, clips_dict, voltage, data_dict, sort_info)
    sort_info['clip_width'] = bp_clip_width
    bp_chan_win, _ = time_window_to_samples(sort_info['clip_width'], sort_info['sampling_rate'])
    sort_info['n_samples_per_chan'] = bp_chan_win[1] - bp_chan_win[0]

    templates = []
    next_label = 0
    for n in neurons:
        if not n['deleted_as_redundant']:
            clips, _ = get_multichannel_clips(clips_dict, voltage,
                                    n['spike_indices'],
                                    clip_width=bp_clip_width)
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

    # thresh_sigma = 1.645, 1.96, 2.576
    crossings, neuron_labels, bp_bool, clips = binary_pursuit_parallel.binary_pursuit(
                    templates, voltage, sort_info['sampling_rate'],
                    v_dtype, sort_info['clip_width'], sort_info['n_samples_per_chan'],
                    thresh_sigma=1.645, kernels_path=None,
                    max_gpu_memory=max_gpu_memory)

    chans_to_template_labels = {}
    for chan in range(0, sort_info['n_channels']):
        chans_to_template_labels[chan] = []

    for unit in np.unique(neuron_labels):
        # Find this unit's channel as the channel with max SNR of template
        curr_template = np.mean(clips[neuron_labels == unit, :], axis=0)
        unit_best_snr = 0.
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

    # Need to convert binary pursuit output to standard sorting output. This
    # requires data from every channel, even if it is just empty
    seg_data = []
    for chan in range(0, sort_info['n_channels']):
        curr_item = None
        for w_item in seg_w_items:
            if w_item['channel'] == chan:
                curr_item = w_item
                # if chan == 0:
                #     print("REASSIGNING CHAN NEIGHBOR IND TO CHANNEL! (line 179 full_binary_pursuit)")
                # w_item['chan_neighbor_ind'] = chan
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
                            clips[select, chan_ind*sort_info['n_samples_per_chan']:(chan_ind+1)*sort_info['n_samples_per_chan']]
                chan_clips.append(unit_clips)
                # chan_clips.append(clips[select, :])

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
