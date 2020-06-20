import numpy as np
import pickle
import os
from fbp.src.consolidate import SegSummary
from fbp.src.parallel import binary_pursuit_parallel, binary_pursuit_parallel_old



import matplotlib.pyplot as plt




def remove_overlapping_spikes(event_indices, clips, neuron_labels, templates,
                              template_labels, tol_inds):
    """
    """
    keep_bool = np.ones(event_indices.size, dtype=np.bool)
    temp_sse = np.zeros(2)
    curr_index = 0
    next_index = 1
    while next_index < event_indices.size:
        if event_indices[next_index] - event_indices[curr_index] <= tol_inds:
            curr_temp_ind = next((idx[0] for idx, val in
                            np.ndenumerate(template_labels) if val == neuron_labels[curr_index]), None)
            temp_sse[0] = np.sum((clips[curr_index, :] - templates[curr_temp_ind]) ** 2)
            next_temp_ind = next((idx[0] for idx, val in
                            np.ndenumerate(template_labels) if val == neuron_labels[next_index]), None)
            temp_sse[1] = np.sum((clips[next_index, :] - templates[next_temp_ind]) ** 2)
            if temp_sse[0] <= temp_sse[1]:
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


def full_binary_pursuit(work_items, data_dict, seg_number,
                        sort_info, v_dtype, overlap_ratio_threshold=2,
                        absolute_refractory_period=12e-4,
                        kernels_path=None, max_gpu_memory=None):

    # Need to build this in format used for consolidate functions
    seg_data = []
    seg_w_items = []
    for wi_ind, w_item in enumerate(work_items):
        if w_item['seg_number'] != seg_number:
            continue
        # Always do this to match with seg_data
        seg_w_items.append(w_item)
        if w_item['ID'] in data_dict['results_dict'].keys():
            if len(data_dict['results_dict'][w_item['ID']][0]) == 0:
                # This work item found nothing (or raised an exception)
                seg_data.append([[], [], [], [], w_item['ID']])
                continue
            with open(sort_info['tmp_clips_dir'] + '/temp_clips' + str(w_item['ID']) + '.pickle', 'rb') as fp:
                clips = pickle.load(fp)
            os.remove(sort_info['tmp_clips_dir'] + '/temp_clips' + str(w_item['ID']) + '.pickle')
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
    seg_summary.sharpen_across_chans()
    # seg_summary.remove_redundant_neurons(overlap_ratio_threshold=overlap_ratio_threshold)
    neurons = seg_summary.summaries

    templates = []
    template_labels = []
    next_label = 0
    for n in neurons:
        if not n['deleted_as_redundant']:
            templates.append(n['expanded_template'])
            template_labels.append(next_label)
            next_label += 1
    template_labels = np.array(template_labels, dtype=np.int64)

    del seg_summary

    seg_volts_buffer = data_dict['segment_voltages'][seg_number][0]
    seg_volts_shape = data_dict['segment_voltages'][seg_number][1]
    voltage = np.frombuffer(seg_volts_buffer, dtype=v_dtype).reshape(seg_volts_shape)

    templates = np.vstack(templates)
    print("Starting full binary pursuit search with", template_labels.shape[0], "templates in segment", seg_number)
    # plt.plot(templates.T)
    # plt.show()

    ocrossings, oneuron_labels, obp_bool, oclips = binary_pursuit_parallel_old.binary_pursuit(
                    templates, voltage, template_labels, sort_info['sampling_rate'],
                    v_dtype, sort_info['clip_width'], sort_info['n_samples_per_chan'],
                    thresh_sigma=1.645, kernels_path=None,
                    max_gpu_memory=max_gpu_memory)

    crossings, neuron_labels, bp_bool, clips = binary_pursuit_parallel.binary_pursuit(
                    templates, voltage, template_labels, sort_info['sampling_rate'],
                    v_dtype, sort_info['clip_width'], sort_info['n_samples_per_chan'],
                    thresh_sigma=1.645, kernels_path=None,
                    max_gpu_memory=max_gpu_memory)



    chans_to_template_labels = {}
    for chan in range(0, sort_info['n_channels']):
        chans_to_template_labels[chan] = []
    for unit in np.unique(neuron_labels):
        # Find this unit's channel as the channel with max value of template
        curr_template = np.mean(clips[neuron_labels == unit, :], axis=0)
        curr_chan = np.argmax(np.abs(curr_template)) // sort_info['n_samples_per_chan']
        chans_to_template_labels[curr_chan].append(unit)

    # Need to convert binary pursuit output to standard sorting output. This
    # requires data from every channel, even if it is just empty
    seg_data = []
    for chan in range(0, voltage.shape[0]):
        curr_item = None
        for w_item in work_items:
            if w_item['seg_number'] != seg_number:
                continue
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
