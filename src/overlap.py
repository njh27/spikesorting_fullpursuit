import numpy as np
from scipy.signal import fftconvolve
from scipy import stats
from spikesorting_python.src import segment
from spikesorting_python.src import consolidate



def get_zero_phase_kernel(x, x_center):
    """ Zero pads the 1D kernel x, so that it is aligned with the current element
        of x located at x_center.  This ensures that convolution with the kernel
        x will be zero phase with respect to x_center.
    """

    kernel_offset = x.size - 2 * x_center - 1 # -1 To center ON the x_center index
    kernel_size = np.abs(kernel_offset) + x.size
    if kernel_size // 2 == 0: # Kernel must be odd
        kernel_size -= 1
        kernel_offset -= 1
    kernel = np.zeros(kernel_size)
    if kernel_offset > 0:
        kernel_slice = slice(kernel_offset, kernel.size)
    elif kernel_offset < 0:
        kernel_slice = slice(0, kernel.size + kernel_offset)
    else:
        kernel_slice = slice(0, kernel.size)
    kernel[kernel_slice] = x

    return kernel


def calculate_expected_overlap(spike_indices1, spike_indices2, sampling_rate, overlap_time):
    """ Spike indices MUST BE IN ORDER for this to work"""

    first_index = max(spike_indices1[0], spike_indices2[0])
    last_index = min(spike_indices1[-1], spike_indices2[-1])
    n1_count = np.count_nonzero(np.logical_and(spike_indices1 >= first_index, spike_indices1 <= last_index))
    n2_count = np.count_nonzero(np.logical_and(spike_indices2 >= first_index, spike_indices2 <= last_index))
    num_ms = (np.ceil((last_index - first_index) / (sampling_rate / 1000))).astype(np.int64)
    if num_ms > 0:
        expected_overlap = (overlap_time * 1000 * n1_count * n2_count) / num_ms
    else:
        expected_overlap = 0

    return expected_overlap


def reassign_simultaneous_spiking_clusters(clips, neuron_labels, event_indices, sampling_rate, clip_width, min_var_accounted_for):
    """
        """
    # NOTE: This could be expanded to include combinations of more than 2 units
    # by iterating over nested loops like the 'k' loop below, but it is not clear
    # if this would be useful.

    templates, template_labels = segment.calculate_templates(clips, neuron_labels)
    if template_labels.size < 3:
        # We can't search for combined combinations without at least 3 clusters
        return neuron_labels
    i_inds = []
    j_scores = []
    k_scores = []
    j_k_scores = []
    total_variance = np.array([np.sum(templates[x] ** 2) for x in range(0, len(templates))])
    # Put in order of descending variance
    variance_order = np.argsort(total_variance)[::-1]
    total_variance = total_variance[variance_order]
    total_variance = total_variance[variance_order]
    template_labels = template_labels[variance_order]
    templates = [templates[x] for x in variance_order]
    neuron_event_indices = []
    for n_label in template_labels:
        neuron_event_indices.append(event_indices[neuron_labels == n_label])
    for i in range(0, len(templates)):
        for j in range(0, len(templates)):
            if i == j:
                continue
            if np.sign(templates[i][np.argmax(np.abs(templates[i]))]) != np.sign(templates[j][np.argmax(np.abs(templates[j]))]):
                continue
            if neuron_event_indices[i].size >= neuron_event_indices[j].size:
                # If neuron i is combination of j + x it should have fewer spikes than either
                continue

            j_residual = templates[i] - templates[j]
            j_variance_accounted_for = 1 - (np.sum((templates[i] - templates[j]) ** 2)) / total_variance[i]
            for k in range(0, len(templates)):
                # Allowing j == k allows 2 similar but as yet unseparated units to successfully split
                if k == i:
                    continue
                if neuron_event_indices[i].size >= neuron_event_indices[k].size:
                    # If neuron i is combination of j + k it should have fewer spikes than both
                    continue

                # Allow template k to be shifted and optimally aligned before getting its residual
                shift = np.argmax(np.correlate(j_residual, templates[k], mode='same')) - j_residual.size // 2
                shifted_template = np.zeros(templates[k].size) # zero padded k
                if shift == 0:
                    shifted_template = templates[k]
                elif shift < 0:
                    shifted_template[0:shift] = templates[k][-shift:]
                else:
                    shifted_template[shift:] = templates[k][0:-shift]

                k_variance_accounted_for = 1 - (np.sum((templates[i] - shifted_template) ** 2)) / total_variance[i]
                j_plus_k_variance_accounted_for = 1 - (np.sum((templates[i] - (templates[j] + shifted_template)) ** 2)) / total_variance[i]
                if j_plus_k_variance_accounted_for < min_var_accounted_for or j_plus_k_variance_accounted_for < j_variance_accounted_for:
                    continue

                # If neuron i is truly a combination of j and k, it should have fewer spikes than either
                # and considerably less actual overlaps than expected
                expected_overlap_ij = calculate_expected_overlap(neuron_event_indices[j], neuron_event_indices[i], sampling_rate, np.amax(clip_width))
                expected_overlap_ik = calculate_expected_overlap(neuron_event_indices[k], neuron_event_indices[i], sampling_rate, np.amax(clip_width))
                actual_overlap_ij = np.count_nonzero(consolidate.find_overlapping_spike_bool(neuron_event_indices[j], neuron_event_indices[i], max_samples=sampling_rate*np.amax(clip_width)))
                actual_overlap_ik = np.count_nonzero(consolidate.find_overlapping_spike_bool(neuron_event_indices[k], neuron_event_indices[i], max_samples=sampling_rate*np.amax(clip_width)))
                if actual_overlap_ij > 0.2 * expected_overlap_ij or actual_overlap_ik > 0.2 * expected_overlap_ik:
                    continue

                i_inds.append([i, j, k])
                j_scores.append(j_variance_accounted_for)
                k_scores.append(k_variance_accounted_for)
                j_k_scores.append(j_plus_k_variance_accounted_for)

    if len(i_inds) == 0:
        # No overlaps were found
        return neuron_labels
    i_inds = np.array(i_inds)
    j_scores = np.array(j_scores)
    k_scores = np.array(k_scores)
    j_k_scores = np.array(j_k_scores)

    spikes_to_check = np.unique(i_inds[:, 0])
    already_moved = []
    combo_order = np.empty(spikes_to_check.size, dtype=np.int64)
    spike_order = np.empty(spikes_to_check.size, dtype=np.int64)
    score_order = np.empty(spikes_to_check.size)
    for ind, spk in enumerate(spikes_to_check):
        current_ind = i_inds[:, 0] == spk
        max_score = np.amax(j_k_scores[current_ind])
        combo_order[ind] = np.where(np.logical_and(current_ind, j_k_scores == max_score))[0][0]
        score_order[ind] = j_k_scores[combo_order[ind]]
        spike_order[ind] = spk

    # Go through combo choices in descending score order
    order_inds = np.argsort(score_order)[::-1]
    combo_order = combo_order[order_inds]
    spike_order = spike_order[order_inds]

    already_used = []
    already_removed = []
    for choose_combo, spk in zip(combo_order, spike_order):
        if template_labels[spk] in already_used:
            continue

        if (j_scores[choose_combo] >= k_scores[choose_combo] and
            template_labels[i_inds[choose_combo, 1]] not in already_removed):
            # assign everything to neuron j, i.e. template_labels[i_inds[choose_combo, 1]]
            neuron_labels[neuron_labels == template_labels[spk]] = template_labels[i_inds[choose_combo, 1]]
            already_used.append(template_labels[i_inds[choose_combo, 1]])
            already_removed.append(template_labels[spk])
        elif template_labels[i_inds[choose_combo, 2]] not in already_removed:
            neuron_labels[neuron_labels == template_labels[spk]] = template_labels[i_inds[choose_combo, 2]]
            already_used.append(template_labels[i_inds[choose_combo, 2]])
            already_removed.append(template_labels[spk])
        else:
            # Other label was already removed so can't merge into it - do nothing
            pass

    return neuron_labels


def remove_overlapping_spikes(event_indices, max_samples):
    """ DATA MUST BE SORTED IN ORDER OF EVENT INDICES FOR THIS TO WORK.
    """

    # Find overlapping spikes, excluding equal values
    overlapping_spike_bool = consolidate.find_overlapping_spike_bool(event_indices, event_indices, max_samples=max_samples, except_equal=True)
    # Also find any duplicate values and keep only one of them
    repeats_bool = np.ones(event_indices.size, dtype='bool')
    repeats_bool[np.unique(event_indices, return_index=True)[1]] = False
    removed_index = np.logical_or(overlapping_spike_bool, repeats_bool)
    event_indices = event_indices[~removed_index]

    return event_indices, removed_index


def find_multichannel_max_neuron(Probe, channel, neighbors, check_time, indices,
        labels, templates, template_labels, chan_win, spike_biases,
        template_error, new_indices, new_labels):
    """ This assumes templates are not normalized and can thus be directly
    subtracted from voltage. This function could be far more efficient if we
    tracked our current location in the spike event indices array instead
    of searching the whole thing every time to find the relevant spikes to
    subtract for computing the residuals. """
    samples_per_chan = chan_win[1] - chan_win[0]
    # Window big enough to cover all spikes possibly overlapping with check_time
    min_t = samples_per_chan + int(np.abs(chan_win[0]))
    max_t = samples_per_chan + int(chan_win[1])
    if (Probe.n_samples < (check_time + max_t)) or (check_time < min_t):
        return np.zeros(template_labels.size, dtype=np.float32)
    residual_window = [-1*min_t, max_t]

    delta_likelihood = np.zeros(len(templates), dtype=np.float32)
    # Get residual voltage centered on current time
    spike_times = np.zeros((residual_window[1] - residual_window[0]) + 1, dtype=np.bool)
    events_in_window = np.logical_and(indices >= check_time + residual_window[0], indices <= check_time + residual_window[1])
    new_events_in_window = [True if (x >= check_time + residual_window[0] and x <= check_time + residual_window[1]) else False for x in new_indices]
    residual_voltage = np.copy(np.float32(Probe.get_voltage(neighbors, slice(check_time + residual_window[0], check_time + residual_window[1] + 1, 1))))

    for label_ind, temp in enumerate(templates):
        curr_spikes = indices[np.logical_and(events_in_window, labels == template_labels[label_ind])]
        # Adjust for window
        curr_spikes -= (check_time + residual_window[0])
        new_curr_spikes = []
        for ni in range(0, len(new_indices)):
            if new_events_in_window[ni] and new_labels[ni] == template_labels[label_ind]:
                new_curr_spikes.append(new_indices[ni] - (check_time + residual_window[0]))
        curr_spikes = np.hstack((curr_spikes, new_curr_spikes)).astype(np.int64)
        spike_times[:] = 0  # Reset to zero each iteration
        spike_times[curr_spikes] = 1

        for neigh_ind in range(0, neighbors.size):
            input_slice = slice(neigh_ind*samples_per_chan, samples_per_chan*(neigh_ind+1), 1)
            temp_kernel = np.float32(get_zero_phase_kernel(temp[input_slice], np.abs(chan_win[0])))
            residual_voltage[neigh_ind, :] -= fftconvolve(spike_times, temp_kernel, mode='same')

    # Only need the center point now
    residual_voltage = residual_voltage[:, -1*residual_window[0]+chan_win[0]:-1*residual_window[0]+chan_win[1]]
    for neigh_ind in range(0, neighbors.size):
        input_slice = slice(neigh_ind*samples_per_chan, samples_per_chan*(neigh_ind+1), 1)
        for label_ind, temp in enumerate(templates):
            delta_likelihood[label_ind] += (template_error[label_ind, neigh_ind]
                    + np.dot(residual_voltage[neigh_ind, :], temp[input_slice])
                    - spike_biases[label_ind, neigh_ind])

    return delta_likelihood


def binary_pursuit_secret_spikes(Probe, channel, neuron_labels, event_indices,
        clip_width):
    """
    This function is slow and not efficient in time or memory consumption. The
    GPU version should be preferred. """
    neighbors = np.array(Probe.get_neighbors(channel)).astype(np.int64)
    # Need to find the indices of the current channel within the multichannel template
    chan_win, clip_width = segment.time_window_to_samples(clip_width, Probe.sampling_rate)
    _, chan_neighbor_ind, clip_samples, samples_per_chan, curr_chan_inds = segment.get_windows_and_indices(
            clip_width, Probe.sampling_rate, channel, neighbors)
    default_multi_check = True if neighbors.size > 1 else False
    # Remove any spikes within 1 clip width of each other
    event_indices.sort()
    event_indices, removed_index = remove_overlapping_spikes(event_indices, clip_samples[1]-clip_samples[0])
    neuron_labels = neuron_labels[~removed_index]

    # Get clips for templates to subtract
    clips, valid_inds = segment.get_singlechannel_clips(Probe, channel, event_indices, clip_width=clip_width)
    event_indices, neuron_labels = segment.keep_valid_inds([event_indices, neuron_labels], valid_inds)
    # Reassign any units that might represent a combination of units into one of the two combining units
    neuron_labels = reassign_simultaneous_spiking_clusters(clips, neuron_labels, event_indices, Probe.sampling_rate, clip_width, 0.75)
    # Align everything
    event_indices, neuron_labels, valid_inds = segment.align_events_with_template(Probe, channel, neuron_labels, event_indices, clip_width=clip_width)

    # Get new aligned multichannel clips here for computing voltage residuals.  Still not normalized
    multi_clips, valid_inds = segment.get_multichannel_clips(Probe, neighbors, event_indices, clip_width=clip_width)
    event_indices, neuron_labels = segment.keep_valid_inds([event_indices, neuron_labels], valid_inds)
    multi_clips = np.float32(multi_clips)

    multi_templates, template_labels = segment.calculate_templates(multi_clips, neuron_labels)
    multi_templates = [np.float32(x) for x in multi_templates]
    clips = multi_clips[:, curr_chan_inds]
    templates = [t[curr_chan_inds] for t in multi_templates]

    # Compute residual voltage by subtracting all known spikes
    spike_times = np.zeros(Probe.n_samples, dtype='byte')
    spike_biases = np.zeros((template_labels.size, neighbors.size), dtype=np.float32)
    template_error = np.zeros((template_labels.size, neighbors.size), dtype=np.float32)
    for chan_ind, chan in enumerate(neighbors):
        if chan == channel:
            # Wait to do main channel last so we can keep residual voltage
            continue
        residual_voltage = np.copy(np.float32(Probe.get_voltage(chan)))
        temp_win = [chan_ind * samples_per_chan,
                    chan_ind * samples_per_chan + samples_per_chan]
        n = 0
        for temp_label, temp in zip(template_labels, multi_templates):
            spike_times[:] = 0  # Reset to zero each iteration
            current_event_indices = event_indices[neuron_labels == temp_label]
            spike_times[current_event_indices] = 1
            temp_kernel = np.float32(get_zero_phase_kernel(temp[temp_win[0]:temp_win[1]], np.abs(chan_win[0])))
            residual_voltage -= fftconvolve(spike_times, temp_kernel, mode='same')
            n += 1
        n = 0
        for temp_label, temp in zip(template_labels, multi_templates):
            temp_kernel = np.float32(get_zero_phase_kernel(temp[temp_win[0]:temp_win[1]], np.abs(chan_win[0])))
            spike_biases[n, chan_ind] = np.median(np.abs(fftconvolve(residual_voltage, temp_kernel, mode='same')))
            template_error[n, chan_ind] = -0.5 * np.dot(
                temp[temp_win[0]:temp_win[1]], temp[temp_win[0]:temp_win[1]])
            n += 1
    residual_voltage = np.copy(np.float32(Probe.get_voltage(channel)))
    n = 0
    for temp_label, temp in zip(template_labels, multi_templates):
        spike_times[:] = 0  # Reset to zero each iteration
        current_event_indices = event_indices[neuron_labels == temp_label]
        spike_times[current_event_indices] = 1
        temp_kernel = np.float32(get_zero_phase_kernel(temp[curr_chan_inds], np.abs(chan_win[0])))
        residual_voltage -= fftconvolve(spike_times, temp_kernel, mode='same')
        n += 1
    n = 0
    for temp_label, temp in zip(template_labels, multi_templates):
        temp_kernel = np.float32(get_zero_phase_kernel(temp[curr_chan_inds], np.abs(chan_win[0])))
        spike_biases[n, chan_neighbor_ind] = np.median(np.abs(fftconvolve(residual_voltage, temp_kernel, mode='same')))
        template_error[n, chan_neighbor_ind] = -0.5 * np.dot(
            temp[curr_chan_inds], temp[curr_chan_inds])
        n += 1
    new_event_indices = []
    new_event_labels = []

    min_t = int(2 * np.abs(chan_win[0]))
    max_t = int(residual_voltage.size - (4 * chan_win[1]))
    double_chan_win = 2 * np.array(chan_win)
    delta_likelihood = np.zeros((len(templates), double_chan_win[1] - chan_win[0]), dtype=np.float32)
    test_likelihood = np.zeros_like(delta_likelihood)

    dl_update_start_t = min_t + chan_win[0]
    dl_update_stop_t = min_t + double_chan_win[1]
    dl_update_start_ind = 0

    rollback_t = None
    t = min_t
    while t <= max_t:
        # Compute delta likelihood function and its max for each neuron in current window
        for dl_index, time in enumerate(range(dl_update_start_t, dl_update_stop_t)):
            x = residual_voltage[time+chan_win[0]:time+chan_win[1]]
            for unit in range(0, delta_likelihood.shape[0]):
                delta_likelihood[unit, dl_update_start_ind + dl_index] = (
                    template_error[unit, chan_neighbor_ind]
                    + np.dot(x, templates[unit])
                    - spike_biases[unit, chan_neighbor_ind])

        max_neuron = None
        single_chan_dl_cross = delta_likelihood > 0
        single_chan_peaks = np.any(single_chan_dl_cross, axis=0)
        if np.any(single_chan_peaks):
            single_chan_likelihood = np.copy(delta_likelihood)
            if default_multi_check:
                for ct in range(0, single_chan_peaks.size):
                    if not single_chan_peaks[ct]:
                        continue
                    check_time = int(ct + t + chan_win[0])
                    delta_likelihood[:, ct] = find_multichannel_max_neuron(Probe, channel,
                                        neighbors, check_time, event_indices, neuron_labels,
                                        multi_templates, template_labels,
                                        chan_win, spike_biases, template_error,
                                        new_event_indices, new_event_labels)
            # Enforce AND operation for single and multi
            test_likelihood[:] = 0.
            test_likelihood[single_chan_dl_cross] = delta_likelihood[single_chan_dl_cross]

            # Find the neuron with highest delta likelihood, and its time index if greater than 0
            max_delta_likelihood = np.argmax(test_likelihood, axis=1)
            max_neuron_delta_likelihood = -np.inf
            inds_to_check = []
            ind_likelihood = []
            for ind in range(0, max_delta_likelihood.size):
                if test_likelihood[ind, max_delta_likelihood[ind]] > 0:
                    if test_likelihood[ind, max_delta_likelihood[ind]] > max_neuron_delta_likelihood:
                        max_neuron_delta_likelihood = test_likelihood[ind, max_delta_likelihood[ind]]
                        max_neuron = ind

        if max_neuron is None:
            # highest delta likelihood in this window is <= 0 so move on to next non-overlapping window
            t += double_chan_win[1] + np.abs(chan_win[0])
            dl_update_start_t = t + chan_win[0]
            dl_update_stop_t = t + double_chan_win[1]
            dl_update_start_ind = 0
            continue

        if max_delta_likelihood[max_neuron] > (chan_win[1] - chan_win[0]):
            # Best spike falls beyond current clip so move t there and repeat on new window
            # Remember we have not added a spike/updated residuals at this point so the
            # update time window is NOT the clip width centered on new t

            # Need to remember this t in case it is skipped
            if rollback_t is None:
                if (np.any(np.any(delta_likelihood[:, 0:(chan_win[1]
                            - chan_win[0])] > 0, axis=1)) or
                    np.any(np.any(single_chan_likelihood[:, 0:(chan_win[1]
                                - chan_win[0])] > 0, axis=1))):
                    rollback_t = t

            # Need to update delta_likelihood at max
            t += max_delta_likelihood[max_neuron]
            dl_update_start_t = t + chan_win[0]
            dl_update_stop_t = t + double_chan_win[1]
            dl_update_start_ind = 0
            continue
        else:
            # Best spike falls within current window, so add it
            new_event_time = max_delta_likelihood[max_neuron] + t + chan_win[0]
            residual_voltage[new_event_time+chan_win[0]:new_event_time+chan_win[1]] -= templates[max_neuron]
            new_event_indices.append(new_event_time)
            new_event_labels.append(template_labels[max_neuron])
            if rollback_t:
                # We have now added the spike above, but to get here we previously skipped
                # a time point that may have a spike and/or dependency on this spike's
                # addition so move t back before where we skipped, to the furtherst
                # point back that could have been affected
                t = rollback_t - chan_win[1]
                rollback_t = None
                dl_update_start_t = t + chan_win[0]
                dl_update_stop_t = t + double_chan_win[1]
                dl_update_start_ind = 0
            else:
                # Otherwise move back to the minimum time point that could be
                # affected by the new spike time
                t = new_event_time - chan_win[1]
                dl_update_start_t = t + chan_win[0]
                dl_update_stop_t = t + double_chan_win[1]
                dl_update_start_ind = 0
        if t < min_t:
            # Move forward regardless and don't look back
            rollback_t = None
            t = min_t
            t += double_chan_win[1] + np.abs(chan_win[0])
            dl_update_start_t = t + chan_win[0]
            dl_update_stop_t = t + double_chan_win[1]
            dl_update_start_ind = 0

    # Add new found spikes to old ones
    event_indices = np.hstack((event_indices, np.array(new_event_indices, dtype=np.int64)))
    neuron_labels = np.hstack((neuron_labels, new_event_labels))
    secret_spike_bool = np.zeros(event_indices.size, dtype='bool')
    if len(new_event_indices) > 0:
        secret_spike_bool[-len(new_event_indices):] = True

    print("Found a total of", np.count_nonzero(secret_spike_bool), "secret spikes", flush=True)

    return event_indices, neuron_labels, secret_spike_bool
