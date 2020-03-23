import numpy as np
from scipy import signal
import copy
from spikesorting_python.src import sort



"""
    threshold(probe)

Determines the per-electrode threshold necessary for the detection of spikes.
This function returns a vector of thresholds (one for each channel). These
represent the absolute value of the threshold.
"""
def median_threshold(voltage, sigma):
    if voltage.ndim == 1:
        voltage = np.expand_dims(voltage, 0)
    num_electrodes = voltage.shape[0]
    thresholds = np.empty((num_electrodes, ))
    for chan in range(0, num_electrodes):
        abs_voltage = np.abs(voltage[chan, :])
        thresholds[chan] = np.nanmedian(abs_voltage) / 0.6745
    thresholds *= sigma

    return thresholds


"""
    identify_threshold_crossings(Probe, thresholds[, skip, align_window, above_threshold_time])

Identify possible events on each electrode. These events are those that pass
the threshold given for each channel (the length of thresholds should be the
length of the number of electrodes). We include a blanking period to avoid multiple
spike detections.

The returned value is a vector of vectors (no a complete matrix). Each vector
inside the exterior vector contains the indices of the threshold crossings for a
given channel. The length of this vector corresponds to the number of spike
events detected on that channel.
"""
def identify_threshold_crossings(Probe, thresholds, skip=0.33e-3, align_window=[-4e-4, 4e-4]):

    thresholds = np.array(thresholds)
    if Probe.num_electrodes != thresholds.size:
        raise ValueError("Thresholds are invalid - expected length ", Probe.num_electrodes)

    skip_indices = max(int(round(skip * Probe.sampling_rate)), 1) - 1
    events = [[] for x in range(0, Probe.num_electrodes)]
    for chan in range(0, Probe.num_electrodes):
        # Working with ABSOLUTE voltage mostly
        voltage = np.abs(Probe.get_voltage(chan))
        first_thresh_index = np.zeros(voltage.shape[0], dtype='bool')
        # Find points above threshold where preceeding sample was below threshold (excludes first point)
        first_thresh_index[1:] = np.logical_and(voltage[1:] > thresholds[chan], voltage[0:-1] <= thresholds[chan])
        events[chan] = np.nonzero(first_thresh_index)[0]

        # Realign event times on min or max in align_window
        events[chan] = align_events(Probe, chan, events[chan], align_window)

        # Remove events that follow preceeding valid event by less than skip_indices samples
        bad_index = np.zeros(events[chan].shape[0], dtype='bool')
        last_n = 0
        for n in range(1, events[chan].shape[0]):
            if events[chan][n] - events[chan][last_n] < skip_indices:
                bad_index[n] = True
            else:
                last_n = n
        events[chan] = events[chan][~bad_index]

    return events


def make_noise_crossings(Probe, chan, real_crossings, n_noise_crossings, clip_width=[-2e-4, 4e-4], skip=0.33e-3, align_window=[-4e-4, 4e-4]):

    skip_indices = max(int(round(skip * Probe.sampling_rate)), 1) - 1
    window = time_window_to_samples(align_window, Probe.sampling_rate)[0]

    # Compute window centered on each real crossing such that randomly chosen noise events
    # will not overlap with actual spikes, given the clip_width and align_windows
    total_crossing_window = time_window_to_samples(np.add(clip_width, align_window), Probe.sampling_rate)[0]
    remove_inds = []
    for cr in real_crossings:
        remove_inds.append(np.arange(cr+total_crossing_window[0], cr+total_crossing_window[1]))
    remove_inds = np.hstack(remove_inds)
    # Ensure these cannot overlap with beginning and end of voltage trace
    noise_choices = np.arange(np.abs(total_crossing_window[0]), Probe.n_samples-np.abs(total_crossing_window[1]))
    noise_choices = noise_choices[np.in1d(noise_choices, remove_inds, invert=True)]
    events = np.random.choice(noise_choices, n_noise_crossings, replace=False)
    events.sort()

    n_iter = 0
    while True:
        if n_iter > 10:
            break
        # Realign event times on min or max in align_window
        events = align_events(Probe, chan, events, align_window)

        # Remove events that follow preceeding valid event by less than skip_indices samples
        bad_index = np.zeros(events.shape[0], dtype='bool')
        last_n = 0
        for n in range(1, events.shape[0]):
            if events[n] - events[last_n] < skip_indices:
                bad_index[n] = True
            else:
                last_n = n
        n_bad = np.count_nonzero(bad_index)
        if n_bad > 0:
            events = events[~bad_index]
            events = np.hstack((events, np.random.choice(noise_choices, n_bad, replace=False)))
            events.sort()
        else:
            break
        n_iter += 1

    return events


"""
    align_events(Probe, probe_channel, events, align_window)

Given threshold crossing event indices for a single channel,
here, we attempt to align spikes based on their maximum absolute value. That is,
we change the values of events within the time align_window to match the maximum
value of the signal within align_window. This is beneficial when computing
the principle components of the signals, since it reduces the variance due to
random jittering of the signal.  This is designed to work on ONE CHANNEL at a time
to save memory.
"""
def align_events(Probe, probe_channel, events, align_window):

    if events.ndim > 1:
        raise ValueError("Input event_indices must be a 1 dimensional array of event indices")
    window = time_window_to_samples(align_window, Probe.sampling_rate)[0]

    if type(probe_channel) is not int:
        raise ValueError("Input probe channel must be a single scalar integer value!")

    for evt in range(0, events.size):
        start = max(0, events[evt] + window[0]) # Start maximally at 0 or event plus window
        stop = min(Probe.n_samples - 1, events[evt] + window[1])# Stop minmally at event plus window or last index

        # We want to perform an "optimal" alignment in the face of noise. If
        # we choose just the max abs value and a positive and negative going
        # threshold are reasonably close together, then noise will result
        # in sporatically choosing one or the other. Similarly, it we only
        # align to the most positive (peak) or most negative (trough).
        # these values might be close together and cause an alignment failure.
        # Therefore, we look for both the highest peak and the lowest
        # trough and decide on the optimal alignment based on the ratio of
        # these two values.

        window_clip = Probe.get_voltage(probe_channel, slice(start, stop, 1))
        max_index = np.argmax(window_clip) # Gets FIRST max in window
        max_value = window_clip[max_index]
        min_index = np.argmin(window_clip)
        min_value = window_clip[min_index]

        if max_value > -2 * min_value:
            # Max value is huge compared to min value, use max index
            events[evt] = start + max_index
        elif min_value < -2 * max_value:
            # Min value is huge (negative) compared to max, use min index
            events[evt] = start + min_index
        else:
            # The spike is biphasic with similar positive and negative
            # going parts. It is possible that noise would shift our decision
            # point. So in this case, we arbitrarily choose the negative going
            # peak as our reference point.
            events[evt] = start + min_index

    return events


def keep_valid_inds(keep_data_list, valid_inds):

    out_data = []
    for data in keep_data_list:
        out_data.append(data[valid_inds])
    return tuple(x for x in out_data) if len(keep_data_list) > 1 else out_data[0]


def time_window_to_samples(time_window, sampling_rate):
    """ Converts a two element list time window in seconds to a corresponding two
        element list window in units of samples.  Assumes the window is centered
        on a time and therefore the first element MUST be negative or it will be
        converted to a negative number. Second element has 1 added so that
        sample_window[1] is INCLUSIVELY SLICEABLE without adjustment. Also
        returns a copy of the input time_window which may have had its first
        element's sign inverted. """

    new_time_window = copy.copy(time_window)
    if new_time_window[0] > 0:
        new_time_window[0] *= -1
    sample_window = [0, 0]
    sample_window[0] = min(int(round(new_time_window[0] * sampling_rate)), 0)
    sample_window[1] = max(int(round(new_time_window[1] * sampling_rate)), 1) + 1 # Add one so that last element is included

    return sample_window, new_time_window


def get_windows_and_indices(clip_width, sampling_rate, channel, neighbors):
    """ Computes some basic info used in many functions about how clips are
        are formatted and provides window indices and clip indices. """

    curr_chan_win, clip_width = time_window_to_samples(clip_width, sampling_rate)
    chan_neighbor_ind = np.where(neighbors == channel)[0][0]
    samples_per_chan = curr_chan_win[1] - curr_chan_win[0]
    curr_chan_inds = np.arange(samples_per_chan * chan_neighbor_ind, samples_per_chan * chan_neighbor_ind + samples_per_chan, 1)

    return clip_width, chan_neighbor_ind, curr_chan_win, samples_per_chan, curr_chan_inds


def calculate_templates(clips, neuron_labels):
    """ Computes the median template from clips for each unique label in neuron_labels.
        Output is a list of templates and a numpy array of the unique labels to which
        each template corresponds. """

    labels = np.unique(neuron_labels)
    # Use clips to get templates for each label in order
    templates = []
    for n in labels:
        templates.append(np.nanmean(clips[neuron_labels == n, :], axis=0))

    return templates, labels


def align_events_with_template(Probe, channel, neuron_labels, event_indices, clip_width=[-2e-4, 4e-4]):
    """ Takes the input data for ONE channel and computes the cross correlation
        of each spike with each template on the channel USING SINGLE CHANNEL CLIPS
        ONLY.  The spike time is then aligned with the peak cross correlation lag.
        This outputs new event indices reflecting this alignment, that can then be
        used to input into final sorting, as in cluster sharpening. """

    window, clip_width = time_window_to_samples(clip_width, Probe.sampling_rate)
    # Create clips twice as wide as current clip width being careful to ensure that rounding
    # from time to samples is preserved in each
    cc_clip_width = [0, 0]
    cc_clip_width[0] = 2 * window[0] / Probe.sampling_rate
    cc_clip_width[1] = 2 * (window[1]-1) / Probe.sampling_rate
    # Find indices within extra wide clips that correspond to the original clipwidth for template
    temp_index = [0, 0]
    temp_index[0] = -1 * min(int(round(clip_width[0] * Probe.sampling_rate)), 0)
    temp_index[1] = 2 * temp_index[0] + max(int(round(clip_width[1] * Probe.sampling_rate)), 1) + 1 # Add one so that last element is included
    clips, valid_inds = get_singlechannel_clips(Probe, channel, event_indices, clip_width=cc_clip_width)
    event_indices = event_indices[valid_inds]
    neuron_labels = neuron_labels[valid_inds]
    templates, labels = calculate_templates(clips[:, temp_index[0]:temp_index[1]], neuron_labels)

    # Align all waves with their own template
    for wave in range(0, clips.shape[0]):
        cross_corr = np.correlate(clips[wave, :], templates[np.nonzero(labels == neuron_labels[wave])[0][0]], mode='valid')
        event_indices[wave] += np.argmax(cross_corr) - int(temp_index[0])

    return event_indices, neuron_labels, valid_inds


def align_adjusted_clips_with_template(Probe, channel, neighbors, clips, event_indices, neuron_labels, thresholds, clip_width):
    """
        """

    # Get indices for this channel within clip width and double wide clipwidth
    clip_width, chan_neighbor_ind, curr_chan_win, samples_per_chan, curr_chan_inds = get_windows_and_indices(clip_width, Probe.sampling_rate, channel, neighbors)
    cc_clip_width, cc_chan_neighbor_ind, cc_curr_chan_win, cc_samples_per_chan, cc_curr_chan_inds = get_windows_and_indices(2*np.array(clip_width), Probe.sampling_rate, channel, neighbors)
    cc_curr_chan_center_index = cc_curr_chan_inds[0] + np.abs(cc_curr_chan_win[0])

    # Get double wide adjusted clips but do templates and alignment only on current channel
    adjusted_clips, event_indices, neuron_labels, valid_event_indices = get_adjusted_clips(Probe, channel, neighbors, clips, event_indices, neuron_labels, thresholds, clip_width, cc_clip_width)
    templates, labels = calculate_templates(adjusted_clips[:, cc_curr_chan_center_index+curr_chan_win[0]:cc_curr_chan_center_index+curr_chan_win[1]], neuron_labels)
    aligned_adjusted_clips = np.empty((adjusted_clips.shape[0], samples_per_chan * len(neighbors)))

    # Align all waves with their own template
    for wave in range(0, adjusted_clips.shape[0]):
        cross_corr = np.correlate(adjusted_clips[wave, cc_curr_chan_inds], templates[np.nonzero(labels == neuron_labels[wave])[0][0]], mode='valid')
        shift = np.argmax(cross_corr) - int(np.abs(curr_chan_win[0]))
        event_indices[wave] += shift
        for n in range(0, len(neighbors)):
            n_chan_win = [n*samples_per_chan, (n+1)*samples_per_chan]
            cc_n_chan_win = [n*cc_samples_per_chan + np.abs(cc_curr_chan_win[0]) + curr_chan_win[0] + shift,
                             n*cc_samples_per_chan + np.abs(cc_curr_chan_win[0]) + curr_chan_win[1] + shift]
            aligned_adjusted_clips[wave, n_chan_win[0]:n_chan_win[1]] = adjusted_clips[wave, cc_n_chan_win[0]:cc_n_chan_win[1]]

    return aligned_adjusted_clips, event_indices, neuron_labels, valid_event_indices


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


"""
    get_clips(Probe, event_indices, clip_width)

Given a probe and the threshold crossings, return a matrix of clips for a
given set of threshold crossings. We center the clip on the threshold crossing
index. The width of the segment is passed to the function in units of seconds.
This is done over all channels on probes, so threshold crossings input in
event_indices must be a list where each element contains a numpy array of
threshold crossing indices for the corresponding electrode in the Probe object.
event_indices that yield a clip width beyond data boundaries are ignored. """
def get_singlechannel_clips(Probe, channel, event_indices, clip_width=[-2e-4, 4e-4], thresholds=None):

    window, clip_width = time_window_to_samples(clip_width, Probe.sampling_rate)
    # Ignore spikes whose clips extend beyond the data and create mask for removing them
    valid_event_indices = np.ones_like(event_indices, dtype='bool')
    start_ind = 0
    n = event_indices[start_ind]
    while n + window[0] < 0:
        valid_event_indices[start_ind] = False
        start_ind += 1
        if start_ind == event_indices.size:
            # There are no valid indices
            valid_event_indices[:] = False
            return None, valid_event_indices
        n = event_indices[start_ind]
    stop_ind = event_indices.shape[0] - 1
    n = event_indices[stop_ind]
    while n + window[1] > Probe.n_samples:
        valid_event_indices[stop_ind] = False
        stop_ind -= 1
        if stop_ind < 0:
            # There are no valid indices
            valid_event_indices[:] = False
            return None, valid_event_indices
        n = event_indices[stop_ind]
    spike_clips = np.empty((np.count_nonzero(valid_event_indices), window[1] - window[0]))
    for out_ind, spk in enumerate(range(start_ind, stop_ind+1)): # Add 1 to index through last valid index
        spike_clips[out_ind, :] = Probe.get_voltage(channel, slice(event_indices[spk]+window[0], event_indices[spk]+window[1], 1))
        if thresholds is not None:
            # Rescale by channel noise
            spike_clips[out_ind, :] /= thresholds

    return spike_clips, valid_event_indices

"""
    get_multichannel_clips(Probe, channels, event_indices, clip_width)

    This is like get_clips except it concatenates the clips for each channel
    input in the list 'channels' in the order that they appear.  Event indices
    is a single one dimensional array of indices over which clips from all input
    channels will be aligned.  """
def get_multichannel_clips(Probe, channels, event_indices, clip_width=[-2e-4, 4e-4], thresholds=None):

    if event_indices.ndim > 1:
        raise ValueError("Event_indices must be one dimensional array of indices")

    window, clip_width = time_window_to_samples(clip_width, Probe.sampling_rate)
    # Ignore spikes whose clips extend beyond the data and create mask for removing them
    valid_event_indices = np.ones_like(event_indices, dtype='bool')
    start_ind = 0
    n = event_indices[start_ind]

    while (n + window[0]) < 0:
        valid_event_indices[start_ind] = False
        start_ind += 1
        if start_ind == event_indices.size:
            # There are no valid indices
            valid_event_indices[:] = False
            return None, valid_event_indices
        n = event_indices[start_ind]
    stop_ind = event_indices.shape[0] - 1
    n = event_indices[stop_ind]
    while (n + window[1]) >= Probe.n_samples:
        valid_event_indices[stop_ind] = False
        stop_ind -= 1
        if stop_ind < 0:
            # There are no valid indices
            valid_event_indices[:] = False
            return None, valid_event_indices
        n = event_indices[stop_ind]
    spike_clips = np.empty((np.count_nonzero(valid_event_indices), (window[1] - window[0]) * len(channels)))
    for out_ind, spk in enumerate(range(start_ind, stop_ind+1)): # Add 1 to index through last valid index
        chan_ind = 0
        start = 0
        for chan in channels:
            chan_ind += 1
            stop = chan_ind * (window[1] - window[0])
            spike_clips[out_ind, start:stop] = Probe.get_voltage(chan, slice(event_indices[spk]+window[0], event_indices[spk]+window[1], 1))
            if thresholds is not None:
                # Rescale by channel noise
                spike_clips[out_ind, start:stop] /= thresholds[chan]
            # Subtract start ind above to adjust for discarded events
            start = stop

    return spike_clips, valid_event_indices


def get_adjusted_clips(Probe, channel, neighbors, clips, event_indices, neuron_labels, thresholds, input_clip_width, output_clip_width):
    """ Input clips are assumed to be threshold normalized.  Templates
        will be made from clips, then un-normalized for subtraction from the voltage trace, which is assumed to
        be NON-normalized.  These are normalized again so that threshold normalized adjusted clips are returned.
        If none of this is true, using all input thresholds == 1 will have no effect.
    """

    output_clip_width, _, output_chan_win, output_samples_per_chan, _ = get_windows_and_indices(output_clip_width, Probe.sampling_rate, channel, neighbors)
    input_clip_width, _, input_chan_win, input_samples_per_chan, _ = get_windows_and_indices(input_clip_width, Probe.sampling_rate, channel, neighbors)
    valid_event_indices = np.logical_and(event_indices > np.abs(output_chan_win[0]), event_indices < Probe.n_samples - output_chan_win[1])
    event_indices, neuron_labels = keep_valid_inds([event_indices, neuron_labels], valid_event_indices)
    clips = clips[valid_event_indices, :]
    templates, labels = calculate_templates(clips, neuron_labels)
    neighbors = np.array(neighbors)

    # Need to un-normalize templates because Probe voltage is NOT normalized to threshold
    for t in templates:
        for ind, neigh in enumerate(np.nditer(neighbors)):
            t[ind*output_samples_per_chan:output_samples_per_chan*(ind+1)] *= thresholds[neigh]

    # Get adjusted clips at all spike times
    adjusted_clips = np.empty((event_indices.size, output_samples_per_chan * neighbors.size))
    spike_times = np.zeros(Probe.n_samples, dtype='byte')
    for neigh_ind, chan in enumerate(np.nditer(neighbors)):
        # First get residual voltage for the current channel by subtracting INPUT clips
        input_slice = slice(neigh_ind*input_samples_per_chan, input_samples_per_chan*(neigh_ind+1), 1)
        output_slice = slice(neigh_ind*output_samples_per_chan, output_samples_per_chan*(neigh_ind+1), 1)
        residual_voltage = np.copy(Probe.get_voltage(chan))
        for label_ind, temp in enumerate(templates):
            spike_times[:] = 0  # Reset to zero each iteration
            spike_times[event_indices[neuron_labels == labels[label_ind]]] = 1
            temp_kernel = get_zero_phase_kernel(temp[input_slice], np.abs(input_chan_win[0]))
            residual_voltage -= signal.fftconvolve(spike_times, temp_kernel, mode='same')

        # Now add each INPUT spike one at a time to residual voltage to get adjusted_clips
        for label_ind, temp in enumerate(templates):
            current_spike_indices = np.where(neuron_labels == labels[label_ind])[0]
            for spk_event in current_spike_indices:
                residual_voltage[event_indices[spk_event]+input_chan_win[0]:event_indices[spk_event]+input_chan_win[1]] += temp[input_slice]
                adjusted_clips[spk_event, output_slice] = residual_voltage[event_indices[spk_event]+output_chan_win[0]:event_indices[spk_event]+output_chan_win[1]]
                adjusted_clips[spk_event, output_slice] /= thresholds[chan]
                residual_voltage[event_indices[spk_event]+input_chan_win[0]:event_indices[spk_event]+input_chan_win[1]] -= temp[input_slice]

    return adjusted_clips, event_indices, neuron_labels, valid_event_indices


""" Below are the stepchild function versions of some of the above functions.  They are needed
    for add_secret_spikes so that computations can be performed on the adjusted voltage rather
    than the original probe voltage. Reference above originals for more comments on functionality.
    They all take an arbitrary 1D voltage array as input rather than a channel number from
    which voltage is sought via the Probe object. """

def identify_threshold_crossings_voltage_in(Probe, voltage, threshold, skip=0.33e-3, align_window=[-4e-4, 4e-4]):

    skip_indices = max(int(round(skip * Probe.sampling_rate)), 1) - 1
    # Working with ABSOLUTE voltage mostly
    abs_voltage = np.abs(voltage)
    first_thresh_index = np.zeros(abs_voltage.shape[0], dtype='bool')
    # Find points above threshold where preceeding sample was below threshold (excludes first point)
    first_thresh_index[1:] = np.logical_and(abs_voltage[1:] > threshold, abs_voltage[0:-1] <= threshold)
    events = np.nonzero(first_thresh_index)[0]

    # Realign event times on min or max in align_window
    events = align_events_voltage_in(Probe, voltage, events, align_window)

    if skip_indices > 0:
        # Remove events that follow preceeding valid event by less than skip_indices samples
        bad_index = np.zeros(events.shape[0], dtype='bool')
        last_n = 0
        for n in range(1, events.shape[0]):
            if events[n] - events[last_n] < skip_indices:
                bad_index[n] = True
            else:
                last_n = n
        events = events[~bad_index]

    return events


def align_events_voltage_in(Probe, voltage, events, align_window):

    if events.ndim > 1:
        raise ValueError("Input event_indices must be a 1 dimensional array of event indices")

    window = time_window_to_samples(align_window, Probe.sampling_rate)[0]
    for evt in range(0, events.size):
        start = max(0, events[evt] + window[0]) # Start maximally at 0 or event plus window
        stop = min(Probe.n_samples - 1, events[evt] + window[1])# Stop minmally at event plus window or last index

        window_clip = voltage[start:stop]
        max_index = np.argmax(window_clip) # Gets FIRST max in window
        max_value = window_clip[max_index]
        min_index = np.argmin(window_clip)
        min_value = window_clip[min_index]

        if max_value > -2 * min_value:
            # Max value is huge compared to min value, use max index
            events[evt] = start + max_index
        elif min_value < -2 * max_value:
            # Min value is huge (negative) compared to max, use min index
            events[evt] = start + min_index
        else:
            events[evt] = start + min_index

    return events


def get_singlechannel_clips_voltage_in(Probe, voltage, event_indices, clip_width=[-2e-4, 4e-4], thresholds=None):

    window, clip_width = time_window_to_samples(clip_width, Probe.sampling_rate)
    # Ignore spikes whose clips extend beyond the data and create mask for removing them
    valid_event_indices = np.ones_like(event_indices, dtype='bool')
    start_ind = 0
    n = event_indices[start_ind]
    while n + window[0] < 0:
        valid_event_indices[start_ind] = False
        start_ind += 1
        if start_ind == event_indices.size:
            # There are no valid indices
            valid_event_indices[:] = False
            return None, valid_event_indices
        n = event_indices[start_ind]
    stop_ind = event_indices.shape[0] - 1
    n = event_indices[stop_ind]
    while n + window[1] > Probe.n_samples:
        valid_event_indices[stop_ind] = False
        stop_ind -= 1
        if stop_ind < 0:
            # There are no valid indices
            valid_event_indices[:] = False
            return None, valid_event_indices
        n = event_indices[stop_ind]
    spike_clips = np.empty((np.count_nonzero(valid_event_indices), window[1] - window[0]))
    for out_ind, spk in enumerate(range(start_ind, stop_ind+1)): # Add 1 to index through last valid index
        spike_clips[out_ind, :] = voltage[event_indices[spk]+window[0]:event_indices[spk]+window[1]]
        if thresholds is not None:
            # Rescale by channel noise
            spike_clips[out_ind, :] /= thresholds

    return spike_clips, valid_event_indices
