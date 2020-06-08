import numpy as np
from scipy import signal
import copy
from so_sorting.src import sort

import matplotlib.pyplot as plt


"""
    threshold(probe)

Determines the per-channel threshold necessary for the detection of spikes.
This function returns a vector of thresholds (one for each channel). These
represent the absolute value of the threshold.
"""
def median_threshold(voltage, sigma):
    if voltage.ndim == 1:
        voltage = np.expand_dims(voltage, 0)
    num_channels = voltage.shape[0]
    thresholds = np.empty((num_channels, ))
    for chan in range(0, num_channels):
        abs_voltage = np.abs(voltage[chan, :])
        thresholds[chan] = np.nanmedian(abs_voltage) / 0.6745
    thresholds *= sigma

    return thresholds


"""
    identify_threshold_crossings(Probe, thresholds[, skip, align_window, above_threshold_time])

Identify possible events on each channel. These events are those that pass
the threshold given for each channel (the length of thresholds should be the
length of the number of channels). We include a blanking period to avoid multiple
spike detections.

The returned value is a vector of vectors (no a complete matrix). Each vector
inside the exterior vector contains the indices of the threshold crossings for a
given channel. The length of this vector corresponds to the number of spike
events detected on that channel.
"""
def identify_threshold_crossings(Probe, chan, threshold, skip=0., align_window=[-5e-4, 5e-4]):

    skip_indices = max(int(round(skip * Probe.sampling_rate)), 1) - 1
    # Working with ABSOLUTE voltage here
    voltage = np.abs(Probe.get_voltage(chan))
    first_thresh_index = np.zeros(voltage.shape[0], dtype=np.bool)
    # Find points above threshold where preceeding sample was below threshold (excludes first point)
    first_thresh_index[1:] = np.logical_and(voltage[1:] > threshold, voltage[0:-1] <= threshold)
    events = np.nonzero(first_thresh_index)[0]

    # Realign event times on min or max in align_window
    window = time_window_to_samples(align_window, Probe.sampling_rate)[0]
    for evt in range(0, events.size):
        start = max(0, events[evt] + window[0]) # Start maximally at 0 or event plus window
        stop = min(Probe.n_samples - 1, events[evt] + window[1])# Stop minmally at event plus window or last index
        window_clip = Probe.voltage[chan, start:stop]
        max_index = np.argmax(window_clip) # Gets FIRST max in window
        max_value = window_clip[max_index]
        min_index = np.argmin(window_clip)
        min_value = window_clip[min_index]
        if max_value > -1 * min_value:
            # Max value is huge compared to min value, use max index
            events[evt] = start + max_index
        elif min_value < -1 * max_value:
            # Min value is huge (negative) compared to max, use min index
            events[evt] = start + min_index
        else:
            # Arbitrarily choose the negative going peak
            events[evt] = start + min_index

    # Remove events that follow preceeding valid event by less than skip_indices samples
    bad_index = np.zeros(events.shape[0], dtype=np.bool)
    last_n = 0
    for n in range(1, events.shape[0]):
        if events[n] - events[last_n] < skip_indices:
            bad_index[n] = True
        else:
            last_n = n
    events = events[~bad_index]

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
    chan_neighbor_ind = next((idx[0] for idx, val in np.ndenumerate(neighbors) if val == channel), None)
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


def align_events_with_template(Probe, channel, neuron_labels, event_indices, clip_width):
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

    # Align all clips with their own template
    for c in range(0, clips.shape[0]):
        cross_corr = np.correlate(clips[c, :], templates[np.nonzero(labels == neuron_labels[c])[0][0]], mode='valid')
        event_indices[c] += np.argmax(cross_corr) - int(temp_index[0])

    return event_indices, neuron_labels, valid_inds


def align_events_with_best_template(Probe, channel, neuron_labels, event_indices, clip_width):
    """ Takes the input data for ONE channel and computes the cross correlation
        of each spike with each template on the channel USING SINGLE CHANNEL CLIPS
        ONLY.  The spike time is then aligned with the peak cross correlation lag.
        This outputs new event indices reflecting this alignment, that can then be
        used to input into final sorting, as in cluster sharpening. """

    window, clip_width = time_window_to_samples(clip_width, Probe.sampling_rate)
    clips, valid_inds = get_singlechannel_clips(Probe, channel, event_indices, clip_width=clip_width)
    event_indices = event_indices[valid_inds]
    neuron_labels = neuron_labels[valid_inds]
    overlaps = np.zeros(event_indices.size, dtype=np.bool)
    templates, labels = calculate_templates(clips, neuron_labels)
    templates = [(t)/np.amax(np.abs(t)) for t in templates]
    window = np.abs(window)
    center = max(window)

    # Align all clips with best template
    for c in range(0, clips.shape[0]):
        best_peak = -np.inf
        best_shift = 0
        for temp_ind in range(0, len(templates)):
            cross_corr = np.correlate(clips[c, :], templates[temp_ind], mode='full')
            max_ind = np.argmax(cross_corr)
            if cross_corr[max_ind] > best_peak:
                best_peak = cross_corr[max_ind]
                shift = max_ind - center - window[0]
                if shift <= -window[0]//2 or shift >= window[1]//2:
                    overlaps[c] = True
                    continue
                best_shift = shift
        event_indices[c] += best_shift
    event_indices = event_indices[~overlaps]
    neuron_labels = neuron_labels[~overlaps]

    return event_indices, neuron_labels, valid_inds


def align_templates(Probe, channel, neuron_labels, event_indices, clip_width):
    """ Aligns templates to each other and shift their event indices accordingly.

    This function determines the template for each cluster and then asks whether
    the majority of clusters have templates with larger peak or valley. All
    templates are then aligned on peak or valley according to the majority and
    their constituent event_indices are shifted accordingly and returned. The
    purpose is to align all events while ignoring the jitter in peak valley
    alignment strategies caused by noise. """
    window, clip_width = time_window_to_samples(clip_width, Probe.sampling_rate)
    clips, valid_inds = get_singlechannel_clips(Probe, channel, event_indices, clip_width=clip_width)
    event_indices = event_indices[valid_inds]
    neuron_labels = neuron_labels[valid_inds]
    window = np.abs(window)
    templates, labels = calculate_templates(clips, neuron_labels)

    temp_peaks = []
    for t_ind, t in enumerate(templates):
        t_select = neuron_labels == labels[t_ind]
        t_weight = np.count_nonzero(t_select) / clips.shape[0]
        temp_peaks.append((np.amax(t) + np.amin(t)) * t_weight)

    if np.mean(temp_peaks) > 0:
        bias_up = True
    else:
        bias_up = False
    for t_ind in range(0, len(templates)):
        t = templates[t_ind]
        t_select = neuron_labels == labels[t_ind]
        min_t = np.amin(t)
        max_t = np.amax(t)
        if bias_up:
            # Align everything on peak
            shift = np.argmax(t)
        else:
            # Align everything on valley
            shift = np.argmin(t)
        event_indices[t_select] += shift - window[0] - 1

    return event_indices, neuron_labels, valid_inds


def wavelet_align_events(Probe, channel, event_indices, clip_width, band_width):
    """ Takes the input data for ONE channel and computes the cross correlation
        of each spike with each template on the channel USING SINGLE CHANNEL CLIPS
        ONLY.  The spike time is then aligned with the peak cross correlation lag.
        This outputs new event indices reflecting this alignment, that can then be
        used to input into final sorting, as in cluster sharpening. """
    # First align window index should be negative
    window, clip_width = time_window_to_samples(clip_width, Probe.sampling_rate)
    clips, valid_inds = get_singlechannel_clips(Probe, channel, event_indices, clip_width=clip_width)
    event_indices = event_indices[valid_inds]
    overlaps = np.zeros(event_indices.size, dtype=np.bool)
    # Create a mexican hat central template, centered on the current clip width
    window = np.abs(window)
    center = max(window)
    temp_scales = []
    scale = 1
    # Minimum oscillation that will fit in this clip width
    min_win_freq = 1./((window[1] + window[0])/Probe.sampling_rate)
    align_band_width = [band_width[0], band_width[1]]
    align_band_width[0] = max(min_win_freq, align_band_width[0])

    # Find center frequency of wavelet Fc. Uses the method in PyWavelets
    # central_frequency function
    central_template = signal.ricker(2 * center+1, scale)
    index = np.argmax(np.abs(np.fft.fft(central_template)[1:])) + 2
    if index > len(central_template) / 2:
        index = len(central_template) - index + 2
    Fc = 1.0 / (central_template.shape[0] / (index - 1))

    # Start scale at max bandwidth
    scale = Fc * Probe.sampling_rate / align_band_width[1]
    # Build scaled templates for multiple of two frequencies within band width
    pseudo_frequency = Fc / (scale * (1/Probe.sampling_rate))
    while pseudo_frequency >= align_band_width[0]:
        # Clips have a center and are odd, so this will match
        central_template = signal.ricker(2 * center+1, scale)
        temp_scales.append(central_template)
        scale *= 2
        pseudo_frequency = Fc / (scale * (1/Probe.sampling_rate))

    if len(temp_scales) == 0:
        # Choose single template at center of frequency band
        scale = Fc * Probe.sampling_rate / (align_band_width[0] + (align_band_width[1] - align_band_width[0]))
        central_template = signal.ricker(2 * center+1, scale)
        temp_scales.append(central_template)

    # Align all clips on the mexican hat central template
    for c in range(0, clips.shape[0]):
        best_peak = -np.inf
        # First find the best frequency (here 'template') for this clip
        for temp_ind in range(0, len(temp_scales)):
            cross_corr = np.convolve(clips[c, :], temp_scales[temp_ind], mode='full')
            max_ind = np.argmax(cross_corr)
            min_ind = np.argmin(cross_corr)
            if cross_corr[max_ind] > best_peak:
                curr_peak = cross_corr[max_ind]
            elif -1.*cross_corr[min_ind] > best_peak:
                curr_peak = -1.*cross_corr[min_ind]
            if curr_peak > best_peak:
                best_temp_ind = temp_ind
                best_peak = curr_peak
                best_corr = cross_corr
                best_max = max_ind
                best_min = min_ind

        # Now use the best frequency convolution to align by weighting the clip
        # values by the convolution
        if -best_corr[best_min] > best_corr[best_max]:
            # Dip in best corr is greater than peak, so invert it so we can
            # use following logic assuming working from peak
            best_corr *= -1
            best_max, best_min = best_min, best_max
        prev_min_ind = best_max
        while prev_min_ind > 0:
            prev_min_ind -= 1
            if best_corr[prev_min_ind] >= best_corr[prev_min_ind+1]:
                prev_min_ind += 1
                break
        prev_max_ind = prev_min_ind
        while prev_max_ind > 0:
            prev_max_ind -= 1
            if best_corr[prev_max_ind] <= best_corr[prev_max_ind+1]:
                prev_max_ind += 1
                break
        next_min_ind = best_max
        while next_min_ind < best_corr.shape[0]-1:
            next_min_ind += 1
            if best_corr[next_min_ind] >= best_corr[next_min_ind-1]:
                next_min_ind -= 1
                break
        next_max_ind = next_min_ind
        while next_max_ind < best_corr.shape[0]-1:
            next_max_ind += 1
            if best_corr[next_max_ind] <= best_corr[next_max_ind-1]:
                next_max_ind -= 1
                break
        # Weighted average from 1 cycle before to 1 cycle after peak
        avg_win = np.arange(prev_max_ind, next_max_ind+1)
        # Weighted by convolution values
        corr_weights = np.abs(best_corr[avg_win])
        best_arg = np.average(avg_win, weights=corr_weights)
        best_arg = np.around(best_arg).astype(np.int64)

        shift = best_arg - center - window[0]
        if shift <= -window[0] or shift >= window[1]:
            # If optimal shift is finding a different spike beyond window,
            # delete this spike as it violates our dead time between spikes
            overlaps[c] = True
            continue
        event_indices[c] += shift
    event_indices = event_indices[~overlaps]

    return event_indices


def align_adjusted_clips_with_template(Probe, channel, neighbors, clips, event_indices, neuron_labels, clip_width):
    """
        """

    # Get indices for this channel within clip width and double wide clipwidth
    clip_width, chan_neighbor_ind, curr_chan_win, samples_per_chan, curr_chan_inds = get_windows_and_indices(clip_width, Probe.sampling_rate, channel, neighbors)
    cc_clip_width, cc_chan_neighbor_ind, cc_curr_chan_win, cc_samples_per_chan, cc_curr_chan_inds = get_windows_and_indices(2*np.array(clip_width), Probe.sampling_rate, channel, neighbors)
    cc_curr_chan_center_index = cc_curr_chan_inds[0] + np.abs(cc_curr_chan_win[0])

    # Get double wide adjusted clips but do templates and alignment only on current channel
    adjusted_clips, event_indices, neuron_labels, valid_event_indices = get_adjusted_clips(Probe, channel, neighbors, clips, event_indices, neuron_labels, clip_width, cc_clip_width)
    templates, labels = calculate_templates(adjusted_clips[:, cc_curr_chan_center_index+curr_chan_win[0]:cc_curr_chan_center_index+curr_chan_win[1]], neuron_labels)
    aligned_adjusted_clips = np.empty((adjusted_clips.shape[0], samples_per_chan * len(neighbors)))

    # Align all clips with their own template
    for c in range(0, adjusted_clips.shape[0]):
        cross_corr = np.correlate(adjusted_clips[c, cc_curr_chan_inds], templates[np.nonzero(labels == neuron_labels[c])[0][0]], mode='valid')
        shift = np.argmax(cross_corr) - int(np.abs(curr_chan_win[0]))
        event_indices[c] += shift
        for n in range(0, len(neighbors)):
            n_chan_win = [n*samples_per_chan, (n+1)*samples_per_chan]
            cc_n_chan_win = [n*cc_samples_per_chan + np.abs(cc_curr_chan_win[0]) + curr_chan_win[0] + shift,
                             n*cc_samples_per_chan + np.abs(cc_curr_chan_win[0]) + curr_chan_win[1] + shift]
            aligned_adjusted_clips[c, n_chan_win[0]:n_chan_win[1]] = adjusted_clips[c, cc_n_chan_win[0]:cc_n_chan_win[1]]

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


"""
    get_clips(Probe, event_indices, clip_width)

Given a probe and the threshold crossings, return a matrix of clips for a
given set of threshold crossings. We center the clip on the threshold crossing
index. The width of the segment is passed to the function in units of seconds.
This is done over all channels on probes, so threshold crossings input in
event_indices must be a list where each element contains a numpy array of
threshold crossing indices for the corresponding channel in the Probe object.
event_indices that yield a clip width beyond data boundaries are ignored. """
def get_singlechannel_clips(Probe, channel, event_indices, clip_width):

    window, clip_width = time_window_to_samples(clip_width, Probe.sampling_rate)
    # Ignore spikes whose clips extend beyond the data and create mask for removing them
    valid_event_indices = np.ones_like(event_indices, dtype=np.bool)
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

    return spike_clips, valid_event_indices

"""
    get_multichannel_clips(Probe, channels, event_indices, clip_width)

    This is like get_clips except it concatenates the clips for each channel
    input in the list 'channels' in the order that they appear.  Event indices
    is a single one dimensional array of indices over which clips from all input
    channels will be aligned.  """
def get_multichannel_clips(Probe, channels, event_indices, clip_width):

    if event_indices.ndim > 1:
        raise ValueError("Event_indices must be one dimensional array of indices")

    window, clip_width = time_window_to_samples(clip_width, Probe.sampling_rate)
    # Ignore spikes whose clips extend beyond the data and create mask for removing them
    valid_event_indices = np.ones_like(event_indices, dtype=np.bool)
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
            # Subtract start ind above to adjust for discarded events
            start = stop

    return spike_clips, valid_event_indices


def get_adjusted_clips(Probe, channel, neighbors, clips, event_indices, neuron_labels, thresholds, input_clip_width, output_clip_width):
    """
    """
    output_clip_width, _, output_chan_win, output_samples_per_chan, _ = get_windows_and_indices(output_clip_width, Probe.sampling_rate, channel, neighbors)
    input_clip_width, _, input_chan_win, input_samples_per_chan, _ = get_windows_and_indices(input_clip_width, Probe.sampling_rate, channel, neighbors)
    valid_event_indices = np.logical_and(event_indices > np.abs(output_chan_win[0]), event_indices < Probe.n_samples - output_chan_win[1])
    event_indices, neuron_labels = keep_valid_inds([event_indices, neuron_labels], valid_event_indices)
    clips = clips[valid_event_indices, :]
    templates, labels = calculate_templates(clips, neuron_labels)
    neighbors = np.array(neighbors)

    # Get adjusted clips at all spike times
    adjusted_clips = np.empty((event_indices.size, output_samples_per_chan * neighbors.size))
    spike_times = np.zeros(Probe.n_samples, dtype=np.byte)
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
                residual_voltage[event_indices[spk_event]+input_chan_win[0]:event_indices[spk_event]+input_chan_win[1]] -= temp[input_slice]

    return adjusted_clips, event_indices, neuron_labels, valid_event_indices
