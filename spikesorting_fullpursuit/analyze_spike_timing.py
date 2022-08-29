import numpy as np


""" This module contains helper functions for analysis of spike timing among
sorted units in the form of creating CCGs and identifying identical spikes. """

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
    samples_axis = (np.floor((samples_axis + d_samples/2) / d_samples)).astype(np.int64)
    if spikes_1.size == 0  or spikes_2.size == 0:
        # CCG is zeros if one unit has no spikes
        if return_trains:
            return np.zeros(samples_axis.shape[0], dtype=np.int64), samples_axis, None, None
        else:
            return np.zeros(samples_axis.shape[0], dtype=np.int64), samples_axis

    # Convert the spike indices to units of d_samples
    spikes_1 = (np.floor((spikes_1 + d_samples/2) / d_samples)).astype(np.int64)
    spikes_2 = (np.floor((spikes_2 + d_samples/2) / d_samples)).astype(np.int64)

    # Convert to spike trains
    train_len = int(max(spikes_1[-1], spikes_2[-1])) + 1 # This is samples, so need to add 1
    spike_trains_1 = np.zeros(train_len, dtype="bool")
    spike_trains_1[spikes_1] = True
    spike_trains_2 = np.zeros(train_len, dtype="bool")
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


def find_overlapping_spike_bool(spikes_1, spikes_2, overlap_tol, except_equal=False):
    """ Finds an index into spikes_1 that indicates whether a spike index of spike_2
        occurs within +/- overlap_tol of a spike index in spikes_1.  Spikes_1 and 2
        are numpy arrays of spike indices in units of samples. Input spikes_1 and
        spikes_2 MUST be SORTED because this function won't work if they
        are not ordered."""
    overlap_bool = np.zeros(spikes_1.shape[0], dtype="bool")
    ind1 = 0
    ind2 = 0
    while (ind1 < spikes_1.shape[0]) and (ind2 < spikes_2.shape[0]):
        if except_equal and spikes_1[ind1] == spikes_2[ind2]:
            ind1 += 1
        elif spikes_1[ind1] < spikes_2[ind2] - overlap_tol:
            ind1 += 1
        elif spikes_1[ind1] > spikes_2[ind2] + overlap_tol:
            ind2 += 1
        else:
            while (ind1 < spikes_1.shape[0]) and spikes_1[ind1] <= spikes_2[ind2] + overlap_tol:
                overlap_bool[ind1] = True
                ind1 += 1

    return overlap_bool


def keep_binary_pursuit_duplicates(event_indices, binary_pursuit_bool, tol_inds):
    """ Preferentially KEEPS spikes found in binary pursuit.
    """
    keep_bool = np.ones(event_indices.size, dtype="bool")
    curr_index = 0
    next_index = 1
    while next_index < event_indices.size:
        if event_indices[next_index] - event_indices[curr_index] <= tol_inds:
            if binary_pursuit_bool[curr_index] and ~binary_pursuit_bool[next_index]:
                keep_bool[next_index] = False
                # Move next index, keep current index
                next_index += 1
            elif ~binary_pursuit_bool[curr_index] and binary_pursuit_bool[next_index]:
                keep_bool[curr_index] = False
                # Move current index to next, move next index
                curr_index = next_index
                next_index += 1
            else:
                # Two spikes with same index, neither from binary pursuit.
                #  Should be chosen based on templates or some other means.
                # Move both indices to next pair
                curr_index = next_index
                next_index += 1
        else:
            # Move both indices to next pair
            curr_index = next_index
            next_index += 1

    return keep_bool


def remove_binary_pursuit_duplicates(event_indices, clips, unit_template,
                                     binary_pursuit_bool, tol_inds):
    """ Preferentially removes overlapping spikes that were both found by binary
    pursuit. This can remove double dipping artifacts as binary pursuit attempts
    to minimize residual error. This only applies for data sorted within the
    same segment and belonging to the same unit, so inputs should reflect that. """
    keep_bool = np.ones(event_indices.size, dtype="bool")
    temp_sse = np.zeros(2)
    curr_index = 0
    next_index = 1
    while next_index < event_indices.size:
        if event_indices[next_index] - event_indices[curr_index] <= tol_inds:
            # Violate window
            if binary_pursuit_bool[curr_index] and binary_pursuit_bool[next_index]:
                # AND both found by binary pursuit
                temp_sse[0] = np.sum((clips[curr_index, :] - unit_template) ** 2)
                temp_sse[1] = np.sum((clips[next_index, :] - unit_template) ** 2)
                if temp_sse[0] <= temp_sse[1]:
                    # current spike is better or equal
                    keep_bool[next_index] = False
                    # Move next index, keep current index
                    next_index += 1
                else:
                    # next spike is better
                    keep_bool[curr_index] = False
                    # Move current index to next, move next index
                    curr_index = next_index
                    next_index += 1
            elif binary_pursuit_bool[curr_index] and ~binary_pursuit_bool[next_index]:
                # Move next index, keep current index
                next_index += 1
            else:
                # Move current index to next, move next index
                curr_index = next_index
                next_index += 1
        else:
            # Move both indices to next pair
            curr_index = next_index
            next_index += 1

    return keep_bool


def remove_spike_event_duplicates(event_indices, clips, unit_template, tol_inds):
    """
    """
    keep_bool = np.ones(event_indices.size, dtype="bool")
    temp_sse = np.zeros(2)
    curr_index = 0
    next_index = 1
    while next_index < event_indices.size:
        if event_indices[next_index] - event_indices[curr_index] <= tol_inds:
            temp_sse[0] = np.sum((clips[curr_index, :] - unit_template) ** 2)
            temp_sse[1] = np.sum((clips[next_index, :] - unit_template) ** 2)
            if temp_sse[0] <= temp_sse[1]:
                # current spike is better or equal
                keep_bool[next_index] = False
                # Move next index, keep current index
                next_index += 1
            else:
                # next spike is better
                keep_bool[curr_index] = False
                # Move current index to next, move next index
                curr_index = next_index
                next_index += 1
        else:
            # Move both indices to next pair
            curr_index = next_index
            next_index += 1

    return keep_bool


def remove_spike_event_duplicates_across_chans(combined_neuron):
    """
    """
    event_indices = combined_neuron['spike_indices']
    keep_bool = np.ones(event_indices.size, dtype="bool")
    temp_sse = np.zeros(2)
    curr_index = 0
    next_index = 1
    while next_index < event_indices.size:
        if event_indices[next_index] - event_indices[curr_index] <= combined_neuron['duplicate_tol_inds']:
            # Compare each spike to the template for its own channel and choose
            # the one that matches its own assigned template/channel best
            curr_chan, next_chan = None, None
            for chan in combined_neuron['channel_selector'].keys():
                if combined_neuron['channel_selector'][chan][curr_index]:
                    curr_chan = chan
                if combined_neuron['channel_selector'][chan][next_index]:
                    next_chan = chan
            temp_sse[0] = np.sum((combined_neuron['clips'][curr_index, :] - combined_neuron['template'][curr_chan]) ** 2)
            temp_sse[1] = np.sum((combined_neuron['clips'][next_index, :] - combined_neuron['template'][next_chan]) ** 2)
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


def compute_spike_trains(spike_indices, bin_width_samples, min_max_indices):
    """ Turns input spike indices into a binary spike train such that each
    element of output is True if 1 or more spikes occured in that time bin and
    False otherwise. """
    if len(min_max_indices) == 1:
        min_max_indices = [0, min_max_indices]
    bin_width_samples = int(bin_width_samples)
    train_len = int(np.ceil((min_max_indices[1] - min_max_indices[0] + bin_width_samples/2) / bin_width_samples) + 1) # Add one because it's a time/samples slice

    # Select spikes in desired index range
    select = np.logical_and(spike_indices >= min_max_indices[0], spike_indices <= min_max_indices[1])
    # Convert the spike indices to units of bin_width_samples
    spikes = (np.floor((spike_indices[select] + bin_width_samples/2 - min_max_indices[0]) / bin_width_samples)).astype(np.int64)
    spike_train = np.zeros(train_len, dtype="bool")
    spike_train[spikes] = True

    return spike_train


def calc_overlap_ratio(n1_spikes, n2_spikes, max_samples):
    """ Calculates the percentage of spike indices from n1 and n2 spikes that
    coincide within max_samples samples of each other. Only the time period
    in which both n1 and n2 have indices that overlap each other are considered.
    Spike indices are assumed to be ordered.
    """
    # Overlap time is the time that their spikes coexist
    overlap_win = [max(n1_spikes[0], n2_spikes[0]),
                   min(n1_spikes[-1], n2_spikes[-1])]
    n1_start = next((idx[0] for idx, val in np.ndenumerate(n1_spikes)
                         if val >= overlap_win[0]), None)
    if n1_start is None:
        # neuron 1 has no spikes in the overlap
        return 0.
    n2_start = next((idx[0] for idx, val in np.ndenumerate(n2_spikes)
                         if val >= overlap_win[0]), None)
    if n2_start is None:
        # neuron 2 has no spikes in the overlap
        return 0.
    # Used as slice, so strictly >, not >=
    n1_stop = next((idx[0] for idx, val in np.ndenumerate(n1_spikes)
                        if val > overlap_win[1]), None)
    n2_stop = next((idx[0] for idx, val in np.ndenumerate(n2_spikes)
                        if val > overlap_win[1]), None)

    n1_spike_train = compute_spike_trains(n1_spikes[n1_start:n1_stop],
                                          max_samples, overlap_win)
    n2_spike_train = compute_spike_trains(n2_spikes[n2_start:n2_stop],
                                          max_samples, overlap_win)
    num_hits = np.count_nonzero(np.logical_and(n1_spike_train, n2_spike_train))
    if num_hits == 0:
        return 0.
    n1_misses = np.count_nonzero(np.logical_and(n1_spike_train, ~n2_spike_train))
    n2_misses = np.count_nonzero(np.logical_and(n2_spike_train, ~n1_spike_train))
    overlap_ratio = max(num_hits / (num_hits + n1_misses),
                        num_hits / (num_hits + n2_misses))
    return overlap_ratio


def calc_expected_overlap(spike_inds_1, spike_inds_2, overlap_time, sampling_rate):
    """ Returns the expected number of overlapping spikes between neuron 1 and
    neuron 2 within a time window 'overlap_time' assuming independent spiking.
    As usual, spike_indices within each neuron must be sorted. """
    first_index = max(spike_inds_1[0], spike_inds_2[0])
    last_index = min(spike_inds_1[-1], spike_inds_2[-1])
    num_ms = int(np.ceil((last_index - first_index) / (sampling_rate / 1000)))
    if num_ms <= 0:
        # Neurons never fire at the same time, so expected overlap is 0
        return 0., 0., 0.

    # Find spike indices from each neuron that fall within the same time window
    # Should be impossible to return None because of num_ms check above
    n1_start = next((idx[0] for idx, val in np.ndenumerate(spike_inds_1) if val >= first_index), None)
    n1_stop = next((idx[0] for idx, val in np.ndenumerate(spike_inds_1[::-1]) if val <= last_index), None)
    n1_stop = spike_inds_1.shape[0] - n1_stop - 1 # Not used as slice so -1
    n2_start = next((idx[0] for idx, val in np.ndenumerate(spike_inds_2) if val >= first_index), None)
    n2_stop = next((idx[0] for idx, val in np.ndenumerate(spike_inds_2[::-1]) if val <= last_index), None)
    n2_stop = spike_inds_2.shape[0] - n2_stop - 1 # Not used as slice so -1

    # Infer number of spikes in overlapping window based on first and last index
    n1_count = n1_stop - n1_start
    n2_count = n2_stop - n2_start
    expected_overlap = (overlap_time * 1000 * n1_count * n2_count) / num_ms

    return expected_overlap, n1_count, n2_count


def calc_expected_overlap_ratio(spike_inds_1, spike_inds_2, overlap_time, sampling_rate):
    """ Returns the expected number of overlapping spikes between neuron 1 and
    neuron 2 within a time window 'overlap_time' assuming independent spiking.
    Ratio is computed by normalizing to the maximum  number of overlapping spikes
    that could have occurred.
    As usual, spike_indices within each neuron must be sorted. """
    expected_overlap, n1_count, n2_count = calc_expected_overlap(spike_inds_1, spike_inds_2, overlap_time, sampling_rate)
    if n1_count == 0 and n2_count == 0:
        return 0.
    expected_overlap_ratio = expected_overlap / min(n1_count, n2_count)

    return expected_overlap_ratio


def calc_ccg_overlap_ratios(spike_inds_1, spike_inds_2, overlap_time, sampling_rate):
    """ Returns the expected number of overlapping spikes between neuron 1 and
    neuron 2 within a time window 'overlap_time'. This is done based on the CCG
    between the two units, comparing the center bin to the nearest neighboring
    bins. This accounts for spike correlations between the units instead of
    assuming a stationary, independent firing rate.
    As usual, spike_indices within each neuron must be sorted. """
    first_index = max(spike_inds_1[0], spike_inds_2[0])
    last_index = min(spike_inds_1[-1], spike_inds_2[-1])
    num_ms = int(np.ceil((last_index - first_index) / (sampling_rate / 1000)))
    if num_ms <= 0 or overlap_time <= 0:
        # Neurons never fire at the same time, so expected overlap is 0
        return 0., 0., 0.

    # Find spike indices from each neuron that fall within the same time window
    # Should be impossible to return None because of num_ms check above
    n1_start = next((idx[0] for idx, val in np.ndenumerate(spike_inds_1) if val >= first_index), None)
    n1_stop = next((idx[0] for idx, val in np.ndenumerate(spike_inds_1[::-1]) if val <= last_index), None)
    n1_stop = spike_inds_1.shape[0] - n1_stop - 1 # Not used as slice so -1
    n2_start = next((idx[0] for idx, val in np.ndenumerate(spike_inds_2) if val >= first_index), None)
    n2_stop = next((idx[0] for idx, val in np.ndenumerate(spike_inds_2[::-1]) if val <= last_index), None)
    n2_stop = spike_inds_2.shape[0] - n2_stop - 1 # Not used as slice so -1

    # Infer number of spikes in overlapping window based on first and last index
    n1_count = n1_stop - n1_start
    n2_count = n2_stop - n2_start
    n_overlap_spikes = min(n1_count, n2_count)
    if n_overlap_spikes == 0:
        # Neurons never fire at the same time, so expected overlap is 0
        return 0., 0., 0.

    # CCG bins based on center, meaning that zero centered bin contains values
    # from +/- half bin width. Therefore we must double overlap_time to get
    # overlaps at +/- overlap_time
    overlap_time *= 2
    bin_samples = int(round(overlap_time * sampling_rate))
    samples_window = 10 * bin_samples
    counts, x_vals = zero_symmetric_ccg(spike_inds_1, spike_inds_2,
                                        samples_window, bin_samples)
    counts = counts / counts.shape[0] # Normalize by N bins
    zero_ind = np.flatnonzero(x_vals == 0)[0]
    # Find maximal neighboring count and use as expected value
    if counts[zero_ind - 1] >= counts[zero_ind + 1]:
        expected_bin = zero_ind - 1
    else:
        expected_bin = zero_ind + 1
    # Find maximal change between bins, excluding center
    max_left = np.amax(np.abs(np.diff(counts[0:zero_ind])))
    max_right = np.amax(np.abs(np.diff(counts[zero_ind+1:])))
    delta_max = max(max_left, max_right)

    # Overlap ratio relative to spikes in overlapping time window
    expected_overlap_ratio = counts[expected_bin] / n_overlap_spikes
    actual_overlap_ratio = counts[zero_ind] / n_overlap_spikes
    # Normalize delta max also
    delta_max = delta_max / n_overlap_spikes

    return expected_overlap_ratio, actual_overlap_ratio, delta_max


def calc_spike_half_width(template):
    """ Computes half width of spike as number of indices between peak and valley. """
    peak_ind = np.argmax(template)
    valley_ind = np.argmin(template)
    if peak_ind >= valley_ind:
        # peak is after valley
        spike_width = peak_ind - valley_ind
    else:
        spike_width = valley_ind - peak_ind

    return spike_width


def calc_template_half_width(template):
    """ Computes half width of spike as number of indices between peak and valley. """
    peak_ind = np.argmax(template)
    valley_ind = np.argmin(template)
    if peak_ind >= valley_ind:
        # peak is after valley
        spike_width = peak_ind - valley_ind
    else:
        spike_width = valley_ind - peak_ind

    return spike_width


def calc_fraction_mua_to_peak(spike_indices, sampling_rate, duplicate_tol_inds,
                         absolute_refractory_period, check_window=0.5):
    """ Estimates the fraction of multiunit activity/noise contained in the
    input spike indices relative to the peak firing rate.

    The ISI distribution is created by binning spike indices from duplicate
    tolerance indices to the end of check window with a bin width. The ISI
    violations are computed as the number of events between duplicate tolerance
    and the absolute refractory period. The returned fraction MUA is this
    number divided by the maximum ISI bin count. """
    all_isis = np.diff(spike_indices)
    refractory_inds = int(round(absolute_refractory_period * sampling_rate))
    bin_width = refractory_inds - duplicate_tol_inds
    if bin_width <= 0:
        print("LINE 555: duplicate_tol_inds encompasses absolute_refractory_period so fraction MUA cannot be computed.")
        return np.nan
    check_inds = int(round(check_window * sampling_rate))
    bin_edges = np.arange(duplicate_tol_inds, check_inds + bin_width, bin_width)
    counts, xval = np.histogram(all_isis, bin_edges)
    isi_peak = np.amax(counts)
    num_isi_violations = counts[0]
    if num_isi_violations < 0:
        num_isi_violations = 0
    if isi_peak == 0:
        isi_peak = max(num_isi_violations, 1.)
    fraction_mua_to_peak = num_isi_violations / isi_peak

    return fraction_mua_to_peak


def calc_isi_violation_rate(spike_indices, sampling_rate,
        absolute_refractory_period, duplicate_tol_inds):
    """ Estimates the fraction of multiunit activity/noise contained in the
    input spike indices relative to the average firing rate.

    The ISI violations are counted in the window from duplicate tolerance
    indices to the absolute refractory period. The units firing rate in this
    window is then divided by the average unit firing rate to compute the
    returned ISI violation firing rate. """
    all_isis = np.diff(spike_indices)
    duplicate_time = duplicate_tol_inds / sampling_rate
    if absolute_refractory_period - duplicate_time <= 0:
        print("LINE 583: duplicate_tol_inds encompasses absolute_refractory_period so fraction MUA cannot be computed.")
        return np.nan
    num_isi_violations = np.count_nonzero(all_isis / sampling_rate < absolute_refractory_period)
    n_duplicates = np.count_nonzero(all_isis <= duplicate_tol_inds)
    # Remove duplicate spikes from this computation and adjust the number
    # of spikes and time window accordingly
    num_isi_violations -= n_duplicates
    isi_violation_rate = num_isi_violations \
                         * (1.0 / (absolute_refractory_period - duplicate_time))\
                         / (spike_indices.shape[0] - n_duplicates)
    return isi_violation_rate


def mean_firing_rate(spike_indices, sampling_rate):
    """ Compute mean firing rate for input spike indices. Spike indices must be
    in sorted order for this to work. """
    if spike_indices[0] == spike_indices[-1]:
        return 0. # Only one spike (could be repeated)
    mean_rate = sampling_rate * spike_indices.size / (spike_indices[-1] - spike_indices[0])
    return mean_rate


def calc_fraction_mua(spike_indices, sampling_rate, duplicate_tol_inds,
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
        return (isi_violation_rate / mean_rate)
