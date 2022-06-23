import numpy as np
from spikesorting_fullpursuit.parallel.segment_parallel import time_window_to_samples, get_singlechannel_clips



def abs2(x):
    """ Gets absolute value squared of numpy complex number """
    return x.real**2 + x.imag**2


"""
    wiener(input, signal, noise)

Compute the wiener filter on the input signal given an estimate of the
true (noiseless) signal and an estimate of the noise (signal removed).

This function optimally smooths the noise and signal power spectra using
a running average specified by `smooth`. The smooth parameter is provided
in integer samples (i.e., the boxcar window).
"""
def wiener(original_voltage, signal_voltage, noise_voltage, smooth=1):

    assert original_voltage.ndim == 2
    assert (signal_voltage.shape == original_voltage.shape) and (noise_voltage.shape == original_voltage.shape)

    original_ft = np.fft.rfft(original_voltage, axis=1) # Input in frequency domain
    S = abs2(np.fft.rfft(signal_voltage, axis=1)) # Signal power spectrum
    N = abs2(np.fft.rfft(noise_voltage, axis=1)) # Noise power spectrum
    smooth = int(smooth)
    if smooth > 1:
        S_smoothed = np.zeros(S.shape[1])
        N_smoothed = np.zeros(N.shape[1])
        for chan in range(0, S.shape[0]):
            # Rolling sum implementation of box-car filter separately of N
            # and S using the binsize passed in the `smooth` variable.
            half_bin_size = int(smooth / 2)
            rolling_sum_num_points = min(half_bin_size, N.shape[1])
            N_rolling_sum = np.sum(N[chan, 0:rolling_sum_num_points])
            S_rolling_sum = np.sum(S[chan, 0:rolling_sum_num_points])
            for i in range(0, S.shape[1]):
                if i + half_bin_size < S.shape[1]:
                    N_rolling_sum += N[chan, i+half_bin_size]
                    S_rolling_sum += S[chan, i+half_bin_size]
                    rolling_sum_num_points += 1
                if ( (i - half_bin_size) > 0 ):
                    N_rolling_sum -= N[chan, i-half_bin_size]
                    S_rolling_sum -= S[chan, i-half_bin_size]
                    rolling_sum_num_points -= 1
                S_smoothed[i] = S_rolling_sum / rolling_sum_num_points
                N_smoothed[i] = N_rolling_sum / rolling_sum_num_points
            S[chan, :] = S_smoothed
            N[chan, :] = N_smoothed
            # Reset to zeros (not necessary but just in case)
            S_smoothed[:] = 0.
            N_smoothed[:] = 0.

    filtered_signal = np.fft.irfft(original_ft * wiener_optimal_filter(S, N), axis=1)
    return filtered_signal


def wiener_optimal_filter(signal_power, noise_power, epsilon=1e-9):
    # Without blurring, the filter is:
    #   |S|² / (|S|² + |N|²)
    return (signal_power) / (signal_power + noise_power + epsilon)


def wiener_filter_segment(work_items, data_dict, seg_number, sort_info,
                            v_dtype):
    """ Does the Wiener filter on the segment voltage provided. The new filtered
    voltage OVERWRITES the input segment voltage! """

    # Get numpy view of voltage for clips and binary pursuit
    seg_volts_buffer = data_dict['segment_voltages'][seg_number][0]
    seg_volts_shape = data_dict['segment_voltages'][seg_number][1]
    voltage = np.frombuffer(seg_volts_buffer, dtype=v_dtype).reshape(seg_volts_shape)
    volt_noise = np.copy(voltage)
    volt_signal = np.zeros(seg_volts_shape, dtype=v_dtype)

    # Determine the set of work items for this segment
    seg_w_items = [w for w in work_items if w['seg_number'] == seg_number]
    # Make a dictionary with all info needed for get_multichannel_clips
    clips_dict = {'sampling_rate': sort_info['sampling_rate'],
                  'n_samples': seg_w_items[0]['n_samples'],
                  'v_dtype': v_dtype}
    clip_window, clip_width = time_window_to_samples(sort_info['clip_width'], sort_info['sampling_rate'])

    # Need to go through each work item for this seg_number
    for w_item in seg_w_items:
        if w_item['ID'] in data_dict['results_dict'].keys():
            if len(data_dict['results_dict'][w_item['ID']][0]) == 0:
                # This work item found nothing (or raised an exception)
                continue
            # Get the clips for each channel in neighborhood of this work item
            for chan in w_item['neighbors']:
                clips, valid_inds = get_singlechannel_clips(clips_dict,
                                        voltage[chan, :],
                                        data_dict['results_dict'][w_item['ID']][0],
                                        sort_info['clip_width'])
                if clips is None:
                    raise RuntimeError("NO CLIPS?")
                # For each neuron compute its template
                for n_id in np.unique(data_dict['results_dict'][w_item['ID']][1]):
                    n_inds = data_dict['results_dict'][w_item['ID']][1] == n_id
                    n_template = np.mean(clips[n_inds, :], axis=0)
                    n_inds = np.logical_and(n_inds, valid_inds)
                    # Subtract the template of each neuron and add to signal
                    for spk in data_dict['results_dict'][w_item['ID']][0][n_inds]:
                        volt_noise[chan, spk+clip_window[0]:spk+clip_window[1]] -= n_template
                        volt_signal[chan, spk+clip_window[0]:spk+clip_window[1]] += n_template
        else:
            # This work item found nothing (or raised an exception)
            continue

    if ( (sort_info['wiener_filter_smoothing'] is None) or
         (sort_info['wiener_filter_smoothing'] < 1) ):
        wiener_filter_smooth_indices = 0
    else:
        wiener_filter_smooth_indices = ( (sort_info['wiener_filter_smoothing'] * voltage.shape[1])
                                        / (sort_info['sampling_rate'] // 2) )
    filtered_voltage = wiener(voltage, volt_signal, volt_noise, wiener_filter_smooth_indices)
    filtered_voltage = filtered_voltage * (np.std(voltage, axis=1) / np.std(filtered_voltage, axis=1))[:, None]
    # Copy Winer filter segment voltage to the raw array buffer so we
    # can re-use it for sorting
    np.copyto(voltage, filtered_voltage)

    return filtered_voltage

#
