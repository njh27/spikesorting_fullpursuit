import numpy as np
from copy import deepcopy
import os
from spikesorting_fullpursuit.parallel.segment_parallel import time_window_to_samples, get_singlechannel_clips, get_multichannel_clips
from spikesorting_fullpursuit.consolidate import SegSummary
from spikesorting_fullpursuit.preprocessing import calculate_robust_template



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
def wiener(original_voltage, signal_voltage, noise_voltage, smooth,
           data_dict, seg_number, sort_info):

    if original_voltage.ndim == 1:
        original_voltage = np.expand_dims(original_voltage, 0)
        signal_voltage = np.expand_dims(signal_voltage, 0)
        noise_voltage = np.expand_dims(noise_voltage, 0)
    assert (signal_voltage.shape == original_voltage.shape) and (noise_voltage.shape == original_voltage.shape)

    original_mem = np.fft.rfft(original_voltage, axis=1) # Input in frequency domain
    o_ft_fname = os.path.join(sort_info['memmap_dir'], "orig_ft{0}.bin".format(str(seg_number)))
    original_ft = np.memmap(o_ft_fname,
                            dtype=original_mem.dtype,
                            mode='w+', shape=original_mem.shape)
    original_ft[:] = original_mem[:]
    original_ft.flush()
    del original_mem

    S_mem = abs2(np.fft.rfft(signal_voltage, axis=1)) # Signal power spectrum
    S_ft_fname = os.path.join(sort_info['memmap_dir'], "S_ft{0}.bin".format(str(seg_number)))
    S = np.memmap(S_ft_fname,
                  dtype=S_mem.dtype,
                  mode='w+', shape=S_mem.shape)
    S[:] = S_mem[:]
    S.flush()
    del S_mem

    N_mem = abs2(np.fft.rfft(noise_voltage, axis=1)) # Noise power spectrum
    N_ft_fname = os.path.join(sort_info['memmap_dir'], "N_ft{0}.bin".format(str(seg_number)))
    N = np.memmap(N_ft_fname,
                  dtype=N_mem.dtype,
                  mode='w+', shape=N_mem.shape)
    N[:] = N_mem[:]
    N.flush()
    del N_mem

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

    # Cleanup memmaps
    original_ft._mmap.close()
    del original_ft
    os.remove(o_ft_fname)
    S._mmap.close()
    del S
    os.remove(S_ft_fname)
    N._mmap.close()
    del N
    os.remove(N_ft_fname)

    return filtered_signal


def wiener_all(original_voltage, signal_voltage, noise_voltage, smooth,
               data_dict, seg_number, sort_info):
    """ Performs Wiener filter over data across all channels in voltage at once
    using the same filter for every channel. The idea is to help avoid
    potential pitfalls of using a very small/noisy "signal" on channels without
    threshold crossings that could in turn amplify this noise via filtering.
    """
    assert (signal_voltage.shape == original_voltage.shape) and (noise_voltage.shape == original_voltage.shape)
    voltage_shape = original_voltage.shape
    # get 1D view of input voltage arrays
    ov = original_voltage.ravel(order="C")
    sv = signal_voltage.ravel(order="C")
    nv = noise_voltage.ravel(order="C")
    # Get FFTs and memmap them
    original_mem = np.fft.rfft(ov) # Input in frequency domain
    o_ft_fname = os.path.join(sort_info['memmap_dir'], "orig_ft{0}.bin".format(str(seg_number)))
    original_ft = np.memmap(o_ft_fname,
                            dtype=original_mem.dtype,
                            mode='w+', shape=original_mem.shape)
    original_ft[:] = original_mem[:]
    original_ft.flush()
    del original_mem

    S_mem = abs2(np.fft.rfft(sv)) # Signal power spectrum
    S_ft_fname = os.path.join(sort_info['memmap_dir'], "S_ft{0}.bin".format(str(seg_number)))
    S = np.memmap(S_ft_fname,
                  dtype=S_mem.dtype,
                  mode='w+', shape=S_mem.shape)
    S[:] = S_mem[:]
    S.flush()
    del S_mem

    N_mem = abs2(np.fft.rfft(nv)) # Noise power spectrum
    N_ft_fname = os.path.join(sort_info['memmap_dir'], "N_ft{0}.bin".format(str(seg_number)))
    N = np.memmap(N_ft_fname,
                  dtype=N_mem.dtype,
                  mode='w+', shape=N_mem.shape)
    N[:] = N_mem[:]
    N.flush()
    del N_mem

    smooth = int(smooth)
    if smooth > 1:
        S_smoothed = np.zeros(S.shape[0])
        N_smoothed = np.zeros(N.shape[0])
        # Rolling sum implementation of box-car filter separately of N
        # and S using the binsize passed in the `smooth` variable.
        half_bin_size = int(smooth / 2)
        rolling_sum_num_points = min(half_bin_size, N.shape[0])
        N_rolling_sum = np.sum(N[0:rolling_sum_num_points])
        S_rolling_sum = np.sum(S[0:rolling_sum_num_points])
        for i in range(0, S.shape[0]):
            if i + half_bin_size < S.shape[0]:
                N_rolling_sum += N[i+half_bin_size]
                S_rolling_sum += S[i+half_bin_size]
                rolling_sum_num_points += 1
            if ( (i - half_bin_size) > 0 ):
                N_rolling_sum -= N[i-half_bin_size]
                S_rolling_sum -= S[i-half_bin_size]
                rolling_sum_num_points -= 1
            S_smoothed[i] = S_rolling_sum / rolling_sum_num_points
            N_smoothed[i] = N_rolling_sum / rolling_sum_num_points
    else:
        S_smoothed = S
        N_smoothed = N

    # Cleanup memmaps since using "smoothed" array now
    original_ft._mmap.close()
    del original_ft
    os.remove(o_ft_fname)
    S._mmap.close()
    del S
    os.remove(S_ft_fname)
    N._mmap.close()
    del N
    os.remove(N_ft_fname)

    filtered_signal = np.fft.irfft(original_ft * wiener_optimal_filter(S_smoothed, N_smoothed))
    filtered_signal = np.reshape(filtered_signal, voltage_shape, order="C")

    return filtered_signal


def wiener_optimal_filter(signal_power, noise_power, epsilon=1e-9):
    # Without blurring, the filter is:
    #   |S|² / (|S|² + |N|²)
    return (signal_power) / (signal_power + noise_power + epsilon)


def wiener_filter_segment(work_items, data_dict, seg_number, sort_info,
                            v_dtype):
    """ Does the Wiener filter on the segment voltage provided. The new filtered
    voltage OVERWRITES the input segment voltage! """

    # Initialize voltages
    voltage = np.memmap(data_dict['seg_v_files'][seg_number][0],
                        dtype=data_dict['seg_v_files'][seg_number][1],
                        mode='r+',
                        shape=data_dict['seg_v_files'][seg_number][2])
    v_noise_fname = os.path.join(sort_info['memmap_dir'], "volt_noise{0}.bin".format(str(seg_number)))
    volt_noise = np.memmap(v_noise_fname,
                           dtype=data_dict['seg_v_files'][seg_number][1],
                           mode='w+', shape=data_dict['seg_v_files'][seg_number][2])
    volt_noise[:] = voltage[:]
    volt_noise.flush()
    v_signal_fname = os.path.join(sort_info['memmap_dir'], "volt_signal{0}.bin".format(str(seg_number)))
    volt_signal = np.memmap(v_signal_fname,
                           dtype=data_dict['seg_v_files'][seg_number][1],
                           mode='w+', shape=data_dict['seg_v_files'][seg_number][2])

    # Determine the set of work items for this segment
    seg_w_items = [w for w in work_items if w['seg_number'] == seg_number]
    # Make a dictionary with all info needed for get_multichannel_clips
    clips_dict = {'sampling_rate': sort_info['sampling_rate'],
                  'n_samples': seg_w_items[0]['n_samples'],
                  'v_dtype': v_dtype}
    clip_window, clip_width = time_window_to_samples(sort_info['clip_width'], sort_info['sampling_rate'])

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
                              np.zeros(len(data_dict['results_dict'][w_item['ID']][0]), dtype="bool"),
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
                        absolute_refractory_period=sort_info['absolute_refractory_period'],
                        verbose=False)
    if len(seg_summary.summaries) == 0:
        print("Found no neuron templates for Wiener filter!")
        return None
    print("SHARPENING WIENERS!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("start neurons", len(seg_summary.summaries))
    seg_summary.sharpen_across_chans()
    print("sharpened neurons", len(seg_summary.summaries))

    # Need to go through each neuron for this seg_number
    for n in seg_summary.summaries:
        # Get the clips for each channel in neighborhood of this work item
        for chan in n['neighbors']:
            clips, valid_inds = get_singlechannel_clips(clips_dict,
                                    voltage[chan, :],
                                    n['spike_indices'],
                                    sort_info['clip_width'])
            # Make noise and signal by adding/subtracting this neurons template
            n['spike_indices'] = n['spike_indices'][valid_inds]
            robust_template = calculate_robust_template(clips)
            for spk in n['spike_indices']:
                volt_noise[chan, spk+clip_window[0]:spk+clip_window[1]] -= robust_template
                volt_signal[chan, spk+clip_window[0]:spk+clip_window[1]] += robust_template

    # Put back the original neighbors to work items since changed in-place
    for w_item_ind, w_item in enumerate(seg_w_items):
        if w_item['ID'] in data_dict['results_dict'].keys():
            w_item['neighbors'] = original_neighbors[w_item_ind]

    volt_noise.flush()
    volt_signal.flush()
    print("Starting Wiener filter")
    if ( (sort_info['wiener_filter_smoothing'] is None) or
         (sort_info['wiener_filter_smoothing'] < 1) ):
        wiener_filter_smooth_indices = 0
    elif sort_info['same_wiener']:
        # Use the same Wiener filter for all channels computed over all data
        wiener_filter_smooth_indices = ( (sort_info['wiener_filter_smoothing'] * voltage.size)
                                        / (sort_info['sampling_rate'] // 2) )
        filtered_voltage = wiener_all(voltage, volt_signal, volt_noise,
                                      wiener_filter_smooth_indices, data_dict,
                                      seg_number, sort_info)
    else:
        wiener_filter_smooth_indices = ( (sort_info['wiener_filter_smoothing'] * voltage.shape[1])
                                        / (sort_info['sampling_rate'] // 2) )
        filtered_voltage = wiener(voltage, volt_signal, volt_noise,
                                  wiener_filter_smooth_indices, data_dict,
                                  seg_number, sort_info)

    # Rescale filtered voltage to original space and Copy Winer filter segment
    # voltage to the raw array buffer so we can re-use it for sorting
    wiener_scale = (np.std(voltage, axis=1) / np.std(filtered_voltage, axis=1))
    filtered_voltage = filtered_voltage * wiener_scale[:, None]
    np.copyto(voltage, filtered_voltage)

    # Delete memmap references and files made for Wiener filter
    volt_noise._mmap.close()
    del volt_noise
    os.remove(v_noise_fname)
    volt_signal._mmap.close()
    del volt_signal
    os.remove(v_signal_fname)

    return None

#
