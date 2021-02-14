import numpy as np
from scipy.stats import norm
from spikesorting_fullpursuit.consolidate import optimal_align_templates
from spikesorting_fullpursuit.parallel.segment_parallel import time_window_to_samples
from spikesorting_fullpursuit.analyze_spike_timing import find_overlapping_spike_bool



def compute_metrics(templates, voltage, n_noise_samples, sort_info, thresholds):
    """ Calculate variance and template sum squared metrics needed to compute
    separability_metrics between units and the delta likelihood function for binary
    pursuit. """

    # Ease of use variables
    n_chans = sort_info['n_channels']
    n_templates = len(templates)
    template_samples_per_chan = sort_info['n_samples_per_chan']
    window, clip_width = time_window_to_samples(sort_info['clip_width'], sort_info['sampling_rate'])
    max_clip_samples = np.amax(np.abs(window)) + 1

    separability_metrics = {}
    separability_metrics['templates'] = np.vstack(templates)
    # Compute our template sum squared error (see note below).
    separability_metrics['template_SS'] = np.sum(separability_metrics['templates'] ** 2, axis=1)
    separability_metrics['template_SS_by_chan'] = np.zeros((n_templates, n_chans))
    # Need to get sum squares and noise covariance separate for each channel and template
    separability_metrics['channel_covariance_mats'] = []


    for chan in range(0, n_chans):
        n_noise_clips_found = 0
        noise_clips_found = []
        while n_noise_clips_found < n_noise_samples:
            rand_inds = np.random.randint(max_clip_samples, voltage.shape[1] - max_clip_samples, n_noise_samples-n_noise_clips_found)
            noise_clips, _ = get_singlechannel_clips(voltage, chan, rand_inds, window)
            noise_clips = noise_clips[np.all(noise_clips < thresholds[chan], axis=1), :]
            n_noise_clips_found += noise_clips.shape[0]
            noise_clips_found.append(noise_clips)
        noise_clips_found = np.vstack(noise_clips_found)
        chan_cov = np.cov(noise_clips_found, rowvar=False)
        separability_metrics['channel_covariance_mats'].append(chan_cov)

    separability_metrics['neuron_biases'] = np.zeros(n_templates)
    for n in range(0, n_templates):
        for chan in range(0, n_chans):
            t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
            separability_metrics['template_SS_by_chan'][n, chan] = np.sum(
                    separability_metrics['templates'][n, t_win[0]:t_win[1]] ** 2)
            separability_metrics['neuron_biases'][n] += (separability_metrics['templates'][n, t_win[0]:t_win[1]][None, :]
                                            @ separability_metrics['channel_covariance_mats'][chan]
                                            @ separability_metrics['templates'][n, t_win[0]:t_win[1]][:, None])
        # Convert bias from variance to standard deviations
        separability_metrics['neuron_biases'][n] = sort_info['sigma_noise_penalty'] * np.sqrt(separability_metrics['neuron_biases'][n])

    separability_metrics['gamma_noise'] = np.zeros(n_chans)

    separability_metrics['std_noise'] = np.zeros(n_chans)
    # Compute bias separately for each neuron, on each channel
    for chan in range(0, n_chans):
        # Convert channel threshold to normal standard deviation
        separability_metrics['std_noise'][chan] = thresholds[chan] / sort_info['sigma']
        # gamma_noise is used only for overlap recheck indices noise term for sum of 2 templates
        separability_metrics['gamma_noise'][chan] = sort_info['sigma_noise_penalty'] * separability_metrics['std_noise'][chan]
        # for n in range(0, n_templates):
        #     separability_metrics['neuron_biases'][n] += np.sqrt(separability_metrics['template_SS_by_chan'][n, chan]) * separability_metrics['gamma_noise'][chan]

    return separability_metrics


def pairwise_separability(separability_metrics, sort_info):
    """
    Note that output matrix of separabilty errors is not necessarily symmetric
    due to the shifting alignment of templates.
    """
    n_chans = sort_info['n_channels']
    template_samples_per_chan = sort_info['n_samples_per_chan']
    max_shift = (template_samples_per_chan // 4) - 1
    n_neurons = separability_metrics['templates'].shape[0]

    # Compute separability measure for all pairs of neurons
    pair_separability_matrix = np.zeros((n_neurons, n_neurons))
    for n1 in range(0, n_neurons):
        E_L_n1 = 0.5 * separability_metrics['template_SS'][n1] - separability_metrics['neuron_biases'][n1]
        for n2 in range(0, separability_metrics['templates'].shape[0]):
            # Need to optimally align n2 template with n1 template for computing
            # the covariance/difference of their Likelihood functions
            shift_temp1, shift_temp2, optimal_shift, shift_samples_per_chan = optimal_align_templates(
                                separability_metrics['templates'][n1, :],
                                separability_metrics['templates'][n2, :],
                                n_chans, max_shift=max_shift, align_abs=True)
            var_diff = 0
            for chan in range(0, n_chans):
                s_win = [chan*shift_samples_per_chan, (chan+1)*shift_samples_per_chan]
                # Adjust covariance matrix to new templates, shifting according to template 1
                if optimal_shift >= 0:
                    s_chan_cov = separability_metrics['channel_covariance_mats'][chan][optimal_shift:, optimal_shift:]
                else:
                    s_chan_cov = separability_metrics['channel_covariance_mats'][chan][0:optimal_shift, 0:optimal_shift]
                # Multiply template with covariance then template transpose to
                # get variance of the likelihood function (or the difference)
                diff_template = shift_temp1[s_win[0]:s_win[1]] - shift_temp2[s_win[0]:s_win[1]]
                var_diff += (diff_template[None, s_win[0]:s_win[1]]
                             @ s_chan_cov
                             @ diff_template[s_win[0]:s_win[1], None])

            n_1_n2_SS = np.dot(shift_temp1, shift_temp2)
            # Assuming full template 2 data here as this will be counted in
            # computation of likelihood function
            E_L_n2 = (n_1_n2_SS - 0.5 * np.dot(separability_metrics['templates'][n2, :], separability_metrics['templates'][n2, :])
                        - separability_metrics['neuron_biases'][n2])

            E_diff_n1_n2 = E_L_n1 - E_L_n2
            print("exepcted diff", E_diff_n1_n2, "var diff", var_diff)
            if var_diff > 0:
                pair_separability_matrix[n1, n2] = norm.cdf(0, E_diff_n1_n2, np.sqrt(var_diff))
            else:
                pair_separability_matrix[n1, n2] = 0

    return pair_separability_matrix


def get_singlechannel_clips(voltage, channel, spike_times, window):

    if spike_times.ndim > 1:
        raise ValueError("Event_indices must be one dimensional array of indices")

    # Ignore spikes whose clips extend beyond the data and create mask for removing them
    valid_event_indices = np.ones(spike_times.shape[0], dtype=np.bool)
    start_ind = 0
    n = spike_times[start_ind]

    while (n + window[0]) < 0:
        valid_event_indices[start_ind] = False
        start_ind += 1
        if start_ind == spike_times.size:
            # There are no valid indices
            valid_event_indices[:] = False
            return None, valid_event_indices
        n = spike_times[start_ind]
    stop_ind = spike_times.shape[0] - 1
    n = spike_times[stop_ind]
    while (n + window[1]) >= voltage.shape[1]:
        valid_event_indices[stop_ind] = False
        stop_ind -= 1
        if stop_ind < 0:
            # There are no valid indices
            valid_event_indices[:] = False
            return None, valid_event_indices
        n = spike_times[stop_ind]

    spike_clips = np.empty((np.count_nonzero(valid_event_indices), window[1] - window[0]))
    for out_ind, spk in enumerate(range(start_ind, stop_ind+1)): # Add 1 to index through last valid index
        spike_clips[out_ind, :] = voltage[channel, spike_times[spk]+window[0]:spike_times[spk]+window[1]]

    return spike_clips, valid_event_indices


def empirical_separability(voltage, spike_times, templates, window_samples, separability_metrics, add_spikes=False):
    """ """

    overlapping_spike_bool_0_1 = find_overlapping_spike_bool(spike_times[0], spike_times[1], overlap_tol=window_samples[1])
    overlapping_spike_bool_1_0 = find_overlapping_spike_bool(spike_times[1], spike_times[0], overlap_tol=window_samples[1])
    spike_times[0] = spike_times[0][~overlapping_spike_bool_0_1]
    spike_times[1] = spike_times[1][~overlapping_spike_bool_1_0]

    LL_E_diff_mat = np.zeros((templates.shape[0], templates.shape[0]))
    LL_Var_diff_mat = np.zeros((templates.shape[0], templates.shape[0]))
    misclassification_errors = np.zeros((templates.shape[0], templates.shape[0]))
    n_output_clips = []
    for n1 in range(0, len(spike_times)):
        spike_clips, valid_event_indices = get_multichannel_clips(
                            voltage, spike_times[n1]+1, window_samples)

        n_output_clips.append(spike_clips)
        LL = []
        for n2 in range(0, len(spike_times)):

            LL.append(spike_clips @ templates[n2, :][:, None] - 0.5*np.sum(templates[n2, :] ** 2) - separability_metrics['neuron_biases'][n2])

            var_L_n1 = 0
            template_samples_per_chan = templates.shape[1] // voltage.shape[0]
            for chan in range(0, voltage.shape[0]):
                t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
                # Multiply template with covariance then template transpose
                var_L_n1 += (templates[n2, :][None, t_win[0]:t_win[1]]
                                 @ separability_metrics['channel_covariance_mats'][chan]
                                 @ templates[n2, :][t_win[0]:t_win[1], None])
            print("VAR", n2, "given", n1, np.var(LL[-1]), var_L_n1)

        for n2 in range(0, len(spike_times)):
            LL_E_diff_mat[n1, n2] = np.mean(LL[n1] - LL[n2])
            LL_Var_diff_mat[n1, n2] = np.var(LL[n1] - LL[n2])

            print("Empirical EXP of DIFFERENCE", LL_E_diff_mat[n1, n2])

            E_L_n1 = 0.5 * np.sum(templates[n1, :] ** 2) - separability_metrics['neuron_biases'][n1]
            E_L_n2 = (np.dot(templates[n1, :], templates[n2, :])
                      -0.5 * np.sum(templates[n2, :] ** 2) - separability_metrics['neuron_biases'][n2])
            print("Calculated EXP of DIFFERENCE", E_L_n1 - E_L_n2)

            var_diff = 0
            template_samples_per_chan = templates.shape[1] // voltage.shape[0]
            for chan in range(0, voltage.shape[0]):
                t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
                # Multiply template with covariance then template transpose
                diff_template = templates[n1, :] - templates[n2, :]
                var_diff += (diff_template[None, t_win[0]:t_win[1]]
                                 @ separability_metrics['channel_covariance_mats'][chan]
                                 @ diff_template[t_win[0]:t_win[1], None])
            print("Empirical VAR of DIFFERENCE", LL_Var_diff_mat[n1, n2])
            print("Calculated VAR of DIFFERENCE", var_diff)

            misclassification_errors[n1, n2] = np.count_nonzero(LL[n2] > LL[n1]) / spike_clips.shape[0]

    return misclassification_errors, LL_E_diff_mat, LL_Var_diff_mat, n_output_clips


def get_multichannel_clips(voltage, spike_times, window):

    if spike_times.ndim > 1:
        raise ValueError("Event_indices must be one dimensional array of indices")

    # Ignore spikes whose clips extend beyond the data and create mask for removing them
    valid_event_indices = np.ones(spike_times.shape[0], dtype=np.bool)
    start_ind = 0
    n = spike_times[start_ind]

    while (n + window[0]) < 0:
        valid_event_indices[start_ind] = False
        start_ind += 1
        if start_ind == spike_times.size:
            # There are no valid indices
            valid_event_indices[:] = False
            return None, valid_event_indices
        n = spike_times[start_ind]
    stop_ind = spike_times.shape[0] - 1
    n = spike_times[stop_ind]
    while (n + window[1]) >= voltage.shape[1]:
        valid_event_indices[stop_ind] = False
        stop_ind -= 1
        if stop_ind < 0:
            # There are no valid indices
            valid_event_indices[:] = False
            return None, valid_event_indices
        n = spike_times[stop_ind]
    spike_clips = np.empty((np.count_nonzero(valid_event_indices), (window[1] - window[0]) * voltage.shape[0]))
    for out_ind, spk in enumerate(range(start_ind, stop_ind+1)): # Add 1 to index through last valid index
        chan_ind = 0
        start = 0
        for chan in range(0, voltage.shape[0]):
            chan_ind += 1
            stop = chan_ind * (window[1] - window[0])
            spike_clips[out_ind, start:stop] = voltage[chan, spike_times[spk]+window[0]:spike_times[spk]+window[1]]
            # Subtract start ind above to adjust for discarded events
            start = stop

    return spike_clips, valid_event_indices
