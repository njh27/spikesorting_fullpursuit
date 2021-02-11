import numpy as np
from scipy.stats import norm
from spikesorting_fullpursuit.consolidate import optimal_align_templates



def compute_metrics(templates, neuron_clips, sort_info, thresholds):
    """ Calculate variance and template sum squared metrics needed to compute
    separability_metrics between units and the delta likelihood function for binary
    pursuit. """

    # Ease of use variables
    n_chans = sort_info['n_channels']
    n_templates = len(templates)
    template_samples_per_chan = sort_info['n_samples_per_chan']

    separability_metrics = {}
    separability_metrics['templates'] = np.vstack(templates)
    # Compute our template sum squared error (see note below).
    separability_metrics['template_SS'] = np.sum(separability_metrics['templates'] ** 2, axis=1)
    separability_metrics['template_SS_by_chan'] = np.zeros((n_templates, n_chans))
    # Need to get sum squares and noise covariance separate for each channel and template
    separability_metrics['neuron_covariance_mats'] = []
    for n in range(0, n_templates):
        chan_covariance_mats = []
        for chan in range(0, n_chans):
            t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
            chan_cov = np.cov(neuron_clips[n][:, t_win[0]:t_win[1]], rowvar=False)
            chan_covariance_mats.append(chan_cov)
            separability_metrics['template_SS_by_chan'][n, chan] = np.sum(
                    separability_metrics['templates'][n, t_win[0]:t_win[1]] ** 2)
        separability_metrics['neuron_covariance_mats'].append(chan_covariance_mats)

    separability_metrics['gamma_noise'] = np.zeros(n_chans)
    separability_metrics['neuron_biases'] = np.zeros(n_templates)
    separability_metrics['std_noise'] = np.zeros(n_chans)
    # Compute bias separately for each neuron, on each channel
    for chan in range(0, n_chans):
        # Convert channel threshold to normal standard deviation
        separability_metrics['std_noise'][chan] = thresholds[chan] / sort_info['sigma']
        separability_metrics['gamma_noise'][chan] = sort_info['sigma_noise_penalty'] * separability_metrics['std_noise'][chan]
        for n in range(0, n_templates):
            separability_metrics['neuron_biases'][n] += np.sqrt(separability_metrics['template_SS_by_chan'][n, chan]) * separability_metrics['gamma_noise'][chan]

    return separability_metrics


def pairwise_separability(separability_metrics, sort_info):

    n_chans = sort_info['n_channels']
    template_samples_per_chan = sort_info['n_samples_per_chan']
    print("!!! ADJUSTED TEMPLATE SIZE SINCE WAS SCREWED UP AT LINE 53 !!!!")
    template_samples_per_chan -= 1
    max_shift = (template_samples_per_chan // 4) - 1
    n_neurons = separability_metrics['templates'].shape[0]

    # Compute separability measure for all pairs of neurons
    pair_separability_matrix = np.zeros((n_neurons, n_neurons))
    for n1 in range(0, n_neurons):
        var_L_n1 = 0
        for chan in range(0, n_chans):
            t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
            # Multiply template with covariance then template transpose
            var_L_n1 += (2 * separability_metrics['templates'][n1, :][None, t_win[0]:t_win[1]]
                             @ separability_metrics['neuron_covariance_mats'][n1][chan]
                             @ separability_metrics['templates'][n1, :][t_win[0]:t_win[1], None])

        E_L_n1 = 0.5 * separability_metrics['template_SS'][n1] - separability_metrics['neuron_biases'][n1]

        print("EXP", n1, E_L_n1)
        print("VAR", n1, var_L_n1)

        for n2 in range(0, separability_metrics['templates'].shape[0]):
            var_L_n2 = 0
            cov_L_n1_L_n2 = 0
            # Need to optimally align n2 template with n1 template for computing
            # the covariance of their Likelihood functions
            shift_temp1, shift_temp2, _, shift_samples_per_chan = optimal_align_templates(
                                separability_metrics['templates'][n1, :],
                                separability_metrics['templates'][n2, :],
                                n_chans, max_shift=max_shift, align_abs=True)

            for chan in range(0, n_chans):
                t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
                var_L_n2 += (2 * separability_metrics['templates'][n2, :][None, t_win[0]:t_win[1]]
                                 @ separability_metrics['neuron_covariance_mats'][n1][chan]
                                 @ separability_metrics['templates'][n2, :][t_win[0]:t_win[1], None])
                # Use product of shifted aligned templates for covariance
                s_win = [chan*shift_samples_per_chan, (chan+1)*shift_samples_per_chan]
                cov_L_n1_L_n2 += (2 * shift_temp1[None, s_win[0]:s_win[1]]
                                  @ separability_metrics['neuron_covariance_mats'][n1][chan] 
                                  @ shift_temp2[s_win[0]:s_win[1], None])


            n_1_n2_SS = np.sum(shift_temp1 * shift_temp2)
            E_L_n2 =  (n_1_n2_SS - 0.5 * np.sum(shift_temp2 ** 2)
                        - separability_metrics['neuron_biases'][n2])

            E_diff_n1_n2 = E_L_n1 - E_L_n2
            # print("EXP 1", E_L_n1, "EXP 2", E_L_n2)
            # print("VAR 1", var_L_n1, "VAR 2", var_L_n2)
            Var_diff_n1_n2 = var_L_n1 + var_L_n2 - 2 * cov_L_n1_L_n2
            # print("EXP DIFF", E_diff_n1_n2)
            # print("VAR DIFF", Var_diff_n1_n2)

            if Var_diff_n1_n2 > 0:
                pair_separability_matrix[n1, n2] = norm.cdf(0, E_diff_n1_n2, np.sqrt(Var_diff_n1_n2))
            else:
                pair_separability_matrix[n1, n2] = 0

    return pair_separability_matrix


def empirical_separability(voltage, spike_times, templates, window_samples, separability_metrics, add_spikes=False):
    """ """

    LL_E_diff_mat = np.zeros((templates.shape[0], templates.shape[0]))
    LL_Var_diff_mat = np.zeros((templates.shape[0], templates.shape[0]))
    misclassification_errors = np.zeros((templates.shape[0], templates.shape[0]))
    n_output_clips = []
    for neuron_clips in range(0, len(spike_times)):
        spike_clips, valid_event_indices = get_multichannel_clips(
                            voltage, spike_times[neuron_clips], window_samples)

        n_output_clips.append(spike_clips)
        LL = []
        for neuron_LL in range(0, len(spike_times)):
            LL.append(spike_clips @ templates[neuron_LL, :][:, None] - 0.5*np.sum(templates[neuron_LL, :] ** 2) - separability_metrics['neuron_biases'][neuron_LL])

            if neuron_clips == neuron_LL:
                print("EXP", neuron_LL, "for V=", neuron_clips, np.mean(LL[-1]))
                print("VAR", neuron_LL, "for V=", neuron_clips, np.var(LL[-1]))



        for neuron_LL in range(0, len(spike_times)):
            LL_E_diff_mat[neuron_clips, neuron_LL] = np.mean(LL[neuron_clips] - LL[neuron_LL])
            LL_Var_diff_mat[neuron_clips, neuron_LL] = np.var(LL[neuron_clips] - LL[neuron_LL])

            misclassification_errors[neuron_clips, neuron_LL] = np.count_nonzero(LL[neuron_LL] > LL[neuron_clips]) / spike_clips.shape[0]

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
