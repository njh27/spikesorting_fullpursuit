import numpy as np
from spikesorting_fullpursuit.consolidate import optimal_align_templates


def compute_metrics(templates, voltage, sort_info, thresholds=None):
    """ Calculate variance and template sum squared metrics needed to compute
    separability_metrics between units and the delta likelihood function for binary
    pursuit. """

    # Ease of use variables
    n_chans = voltage.shape[0]
    n_templates = templates.shape[0]
    template_samples_per_chan = sort_info['n_samples_per_chan']

    separability_metrics = {}
    separability_metrics['templates'] = templates
    # Compute our template sum squared error (see note below).
    # This is a num_templates vector
    separability_metrics['template_SS'] = np.sum(templates * templates, axis=1)
    separability_metrics['template_SS_by_chan'] = np.zeros((n_templates, n_chans))
    # Need to get convolution kernel separate for each channel and each template
    for n in range(0, n_templates):
        for chan in range(0, n_chans):
            t_win = [chan*template_samples_per_chan, chan*template_samples_per_chan + template_samples_per_chan]
            separability_metrics['template_SS_by_chan'][n, chan] = np.sum(templates[n, t_win[0]:t_win[1]] ** 2)

    if thresholds is None:
        thresholds = np.empty((n_chans, ))
        for chan in range(0, voltage.shape[0]):
            abs_voltage = np.abs(voltage[chan, :])
            thresholds[chan] = sort_info['sigma'] * np.nanmedian(abs_voltage) / 0.6745

    separability_metrics['gamma_noise'] = np.zeros(n_chans)
    separability_metrics['neuron_biases'] = np.zeros(templates.shape[0])
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

    n_chans = sort_info['n_chans']
    samples_per_chan = sort_info['n_samples_per_chan']
    max_xcorr_shift = (samples_per_chan // 2) - 1

    # Compute separability measure for all pairs of neurons
    pair_separability_matrix = np.zeros((separability_metrics['templates'].shape[0], separability_metrics['templates'].shape[0]))
    for n1 in range(0, separability_metrics['templates'].shape[0]):
        var_L_n1 = 0
        for chan in range(0, n_chans):
            var_L_n1 += (separability_metrics['std_noise'][chan] ** 2) * separability_metrics['template_SS_by_chan'][n1, chan]
        E_L_n1 = 0.5 * separability_metrics['template_SS'][n1] - separability_metrics['neuron_biases'][n1]

        for n2 in range(n1, separability_metrics['templates'].shape[0]):
            var_L_n2 = 0
            cov_L_n1_L_n2 = 0

            # Need to optimally align n2 template with n1 template
            optimal_shift, shift_temp1, shift_temp2 = optimal_align_templates(
                                separability_metrics['templates'][n1, :],
                                separability_metrics['templates'][n2, :],
                                n_chans, max_shift=max_xcorr_shift)

            for chan in range(0, n_chans):
                chan_var_noise = separability_metrics['std_noise'][chan] ** 2
                var_L_n2 += chan_var_noise * separability_metrics['template_SS_by_chan'][n2, chan]

                chan_win = [chan*samples_per_chan, chan*samples_per_chan + samples_per_chan]
                separability_metrics['template_SS_by_chan'][n, chan] = np.sum(templates[n, t_win[0]:t_win[1]] ** 2)

                cov_L_n1_L_n2 +=
            n_1_n2_SS = np.amax()
            E_L_n2 =  (n_1_n2_SS - 0.5 * separability_metrics['template_SS'][n2]
                        - separability_metrics['neuron_biases'][n2])

            E_diff_n1_n2 = E_L_n1 - E_L_n2
            cov_L_n1_L_n2 = n_1_n2_SS *
            Var_diff_n1_n2 = var_L_n1 + var_L_n2 - 2 * cov_L_n1_L_n2

            d_prime = np.abs(E_L_n1 - E_L_n2) / np.sqrt(0.5 * (var_L_n1 + var_L_n2))
            pair_separability_matrix[n1, n2] = d_prime
            pair_separability_matrix[n2, n1] = d_prime




    return
