import numpy as np



def compute_metrics(templates, voltage, sort_info, thresholds=None):
    """ Calculate variance and template sum squared metrics needed to compute
    separability_metrics between units and the delta likelihood function for binary
    pursuit. """

    # Ease of use variables
    n_chans = voltage.shape[0]
    n_templates = templates.shape[0]
    template_samples_per_chan = sort_info['n_samples_per_chan']

    separability_metrics = {}
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
