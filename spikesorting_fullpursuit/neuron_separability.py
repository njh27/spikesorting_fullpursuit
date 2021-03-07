import numpy as np
from scipy.stats import norm
from spikesorting_fullpursuit.consolidate import optimal_align_templates
from spikesorting_fullpursuit.analyze_spike_timing import find_overlapping_spike_bool
from spikesorting_fullpursuit.parallel.segment_parallel import time_window_to_samples

import matplotlib.pyplot as plt



def compute_template_likelihood_variance(template, chan_covariance_mats,
                                         template_samples_per_chan):

    n_chans = len(chan_covariance_mats)

    for chan in range(0, n_chans):
        t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
        separability_metrics['template_SS_by_chan'][n, chan] = np.sum(
                separability_metrics['templates'][n, t_win[0]:t_win[1]] ** 2)

        separability_metrics['neuron_variances'][n] += (separability_metrics['templates'][n, t_win[0]:t_win[1]][None, :]
                    @ separability_metrics['channel_covariance_mats'][chan]
                    @ separability_metrics['templates'][n, t_win[0]:t_win[1]][:, None])


def decision_boundary_equal_var(mu_1, mu_2, var, p_1):
    """ Helper function to return the decision boundary between two 1D
    Gaussians with equal variance and Gaussian with mean=mu_1 has prior
    probability of p_1.
    This is the solution to:
        p_1 * N(mu_1, sqrt(var)) = p_2 * N(mu_2, sqrt(var))
    where N(mu, sigma) is the normal distribution PDF with mean=mu and
    standard deviation=sigma."""
    if p_1 < 0 or p_1 > 1:
        raise ValueError("Prior probably must lie in the interval [0, 1]")
    if var <= 0:
        raise ValueError("Variance var must be a positive number")
    if mu_1 == mu_2:
        raise ValueError("Means of the Gaussians cannot be equal")
    p_2 = 1 - p_1
    numerator = mu_1 ** 2 - mu_2 ** 2
    denominator = 2 * (mu_1 - mu_2)
    numerator_ll = 2 * var * np.log(p_1/p_2)
    decision_boundary = numerator / denominator - numerator_ll / denominator
    return decision_boundary


def decision_boundary(mu_1, mu_2, var_1, var_2):
    """ Helper function to return the decision boundary between 1D Gaussians with
    unequal variance but equal prior probabilities.
    This is the solution to:
        N(mu_1, sqrt(var_1)) = N(mu_2, sqrt(var_2))
    where N(mu, sigma) is the normal distribution PDF with mean=mu and
    standard deviation=sigma."""
    if var_1 <= 0 or var_2 <= 0:
        raise ValueError("Variances must be a positive numbers")
    if mu_1 == mu_2:
        raise ValueError("Means of the Gaussians cannot be equal")
    # coefficients of quadratic equation ax^2 + bx + c = 0
    a = var_1 - var_2
    b = 2 * (mu_1 * var_2 - mu_2 * var_1)
    c = mu_2 ** 2.0 * var_1 - mu_1 ** 2.0 * var_2 - 2 * var_1 * var_2 * np.log(np.sqrt(var_1)/np.sqrt(var_2))
    x1 = (-b + np.sqrt(b**2.0 - 4.0 * a * c)) / (2.0 * a)
    x2 = (-b - np.sqrt(b**2.0 - 4.0 * a * c)) / (2.0 * a)
    # Choose the solution where the PDFs are maximized
    if norm.pdf(x1, mu_1, np.sqrt(var_1)) > norm.pdf(x2, mu_1, np.sqrt(var_1)):
        decision_boundary = x1
    else:
        decision_boundary = x2
    return decision_boundary


def check_template_pair(template_1, template_2, chan_covariance_mats, sort_info):
    """
    Intended for testing whether a sum of templates is equal to a given template.
    Templates are assumed to be aligned with one another as no shifting is performed.
    """
    n_chans = sort_info['n_channels']
    template_samples_per_chan = sort_info['n_samples_per_chan']

    # Compute separability given V = template_1. Since we are not using the
    # noise bias term here, this is equal to V = template_2
    E_L_t1 = 0.5 * np.dot(template_1, template_1)
    E_L_t2 = np.dot(template_1, template_2) - 0.5 * np.dot(template_2, template_2)
    Var_L_t1 = 0
    Var_L_t2 = 0
    var_diff = 0
    for chan in range(0, n_chans):
        t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
        Var_L_t1 += (template_1[t_win[0]:t_win[1]][None, :]
                     @ chan_covariance_mats[chan]
                     @ template_1[t_win[0]:t_win[1]][:, None])
        Var_L_t2 += (template_2[t_win[0]:t_win[1]][None, :]
                     @ chan_covariance_mats[chan]
                     @ template_2[t_win[0]:t_win[1]][:, None])

        diff_template = template_1[t_win[0]:t_win[1]] - template_2[t_win[0]:t_win[1]]
        var_diff += (diff_template[None, :]
                     @ chan_covariance_mats[chan]
                     @ diff_template[:, None])

    # Expected difference between t1 and t2 likelihood functions
    E_diff_t1_t2 = E_L_t1 - E_L_t2
    if var_diff > 0:
        # Probability likelihood n1 - n2 < 0
        p_diff = norm.cdf(0, E_diff_t1_t2, np.sqrt(var_diff))
        # Probability both likelihoods are less than zero and neither assigned
        p_L_n1_lt0 = norm.cdf(0, E_L_t1, np.sqrt(Var_L_t1))
        p_L_n2_lt0 = norm.cdf(0, E_L_t2, np.sqrt(Var_L_t2))
        p_both_lt0 = p_L_n1_lt0*p_L_n2_lt0
        # Probability n2 is less than zero, so isn't incorrectly assigned
        p_L_n2_gt0 = norm.sf(0, E_L_t2, np.sqrt(Var_L_t2))
        # Probability a spike from unit n1 will be assigned to unit n2,
        # assuming independence of the probability n1/n2 are less than 0
        p_confusion = p_diff * p_L_n2_gt0 * (1 - p_both_lt0)
    else:
        p_confusion = 1.

    return p_confusion


def compute_metrics(templates, channel_covariance_mats, shifted_sum_variances,
                    n_noise_samples, sort_info, rand_state=None):
    """ Calculate variance and template sum squared metrics needed to compute
    separability_metrics between units and the delta likelihood function for binary
    pursuit. """

    # Ease of use variables
    n_chans = sort_info['n_channels']
    n_templates = len(templates)
    template_samples_per_chan = sort_info['n_samples_per_chan']
    chan_win, _ = time_window_to_samples(sort_info['clip_width'], sort_info['sampling_rate'])

    separability_metrics = {}
    separability_metrics['templates'] = np.vstack(templates)
    # Compute our template sum squared error (see note below).
    separability_metrics['template_SS'] = np.sum(separability_metrics['templates'] ** 2, axis=1)
    separability_metrics['template_SS_by_chan'] = np.zeros((n_templates, n_chans))
    # Get channel covariance of appropriate size from the extra large covariance matrices
    separability_metrics['channel_covariance_mats'] = [x[0:template_samples_per_chan, 0:template_samples_per_chan] for x in channel_covariance_mats]
    separability_metrics['template_shifted_sum_lower_thresholds'] = sort_info['sigma_noise_penalty'] * np.sqrt(shifted_sum_variances)

    # Compute bias for each neuron from its per channel variance
    separability_metrics['neuron_variances'] = np.zeros(n_templates)
    separability_metrics['neuron_lower_thresholds'] = np.zeros(n_templates)
    separability_metrics['neuron_upper_thresholds'] = np.zeros(n_templates)
    for n in range(0, n_templates):
        expectation = 0.5 * separability_metrics['template_SS'][n]
        variance = 0
        noise_only = 0
        for chan in range(0, n_chans):
            t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
            separability_metrics['template_SS_by_chan'][n, chan] = np.sum(
                    separability_metrics['templates'][n, t_win[0]:t_win[1]] ** 2)

            separability_metrics['neuron_variances'][n] += (separability_metrics['templates'][n, t_win[0]:t_win[1]][None, :]
                        @ separability_metrics['channel_covariance_mats'][chan]
                        @ separability_metrics['templates'][n, t_win[0]:t_win[1]][:, None])

        separability_metrics['neuron_lower_thresholds'][n] = max(sort_info['sigma_noise_penalty'] * np.sqrt(separability_metrics['neuron_variances'][n]), 0)

        print("OLD threshold:", separability_metrics['neuron_lower_thresholds'][n])

        new_threshold = decision_boundary_equal_var(-expectation, expectation,
                            separability_metrics['neuron_variances'][n],
                            sort_info['p_noise'])

        print("NEW threshold:", new_threshold)
        # separability_metrics['neuron_lower_thresholds'][n] = new_threshold

        separability_metrics['neuron_upper_thresholds'][n] = expectation + sort_info['sigma_template_ci'] * np.sqrt(separability_metrics['neuron_variances'][n])

    return separability_metrics


def shifted_template_sum_variance(templates, n_chans, n_samples_per_chan,
                                    n_max_shift_inds, channel_covariance_mats):
    """
    Note that for symmetry in matrix storage, t1 t2 at zero shift will be
    repeated for each pair.
    """
    n_templates = templates.shape[0]
    # Add 1 to max shift inds to include shift = 0
    n_shifts = n_max_shift_inds + 1
    shifted_sum_variances = np.zeros((n_templates * n_templates, n_shifts))
    shifted_sum_variances_single = np.zeros((n_templates * n_templates, n_shifts))
    chan_shifted_sum_templates = np.zeros((n_shifts, n_samples_per_chan + n_shifts))
    chan_shifted_sum_templates_single = np.zeros((n_samples_per_chan + n_shifts))
    for t1_ind in range(0, n_templates):
        for t2_ind in range(0, n_templates):
            # Row where shifted t1 and t2 variances stored for output
            shift_sum_row = t1_ind * n_templates + t2_ind
            for chan in range(0, n_chans):
                t_win = [chan*n_samples_per_chan, (chan+1)*n_samples_per_chan]
                # t1 is fixed at beginning of summed template
                chan_shifted_sum_templates[:, 0:n_samples_per_chan] = templates[t1_ind, t_win[0]:t_win[1]]
                # Add in t2 at each shifted lag relative to t1
                for shift_ind in range(0, n_shifts):
                    chan_shifted_sum_templates_single[0:n_samples_per_chan] = templates[t1_ind, t_win[0]:t_win[1]]
                    chan_shifted_sum_templates_single[shift_ind:(shift_ind+n_samples_per_chan)] += templates[t2_ind, t_win[0]:t_win[1]]

                    shifted_sum_variances_single[shift_sum_row, shift_ind] += (chan_shifted_sum_templates_single[None, :]
                                 @ channel_covariance_mats[chan]
                                 @ chan_shifted_sum_templates_single[:, None])

                    chan_shifted_sum_templates[shift_ind, shift_ind:(shift_ind+n_samples_per_chan)] += templates[t2_ind, t_win[0]:t_win[1]]
                # Add variance for each channel
                # Use matrix multiplication as loop shortcut for first step
                # then sum because result is not true matrix multiplication
                shifted_sum_variances[shift_sum_row, :] += np.sum(
                            (chan_shifted_sum_templates @ channel_covariance_mats[chan])
                            * chan_shifted_sum_templates, axis=1)
                # Reset these templates to 0 for next channel
                chan_shifted_sum_templates[:] = 0.0
                chan_shifted_sum_templates_single[:] = 0.0

    print("max var diffs", np.amax(np.amax(np.abs(shifted_sum_variances_single - shifted_sum_variances))))

    return shifted_sum_variances_single


def noise_threshold_templates(separability_metrics, sort_info):
    """ Identify templates below noise threshold for deletion and reset the
    lower threshold for remaining templates that might be affected. """

    n_chans = sort_info['n_channels']
    template_samples_per_chan = sort_info['n_samples_per_chan']
    max_shift = (template_samples_per_chan // 4) - 1
    n_neurons = separability_metrics['templates'].shape[0]

    noisy_units = np.zeros(n_neurons, dtype=np.bool)
    # Find neurons that are too close to noise and need to be deleted
    for neuron in range(0, n_neurons):
        neuron_threshold = 2 * sort_info['sigma_noise_penalty'] * np.sqrt(separability_metrics['neuron_variances'][neuron])
        if separability_metrics['template_SS'][neuron] < neuron_threshold:
            noisy_units[neuron] = True

def pairwise_separability(separability_metrics, sort_info):
    """
    Note that output matrix of separabilty errors is not necessarily symmetric
    due to the multiplication of the probability both units' likelihoods are
    less than zero.
    """
    n_chans = sort_info['n_channels']
    template_samples_per_chan = sort_info['n_samples_per_chan']
    max_shift = (template_samples_per_chan // 4) - 1
    n_neurons = separability_metrics['templates'].shape[0]

    # Compute separability from noise for each neuron
    neuron_noise_separability = np.zeros(n_neurons)
    # Compute separability measure for all pairs of neurons
    pair_separability_matrix = np.zeros((n_neurons, n_neurons))
    pair_decision_matrix = np.zeros((n_neurons, n_neurons))
    for n1 in range(0, n_neurons):
        E_L_n1 = 0.5 * separability_metrics['template_SS'][n1]

        E_L_n1_noise = -0.5 * separability_metrics['template_SS'][n1]
        neuron_noise_separability[n1] = norm.cdf(
                    separability_metrics['neuron_lower_thresholds'][n1], E_L_n1, np.sqrt(separability_metrics['neuron_variances'][n1]))

        for n2 in range(0, separability_metrics['templates'].shape[0]):
            # Need to optimally align n2 template with n1 template for computing
            # the covariance/difference of their Likelihood functions
            shift_temp1, shift_temp2, optimal_shift, shift_samples_per_chan = optimal_align_templates(
                                separability_metrics['templates'][n1, :],
                                separability_metrics['templates'][n2, :],
                                n_chans, max_shift=max_shift, align_abs=True)
            print("Using optimal shift of", optimal_shift)
            var_diff = 0
            var_1 = 0
            var_2 = 0
            covar = 0
            Var_L_n2 = 0
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
                var_diff += (diff_template[None, :]
                             @ s_chan_cov
                             @ diff_template[:, None])

                t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
                Var_L_n2 += (separability_metrics['templates'][n2, t_win[0]:t_win[1]][None, :]
                             @ separability_metrics['channel_covariance_mats'][chan]
                             @ separability_metrics['templates'][n2, t_win[0]:t_win[1]][:, None])

                # var_1 += (shift_temp1[s_win[0]:s_win[1]][None, :]
                #          @ s_chan_cov
                #          @ shift_temp1[s_win[0]:s_win[1]][:, None])
                # var_2 += (shift_temp2[s_win[0]:s_win[1]][None, :]
                #          @ s_chan_cov
                #          @ shift_temp2[s_win[0]:s_win[1]][:, None])
                # covar += (shift_temp1[s_win[0]:s_win[1]][None, :]
                #          @ s_chan_cov
                #          @ shift_temp2[s_win[0]:s_win[1]][:, None])

            # var_diff = var_1 + var_2 - 2*covar
            n_1_n2_SS = np.dot(shift_temp1, shift_temp2)
            # Assuming full template 2 data here as this will be counted in
            # computation of likelihood function
            E_L_n2 = n_1_n2_SS - 0.5 * separability_metrics['template_SS'][n2]

            # Expected difference between n1 and n2 likelihood functions
            E_diff_n1_n2 = E_L_n1 - E_L_n2
            if var_diff > 0:
                pair_separability_matrix[n1, n2] = norm.cdf(0, E_diff_n1_n2, np.sqrt(var_diff))


                E_L_n2_n2 = 0.5 * separability_metrics['template_SS'][n2]
                pair_decision_matrix[n1, n2] = E_L_n2 + (E_L_n2_n2 - E_L_n2)/2
                # print("Densities:", norm.cdf(pair_decision_matrix[n1, n2], E_L_n2_n2, np.sqrt(Var_L_n2)))
                # xvals = np.linspace(-10, 50, 1000)
                # yvals1 = norm.pdf(xvals, E_L_n2_n2, np.sqrt(Var_L_n2))
                # yvals2 = norm.pdf(xvals, E_L_n2, np.sqrt(Var_L_n2))
                #
                # plt.plot(xvals, np.squeeze(yvals1))
                # plt.plot(xvals, np.squeeze(yvals2))
                # plt.axvline(pair_decision_matrix[n1, n2])
                # plt.show()
                # Probability likelihood n1 - n2 < 0
                # p_diff = norm.cdf(0, E_diff_n1_n2, np.sqrt(var_diff))
                # Probability both likelihoods are less than zero and neither assigned
                # p_L_n1_lt0 = norm.cdf(0, E_L_n1, np.sqrt(var_1))
                # p_L_n2_lt0 = norm.cdf(0, E_L_n2, np.sqrt(var_2))
                # p_both_lt0 = p_L_n1_lt0*p_L_n2_lt0
                # Probability n2 is less than zero, so isn't incorrectly assigned
                # p_L_n2_gt0 = norm.sf(0, E_L_n2, np.sqrt(var_2))
                # Probability a spike from unit n1 will be assigned to unit n2,
                # assuming independence of the probability n1/n2 are less than 0
                # pair_separability_matrix[n1, n2] = p_diff * p_L_n2_gt0 * (1 - p_both_lt0)
            else:
                # No variance (should be the same unit)
                pair_separability_matrix[n1, n2] = 0

    return pair_separability_matrix, neuron_noise_separability, pair_decision_matrix


def empirical_separability(voltage, spike_times, templates, window_samples, separability_metrics, add_spikes=False):
    """ """

    overlapping_bools = []
    for s1 in range(0, len(spike_times)):
        overlapping_bools.append(np.ones(spike_times[s1].shape[0], dtype=np.bool))
        for s2 in range(0, len(spike_times)):
            if s1 == s2:
                continue
            overlapping_spike_bool = find_overlapping_spike_bool(spike_times[s1], spike_times[s2], overlap_tol=window_samples[1])
            overlapping_bools[s1] = np.logical_and(overlapping_bools[s1], ~overlapping_spike_bool)
    for ob in range(0, len(spike_times)):
        spike_times[ob] = spike_times[ob][overlapping_bools[ob]]

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

            LL.append(spike_clips @ templates[n2, :][:, None] - 0.5*np.sum(templates[n2, :] ** 2))

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

            E_L_n1 = 0.5 * np.sum(templates[n1, :] ** 2)
            E_L_n2 = (np.dot(templates[n1, :], templates[n2, :])
                      -0.5 * np.sum(templates[n2, :] ** 2))
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
