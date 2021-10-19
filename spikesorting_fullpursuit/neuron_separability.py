import numpy as np
from scipy.stats import norm
from spikesorting_fullpursuit.consolidate import optimal_align_templates
from spikesorting_fullpursuit.analyze_spike_timing import find_overlapping_spike_bool
from spikesorting_fullpursuit.parallel.segment_parallel import time_window_to_samples

import matplotlib.pyplot as plt

def find_decision_boundary_equal_var(mu_1, mu_2, var, p_1=0.5):
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


def find_decision_boundary(mu_1, mu_2, var_1, var_2):
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


def check_template_pair_template(template_1, template_2, template_1_covar):
    """
    Intended for testing whether a sum of templates is equal to a given
    template. Templates are assumed to be aligned with one another as no
    shifting is performed. Probability of confusing the templates is
    returned. """
    # Compute separability given V = template_1.
    E_L_t1 = 0.5 * np.dot(template_1, template_1)
    E_L_t2 = np.dot(template_1, template_2) - 0.5 * np.dot(template_2, template_2)
    diff_template = template_1 - template_2
    var_diff = (diff_template[None, :]
                @ template_1_covar
                @ diff_template[:, None])

    # Expected difference between t1 and t2 likelihood functions
    E_diff_t1_t2 = np.abs(E_L_t1 - E_L_t2)
    if var_diff > 0:
        # Probability likelihood t1 - t2 < 0
        p_confusion = norm.cdf(0, E_diff_t1_t2, np.sqrt(var_diff))
    else:
        p_confusion = 1.

    return p_confusion


def check_template_pair(template_1, template_2, chan_covariance_mats, sort_info):
    """
    Intended for testing whether a sum of templates is equal to a given
    template. Templates are assumed to be aligned with one another as no
    shifting is performed. Probability of confusiong the templates is
    returned. This confusion is symmetric, i.e. p_confusion template_1 assigned
    to template_2 equals p_confusion template_2 assigned to template_1. """
    n_chans = sort_info['n_channels']
    template_samples_per_chan = sort_info['n_samples_per_chan']

    # Compute separability given V = template_1.
    E_L_t1 = 0.5 * np.dot(template_1, template_1)
    E_L_t2 = np.dot(template_1, template_2) - 0.5 * np.dot(template_2, template_2)
    var_diff = 0

    E_t2 = 0.5 * np.dot(template_2, template_2)
    var_t1 = 0
    var_t2 = 0
    full_diff_template = template_1 - template_2
    E_diff_template = 0.5 * np.dot(full_diff_template, full_diff_template)
    for chan in range(0, n_chans):
        t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
        diff_template = template_1[t_win[0]:t_win[1]] - template_2[t_win[0]:t_win[1]]
        var_diff += (diff_template[None, :]
                     @ chan_covariance_mats[chan]
                     @ diff_template[:, None])

        chan_t1 = template_1[t_win[0]:t_win[1]]
        var_t1 += (chan_t1[None, :]
                     @ chan_covariance_mats[chan]
                     @ chan_t1[:, None])
        chan_t2 = template_2[t_win[0]:t_win[1]]
        var_t2 += (chan_t2[None, :]
                     @ chan_covariance_mats[chan]
                     @ chan_t2[:, None])

    # Expected difference between t1 and t2 likelihood functions
    E_diff_t1_t2 = np.abs(E_L_t1 - E_L_t2)
    if var_diff > 0:
        # Probability likelihood nt - t2 < 0
        print("!!!!! DIVIDING VARIANCE BY 4 ON LINE 84 OF NEURON SEPARABILITY !!!!!!!!!")
        p_confusion = norm.cdf(0, E_diff_t1_t2, np.sqrt(var_diff))
        new_confusion = norm.cdf(0, E_diff_t1_t2, np.sqrt(var_diff + var_t1 + var_t2))
        p_confusion = new_confusion
        print("OLD confusion", p_confusion, "NEW confusion", new_confusion)
        print("Diff SS", np.sum(full_diff_template**2))
        print("T1 T2 SS", np.sum(template_1**2), np.sum(template_2**2))
    else:
        p_confusion = 1.

    return p_confusion


def compute_separability_metrics(templates, channel_covariance_mats,
                                 sort_info, template_covar):
    """ Calculate the various variance and template metrics needed to compute
    separability_metrics between units and the delta likelihood function for
    binary pursuit."""

    # Ease of use variables
    n_chans = sort_info['n_channels']
    n_templates = len(templates)
    template_samples_per_chan = sort_info['n_samples_per_chan']

    separability_metrics = {}
    # Store the samples per channel used in binary pursuit
    separability_metrics['bp_n_samples_per_chan'] = sort_info['n_samples_per_chan']
    separability_metrics['templates'] = np.vstack(templates)
    # Compute our template sum squared error (see note below).
    separability_metrics['template_SS'] = np.sum(separability_metrics['templates'] ** 2, axis=1)
    separability_metrics['template_SS_by_chan'] = np.zeros((n_templates, n_chans))
    # Get channel covariance of appropriate size from the extra large covariance matrices
    separability_metrics['channel_covariance_mats'] = channel_covariance_mats
    separability_metrics['template_covariance_mats'] = template_covar
    separability_metrics['contamination'] = np.zeros(n_templates)
    separability_metrics['peak_channel'] = np.zeros(n_templates, dtype=np.int64)

    separability_metrics['channel_p_noise'] = np.zeros(n_chans)
    # Expected number of threshold crossing due to noise given sigma
    noise_crossings = 2 * norm.cdf(-sort_info['sigma'], 0, 1) * sort_info['n_samples']
    for chan in range(0, n_chans):
        non_noise_crossings = sort_info['n_threshold_crossings'][chan] - noise_crossings
        # Can't be less than zero
        non_noise_crossings = max(non_noise_crossings, 0.0)
        # separability_metrics['channel_p_noise'][chan] = 1.0 - (non_noise_crossings / sort_info['n_samples'])
        separability_metrics['channel_p_noise'][chan] = ((sort_info['n_samples'] -
                sort_info['n_threshold_crossings'][chan]) / sort_info['n_samples'])

    # Compute variance for each neuron and the boundaries with noise for expected
    # likelihood function distribution given this variance
    separability_metrics['neuron_variances'] = np.zeros(n_templates)
    separability_metrics['neuron_lower_thresholds'] = np.zeros(n_templates)
    separability_metrics['neuron_lower_CI'] = np.zeros(n_templates)
    for n in range(0, n_templates):
        expectation = 0.5 * separability_metrics['template_SS'][n]
        for chan in range(0, n_chans):
            t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
            separability_metrics['template_SS_by_chan'][n, chan] = np.sum(
                    separability_metrics['templates'][n, t_win[0]:t_win[1]] ** 2)

            separability_metrics['neuron_variances'][n] += (separability_metrics['templates'][n, t_win[0]:t_win[1]][None, :]
                        @ separability_metrics['channel_covariance_mats'][n][chan]
                        @ separability_metrics['templates'][n, t_win[0]:t_win[1]][:, None])
        print("Channelwise variance", separability_metrics['neuron_variances'][n])
        separability_metrics['neuron_lower_thresholds'][n] = (-1*expectation + sort_info['sigma_bp_noise']
                                * np.sqrt(separability_metrics['neuron_variances'][n]))
        print("Channelwise threshold", separability_metrics['neuron_lower_thresholds'][n])

        separability_metrics['neuron_variances'][n] = (
                        separability_metrics['templates'][n, :][None, :]
                        @ separability_metrics['template_covariance_mats'][n]
                        @ separability_metrics['templates'][n, :][:, None])
        print("Full template variance", separability_metrics['neuron_variances'][n])
        # Set threshold in standard deviations from decision boundary at 0
        # separability_metrics['neuron_lower_thresholds'][n] = (
        #                         sort_info['sigma_bp_noise']
        #                         * np.sqrt(separability_metrics['neuron_variances'][n]))

        separability_metrics['neuron_lower_CI'][n] = (expectation - sort_info['sigma_bp_CI']
                                * np.sqrt(separability_metrics['neuron_variances'][n]))

        # Set threshold in standard deviations from expected value at voltage = 0
        separability_metrics['neuron_lower_thresholds'][n] = (-1*expectation + sort_info['sigma_bp_noise']
                                * np.sqrt(separability_metrics['neuron_variances'][n]))
        print("Lower CI", separability_metrics['neuron_lower_CI'][n])
        print("Lower THRESHOLD", separability_metrics['neuron_lower_thresholds'][n])
        # separability_metrics['neuron_lower_thresholds'][n] = max(separability_metrics['neuron_lower_thresholds'][n], 0)
        # separability_metrics['neuron_lower_thresholds'][n] = max(separability_metrics['neuron_lower_thresholds'][n],
        #                                                         separability_metrics['neuron_lower_CI'][n])
        # print("Unit", n, "variance=", separability_metrics['neuron_variances'][n], "Mean=", expectation)
        # print("Template var is", template_var[n], "Likelihood var is", separability_metrics['neuron_variances'][n])
        # print("Their difference is", template_var[n] - separability_metrics['neuron_variances'][n])

        # Determine peak channel for this unit
        separability_metrics['peak_channel'][n] = ( np.argmax(np.abs(
                                    separability_metrics['templates'][n, :]))
                                    // template_samples_per_chan )
        separability_metrics['contamination'][n] = norm.sf(
                            separability_metrics['neuron_lower_thresholds'][n],
                            -expectation,
                            np.sqrt(separability_metrics['neuron_variances'][n]))

    return separability_metrics


def find_noisy_templates(separability_metrics, sort_info):
    """ Identify templates as noisy if their lower confidene bound is less than
    or equal to the decision boundary at 0. """
    n_neurons = separability_metrics['templates'].shape[0]

    noisy_templates = np.zeros(n_neurons, dtype="bool")
    # Find neurons that are too close to decision boundary and need to be deleted
    for neuron in range(0, n_neurons):
        # This implies that the neuron's expected value given a spike is present
        # has a distribution that overlaps the distribution centered at 0.0
        # within sigma_bp_noise standard deviations of each other
        if 2 * separability_metrics['neuron_lower_thresholds'][neuron] > 0.5 * separability_metrics['template_SS'][neuron]:
            noisy_templates[neuron] = True
            print("Unit", neuron, "is noisy.")
        # if separability_metrics['neuron_lower_thresholds'][neuron] > 0.0:
        #     noisy_templates[neuron] = True
        #     print("Unit", neuron, "is noisy.")

    return noisy_templates


def check_noise_templates(separability_metrics, sort_info,
                                        noisy_templates):
    """ This function first decides whether the templates indicated as noise by
    noisy_templates will be useful for sorting other units. If it is, then it
    is kept. If its detectability is so small or dissimilar to other units,
    then it will be deleted permaneantly. """
    n_chans = sort_info['n_channels']
    max_shift = (sort_info['n_samples_per_chan'] // 4) - 1
    n_neurons = separability_metrics['templates'].shape[0]

    new_noisy_templates = np.copy(noisy_templates)
    for noise_n in range(0, n_neurons):
        # Find each noisy template that could be deleted
        if not noisy_templates[noise_n]:
            continue
        for n in range(0, n_neurons):
            # For each template that will NOT be deleted
            if noisy_templates[n]:
                continue
            # First get aligned templates
            shift_temp_n, shift_temp_noise_n, _, _ = optimal_align_templates(
                                separability_metrics['templates'][n, :],
                                separability_metrics['templates'][noise_n, :],
                                n_chans, max_shift=max_shift, align_abs=True)
            # Expected value of Likelihood function for good neuron, n, given
            # that the true voltage is the noise neuron, noise_n
            expectation_n_noise_n = np.dot(shift_temp_n, shift_temp_noise_n) \
                                    - 0.5 * separability_metrics['template_SS'][n]
            # Variance of likelihood for good neuron given noise unit spike
            var_n_noise_n = (separability_metrics['templates'][n, :][None, :]
                            @ separability_metrics['template_covariance_mats'][noise_n]
                            @ separability_metrics['templates'][n, :][:, None])
            if var_n_noise_n == 0.0:
                # Templates do not overlap across channels
                continue
            # Find the upper bound of the distribution of the likelihood
            # function for neuron n, given that voltage = noise_n
            # noise_match_upper_bound = expectation_n_noise_n + sort_info['sigma_bp_noise'] \
            #                     * np.sqrt(var_n_noise_n)
            # Probability a noise spike exceeds threshold and added to good unit
            p_noise_added = norm.sf(separability_metrics['neuron_lower_thresholds'][n],
                                    expectation_n_noise_n, np.sqrt(var_n_noise_n))
            # Probability of adding a noise false positive given that good
            # neuron is not present as definted by bp noise threshold
            p_n_added_noise = norm.sf(sort_info['sigma_bp_noise'], 0, 1)
            if p_noise_added > p_n_added_noise:
            # if noise_match_upper_bound > separability_metrics['neuron_lower_thresholds'][n]:
                # The likelihood function for good template n given the noise
                # template has enough probability of exceeding threshold to be
                # either incorrectly added to the good unit or improve its
                # sorting so keep it
                new_noisy_templates[noise_n] = False
                print("Keeping unit", noise_n, "to improve sorting.")
                break
            # If we make it here for each neuron, then if we delete the noise
            # unit, spikes from that unit are unlikely to be added to any of
            # the good units and it is of low enough quality that it won't
            # help sorting

    return separability_metrics, new_noisy_templates


def delete_noise_units(separability_metrics, noisy_templates):
    """ Remove data associated with deleted noise templates from
    separability_metrics and reassign values IN PLACE to separability_metrics. """
    for key in separability_metrics.keys():
        if key in ['channel_covariance_mats', 'channel_p_noise',
                    'bp_n_samples_per_chan']:
            continue
        elif key in ['templates', 'template_SS_by_chan']:
            separability_metrics[key] = separability_metrics[key][~noisy_templates, :]
        elif key in ['template_SS', 'neuron_variances',
                     'neuron_lower_thresholds', 'neuron_lower_CI',
                     'contamination', 'peak_channel']:
            separability_metrics[key] = separability_metrics[key][~noisy_templates]
        elif key in ['template_covariance_mats']:
            for x in reversed(range(0, len(separability_metrics[key]))):
                if noisy_templates[x]:
                    del separability_metrics[key][x]
        else:
            print("!!! Could not find a condition for a key in separability_metrics!!!", key)

    print("Removed", np.count_nonzero(noisy_templates), "templates as noise")
    return separability_metrics


def set_bp_threshold(separability_metrics):
    """ Set binary pursuit threshold to the max of the decision boundary or the
    lower confidence bound. """
    for n in range(0, separability_metrics['neuron_lower_CI'].shape[0]):
        separability_metrics['neuron_lower_thresholds'][n] = max(
                    separability_metrics['neuron_lower_thresholds'][n], 0)
        separability_metrics['neuron_lower_thresholds'][n] = max(
                            separability_metrics['neuron_lower_thresholds'][n],
                            separability_metrics['neuron_lower_CI'][n])

    return separability_metrics


def add_n_spikes(separability_metrics, neuron_labels):
    """ Adds number of spikes assigned to each unit in separabilit_metrics
    after binary pursuit is run. Spikes are added by modifying the input
    separability_metrics dictionary. Called in binary_pursuit_parallel.
    """
    separability_metrics['n_spikes'] = np.zeros(separability_metrics['templates'].shape[0])
    for n in range(0, separability_metrics['templates'].shape[0]):
        separability_metrics['n_spikes'][n] = np.count_nonzero(neuron_labels == n)

    return None


def pairwise_separability(separability_metrics, sort_info):
    """
    Uses separability metrics to determine the overlap between pairs of units
    and estimate the probability of confusing spikes between the units. Similar
    estimates are made for the probability of adding noise to a unit and missing
    spikes from a unit.
    """
    n_chans = sort_info['n_channels']
    template_samples_per_chan = separability_metrics['bp_n_samples_per_chan']
    max_shift = (template_samples_per_chan // 4) - 1
    n_neurons = separability_metrics['templates'].shape[0]

    # Compute separability from noise for each neuron
    noise_misses = np.zeros(n_neurons)
    noise_contamination = np.zeros(n_neurons)
    # Compute separability measure for all pairs of neurons
    pair_separability_matrix = np.zeros((n_neurons, n_neurons))
    for n1 in range(0, n_neurons):
        # Expected n1 likelihood given n1 spike
        E_L_n1 = 0.5 * separability_metrics['template_SS'][n1]
        noise_misses[n1] = norm.cdf(
                    separability_metrics['neuron_lower_thresholds'][n1], E_L_n1, np.sqrt(separability_metrics['neuron_variances'][n1]))

        E_L_n1_noise = -0.5 * separability_metrics['template_SS'][n1]
        p_spike_added_given_noise = norm.sf(
                    separability_metrics['neuron_lower_thresholds'][n1],
                    E_L_n1_noise, np.sqrt(separability_metrics['neuron_variances'][n1]))
        p_spike_added = separability_metrics['n_spikes'][n1] / sort_info['n_samples']
        # Probability of adding a spike must be at >= conditional probability
        p_spike_added = max(p_spike_added, p_spike_added_given_noise)
        noise_contamination[n1] = (p_spike_added_given_noise *
                separability_metrics['channel_p_noise'][separability_metrics['peak_channel'][n1]]
                / p_spike_added)

        for n2 in range(0, separability_metrics['templates'].shape[0]):
            # if separability_metrics['peak_channel'][n1] != separability_metrics['peak_channel'][n2]:
            #     print("Skipping comparison not on the same channel")
            #     continue
            # Need to optimally align n2 template with n1 template for computing
            # the covariance/difference of their Likelihood functions
            shift_temp1, shift_temp2, optimal_shift, shift_samples_per_chan = optimal_align_templates(
                                separability_metrics['templates'][n1, :],
                                separability_metrics['templates'][n2, :],
                                n_chans, max_shift=max_shift, align_abs=True)
            # var_diff = 0
            # for chan in range(0, n_chans):
            #     s_win = [chan*shift_samples_per_chan, (chan+1)*shift_samples_per_chan]
            #     # Adjust covariance matrix to new templates, shifting according to template 1
            #     if optimal_shift >= 0:
            #         s_chan_cov = separability_metrics['channel_covariance_mats'][chan][optimal_shift:, optimal_shift:]
            #     else:
            #         s_chan_cov = separability_metrics['channel_covariance_mats'][chan][0:optimal_shift, 0:optimal_shift]
            #     # Multiply template with covariance then template transpose to
            #     # get variance of the likelihood function (or the difference)
            #     diff_template = shift_temp1[s_win[0]:s_win[1]] - shift_temp2[s_win[0]:s_win[1]]
            #     var_diff += (diff_template[None, :]
            #                  @ s_chan_cov
            #                  @ diff_template[:, None])
            # Need to zero pad the shifted templates so that we can use the
            # covariance matrix
            pad_shift_temp1 = np.zeros(separability_metrics['templates'][n1, :].shape[0])
            pad_shift_temp2 = np.zeros(separability_metrics['templates'][n2, :].shape[0])
            for chan in range(0, n_chans):
                t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
                s_win = [chan*shift_samples_per_chan, (chan+1)*shift_samples_per_chan]
                # Adjust covariance matrix to new templates, shifting according to template 1
                if optimal_shift >= 0:
                    pad_shift_temp1[t_win[0]+optimal_shift:t_win[1]] = shift_temp1[s_win[0]:s_win[1]]
                    pad_shift_temp2[t_win[0]+optimal_shift:t_win[1]] = shift_temp2[s_win[0]:s_win[1]]
                else:
                    pad_shift_temp1[t_win[0]:t_win[1]+optimal_shift] = shift_temp1[s_win[0]:s_win[1]]
                    pad_shift_temp2[t_win[0]:t_win[1]+optimal_shift] = shift_temp2[s_win[0]:s_win[1]]
            diff_template = pad_shift_temp1 - pad_shift_temp2
            # Use n1 covariance because computing conditioned on n1 spike
            var_diff = (diff_template[None, :]
                        @ separability_metrics['template_covariance_mats'][n1]
                        @ diff_template[:, None])

            # Expected n2 likelihood given n1 spike
            n_1_n2_SS = np.dot(shift_temp1, shift_temp2)
            # Assuming full template 2 data here as this will be counted in
            # computation of likelihood function
            E_L_n2 = n_1_n2_SS - 0.5 * separability_metrics['template_SS'][n2]

            # Expected difference between n1 and n2 likelihood functions
            E_diff_n1_n2 = E_L_n1 - E_L_n2
            if var_diff > 0:
                # Probability n2 likelihood greater than n1
                pair_separability_matrix[n1, n2] = norm.cdf(0, E_diff_n1_n2, np.sqrt(var_diff))
            else:
                # No variance (should be the same unit)
                pair_separability_matrix[n1, n2] = 0

    return pair_separability_matrix, noise_contamination, noise_misses
