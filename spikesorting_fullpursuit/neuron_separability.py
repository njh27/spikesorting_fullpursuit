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
    for chan in range(0, n_chans):
        t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
        diff_template = template_1[t_win[0]:t_win[1]] - template_2[t_win[0]:t_win[1]]
        var_diff += (diff_template[None, :]
                     @ chan_covariance_mats[chan]
                     @ diff_template[:, None])

    # Expected difference between t1 and t2 likelihood functions
    E_diff_t1_t2 = E_L_t1 - E_L_t2
    if var_diff > 0:
        # Probability likelihood nt - t2 < 0
        p_confusion = norm.cdf(0, E_diff_t1_t2, np.sqrt(var_diff))
    else:
        p_confusion = 1.

    return p_confusion


def compute_separability_metrics(templates, channel_covariance_mats,
                                 sort_info):
    """ Calculate variance and template sum squared metrics needed to compute
    separability_metrics between units and the delta likelihood function for
    binary pursuit. After running binary pursuit, the number of spikes added
    to each unit will be appended using "add_n_spikes" below."""

    # Ease of use variables
    n_chans = sort_info['n_channels']
    n_templates = len(templates)
    template_samples_per_chan = sort_info['n_samples_per_chan']

    separability_metrics = {}
    separability_metrics['templates'] = np.vstack(templates)
    # Compute our template sum squared error (see note below).
    separability_metrics['template_SS'] = np.sum(separability_metrics['templates'] ** 2, axis=1)
    separability_metrics['template_SS_by_chan'] = np.zeros((n_templates, n_chans))
    # Get channel covariance of appropriate size from the extra large covariance matrices
    separability_metrics['channel_covariance_mats'] = channel_covariance_mats
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
    for n in range(0, n_templates):
        expectation = 0.5 * separability_metrics['template_SS'][n]
        for chan in range(0, n_chans):
            t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
            separability_metrics['template_SS_by_chan'][n, chan] = np.sum(
                    separability_metrics['templates'][n, t_win[0]:t_win[1]] ** 2)

            separability_metrics['neuron_variances'][n] += (separability_metrics['templates'][n, t_win[0]:t_win[1]][None, :]
                        @ separability_metrics['channel_covariance_mats'][chan]
                        @ separability_metrics['templates'][n, t_win[0]:t_win[1]][:, None])

        separability_metrics['neuron_lower_thresholds'][n] = (
                                -expectation + sort_info['sigma_lower_bound']
                                * np.sqrt(separability_metrics['neuron_variances'][n]))
        separability_metrics['neuron_lower_thresholds'][n] = max(0,
                            separability_metrics['neuron_lower_thresholds'][n])
        print("Lower threshold", n, "is:", separability_metrics['neuron_lower_thresholds'][n])
        noise_bound = (expectation - (sort_info['sigma_lower_bound'])
                        * np.sqrt(separability_metrics['neuron_variances'][n]))
        print("Noise bound is", noise_bound)
        separability_metrics['neuron_lower_thresholds'][n] = max(noise_bound, 0)

        # Determine peak channel for this unit
        separability_metrics['peak_channel'][n] = ( np.argmax(np.abs(
                                    separability_metrics['templates'][n, :]))
                                    // template_samples_per_chan )
        # Decision boundary between this unit and noise using noise probability
        # for this unit's peak channel
        # separability_metrics['neuron_lower_thresholds'][n] = find_decision_boundary_equal_var(
        #                             -expectation, expectation,
        #                             separability_metrics['neuron_variances'][n],
        #                             separability_metrics['channel_p_noise'][separability_metrics['peak_channel'][n]])
        # Expected proportion of true noise events that exceed lower threshold
        # and could be added to neuron n
        separability_metrics['contamination'][n] = norm.sf(
                            separability_metrics['neuron_lower_thresholds'][n],
                            -expectation,
                            np.sqrt(separability_metrics['neuron_variances'][n]))

    return separability_metrics


def add_n_spikes(separability_metrics, neuron_labels):
    """ Adds number of spikes assigned to each unit in separabilit_metrics
    after binary pursuit is run. Spikes are added by modifying the input
    separability_metrics dictionary.
    """
    separability_metrics['n_spikes'] = np.zeros(separability_metrics['templates'].shape[0])
    for n in range(0, separability_metrics['templates'].shape[0]):
        separability_metrics['n_spikes'][n] = np.count_nonzero(neuron_labels == n)

    # n_spikes added to existing separability_metrics
    return None


def find_noisy_templates(separability_metrics, sort_info):
    """ Identify templates as noisy if their lower confidene bound is less than
    their lower threshold. In separability metrics are originally computed
    with the lower threshold equal to the noise decision boundary. """
    n_neurons = separability_metrics['templates'].shape[0]

    noisy_templates = np.zeros(n_neurons, dtype=np.bool)
    # Find neurons that are too close to noise and need to be deleted
    for neuron in range(0, n_neurons):
        expectation = 0.5 * separability_metrics['template_SS'][neuron]
        lower_confidence_bound = expectation - sort_info['sigma_lower_bound'] \
                        * np.sqrt(separability_metrics['neuron_variances'][neuron])
        if separability_metrics['neuron_lower_thresholds'][neuron] > lower_confidence_bound:
            print("Neuron", neuron, "is NOISE")
            # print("BUT NOT DELETING IT")
            noisy_templates[neuron] = True

    return noisy_templates


def rethreshold_noise_and_templates(separability_metrics, sort_info,
                                        noisy_templates):
    """ This function first decides whether the templates indicated as noise by
    noisy_templates are required to maintain sorting accuracy above sigma noise
    penalty for each neuron. If it is not needed, remaining neurons are given
    a new lower boundary according to their decision boundary with the deleted
    units. Data for all noisy_templates are then removed from
    separability_metrics. """

    # print("SKIPPING rethreshold_noise_and_templates LINE 212")
    # return separability_metrics, noisy_templates

    n_chans = sort_info['n_channels']
    max_shift = (sort_info['n_samples_per_chan'] // 4) - 1
    n_neurons = separability_metrics['templates'].shape[0]

    for noise_n in range(0, n_neurons):
        # Find each noisy template that will be deleted
        if not noisy_templates[noise_n]:
            continue
        reset_boundaries = False
        max_boundaries = np.copy(separability_metrics['neuron_lower_thresholds'])
        max_contamination = np.copy(separability_metrics['contamination'])
        for n in range(0, n_neurons):
            # For each template that will NOT be deleted
            if noisy_templates[n]:
                continue
            # Reset the noise boundary for this template to be the maximum value
            # of its current boundary and all noise template boundaries
            expectation_n = 0.5 * separability_metrics['template_SS'][n]
            shift_temp_n, shift_temp_noise_n, _, _ = optimal_align_templates(
                                separability_metrics['templates'][n, :],
                                separability_metrics['templates'][noise_n, :],
                                n_chans, max_shift=max_shift, align_abs=True)
            # Expected value of Likelihood function for good neuron, n, given
            # that the true voltage is the noise neuron, noise_n
            expectation_n_noise_n = np.dot(shift_temp_n, shift_temp_noise_n) \
                                    - 0.5 * separability_metrics['template_SS'][n]
            decision_boundary = find_decision_boundary_equal_var(
                                        expectation_n, expectation_n_noise_n,
                                        separability_metrics['neuron_variances'][n])
            # Check if lower threshold must be increased in event this noise
            # neuron is deleted

            """ I THINK THIS ONLY NEEDS TO ASK IF GOOD NEURON N, GIVEN A NOISE SPIKE, EXCEEDS GOOD NEURON N THRESHOLD. i.e.
            the probability that a noise spike will be added to good neuron N if the noise spike template is deleted. """
            if decision_boundary > separability_metrics['neuron_lower_thresholds'][n]:
                # If choosing to delete this noise neuron will drive the decision
                # boundary too high for neuron n, remove noise neuron from the
                # noisy templates so it's not deleted.
                lower_bound = expectation_n - sort_info['sigma_lower_bound'] \
                                    * np.sqrt(separability_metrics['neuron_variances'][n])
                if decision_boundary > lower_bound:
                    noisy_templates[noise_n] = False
                    reset_boundaries = False
                    # If keeping this noise neuron, set its boundary high so
                    # binary pursuit doesn't spend forever adding to it. Set to
                    # the mean expected value
                    separability_metrics['neuron_lower_thresholds'][noise_n] = 0.5 * separability_metrics['template_SS'][noise_n]
                    print("Noise neuron", noise_n, "was kept to improve sorting for neuron", n)
                    break
                else:
                    # Need to delay reassignment of new boundaries until we are
                    # sure the current noise neuron will be deleted
                    reset_boundaries = True
                    max_boundaries[n] = decision_boundary
                    # Calculate proportion of deleted noise unit spikes expected
                    # to be added to unit n given this boundary
                    p_contamination = norm.sf(decision_boundary,
                                expectation_n_noise_n,
                                np.sqrt(separability_metrics['neuron_variances'][n]))
                    max_contamination[n] = p_contamination

        if reset_boundaries:
            separability_metrics['neuron_lower_thresholds'] = max_boundaries
            separability_metrics['contamination'] = max_contamination

    return separability_metrics, noisy_templates


def delete_noise_assign_thresholds(separability_metrics, noisy_templates):
    """ Remove data associated with noise templates from separability_metrics
    and reassign values IN PLACE to separability metrics dictionary. """
    print("SKIPPING delete_noise_assign_thresholds LINE 283")
    return separability_metrics
    for key in separability_metrics.keys():
        if key in ['channel_covariance_mats', 'channel_p_noise']:
            continue
        elif key in ['templates', 'template_SS_by_chan']:
            separability_metrics[key] = separability_metrics[key][~noisy_templates, :]
        elif key in ['template_SS', 'neuron_variances',
                     'neuron_lower_thresholds', 'contamination', 'peak_channel']:
            separability_metrics[key] = separability_metrics[key][~noisy_templates]
        else:
            print("!!! Could not find a condition for metrics key !!!", key)

    return separability_metrics


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
    noise_misses = np.zeros(n_neurons)
    noise_contamination = np.zeros(n_neurons)
    # Compute separability measure for all pairs of neurons
    pair_separability_matrix = np.zeros((n_neurons, n_neurons))
    for n1 in range(0, n_neurons):
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
                var_diff += (diff_template[None, :]
                             @ s_chan_cov
                             @ diff_template[:, None])

            n_1_n2_SS = np.dot(shift_temp1, shift_temp2)
            # Assuming full template 2 data here as this will be counted in
            # computation of likelihood function
            E_L_n2 = n_1_n2_SS - 0.5 * separability_metrics['template_SS'][n2]

            # Expected difference between n1 and n2 likelihood functions
            E_diff_n1_n2 = E_L_n1 - E_L_n2
            if var_diff > 0:
                pair_separability_matrix[n1, n2] = norm.cdf(0, E_diff_n1_n2, np.sqrt(var_diff))
            else:
                # No variance (should be the same unit)
                pair_separability_matrix[n1, n2] = 0

    return pair_separability_matrix, noise_contamination, noise_misses


# def shifted_template_sum_variance(templates, n_chans, n_samples_per_chan,
#                                     n_max_shift_inds, channel_covariance_mats):
#     """
#     Note that for symmetry in matrix storage, t1 t2 at zero shift will be
#     repeated for each pair.
#     """
#     n_templates = templates.shape[0]
#     # Add 1 to max shift inds to include shift = 0
#     n_shifts = n_max_shift_inds + 1
#     shifted_sum_variances = np.zeros((n_templates * n_templates, n_shifts))
#     shifted_sum_variances_single = np.zeros((n_templates * n_templates, n_shifts))
#     chan_shifted_sum_templates = np.zeros((n_shifts, n_samples_per_chan + n_shifts))
#     chan_shifted_sum_templates_single = np.zeros((n_samples_per_chan + n_shifts))
#     for t1_ind in range(0, n_templates):
#         for t2_ind in range(0, n_templates):
#             # Row where shifted t1 and t2 variances stored for output
#             shift_sum_row = t1_ind * n_templates + t2_ind
#             for chan in range(0, n_chans):
#                 t_win = [chan*n_samples_per_chan, (chan+1)*n_samples_per_chan]
#                 # t1 is fixed at beginning of summed template
#                 chan_shifted_sum_templates[:, 0:n_samples_per_chan] = templates[t1_ind, t_win[0]:t_win[1]]
#                 # Add in t2 at each shifted lag relative to t1
#                 for shift_ind in range(0, n_shifts):
#                     chan_shifted_sum_templates_single[0:n_samples_per_chan] = templates[t1_ind, t_win[0]:t_win[1]]
#                     chan_shifted_sum_templates_single[shift_ind:(shift_ind+n_samples_per_chan)] += templates[t2_ind, t_win[0]:t_win[1]]
#
#                     shifted_sum_variances_single[shift_sum_row, shift_ind] += (chan_shifted_sum_templates_single[None, :]
#                                  @ channel_covariance_mats[chan]
#                                  @ chan_shifted_sum_templates_single[:, None])
#
#                     chan_shifted_sum_templates[shift_ind, shift_ind:(shift_ind+n_samples_per_chan)] += templates[t2_ind, t_win[0]:t_win[1]]
#                 # Add variance for each channel
#                 # Use matrix multiplication as loop shortcut for first step
#                 # then sum because result is not true matrix multiplication
#                 shifted_sum_variances[shift_sum_row, :] += np.sum(
#                             (chan_shifted_sum_templates @ channel_covariance_mats[chan])
#                             * chan_shifted_sum_templates, axis=1)
#                 # Reset these templates to 0 for next channel
#                 chan_shifted_sum_templates[:] = 0.0
#                 chan_shifted_sum_templates_single[:] = 0.0
#
#     print("max var diffs", np.amax(np.amax(np.abs(shifted_sum_variances_single - shifted_sum_variances))))
#
#     return shifted_sum_variances_single


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
