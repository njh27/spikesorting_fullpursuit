import numpy as np
from numpy import linalg as la
from scipy.fftpack import dct, idct, fft, ifft
from scipy.optimize import fsolve, fminbound
from spikesorting_fullpursuit import isotonic
from spikesorting_fullpursuit.c_cython import sort_cython
from spikesorting_fullpursuit import multinomial_gof



def initial_cluster_farthest(data, median_cluster_size, choose_percentile=0.95, n_random=0):
    """
    Create distance based cluster labels along the rows of data.

    Returns a vector containing the labels for each data point.

    Data are iteratively clustered based on Euclidean distance until the median
    number of points in each cluster is <= median_cluster_size or each point is
    the only member of its own cluster. For each iteration, a new cluster center
    is chosen. First, the distance of each point from its nearest cluster center
    is computed. Second, from these distances, the point lying at the 99th
    percentile is chosen to be a new cluster center.  Finally, all points closer
    to this center than their current cluster center are assigned to the new
    cluster and the process is repeated.  This is similar to the k-means++
    algorithm except that it is deterministic, always choosing at the 99th
    percentile.

    Parameters
    ----------
    data : numpy ndarray
        Each row of data will be treated as an observation and each column as a
        dimension over which distance will be computed.  Must be two dimensional.
    median_cluster_size : {int, float, ndarray etc.}
        Must be a single, scalar value regardless of type. New cluster centers
        will be added until the median number of points from data that are
        nearest a cluster center is less than or equal to this number (see
        Notes below).

    Returns
    -------
    labels : numpy ndarray of dtype int64
        A new array holding the numerical labels indicating the membership of
        each point input in data. Array is the same size as data.shape[0].
    """
    if data.ndim <= 1 or data.size == 1:
        # Only 1 spike so return 1 label!
        return np.zeros(1, dtype=np.int64)

    # Begin with a single cluster (all data belong to the same cluster)
    labels = np.zeros((data.shape[0]), dtype=np.int64)
    label_counts = labels.size
    current_num_centers = 0
    if labels.size <= median_cluster_size:
        return labels
    if median_cluster_size <= 2:
        labels = np.arange(0, labels.size, dtype=np.int64)
        return labels
    centers = np.mean(data, axis=0)
    distances = np.sum((data - centers)**2, axis=1)
    if np.all(np.all(distances == 0)):
        # All scores are the same, so return all same label
        return labels

    if n_random > 0:
        if n_random >= labels.size:
            return np.arange(0, labels.size, dtype=np.int64)
        n_random = np.ceil(n_random).astype(np.int64)
        for nl in range(0, n_random):
            rand_ind = np.random.choice(data.shape[0], 1,
                            p=(distances/np.sum(distances)), replace=False)
            current_num_centers += 1
            new_center = data[rand_ind, :]
            centers = np.vstack((centers, new_center))
            temp_distance = np.sum((data - new_center)**2, axis=1)
            select = temp_distance < distances
            labels[select] = current_num_centers
            distances[select] = temp_distance[select]
            _, label_counts = np.unique(labels, return_counts=True)
            if current_num_centers == labels.size:
                break
    pre_centers = current_num_centers

    _, label_counts = np.unique(labels, return_counts=True)
    # Convert percentile to an index
    n_percentile = np.ceil((labels.size-1) * (1 - choose_percentile)).astype(np.int64)
    while np.median(label_counts) > median_cluster_size and current_num_centers < labels.size:
        current_num_centers += 1
        # Partition the negative distances (ascending partition)
        new_index = np.argpartition(-distances, n_percentile)[n_percentile]
        # Choose data at percentile index as the center of the next cluster
        new_center = data[new_index, :]
        centers = np.vstack((centers, new_center))
        temp_distance = np.sum((data - new_center)**2, axis=1)
        # Add any points closer to new center than their previous center
        select = temp_distance < distances
        labels[select] = current_num_centers
        distances[select] = temp_distance[select]
        _, label_counts = np.unique(labels, return_counts=True)

    return labels


def assign_nearest_label(data, labels):
    """ Reassigns all the data to their nearest cluster center according to labels.
    """
    unq_labels = np.unique(labels)
    centers = {}
    distances = np.zeros((data.shape[0]))
    # Get each label center and assign its current points to it
    for lab in unq_labels:
        sel_label = labels == lab
        centers[lab] = np.mean(data[sel_label, :], axis=0)
        distances[sel_label] = np.sum((data[sel_label, :] - centers[lab])**2, axis=1)

    # No go through each center and assign nearest to it
    for label, center in centers.items():
        temp_distances = np.sum((data - center)**2, axis=1)
        sel_relabel = temp_distances < distances
        labels[sel_relabel] = label
        distances[sel_relabel] = temp_distances[sel_relabel]

    return labels


def reorder_labels(labels):
    """
    Rename labels from 0 to n-1, where n is the number of unique labels.

    Returns None. Input labels is altered in place.

    Following sorting, the labels for individual spikes can be any number up to
    the maximum number of clusters used for initial sorting (worst case
    scenario, this could be 0:M-1, where M is the number of spikes). This
    function reorders the labels so that they nicely go from 0:num_clusters.

    Parameters
    ----------
    labels : numpy ndarray
        A one dimensional array of numerical labels.

    Returns
    -------
    None :
        The array labels is changed in place.
    """

    if labels.size == 0:
        return
    unique_labels = np.unique(labels)
    new_label = 0
    for old_label in unique_labels:
        labels[labels == old_label] = new_label
        new_label += 1

    return None


def max_sse_window(observed, expected):
    """
    Compute an approximate KS statistic for Kolmogorov-Smirnov test from binned
    data.

    Returns an approximate KS statistic and its approximate p-value as well as
    the point of maximum difference from the input distributions.

    This is intended as a helper function for the current sorting algorithm and
    is NOT a proper Kolmogorov-Smirnov test in its own right.  It is inspired
    by the KS test but violates some fundamental assumptions. The actual KS test
    and associated distributions require that the KS statistic be computed from
    data that are not binned or smoothed.  This function only approximates these
    tests for the sake of speed by by using binned data in the face of
    potentially very large sample sizes.  It does this by computing the sample
    size N as the number of observations in each binned distribution (the sum
    of counts) and penalizing  by a degrees of freedom equal to the number of
    bins (the length of counts). In the current context the number of bins is
    the square root of N.

    Parameters
    ----------
    counts_1, counts_2 : numpy ndarray
        These vectors contain the binned observation counts for the underlying
        distributions to be compared.  It is assumed that both were binned over
        the same range, and are orded in accord with this axis and with each
        other.  The CDF of each distribution will be computed and compared
        between them under this assumption.

    Returns
    -------
    ks : float
        The pseudo KS test statistic as computed from the binned data in
        counts_1 and counts_2 using N values adjusted by the number of bins used.
    I : numpy ndarray of type int
        The index into the counts where the maximum difference in the empirical
        CDF was found (based on the input binned data).  Used to indicate the
        point of maximum divergence between the distributions represented by
        counts_1 and counts_2.
    p_value : float
        The p-value for the KS test between the distributions represented by
        counts_1 and counts_2 as returned by compute_ks4_p_value.

    See Also
    --------
    compute_ks4_p_value

    Indices are sliceable, and at least 0, 1. They are adjusted to include the
    point just before and just after the most deviant SSE window if
    possible."""

    max_sse = 0
    best_start = 0
    best_stop = observed.size
    search_start = True
    d_observed = np.diff(observed)
    d_expected = np.diff(expected)
    check_ind = 0
    while check_ind < observed.size:
        if search_start:
            if check_ind == observed.size - 1:
                break # Can't look for start at last index
            elif check_ind == 0:
                if observed[check_ind] != expected[check_ind]:
                    # Will act as an "OR" operator when paired with "if" below
                    curr_start = check_ind
                    search_start = False
            if d_observed[check_ind] != d_expected[check_ind]:
                curr_start = check_ind
                search_start = False
        else:
            if check_ind == observed.size - 1:
                # Hit last index so end window and check it
                curr_stop = check_ind+1
                curr_sse = np.sum((observed[curr_start:curr_stop] - expected[curr_start:curr_stop]) ** 2)
                if curr_sse >= max_sse: # Equal allows finding last most point
                    best_start = curr_start
                    best_stop = curr_stop
                    max_sse = curr_sse
                break
            if (d_observed[check_ind] == d_expected[check_ind]) and (d_observed[check_ind] != 0):
                curr_stop = min(check_ind + 1, observed.size)
                curr_sse = np.sum((observed[curr_start:curr_stop] - expected[curr_start:curr_stop]) ** 2)
                if curr_sse >= max_sse: # Equal allows finding last most point
                    best_start = curr_start
                    best_stop = curr_stop
                    max_sse = curr_sse
                search_start = True
        check_ind += 1

    return max_sse, best_start, best_stop


def choose_optimal_cutpoint(cutpoint_ind, residual_densities, x_axis):
    """
    """
    if cutpoint_ind == 0:
        first_same_index = 0
        last_same_index = 0
        check_ind = 0
        while residual_densities[check_ind + 1] == residual_densities[cutpoint_ind]:
            last_same_index = check_ind + 1
            check_ind += 1
            if check_ind == residual_densities.shape[0] - 1:
                break
    elif cutpoint_ind == residual_densities.shape[0] - 1:
        first_same_index = residual_densities.shape[0] - 1
        last_same_index = residual_densities.shape[0] - 1
        check_ind = residual_densities.shape[0] - 1
        while residual_densities[check_ind - 1] == residual_densities[cutpoint_ind]:
            first_same_index = check_ind - 1
            check_ind -= 1
            if check_ind == 0:
                break
    else:
        first_same_index = cutpoint_ind
        check_ind = cutpoint_ind
        while residual_densities[check_ind - 1] == residual_densities[cutpoint_ind]:
            first_same_index = check_ind - 1
            check_ind -= 1
            if check_ind == 0:
                break
        last_same_index = cutpoint_ind
        check_ind = cutpoint_ind
        while residual_densities[check_ind + 1] == residual_densities[cutpoint_ind]:
            last_same_index = check_ind + 1
            check_ind += 1
            if check_ind == residual_densities.shape[0] - 1:
                break

    # Do not want to cut at 0 or between bin zero and 1; same for end
    # first_same_index = max(first_same_index , 1)
    # last_same_index = min(last_same_index, residual_densities.shape[0] - 1)
    cutpoint = x_axis[first_same_index] + (
                    x_axis[last_same_index] - x_axis[first_same_index])/2

    return cutpoint


def iso_cut(projection, p_value_cut_thresh):
    """ This helper function determines the optimal cutpoint given a distribution.
    First, it tests to determine whether the histogram has a single peak If not,
    it returns the optimal cut point. """
    N = projection.size
    if N <= 2:
        # Don't try any comparison with only two samples since can't split
        # merge_clusters shouldn't get to this point
        return 1., None

    num_bins = np.ceil(np.sqrt(N)).astype(np.int64)
    if num_bins < 2: num_bins = 2

    smooth_pdf, x_axis, _ = sort_cython.kde(projection, num_bins)
    # smooth_pdf, x_axis, _ = kde_builtin(projection, num_bins)
    if x_axis.size == 1:
        # All data are in same bin so merge (shouldn't happen)
        return 1., None
    # Output density of kde does not sum to one, so normalize it.
    smooth_pdf = smooth_pdf / np.sum(smooth_pdf)
    # kde function estimates at power of two spacing levels so compute num_points
    num_points = smooth_pdf.size
    if np.any(np.isnan(smooth_pdf)):
        return 1., None # (shouldn't happen)

    # Approximate observations per spacing used for computing n for statistics
    smooth_pdf[smooth_pdf < 0] = 0
    obs_counts = np.around(smooth_pdf * N).astype(np.int64)
    remove_counts = np.logical_and(obs_counts <= 0, smooth_pdf <= 0)
    smooth_pdf = smooth_pdf[~remove_counts]
    obs_counts = obs_counts[~remove_counts]
    x_axis = x_axis[~remove_counts]
    num_points = obs_counts.shape[0]

    # Generate a triange weighting to bias regression towards center of distribution
    if num_points % 2 == 0:
        iso_fit_weights = np.hstack((np.arange(1, num_points // 2 + 1), np.arange(num_points // 2, 0, -1)))
    else:
        iso_fit_weights = np.hstack((np.arange(1, num_points // 2 + 1), np.arange(num_points // 2 + 1, 0, -1)))
    densities_unimodal_fit, peak_density_ind = isotonic.unimodal_prefix_isotonic_regression_l2(smooth_pdf, iso_fit_weights)
    densities_unimodal_fit = densities_unimodal_fit / np.sum(densities_unimodal_fit)
    null_counts = np.around(densities_unimodal_fit * N).astype(np.int64)

    if np.all(obs_counts == null_counts):
        return 1., None
    if num_points <= 4:
        m_gof = multinomial_gof.MultinomialGOF(
                    obs_counts,
                    densities_unimodal_fit,
                    p_threshold=p_value_cut_thresh)
        p_value = m_gof.twosided_exact_test()
        cutpoint = None
        if p_value < p_value_cut_thresh:
            cutpoint_ind = np.argmax(null_counts - obs_counts)
            cutpoint = choose_optimal_cutpoint(cutpoint_ind, null_counts - obs_counts, x_axis)
            # print("Early EXACT critical cut at p=", p_value,"!")
        return p_value, cutpoint

    sse_left, left_start, left_stop = max_sse_window(obs_counts[0:peak_density_ind+1],
                             null_counts[0:peak_density_ind+1])
    sse_right, right_start, right_stop = max_sse_window(obs_counts[peak_density_ind:][-1::-1],
                               null_counts[peak_density_ind:][-1::-1])
    if sse_left > sse_right:
        # max_sse_window returns sliceable indices so don't adjust
        critical_range = np.arange(left_start, left_stop)
        citical_side = 'left'
    else:
        # right side values were computed backward in max_sse_window so fix it
        flip_right_start = len(x_axis) - right_stop
        flip_right_stop = len(x_axis) - right_start
        critical_range = np.arange(flip_right_start, flip_right_stop)
        citical_side = 'right'

    m_gof = multinomial_gof.MultinomialGOF(
                obs_counts[critical_range],
                densities_unimodal_fit[critical_range],
                p_threshold=p_value_cut_thresh)
    log_combinations = m_gof.get_log_n_total_combinations()
    n_perms = np.int64(np.ceil(1/p_value_cut_thresh) * 100)
    if log_combinations <= np.log(n_perms):
        # Fewer than n_perms combinations exist so do exact test
        p_value = m_gof.twosided_exact_test()
    else:
        p_value = m_gof.random_perm_test(n_perms=n_perms)

    # Only compute cutpoint if we plan on using it, also skipped if p_value is np.nan
    cutpoint = None
    if p_value < p_value_cut_thresh:
        residual_densities = obs_counts - null_counts

        if np.any(residual_densities > 0.):
            residual_densities_fit, _ = isotonic.unimodal_prefix_isotonic_regression_l2(-1 * residual_densities, np.ones_like(x_axis))
            if (peak_density_ind <= obs_counts.size/2):
                n_consec_0 = 0
                start_consec = False
                ind = peak_density_ind + 1
                while ind < obs_counts.size:
                    if residual_densities_fit[ind] < 0.:
                        break
                    if ( (obs_counts[ind] == 0.) and (not start_consec) ):
                        start_consec = True
                        n_consec_0 = 1
                    if obs_counts[ind] != 0.:
                        start_consec = False
                    if ( (obs_counts[ind] == 0.) and (start_consec) ):
                        n_consec_0 += 1
                    ind += 1
                if ( (n_consec_0 / obs_counts.size > 0.1) and (n_consec_0 > 3) ):
                    cutpoint = choose_optimal_cutpoint(ind - 1, residual_densities_fit, x_axis)
                    return p_value, cutpoint
            else:
                n_consec_0 = 0
                start_consec = False
                ind = peak_density_ind - 1
                while ind >= 0:
                    if residual_densities_fit[ind] < 0.:
                        break
                    if ( (obs_counts[ind] == 0.) and (not start_consec) ):
                        start_consec = True
                        n_consec_0 = 1
                    if obs_counts[ind] != 0.:
                        start_consec = False
                    if ( (obs_counts[ind] == 0.) and (start_consec) ):
                        n_consec_0 += 1
                    ind -= 1
                if ( (n_consec_0 / obs_counts.size > 0.1) and (n_consec_0 > 3) ):
                    cutpoint = choose_optimal_cutpoint(ind + 1, residual_densities_fit, x_axis)
                    return p_value, cutpoint

        # Multiply by negative residual densities since isotonic.unimodal_prefix_isotonic_regression_l2 only does UP-DOWN
        residual_densities_fit, _ = isotonic.unimodal_prefix_isotonic_regression_l2(-1 * residual_densities[critical_range], np.ones_like(critical_range))
        # A full critical range suggests possible bad fit because distribution is extremely bimodal so check them
        found_dip = False
        if ( (peak_density_ind < obs_counts.size/2) and (critical_range[-1] == obs_counts.size - 1) ):
            min_null_fit = np.amin(null_counts)
            for cutpoint_ind in range(0, residual_densities_fit.size):
                if ( (null_counts[cutpoint_ind] == min_null_fit) and (residual_densities_fit[cutpoint_ind] < 0) ):
                    # This means the null fit missed a second hump because it was so extreme
                    found_dip = True
                    break
            if cutpoint_ind > 0:
                cutpoint_ind -= 1
        elif ( (peak_density_ind > obs_counts.size/2) and (critical_range[-1] == obs_counts.size - 1) ):
            min_null_fit = np.amin(null_counts)
            for cutpoint_ind in range(residual_densities_fit.size - 1, -1, -1):
                if ( (null_counts[cutpoint_ind] == min_null_fit) and (residual_densities_fit[cutpoint_ind] < 0) ):
                    # This means the null fit missed a second hump because it was so extreme
                    found_dip = True
                    break
            if cutpoint_ind < (residual_densities_fit.size - 1):
                cutpoint_ind += 1
        if not found_dip:
            if citical_side == 'left':
                # Ensure we cut closest to center in event of a tie
                cutpoint_ind = np.argmax(residual_densities_fit[-1::-1])
                cutpoint_ind = len(critical_range) - cutpoint_ind - 1
            else:
                cutpoint_ind = np.argmax(residual_densities_fit)
        cutpoint = choose_optimal_cutpoint(cutpoint_ind, residual_densities_fit,
                    x_axis[critical_range])
        
    return p_value, cutpoint


"""
    merge_clusters(data, labels [, ])

This is the workhorse function when performing clustering. It joins individual
clusters together to form larger clusters until it reaches convergence.  Returns
the cluster labels for each spike input.

Explanation of parameters:
 - comparison_pca. When we choose two clusters to compare, should we re-compute
   the principle components (using these "local" scores rather than the global
   components). This can help separate larger clusters into smaller clusters.
 - merge_only. Only perform merges, do not split.
"""
def merge_clusters(data, labels, p_value_cut_thresh=0.01, whiten_clusters=True,
                   merge_only=False, split_only=False, max_iter=20000,
                   flip_labels=True, verbose=False):

    def whiten_cluster_pairs(scores, labels, c1, c2):
        centroid_1 = sort_cython.compute_cluster_centroid(scores, labels, c1)
        centroid_2 = sort_cython.compute_cluster_centroid(scores, labels, c2)
        V = centroid_2 - centroid_1
        avg_cov = np.cov(scores[np.logical_or(labels == c1, labels == c2), :], rowvar=False)
        if np.abs(la.det(avg_cov)) > 1e-6:
            inv_average_covariance = la.inv(avg_cov)
            V = np.matmul(V, inv_average_covariance)

        return V

    # This helper function determines if we should perform merging of two clusters
    # This function returns a boolean if we should merge the clusters
    def merge_test(scores, labels, c1, c2):
        if scores.shape[1] > 1:
            # Get V, the vector connecting the two centroids either with or without whitening
            if whiten_clusters:
                V = whiten_cluster_pairs(scores, labels, c1, c2)
            else:
                centroid_1 = sort_cython.compute_cluster_centroid(scores, labels, c1)
                centroid_2 = sort_cython.compute_cluster_centroid(scores, labels, c2)
                V = centroid_2 - centroid_1
            norm_V = la.norm(V)
            if norm_V == 0:
                # The two cluster centroids are identical so merge
                return True
            V = V / norm_V # Scale by the magnitude to get a unit vector in the appropriate direction
            # Compute the projection of all points from C1 and C2 onto the line
            projection = np.matmul(scores, V)
        else:
            # Can't whiten one and project one dimensional scores, they are already the
            # 1D projection
            projection = np.squeeze(np.copy(scores))

        distances1 = np.sum((scores[labels == c1, :] - np.mean(scores[labels == c1, :], axis=0))**2, axis=1)
        distances2 = np.sum((scores[labels == c2, :] - np.mean(scores[labels == c2, :], axis=0))**2, axis=1)

        p_value, optimal_cut = iso_cut(projection[np.logical_or(labels == c1, labels == c2)], p_value_cut_thresh)
        if p_value >= p_value_cut_thresh: #or np.isnan(p_value):
            # These two clusters should be combined
            if split_only:
                return False
            else:
                return True
        elif merge_only:
            return False # These clusters should be split, but our options say no with merge only.
        else:
            # Reassign based on the optimal value
            select_greater = np.logical_and(np.logical_or(labels == c1, labels == c2), (projection > optimal_cut + 1e-6))
            select_less = np.logical_and(np.logical_or(labels == c1, labels == c2), ~select_greater)
            if flip_labels:
                # Make label with most data going in the same as that going out
                assign_max_c1 = True if np.count_nonzero(labels == c1) >= np.count_nonzero(labels == c2) else False
                if np.count_nonzero(select_greater) >= np.count_nonzero(select_less):
                    if assign_max_c1:
                        labels[select_greater] = c1
                        labels[select_less] = c2
                    else:
                        labels[select_greater] = c2
                        labels[select_less] = c1
                else:
                    if assign_max_c1:
                        labels[select_greater] = c2
                        labels[select_less] = c1
                    else:
                        labels[select_greater] = c1
                        labels[select_less] = c2
                if np.count_nonzero(labels == c1) == 0 or np.count_nonzero(labels == c2) == 0:
                    # Our optimal split forced a merge. This can happen even with
                    # 'split_only' set to True.
                    return True
            else:
                # Reassign labels to cluster keeping label numbers that minimize
                # projection distance between original and new center
                original_c1 = np.median(projection[labels == c1])
                original_c2 = np.median(projection[labels == c2])
                if original_c1 >= original_c2:
                    labels[select_greater] = c1
                    labels[select_less] = c2
                else:
                    labels[select_greater] = c2
                    labels[select_less] = c1

            new_distances1 = np.sum((scores[labels == c1, :] - np.mean(scores[labels == c1, :], axis=0))**2, axis=1)
            new_distances2 = np.sum((scores[labels == c2, :] - np.mean(scores[labels == c2, :], axis=0))**2, axis=1)

            if np.mean(new_distances1) > np.mean(distances1):
                print("Cluster 1 distances got bigger...")
            if np.mean(new_distances2) > np.mean(distances2):
                print("Cluster 2 distances got bigger...")

            return False

    # START ACTUAL OUTER FUNCTION
    if labels.size == 0:
        return labels
    unique_labels, u_counts = np.unique(labels, return_counts=True)
    if unique_labels.size <= 1:
        # Return all labels merged into most prevalent label
        labels[:] = unique_labels[np.argmax(u_counts)]
        return labels
    elif data is None:
        return labels
    elif data.ndim == 1 or data.shape[0] == 1:
        return labels
    if data.size == 1:
        # PCA failed because there is no variance between data points
        # so return all labels as the same, merged into most common label
        labels[:] = unique_labels[np.argmax(u_counts)]
        return labels

    previously_compared_pairs = []
    num_iter = 0
    none_merged = True
    while True:
        if num_iter > max_iter:
            print("Maximum number of iterations exceeded")
            return labels
        minimum_distance_pairs = sort_cython.identify_clusters_to_compare(data, labels, previously_compared_pairs)
        if len(minimum_distance_pairs) == 0 and none_merged:
            break # Done, no more clusters to compare
        none_merged = True
        # if len(minimum_distance_pairs) != 0:
        #     p_value_cut_thresh = original_pval / len(minimum_distance_pairs)
        for c1, c2 in minimum_distance_pairs:
            if verbose: print("Comparing ", c1, " with ", c2)
            n_c1 = np.count_nonzero(labels == c1)
            n_c2 = np.count_nonzero(labels == c2)
            if n_c1 > 1 or n_c2 > 1:
                merge = merge_test(data, labels, c1, c2)
            else:
                # c1 and c2 have one spike each so merge them (algorithm can't
                # split in this case and they are mutually closest pairs)
                merge = True
            if merge:
                # Combine the two clusters together, merging into larger cluster label
                if n_c1 >= n_c2:
                    labels[labels == c2] = c1
                else:
                    labels[labels == c1] = c2
                if verbose: print("Iter: ", num_iter, ", Unique clusters: ", np.unique(labels).size)
                none_merged = False
                # labels changed, so any previous comparison is no longer valid and is removed
                for ind, pair in reversed(list(enumerate(previously_compared_pairs))):
                    if c1 in pair or c2 in pair:
                        del previously_compared_pairs[ind]
            else:
                previously_compared_pairs.append((c1, c2))
                if verbose: print("split clusters, ", c1, c2)
        num_iter += 1

    return labels




def kde_builtin(data, n):
    """
    ----------------------------------------------------------------------------
    THIS IS THE ORIGINAL PYTHON IMPLEMENTATION THAT IS SUPER SLOW AND USED ONLY
    FOR TESTING AGAINST THE COMPILED VERSION IN sort_cython.pyx
    ----------------------------------------------------------------------------
    Kernel density estimate (KDE) with automatic bandwidth selection.

    Returns an array of the KDE and the bandwidth used to compute it.

    This code was adapted to Python from the original MatLab script distributed
    by Zdravko Botev (see Notes below).
    "Reliable and extremely fast kernel density estimator for one-dimensional
    data. Gaussian kernel is assumed and the bandwidth is chosen automatically.
    Unlike many other implementations, this one is immune to problems caused by
    multimodal densities with widely separated modes (see example). The
    estimation does not deteriorate for multimodal densities, because we never
    assume a parametric model for the data."

    Parameters
    ----------
    data : numpy ndarray
        a vector of data from which the density estimate is constructed
    n : float
        the number of mesh points used in the uniform discretization of the
        input data over the interval [MIN, MAX] (min and max is determined from
        the range of values input in data). n has to be a power of two. if n is
        not a power of two, then n is rounded up to the next power of two,
        i.e., n is set to n = 2 ** np.ceil(np.log2(n)).

    Returns
    -------
    density : numpy ndarray
        Column vector of length 'n' with the values of the density estimate at
        the xmesh grid points.
    xmesh : numpy ndarray
        The grid over which the density estimate was computed.
    bandwidth : float
        The optimal bandwidth (Gaussian kernel assumed).

    Example
    -------
    import numpy as np
    import matplotlib.pyplot as plt

    data = np.hstack((np.random.randn(100), np.random.randn(100)*2+35,
                    np.random.randn(100)+55))
    density, xmesh, bandwidth = sort.kde(data, 2**14)
    counts, xvals = np.histogram(data, bins=100)
    plt.bar(xvals[0:-1], counts, width=1, align='edge', color=[.5, .5, .5])
    plt.plot(xmesh, density * (np.amax(counts) / np.amax(density)), color='r')
    plt.xlim(-5, 65)
    plt.show()

    Notes
    -----
    New comments for this Python translation are in triple quotes below.  The
    original comments in the distributed MatLab implementation are indicated
    with hashtag style.

    MatLab code downloaded from:
    https://www.mathworks.com/matlabcentral/fileexchange/
                                14034-kernel-density-estimator
    with references and author information here:
    https://web.maths.unsw.edu.au/~zdravkobotev/
    and here:
    Kernel density estimation via diffusion
    Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010)
    Annals of Statistics, Volume 38, Number 5, pages 2916-2957.

    I removed the original 'MIN' and 'MAX' inputs and instead always set them to
    the default values.  Originally these inputs were optional, and defined as:
    MIN, MAX  - defines the interval [MIN,MAX] on which the density estimate is
                constructed the default values of MIN and MAX are:
                MIN=min(data)-Range/10 and MAX=max(data)+Range/10, where
                Range=max(data)-min(data)

    I removed the original 'cdf' output and only output the 'density' pdf.
    Original cdf output definition was:
        cdf  - column vector of length 'n' with the values of the cdf

    I removed all plotting functionality of the original function.
    """

    def dct1d(data):
        """ I changed this to use the scipy discrete cosine transform instead.
        The only difference is the scipy needs to be set to 1 in the first
        element.  I kept the original function below for reference.
        """
        a = dct(data)
        """ The original implementation below returns 1 for first element, so
        # enforce that here """
        a[0] = 1.
        return a
        """
        --------------------------------------------------------------------
                ORIGINAL FUNCTION BELOW
        --------------------------------------------------------------------
        """
        # computes the discrete cosine transform of the column vector data
        # Compute weights to multiply DFT coefficients
        data_copy = np.copy(data)
        weight = np.hstack((1, 2*(np.exp(-1 * 1j * np.arange(1, data_copy.size) * np.pi / (2 * data_copy.size)))))
        # Re-order the elements of the columns of x
        data_copy = np.hstack((data_copy[0::2], data_copy[-1:0:-2]))
        #Multiply FFT by weights:
        return np.real(weight * fft(data_copy))

    def idct1d(data):
        # computes the inverse discrete cosine transform
        # Compute weights
        weights = data.size * np.exp(1j * np.arange(0, data.size) * np.pi / (2 * data.size))
        # Compute x tilde using equation (5.93) in Jain
        data = np.real(ifft(weights * data))
        # Re-order elements of each column according to equations (5.93) and
        # (5.94) in Jain
        out = np.zeros(data.size)
        out_midslice = int(data.size / 2)
        out[0::2] = data[0:out_midslice]
        out[1::2] = data[-1:out_midslice-1:-1]
        #   Reference:
        #      A. K. Jain, "Fundamentals of Digital Image
        #      Processing", pp. 150-153.

        return out

    def fixed_point(t, N, I, a2):
        # this implements the function t-zeta*gamma^[l](t)
        l = 7
        # f_fac = np.sum(I ** l * a2 * np.exp(-I * np.pi ** 2 * t))
        # This line removes I ** l and keeps things in range of float64
        f_fac = np.sum(np.exp(np.log(I) * l + np.log(a2) - I * np.pi ** 2 * t))
        if f_fac < 1e-6 or N == 0:
            # Prevent zero division, which converges to negative infinity
            return -np.inf
        f = 2 * np.pi ** (2*l) * f_fac
        for s in range(l - 1, 1, -1): # s=l-1:-1:2
            K0 = np.prod(np.arange(1, 2*s, 2)) / np.sqrt(2*np.pi)
            const = (1 + (1/2) ** (s + 1/2)) / 3
            time = (2 * const * K0 / N / f) ** (2 / (3 + 2*s))
            # f_fac = np.sum(I ** s * a2 * np.exp(-I * np.pi ** 2 * time))
            # This line removes I ** s and keeps things in range of float64
            f_fac = np.sum(np.exp(np.log(I) * s + np.log(a2) - I * np.pi ** 2 * time))
            if f_fac < 1e-6:
                # Prevent zero division, which converges to negative infinity
                f = -1.0
                break
            f = 2 * np.pi ** (2*s) * f_fac

        if f > 0.0:
            return t - (2 * N * np.sqrt(np.pi) * f) ** (-2/5)
        else:
            return -np.inf

    def fixed_point_abs(t, N, I, a2):
        """ I added this for the case where no root is found and we seek the
        minimum absolute value in the main optimization while loop below.  It
        is identical to 'fixed_point' above but returns the absolute value.
        """
        f_t = fixed_point(t, N, I, a2)
        # Get absolute value
        if f_t >= 0.0:
            return f_t
        else:
            return -1.0 * f_t

    def bound_grad_desc_fixed_point_abs(N, I, a2, lower, upper, xtol, ytol):
        """ I added this for the case where no root is found and we seek the
        minimum absolute value in the main optimization while loop below.  It
        is identical to 'fixed_point' above but returns the absolute value.
        """
        alpha = 1e-4 # Choose learning rate
        max_iter = 1000
        t_star = lower
        f_min = np.inf
        t = lower

        # Choose starting point as lowest over 10 intervals
        dt = (upper - lower) / 10
        if dt < xtol:
            dt = xtol
        while t <= upper:
            f_t = fixed_point_abs(t, N, I, a2)
            if f_t < f_min:
                t_star = t
                f_min = f_t
            t += dt
        if np.isinf(f_min):
            return 0.0
        # reset t and f_t to lowest point to start search
        t = t_star
        f_t = f_min
        n_iters = 0
        while True:
            f_dt_pl = fixed_point_abs(t + dt, N, I, a2)
            f_dt_mn = fixed_point_abs(t - dt, N, I, a2)
            d_f_t_dt = (f_dt_pl - f_dt_mn) / (2*dt)
            if np.isinf(d_f_t_dt):
                t_star = t
                break

            # Update t according to gradient d_f_t_dt
            next_t = t - alpha * d_f_t_dt
            # If next_t is beyond bounds, choose point halfway
            if next_t >= upper:
                next_t = (upper - t)/2 + t
            if next_t <= lower:
                next_t = (t - lower)/2 + lower
            f_t = fixed_point_abs(next_t, N, I, a2)

            # Get absolute value of change in f_t and t
            f_step = f_t - f_min
            if f_step < 0.0:
                f_step *= -1.0
            t_step = t - next_t
            if t_step < 0.0:
                t_step *= -1.0

            if (f_step < ytol) or (t_step < xtol):
                # So little change we declare ourselves done
                t_star = t
                break
            t = next_t
            dt = t_step
            f_min = f_t
            n_iters += 1
            if n_iters > max_iter:
                t_star = t
                break

        # if do_print: print("SOLUTION CONVERGED IN ", n_iters, "ITERS to", t_star, upper)
        return t_star

    n = 2 ** np.ceil(np.log2(n)) # round up n to the next power of 2
    # define the interval [MIN, MAX]
    MIN = np.amin(data)
    MAX = np.amax(data)
    # Range = maximum - minimum
    # MIN = minimum# - Range / 20 # was divided by 2
    # MAX = maximum# + Range / 20

    density = np.array([1])
    xmesh = np.array([MAX])
    bandwidth = 0
    if MIN == MAX:
        return density, xmesh, bandwidth

    # set up the grid over which the density estimate is computed
    R = MAX - MIN
    dx = R / (n - 1)
    xmesh = MIN + np.arange(0, R+dx, dx)
    N = np.unique(data).size
    # bin the data uniformly using the grid defined above
    """ ADD np.inf as the final bin edge to get MatLab histc like behavior """
    initial_data = np.histogram(data, bins=np.hstack((xmesh, np.inf)))[0] / N
    initial_data = initial_data / np.sum(initial_data)
    a = dct1d(initial_data) # discrete cosine transform of initial data

    # now compute the optimal bandwidth^2 using the referenced method
    I = np.arange(1, n, dtype=np.float64) ** 2 # Do I as float64 so it doesn't overflow in fixed_point
    a2 = (a[1:] / 2) ** 2
    N_tol = 50 * (N <= 50) + 1050 * (N >= 1050) + N * (np.logical_and(N < 1050, N>50))
    tol = 10.0 ** -12.0 + 0.01 * (N_tol - 50.0) / 1000.0
    fixed_point_0 = fixed_point(0, N, I, a2)
    fmin_val = np.inf
    f_0 = fixed_point_0
    tol_0 = 0
    """ This is the main optimization loop to solve for t_star """
    while True:
        f_tol = fixed_point(tol, N, I, a2)
        """ Attempt to find a zero crossing in the fixed_point function moving
        stepwise from fixed_point_0 """
        if np.sign(f_0) != np.sign(f_tol):
            # use fzero to solve the equation t=zeta*gamma^[5](t)
            """ I am using fsolve here rather than MatLab 'fzero' """
            # t_star = fminbound(fixed_point_abs, tol_0, tol, args=(N, I, a2))
            t_star = bound_grad_desc_fixed_point_abs(N, I, a2, tol_0, tol, 1e-6, 1e-6)
            break
        else:
            tol_0 = tol
            tol = min(tol * 2, 1.0) # double search interval
            f_0 = f_tol
        if tol == 1.0: # if all else fails
            """ Failed to find zero crossing so find absolute minimum value """
            # t_star = fminbound(fixed_point_abs, 0, 1.0, args=(N, I, a2))
            t_star = bound_grad_desc_fixed_point_abs(N, I, a2, 0, 1.0, 1e-6, 1e-6)
            break
    # smooth the discrete cosine transform of initial data using t_star
    a_t = a * np.exp(-1 * np.arange(0, n) ** 2 * np.pi ** 2 * t_star / 2)
    # now apply the inverse discrete cosine transform
    density = idct1d(a_t) / R
    # take the rescaling of the data into account
    bandwidth = np.sqrt(t_star) * R
    """ I set this to zero instead of a small epsilon value. """
    density[density < 0] = 0; # remove negatives due to round-off error

    return density, xmesh, bandwidth
