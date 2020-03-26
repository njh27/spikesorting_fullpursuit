import numpy as np
from numpy import linalg as la
from scipy import signal, linalg
from scipy.spatial.distance import pdist
from spikesorting_python.src.c_cython import sort_cython

import matplotlib.pyplot as plt

def get_full_zca_matrix(data, rowvar=True):
    """ Computes ZCA matrix for data. rowvar=False means that COLUMNS represent
        variables and ROWS represent observations.  Else the opposite is used.
        ZCA whitening is done with this matrix by calling:
            zca_filtered_data = np.dot(zca_matrix, data).
        The ZCA procedure was taken (and reformatted into 2 lines) from:
        https://github.com/zellyn/deeplearning-class-2011/blob/master/ufldl/pca_2d/pca_2d.py
        """
    if data.ndim == 1:
        return 1.
    elif data.shape[0] == 1:
        return 1.

    sigma = np.cov(data, rowvar=rowvar)
    U, S, _ = linalg.svd(sigma)
    zca_matrix = U @ np.diag(1.0 / np.sqrt(S + 1e-9)) @ U.T

    return zca_matrix


def get_noise_sampled_zca_matrix(voltage_data, thresholds, sigma, thresh_cushion, n_samples=1e7):
    """
        The ZCA procedure was taken (and reformatted into 2 lines) from:
        https://github.com/zellyn/deeplearning-class-2011/blob/master/ufldl/pca_2d/pca_2d.py
        """
    if voltage_data.ndim == 1:
        return 1.
    zca_thresholds = thresholds
    # convert cushion to zero centered window
    thresh_cushion = (thresh_cushion * 2 + 1)
    volt_thresh_bool = np.zeros(voltage_data.shape, dtype='bool')
    for chan_v in range(0, volt_thresh_bool.shape[0]):
        volt_thresh_bool[chan_v, :] = np.rint(signal.fftconvolve(np.abs(voltage_data[chan_v, :]) > zca_thresholds[chan_v], np.ones(thresh_cushion), mode='same')).astype('bool')
    sigma = np.empty((voltage_data.shape[0], voltage_data.shape[0]))
    for i in range(0, voltage_data.shape[0]):
        # Compute i variance for diagonal elements of sigma
        valid_samples = voltage_data[i, :][~volt_thresh_bool[i, :]]
        if n_samples > valid_samples.size:
            out_samples = valid_samples
        else:
            out_samples = np.random.choice(valid_samples, int(n_samples), replace=True)
        sigma[i, i] = np.var(out_samples, ddof=1)
        ij_samples = np.full((2, voltage_data.shape[0]), np.nan)
        for j in range(i+1, voltage_data.shape[0]):
            row_inds = np.array([i, j], dtype=np.int64)
            valid_samples = np.nonzero(~np.any(volt_thresh_bool[row_inds, :], axis=0))[0]
            if n_samples > valid_samples.size:
                out_samples = valid_samples
            else:
                out_samples = np.random.choice(valid_samples, int(n_samples), replace=True)

            sigma[i, j] = np.dot(voltage_data[i, out_samples], voltage_data[j, out_samples]) / (out_samples.size-1)
            sigma[j, i] = sigma[i, j]
            ij_samples[0, j] = out_samples.size
            ij_samples[1, j] = valid_samples.size
        if i < voltage_data.shape[0] - 1:
            print("ZCA channel", i, "is from approximately", np.around(np.nanmean(ij_samples[0, :])), "samples of", np.around(np.nanmean(ij_samples[1, :])), "available points")
        else:
            print("ZCA channel", i, "is from approximately", out_samples.size, "samples of", valid_samples.size, "available points")

    U, S, _ = linalg.svd(sigma)
    zca_matrix = U @ np.diag(1.0 / np.sqrt(S + 1e-9)) @ U.T

    return zca_matrix


"""
    pca(spikes)

Given a set of spikes which is an MxN matrix (M spikes x N timepoints), we
determine the principle components given a set of spikes. The principle components
are returned in order of decreasing variance (i.e, the first components returned
have the highest variance).

Each column in the returned output corresponds to one principle component. To
compute the "weight" for each PCA, simply multiply matrix wise.
 pca(spikes)[:, 1] * spikes[1, :]
to get the weight of the first principle component for the first spike.
If pc_count is a scalar
"""
def pca_scores(spikes, compute_pcs=None, pcs_as_index=True, return_V=False, return_E=False):

    if spikes.ndim != 2:
        raise ValueError("Input 'spikes' must be a 2 dimensional array to compute PCA")
    if spikes.shape[0] == 1:
        raise ValueError("Must input more than 1 spike to compute PCA")
    if compute_pcs is None:
        compute_pcs = spikes.shape[1]
    spike_std = np.std(spikes, axis=0)

    if np.all(spike_std != 0):
        spike_copy = np.copy(spikes)
        spike_copy -= np.mean(spike_copy, axis=0)
        spike_copy /= spike_std
        C = np.cov(spike_copy, rowvar=False)
        E, V = la.eigh(C)

        # If pcs_as_index is true, treat compute_pcs as index of specific components
        # else compute_pcs must be a scalar and we index from 0:compute_pcs
        if pcs_as_index:
            key = np.argsort(E)[::-1][compute_pcs]
        else:
            key = np.argsort(E)[::-1][:compute_pcs]

        E, V = E[key], V[:, key]
        U = np.matmul(spike_copy, V)
    else:
        # No variance, all PCs are equal! Set to 0
        if compute_pcs is None:
            compute_pcs = spikes.shape[1]
        U = np.array(np.nan) #np.zeros((spikes.shape[0], compute_pcs))
        V = np.array(np.nan) #np.zeros((spikes.shape[1], compute_pcs))
        E = np.array(np.nan) #np.ones(compute_pcs)

    if return_V and return_E:
        return U, V, E
    elif return_V:
        return U, V
    elif return_E:
        return U, E
    else:
        return U


"""
Used as an alternative to 'max_pca_components_cross_validation'.
This function computes the reconstruction based on each principal component
separately and then reorders the principal components according to their
reconstruction accuracy rather than variance accounted for.  It then iterates
through the reconstructions adding one PC at a time in this new order and at each
step computing the ratio of improvement from the addition of a PC.  All PCs up to
and including the first local maxima of this VAF function are output as the
the optimal ones to use. """
def optimal_reconstruction_pca_order(spikes, check_components=None,
                                     max_components=None, min_components=0):
    # Limit max-components based on the size of the dimensions of spikes
    if max_components is None:
        max_components = spikes.shape[1]
    if check_components is None:
        check_components = spikes.shape[1]
    max_components = np.amin([max_components, spikes.shape[1]])
    check_components = np.amin([check_components, spikes.shape[1]])
    if (max_components <= 1) or (check_components <= 1):
        # Only choosing from among the first PC so just return index to first PC
        return np.array([0])

    # Get residual sum of squared error for each PC separately
    resid_error = np.zeros(check_components)
    _, components = pca_scores(spikes, check_components, pcs_as_index=False, return_V=True)
    for comp in range(0, check_components):
        reconstruction = (spikes @ components[:, comp][:, None]) @ components[:, comp][:, None].T
        RESS = np.mean(np.mean((reconstruction - spikes) ** 2, axis=1), axis=0)
        resid_error[comp] = RESS

    # Optimal order of components based on reconstruction accuracy
    comp_order = np.argsort(resid_error)

    # Find improvement given by addition of each ordered PC
    vaf = np.zeros(check_components)
    PRESS = np.mean(np.mean((spikes) ** 2, axis=1), axis=0)
    RESS = np.mean(np.mean((spikes - np.mean(np.mean(spikes, axis=0))) ** 2, axis=1), axis=0)
    vaf[0] = 1. - RESS / PRESS

    PRESS = RESS
    for comp in range(1, check_components):
        reconstruction = (spikes @ components[:, comp_order[0:comp]]) @ components[:, comp_order[0:comp]].T
        RESS = np.mean(np.mean((reconstruction - spikes) ** 2, axis=1), axis=0)
        vaf[comp] = 1. - RESS / PRESS
        PRESS = RESS
        # Choose first local maxima
        if (vaf[comp] < vaf[comp - 1]):
          break
        if comp == max_components:
          # Won't use more than this so break
          break

    max_vaf_components = comp

    # plt.plot(vaf)
    # plt.scatter(max_vaf_components, vaf[max_vaf_components])
    # plt.show()
    # plt.plot(components[:, comp_order[0:comp]])
    # plt.show()

    is_worse_than_mean = False
    if vaf[1] < 0:
        # First PC is worse than the mean
        is_worse_than_mean = True
        max_vaf_components = 1

    # This is to account for slice indexing and edge effects
    if max_vaf_components >= vaf.size - 1:
        # This implies that we found no maxima before reaching the end of vaf
        if vaf[-1] > vaf[-2]:
            # vaf still increasing so choose last point
            max_vaf_components = vaf.size
        else:
            # vaf has become flat so choose second to last point
            max_vaf_components = vaf.size - 1
    if max_vaf_components < min_components:
        max_vaf_components = min_components
    if max_vaf_components > max_components:
        max_vaf_components = max_components

    return comp_order[0:max_vaf_components], is_worse_than_mean


def compute_pca(clips, check_components, max_components, add_peak_valley=False,
                curr_chan_inds=None):
    if add_peak_valley and curr_chan_inds is None:
        raise ValueError("Must supply indices for the main channel if using peak valley")
    # use_components1, _ = optimal_reconstruction_pca_order(clips, check_components, max_components)
    if clips.flags['C_CONTIGUOUS']:
        use_components, _ = sort_cython.optimal_reconstruction_pca_order(clips, check_components, max_components)
    else:
        use_components, _ = sort_cython.optimal_reconstruction_pca_order_F(clips, check_components, max_components)
    # print("Automatic component detection chose", use_components, "PCA components.", flush=True)
    scores = pca_scores(clips, use_components, pcs_as_index=True)
    if add_peak_valley:
        peak_valley = (np.amax(clips[:, curr_chan_inds], axis=1) - np.amin(clips[:, curr_chan_inds], axis=1)).reshape(clips.shape[0], -1)
        peak_valley /= np.amax(np.abs(peak_valley)) # Normalized from -1 to 1
        peak_valley *= np.amax(np.amax(np.abs(scores))) # Normalized to same range as PC scores
        scores = np.hstack((scores, peak_valley))

    return scores


def compute_pca_by_channel(clips, curr_chan_inds, check_components,
                           max_components, add_peak_valley=False):
    if add_peak_valley and curr_chan_inds is None:
        raise ValueError("Must supply indices for the main channel if using peak valley")
    pcs_by_chan = []
    # Do current channel first
    # use_components, _ = optimal_reconstruction_pca_order(clips[:, curr_chan_inds], check_components, max_components, min_components=0)
    use_components, _ = sort_cython.optimal_reconstruction_pca_order_F(clips[:, curr_chan_inds], check_components, max_components, min_components=0)
    # print("Automatic component detection (get by channel) chose", use_components, "PCA components.", flush=True)
    scores = pca_scores(clips[:, curr_chan_inds], use_components, pcs_as_index=True)
    if add_peak_valley:
        peak_valley = (np.amax(clips[:, curr_chan_inds], axis=1) - np.amin(clips[:, curr_chan_inds], axis=1)).reshape(clips.shape[0], -1)
        peak_valley /= np.amax(np.abs(peak_valley)) # Normalized from -1 to 1
        peak_valley *= np.amax(np.amax(np.abs(scores))) # Normalized to same range as PC scores
        scores = np.hstack((scores, peak_valley))
    pcs_by_chan.append(scores)
    n_curr_max = use_components.size

    samples_per_chan = curr_chan_inds.size
    n_estimated_chans = clips.shape[1] // samples_per_chan
    for ch in range(0, n_estimated_chans):
        if ch*samples_per_chan == curr_chan_inds[0]:
            continue
        ch_inds = np.arange(ch*samples_per_chan, (ch+1)*samples_per_chan)
        # use_components, is_worse_than_mean = optimal_reconstruction_pca_order(clips[:, ch_inds], check_components, max_components)
        use_components, is_worse_than_mean = sort_cython.optimal_reconstruction_pca_order_F(clips[:, ch_inds], check_components, max_components)
        if is_worse_than_mean:
            # print("Automatic component detection (get by channel) chose !NO! PCA components.", flush=True)
            continue
        # if use_components.size > n_curr_max:
        #     use_components = use_components[0:n_curr_max]
        # print("Automatic component detection (get by channel) chose", use_components, "PCA components.", flush=True)
        scores = pca_scores(clips[:, ch_inds], use_components, pcs_as_index=True)
        pcs_by_chan.append(scores)

    return np.hstack(pcs_by_chan)


def minimal_redundancy_template_order(spikes, templates, max_templates=None, first_template_ind=None):
    """
    """
    if max_templates is None:
        max_templates = templates.shape[0]

    templates_copy = np.copy(templates).T
    new_temp_order = np.zeros_like(templates_copy)
    temps_remaining = [int(x) for x in range(0, templates_copy.shape[1])]

    first_template_ind = first_template_ind / np.sum(first_template_ind)
    template_sizes = np.sum(templates_copy ** 2, axis=0) * first_template_ind

    # if first_template_ind is None:
    #     first_template_ind = np.argmax(template_sizes)
    # new_temp_order[:, 0] = templates_copy[:, first_template_ind]
    # temps_remaining.remove(first_template_ind)
    # total_function = [template_sizes[first_template_ind]]

    f_ind = np.argmax(template_sizes)
    new_temp_order[:, 0] = templates_copy[:, f_ind]
    temps_remaining.remove(f_ind)
    total_function = [template_sizes[f_ind]]

    temp_seeking = 1
    while len(temps_remaining) > 0:
        test_templates = new_temp_order[:, 0:temp_seeking]
        dot_products = np.zeros(len(temps_remaining))
        for t_ind, t in enumerate(temps_remaining):
            # dot_products[t_ind] = np.abs(template_sizes[t] - np.median(templates_copy[:, t] @ test_templates))
            # dot_products[t_ind] = np.amin(np.abs(template_sizes[t] - templates_copy[:, t] @ test_templates))
            dot_products[t_ind] = np.sum(templates_copy[:, t] @ test_templates) / template_sizes[t]
        # next_best_temp = temps_remaining[np.argmax(dot_products)]
        # total_function.append((dot_products[np.argmax(dot_products)]))
        next_best_temp = temps_remaining[np.argmin(dot_products)]
        total_function.append((np.abs(dot_products[np.argmin(dot_products)])))
        new_temp_order[:, temp_seeking] = templates_copy[:, next_best_temp]
        temps_remaining.remove(next_best_temp)

        temp_seeking += 1

    total_function = np.hstack(total_function)
    vaf = np.zeros_like(total_function)
    total_function[0] = np.inf
    for df in range(1, total_function.size):
        this_vaf = total_function[df] / total_function[df-1]
        vaf[df] = this_vaf
        # if vaf[df] < vaf[df-1]:
        #     break
    df = np.argmax(vaf) + 1
    if df < 2:
        df = 2
    if df > max_templates:
        df = max_templates
    new_temp_order = new_temp_order[:, 0:df]
    print("CHOSE", df, "TEMPLATES FOR PROJECTION")
    # for tdf in range(0, df):
    #     plt.plot(new_temp_order[:, tdf])
    #     plt.show()
    # plt.plot(vaf)
    # plt.show()
    return new_temp_order.T


def compute_template_projection(clips, labels, curr_chan_inds, add_peak_valley=False, max_templates=None):
    """
    """
    if add_peak_valley and curr_chan_inds is None:
        raise ValueError("Must supply indices for the main channel if using peak valley")
    # Compute the weights using projection onto each neurons' template
    unique_labels, u_counts = np.unique(labels, return_counts=True)
    if max_templates is None:
        max_templates = unique_labels.size
    templates = np.empty((unique_labels.size, clips.shape[1]))
    for ind, l in enumerate(unique_labels):
        templates[ind, :] = np.mean(clips[labels == l, :], axis=0)
    templates = minimal_redundancy_template_order(clips, templates,
                    max_templates=max_templates, first_template_ind=u_counts)
    # Keep at most the max_templates templates
    templates = templates[0:min(templates.shape[0], max_templates), :]
    scores = clips @ templates.T

    if add_peak_valley:
        peak_valley = (np.amax(clips[:, curr_chan_inds], axis=1) - np.amin(clips[:, curr_chan_inds], axis=1)).reshape(clips.shape[0], -1)
        peak_valley /= np.amax(np.abs(peak_valley)) # Normalized from -1 to 1
        peak_valley *= np.amax(np.amax(np.abs(scores))) # Normalized to same range as PC scores
        scores = np.hstack((scores, peak_valley))

    return scores


def compute_template_pca(clips, labels, curr_chan_inds, check_components, max_components, add_peak_valley=False):
    if add_peak_valley and curr_chan_inds is None:
        raise ValueError("Must supply indices for the main channel if using peak valley")
    # Compute the weights using the PCA for neuron templates
    unique_labels, u_counts = np.unique(labels, return_counts=True)
    if unique_labels.size == 1:
        # Can't do PCA on one template
        return None
    templates = np.empty((unique_labels.size, clips.shape[1]))
    for ind, l in enumerate(unique_labels):
        templates[ind, :] = np.mean(clips[labels == l, :], axis=0) * np.sqrt(u_counts[ind] / labels.size)

    # use_components, _ = optimal_reconstruction_pca_order(templates, check_components, max_components)
    use_components, _ = sort_cython.optimal_reconstruction_pca_order(templates, check_components, max_components)
    # print("Automatic component detection (FULL TEMPLATES) chose", use_components, "PCA components.")
    _, score_mat = pca_scores(templates, use_components, pcs_as_index=True, return_V=True)
    scores = clips @ score_mat
    if add_peak_valley:
        peak_valley = (np.amax(clips[:, curr_chan_inds], axis=1) - np.amin(clips[:, curr_chan_inds], axis=1)).reshape(clips.shape[0], -1)
        peak_valley /= np.amax(np.abs(peak_valley)) # Normalized from -1 to 1
        peak_valley *= np.amax(np.amax(np.abs(scores))) # Normalized to same range as PC scores
        scores = np.hstack((scores, peak_valley))

    return scores


def compute_template_pca_by_channel(clips, labels, curr_chan_inds, check_components, max_components, add_peak_valley=False):
    if add_peak_valley and curr_chan_inds is None:
        raise ValueError("Must supply indices for the main channel if using peak valley")
    # Compute the weights using the PCA for neuron templates
    unique_labels, u_counts = np.unique(labels, return_counts=True)
    if unique_labels.size == 1:
        # Can't do PCA on one template
        return None
    templates = np.empty((unique_labels.size, clips.shape[1]))
    for ind, l in enumerate(unique_labels):
        templates[ind, :] = np.mean(clips[labels == l, :], axis=0) * np.sqrt(u_counts[ind] / labels.size)

    pcs_by_chan = []
    # Do current channel first
    # use_components, _ = optimal_reconstruction_pca_order(templates[:, curr_chan_inds], check_components, max_components)
    use_components, _ = sort_cython.optimal_reconstruction_pca_order_F(templates[:, curr_chan_inds], check_components, max_components)
    # print("Automatic component detection (TEMPLATES by channel) chose", use_components, "PCA components.")
    _, score_mat = pca_scores(templates[:, curr_chan_inds], use_components, pcs_as_index=True, return_V=True)
    scores = clips[:, curr_chan_inds] @ score_mat
    if add_peak_valley:
        peak_valley = (np.amax(clips[:, curr_chan_inds], axis=1) - np.amin(clips[:, curr_chan_inds], axis=1)).reshape(clips.shape[0], -1)
        peak_valley /= np.amax(np.abs(peak_valley)) # Normalized from -1 to 1
        peak_valley *= np.amax(np.amax(np.abs(scores))) # Normalized to same range as PC scores
        scores = np.hstack((scores, peak_valley))
    pcs_by_chan.append(scores)
    n_curr_max = use_components.size

    samples_per_chan = curr_chan_inds.size
    for ch in range(0, clips.shape[1] // samples_per_chan):
        if ch*samples_per_chan == curr_chan_inds[0]:
            continue
        ch_inds = np.arange(ch*samples_per_chan, (ch+1)*samples_per_chan)
        # use_components, is_worse_than_mean = optimal_reconstruction_pca_order(templates[:, ch_inds], check_components, max_components)
        use_components, is_worse_than_mean = sort_cython.optimal_reconstruction_pca_order_F(templates[:, ch_inds], check_components, max_components)
        if is_worse_than_mean:
            # print("Automatic component detection (TEMPLATES by channel) chose !NO! PCA components.", flush=True)
            continue
        # if use_components.size > n_curr_max:
        #     use_components = use_components[0:n_curr_max]
        # print("Automatic component detection (TEMPLATES by channel) chose", use_components, "PCA components.")
        _, score_mat = pca_scores(templates[:, ch_inds], use_components, pcs_as_index=True, return_V=True)
        scores = clips[:, ch_inds] @ score_mat
        pcs_by_chan.append(scores)

    return np.hstack(pcs_by_chan)
