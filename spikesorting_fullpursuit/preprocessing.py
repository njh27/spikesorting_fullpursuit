import numpy as np
from numpy import linalg as la
from scipy import signal, linalg
from scipy.spatial.distance import pdist
from spikesorting_fullpursuit.c_cython import sort_cython



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

    # Scale each of the rows of Z
    # This ensures that the output of the matrix multiplication (Z * channel voltages)
    # remains in the same range (doesn't clip ints)
    zca_matrix = zca_matrix / np.diag(zca_matrix)[:, None]

    return zca_matrix


def get_noise_sampled_zca_matrix(voltage_data, thresholds, sigma, thresh_cushion,
                                 n_samples=1e6):
    """
        The ZCA procedure was taken (and reformatted into 2 lines) from:
        https://github.com/zellyn/deeplearning-class-2011/blob/master/ufldl/pca_2d/pca_2d.py
        """
    if voltage_data.ndim == 1:
        return 1.
    zca_thresholds = np.copy(thresholds)
    # convert cushion to zero centered window
    thresh_cushion = (thresh_cushion * 2 + 1)
    volt_thresh_bool = np.zeros(voltage_data.shape, dtype="bool")
    for chan_v in range(0, volt_thresh_bool.shape[0]):
        volt_thresh_bool[chan_v, :] = np.rint(signal.fftconvolve(np.abs(voltage_data[chan_v, :]) > zca_thresholds[chan_v], np.ones(thresh_cushion), mode='same')).astype("bool")
    sigma = np.empty((voltage_data.shape[0], voltage_data.shape[0]))
    for i in range(0, voltage_data.shape[0]):
        # Compute i variance for diagonal elements of sigma
        valid_samples = voltage_data[i, :][~volt_thresh_bool[i, :]]
        if valid_samples.size == 0:
            raise ValueError("No data points under threshold to compute ZCA. Check threshold sigma and clip width.")
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

    U, S, _ = linalg.svd(sigma)
    zca_matrix = U @ np.diag(1.0 / np.sqrt(S + 1e-9)) @ U.T
    # Scale each of the rows of Z
    # This ensures that the output of the matrix multiplication (Z * channel voltages)
    # remains in the same range (doesn't clip ints)
    zca_matrix = zca_matrix / np.diag(zca_matrix)[:, None]
    # print("TURNED OFF UNDO ZCA SCALING LINE 81 PREPROCESSING")

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
    if spikes.shape[0] <= 1:
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
        # No variance, all PCs are equal! Set to None(s)
        U = None
        V = None
        E = None

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

    # Get residual sum of squared error for each PC separately
    resid_error = np.zeros(check_components)
    _, components = pca_scores(spikes, check_components, pcs_as_index=False, return_V=True)
    if components is None:
        # Couldn't compute PCs
        return np.zeros(1, dtype=np.int64), True
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
                curr_chan_inds=None, n_samples=1e5):
    if add_peak_valley and curr_chan_inds is None:
        raise ValueError("Must supply indices for the main channel if using peak valley")
    # Get a sample of the current clips for PCA reconstruction so that memory
    # usage does not explode. Copy from memmap clips to memory
    # PCA order functions use double precision and are compiled that way
    mem_order = "F" if clips.flags['F_CONTIGUOUS'] else "C"
    if n_samples > clips.shape[0]:
        sample_clips = np.empty(clips.shape, dtype=np.float64, order=mem_order)
        np.copyto(sample_clips, clips)
    else:
        sample_clips = np.empty((int(n_samples), clips.shape[1]), dtype=np.float64, order=mem_order)
        sel_inds = np.random.choice(clips.shape[0], int(n_samples), replace=True)
        np.copyto(sample_clips, clips[sel_inds, :])

    if mem_order == "C":
        use_components, _ = sort_cython.optimal_reconstruction_pca_order(sample_clips, check_components, max_components)
    else:
        use_components, _ = sort_cython.optimal_reconstruction_pca_order_F(sample_clips, check_components, max_components)
    # Return the PC matrix computed over sampled data
    scores, V = pca_scores(sample_clips, use_components, pcs_as_index=True, return_V=True)
    if scores is None:
        # Usually means all clips were the same or there was <= 1, so PCs can't
        # be computed. Just return everything as the same score
        return np.zeros((clips.shape[0], 1))
    else:
        # Convert ALL clips to scores
        scores = np.matmul(clips, V)
    if add_peak_valley:
        peak_valley = (np.amax(clips[:, curr_chan_inds], axis=1) - np.amin(clips[:, curr_chan_inds], axis=1)).reshape(clips.shape[0], -1)
        peak_valley /= np.amax(np.abs(peak_valley)) # Normalized from -1 to 1
        peak_valley *= np.amax(np.amax(np.abs(scores))) # Normalized to same range as PC scores
        scores = np.hstack((scores, peak_valley))

    return scores


def compute_pca_by_channel(clips, curr_chan_inds, check_components,
                           max_components, add_peak_valley=False, n_samples=1e5):
    if add_peak_valley and curr_chan_inds is None:
        raise ValueError("Must supply indices for the main channel if using peak valley")
    pcs_by_chan = []
    eigs_by_chan = []
    # Do current channel first
    # use_components, _ = optimal_reconstruction_pca_order(clips[:, curr_chan_inds], check_components, max_components, min_components=0)
    # NOTE: Slicing SWITCHES C and F ordering so check!
    mem_order = "F" if clips[:, curr_chan_inds[0]:curr_chan_inds[-1]+1].flags['F_CONTIGUOUS'] else "C"
    if n_samples > clips.shape[0]:
        sample_clips = np.empty((clips.shape[0], len(curr_chan_inds)), dtype=np.float64, order=mem_order)
        np.copyto(sample_clips, clips[:, curr_chan_inds[0]:curr_chan_inds[-1]+1])
        sel_inds = np.arange(0, clips.shape[0])
    else:
        sample_clips = np.empty((int(n_samples), len(curr_chan_inds)), dtype=np.float64, order=mem_order)
        sel_inds = np.random.choice(clips.shape[0], int(n_samples), replace=True)
        np.copyto(sample_clips, clips[sel_inds, curr_chan_inds[0]:curr_chan_inds[-1]+1])

    if mem_order == "C":
        use_components, _ = sort_cython.optimal_reconstruction_pca_order(sample_clips, check_components, max_components)
    else:
        use_components, _ = sort_cython.optimal_reconstruction_pca_order_F(sample_clips, check_components, max_components)
    scores, V, eigs = pca_scores(sample_clips, use_components, pcs_as_index=True, return_E=True, return_V=True)
    if scores is None:
        # Usually means all clips were the same or there was <= 1, so PCs can't
        # be computed. Just return everything as the same score
        return np.zeros((clips.shape[0], 1))
    else:
        # Convert ALL clips to scores
        scores = np.matmul(clips[:, curr_chan_inds[0]:curr_chan_inds[-1]+1], V)
    if add_peak_valley:
        peak_valley = (np.amax(clips[:, curr_chan_inds[0]:curr_chan_inds[-1]+1], axis=1) - np.amin(clips[:, curr_chan_inds[0]:curr_chan_inds[-1]+1], axis=1)).reshape(clips.shape[0], -1)
        peak_valley /= np.amax(np.abs(peak_valley)) # Normalized from -1 to 1
        peak_valley *= np.amax(np.amax(np.abs(scores))) # Normalized to same range as PC scores
        scores = np.hstack((scores, peak_valley))
    pcs_by_chan.append(scores)
    eigs_by_chan.append(eigs)
    n_curr_max = use_components.size

    samples_per_chan = curr_chan_inds.size
    n_estimated_chans = clips.shape[1] // samples_per_chan
    for ch in range(0, n_estimated_chans):
        if ch*samples_per_chan == curr_chan_inds[0]:
            continue
        ch_win = [ch*samples_per_chan, (ch+1)*samples_per_chan]
        # Copy channel data to memory in sample clips using same clips as before
        np.copyto(sample_clips, clips[sel_inds, ch_win[0]:ch_win[1]])
        if mem_order == "C":
            use_components, is_worse_than_mean = sort_cython.optimal_reconstruction_pca_order(sample_clips, check_components, max_components)
        else:
            use_components, is_worse_than_mean = sort_cython.optimal_reconstruction_pca_order_F(sample_clips, check_components, max_components)
        if is_worse_than_mean:
            # print("Automatic component detection (get by channel) chose !NO! PCA components.", flush=True)
            continue
        scores, V, eigs = pca_scores(clips[:, ch_win[0]:ch_win[1]], use_components, pcs_as_index=True, return_E=True, return_V=True)
        if scores is not None:
            # Convert ALL clips to scores
            scores = np.matmul(clips[:, ch_win[0]:ch_win[1]], V)
        pcs_by_chan.append(scores)
        eigs_by_chan.append(eigs)

    # Keep only the max components by eigenvalue
    pcs_by_chan = np.hstack(pcs_by_chan)
    if pcs_by_chan.shape[1] > max_components:
        eigs_by_chan = np.hstack(eigs_by_chan)
        comp_order = np.argsort(eigs_by_chan)
        pcs_by_chan = pcs_by_chan[:, comp_order]
        pcs_by_chan = pcs_by_chan[:, 0:max_components]
        pcs_by_chan = np.ascontiguousarray(pcs_by_chan)

    return pcs_by_chan


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


def compute_template_pca(clips, labels, curr_chan_inds, check_components,
        max_components, add_peak_valley=False, use_weights=True):
    if add_peak_valley and curr_chan_inds is None:
        raise ValueError("Must supply indices for the main channel if using peak valley")
    # Compute the weights using the PCA for neuron templates
    unique_labels, u_counts = np.unique(labels, return_counts=True)
    if unique_labels.size == 1:
        # Can't do PCA on one template
        return None
    templates = np.empty((unique_labels.size, clips.shape[1]))
    for ind, l in enumerate(unique_labels):
        templates[ind, :] = np.mean(clips[labels == l, :], axis=0)
        if use_weights:
            templates[ind, :] *= np.sqrt(u_counts[ind] / labels.size)

    # use_components, _ = optimal_reconstruction_pca_order(templates, check_components, max_components)
    # PCA order functions use double precision and are compiled that way, so cast
    # here and convert back afterward instead of carrying two copies. Scores
    # will then be output as doubles.
    clip_dtype = clips.dtype
    clips = clips.astype(np.float64)
    if templates.flags['C_CONTIGUOUS']:
        use_components, _ = sort_cython.optimal_reconstruction_pca_order(templates, check_components, max_components)
    else:
        use_components, _ = sort_cython.optimal_reconstruction_pca_order_F(templates, check_components, max_components)
    # print("Automatic component detection (FULL TEMPLATES) chose", use_components, "PCA components.")
    _, score_mat = pca_scores(templates, use_components, pcs_as_index=True, return_V=True)
    if score_mat is None:
        # Usually means all clips were the same or there was <= 1, so PCs can't
        # be computed. Just return everything as the same score
        return np.zeros((clips.shape[0], 1))
    scores = clips @ score_mat
    if add_peak_valley:
        peak_valley = (np.amax(clips[:, curr_chan_inds], axis=1) - np.amin(clips[:, curr_chan_inds], axis=1)).reshape(clips.shape[0], -1)
        peak_valley /= np.amax(np.abs(peak_valley)) # Normalized from -1 to 1
        peak_valley *= np.amax(np.amax(np.abs(scores))) # Normalized to same range as PC scores
        scores = np.hstack((scores, peak_valley))
    clips = clips.astype(clip_dtype)

    return scores


def compute_template_pca_by_channel(clips, labels, curr_chan_inds,
        check_components, max_components, add_peak_valley=False, use_weights=True):
    if curr_chan_inds is None:
        raise ValueError("Must supply indices for the main channel for computing PCA by channel")
    # Compute the weights using the PCA for neuron templates
    unique_labels, u_counts = np.unique(labels, return_counts=True)
    if unique_labels.size == 1:
        # Can't do PCA on one template
        return None
    templates = np.empty((unique_labels.size, clips.shape[1]))
    for ind, l in enumerate(unique_labels):
        templates[ind, :] = np.median(clips[labels == l, :], axis=0)
        if use_weights:
            templates[ind, :] *= np.sqrt(u_counts[ind] / labels.size)

    pcs_by_chan = []
    # Do current channel first
    # use_components, _ = optimal_reconstruction_pca_order(templates[:, curr_chan_inds], check_components, max_components)
    is_c_contiguous = templates[:, curr_chan_inds].flags['C_CONTIGUOUS']
    # PCA order functions use double precision and are compiled that way, so cast
    # here and convert back afterward instead of carrying two copies. Scores
    # will then be output as doubles.
    clip_dtype = clips.dtype
    clips = clips.astype(np.float64)
    if is_c_contiguous:
        use_components, _ = sort_cython.optimal_reconstruction_pca_order(templates[:, curr_chan_inds], check_components, max_components)
    else:
        use_components, _ = sort_cython.optimal_reconstruction_pca_order_F(templates[:, curr_chan_inds], check_components, max_components)
    # print("Automatic component detection (TEMPLATES by channel) chose", use_components, "PCA components.")
    _, score_mat = pca_scores(templates[:, curr_chan_inds], use_components, pcs_as_index=True, return_V=True)
    if score_mat is None:
        # Usually means all clips were the same or there was <= 1, so PCs can't
        # be computed. Just return everything as the same score
        print("FAILED TO FIND PCS !!!")
        return np.zeros((clips.shape[0], 1))
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
        if is_c_contiguous:
            use_components, is_worse_than_mean = sort_cython.optimal_reconstruction_pca_order(templates[:, ch_inds], check_components, max_components)
        else:
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
    clips = clips.astype(clip_dtype)

    return np.hstack(pcs_by_chan)


def keep_max_on_main(clips, main_chan_inds):

    keep_clips = np.ones(clips.shape[0], dtype="bool")
    for c in range(0, clips.shape[0]):
        max_ind = np.argmax(np.abs(clips[c, :]))
        if max_ind < main_chan_inds[0] or max_ind > main_chan_inds[-1]:
            keep_clips[c] = False

    return keep_clips


def cleanup_clusters(clips, neuron_labels):

    keep_clips = np.ones(clips.shape[0], dtype="bool")

    total_SSE_clips = np.sum(clips ** 2, axis=1)
    total_mean_SSE_clips = np.mean(total_SSE_clips)
    total_STD_clips = np.std(total_SSE_clips)
    overall_deviant = np.logical_or(total_SSE_clips > total_mean_SSE_clips + 5*total_STD_clips,
                        total_SSE_clips < total_mean_SSE_clips - 5*total_STD_clips)
    keep_clips[overall_deviant] = False

    for nl in np.unique(neuron_labels):
        select_nl = neuron_labels == nl
        select_nl[overall_deviant] = False
        nl_template = np.mean(clips[select_nl, :], axis=0)
        nl_SSE_clips = np.sum((clips[select_nl, :] - nl_template) ** 2, axis=1)
        nl_mean_SSE_clips = np.mean(nl_SSE_clips)
        nl_STD_clips = np.std(nl_SSE_clips)
        for nl_ind in range(0, clips.shape[0]):
            if not select_nl[nl_ind]:
                continue
            curr_SSE = np.sum((clips[nl_ind, :] - nl_template) ** 2)
            if np.logical_or(curr_SSE > nl_mean_SSE_clips + 2*nl_STD_clips,
                             curr_SSE < nl_mean_SSE_clips - 2*nl_STD_clips):
                keep_clips[nl_ind] = False

    return keep_clips

def calculate_robust_template(clips):

    if clips.shape[0] == 1 or clips.ndim == 1:
        # Only 1 clip so nothing to average over
        return clips
    robust_template = np.zeros(clips.shape[1], dtype=clips.dtype)
    sample_medians = np.median(clips, axis=0)
    for sample in range(0, clips.shape[1]):
        # Compute MAD with standard deviation conversion factor
        sample_MAD = np.median(np.abs(clips[:, sample] - sample_medians[sample])) / 0.6745
        # Samples within 1 MAD
        select_1MAD = np.logical_and(clips[:, sample] > sample_medians[sample] - sample_MAD,
                                     clips[:, sample] < sample_medians[sample] + sample_MAD)
        if ~np.any(select_1MAD):
            # Nothing within 1 MAD STD so just fall back on median
            robust_template[sample] = sample_medians[sample]
        else:
            # Robust template as median of samples within 1 MAD
            robust_template[sample] = np.median(clips[select_1MAD, sample])

    return robust_template


def keep_cluster_centroid(clips, neuron_labels, n_keep=100):
    keep_clips = np.ones(clips.shape[0], dtype="bool")
    if n_keep > clips.shape[0]:
        # Everything will be kept no matter what so just exit
        return keep_clips
    for nl in np.unique(neuron_labels):
        select_nl = neuron_labels == nl
        nl_template = np.mean(clips[select_nl, :], axis=0)
        nl_distances = np.sum((clips[select_nl, :] - nl_template[None, :]) ** 2, axis=1)
        dist_order = np.argpartition(nl_distances, min(n_keep, nl_distances.shape[0]-1))[0:min(n_keep, nl_distances.shape[0])]
        select_dist = np.zeros(nl_distances.shape[0], dtype="bool")
        select_dist[dist_order] = True
        keep_clips[select_nl] = select_dist

    return keep_clips
