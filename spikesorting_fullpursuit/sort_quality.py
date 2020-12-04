import numpy as np



def calc_m_overlap_ab(clips_a, clips_b, N=100, k_neighbors=10):
    """
    """
    Na = min(N, clips_a.shape[0])
    Nb = min(N, clips_b.shape[0])

    a_dist_b = clips_a - np.mean(clips_b, axis=0)[None, :]
    a_dist_b = -1*(a_dist_b ** 2).sum(axis=1)
    check_a_inds = np.argpartition(a_dist_b, Na)[:Na]
    b_dist_a = clips_b - np.mean(clips_a, axis=0)[None, :]
    b_dist_a = -1*(b_dist_a ** 2).sum(axis=1)
    check_b_inds = np.argpartition(b_dist_a, Nb)[:Nb]

    if k_neighbors > clips_a.shape[0] or k_neighbors > clips_b.shape[0]:
        k_neighbors = min(clips_a.shape[0], clips_b.shape[0])
        print("Input K neighbors for calc_m_overlap_ab exceeds at least one cluster size. Resetting to", k_neighbors)
    clips_aUb = np.vstack((clips_a, clips_b))

    # Number of neighbors of points in a near b that are assigned to b
    k_in_a = 0.
    for a_ind in range(0, Na):
        x = clips_a[check_a_inds[a_ind], :]
        # Distance of x from all points in a union b
        x_dist = np.sum((x - clips_aUb) ** 2, axis=1)
        k_nearest_inds = np.argpartition(x_dist, k_neighbors)[:k_neighbors]
        k_in_a += np.count_nonzero(k_nearest_inds >= clips_a.shape[0])
    #/ k_neighbors) / clips_aUb.shape[0]

    k_in_b = 0.
    for b_ind in range(0, Nb):
        x = clips_b[check_b_inds[b_ind], :]
        # Distance of x from all points in a union b
        x_dist = np.sum((x - clips_aUb) ** 2, axis=1)
        k_nearest_inds = np.argpartition(x_dist, k_neighbors)[:k_neighbors]
        k_in_b += np.count_nonzero(k_nearest_inds < clips_a.shape[0])
    # / k_neighbors) / clips_aUb.shape[0]
    m_overlap_ab = (k_in_a + k_in_b) / ((Na + Nb) * k_neighbors)

    return m_overlap_ab


def calc_m_isolation(clips, neuron_labels, N=100, k_neighbors=10):
    """
    """
    cluster_labels = np.unique(neuron_labels)
    if cluster_labels.size == 1:
        return np.ones(1), cluster_labels
    m_overlap_ab = np.zeros((cluster_labels.shape[0], cluster_labels.shape[0]))
    m_isolation = np.zeros(cluster_labels.shape[0])
    # First get all pairwise overlap measures. These are symmetric
    for a in range(0, cluster_labels.shape[0]):
        clips_a = clips[neuron_labels == cluster_labels[a], :]
        for b in range(a+1, cluster_labels.shape[0]):
            clips_b = clips[neuron_labels == cluster_labels[b], :]
            m_overlap_ab[a, b] = calc_m_overlap_ab(clips_a, clips_b, N, k_neighbors)
            m_overlap_ab[b, a] = m_overlap_ab[a, b]
        m_isolation[a] = np.amax(m_overlap_ab[a, :])

    return m_isolation
