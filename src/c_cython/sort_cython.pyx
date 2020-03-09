cimport numpy as np
import numpy as np
from libc.stdint cimport int64_t, int32_t
from numpy.math cimport INFINITY, NAN
from numpy import linalg as la
cimport cython
from libc.math cimport exp, sqrt, M_PI, M_SQRT2, log2, ceil, floor, isfinite
from libc.stdlib cimport calloc, free
from scipy.fftpack import dct, ifft



def pca_scores(spikes, compute_pcs=None, pcs_as_index=True, return_V=False, return_E=False):
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)    # turn division by zero checking off
def optimal_reconstruction_pca_order(np.ndarray[double, ndim=2, mode="c"] spikes not None,
        check_components=None, max_components=None, int64_t min_components=0):
  """
  Used as an alternative to 'max_pca_components_cross_validation'.
  This function computes the reconstruction based on each principal component
  separately and then reorders the principal components according to their
  reconstruction accuracy rather than variance accounted for.  It then iterates
  through the reconstructions adding one PC at a time in this new order and at each
  step computing the ratio of improvement from the addition of a PC.  All PCs up to
  and including the first local maxima of this VAF function are output as the
  the optimal ones to use. """

  cdef Py_ssize_t comp, x, y, c_ord, check_comp_ssize_t, nv
  cdef double curr_answer
  cdef double *spikes_ptr
  cdef Py_ssize_t spikes_x = spikes.shape[0]
  cdef Py_ssize_t spikes_y = spikes.shape[1]

  # Limit max-components based on the size of the dimensions of spikes
  if max_components is None:
    max_components = spikes_y
  if check_components is None:
    check_components = spikes_y
  max_components = np.amin([max_components, spikes_y])
  check_components = np.amin([check_components, spikes_y])
  if (max_components <= 1) or (check_components <= 1):
    # Only choosing from among the first PC so just return index to first PC
    return np.array([0])
  check_comp_ssize_t = <Py_ssize_t>check_components

  # Get residual sum of squared error for each PC separately
  cdef np.ndarray[double, ndim=1, mode="c"] resid_error = np.zeros(check_components)
  cdef double *resid_error_ptr = &resid_error[0]
  cdef np.ndarray[double, ndim=2, mode="fortran"] components
  _, components = pca_scores(spikes, check_components, pcs_as_index=False, return_V=True)
  cdef double *components_ptr = &components[0, 0]
  cdef double *reconstruction_ptr

  cdef double RESS = 0.0
  cdef int64_t idx_sp, idx_rec
  for comp in range(0, check_comp_ssize_t):
    # Compute (spikes @ components[:, comp][:, None]). Here we can store the
    # result of this first multiplication in reconstruction
    reconstruction_ptr = <double *> calloc(spikes_x * spikes_y, sizeof(double))
    spikes_ptr = &spikes[0,0]
    idx_sp = 0
    for x in range(0, spikes_x):
      idx_rec = x * spikes_y + comp
      for y in range(0, spikes_y):
        reconstruction_ptr[idx_rec] += spikes_ptr[idx_sp] * components_ptr[spikes_y * comp + y]
        idx_sp += 1

    # reconstruction_ptr = &reconstruction[0, 0]
    spikes_ptr = &spikes[0,0]
    idx_sp = 0
    # Now compute result above (reconstruction[x, comp]) @ components[:, comp][:, None].T
    for x in range(0, spikes_x):
      curr_answer = reconstruction_ptr[x * spikes_y + comp]
      for y in range(0, spikes_y):
        reconstruction_ptr[idx_sp] = curr_answer * components_ptr[spikes_y * comp + y]
        RESS += (reconstruction_ptr[idx_sp] - spikes_ptr[idx_sp]) ** 2
        idx_sp += 1

    # Compute mean residual error over all points
    RESS /= spikes_x * spikes_y
    resid_error_ptr[comp] = RESS
    free(reconstruction_ptr)
    RESS = 0.0

  # Optimal order of components based on reconstruction accuracy
  comp_order = np.argsort(resid_error)

  # Find improvement given by addition of each ordered PC
  cdef np.ndarray[double, ndim=1, mode="c"] vaf = np.zeros(check_comp_ssize_t)
  cdef double *vaf_ptr = &vaf[0]
  # Start with RESS as total error, i.e. no improvement
  for x in range(0, spikes_x):
    for y in range(0, spikes_y):
      RESS += spikes[x, y] ** 2
  RESS /= spikes_x * spikes_y
  if RESS < 1.0e-14:
    if min_components == 0:
      return np.array([])
    else:
      return np.arange(0, min_components)

  cdef double PRESS = resid_error[comp_order[0]]
  cdef Py_ssize_t max_vaf_components
  for comp in range(1, check_comp_ssize_t):
    reconstruction = (spikes @ components[:, comp_order[0:comp]]) @ components[:, comp_order[0:comp]].T
    RESS = np.mean(np.mean((reconstruction - spikes) ** 2, axis=1), axis=0)
    vaf_ptr[comp] = 1 - RESS / PRESS
    PRESS = RESS

    # Choose first local maxima as point at which there is decrease in vaf
    if (vaf_ptr[comp] > vaf_ptr[comp - 1]) and (comp > 2):
      max_vaf_components = comp # Used as slice so this includes peak
      break
    if comp == max_components:
      # Won't use more than this so break
      max_vaf_components = comp
      break

  # This is to account for slice indexing and edge effects
  if comp >= check_comp_ssize_t - 1:
    # This implies that we found no maxima before reaching the end of vaf
    if vaf_ptr[check_comp_ssize_t] > vaf_ptr[check_comp_ssize_t - 1]:
      # vaf still increasing so choose last point
      max_vaf_components = check_comp_ssize_t
    else:
      # vaf has become flat so choose second to last point
      max_vaf_components = check_comp_ssize_t - 1
  if max_vaf_components > max_components:
      max_vaf_components = max_components
  if max_vaf_components < min_components:
      max_vaf_components = min_components
  return comp_order[0:max_vaf_components]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)    # turn division by zero checking off
def optimal_reconstruction_pca_order_F(np.ndarray[double, ndim=2, mode="fortran"] spikes not None,
        check_components=None, max_components=None, int64_t min_components=0):
  """
  Used as an alternative to 'max_pca_components_cross_validation'.
  This function computes the reconstruction based on each principal component
  separately and then reorders the principal components according to their
  reconstruction accuracy rather than variance accounted for.  It then iterates
  through the reconstructions adding one PC at a time in this new order and at each
  step computing the ratio of improvement from the addition of a PC.  All PCs up to
  and including the first local maxima of this VAF function are output as the
  the optimal ones to use. """

  cdef Py_ssize_t comp, x, y, c_ord, check_comp_ssize_t, nv
  cdef double curr_answer
  cdef double *spikes_ptr
  cdef Py_ssize_t spikes_x = spikes.shape[0]
  cdef Py_ssize_t spikes_y = spikes.shape[1]

  # Limit max-components based on the size of the dimensions of spikes
  if max_components is None:
    max_components = spikes_y
  if check_components is None:
    check_components = spikes_y
  max_components = np.amin([max_components, spikes_y])
  check_components = np.amin([check_components, spikes_y])
  if (max_components <= 1) or (check_components <= 1):
    # Only choosing from among the first PC so just return index to first PC
    return np.array([0])
  check_comp_ssize_t = <Py_ssize_t>check_components

  # Get residual sum of squared error for each PC separately
  cdef np.ndarray[double, ndim=1, mode="c"] resid_error = np.zeros(check_components)
  cdef double *resid_error_ptr = &resid_error[0]
  cdef np.ndarray[double, ndim=2, mode="fortran"] components
  _, components = pca_scores(spikes, check_components, pcs_as_index=False, return_V=True)
  cdef double *components_ptr = &components[0, 0]
  cdef double *reconstruction_ptr

  cdef double RESS = 0.0
  cdef int64_t idx_sp, idx_rec
  for comp in range(0, check_comp_ssize_t):
    # Compute (spikes @ components[:, comp][:, None]). Here we can store the
    # result of this first multiplication in reconstruction
    reconstruction_ptr = <double *> calloc(spikes_x * spikes_y, sizeof(double))
    spikes_ptr = &spikes[0,0]
    idx_sp = 0
    for x in range(0, spikes_x):
      idx_rec = x * spikes_y + comp
      for y in range(0, spikes_y):
        # This more complicated indexing into F ordered spikes array still seems
        # faster assuming that x > y (as is usually the case)
        reconstruction_ptr[idx_rec] += spikes_ptr[spikes_x * y + x] * components_ptr[spikes_y * comp + y]
        idx_sp += 1

    # reconstruction_ptr = &reconstruction[0, 0]
    spikes_ptr = &spikes[0,0]
    idx_sp = 0
    # Now compute result above (reconstruction[x, comp]) @ components[:, comp][:, None].T
    for x in range(0, spikes_x):
      curr_answer = reconstruction_ptr[x * spikes_y + comp]
      for y in range(0, spikes_y):
        reconstruction_ptr[idx_sp] = curr_answer * components_ptr[spikes_y * comp + y]
        RESS += (reconstruction_ptr[idx_sp] - spikes_ptr[spikes_x * y + x]) ** 2
        idx_sp += 1

    # Compute mean residual error over all points
    RESS /= spikes_x * spikes_y
    resid_error_ptr[comp] = RESS
    free(reconstruction_ptr)
    RESS = 0.0

  # Optimal order of components based on reconstruction accuracy
  comp_order = np.argsort(resid_error)

  # Find improvement given by addition of each ordered PC
  cdef np.ndarray[double, ndim=1, mode="c"] vaf = np.zeros(check_comp_ssize_t)
  cdef double *vaf_ptr = &vaf[0]
  # Start with RESS as total error, i.e. no improvement
  for x in range(0, spikes_x):
    for y in range(0, spikes_y):
      RESS += spikes[x, y] ** 2
  RESS /= spikes_x * spikes_y
  if RESS < 1.0e-14:
    if min_components == 0:
      return np.array([])
    else:
      return np.arange(0, min_components)

  cdef double PRESS = resid_error[comp_order[0]]
  cdef Py_ssize_t max_vaf_components
  for comp in range(1, check_comp_ssize_t):
    reconstruction = (spikes @ components[:, comp_order[0:comp]]) @ components[:, comp_order[0:comp]].T
    RESS = np.mean(np.mean((reconstruction - spikes) ** 2, axis=1), axis=0)
    vaf_ptr[comp] = 1 - RESS / PRESS
    PRESS = RESS

    # Choose first local maxima as point at which there is decrease in vaf
    if (vaf[comp] > vaf[comp - 1]) and (comp > 2):
      max_vaf_components = comp # Used as slice so this includes peak
      break
    if comp == max_components:
      # Won't use more than this so break
      max_vaf_components = comp
      break

  # This is to account for slice indexing and edge effects
  if comp >= check_comp_ssize_t - 1:
    # This implies that we found no maxima before reaching the end of vaf
    if vaf_ptr[check_comp_ssize_t] > vaf_ptr[check_comp_ssize_t - 1]:
      # vaf still increasing so choose last point
      max_vaf_components = check_comp_ssize_t
    else:
      # vaf has become flat so choose second to last point
      max_vaf_components = check_comp_ssize_t - 1
  if max_vaf_components > max_components:
    max_vaf_components = max_components
  if max_vaf_components < min_components:
    max_vaf_components = min_components
  return comp_order[0:max_vaf_components]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)    # turn division by zero checking off
def compute_cluster_centroid(double[:, ::1] scores not None,
      int64_t[::1] labels not None, int64_t label):
  """np.ndarray[np.double_t, ndim=1]
  Compute the mean of scores corresponding to a given label along axis=0.

  Returns a vector of the selected mean scores. If label is not found in the
  vector labels, centroid is return as a numpy array of np.nan.

  Compute the centroids of the a given cluster given the scores and labels
  for each spike. The centroid is just given by the mean of all of the
  selected points corresponding to the input label in the vector labels.

  Parameters
  ----------
  scores : numpy ndarray float64
      Each row of data will be treated as an observation and each column as a
      dimension over which distance will be computed.  Must be two
      dimensional.
  labels : numpy ndarray int64_t
      Must be a one dimensional vector such that labels.size =
      scores.shape[0].  Each element of label indicates the cluster group to
      which the corresponding row of scores belongs.
  label : int64_t
      Indicates the cluster label to average over.  All labels == label will
      be used to compute the average for the output.

  Returns
  -------
  centroid : numpy ndarray
      A new array the size of scores.shape[1] indicating the mean value
      over all scores belonging to the cluster indicated by label.
  """
  cdef Py_ssize_t x, y
  cdef int64_t n_points = 0
  cdef np.ndarray[double, ndim=1, mode="c"] centroid = np.zeros(scores.shape[1])
  cdef double *centroid_ptr = &centroid[0]

  for x in range(0, labels.shape[0]):
    if labels[x] == label:
      n_points += 1
      for y in range(0, scores.shape[1]):
        centroid_ptr[y] += scores[x, y]
  if n_points > 0:
    for y in range(0, centroid.shape[0]):
      centroid_ptr[y] /= n_points
  else:
    # No centroid matching requested label so returns nans
    for y in range(0, centroid.shape[0]):
      centroid_ptr[y] = NAN
  return np.asarray(centroid)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)    # turn division by zero checking off
def identify_clusters_to_compare(double[:, ::1] scores, int64_t[::1] labels, list previously_compared_pairs):
  """ This helper function identifies clusters to compare. It does this by
  determining mutually close clusters (as defined by the L2 norm). That is,
  it finds the first two clusters that are the closest to each other.
  If such a pair is not found, it returns a tuple containing (0, 0)
  """
  cdef Py_ssize_t i, x, y
  cdef int64_t n_points
  cdef long c1, c2
  cdef double dist_min

  # Get memory views of all important arrays
  # cdef double[:, ::1] scores_view = scores
  # cdef int64_t[::1] labels_view = labels
  unique_labels = np.unique(labels)
  cdef int64_t[::1] unique_labels_view = unique_labels
  centroids = np.zeros((unique_labels_view.shape[0], scores.shape[1]), dtype=np.double)
  cdef double[:, ::1] centroids_view = centroids

  # Get cluster centroid for each label
  for i in range (0, centroids_view.shape[0]):
    n_points = 0
    for x in range(0, labels.shape[0]):
      if labels[x] == unique_labels_view[i]:
        n_points += 1
        for y in range(0, centroids_view.shape[1]):
          centroids_view[i, y] += scores[x, y]
    for y in range(0, centroids_view.shape[1]):
      centroids_view[i, y] /= n_points

  # Compute the squareds distance between each cluster and all other clusters
  # MUST initialize min_distance_cluster to zeros to account for previous pairs
  min_distance_cluster = np.zeros((centroids_view.shape[0], ), dtype=np.intp)
  cdef Py_ssize_t[::1] min_distance_cluster_view = min_distance_cluster
  this_distance = np.zeros((centroids_view.shape[0], ), dtype=np.double)
  cdef double[::1] this_distance_view = this_distance
  for i in range (0, centroids_view.shape[0]):
    # reset distances to zero
    for x in range(0, this_distance_view.shape[0]):
      this_distance_view[x] = 0.0
    for x in range(0, centroids_view.shape[0]):
      for y in range(0, centroids_view.shape[1]):
        this_distance_view[x] += (centroids_view[x, y] - centroids_view[i, y]) ** 2
    this_distance_view[i] = INFINITY

    # Remove previously compared pairs
    for x in range(0, len(previously_compared_pairs)):
      c1 = previously_compared_pairs[x][0]
      c2 = previously_compared_pairs[x][1]
      if c1 == unique_labels_view[i]:
        for y in range(0, unique_labels_view.shape[0]):
          if unique_labels_view[y] == c2:
            this_distance_view[y] = INFINITY
            break
      elif c2 == unique_labels_view[i]:
        for y in range(0, unique_labels_view.shape[0]):
          if unique_labels_view[y] == c1:
            this_distance_view[y] = INFINITY
            break

    # Find np.argmin(this_distance)
    dist_min = INFINITY
    for x in range(0, this_distance_view.shape[0]):
      if this_distance_view[x] < dist_min:
        dist_min = this_distance_view[x]
        min_distance_cluster_view[i] = x
    # if isfinite(dist_min) == 0:
    #   min_distance_cluster_view[i] = 0

  # Now that we have the distances, we want to find the first pairs that
  # are mutally each other's minimum distance
  cdef list minimum_distance_pairs = []
  for x in range (0, unique_labels_view.shape[0]):
    for y in range (x+1, unique_labels_view.shape[0]): # NOTE: Assumes unique labels are sorted
      if (min_distance_cluster_view[x] == y) and (min_distance_cluster_view[y] == x):
        minimum_distance_pairs.append([unique_labels_view[x], unique_labels_view[y]])
  return minimum_distance_pairs


# def idct1d(data):
#   # computes the inverse discrete cosine transform
#   # Compute weights
#   weights = data.size * np.exp(1j * np.arange(0, data.size) * np.pi / (2 * data.size))
#   # Compute x tilde using equation (5.93) in Jain
#   data = np.real(ifft(weights * data))
#   # Re-order elements of each column according to equations (5.93) and
#   # (5.94) in Jain
#   out = np.zeros(data.size)
#   out_midslice = int64_t(data.size / 2)
#   out[0::2] = data[0:out_midslice]
#   out[1::2] = data[-1:out_midslice-1:-1]
#   #   Reference:
#   #      A. K. Jain, "Fundamentals of Digital Image
#   #      Processing", pp. 150-153.
#
#   return out


cdef int64_t sign_fun(double x):
  # Returns 1 if x > 0 , -1 if x < 0, and 0 if x == 0
  return <int64_t>(x > 0) - (x < 0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)    # turn division by zero checking off
cdef double fixed_point(double t, int64_t N, int64_t[::1] I, double[::1] a2):
  # this implements the function t-zeta*gamma^[l](t)
  cdef int64_t s, K0_int, s2
  cdef int64_t l = 7
  cdef double f_fac, f, K0, const, time
  cdef Py_ssize_t x
  cdef Py_ssize_t I_size = I.shape[0]

  f_fac = 0.0
  for x in range(I_size):
    f_fac += I[x] ** l * a2[x] * exp(-1.0*I[x] * M_PI * M_PI * t)
  if f_fac < 1.0e-6 or N == 0:
    # Prevent zero division, which converges to negative infinity
    return -INFINITY
  f = 2.0 * M_PI ** (2.0*l) * f_fac

  for s in range(l - 1, 1, -1):
    K0_int = 1
    if s > 1:
      for s2 in range(3, 2*s, 2):
        K0_int *= s2
    K0 = K0_int / (M_SQRT2 * sqrt(M_PI))
    const = (1.0 + (0.5) ** (s + 0.5)) / 3.0
    time = (2.0 * const * K0 / N / f) ** (2.0 / (3.0 + 2.0*s))
    f_fac = 0.0
    for x in range(I_size):
      f_fac += I[x] ** s * a2[x] * exp(-1.0*I[x] * M_PI * M_PI * time)
    if f_fac < 1.0e-6:
      # Prevent zero division, which converges to negative infinity
      f = -1.0
      break
    f = 2.0 * M_PI ** (2.0*s) * f_fac

  if f > 0.0:
    return t - (2.0 * N * sqrt(M_PI) * f) ** (-2.0/5.0)
  else:
    return -INFINITY


cdef double fixed_point_abs(double t, int64_t N, int64_t[::1] I, double[::1] a2):
  """ I added this for the case where no root is found and we seek the
  minimum absolute value in the main optimization while loop below.  It
  returns the absolute value of "fixed_point".
  """
  # this implements the function t-zeta*gamma^[l](t)
  cdef double f_t

  f_t = fixed_point(t, N, I, a2)
  # Get absolute value
  if f_t >= 0.0:
    return f_t
  else:
    return -1.0 * f_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)    # turn division by zero checking off
cdef double bound_grad_desc_fixed_point_abs(int64_t N, int64_t[::1] I, double[::1] a2,
      double lower, double upper, double xtol, double ytol):
  """ I added this for the case where no root is found and we seek the
  minimum absolute value in the main optimization while loop below.  It
  is identical to 'fixed_point' above but returns the absolute value.
  """
  cdef double f_t, f_dt_pl, f_dt_mn, d_f_t_dt, next_t, f_step, t_step
  cdef double alpha = 1.0e-4 # Choose learning rate
  cdef int64_t max_iter = 1000
  cdef int64_t n_iters = 0
  cdef double t_star = lower
  cdef double f_min = INFINITY
  cdef double t = lower
  cdef double dt = (upper - lower) / 10

  if dt < xtol:
    dt = xtol
  # Choose starting point as lowest over 10 intervals
  while t <= upper:
    f_t = fixed_point_abs(t, N, I, a2)
    if f_t < f_min:
      t_star = t
      f_min = f_t
    t += dt
  if isfinite(f_min) == 0:
    return 0.0

  # reset t and f_t to lowest point to start search
  t = t_star
  f_t = f_min
  while 1:
    # Compute derivative at t using +/- dt
    f_dt_pl = fixed_point_abs(t + dt, N, I, a2)
    f_dt_mn = fixed_point_abs(t - dt, N, I, a2)
    d_f_t_dt = (f_dt_pl - f_dt_mn) / (2*dt)
    if isfinite(d_f_t_dt) == 0:
      # This shouldn't happen, but if derivative is infinite return preceeding t
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
    # print("NEXT T", next_t)
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
    # Reassign and remember stuff for next iteration
    t = next_t
    dt = t_step
    f_min = f_t
    n_iters += 1
    if n_iters > max_iter:
      t_star = t
      break

  return t_star


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)    # turn division by zero checking off
def kde(double[::1] data, int64_t n_points):
  """
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
  data : numpy ndarray double
      a vector of data from which the density estimate is constructed
  n : int64_t
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
  density, xmesh, bandwidth = sort_cython.kde(data, 2**14)
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
  cdef double R, dx, tol, fixed_point_0, f_0, tol_0, f_tol, t_star
  cdef int64_t N, n
  cdef Py_ssize_t x
  cdef Py_ssize_t data_size = data.shape[0]
  cdef double MIN = INFINITY, MAX = -INFINITY

  n = <int64_t>1 << <int64_t>(ceil(log2(n_points))) # round up n to the next power of 2
  # define the interval [MIN, MAX]
  for x in range(0, data_size):
    if data[x] > MAX:
      MAX = data[x]
    if data[x] < MIN:
      MIN = data[x]

  if MIN == MAX:
    early_density = np.ones((1, ), dtype=np.double)
    xmesh = np.ones((1, ), dtype=np.double)
    xmesh[0] = MAX
    bandwidth = 0.0
    return early_density, xmesh, bandwidth

  # set up the grid over which the density estimate is computed
  R = MAX - MIN
  dx = R / (n - 1.0)
  cdef double val = MIN
  cdef int64_t xmesh_size = <int64_t>((R+dx)/dx)
  xmesh = np.empty((xmesh_size, ), dtype=np.double)
  cdef double[::1] xmesh_view = xmesh
  for x in range(0, xmesh_size):
    xmesh_view[x] = val
    val += dx

  # bin the data uniformly using the grid defined above
  hist_bins = np.empty((xmesh_size + 1, ), dtype=np.double)
  cdef double[::1] hist_bins_view = hist_bins
  for x in range(0, xmesh_size):
    hist_bins_view[x] = xmesh_view[x]
  """ ADD np.inf as the final bin edge to get MatLab histc like behavior """
  hist_bins_view[x+1] = INFINITY
  N = np.unique(data).shape[0]
  initial_data = np.histogram(data, bins=hist_bins)[0] / N
  initial_data = initial_data / np.sum(initial_data)
  a = dct(initial_data) # discrete cosine transform of initial data
  cdef double[::1] a_view = a
  """ The original implementation below returns 1 for first element, so
  # enforce that here """
  a_view[0] = 1.0

  # now compute the optimal bandwidth^2 using the referenced method
  I = np.empty((n - 1, ), dtype=np.int64)
  cdef int64_t[::1] I_view = I
  cdef int64_t I_i = 1
  for x in range(0, n-1):
    I_view[x] = I_i ** 2
    I_i += 1
  a2 = np.empty((a_view.shape[0] - 1, ), dtype=np.double)
  cdef double[::1] a2_view = a2
  for x in range(0, a_view.shape[0]-1):
    a2_view[x] = (a_view[x+1] / 2) ** 2

  cdef int64_t N_tol = 50 * (N <= 50) + 1050 * (N >= 1050) + N * ((N < 1050) * (N > 50))
  tol = 10.0 ** -12.0 + 0.01 * (N_tol - 50.0) / 1000.0
  fixed_point_0 = fixed_point(0.0, N, I, a2)
  f_0 = fixed_point_0
  tol_0 = 0.0
  """ This is the main optimization loop to solve for t_star """
  while 1:
    f_tol = fixed_point(tol, N, I, a2)
    """ Attempt to find a zero crossing in the fixed_point function moving
    stepwise from fixed_point_0 """
    if sign_fun(f_0) != sign_fun(f_tol):
      # use fzero to solve the equation t=zeta*gamma^[5](t)
      """ I am using my own gradient descent with bounds made for this purpose
          rather than MatLab 'fzero' """
      t_star = bound_grad_desc_fixed_point_abs(N, I, a2, tol_0, tol, 1.0e-6, 1.0e-6)
      break
    else:
      tol_0 = tol
      tol = min(tol * 2.0, 1.0) # double search interval
      f_0 = f_tol
    if tol == 1.0: # if all else fails
      """ Failed to find zero crossing so find absolute minimum value using
          the same bounded gradient descent algorithm as above"""
      t_star = bound_grad_desc_fixed_point_abs(N, I, a2, 0, 1.0, 1.0e-6, 1.0e-6)
      break

  # smooth the discrete cosine transform of initial data using t_star
  a_t = np.empty_like(a, dtype=np.double)
  cdef double[::1] a_t_view = a_t
  for x in range(0, a_t_view.shape[0]):
    a_t_view[x] = a_view[x] * exp(-1.0 * x ** 2 * M_PI ** 2 * t_star / 2)

  # Computes the inverse discrete cosine transform as formerly done in "idct1d"
  #   Reference:
  #      A. K. Jain, "Fundamentals of Digital Image
  #      Processing", pp. 150-153.
  # Compute weights
  weights = a_t.size * np.exp(1j * np.arange(0, a_t.size) * np.pi / (2 * a_t.size))
  # Compute x tilde using equation (5.93) in Jain
  a_t = np.real(ifft(weights * a_t))
  # Re-order elements of each column according to equations (5.93) and
  # (5.94) in Jain
  cdef np.ndarray[double, ndim=1, mode="c"] density = np.zeros(a_t.size)
  cdef double[::1] density_view = density
  cdef int64_t density_midslice
  density_midslice = <int64_t>(floor(a_t_view.shape[0] / 2))
  density[0::2] = a_t[0:density_midslice]
  density[1::2] = a_t[a_t_view.shape[0]:density_midslice-1:-1]
  density /= R

  # take the rescaling of the data into account
  bandwidth = sqrt(t_star) * R
  """ I set this to zero instead of a small epsilon value. """
  density[density < 0] = 0; # remove negatives due to round-off error

  return density, xmesh, bandwidth
