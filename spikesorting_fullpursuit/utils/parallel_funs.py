import numpy as np
from scipy.signal import filtfilt, butter
import multiprocessing as mp
import psutil
import mkl
from spikesorting_fullpursuit.utils.robust_covariance import MinCovDet



### FUNCTIONS FOR PARALLEL FILTERING ###
def init_filt_pool_dict(volt_array, volt_shape):
    global pool_dict
    pool_dict = {}
    pool_dict['share_voltage'] = volt_array
    pool_dict['share_voltage_shape'] = volt_shape
    return


def filter_one_chan(chan, b_filt, a_filt, voltage_type):

    mkl.set_num_threads(2)
    voltage = np.frombuffer(pool_dict['share_voltage'], dtype=voltage_type).reshape(pool_dict['share_voltage_shape'])
    filt_voltage = filtfilt(b_filt, a_filt, voltage[chan, :], padlen=None).astype(voltage_type)
    return filt_voltage


def filter_parallel(Probe, low_cutoff=300, high_cutoff=6000):

    print("Allocating filter array and copying voltage")
    filt_X = mp.RawArray(np.ctypeslib.as_ctypes_type(Probe.v_dtype), Probe.voltage.size)
    filt_X_np = np.frombuffer(filt_X, dtype=Probe.v_dtype).reshape(Probe.voltage.shape)
    np.copyto(filt_X_np, Probe.voltage)
    low_cutoff = low_cutoff / (Probe.sampling_rate / 2)
    high_cutoff = high_cutoff / (Probe.sampling_rate / 2)
    b_filt, a_filt = butter(1, [low_cutoff, high_cutoff], btype='band')
    print("Performing voltage filtering")
    filt_results = []
    # Number of processes was giving me out of memory error on Windows 10 until
    # I dropped it to half number of cores.
    with mp.Pool(processes=psutil.cpu_count(logical=False)//2, initializer=init_filt_pool_dict, initargs=(filt_X, Probe.voltage.shape)) as pool:
        try:
            for chan in range(0, Probe.num_channels):
                filt_results.append(pool.apply_async(filter_one_chan, args=(chan, b_filt, a_filt,  Probe.v_dtype)))
        finally:
            pool.close()
            pool.join()
    filt_voltage = np.vstack([x.get() for x in filt_results])

    return filt_voltage


### FUNCTIONS FOR PARALLEL NOISE COVARIANCE ###
def get_singlechannel_clips(voltage, channel, spike_times, window):

    if spike_times.ndim > 1:
        raise ValueError("Event_indices must be one dimensional array of indices")

    # Ignore spikes whose clips extend beyond the data and create mask for removing them
    valid_event_indices = np.ones(spike_times.shape[0], dtype="bool")
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

    spike_clips = np.empty((np.count_nonzero(valid_event_indices), window[1] - window[0]))
    for out_ind, spk in enumerate(range(start_ind, stop_ind+1)): # Add 1 to index through last valid index
        spike_clips[out_ind, :] = voltage[channel, spike_times[spk]+window[0]:spike_times[spk]+window[1]]

    return spike_clips, valid_event_indices


def init_covar_pool_dict(volt_array, volt_shape, volt_dtype, window,
                         n_samples, rand_state):
    global pool_dict
    pool_dict = {}
    pool_dict['share_voltage'] = volt_array
    pool_dict['share_voltage_shape'] = volt_shape
    pool_dict['share_voltage_dtype'] = volt_dtype
    pool_dict['window'] = window
    pool_dict['n_samples'] = n_samples
    pool_dict['rand_state'] = rand_state
    return


def covar_one_chan(chan):

    if pool_dict['rand_state'] is not None:
        np.random.set_state(pool_dict['rand_state'])
    # Limit threads within each process
    mkl.set_num_threads(1)
    voltage = np.frombuffer(pool_dict['share_voltage'],
                dtype=pool_dict['share_voltage_dtype']).reshape(pool_dict['share_voltage_shape'])

    rand_inds = np.random.randint(-1*pool_dict['window'][0],
                            voltage.shape[1] - (pool_dict['window'][1] + 1),
                            pool_dict['n_samples'])
    noise_clips, _ = get_singlechannel_clips(voltage, chan, rand_inds, pool_dict['window'])
    # Get robust covariance to avoid outliers
    # Random state doesn't matter with support_fraction=1.
    rob_cov = MinCovDet(store_precision=False, assume_centered=True,
                         support_fraction=1., random_state=None)
    rob_cov.fit(noise_clips)

    return rob_cov.covariance_


def noise_covariance_parallel(voltage_array, window, n_samples=100000,
                              rand_state=None):
    """ Computes the covariance of the noise over time within the clip_width.
    """
    if window[0] > 0:
        # First element of window must be <= 0
        window[0] *= -1
    if voltage_array.ndim == 1:
        voltage_array = voltage_array.reshape(1, -1)
    # Allocate shared array across processes for voltage
    shared_v_array = mp.RawArray(np.ctypeslib.as_ctypes_type(voltage_array.dtype), voltage_array.size)
    # Create a numpy view of shared array and copy data
    shared_v_array_np = np.frombuffer(shared_v_array, dtype=voltage_array.dtype).reshape(voltage_array.shape)
    np.copyto(shared_v_array_np, voltage_array)

    chan_covar_mats = []
    with mp.Pool(processes=psutil.cpu_count(logical=False),
                 initializer=init_covar_pool_dict, initargs=(shared_v_array,
                 voltage_array.shape, voltage_array.dtype, window,
                 n_samples, rand_state)) as pool:
        try:
            for chan in range(0, voltage_array.shape[0]):
                chan_covar_mats.append(pool.apply_async(covar_one_chan, args=(chan, )))
        finally:
            pool.close()
            pool.join()
    # Return the outputs in channel order
    chan_covar_mats = [x.get() for x in chan_covar_mats]

    return chan_covar_mats
