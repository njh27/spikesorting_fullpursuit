import sys
import numpy as np
from scipy.signal import filtfilt, butter
import mkl
import multiprocessing as mp
import psutil
import pickle
sys.path.append('c:\\users\\plexon\\documents\\python scripts') # for PL2_read
import PL2_read
from spikesorting_fullpursuit import electrode
from spikesorting_fullpursuit.parallel.spikesorting_parallel import spike_sort_parallel



def init_load_voltage():
    mkl.set_num_threads(1)


def load_voltage_parallel(PL2Reader, read_source):
    """
    """

    include_chans = []
    for cc in range(0, len(PL2Reader.info['analog_channels'])):
        if PL2Reader.info['analog_channels'][cc]['source_name'].upper() == read_source and bool(PL2Reader.info['analog_channels'][cc]['enabled']):
            include_chans.append(cc)

    order_results = []
    with mp.Pool(processes=psutil.cpu_count(logical=True), initializer=init_load_voltage, initargs=()) as pool:
        try:
            for chan in range(0, len(include_chans)):
                order_results.append(pool.apply_async(PL2Reader.load_analog_channel, args=(include_chans[chan], False, None)))
        finally:
            pool.close()
            pool.join()
    voltage = np.vstack([x.get() for x in order_results])

    return voltage


def init_pool_dict(volt_array, volt_shape, init_dict=None):
    global pool_dict
    pool_dict = {}
    pool_dict['share_voltage'] = volt_array
    pool_dict['share_voltage_shape'] = volt_shape
    if init_dict is not None:
        for k in init_dict.keys():
            pool_dict[k] = init_dict[k]
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
    with mp.Pool(processes=psutil.cpu_count(logical=False)//2, initializer=init_pool_dict, initargs=(filt_X, Probe.voltage.shape)) as pool:
        try:
            for chan in range(0, Probe.num_channels):
                filt_results.append(pool.apply_async(filter_one_chan, args=(chan, b_filt, a_filt,  Probe.v_dtype)))
        finally:
            pool.close()
            pool.join()
    filt_voltage = np.vstack([x.get() for x in filt_results])

    return filt_voltage


if __name__ == '__main__':
    """
    """
    if len(sys.argv) < 3:
        raise ValueError("Requires 2 inputs. Voltage file and save destination.")
    fname_PL2 = sys.argv[1]
    save_fname = sys.argv[2]
    if not '.pickle' in save_fname[-7:]:
        save_fname = save_fname + '.pickle'
    log_dir = save_fname[0:-7] + '_std_logs' # Set default log name to match save name

    print("Sorting data from file: ", fname_PL2)
    print("Output will be saved as: ", save_fname)

    # python.exe fbp_sort_learndirtune_pl2.py c:\nate\LearnDirTunePurk_Yoda_49.pl2 c:\nate\sorted_yoda_49.pickle

    # Setup the sorting parameters dictionary.
    spike_sort_args = {'sigma': 4.,
                       'verbose': True,
                       'test_flag': False,
                       'log_dir': log_dir,
                       'clip_width': [-8e-4, 12e-4],
                       'do_branch_PCA': True,
                       'do_branch_PCA_by_chan': False,
                       'filter_band': (300, 6000),
                       'do_ZCA_transform': True,
                       'use_rand_init': True,
                       'add_peak_valley': False,
                       'check_components': 20,
                       'max_components': 5,
                       'min_firing_rate': 0.5,
                       'p_value_cut_thresh': 0.05,
                       'max_gpu_memory': None,
                       'save_1_cpu': True,
                       'segment_duration': 300,
                       'segment_overlap': 150,
                       'sort_peak_clips_only': True,
                       # sigma_noise_penalty = 90%: 1.645, 95%: 1.96, 99%: 2.576; NOTE: these are used one sided
                       'sigma_noise_penalty': 1.645, # Number of noise standard deviations to penalize binary pursuit by. Higher numbers reduce false positives, increase false negatives
                       'get_adjusted_clips': False,
                       'max_binary_pursuit_clip_width_factor': 2.0,
                       }

    # use_voltage_file = 'C:\\Users\\plexon\\Documents\\Python Scripts\\voltage_49_1min'
    use_voltage_file = None

    if use_voltage_file is None:
        spike_sort_args['do_ZCA_transform'] = True
    else:
        spike_sort_args['do_ZCA_transform'] = False

    pl2_reader = PL2_read.PL2Reader(fname_PL2)

    if use_voltage_file is None or not use_voltage_file:
        if not spike_sort_args['do_ZCA_transform']:
            print("!!! WARNING !!!: ZCA transform is OFF but data is being loaded from PL2 file.")
        print("Reading voltage from file")
        voltage_array = load_voltage_parallel(pl2_reader, 'SPKC')
        voltage_array = voltage_array.astype(np.float32)
        print("INITIAL RAW VOLTAGE TYPE IS", voltage_array.dtype)
        t_t_start = int(40000 * 60 * 30)
        t_t_stop =  int(40000 * 60 * 40)
        if voltage_array.shape[0] == 32:
            SProbe = electrode.SProbe16by2(pl2_reader.info['timestamp_frequency'], voltage_array=voltage_array)#[:, t_t_start:t_t_stop])
        elif voltage_array.shape[0] == 1:
            SProbe = electrode.SingleElectrode(pl2_reader.info['timestamp_frequency'], voltage_array=voltage_array)
        else:
            raise ValueError("Cannot determine which type of electrode to use")
    else:
        with open(use_voltage_file, 'rb') as fp:
            voltage_array = pickle.load(fp)
        SProbe = electrode.SProbe16by2(pl2_reader.info['timestamp_frequency'], voltage_array=voltage_array)

    voltage_array = filter_parallel(SProbe, low_cutoff=spike_sort_args['filter_band'][0], high_cutoff=spike_sort_args['filter_band'][1])
    print("FILTERED RAW VOLTAGE TYPE IS", voltage_array.dtype)
    if voltage_array.shape[0] == 32:
        SProbe = electrode.SProbe16by2(pl2_reader.info['timestamp_frequency'], voltage_array=voltage_array)
    elif voltage_array.shape[0] == 1:
        SProbe = electrode.SingleElectrode(pl2_reader.info['timestamp_frequency'], voltage_array=voltage_array)
    else:
        raise ValueError("Cannot determine which type of electrode to use")

    SProbe.filter_band = spike_sort_args['filter_band']
    print("Start sorting")
    sort_data, work_items, sorter_info = spike_sort_parallel(SProbe, **spike_sort_args)

    print("Saving neurons file as", save_fname)
    with open(save_fname, 'wb') as fp:
        pickle.dump((sort_data, work_items, sorter_info), fp, protocol=-1)
