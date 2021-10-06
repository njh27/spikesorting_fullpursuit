import sys
import numpy as np
import pickle
from spikesorting_fullpursuit import electrode
from spikesorting_fullpursuit.parallel.spikesorting_parallel import spike_sort_parallel

# python.exe fbp_sort_sf2_distance_probe.py c:\nate\spikeforest2_downloads\kampff_c26

if __name__ == '__main__':
    """
    """
    if len(sys.argv) < 2:
        raise ValueError("Requires 2 inputs. Voltage file and save destination.")
    data_folder = sys.argv[1]
    save_fname = data_folder + '/sorted.pickle'
    if not '.pickle' in save_fname[-7:]:
        save_fname = save_fname + '.pickle'
    log_dir = save_fname[0:-7] + '_std_logs' # Set default log name to match save name

    print("Sorting data from file: ", data_folder)
    print("Output will be saved as: ", save_fname)

    # Setup the sorting parameters dictionary.
    spike_sort_args = {'sigma': 4.,
                       'verbose': True,
                       'test_flag': False,
                       'log_dir': log_dir,
                       'clip_width': [-10e-4, 25e-4],
                       'do_branch_PCA': True,
                       'do_branch_PCA_by_chan': False,
                       'filter_band': (300, 6000),
                       'do_ZCA_transform': True,
                       'use_rand_init': True,
                       'add_peak_valley': False,
                       'check_components': 20,
                       'max_components': 5,
                       'min_firing_rate': 0.1,
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

    raw_voltage = np.load(data_folder + '/volt_array.npy')
    raw_voltage = np.float32(raw_voltage)
    xy_layout = np.load(data_folder + '/layout.npy')
    try:
        sampling_rate = np.load(data_folder + "/sampling_rate.npy")
    except:
        print("no sampling rate file present. Creating one at 30 kHz.")
        np.save(data_folder + "/sampling_rate.npy", np.array(30000))
        sampling_rate = 30000
    num_electrodes = raw_voltage.shape[0]

    t_start = int(30000 * 60 * 0)
    t_stop = int(30000 * 60 * 5)
    Probe = electrode.DistanceBasedProbe(sampling_rate, num_electrodes, xy_layout, voltage_array=raw_voltage)#[:, t_start:t_stop])
    Probe.bandpass_filter(spike_sort_args['filter_band'][0], spike_sort_args['filter_band'][1])

    print("Start sorting")
    sort_data, work_items, sorter_info = spike_sort_parallel(Probe, **spike_sort_args)

    print("Saving neurons file as", save_fname)
    with open(save_fname, 'wb') as fp:
        pickle.dump((sort_data, work_items, sorter_info), fp, protocol=-1)
