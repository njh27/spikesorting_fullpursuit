import sys
import numpy as np
import pickle
from spikesorting_fullpursuit import electrode
from spikesorting_fullpursuit.parallel.spikesorting_parallel import spike_sort_parallel

"""
 ex. python test_demo_run_sort.py ./mysavedir/test_voltage.npy ./mysavedir/sorted_demo.pickle

NOTE:
 If running this script causes your computer to hang or crash, you might try testing
 first with 'do_overlap_recheck' set to "False". The overlap recheck can be time
 consuming and may cause your GPU to either crash or timeout. It is likely that
 the watchdog timer for your operating system or graphics card will need to be
 increased in order to successfully run. Alternatively, you could run using a
 smaller 'max_gpu_memory' or run with smaller segment durations, both of which
 will sort less data in each GPU call and therefore might run faster without
 causing a timeout. Be sure that the program is discovering and using the
 desired GPU.
 """

if __name__ == '__main__':
    """
    Sets up a probe and filters the voltage for the numpy voltage file input.
    Then spike sorting is run and the output is saved.
    """
    if len(sys.argv) < 3:
        raise ValueError("Requires 2 inputs. Voltage file and save destination.")
    volt_fname = sys.argv[1]
    save_fname = sys.argv[2]
    if not '.pickle' in save_fname[-7:]:
        save_fname = save_fname + '.pickle'
    log_dir = save_fname[0:-7] + '_std_logs' # Set default log name to match save name

    print("Sorting data from file: ", volt_fname)
    print("Output will be saved as: ", save_fname)

    # Setup the sorting parameters dictionary.
    spike_sort_args = {
            'sigma': 4.0, # Threshold based on noise level
            'clip_width': [-10e-4, 10e-4], # Width of clip in seconds
            'p_value_cut_thresh': 0.01,
            'segment_duration': 300,
            'segment_overlap': 150,
            'do_branch_PCA': True,
            'do_branch_PCA_by_chan': True,
            'do_overlap_recheck': True,
            'filter_band': (300, 6000),
            'do_ZCA_transform': True,
            'check_components': 20,
            'max_components': 5,
            'min_firing_rate': 0.1,
            'use_rand_init': True,
            'add_peak_valley': False,
            'max_gpu_memory': .1 * (1024 * 1024 * 1024),
            'save_1_cpu': True,
            'sort_peak_clips_only': True,
            'n_cov_samples': 20000,
            'sigma_bp_noise': 2.326,
            'sigma_bp_CI': 12.0,
            'absolute_refractory_period': 10e-4,
            'get_adjusted_clips': False,
            'max_binary_pursuit_clip_width_factor': 1.0,
            'verbose': True,
            'test_flag': False,
            'log_dir': log_dir,
            }

    # Load the numpy voltage array
    raw_voltage = np.load(volt_fname)
    raw_voltage = np.float32(raw_voltage)

    # Create the electrode object that specifies neighbor function for our current
    # tetrode test dataset
    Probe = electrode.SingleTetrode(sampling_rate=40000, voltage_array=raw_voltage)
    # We need to filter our voltage before passing it to the spike sorter. Just
    # use the one in Probe class
    Probe.bandpass_filter_parallel(spike_sort_args['filter_band'][0], spike_sort_args['filter_band'][1])

    print("Start sorting")
    sort_data, work_items, sorter_info = spike_sort_parallel(Probe, **spike_sort_args)

    print("Saving neurons file as", save_fname)
    with open(save_fname, 'wb') as fp:
        pickle.dump((sort_data, work_items, sorter_info), fp, protocol=-1)
