import sys
import numpy as np
import pickle
from spikesorting_fullpursuit.test import gen_dataset



"""
Will save a file called "test_voltage.npy" to the directory specified in the
first input argument and save the ground truth spike times in the directory
given in the second argument.
ex.
    python make_and_save_voltage.py /mydir/test_voltage.npy /mydir/test_ground_truth.pickle

    creates a numpy voltage array saved at: /mydir/test_voltage.npy
    and an array of ground truth spike times at /mydir/test_ground_truth.pickle

"""
if __name__ == '__main__':
    """
    """
    if len(sys.argv) < 3:
        raise ValueError("Requires 2 inputs specifying save destination for voltage and ground truth spike times.")
    volt_fname = sys.argv[1]
    volt_fname = volt_fname.rstrip("/")
    if not '.npy' in volt_fname[-4:]:
        volt_fname = volt_fname + '.npy'
    gt_fname = sys.argv[2]
    gt_fname = gt_fname.rstrip("/")
    if not '.pickle' in gt_fname[-7:]:
        gt_fname = gt_fname + '.pickle'

    n_chans = 4 # Number of channels to make in test dataset
    v_duration = 30 # Test dataset duration in seconds
    random_seed = None # Set seed of numpy random number generator for spike times
    neuron_templates = None # Just use default pre-loaded template waveforms to generate spike voltage traces
    frequency_range = (300, 6000) # Frequencies of dataset in Hz
    samples_per_second = 40000 # Sampling rate of 40kHz
    amplitude = 1 # Amplitude of 3 standard deviations of noise
    percent_shared_noise = .3 # Create shared noise across channels
    correlate1_2 = (.10, 10) # Set 15% of neuron 2 spikes to occur within 10 samples of a neuron 1 spike
    electrode_type = 'tetrode' # Choose pre-loaded electrode type of tetrode and all channels in neighborhood
    voltage_dtype = np.float32 # Create voltage array as float 32

    # Create the test dataset object
    test_data = gen_dataset.TestDataset(n_chans, v_duration, random_seed, neuron_templates, frequency_range,
                                        samples_per_second, amplitude, percent_shared_noise,
                                        correlate1_2, electrode_type, voltage_dtype)

    # Generate the noise voltage array, without spikes, assigned to test_date.voltage_array
    test_data.gen_noise_voltage_array()

    # Specify the neurons' properties in the dataset
    firing_rates = np.array([90, 100]) # Firing rates
    template_inds = np.array([1, 0]) # Templates used for waveforms
    chan_scaling_factors = np.array([[1.85, 2.25, 1.65, .5], [3.85, 3.95, 1.95, 3.7]]) # Amplitude of neurons on each of the 4 channels
    refractory_win = 1.5e-3 # Set refractory period at 1.5 ms
    # Generate the test dataset by choosing spike times and adding them according to the specified properties
    test_data.gen_test_dataset(firing_rates, template_inds, chan_scaling_factors, refractory_win)

    print("Saving test voltage array as: ", volt_fname)
    np.save(volt_fname, test_data.Probe.voltage)

    print("Saving ground truth file as", gt_fname)
    with open(gt_fname, 'wb') as fp:
        pickle.dump(test_data.actual_IDs, fp, protocol=-1)
