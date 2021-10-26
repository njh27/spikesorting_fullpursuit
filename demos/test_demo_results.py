import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import spikesorting_fullpursuit as fbp



"""
 ex. python test_demo_results.py ./mysavedir/sorted_neurons.pickle

 """

if __name__ == '__main__':
    """
    Sets up a probe and filters the voltage for the numpy voltage file input.
    Then spike sorting is run and the output is saved.
    """
    if len(sys.argv) < 2:
        raise ValueError("Requires 1 inputs. The file of postprocessed neurons.")
    fname_sorted_data = sys.argv[1]

    print("Loading neurons from file: ", fname_sorted_data)

    with open(fname_sorted_data, 'rb') as fp:
        neurons = pickle.load(fp)

    # Print out some basic information about our sorted units like number of spikes, firing rate, SNR, proportion MUA ISI violations
    print("Found", len(neurons), "total units with properties:")
    fmtL = "Unit: {:.0f} on chans {}; n spikes = {:.0f}; FR = {:.0f}; Dur = {:.0f}; SNR = {:.2f}; MUA = {:.2f}; TolInds = {:.0f}"
    for ind, n in enumerate(neurons):
        print_vals = [ind, n['channel'], n['spike_indices'].size, n['firing_rate'], n['duration_s'], n['snr']['average'], n['fraction_mua'], n['duplicate_tol_inds']]
        print(fmtL.format(*print_vals))

    # Match the ground truth units to the sorted neurons with the most true positives
    test_match_to_neurons = {}
    for test_num in range(0, len(test_data.actual_IDs)):
        max_true_positives = -np.inf
        for unit_num in range(0, len(neurons)):
            overlapping_spike_bool = fbp.analyze_spike_timing.find_overlapping_spike_bool(test_data.actual_IDs[test_num],
                                                                                          neurons[unit_num]['spike_indices'], overlap_tol=10)
            true_positives = np.count_nonzero(overlapping_spike_bool)
            if true_positives > max_true_positives:
                max_true_positives = true_positives
                test_match_to_neurons[test_num] = unit_num

    for test_num in range(0, len(test_data.actual_IDs)):
        print("Matched actual unit", test_num, "to sorted neuron", test_match_to_neurons[test_num])
