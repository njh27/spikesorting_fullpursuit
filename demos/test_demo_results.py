import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import spikesorting_fullpursuit as fbp



"""
 ex. python test_demo_results.py ./mysavedir/sorted_neurons.pickle ./test_ground_truth.pickle

 """

if __name__ == '__main__':
    """
    Sets up a probe and filters the voltage for the numpy voltage file input.
    Then spike sorting is run and the output is saved.
    """
    if len(sys.argv) < 3:
        raise ValueError("Requires 2 inputs. The file of postprocessed neurons and the ground truth for comparison.")
    fname_sorted_data = sys.argv[1]
    fname_ground_truth = sys.argv[2]

    print("Loading neurons from file: ", fname_sorted_data)
    with open(fname_sorted_data, 'rb') as fp:
        neurons = pickle.load(fp)

    print("Loading ground truth from file: ", fname_ground_truth)
    with open(fname_ground_truth, 'rb') as fp:
        ground_truth = pickle.load(fp)

    # Print out some basic information about our sorted units like number of spikes, firing rate, SNR, proportion MUA ISI violations
    print("Found", len(neurons), "total units with properties:")
    fmtL = "Unit: {:.0f} on chans {}; n spikes = {:.0f}; FR = {:.0f}; Dur = {:.0f}; SNR = {:.2f}; MUA = {:.2f}; TolInds = {:.0f}"
    for ind, n in enumerate(neurons):
        print_vals = [ind, n['channel'], n['spike_indices'].size, n['firing_rate'], n['duration_s'], n['snr']['average'], n['fraction_mua'], n['duplicate_tol_inds']]
        print(fmtL.format(*print_vals))

    # Match the ground truth units to the sorted neurons with the most true positives
    test_match_to_neurons = {}
    for test_num in range(0, len(ground_truth)):
        max_true_positives = -np.inf
        for unit_num in range(0, len(neurons)):
            overlapping_spike_bool = fbp.analyze_spike_timing.find_overlapping_spike_bool(ground_truth[test_num],
                                                                                          neurons[unit_num]['spike_indices'], overlap_tol=10)
            true_positives = np.count_nonzero(overlapping_spike_bool)
            if true_positives > max_true_positives:
                max_true_positives = true_positives
                test_match_to_neurons[test_num] = unit_num

    for test_num in range(0, len(ground_truth)):
        print("Matched actual unit", test_num, "to sorted neuron", test_match_to_neurons[test_num])

    # Plot the templates of the sorted neurons
    print("Figure 1 shows templates of the discovered units")
    fig, axes = plt.subplots(1)
    for n in test_match_to_neurons:
        n_num = test_match_to_neurons[n]
        for chan in neurons[n_num]['channel']:
            axes.plot(neurons[n_num]['template'][chan][0:], label="Unit " + str(n))
    fig.suptitle('Sorted unit templates', fontsize=16)
    leg = axes.legend()

    # Print true positive and false discoveries for best matching to ground truth neuron 1
    ground_truth_unit = 0
    tol_inds = 10 # Match within a tolerance of 10 time samples
    overlapping_spike_bool = fbp.analyze_spike_timing.find_overlapping_spike_bool(ground_truth[ground_truth_unit], neurons[test_match_to_neurons[ground_truth_unit]]['spike_indices'], overlap_tol=tol_inds)
    true_positives = np.count_nonzero(overlapping_spike_bool)

    print("False discoveries are", 100 * (neurons[test_match_to_neurons[ground_truth_unit]]['spike_indices'].size - true_positives) / neurons[test_match_to_neurons[ground_truth_unit]]['spike_indices'].size)
    print("True positives are", 100 * true_positives / ground_truth[ground_truth_unit].size)

    # Print true positive and false discoveries for best matching to ground truth neuron 2
    ground_truth_unit = 1
    tol_inds = 10 # Match within a tolerance of 10 time samples
    overlapping_spike_bool = fbp.analyze_spike_timing.find_overlapping_spike_bool(ground_truth[ground_truth_unit], neurons[test_match_to_neurons[ground_truth_unit]]['spike_indices'], overlap_tol=tol_inds)
    true_positives = np.count_nonzero(overlapping_spike_bool)

    print("False discoveries are", 100 * (neurons[test_match_to_neurons[ground_truth_unit]]['spike_indices'].size - true_positives) / neurons[test_match_to_neurons[ground_truth_unit]]['spike_indices'].size)
    print("True positives are", 100 * true_positives / ground_truth[ground_truth_unit].size)

    # Plot the CCG between the sorted units and the ground truth units for comparison
    print("Figure 2 shows the CCG of the sorted units (black) and the ground truth (red)")
    fig, axes = plt.subplots(1)
    counts, time_axis = fbp.analyze_spike_timing.zero_symmetric_ccg(neurons[test_match_to_neurons[0]]['spike_indices'],
                                                                    neurons[test_match_to_neurons[1]]['spike_indices'], 20*40, 40)
    axes.bar(time_axis, counts, width=1, color=[.5, .5, .5])
    axes.plot(time_axis, counts, color='k', label="Sorted units")

    # CCG for actual data for comparison
    counts, time_axis = fbp.analyze_spike_timing.zero_symmetric_ccg(ground_truth[0], ground_truth[1], 20*40, 40)
    axes.plot(time_axis, counts, color='r', label="Ground truth")

    axes.axvline(0, color=[.75, .75, .75])
    axes.axvline(10, color=[.75, .75, .75])
    axes.axvline(-10, color=[.75, .75, .75])
    axes.axvline(5, color=[.75, .75, .75])
    axes.axvline(-5, color=[.75, .75, .75])

    fig.suptitle('CCG of sorted units and ground truth', fontsize=16)
    leg = axes.legend()
    plt.show()
