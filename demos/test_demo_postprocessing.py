import sys
import numpy as np
import pickle
import spikesorting_fullpursuit as fbp



"""
 ex. python test_demo_postprocessing.py ./mysavedir/sorted_demo.pickle ./mysavedir/sorted_neurons.pickle

 """

if __name__ == '__main__':
    """
    Sets up a probe and filters the voltage for the numpy voltage file input.
    Then spike sorting is run and the output is saved.
    """
    if len(sys.argv) < 3:
        raise ValueError("Requires 2 inputs. The sorter output file and save destination for postprocessed neurons.")
    fname_sorted_data = sys.argv[1]
    fname_out_data = sys.argv[2]
    if not '.pickle' in save_fname[-7:]:
        save_fname = save_fname + '.pickle'

    print("Loading sorter data from file: ", fname_sorted_data)
    print("Output neurons will be saved as: ", fname_out_data)

    with open(fname_sorted_data, 'rb') as fp:
        sorted_data = pickle.load(fp)

    # Unpack sorter data output
    sort_data, work_items, sort_info = sorted_data[0], sorted_data[1], sorted_data[2]

    max_mua_ratio = 1.2
    min_snr = 0
    absolute_refractory_period = 10e-4
    min_overlapping_spikes = .75
    work_summary = fbp.postprocessing.WorkItemSummary(sort_data, work_items, sort_info,
                                               absolute_refractory_period=absolute_refractory_period, max_mua_ratio=max_mua_ratio,
                                               min_snr=min_snr, min_overlapping_spikes=min_overlapping_spikes,
                                               verbose=False)
