import sys
import numpy as np
import pickle
import spikesorting_fullpursuit as fbp
from spikesorting_fullpursuit.postprocessing import WorkItemSummary



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
    if not '.pickle' in fname_out_data[-7:]:
        fname_out_data = fname_out_data + '.pickle'

    print("Loading sorter data from file: ", fname_sorted_data)
    print("Output neurons will be saved as: ", fname_out_data)

    with open(fname_sorted_data, 'rb') as fp:
        sorted_data = pickle.load(fp)

    # Unpack sorter data output
    sort_data, work_items, sort_info = sorted_data[0], sorted_data[1], sorted_data[2]

    # First step in automated post-processing
    # Set a few variables that can allow easy detection of units that are poor
    absolute_refractory_period = 10e-4 # Refractory period (in ms) will be used to determine potential violations in sorting accuracy
    # Max allowable ratio between refractory period violations and maximal bin of ACG. Units that violate will be deleted. Setting to >= 1. allows all units
    max_mua_ratio = 1.
    min_snr = 0 # Minimal SNR a unit must have to be included in post-processing
    min_overlapping_spikes = .75 # Percentage of spikes required with nearly identical spike times in adjacent segments for them to combine in stitching

    # Create the work_summary postprocessing object
    work_summary = WorkItemSummary(sort_data, work_items,
                       sort_info, absolute_refractory_period=absolute_refractory_period,
                       max_mua_ratio=max_mua_ratio, min_snr=min_snr,
                       min_overlapping_spikes=min_overlapping_spikes, verbose=False)

    # No segments in the demo (segment_duration > duration of synthetic data) but done as example
    work_summary.stitch_segments()

    # Summarize the sorted output data into dictionaries by time segment.
    work_summary.summarize_neurons_by_seg()

    # Finally summarize neurons across channels (combining and removing duplicate
    # neurons across space) to get a list of sorted "neurons"
    neurons = work_summary.summarize_neurons_across_channels(
                    overlap_ratio_threshold=np.inf, min_segs_per_unit=1)

    print("Saving neurons file as", fname_out_data)
    with open(fname_out_data, 'wb') as fp:
        pickle.dump(neurons, fp, protocol=-1)
