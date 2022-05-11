import sys
from spikesorting_fullpursuit.utils.format import to_neuroviz



"""
    convert_to_viz.py fname_neurons [...save_fname, ]
"""
""" /volumes/t7/spikesortingfiles/bin_files/learndirtunepurk_dandy_06_volt.bin """
if __name__ == '__main__':


    fname_neurons = sys.argv[1]
    fname_neurons = fname_neurons.rsplit(".")
    if fname_neurons[-4:] != ".pkl" and fname_neurons[-7:] != ".pickle":
        fname_neurons = fname_neurons + ".pkl"
    if len(sys.argv) == 2:
        save_fname = fname_neurons.split("neurons_")[1]
        save_fname = save_fname.split(".")[0]
        save_fname = save_fname + "_viz"
    elif len(sys.argv) >= 3:
        save_fname = sys.argv[2]
        if save_fname[-4:] == ".pkl" or save_fname[-7:] == ".pickle":
            save_fname = save_fname.split(".")[0] + "_viz.pkl"
        elif save_fname[-4:] == "_viz":
            save_fname = save_fname + ".pkl"
        else:
            print("Save file name does not indicate _viz file or .pkl.")
    to_neuroviz(fname_neurons, save_fname, neuroviz_only=False, filename=None)
