import sys
from spikesorting_fullpursuit.utils.format import to_neuroviz



"""
    convert_to_viz.py fname_neurons [...save_fname, ]
"""
""" /volumes/t7/spikesortingfiles/bin_files/learndirtunepurk_dandy_06_volt.bin """
if __name__ == '__main__':


    fname_neurons = sys.argv[1]
    fname_neurons = fname_neurons.rsplit(".")
    if len(fname_neurons) == 2:
        if fname_neurons[1] != "pkl" and fname_neurons[1] != "pickle":
            fname_neurons = fname_neurons[0] + ".pkl"
        else:
            fname_neurons = fname_neurons[0] + "." + fname_neurons[1]
    elif len(fname_neurons) == 1:
        fname_neurons = fname_neurons + ".pkl"
    else:
        raise ValueError("Cannot parse multiple file extensions on input neurons filename")
    if len(sys.argv) == 2:
        save_fname = fname_neurons.split(".")[0]
        save_fname = save_fname + "_viz.pkl"
    elif len(sys.argv) >= 3:
        save_fname = sys.argv[2]
        if save_fname[-4:] == ".pkl" or save_fname[-7:] == ".pickle":
            save_fname = save_fname.split(".")[0] + "_viz.pkl"
        elif save_fname[-4:] == "_viz":
            save_fname = save_fname + ".pkl"
        else:
            print("Save file name does not indicate _viz file or .pkl.")
    to_neuroviz(fname_neurons, save_fname, neuroviz_only=False, filename=None)
