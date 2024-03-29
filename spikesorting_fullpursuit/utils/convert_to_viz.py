import sys
from spikesorting_fullpursuit.utils.format import to_neuroviz



""" A couple functions useful for converting the out "neurons" dictionary of the
sorter into a format that can be read by "NeuroViz" for manual inspection and
post-processing. """
def f_neurons_to_viz(fname_neurons, save_fname=None, neuroviz_only=False, filename=None):
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
    if save_fname is None:
        save_fname = fname_neurons.split(".")[0]
        save_fname = save_fname + "_viz.pkl"

    to_neuroviz(fname_neurons, save_fname, neuroviz_only=neuroviz_only, filename=filename)

    return None


"""
    convert_to_viz.py fname_neurons [...save_fname, ]
"""
""" /volumes/t7/spikesortingfiles/bin_files/learndirtunepurk_dandy_06_volt.bin """
if __name__ == '__main__':


    fname_neurons = sys.argv[1]
    if len(sys.argv) == 2:
        save_fname = None
    elif len(sys.argv) >= 3:
        save_fname = sys.argv[2]
        if save_fname[-4:] == ".pkl" or save_fname[-7:] == ".pickle":
            save_fname = save_fname.split(".")[0] + "_viz.pkl"
        elif save_fname[-4:] == "_viz":
            save_fname = save_fname + ".pkl"
        else:
            print("Save file name does not indicate _viz file or .pkl.")

    f_neurons_to_viz(fname_neurons, save_fname=save_fname, neuroviz_only=True, filename=None)
