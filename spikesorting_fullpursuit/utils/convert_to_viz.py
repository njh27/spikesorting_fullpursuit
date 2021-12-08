import sys
from spikesorting_fullpursuit.utils.format import to_neuroviz



"""
    convert_to_viz.py fname_neurons
"""

if __name__ == '__main__':


    fname_neurons = sys.argv[1]
    fname_neurons = fname_neurons.rsplit(".")
    if fname_neurons[-4:] != ".pkl" and fname_neurons[-7:] != ".pickle":
        fname_neurons = fname_neurons + ".pkl"
    if fname_neurons[0:8] == "neurons_":
        save_fname = fname_neurons.split("neurons_")[1]
    save_fname = save_fname.split(".")[0]
    save_fname = save_fname + "_viz"
    to_neuroviz(fname_neurons, save_fname, neuroviz_only=False, filename=None)
