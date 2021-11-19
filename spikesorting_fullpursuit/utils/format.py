import pickle
import numpy as np



def to_neuroviz(neurons, save_fname, neuroviz_only=False, filename=None):
    """
    Converts a sorted list of neurons output by FBP sorting to a dictionary
    with parameters suitable for viewing in NeuroViz.

    The new NeuroViz compatible list of neuron dictionaries is saved as a pickle
    file.

    Parameters
    ----------
    neurons : list of dict (or string of pickle file)
        Each element of the list is a dictionary of sorted neurons as output
        by FBP post processing. If input as a string, the function will attempt
        to load that string as a pickle file containing the neurons list.
    filename : string
        Directory where the neurons data originated. Will be used in the output
        'filename__' field for the NeuroViz dictionary. Default of None will
        use the filename field present in the input neurons.
    save_fname : string
        Directory where the NeuroViz compatible dictionary will be saved.
    neuroviz_only : bool
        If True, the saved dictionary will contain only the data required for
        NeuroViz. If False, the required NeuroViz fields will be added to the
        existing dictionaries in neurons in place, modifying the input elements.

    Returns
    -------
    None. The resulting dictionary is saved to save_fname and the input neurons
    are modified in place.
    """

    if isinstance(neurons, str):
        with open(neurons, 'rb') as fp:
            neurons = pickle.load(fp)

    if save_fname[-4:] != ".pkl":
        save_fname = save_fname + ".pkl"
    # Use the default required NeuroViz keys
    nv_keys = ['spike_indices__',
               'filename__',
               'channel_id__',
               'spike_indices_channel__',
               'type__',
               'sampling_rate__']

    if filename is None and neurons[0]['sort_info']['filename'] is None:
        # No filename specified so use default
        print("No filename specified, using 'default_fname'")
        filename = "default_fname"
    elif filename is None:
        filename = neurons[0]['sort_info']['filename']

    if neuroviz_only:
        nv_neurons = []
    for n in neurons:
        if neuroviz_only:
            viz_dict = {}
        for nv_key in nv_keys:
            if nv_key == 'filename__':
                if neuroviz_only:
                    viz_dict['filename__'] = filename
                else:
                    n['filename__'] = filename
            elif nv_key == 'channel_id__':
                if neuroviz_only:
                    viz_dict['channel_id__'] = n['channel'][0] + 1
                else:
                    n['channel_id__'] = n['channel'][0] + 1
            elif nv_key == 'type__':
                if neuroviz_only:
                    viz_dict['type__'] = "Neuron"
                else:
                    n['type__'] = "Neuron"
            elif nv_key == 'sampling_rate__':
                if neuroviz_only:
                    viz_dict['sampling_rate__'] = n['sort_info']['sampling_rate']
                else:
                    n['sampling_rate__'] = n['sort_info']['sampling_rate']
            elif nv_key == 'spike_indices__':
                if neuroviz_only:
                    viz_dict['spike_indices__'] = n['spike_indices'] + 1
                else:
                    n['spike_indices__'] = n['spike_indices'] + 1
            elif nv_key == 'spike_indices_channel__':
                if neuroviz_only:
                    viz_dict['spike_indices_channel__'] = (n['channel'][0] + 1) * np.ones(n['spike_indices'].shape, dtype=np.uint16)
                else:
                    n['spike_indices_channel__'] = (n['channel'][0] + 1) * np.ones(n['spike_indices'].shape, dtype=np.uint16)
            else:
                raise ValueError("Unrecognized key", nv_key, "for NeuroViz dictionary")

        if neuroviz_only:
            nv_neurons.append(viz_dict)

    # Need to save protocol 3 to be compatiblel with Julia
    if neuroviz_only:
        with open(save_fname, 'wb') as fp:
            pickle.dump(nv_neurons, fp, protocol=3)
    else:
        with open(save_fname, 'wb') as fp:
            pickle.dump(neurons, fp, protocol=3)
    print("Saved NeuroViz file:", save_fname)

    return None
