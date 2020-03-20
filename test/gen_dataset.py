import numpy as np
from scipy import signal
from scipy import stats
from spikesorting_python.src import electrode
from spikesorting_python.src import spikesorting
from spikesorting_python.src.parallel import spikesorting_parallel



class TestProbe(electrode.AbstractProbe):

    def __init__(self, sampling_rate, voltage_array, num_channels=None):
        if num_channels is None:
            num_channels = voltage_array.shape[1]
        electrode.AbstractProbe.__init__(self, sampling_rate, num_channels, voltage_array=voltage_array)

    def get_neighbors(self, channel):
        """ Test probe neighbor function simple returns an array of all channel numbers.
            """

        # start = max(channel - 2, 0)
        # stop = min(channel + 3, 4)
        start = 0
        stop = 4

        return np.arange(start, stop)


class TestDataset(object):

    def __init__(self, num_channels, duration, random_seed=None, neuron_templates=None, frequency_range=(300, 6000), samples_per_second=40000, amplitude=1):
        self.num_channels = num_channels
        self.duration = duration
        self.frequency_range = frequency_range
        self.samples_per_second = samples_per_second
        self.amplitude = amplitude
        if neuron_templates is not None:
            self.neuron_templates = neuron_templates
        else:
            self.neuron_templates = self.get_default_templates()
        self.voltage_array = None
        self.actual_IDs = []
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.random_state = np.random.get_state()
        np.random.set_state(self.random_state)

    """
    	gen_poission_spiketrain(rate, refactory period, samples per second [, duration])
    Generate a poisson distributed neuron spike train given the passed firing rate (in Hz).
    The refactory period (in sec) determines the silency period between two adjacent spikes.
    Samples per seconds defines the sampling rate of output spiketrain.
    """
    def gen_poisson_spiketrain(self, firing_rate, tau_ref=1.5e-3):
        # np.random.set_state(self.random_state)

        number_of_samples = int(np.round(self.duration * self.samples_per_second))
        spiketrain = np.zeros(number_of_samples, dtype='bool')
        random_nums = np.random.random(spiketrain.size)

        if tau_ref < 0:
            tau_ref = 0
        elif tau_ref > 0:
            # Adjust the firing rate to take into account the refactory period
            firing_rate = firing_rate * 1.0 / (1.0 - firing_rate * tau_ref)

        # Generate the spiketrain
        last_spike_index = -np.inf
        for i in range(0, spiketrain.size):
            if (i - last_spike_index) / self.samples_per_second > tau_ref:
                spiketrain[i] = random_nums[i] < firing_rate / self.samples_per_second
                if spiketrain[i]:
                    last_spike_index = i

        # self.random_state = np.random.get_state()
        return spiketrain

    def get_default_templates(self):
        templates = np.array([
            [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.001, -0.0, -0.0, 0.0, 0.0, 0.001, 0.002, 0.005, 0.01, 0.019, 0.034, 0.057, 0.1, 0.166, 0.222, 0.167, -0.088, -0.49, -0.855, -1.0, -0.862, -0.526, -0.151, 0.136, 0.293, 0.339, 0.312, 0.251, 0.187, 0.138, 0.109, 0.093, 0.084, 0.076, 0.069, 0.063, 0.058, 0.056, 0.054, 0.052, 0.051, 0.05, 0.049, 0.049, 0.048, 0.048, 0.047, 0.047, 0.046, 0.046, 0.045, 0.044, 0.043, 0.042, 0.041, 0.04, 0.039, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033, 0.032, 0.03, 0.029, 0.027, 0.026, 0.025],
            [0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.001, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.001, 0.001, 0.002, 0.002, 0.003, 0.004, 0.005, 0.005, 0.006, 0.007, 0.007, 0.008, 0.009, 0.011, 0.013, 0.017, 0.024, 0.033, 0.045, 0.062, 0.084, 0.091, 0.033, -0.166, -0.513, -0.858, -1.0, -0.868, -0.555, -0.212, 0.057, 0.218, 0.285, 0.293, 0.274, 0.248, 0.224, 0.204, 0.187, 0.173, 0.16, 0.149, 0.14, 0.132, 0.124, 0.117, 0.11, 0.103, 0.097, 0.091, 0.086, 0.081, 0.076, 0.072, 0.067, 0.064, 0.06, 0.056, 0.053, 0.05, 0.046, 0.043, 0.04, 0.038, 0.035, 0.033, 0.031, 0.028, 0.026, 0.023, 0.021, 0.019, 0.017, 0.015, 0.013, 0.012],
            [-0.108, -0.11, -0.111, -0.112, -0.112, -0.11, -0.107, -0.105, -0.104, -0.101, -0.099, -0.097, -0.095, -0.092, -0.088, -0.083, -0.077, -0.071, -0.065, -0.062, -0.058, -0.053, -0.045, -0.037, -0.03, -0.022, -0.013, -0.003, 0.01, 0.027, 0.045, 0.059, 0.071, 0.081, 0.092, 0.106, 0.12, 0.133, 0.145, 0.158, 0.171, 0.19, 0.215, 0.245, 0.276, 0.305, 0.351, 0.453, 0.642, 0.877, 1.0, 0.883, 0.647, 0.45, 0.342, 0.295, 0.269, 0.242, 0.214, 0.189, 0.171, 0.158, 0.146, 0.134, 0.12, 0.107, 0.094, 0.083, 0.072, 0.06, 0.049, 0.036, 0.022, 0.008, -0.006, -0.019, -0.027, -0.033, -0.04, -0.048, -0.054, -0.059, -0.063, -0.069, -0.073, -0.079, -0.086, -0.092, -0.095, -0.096, -0.098, -0.101, -0.105, -0.105, -0.105, -0.105, -0.105, -0.107, -0.109, -0.111]
            ])

        templates /= np.amax(np.abs(templates), axis=1)[:, None]

        return templates

    """
        gen_bandlimited_noise([frequency_range, samples_per_second, duration, amplitude])

    Generates a timeseries of bandlimited Gaussian noise. This function uses the inverse FFT to generate
    noise in a given bandwidth. Within this bandwidth, the amplitude of each frequency is approximately
    constant (i.e., white), but the phase is random. The amplitude of the output signal is scaled so that
    99% of the values fall within the amplitude criteron (3 standard deviations of zero mean).
    """
    def gen_bandlimited_noise(self):
        # np.random.set_state(self.random_state)

        freqs = np.abs(np.fft.fftfreq(self.samples_per_second * self.duration, 1/self.samples_per_second))
        f = np.zeros(self.samples_per_second * self.duration)
        idx = np.where(np.logical_and(freqs>=self.frequency_range[0], freqs<=self.frequency_range[1]))[0]
        f[idx] = 1
        f = np.array(f, dtype='complex')
        Np = (len(f) - 1) // 2
        phases = np.random.rand(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        f[1:Np+1] *= phases
        f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
        bandlimited_noise = np.fft.ifft(f).real
        bandlimited_noise = bandlimited_noise * self.amplitude / (3 * np.std(bandlimited_noise))

        # self.random_state = np.random.get_state()
        return bandlimited_noise

    def gen_noise_voltage_array(self):

        noise_voltage_array = np.empty((self.num_channels, self.samples_per_second*self.duration))
        for i in range (0, self.num_channels):
            noise_voltage_array[i, :] = self.gen_bandlimited_noise()

        return noise_voltage_array

    def gen_test_dataset(self, firing_rates, template_inds, chan_scaling_factors, refractory_wins=1.5e-3):

        if len(firing_rates) != len(template_inds) or len(firing_rates) != len(chan_scaling_factors):
            raise ValueError("Input rates, templates, and scaling factors must all be the same length!")

        refractory_wins = np.array(refractory_wins)
        if refractory_wins.size == 1:
            refractory_wins = np.repeat(refractory_wins, len(firing_rates))
        # Reset neuron actual IDs for each neuron
        self.actual_IDs = [[] for neur in range(0, len(firing_rates))]
        self.actual_templates = [np.empty((0, self.neuron_templates.shape[1])) for neur in range(0, len(firing_rates))]
        voltage_array = self.gen_noise_voltage_array()
        for neuron in range(0, len(firing_rates)):
            # Generate one spike train for each neuron
            spiketrain = self.gen_poisson_spiketrain(firing_rate=firing_rates[neuron], tau_ref=refractory_wins[neuron])
            self.actual_IDs[neuron] = np.where(spiketrain)[0]
            for chan in range(0, self.num_channels):
                # Apply spike train to every channel this neuron is present on
                convolve_kernel = chan_scaling_factors[neuron][chan] * self.neuron_templates[template_inds[neuron], :] * self.amplitude
                self.actual_templates[neuron] = np.vstack((self.actual_templates[neuron], convolve_kernel))
                if chan_scaling_factors[neuron][chan] <= 0:
                    continue
                voltage_array[chan, :] += signal.fftconvolve(spiketrain, convolve_kernel, mode='same')

        self.voltage_array = voltage_array

    def assess_performance():
        pass

    def sort_test_dataset(self, kwargs):

        spike_sort_kwargs = {'sigma': 4., 'clip_width': [-6e-4, 8e-4],
                              'p_value_cut_thresh': 0.01, 'check_components': None,
                              'max_components': 10,
                              'min_firing_rate': 1, 'do_binary_pursuit': False,
                              'add_peak_valley': False, 'do_ZCA_transform': True,
                              'do_branch_PCA': True, 'max_gpu_memory': None,
                              'use_rand_init': True, 'cleanup_neurons': False,
                              'verbose': True}
        for key in kwargs:
            spike_sort_kwargs[key] = kwargs[key]

        self.Probe = TestProbe(self.samples_per_second, self.voltage_array, self.num_channels)
        neurons = spikesorting.spike_sort(self.Probe, **spike_sort_kwargs)

        return neurons

    def sort_test_dataset_parallel(self, kwargs):

        spike_sort_kwargs = {'sigma': 4., 'clip_width': [-6e-4, 8e-4],
                           'filter_band': self.frequency_range,
                           'p_value_cut_thresh': 0.01, 'check_components': None,
                           'max_components': 10,
                           'min_firing_rate': 1, 'do_binary_pursuit': False,
                           'add_peak_valley': False, 'do_branch_PCA': True,
                           'max_gpu_memory': None, 'use_rand_init': True,
                           'cleanup_neurons': False,
                           'verbose': True,
                           'test_flag': True, 'log_dir': None,
                           'do_ZCA_transform': True}
        for key in kwargs:
            spike_sort_kwargs[key] = kwargs[key]

        self.Probe = TestProbe(self.samples_per_second, self.voltage_array, self.num_channels)
        neurons = spikesorting_parallel.spike_sort_parallel(self.Probe, **spike_sort_kwargs)

        return neurons


    def compare_single_vs_parallel(self, kwargs):

        single_sort_kwargs = {'sigma': 4., 'clip_width': [-6e-4, 8e-4],
                              'p_value_cut_thresh': 0.01, 'check_components': None,
                              'max_components': 10,
                              'min_firing_rate': 1, 'do_binary_pursuit': False,
                              'add_peak_valley': False, 'do_ZCA_transform': True,
                              'do_branch_PCA': True, 'max_gpu_memory': None,
                              'use_rand_init': True,
                              'cleanup_neurons': False,
                              'verbose': True}
        for key in kwargs:
            single_sort_kwargs[key] = kwargs[key]
        par_sort_kwargs = {'sigma': 4., 'clip_width': [-6e-4, 8e-4],
                           'filter_band': self.frequency_range,
                           'p_value_cut_thresh': 0.01, 'check_components': None,
                           'max_components': 10,
                           'min_firing_rate': 1, 'do_binary_pursuit': False,
                           'add_peak_valley': False, 'do_branch_PCA': True,
                           'max_gpu_memory': None, 'use_rand_init': True,
                           'cleanup_neurons': False,
                           'verbose': True,
                           'test_flag': True, 'log_dir': None,
                           'do_ZCA_transform': True}
        for key in single_sort_kwargs.keys():
            par_sort_kwargs[key] = single_sort_kwargs[key]

        self.random_state = np.random.get_state()
        first_state = self.random_state
        np.random.set_state(first_state)
        self.Probe = TestProbe(self.samples_per_second, self.voltage_array, self.num_channels)
        neurons = spikesorting.spike_sort(self.Probe, **single_sort_kwargs)

        np.random.set_state(first_state)
        self.Probe = TestProbe(self.samples_per_second, self.voltage_array, self.num_channels)
        neurons_parallel = spikesorting_parallel.spike_sort_parallel(self.Probe, **par_sort_kwargs)
        self.random_state = first_state

        n_ind = 0
        for neur, npar in zip(neurons, neurons_parallel):
            for key in neur.keys():
                print("testing key", key)
                if key == 'template':
                    continue
                if key == 'sort_quality':
                    continue
                    # assert neur['sort_quality']['false_positives'] == npar['sort_quality']['false_positives'], "Neuron {0} sort quality false positives differ".format(n_ind)
                    # assert neur['sort_quality']['false_positives'] == npar['sort_quality']['false_positives'], "Neuron {0} sort quality false positives differ".format(n_ind)
                elif key == 'waveforms':
                    # assert np.all(np.abs(np.sum(neur[key] - npar[key], axis=1)) < 1e-6), "Neuron {0} waveforms differ".format(n_ind)
                    assert np.allclose(neur[key], npar[key], equal_nan=True), "Neuron {0} waveforms differ".format(n_ind)
                else:
                    assert np.all(neur[key] == npar[key]), "Neuron {0} {1} differ".format(n_ind, key)
                    # assert np.allclose(neur[key], npar[key], equal_nan=True), "Neuron {0} {1} differ".format(n_ind, key)
            n_ind += 1

        return neurons, neurons_parallel
